"""
rate_limit_fallback.py
======================
Retry + exponential backoff (with jitter) for the primary LLM call, then an
automatic fallback to a secondary endpoint.

Design rules
------------
* This module is the ONLY retry layer. Do not wrap it in another retry loop
  (e.g. an outer "llm_chat" retry), or attempts multiply (3 x 10 = 30).
* It never touches Streamlit. It reports progress through on_progress(event),
  and the UI decides what to show. Low-level details go to the logger only.
* Retries only errors that can succeed on a second try: HTTP 429, HTTP 5xx,
  timeouts and network errors. 400/401/404 etc. fail immediately.

Env (all optional)
------------------
    OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL            -> primary
    FALLBACK_API_KEY / FALLBACK_BASE_URL / FALLBACK_MODEL   -> fallback
    LLM_TIMEOUT_S         read timeout per attempt (default 30; set 2-3 to test)
    LLM_CONNECT_TIMEOUT_S connect timeout (default 10)
    MAX_RETRIES           total attempts on primary (default 3)
    BACKOFF_BASE / BACKOFF_MIN / BACKOFF_MAX   seconds
"""

import logging
import os
import random
import threading
import time

import httpx
from dotenv import load_dotenv
from tenacity import Retrying, retry_if_exception, stop_after_attempt

load_dotenv()

logger = logging.getLogger("nexus.llm")

# ---------------------------------------------------------------- env config
PRIMARY_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
PRIMARY_API_KEY = os.getenv("OPENAI_API_KEY", "")
PRIMARY_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", PRIMARY_BASE_URL).rstrip("/")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", PRIMARY_API_KEY)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", PRIMARY_MODEL)

# ----------------------------------------------------- tunable settings
# Read at call time, so tests can monkeypatch them.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
CONNECT_TIMEOUT_S = float(os.getenv("LLM_CONNECT_TIMEOUT_S", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1"))
BACKOFF_MIN = float(os.getenv("BACKOFF_MIN", "1"))
BACKOFF_MAX = float(os.getenv("BACKOFF_MAX", "20"))
LLM_TOTAL_TIMEOUT_S = float(os.getenv("LLM_TOTAL_TIMEOUT_S", "120"))
MIN_ATTEMPT_S = float(os.getenv("MIN_ATTEMPT_S", "5"))
_SLEEP = time.sleep  # tests replace this with an instant recorder

_retry_tls = threading.local()


def time_left():
    """Seconds left in the current retry budget, or None if no budget is active."""
    start = getattr(_retry_tls, "start", None)
    if start is None:
        return None
    return LLM_TOTAL_TIMEOUT_S - (time.monotonic() - start)


class RateLimitError(Exception):
    """HTTP 429. retry_after is seconds from the Retry-After header, if present."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ------------------------------------------------------------- classification
def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _reason(exc: BaseException) -> str:
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    if isinstance(exc, httpx.TransportError):
        return "network"
    return type(exc).__name__


def backoff_seconds(attempt_number: int, exc=None) -> float:
    """Honor Retry-After if given; otherwise exponential backoff with jitter."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        return min(float(retry_after), BACKOFF_MAX)
    ceiling = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt_number - 1)))
    return max(BACKOFF_MIN, random.uniform(ceiling / 2, ceiling))


def _wait(rs) -> float:
    exc = rs.outcome.exception() if rs.outcome else None
    return backoff_seconds(rs.attempt_number, exc)


def raise_for_status_with_retry_after(resp) -> None:
    """429 -> RateLimitError (with Retry-After); other 4xx/5xx -> httpx.HTTPStatusError."""
    if resp.status_code == 429:
        try:
            ra = float(resp.headers.get("Retry-After", ""))
        except ValueError:
            ra = None
        raise RateLimitError("429 Too Many Requests", retry_after=ra)
    resp.raise_for_status()


# ------------------------------------------------------------------- HTTP
def _chat(base_url: str, api_key: str, model: str, prompt: str) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint."""
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(LLM_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
    )
    try:
        resp = client.post(
            "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        )
        raise_for_status_with_retry_after(resp)
        return resp.json()["choices"][0]["message"]["content"]
    finally:
        client.close()


def call_primary_model(prompt: str) -> str:
    return _chat(PRIMARY_BASE_URL, PRIMARY_API_KEY, PRIMARY_MODEL, prompt)


def call_fallback_model(prompt: str) -> str:
    return _chat(FALLBACK_BASE_URL, FALLBACK_API_KEY, FALLBACK_MODEL, prompt)


def _default_reporter(event):
    logger.info("progress: %s", event)


# ------------------------------------------------------------------- retry
def retry_call(fn, *args, max_attempts=None, on_progress=None, **kwargs):
    """
    Call fn(*args, **kwargs); retry retryable errors with backoff. The ONLY retry layer.
    max_attempts=1 means "no retry". on_progress(event) gets:
      {"type":"attempt","n":N,"total":T}
      {"type":"rate_limited","n":N,"total":T,"wait":W}                 # 429
      {"type":"retrying","n":N,"total":T,"wait":W,"reason":"timeout"}  # timeout / 5xx / network
    """
    total = int(max_attempts or MAX_RETRIES)
    emit = on_progress or _default_reporter

    def before(rs):
        logger.info("LLM attempt %d/%d (read timeout %.1fs)", rs.attempt_number, total, LLM_TIMEOUT_S)
        emit({"type": "attempt", "n": rs.attempt_number, "total": total})

    def before_sleep(rs):
        exc = rs.outcome.exception()
        wait = rs.next_action.sleep
        logger.warning(
            "LLM attempt %d/%d failed (%s: %s); backing off %.2fs",
            rs.attempt_number, total, _reason(exc), exc, wait,
        )
        if isinstance(exc, RateLimitError):
            emit({"type": "rate_limited", "n": rs.attempt_number, "total": total, "wait": wait})
        else:
            emit({"type": "retrying", "n": rs.attempt_number, "total": total, "wait": wait, "reason": _reason(exc)})

    retryer = Retrying(
        retry=retry_if_exception(_is_retryable),
        wait=_wait,
        stop=stop_after_attempt(total),
        sleep=_SLEEP,
        reraise=True,
        before=before,
        before_sleep=before_sleep,
    )
    _retry_tls.start = time.monotonic()
    try:
        return retryer(fn, *args, **kwargs)
    finally:
        _retry_tls.start = None


# ------------------------------------------------------------------- main
def answer_with_retry_and_fallback(prompt: str, primary_fn=None, fallback_fn=None, on_progress=None) -> str:
    """
    Try the primary model up to MAX_RETRIES times (backoff + jitter) on retryable
    errors; if all attempts fail, call the fallback model once.
    Events: see retry_call, plus {"type":"exhausted"}, {"type":"fallback"},
    {"type":"fallback_done"}, {"type":"done"}.
    """
    primary_fn = primary_fn or call_primary_model
    fallback_fn = fallback_fn or call_fallback_model
    on_progress = on_progress or _default_reporter
    total = MAX_RETRIES
    started = time.perf_counter()
    try:
        result = retry_call(primary_fn, prompt, max_attempts=total, on_progress=on_progress)
        logger.info("LLM ok in %.1fs", time.perf_counter() - started)
        on_progress({"type": "done"})
        return result
    except Exception as exc:
        if not _is_retryable(exc):
            logger.exception("LLM non-retryable error; not retrying")
            raise
        logger.error(
            "Primary exhausted after %d attempts in %.1fs (%s); switching to fallback",
            total, time.perf_counter() - started, _reason(exc),
        )
        if (FALLBACK_BASE_URL, FALLBACK_MODEL) == (PRIMARY_BASE_URL, PRIMARY_MODEL):
            logger.warning("Fallback is identical to primary; it will likely fail the same way")
        on_progress({"type": "exhausted", "n": total, "reason": _reason(exc)})
        on_progress({"type": "fallback"})
        try:
            fb = fallback_fn(prompt)
        except Exception:
            logger.exception("Fallback model also failed")
            raise
        on_progress({"type": "fallback_done"})
        return fb

