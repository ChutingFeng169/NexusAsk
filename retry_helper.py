"""
retry_helper.py — shared reliability + logging + step tracking for NexusAsk.

1. call_with_retry: wraps any callable with configurable timeout, retry on
   timeout / HTTP 429 / 5xx, exponential backoff with jitter, Retry-After support.
2. StepTracker: records each agent step (action, duration, ok, retries) and
   drives the live st.status widget + the saved step history.
3. setup_logging: configures Python's logging module (stdout + file).
"""
import os, time, random, logging
import httpx

LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
BACKOFF_CAP = 30.0

logger = logging.getLogger("nexusask")


def setup_logging(level=logging.INFO):
    if logger.handlers:
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    try:
        fh = logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexusask.log"))
        fh.setFormatter(fmt); logger.addHandler(fh)
    except Exception:
        pass
    logger.setLevel(level)


setup_logging()

_current_tracker = None


def set_current_tracker(tracker):
    global _current_tracker
    _current_tracker = tracker


def get_current_tracker():
    return _current_tracker


class StepTracker:
    def __init__(self, status_widget=None):
        self.steps = []
        self.status = status_widget
        self._t0 = None
        self._current = None
        self._last_retries = 0

    def start(self, action):
        self._current = action
        self._t0 = time.time()
        self._last_retries = 0
        if self.status:
            try: self.status.update(label=f"{action}…")
            except Exception: pass

    def record(self, action, seconds, ok=True, retries=0):
        self.steps.append({"action": action, "seconds": round(seconds, 1), "ok": ok, "retries": retries})
        if self.status:
            try:
                mark = "✅" if ok else "❌"
                rs = f" (retry {retries})" if retries else ""
                self.status.update(label=f"{action} ({seconds:.1f}s)")
                self.status.markdown(f"**{action}** — {seconds:.1f}s {mark}{rs}")
            except Exception: pass

    def record_done(self, ok=True):
        """Record the current step (started via start()) using _last_retries (set by call_with_retry). Resets _current."""
        if self._current and self._t0:
            elapsed = time.time() - self._t0
            self.record(self._current, elapsed, ok=ok, retries=self._last_retries)
        self._current = None
        self._last_retries = 0

    def retry(self, attempt, max_retries):
        label = self._current or "Working"
        if self.status:
            try: self.status.update(label=f"{label} — retrying ({attempt}/{max_retries})…", state="running")
            except Exception: pass

    def done(self):
        if self.status:
            try: self.status.update(label="✅ Done", state="complete")
            except Exception: pass

    def to_list(self):
        return self.steps


def call_with_retry(fn, *, timeout=LLM_TIMEOUT_S, max_retries=MAX_RETRIES, step_name=None):
    """Wrap a callable with timeout-aware retry.
    Retries on: httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError (429 or 5xx).
    Honors Retry-After. Exponential backoff with jitter. Logs every attempt.
    On final failure: logs traceback, raises friendly RuntimeError.
    Uses the module-level StepTracker (if set) to record steps + show retry status."""
    name = step_name or "call"
    tracker = get_current_tracker()
    for attempt in range(1, max_retries + 2):
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            logger.info(f"[{name}] attempt {attempt}/{max_retries+1} ok in {elapsed:.1f}s (timeout={timeout}s)")
            if tracker:
                tracker._last_retries = max(0, attempt - 1)
            return result
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            elapsed = time.time() - t0
            if attempt <= max_retries:
                wait = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1))) * (0.5 + random.random() * 0.5)
                logger.warning(f"[{name}] attempt {attempt}/{max_retries+1} {type(e).__name__} in {elapsed:.1f}s, retry in {wait:.1f}s")
                if tracker: tracker.retry(attempt, max_retries)
                time.sleep(wait)
            else:
                logger.error(f"[{name}] final failure: {type(e).__name__}: {e}", exc_info=True)
                if tracker and tracker._current: tracker.record(tracker._current, elapsed, ok=False, retries=attempt-1)
                raise RuntimeError("The request timed out or the service is unavailable. Please try again.") from e
        except httpx.HTTPStatusError as e:
            elapsed = time.time() - t0
            status = e.response.status_code
            if (status == 429 or 500 <= status < 600) and attempt <= max_retries:
                ra = e.response.headers.get("Retry-After")
                try: wait = min(BACKOFF_CAP, float(ra)) if ra else min(BACKOFF_CAP, BACKOFF_BASE*(2**(attempt-1)))*(0.5+random.random()*0.5)
                except ValueError: wait = min(BACKOFF_CAP, BACKOFF_BASE*(2**(attempt-1)))*(0.5+random.random()*0.5)
                logger.warning(f"[{name}] attempt {attempt}/{max_retries+1} HTTP {status} in {elapsed:.1f}s, retry in {wait:.1f}s")
                if tracker: tracker.retry(attempt, max_retries)
                time.sleep(wait)
            else:
                logger.error(f"[{name}] final failure: HTTP {status}: {e}", exc_info=True)
                if tracker and tracker._current: tracker.record(tracker._current, elapsed, ok=False, retries=attempt-1)
                raise RuntimeError(f"The service returned HTTP {status}. Please try again.") from e
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{name}] non-retryable: {type(e).__name__}: {e}", exc_info=True)
            if tracker and tracker._current: tracker.record(tracker._current, elapsed, ok=False, retries=0)
            raise RuntimeError("An unexpected error occurred. Please try again.") from e
    raise RuntimeError("Max retries exceeded.")

