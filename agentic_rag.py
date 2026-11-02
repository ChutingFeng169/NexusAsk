"""
agentic_rag.py — Agentic RAG with native tool-calling (2026 pattern), legacy-stack-safe.

Pipeline: [optional] query rewrite -> native tool-calling loop (search_docs / search_web) ->
faithfulness self-correction. Native tool-calling via the DeepSeek proxy (confirmed:
emits real tool_calls). No langgraph (incompatible with pinned langchain-core 0.1.53);
the agent loop is plain Python with native function-calling over httpx.

Reliability design
------------------
* ONE retry layer: every model call goes through rate_limit_fallback.retry_call
  (timeouts, 429, 5xx, network errors; exponential backoff + jitter; logged).
* This module never touches Streamlit. It reports progress with on_progress(event);
  retries show up as {"type": "retry", "n", "total", "wait", "reason"}.
* The query rewrite is a nice-to-have: it is skipped when it cannot help, uses a
  short timeout, never retries, and falls back to the original question.
"""
import json
import logging
import os
import threading
import time
from contextlib import contextmanager

import httpx
from dotenv import load_dotenv
load_dotenv()
import rate_limit_fallback as rlf
from simple_eval import simple_similarity  # reuse faithfulness (embedding sim)
from langsmith import traceable  # LangSmith tracing for the agentic (httpx) calls

logger = logging.getLogger("nexus.agent")

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
MODEL = os.getenv("LLM_MODEL", "")

# Query rewrite settings (env-overridable)
REWRITE_TIMEOUT_S = float(os.getenv("REWRITE_TIMEOUT_S", "4"))
REWRITE_MIN_CHARS = int(os.getenv("REWRITE_MIN_CHARS", "15"))

_client = httpx.Client(
    base_url=BASE_URL,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    timeout=httpx.Timeout(rlf.LLM_TIMEOUT_S, connect=rlf.CONNECT_TIMEOUT_S),
)

# ---------------- retry events -> progress events ----------------
_tls = threading.local()


def _emit_retry(ev):
    cb = getattr(_tls, "on_retry", None)
    if cb:
        try:
            cb(ev)
        except Exception:
            pass


def _retry_to_progress(on_progress):
    """Turn rate_limit_fallback retry events into one simple {"type": "retry"} progress event."""
    def convert(ev):
        if ev.get("type") in ("rate_limited", "retrying"):
            on_progress({
                "type": "retry",
                "n": ev["n"],
                "total": ev.get("total"),
                "wait": ev.get("wait", 0.0),
                "reason": ev.get("reason", "rate_limit"),
            })
    return convert


@contextmanager
def _retry_events(on_progress):
    """While active, retries inside llm_chat / llm_chat_stream are reported via on_progress."""
    _tls.on_retry = _retry_to_progress(on_progress) if on_progress else None
    try:
        yield
    finally:
        _tls.on_retry = None


def _trim_chunks(inputs):
    """process_input for @traceable: replace a large chunks list with a count to avoid trace bloat."""
    if isinstance(inputs, dict) and "chunks" in inputs:
        inputs = dict(inputs)
        c = inputs["chunks"]
        inputs["chunks"] = f"<{len(c) if hasattr(c, '__len__') else '?'} chunks>"
    return inputs


# ---------------- model calls ----------------
def _request_timeout(timeout):
    """Per-attempt timeout, capped to what is left of the overall retry budget."""
    t = float(timeout if timeout is not None else rlf.LLM_TIMEOUT_S)
    left = rlf.time_left()
    if left is not None:
        t = min(t, left)
    return httpx.Timeout(t, connect=min(t, rlf.CONNECT_TIMEOUT_S))


def _post_chat(payload, timeout=None):
    resp = _client.post("/chat/completions", json=payload, timeout=_request_timeout(timeout))
    rlf.raise_for_status_with_retry_after(resp)
    return resp.json()["choices"][0]["message"]


@traceable(run_type="llm", name="llm_chat")
def llm_chat(messages, tools=None, temperature=0, timeout=None, max_attempts=None):
    """Native tool-calling chat. Returns {'content': str, 'tool_calls': list|None}.
    timeout: per-attempt seconds (default LLM_TIMEOUT_S). max_attempts=1 means no retry."""
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    t0 = time.perf_counter()
    msg = rlf.retry_call(_post_chat, payload, timeout, max_attempts=max_attempts, on_progress=_emit_retry)
    logger.info("llm_chat ok in %.1fs", time.perf_counter() - t0)
    return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls")}


def _stream_once(messages, tools, temperature, _out, timeout=None):
    """One streaming attempt. Yields content deltas; fills _out['tool_calls'] at the end."""
    payload = {"model": MODEL, "messages": messages, "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    tcs_acc = []
    with _client.stream("POST", "/chat/completions", json=payload, timeout=_request_timeout(timeout)) as resp:
        rlf.raise_for_status_with_retry_after(resp)
        for line in resp.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
            dtcs = delta.get("tool_calls")
            if dtcs:
                for tc in dtcs:
                    idx = tc.get("index", 0)
                    while len(tcs_acc) <= idx:
                        tcs_acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        tcs_acc[idx]["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        tcs_acc[idx]["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tcs_acc[idx]["function"]["arguments"] += fn["arguments"]
    if _out is not None:
        _out["tool_calls"] = tcs_acc if any(tc["function"]["name"] for tc in tcs_acc) else None


def llm_chat_stream(messages, tools=None, temperature=0, _out=None):
    """Streaming native chat (stream=True + SSE). Yields content deltas token-by-token.
    If _out (a dict) is passed, the accumulated tool_calls are written to _out['tool_calls']
    after the stream ends (so a tool-calling loop can drive itself).
    Retries only while NOTHING has been shown to the user yet (timeout / 429 / 5xx when the
    stream opens); once tokens have been yielded, an error is raised instead of restarting."""
    total = rlf.MAX_RETRIES
    t_start = time.monotonic()
    for attempt in range(1, total + 1):
        started = False
        left = rlf.LLM_TOTAL_TIMEOUT_S - (time.monotonic() - t_start)
        try:
            for piece in _stream_once(messages, tools, temperature, _out, timeout=min(rlf.LLM_TIMEOUT_S, max(left, 0.5))):
                started = True
                yield piece
            return
        except Exception as exc:
            wait = rlf.backoff_seconds(attempt, exc)
            left = rlf.LLM_TOTAL_TIMEOUT_S - (time.monotonic() - t_start) - wait
            if started or attempt >= total or not rlf._is_retryable(exc) or left < rlf.MIN_ATTEMPT_S:
                logger.error("llm_chat_stream failed (attempt %d/%d, started=%s): %s", attempt, total, started, exc)
                raise
            logger.warning("llm_chat_stream attempt %d/%d failed (%s: %s); backing off %.2fs",
                           attempt, total, rlf._reason(exc), exc, wait)
            ev_type = "rate_limited" if isinstance(exc, rlf.RateLimitError) else "retrying"
            _emit_retry({"type": ev_type, "n": attempt, "total": total, "wait": wait, "reason": rlf._reason(exc)})
            rlf._SLEEP(wait)


# ---------------- response language (thread-local) ----------------

def set_response_language(lang):
    """Per-session answer language (thread-local). None / "auto" = match the user's question."""
    _tls.lang = lang if lang and lang != "auto" else None


def _lang_line():
    lang = getattr(_tls, "lang", None)
    return f"\n\nAlways write your final answer in {lang}." if lang else ""


# ---------------- tools ----------------
@traceable(run_type="tool", name="search_docs", process_input=_trim_chunks)
def search_docs(query, chunks, k=5):
    """Document retrieval tool (keyword overlap, same as app's search_chunks)."""
    qw = set(query.lower().split())
    scored = [(len(qw & set(c.lower().split())), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


@traceable(run_type="tool", name="search_web")
def search_web(query, n=3):
    """Web search tool (DuckDuckGo). Returns list of {title, snippet, href}."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [{"error": "web search unavailable (duckduckgo-search not installed)"}]
    out = []
    try:
        for r in DDGS().text(query, max_results=n):
            out.append({"title": r.get("title", ""), "snippet": r.get("body", r.get("snippet", "")), "href": r.get("href", "")})
    except Exception as e:
        out = [{"error": f"web search failed: {e}"}]
    return out


@traceable(run_type="tool", name="calculator")
def calculator(expression):
    """Safe math evaluator tool."""
    import math
    allowed = {k: getattr(math, k) for k in ("sqrt", "log", "log10", "sin", "cos", "tan", "ceil", "floor", "factorial", "pi", "e")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"calculation failed: {e}"}


@traceable(run_type="tool", name="wikipedia_search")
def wikipedia_search(query, sentences=3):
    """Wikipedia summary tool (external knowledge)."""
    try:
        import wikipedia
    except ImportError:
        return {"error": "wikipedia not installed"}
    try:
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=False)
        url = ""
        try:
            url = wikipedia.page(query, auto_suggest=False).url
        except Exception:
            url = ""
        return {"title": query, "summary": summary, "url": url}
    except Exception as e:
        return {"error": f"wikipedia failed: {e}"}


# ---------------- agentic steps ----------------
def should_rewrite(question, chunks):
    """Rewriting only helps document retrieval. Skip it when there are no uploaded documents
    (web / wikipedia / calculator build their own queries) or the question is very short."""
    return bool(chunks) and len((question or "").strip()) >= REWRITE_MIN_CHARS


def rewrite_query(question, chunks=None):
    """LLM rewrites the question for better retrieval.
    Nice-to-have only: short timeout, no retry, falls back to the original question.
    Pass chunks to let it skip itself when rewriting cannot help."""
    if chunks is not None and not should_rewrite(question, chunks):
        return question
    try:
        msg = llm_chat([
            {"role": "system", "content": "Rewrite the user's question to be optimal for document retrieval. Keep the same language. Output ONLY the rewritten question."},
            {"role": "user", "content": question},
        ], timeout=REWRITE_TIMEOUT_S, max_attempts=1)
    except Exception as e:
        logger.warning("query rewrite skipped (%s: %s); using the original question", type(e).__name__, e)
        return question
    return (msg["content"].strip().strip('"') or question)


def _user_message(question, rq):
    text = f"Question: {question}"
    if rq and rq != question:
        text += f"\n(Rewritten for retrieval: {rq})"
    return {"role": "user", "content": text}


def grade_relevance(question, passages):
    """LLM judges if the passages can answer the question. Returns (relevant: bool, reason: str)."""
    if not passages:
        return False, "no passages"
    ctx = "\n---\n".join(str(p) for p in passages)[:3000]
    msg = llm_chat([
        {"role": "system", "content": "You are a relevance grader. Decide if the passages contain info to answer the question. Reply with JSON: {\"relevant\": true|false, \"reason\": \"...\"}."},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{ctx}"},
    ])
    txt = msg["content"].strip()
    try:
        obj = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        return bool(obj.get("relevant")), str(obj.get("reason", ""))
    except Exception:
        return ("true" in txt.lower()), txt[:200]


def faithfulness_check(answer, context):
    """Reuse simple_eval's embedding-similarity faithfulness (answer vs context)."""
    if not answer or not context:
        return 0.0
    ctx = " ".join(context) if isinstance(context, list) else str(context)
    return simple_similarity(answer, ctx)


# ---------------- safety (before_model guard / after_model disclaimer) ----------------
SENSITIVE_WORDS = ["暴力", "色情", "自杀", "恐怖", "毒品", "炸弹", "杀人", "bomb", "kill", "terror"]


def guard_input(text):
    """before_model: if a sensitive word is in the input, short-circuit (jump_to=end)."""
    for w in SENSITIVE_WORDS:
        if w in (text or ""):
            return True, f"抱歉，为了安全，不能处理包含「{w}」的请求。"
    return False, ""


DISCLAIMER = ""  # moved to a CSS element below the chat input (see style.css)


def append_disclaimer(answer):
    """after_model: append a disclaimer to the answer."""
    return (answer or "") + DISCLAIMER


TOOLS = [
    {"type": "function", "function": {"name": "search_docs", "description": "Search the uploaded documents for relevant passages. Use this for questions about the documents.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web (DuckDuckGo) for current/external info. Use this FIRST for current events, news, real-time info, recent facts, or when the user explicitly asks to search the web/website/internet.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate a math expression, e.g. 2+2, sqrt(9), sin(0). Use for calculations.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "math expression"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "wikipedia_search", "description": "Get a Wikipedia summary for a topic (external knowledge). Use for encyclopedic facts.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "topic"}}, "required": ["query"]}}},
]


def _run_tool(name, args, chunks, k):
    q = args.get("query", "")
    t0 = time.perf_counter()
    try:
        if name == "search_docs":
            return search_docs(q, chunks, k)
        if name == "search_web":
            return search_web(q)
        if name == "calculator":
            return calculator(args.get("expression", ""))
        if name == "wikipedia_search":
            return wikipedia_search(q)
        return f"unknown tool: {name}"
    finally:
        logger.info("tool %s finished in %.1fs", name, time.perf_counter() - t0)


def _agentic_answer_impl(question, chunks, k=5, on_progress=None, max_steps=8, max_self_correct=2, memory=None):
    blocked, refusal = guard_input(question)
    if blocked:
        if on_progress:
            on_progress({"type": "blocked", "word": refusal})
        return refusal, [], question
    if should_rewrite(question, chunks):
        if on_progress:
            on_progress({"type": "rewrite"})
        rq = rewrite_query(question, chunks)
        if on_progress:
            on_progress({"type": "rewrite_done", "rewritten": rq})
    else:
        rq = question
    _sys = "You are a Q&A agent with tools. RULES: (1) If the user asks about current events, news, real-time info, recent facts, OR explicitly asks to search the web/website/internet → call search_web FIRST (do NOT answer from the documents). (2) For questions about the uploaded documents → call search_docs. (3) If search_docs returns irrelevant or insufficient info → call search_web. (4) For math → call calculator. (5) For encyclopedic facts → call wikipedia_search. (6) Cite sources. After gathering info, answer concisely. Use remembered user facts to personalize answers when relevant."
    if memory:
        _sys += "\n\n## Remembered about this user (from past conversations)\n" + "\n".join(f"- {m}" for m in memory if m)
    _sys += _lang_line()
    messages = [
        {"role": "system", "content": _sys},
        _user_message(question, rq),
    ]
    gathered = []
    answer = None
    for step in range(1, max_steps + 1):
        if on_progress:
            on_progress({"type": "thinking", "step": step})
        msg = llm_chat(messages, tools=TOOLS)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            answer = msg["content"].strip()
            break
        messages.append({"role": "assistant", "content": msg["content"] or "", "tool_calls": tcs})
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            if on_progress:
                on_progress({"type": "tool_call", "name": name, "args": args})
            res = _run_tool(name, args, chunks, k)
            gathered.append({"tool": name, "query": args.get("query"), "result": res})
            if on_progress:
                on_progress({"type": "tool_result", "name": name, "n": len(res) if isinstance(res, list) else 1})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res, ensure_ascii=False)[:4000]})
    if answer is None:
        answer = "(max steps reached; could not produce an answer)"
    ctx = [s for g in gathered if g["tool"] == "search_docs" and isinstance(g["result"], list) for s in g["result"] if isinstance(s, str)]
    for attempt in range(max_self_correct):
        score = faithfulness_check(answer, ctx) if ctx else 0.0
        if on_progress:
            on_progress({"type": "self_check", "score": round(score, 3), "attempt": attempt + 1})
        if score >= 0.4 or not ctx:
            break
        messages.append({"role": "user", "content": "Your previous answer wasn't well grounded in the retrieved context. Answer again using only the gathered context."})
        answer = llm_chat(messages, tools=None)["content"].strip()
    _uncertain = any(g["tool"] in ("search_web", "wikipedia_search") for g in gathered)
    if _uncertain:
        answer = append_disclaimer(answer)
    _wiki_url = next((g["result"].get("url") for g in gathered if g["tool"] == "wikipedia_search" and isinstance(g["result"], dict) and g["result"].get("url")), None)
    if _wiki_url:
        answer += f"\n\n[Source: Wikipedia]({_wiki_url})"
    return answer, gathered, rq


@traceable(run_type="chain", name="agentic_answer", process_input=_trim_chunks)
def agentic_answer(question, chunks, k=5, on_progress=None, max_steps=8, max_self_correct=2, memory=None):
    """Agentic RAG: [guard] -> [rewrite] -> native tool-calling loop -> faithfulness self-correction -> [disclaimer].
    Returns (answer, gathered, rewritten_query)."""
    with _retry_events(on_progress):
        return _agentic_answer_impl(question, chunks, k, on_progress, max_steps, max_self_correct, memory)


def _agentic_answer_stream_impl(question, chunks, k=5, on_progress=None, max_steps=8, memory=None):
    blocked, refusal = guard_input(question)
    if blocked:
        if on_progress:
            on_progress({"type": "blocked", "word": refusal})
        yield refusal
        return
    if should_rewrite(question, chunks):
        if on_progress:
            on_progress({"type": "rewrite"})
        rq = rewrite_query(question, chunks)
    else:
        rq = question
    _sys = "You are a Q&A agent with tools. RULES: (1) If the user asks about current events, news, real-time info, recent facts, OR explicitly asks to search the web/website/internet \u2192 call search_web FIRST (do NOT answer from the documents). (2) For questions about the uploaded documents \u2192 call search_docs. (3) If search_docs returns irrelevant or insufficient info \u2192 call search_web. (4) For math \u2192 call calculator. (5) For encyclopedic facts \u2192 call wikipedia_search. (6) Cite sources. After gathering info, answer concisely. Use remembered user facts to personalize answers when relevant."
    if memory:
        _sys += "\n\n## Remembered about this user (from past conversations)\n" + "\n".join(f"- {m}" for m in memory if m)
    _sys += _lang_line()
    messages = [
        {"role": "system", "content": _sys},
        _user_message(question, rq),
    ]
    gathered = []
    answer_parts = []
    for step in range(1, max_steps + 1):
        if on_progress:
            on_progress({"type": "thinking", "step": step})
        _out = {}
        for delta in llm_chat_stream(messages, tools=TOOLS, _out=_out):
            answer_parts.append(delta)
            yield delta
        tcs = _out.get("tool_calls")
        if not tcs:
            break
        messages.append({"role": "assistant", "content": "".join(answer_parts) or "", "tool_calls": tcs})
        answer_parts = []
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            if on_progress:
                on_progress({"type": "tool_call", "name": name, "args": args})
            res = _run_tool(name, args, chunks, k)
            gathered.append({"tool": name, "query": args.get("query"), "result": res})
            if on_progress:
                on_progress({"type": "tool_result", "name": name, "n": len(res) if isinstance(res, list) else 1})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res, ensure_ascii=False)[:4000]})
    answer = "".join(answer_parts).strip()
    if not answer:
        answer = "(max steps reached; could not produce an answer)"
        yield answer
    _uncertain = any(g["tool"] in ("search_web", "wikipedia_search") for g in gathered)
    if _uncertain:
        for ch in DISCLAIMER:
            yield ch
    _wiki_url = next((g["result"].get("url") for g in gathered if g["tool"] == "wikipedia_search" and isinstance(g["result"], dict) and g["result"].get("url")), None)
    if _wiki_url:
        src = f"\n\n[Source: Wikipedia]({_wiki_url})"
        for ch in src:
            yield ch


def agentic_answer_stream(question, chunks, k=5, on_progress=None, max_steps=8, memory=None):
    """Streaming agentic RAG: [guard] -> [rewrite] -> native tool-calling loop (streaming the
    final answer token-by-token via llm_chat_stream) -> [disclaimer]. Yields answer tokens.
    NOTE: faithfulness self-correction is skipped in stream mode (the loop's answer is final)."""
    with _retry_events(on_progress):
        yield from _agentic_answer_stream_impl(question, chunks, k, on_progress, max_steps, memory)
