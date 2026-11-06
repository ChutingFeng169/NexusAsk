"""
step_ui.py
==========
One assistant turn in the chat: a single quiet status line, the streamed answer, a stop button
that turns an unfinished answer into an "interrupted" message you can continue, and a failure
row with a red retry icon.

What app.py calls (main chat):
    step_ui.settle_interrupted()                 # top of the chat view, before the history loop
    step_ui.render_history_message(i, m, is_last, email)    # inside the history loop, per message
    step_ui.run_assistant_turn(make_stream, footer=..., continue_idx=...)   # to answer
    step_ui.render_unanswered()                  # after the history loop when nothing is being answered

run_assistant_turn also STORES the finished answer in st.session_state.messages
(app.py must no longer append it itself).

How "stop" works: while an answer is generated, a hidden st.button ("stop_proxy") exists.
autoscroll.py's script turns the chat box's send arrow into a stop square and clicks that
hidden button, which makes Streamlit interrupt the running script and rerun. The text streamed
so far is kept in st.session_state["_gen"]; settle_interrupted() turns it into a message marked
"interrupted". "Click to continue" asks the model to carry on from where it stopped.
"""
import html as _html
import logging
import time

import streamlit as st

import answer_actions
from i18n import t

logger = logging.getLogger("nexus.ui")

CONTINUE_PROMPT = (
    "Your previous reply was cut off. Continue exactly from where it stopped, without repeating "
    "what you already wrote, without any introduction, and do not use any tools."
)


# --------------------------------------------------------------------------- step tracker
class StepTracker:
    """Wraps ONE st.status. Has the same .update(label, state) signature as st.status,
    so the existing agentic reporters keep working. Records (action, seconds) per step."""

    def __init__(self, status):
        self.status = status
        self.steps = []
        self._cur = None
        self._t0 = 0.0
        self._start = time.perf_counter()
        self._final = None  # "complete" / "error" once the agent reports it

    def _close(self):
        if self._cur:
            sec = round(time.perf_counter() - self._t0, 1)
            self.steps.append({"action": self._cur, "seconds": sec})
            try:
                self.status.write(f"{self._cur} — {sec}s")
            except Exception:
                pass
            self._cur = None

    def update(self, label="", state="running", **kw):
        label = (label or "").strip()
        try:
            if state == "running":
                self._close()
                if label:
                    self._cur, self._t0 = label.rstrip("…. "), time.perf_counter()
                    self.status.update(label=label, state="running")
            else:
                self._close()
                self._final = state
                total = time.perf_counter() - self._start
                self.status.update(label=f"{label or t('turn.done')} · {total:.1f}s", state=state)
        except Exception:
            logger.debug("status update failed", exc_info=True)

    def finish(self, ok=True):
        self._close()
        if self._final is None:
            self._final = "complete" if ok else "error"
            total = time.perf_counter() - self._start
            try:
                self.status.update(label=f"{t('turn.done') if ok else t('turn.failed')} · {total:.1f}s", state=self._final)
            except Exception:
                pass


# --------------------------------------------------------------------------- helpers
def _question_before(msgs, idx):
    for j in range(min(idx, len(msgs)) - 1, -1, -1):
        if msgs[j].get("role") == "user":
            return str(msgs[j].get("content", ""))
    return ""


def _is_cjk(ch):
    return "\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef"


def _join(first, more):
    """Glue a continuation onto the text written before the interruption."""
    first, more = first.rstrip(), more.lstrip()
    if not first:
        return more
    if not more:
        return first
    if first.endswith("\n") or _is_cjk(first[-1]) or _is_cjk(more[0]) or more[0] in ".,;:!?)":
        return first + more
    return first + " " + more


def _tracked(stream, gen):
    """Pass tokens through while remembering what has been written (for a stop)."""
    for tok in stream:
        gen["partial"] += tok
        yield tok


# --------------------------------------------------------------------------- failure row (red retry icon)
def _retry_failed(question):
    """Ask the same question again (app.py answers whatever is in pending_q on the next run)."""
    st.session_state["pending_q"] = question


def _auto_question(retry_question):
    """"auto" = the user's last message, but only in the main chat (the review page has its own Retry)."""
    if retry_question != "auto":
        return retry_question or None
    if "review_conversation" in st.session_state:
        return None
    msgs = st.session_state.get("messages", [])
    if msgs and msgs[-1].get("role") == "user":
        return str(msgs[-1].get("content", ""))
    return None


def _failure_row(question, key):
    """The error message with a red retry icon right next to it."""
    with st.container(key=f"fail_{key}"):
        c1, c2 = st.columns([24, 1], vertical_alignment="center")
        c1.markdown(f'<div class="fail-msg">{_html.escape(t("turn.error"))}</div>', unsafe_allow_html=True)
        if question:
            c2.button(t("action.retry"), icon=":material/refresh:", key=f"failrt_{key}", help=t("action.retry"),
                      on_click=_retry_failed, args=(question,))


def render_unanswered():
    """Call after the history loop when no answer is being generated: if the last message is a
    question without an answer, show the error + retry icon so the user is never stuck."""
    msgs = st.session_state.get("messages", [])
    if msgs and msgs[-1].get("role") == "user" and "review_conversation" not in st.session_state:
        with st.chat_message("assistant"):
            _failure_row(str(msgs[-1].get("content", "")), "stale")


# --------------------------------------------------------------------------- interrupted answers
def settle_interrupted():
    """Call at the top of the chat view, before the history loop.
    If the previous run was cut off while answering (stop button, or a new message), keep what was
    written so far as a message marked "interrupted"."""
    gen = st.session_state.pop("_gen", None)
    if not gen:
        return
    msgs = st.session_state.setdefault("messages", [])
    complete = bool(gen.get("complete"))
    partial = gen.get("partial", "")
    if gen.get("mode") == "continue":
        i = gen.get("idx")
        if i is not None and 0 <= i < len(msgs):
            msgs[i]["content"] = _join(gen.get("base", ""), partial)
            msgs[i]["steps"] = (msgs[i].get("steps") or []) + list(gen.get("steps", []))
            if complete:
                msgs[i].pop("interrupted", None)
            else:
                msgs[i]["interrupted"] = True
    elif msgs and msgs[-1].get("role") == "user":
        m = {"role": "assistant", "content": partial, "steps": list(gen.get("steps", []))}
        if not complete:
            m["interrupted"] = True
        msgs.append(m)


def _continue(idx):
    msgs = st.session_state.get("messages", [])
    if idx >= len(msgs):
        return
    m = msgs[idx]
    if not str(m.get("content", "")).strip():
        # nothing was written before the stop: simply ask the question again
        question = _question_before(msgs, idx)
        st.session_state.messages = msgs[:idx]
        st.session_state["pending_q"] = question
        return
    m.pop("interrupted", None)
    st.session_state["pending_continue"] = idx


def render_interrupted(idx, is_last):
    """"Response interrupted." with a "Click to continue" button next to it."""
    with st.container(key=f"intr_{idx}"):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.markdown(f'<div class="intr-msg">{_html.escape(t("turn.interrupted"))}</div>', unsafe_allow_html=True)
        if is_last:
            c2.button(t("turn.continue"), key=f"cont_{idx}", on_click=_continue, args=(idx,))


# --------------------------------------------------------------------------- history
def render_steps(steps):
    """Collapsed 'Show steps' list under a saved assistant message."""
    if not steps:
        return
    with st.expander(t("turn.show_steps")):
        for s in steps:
            st.caption(f"{s['action']} — {s['seconds']}s")


def render_history_message(idx, msg, is_last, email=""):
    """Everything under one saved message: its text, then copy / retry icons (or the interrupted
    notice), then the steps. Use inside `with st.chat_message(role):`."""
    text = str(msg.get("content", ""))
    if msg.get("role") == "user":
        st.write(text)
        answer_actions.render_user(idx, msg)
        return
    if text.strip():
        st.write(text)
    if msg.get("interrupted"):
        render_interrupted(idx, is_last)
    elif text.strip():
        answer_actions.render(idx, msg, email, is_last=is_last)
    render_steps(msg.get("steps"))


# --------------------------------------------------------------------------- one turn
def run_assistant_turn(make_stream, footer=None, retry_question="auto", continue_idx=None):
    """
    Render ONE assistant message: a single quiet status line, then the streamed answer, and store
    the finished answer in st.session_state.messages.

    make_stream(status) -> generator of answer tokens. Pass `status` on to code that
        calls status.update(label=..., state=...).
    footer(answer_text) -> optional; called inside the message after streaming (developer info ...).
        Not called when continuing an interrupted answer.
    retry_question: the question to ask again when the red retry icon is clicked. "auto" uses the
        user's last message (main chat); pass None to show no retry icon.
    continue_idx: index of an interrupted message to carry on; the new text is appended to it.

    Returns (answer_text, steps); answer_text is "" if the turn failed (friendly error + retry icon).
    """
    msgs = st.session_state.setdefault("messages", [])
    base = ""
    if continue_idx is not None and 0 <= continue_idx < len(msgs):
        base = str(msgs[continue_idx].get("content", ""))
    else:
        continue_idx = None
    gen = {"mode": "continue" if continue_idx is not None else "new", "idx": continue_idx,
           "base": base, "partial": "", "steps": [], "complete": False}
    st.session_state["_gen"] = gen

    # hidden proxy for the stop button: its presence tells the page "an answer is being written"
    stop_slot = st.empty()
    with stop_slot.container(key="stop_proxy_wrap"):
        st.button(t("action.stop"), key="stop_proxy")

    with st.chat_message("assistant"):
        status = st.status(t("turn.thinking"), expanded=False)
        tracker = StepTracker(status)
        gen["steps"] = tracker.steps
        answer_box = st.empty()  # created BEFORE any slow call, so stale content is cleared at once
        text = ""
        try:
            with answer_box.container():
                text = st.write_stream(_tracked(make_stream(tracker), gen)) or ""
            if not str(text).strip():
                raise RuntimeError("empty answer")
            tracker.finish(ok=True)
        except Exception:
            logger.exception("assistant turn failed")
            tracker.finish(ok=False)
            stop_slot.empty()
            st.session_state.pop("_gen", None)
            with answer_box.container():
                _failure_row(_auto_question(retry_question), "live")
            return "", tracker.steps
        stop_slot.empty()
        gen["complete"] = True            # from here on a cut-off still counts as a finished answer
        text = str(text)
        if footer and continue_idx is None:
            footer(text)

    msgs = st.session_state.messages
    if continue_idx is not None:
        m = msgs[continue_idx]
        m["content"] = _join(base, text)
        m.pop("interrupted", None)
        m["steps"] = (m.get("steps") or []) + tracker.steps
    else:
        msgs.append({"role": "assistant", "content": text, "steps": tracker.steps})
    st.session_state.pop("_gen", None)
    return text, tracker.steps


