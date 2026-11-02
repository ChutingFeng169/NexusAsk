"""
step_ui.py
==========
One status box per assistant turn, plus a saved step history.

Why this file exists
--------------------
Every place that answers a question (main chat, review follow-up, review retry,
edit-and-resend) used to build its own status widget. This module gives them ONE
shared helper, so the page never shows duplicate status boxes, and the step list
is recorded once and saved with the message.

Usage (see the snippets in the chat reply):
    answer, steps = step_ui.run_assistant_turn(
        lambda status: agentic_answer_cached(q, chunks, k, status, email=user_email))
    st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})

In the history loop:
    step_ui.render_steps(m.get("steps"))
"""

import logging
import time

import streamlit as st

logger = logging.getLogger("nexus.ui")

FRIENDLY_ERROR = "Nexus couldn't reach the model. Please try again."


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
                self.status.update(label=f"{label or 'Done'} · {total:.1f}s", state=state)
        except Exception:
            logger.debug("status update failed", exc_info=True)

    def finish(self, ok=True):
        self._close()
        if self._final is None:
            self._final = "complete" if ok else "error"
            total = time.perf_counter() - self._start
            try:
                self.status.update(label=f"{'Done' if ok else 'Failed'} · {total:.1f}s", state=self._final)
            except Exception:
                pass


def run_assistant_turn(make_stream, footer=None):
    """
    Render ONE assistant message: a single collapsed status box, then the streamed answer.

    make_stream(status) -> generator of answer tokens. Pass `status` on to code that
        calls status.update(label=..., state=...).
    footer(answer_text) -> optional; called inside the message after streaming
        (token count caption, LangSmith trace, ...).

    Returns (answer_text, steps). answer_text is "" if the turn failed; the user sees one
    friendly error and the real exception goes to the log.
    """
    with st.chat_message("assistant"):
        status = st.status("Nexus is thinking…", expanded=False)
        tracker = StepTracker(status)
        answer_box = st.empty()  # created BEFORE any slow call, so stale content is cleared at once
        text = ""
        try:
            with answer_box.container():
                text = st.write_stream(make_stream(tracker)) or ""
            if not str(text).strip():
                raise RuntimeError("empty answer")
            tracker.finish(ok=True)
        except Exception:
            logger.exception("assistant turn failed")
            tracker.finish(ok=False)
            answer_box.error(FRIENDLY_ERROR)
            return "", tracker.steps
        if footer:
            footer(text)
    return str(text), tracker.steps


def render_steps(steps):
    """Collapsed 'Show steps' list under a saved assistant message."""
    if not steps:
        return
    with st.expander("Show steps"):
        for s in steps:
            st.caption(f"{s['action']} — {s['seconds']}s")
