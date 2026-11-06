"""
answer_actions.py
=================
The small grey icon row under an answer, like Claude / Gemini:  copy   retry

    import answer_actions
    answer_actions.render(msg_idx, msg, user_email, is_last)

* Copy  : a tiny HTML iframe (the browser clipboard needs a real click inside the browser).
          render_user() puts the same copy icon under the messages the user sent.
* Retry : only on the newest answer. Drops that answer, removes its entry from the semantic
          cache (otherwise the same cached answer would come straight back), and asks the
          same question again through the normal pending_q flow in app.py.
Styling lives in style.css (search for "answer action icons").
"""
import base64
import hashlib
import html as _html

import streamlit as st
import streamlit.components.v1 as components

from i18n import t

_COPY_HTML = """
<style>
  html, body { margin: 0; background: transparent; }
  button { width: 32px; height: 32px; border: none; background: transparent; border-radius: 8px; cursor: pointer;
           display: flex; align-items: center; justify-content: center; color: #9a978d; padding: 0; }
  button:hover { background: rgba(0,0,0,.06); color: #1f1e1d; }
  svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
  .ok { display: none; }
</style>
<button id="b" title="__LABEL__" aria-label="__LABEL__">
  <svg class="cp" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M5 15V6.5A2.5 2.5 0 0 1 7.5 4H15"/></svg>
  <svg class="ok" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>
</button>
<script>
  const TEXT = new TextDecoder().decode(Uint8Array.from(atob("__B64__"), c => c.charCodeAt(0)));
  const b = document.getElementById('b');
  function fallback() {
    const t = document.createElement('textarea');
    t.value = TEXT; t.style.position = 'fixed'; t.style.opacity = '0';
    document.body.appendChild(t); t.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(t);
  }
  b.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(TEXT); } catch (e) { fallback(); }
    b.querySelector('.cp').style.display = 'none';
    b.querySelector('.ok').style.display = 'block';
    setTimeout(() => { b.querySelector('.cp').style.display = 'block'; b.querySelector('.ok').style.display = 'none'; }, 1500);
  });
</script>
"""


def _copy_html(text):
    """The answer travels as base64, so no character in it can ever break out of the script."""
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return _COPY_HTML.replace("__B64__", b64).replace("__LABEL__", _html.escape(t("action.copy")))


def _embed(html, height, width="stretch"):
    """st.iframe on new Streamlit versions; components.html (removed in newer releases) on old ones."""
    if hasattr(st, "iframe"):
        st.iframe(html, width=width, height=height)
    else:
        components.html(html, height=height, width=None if width == "stretch" else width)


def _question_before(msgs, idx):
    for j in range(min(idx, len(msgs)) - 1, -1, -1):
        if msgs[j].get("role") == "user":
            return str(msgs[j].get("content", ""))
    return ""


def _retry(idx, key):
    """Drop the answer at idx and ask the same question again."""
    msgs = st.session_state.get("messages", [])
    if idx < 1 or idx >= len(msgs) or msgs[idx].get("role") != "assistant":
        return
    question = _question_before(msgs, idx)
    old = str(msgs[idx].get("content", ""))
    st.session_state.messages = msgs[:idx]                       # keeps the user's question as the last message
    st.session_state["semantic_cache"] = [                       # otherwise the cache returns the same answer
        e for e in st.session_state.get("semantic_cache", [])
        if e.get("answer") != old and e.get("question") != question
    ]
    st.session_state["pending_q"] = question                     # app.py answers it on the next run


def render(idx, msg, email, is_last=False):
    """Draw the icon row for the assistant message `msg` that sits (or will sit) at index idx."""
    text = str(msg.get("content", ""))
    key = f"{idx}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}"   # unique per message text
    with st.container(key=f"act_{key}"):
        cols = st.columns([1, 1, 22] if is_last else [1, 23], vertical_alignment="center")
        with cols[0]:
            _embed(_copy_html(text), 36)
        if is_last:
            with cols[1]:
                st.button(t("action.retry"), icon=":material/refresh:", key=f"rt_{key}", help=t("action.retry"),
                          on_click=_retry, args=(idx, key))


def render_user(idx, msg):
    """The copy icon under a message the user sent (right-aligned under the bubble)."""
    text = str(msg.get("content", ""))
    key = f"{idx}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}"
    with st.container(key=f"ucopy_{key}"):
        _embed(_copy_html(text), 36, width=36)
