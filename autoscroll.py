"""
autoscroll.py
=============
Keeps the chat scrolled to the newest text, like Claude / ChatGPT:
  * while an answer streams in, the page follows it;
  * if you scroll up to read something, it stops following (and starts again when you
    scroll back to the bottom);
  * sending a message or pressing retry jumps to the bottom straight away.

    import autoscroll
    autoscroll.install(force=<True when a question is about to be answered>)

Call it ONCE per run, at the top of the chat view (before the message history), with
force=True on the runs where a new answer is generated (when pending_q was set).

It also turns the chat box's send arrow into a STOP square while an answer is being written:
step_ui.run_assistant_turn renders a hidden button (.st-key-stop_proxy_wrap) for exactly that
time; this script draws a square button over the send arrow and, when it is clicked, clicks
that hidden button, which makes Streamlit interrupt the run (step_ui.settle_interrupted then
keeps the text written so far). Styles for both live in style.css.
"""
import inspect
import time

import streamlit as st
import streamlit.components.v1 as components

_JS = r"""
(function () {
  const w = (window.frameElement && window.parent) ? window.parent : window;   // old Streamlit: we live in an iframe
  let doc;
  try { doc = w.document; } catch (e) { return; }
  const FORCE = "__FORCE__";
  const NEAR = 160;                                    // px from the bottom that still counts as "at the bottom"
  const S = (w.__nexusScroll = w.__nexusScroll || { stick: true, token: null, raf: 0, installed: false });

  function scroller() {
    const sels = ['section[data-testid="stMain"]', '[data-testid="stMain"]', 'section.main',
                  '[data-testid="stAppViewContainer"]', '.main'];
    for (const s of sels) {
      const el = doc.querySelector(s);
      if (el && el.scrollHeight > el.clientHeight + 2) return el;
    }
    return doc.scrollingElement || doc.documentElement;
  }
  function toBottom() {
    const el = scroller();
    if (el) el.scrollTop = el.scrollHeight;
  }
  function schedule() {
    if (S.raf) return;
    S.raf = w.requestAnimationFrame(() => { S.raf = 0; if (S.stick) toBottom(); });
  }

  if (!S.installed) {
    S.installed = true;
    // the user scrolling decides whether we keep following
    doc.addEventListener('scroll', (e) => {
      const el = scroller();
      const t = e.target;
      if (!el || (t !== el && t !== doc)) return;
      S.stick = (el.scrollHeight - el.scrollTop - el.clientHeight) < NEAR;
    }, true);
    // new / growing content: follow it
    new w.MutationObserver(schedule).observe(doc.body, { childList: true, subtree: true, characterData: true });
  }

  // ---- stop square over the send arrow while an answer is being written ----
  const X = (w.__nexusStop = w.__nexusStop || { installed: false, btn: null });
  function proxy() {                                   // the hidden Streamlit button, only while generating
    const wrap = doc.querySelector('.st-key-stop_proxy_wrap');
    if (!wrap || (wrap.closest && wrap.closest('[data-stale="true"]'))) return null;
    return wrap.querySelector('button');
  }
  function syncStop() {
    const generating = !!proxy();
    doc.documentElement.classList.toggle('nexus-generating', generating);
    if (!X.btn) return;
    const send = doc.querySelector('[data-testid="stChatInputSubmitButton"]');
    if (!generating || !send) { X.btn.style.display = 'none'; return; }
    const r = send.getBoundingClientRect();
    if (!r.width) { X.btn.style.display = 'none'; return; }
    X.btn.style.display = 'block';
    X.btn.style.left = r.left + 'px';  X.btn.style.top = r.top + 'px';
    X.btn.style.width = r.width + 'px'; X.btn.style.height = r.height + 'px';
  }
  if (!X.installed) {
    X.installed = true;
    X.btn = doc.createElement('button');
    X.btn.type = 'button';
    X.btn.className = 'nexus-stop';
    X.btn.setAttribute('aria-label', 'Stop');
    X.btn.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation();
      const p = proxy();
      if (p) p.click();
    });
    doc.body.appendChild(X.btn);
    new w.MutationObserver(syncStop).observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-stale'] });
    w.setInterval(syncStop, 250);                      // keeps the square glued to the arrow (resizes, growing text box)
  }
  syncStop();

  // a new question or a retry: jump to the bottom and follow for a moment while the page settles
  if (FORCE !== "0" && S.token !== FORCE) {
    S.token = FORCE;
    S.stick = true;
    let n = 0;
    const t = w.setInterval(() => { toBottom(); if (++n > 12) w.clearInterval(t); }, 150);
  }
})();
"""


def _run_js(js):
    """st.html can run scripts in newer Streamlit; older versions need an iframe component."""
    html = f"<script>{js}</script>"
    if "unsafe_allow_javascript" in inspect.signature(st.html).parameters:
        st.html(html, unsafe_allow_javascript=True)
    else:
        components.html(html, height=0)


def install(force=False):
    token = str(time.time_ns()) if force else "0"       # a new token makes the script run its "jump to bottom" again
    with st.container(key="autoscroll"):
        _run_js(_JS.replace("__FORCE__", token))
