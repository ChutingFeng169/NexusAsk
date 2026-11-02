"""sidebar_ui.py — sidebar, search dialog, settings/language/help dialogs.

Exposes one function: render_sidebar(deps), where deps carries:
  user_email, user_name, archive_current, load_user_convs, save_user_convs,
  confirm_delete_conv, confirm_clear_all, confirm_logout, upgrade_title_async,
  _title_in_progress, BASE_DIR

Does NOT import app.py (no circular import). Imports memory + agentic_rag directly.
"""
import json
import os
import re
import threading
import streamlit as st
import memory
import agentic_rag

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_load_convs = None   # set by render_sidebar from deps
_save_convs = None   # set by render_sidebar from deps


def _snippet(c, ql):
    """Short excerpt around the first message hit (empty if the match was only in the title)."""
    for m in c.get("messages", []):
        t = str(m.get("content", ""))
        i = t.lower().find(ql)
        if i >= 0:
            a, b = max(0, i - 30), min(len(t), i + len(ql) + 50)
            return ("…" if a else "") + t[a:b].replace("\n", " ") + ("…" if b < len(t) else "")
    return ""


def find_chats(convs, q, limit=20):
    """[(index, snippet)] of chats whose title or any message contains q (newest first)."""
    ql = (q or "").strip().lower()
    if not ql:
        return []
    out = []
    for idx in range(len(convs) - 1, -1, -1):
        c = convs[idx]
        title = (c.get("title") or c.get("filename", "")).lower()
        snip = _snippet(c, ql)
        if ql in title or snip:
            out.append((idx, snip))
        if len(out) >= limit:
            break
    return out


def _open_chat(idx):
    st.session_state.review_conversation = idx
    for k in ("preview_file", "library_view", "preview_from_library"):
        if k in st.session_state: del st.session_state[k]
    st.rerun()


# ---- per-user settings (saved on disk): answer language + developer info ----
SETTINGS_DIR = os.path.join(_BASE_DIR, "user_settings")
DEFAULT_SETTINGS = {"language": "auto", "dev_info": False}
LANGUAGES = {"auto": "Auto (match my question)", "English": "English", "Simplified Chinese": "中文 (简体)"}


def _settings_file(email):
    return os.path.join(SETTINGS_DIR, f"{re.sub(r'[^a-zA-Z0-9._-]', '_', email)}.json")


def load_user_settings(email):
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(_settings_file(email), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if data.get("language") in LANGUAGES: s["language"] = data["language"]
            s["dev_info"] = bool(data.get("dev_info", False))
    except (OSError, ValueError):
        pass
    return s


def save_user_settings(email, s):
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(_settings_file(email), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def _delete_with_confirm(label, key, action, full_rerun=False):
    """Button -> 'This cannot be undone' -> Yes/Cancel, all inside the dialog."""
    if st.button(label, key=f"{key}_ask"):
        st.session_state[key] = True
    if st.session_state.get(key):
        st.caption("This cannot be undone.")
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete", key=f"{key}_yes", type="primary"):
            action()
            st.session_state[key] = False
            if full_rerun:
                st.rerun()
            st.caption("Deleted.")
        if c2.button("Cancel", key=f"{key}_no"):
            st.session_state[key] = False
            st.rerun(scope="fragment")


@st.dialog("Settings")
def settings_dialog(email, name):
    s = load_user_settings(email)
    st.markdown("**Account**")
    st.caption(f"{name} · {email}")
    st.markdown("**Display**")
    dev = st.toggle("Show developer info under answers", value=s["dev_info"],
                    help="Token count, cache status and the LangSmith trace.")
    if dev != s["dev_info"]:
        s["dev_info"] = dev
        save_user_settings(email, s)
        st.session_state["dev_info"] = dev
    st.markdown("**Your data**")
    st.caption("NexusAsk remembers facts from past chats to personalize answers.")
    _delete_with_confirm("Clear long-term memory", "_cf_mem", lambda: memory.clear_memory(email))
    _delete_with_confirm("Clear chat history", "_cf_chats", lambda: _save_convs(email, []), full_rerun=True)
    _delete_with_confirm("Clear search history", "_cf_search", lambda: clear_searches(email))


@st.dialog("Language")
def language_dialog(email):
    s = load_user_settings(email)
    st.caption("The language NexusAsk answers in. The interface itself stays in English.")
    keys = list(LANGUAGES)
    choice = st.radio("Answer language", keys, index=keys.index(s["language"]),
                      format_func=lambda k: LANGUAGES[k], label_visibility="collapsed")
    if st.button("Save", key="lang_save", type="primary"):
        s["language"] = choice
        save_user_settings(email, s)
        st.rerun()


@st.dialog("Get help", width="large")
def help_dialog():
    st.markdown("**Welcome to NexusAsk**")
    st.write("An assistant that answers from your documents, the web and Wikipedia, and shows you what it did along the way.")
    st.markdown("**What you can do**")
    st.markdown(
        "- **Ask anything.** NexusAsk picks the right tool by itself: your documents, web search, Wikipedia or a calculator.\n"
        "- **Chat about your files.** Use the + next to the message box, or drag and drop. Supported: PDF, Word, PowerPoint, Excel, TXT, Markdown, CSV.\n"
        "- **Find files again** in Library. Everything you upload is kept there.\n"
        "- **Come back to a chat.** Chats are saved automatically. Use New chat to start fresh, the three dots on a chat to rename or delete it, and the magnifier to search every chat.\n"
        "- **See how an answer was made.** Open Show steps under a reply to see each step and how long it took.\n"
        "- **Get personalized answers.** NexusAsk remembers useful facts from your past chats. You can clear this in Settings."
    )
    st.markdown("**Good to know**")
    st.markdown(
        "- Mention a website or say \"search the web\" to make it look online first.\n"
        "- Answers that use the web or Wikipedia end with a reminder that they are AI-generated, so check important facts.\n"
        "- Very similar questions in the same session may be answered from a saved earlier answer, which is faster.\n"
        "- Requests that contain certain sensitive words are declined."
    )


# ---- search history: saved on disk per user, kept until the user removes it ----
SEARCH_DIR = os.path.join(_BASE_DIR, "search_history")
MAX_SEARCHES = 100


def _search_file(email):
    return os.path.join(SEARCH_DIR, f"{re.sub(r'[^a-zA-Z0-9._-]', '_', email)}.json")


def load_search_history(email):
    """Newest first. A missing or damaged file just means an empty history."""
    try:
        with open(_search_file(email), "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(x) for x in data] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_search_history(email, history):
    os.makedirs(SEARCH_DIR, exist_ok=True)
    with open(_search_file(email), "w", encoding="utf-8") as f:
        json.dump(history[:MAX_SEARCHES], f, ensure_ascii=False, indent=2)


def log_search(email, term):
    """Put term at the top (no duplicates)."""
    term = (term or "").strip()
    if not term:
        return
    save_search_history(email, [term] + [x for x in load_search_history(email) if x != term])


def remove_search(email, term):
    save_search_history(email, [x for x in load_search_history(email) if x != term])


def clear_searches(email):
    save_search_history(email, [])


def _on_search_submit(email):
    log_search(email, st.session_state.get("dlg_query", ""))


def _use_search(email, term):
    st.session_state["dlg_query"] = term
    log_search(email, term)


@st.dialog("Search chats", width="large")
def search_dialog(email):
    convs = _load_convs(email)
    history = load_search_history(email)
    q = st.text_input("Search", key="dlg_query", placeholder="Search chats (press Enter)",
                      label_visibility="collapsed", on_change=_on_search_submit, args=(email,))
    ql = q.strip()

    if ql:
        results = find_chats(convs, ql)
        st.markdown('<div class="search-label">Results</div>', unsafe_allow_html=True)
        with st.container(key="search_results"):
            for idx, snip in results:
                if st.button(convs[idx].get("title") or convs[idx].get("filename", "?"), key=f"srch_{idx}", use_container_width=True):
                    _open_chat(idx)
                if snip: st.caption(snip)
            if not results:
                st.caption("No matching chats")
    else:
        if history:
            _h1, _h2 = st.columns([4, 1])
            _h1.markdown('<div class="search-label">Recent searches</div>', unsafe_allow_html=True)
            _h2.button("Clear all", key="srch_clear", on_click=clear_searches, args=(email,))
            with st.container(key="search_recent"):
                for j, h in enumerate(history[:8]):
                    _r1, _r2 = st.columns([12, 1], vertical_alignment="center")
                    _r1.button(h, icon=":material/history:", key=f"srch_h_{j}", use_container_width=True,
                               on_click=_use_search, args=(email, h))
                    _r2.button("Remove", icon=":material/close:", key=f"srch_x_{j}", help="Remove from history",
                               on_click=remove_search, args=(email, h))
        if convs:
            st.markdown('<div class="search-label">Recent chats</div>', unsafe_allow_html=True)
            with st.container(key="search_results"):
                for idx in range(len(convs) - 1, max(len(convs) - 9, -1), -1):
                    if st.button(convs[idx].get("title") or convs[idx].get("filename", "?"), key=f"srch_r_{idx}", use_container_width=True):
                        _open_chat(idx)
        else:
            st.caption("No conversations yet")


def render_sidebar(deps):
    """Render the sidebar. deps carries: user_email, user_name, archive_current,
    load_user_convs, save_user_convs, confirm_delete_conv, confirm_clear_all,
    confirm_logout, upgrade_title_async, _title_in_progress, BASE_DIR."""
    global _load_convs, _save_convs
    _load_convs = deps.load_user_convs
    _save_convs = deps.save_user_convs
    user_email = deps.user_email
    user_name = deps.user_name
    archive_current = deps.archive_current
    load_user_convs = deps.load_user_convs
    save_user_convs = deps.save_user_convs
    confirm_delete_conv = deps.confirm_delete_conv
    confirm_logout = deps.confirm_logout
    upgrade_title_async = deps.upgrade_title_async
    _title_in_progress = deps._title_in_progress

    _settings = load_user_settings(user_email)
    st.session_state["dev_info"] = _settings["dev_info"]
    agentic_rag.set_response_language(_settings["language"])

    with st.sidebar:
        with st.container(key="nav_search"):
            if st.button("Search", icon=":material/search:", key="btn_search", help="Search chats"):
                search_dialog(user_email)

        with st.container(key="brand_wrap"):
            st.markdown('<div class="brand">NexusAsk</div>', unsafe_allow_html=True)

        if st.button("New chat", icon=":material/add:", key="nav_new", use_container_width=True):
            with st.spinner("Saving conversation…"):
                archive_current()
            for key in ["messages", "doc_chunks", "uploaded_filenames", "processed_files", "file_raw_bytes", "applied_chunk_size", "applied_chunk_overlap", "preview_file", "review_conversation", "show_uploader", "library_view", "preview_from_library"]:
                if key in st.session_state: del st.session_state[key]
            if "continuing_conversation_idx" in st.session_state: del st.session_state["continuing_conversation_idx"]
            st.rerun()

        if st.button("Library", icon=":material/folder_open:", key="nav_library", use_container_width=True):
            st.session_state.library_view = True
            if "review_conversation" in st.session_state: del st.session_state.review_conversation
            if "preview_file" in st.session_state: del st.session_state.preview_file
            st.rerun()

        convs = load_user_convs(user_email)
        for _c in convs:
            _cid = _c.get("id")
            if _c.get("title_is_final") is False and _cid and _cid not in _title_in_progress:
                _title_in_progress.add(_cid)
                threading.Thread(target=upgrade_title_async, args=(user_email, _cid), daemon=True).start()

        st.markdown('<div class="side-label">Chats</div>', unsafe_allow_html=True)
        if convs:
            _active_idx = st.session_state.get("review_conversation", st.session_state.get("continuing_conversation_idx"))
            for idx in range(len(convs) - 1, -1, -1):
                _name = convs[idx].get("title") or convs[idx].get("filename", "?")
                _frame_key = f"convframe_active_{idx}" if idx == _active_idx else f"convframe_{idx}"
                with st.container(border=False, key=_frame_key):
                    _t1, _t2 = st.columns([8, 1])
                    with _t1:
                        if st.button(_name, key=f"conv_{idx}", use_container_width=True):
                            st.session_state.review_conversation = idx
                            if "preview_file" in st.session_state: del st.session_state.preview_file
                            st.rerun()
                    with _t2:
                        with st.popover("⋮", use_container_width=True, key=f"menu_{idx}"):
                            if st.button("Delete", icon=":material/delete:", key=f"del_{idx}", use_container_width=True):
                                confirm_delete_conv(user_email, idx)
                            if st.button("Rename", icon=":material/edit:", key=f"ren_{idx}", use_container_width=True):
                                st.session_state["renaming_conv"] = idx
                                st.rerun()
                    if st.session_state.get("renaming_conv") == idx:
                        _new = st.text_input("New name", value=_name, key=f"rn_{idx}", label_visibility="collapsed")
                        _rc1, _rc2 = st.columns(2)
                        if _rc1.button("Save", key=f"rsv_{idx}"):
                            _convs = load_user_convs(user_email)
                            if idx < len(_convs): _convs[idx]["title"] = _new; save_user_convs(user_email, _convs)
                            del st.session_state["renaming_conv"]; st.rerun()
                        if _rc2.button("Cancel", key=f"rnc_{idx}"):
                            del st.session_state["renaming_conv"]; st.rerun()
        else:
            st.caption("No conversations yet")

        with st.container(key="side_account"):
            _initial = (user_name[:1] or "?").upper()
            if not _initial.isalnum():
                _initial = "?"
            st.markdown(f"<style>.st-key-side_account{{--initial:'{_initial}'}}</style>", unsafe_allow_html=True)
            with st.popover(user_name, use_container_width=True):
                st.caption(user_email)
                if st.button("Settings", icon=":material/settings:", key="acct_settings", use_container_width=True):
                    settings_dialog(user_email, user_name)
                if st.button("Language", icon=":material/language:", key="acct_language", use_container_width=True):
                    language_dialog(user_email)
                if st.button("Get help", icon=":material/help:", key="acct_help", use_container_width=True):
                    help_dialog()
                st.divider()
                if st.button("Log out", icon=":material/logout:", key="acct_logout", use_container_width=True):
                    confirm_logout(user_email)


