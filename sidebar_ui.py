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
from functools import wraps
import streamlit as st
import i18n
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


def _clear_with_confirm(label, key, action, message):
    """Click -> "This cannot be undone" with a red Yes -> do it, close the dialog, show `message`."""
    st.button(label, key=key, on_click=lambda k=key: st.session_state.update(_clr_pending=k))
    if st.session_state.get("_clr_pending") == key:
        st.caption(i18n.t("set.cannot_undo"))
        c1, c2 = st.columns(2)
        if c1.button(i18n.t("set.yes"), key=f"yes_{key}"):
            action()
            st.session_state.pop("_clr_pending", None)
            st.session_state["_flash"] = message
            st.rerun()
        c2.button(i18n.t("nav.cancel"), key=f"no_{key}",
                  on_click=lambda: st.session_state.pop("_clr_pending", None))


def _t_dialog(title_key, **dkw):
    """@st.dialog, but the title is translated when the dialog OPENS. A plain
    @st.dialog(i18n.t(...)) in an imported module would freeze the title in the
    language of the very first run, so the decorator is re-applied on every call."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            st.dialog(i18n.t(title_key), **dkw)(fn)(*args, **kwargs)
        return wrapper
    return deco


@_t_dialog("set.title")
def settings_dialog(email, name):
    s = load_user_settings(email)
    st.markdown(f"**{i18n.t('set.account')}**")
    st.caption(f"{name} · {email}")
    st.markdown(f"**{i18n.t('set.display')}**")
    dev = st.toggle(i18n.t("set.dev_toggle"), value=s["dev_info"],
                    help=i18n.t("set.dev_help"))
    if dev != s["dev_info"]:
        s["dev_info"] = dev
        save_user_settings(email, s)
        st.session_state["dev_info"] = dev
    st.markdown(f"**{i18n.t('set.data')}**")
    st.caption(i18n.t("set.memory_note"))
    _clear_with_confirm(i18n.t("set.clear_memory"), "clr_mem", lambda: memory.clear_memory(email), i18n.t("set.memory_cleared"))
    _clear_with_confirm(i18n.t("set.clear_chats"), "clr_chats", lambda: _save_convs(email, []), i18n.t("set.chats_cleared"))
    _clear_with_confirm(i18n.t("set.clear_searches"), "clr_search", lambda: clear_searches(email), i18n.t("set.searches_cleared"))


@_t_dialog("lang.title")
def language_dialog(email):
    s = load_user_settings(email)
    st.caption(i18n.t("lang.note"))
    keys = list(LANGUAGES)
    labels = {"auto": i18n.t("lang.auto"), "English": i18n.t("lang.en"), "Simplified Chinese": i18n.t("lang.zh")}
    choice = st.radio("Answer language", keys, index=keys.index(s["language"]),
                      format_func=lambda k: labels[k], label_visibility="collapsed")
    if st.button(i18n.t("nav.save"), key="lang_save", type="primary"):
        s["language"] = choice
        save_user_settings(email, s)
        st.rerun()


@_t_dialog("help.title", width="large")
def help_dialog():
    st.markdown(i18n.t("help.body"))


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


@_t_dialog("search.title", width="large")
def search_dialog(email):
    convs = _load_convs(email)
    history = load_search_history(email)
    q = st.text_input("Search", key="dlg_query", placeholder=i18n.t("search.placeholder"),
                      label_visibility="collapsed", on_change=_on_search_submit, args=(email,))
    ql = q.strip()

    if ql:
        results = find_chats(convs, ql)
        st.markdown(f'<div class="search-label">{i18n.t("search.results")}</div>', unsafe_allow_html=True)
        with st.container(key="search_results"):
            for idx, snip in results:
                if st.button(convs[idx].get("title") or convs[idx].get("filename", "?"), key=f"srch_{idx}", use_container_width=True):
                    _open_chat(idx)
                if snip: st.caption(snip)
            if not results:
                st.caption(i18n.t("search.none"))
    else:
        if history:
            _h1, _h2 = st.columns([4, 1])
            _h1.markdown(f'<div class="search-label">{i18n.t("search.recent_searches")}</div>', unsafe_allow_html=True)
            _h2.button(i18n.t("search.clear_all"), key="srch_clear", on_click=clear_searches, args=(email,))
            with st.container(key="search_recent"):
                for j, h in enumerate(history[:8]):
                    _r1, _r2 = st.columns([12, 1], vertical_alignment="center")
                    _r1.button(h, icon=":material/history:", key=f"srch_h_{j}", use_container_width=True,
                               on_click=_use_search, args=(email, h))
                    _r2.button(i18n.t("search.remove"), icon=":material/close:", key=f"srch_x_{j}", help=i18n.t("search.remove"),
                               on_click=remove_search, args=(email, h))
        if convs:
            st.markdown(f'<div class="search-label">{i18n.t("search.recent_chats")}</div>', unsafe_allow_html=True)
            with st.container(key="search_results"):
                for idx in range(len(convs) - 1, max(len(convs) - 9, -1), -1):
                    if st.button(convs[idx].get("title") or convs[idx].get("filename", "?"), key=f"srch_r_{idx}", use_container_width=True):
                        _open_chat(idx)
        else:
            st.caption(i18n.t("nav.no_chats"))


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

    _flash = st.session_state.pop("_flash", None)
    if _flash:
        st.toast(_flash)

    _settings = load_user_settings(user_email)
    st.session_state["dev_info"] = _settings["dev_info"]
    agentic_rag.set_response_language(_settings["language"])

    with st.sidebar:
        with st.container(key="nav_search"):
            if st.button("Search", icon=":material/search:", key="btn_search", help=i18n.t("nav.search_help")):
                search_dialog(user_email)

        with st.container(key="brand_wrap"):
            st.markdown('<div class="brand">NexusAsk</div>', unsafe_allow_html=True)

        if st.button(i18n.t("nav.new_chat"), icon=":material/add:", key="nav_new", use_container_width=True):
            with st.spinner(i18n.t("nav.saving")):
                archive_current()
            for key in ["messages", "doc_chunks", "uploaded_filenames", "processed_files", "file_raw_bytes", "applied_chunk_size", "applied_chunk_overlap", "preview_file", "review_conversation", "show_uploader", "library_view", "preview_from_library"]:
                if key in st.session_state: del st.session_state[key]
            if "continuing_conversation_idx" in st.session_state: del st.session_state["continuing_conversation_idx"]
            st.rerun()

        if st.button(i18n.t("nav.library"), icon=":material/folder_open:", key="nav_library", use_container_width=True):
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

        st.markdown(f'<div class="side-label">{i18n.t("nav.chats")}</div>', unsafe_allow_html=True)
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
                            if st.button(i18n.t("nav.delete"), icon=":material/delete:", key=f"del_{idx}", use_container_width=True):
                                confirm_delete_conv(user_email, idx)
                            if st.button(i18n.t("nav.rename"), icon=":material/edit:", key=f"ren_{idx}", use_container_width=True):
                                st.session_state["renaming_conv"] = idx
                                st.rerun()
                    if st.session_state.get("renaming_conv") == idx:
                        _new = st.text_input("New name", value=_name, key=f"rn_{idx}", label_visibility="collapsed")
                        _rc1, _rc2 = st.columns(2)
                        if _rc1.button(i18n.t("nav.save"), key=f"rsv_{idx}"):
                            _convs = load_user_convs(user_email)
                            if idx < len(_convs): _convs[idx]["title"] = _new; save_user_convs(user_email, _convs)
                            del st.session_state["renaming_conv"]; st.rerun()
                        if _rc2.button(i18n.t("nav.cancel"), key=f"rnc_{idx}"):
                            del st.session_state["renaming_conv"]; st.rerun()
        else:
            st.caption(i18n.t("nav.no_chats"))

        with st.container(key="side_account"):
            _initial = (user_name[:1] or "?").upper()
            if not _initial.isalnum():
                _initial = "?"
            st.markdown(f"<style>.st-key-side_account{{--initial:'{_initial}'}}</style>", unsafe_allow_html=True)
            with st.popover(user_name, use_container_width=True):
                st.caption(user_email)
                if st.button(i18n.t("acct.settings"), icon=":material/settings:", key="acct_settings", use_container_width=True):
                    st.session_state.pop("_clr_pending", None)
                    settings_dialog(user_email, user_name)
                if st.button(i18n.t("acct.language"), icon=":material/language:", key="acct_language", use_container_width=True):
                    language_dialog(user_email)
                if st.button(i18n.t("acct.help"), icon=":material/help:", key="acct_help", use_container_width=True):
                    help_dialog()
                st.divider()
                if st.button(i18n.t("acct.logout"), icon=":material/logout:", key="acct_logout", use_container_width=True):
                    confirm_logout(user_email)


