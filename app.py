import os
import re
import json
from dotenv import load_dotenv
# Load .env FIRST, before any langchain/langsmith imports, so tracing env vars exist at import time
load_dotenv()
# Explicitly surface tracing/endpoint config into os.environ (langchain reads these at import time)
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "true"))
if os.getenv("LANGSMITH_API_KEY"): os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_PROJECT"): os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
if os.getenv("OPENAI_API_KEY"): os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if os.getenv("OPENAI_BASE_URL"): os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

import hashlib, json, re, base64
from datetime import datetime
from typing import Optional, List, Mapping, Any
import streamlit as st
import streamlit.components.v1 as components
import time
import threading
import tempfile, httpx, bcrypt, pandas as pd, numpy as np
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic.v1 import PrivateAttr
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
import rate_limit_fallback as rlf
import timeout_handling as th
import async_pipelines as ap
import streaming_tokens_retry as str_mod
import agentic_rag
import memory
import retrieval
import retry_helper
import step_ui
import answer_actions
import autoscroll
import i18n
from i18n import t
from langsmith import Client as LangSmithClient

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 4
SEMANTIC_CACHE_THRESHOLD = 0.9
REAL_MODEL_TIMEOUT = 15.0  # seconds before a real (non-simulated) model call is abandoned -> TimeoutError -> Retry button
LLM_TEMPERATURE = 0.0
MAX_CONTEXT_CHARS = 12000
MAX_CONTEXT_TOKENS = 3000
MAX_STORED_CHUNK_CHARS = 500
BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")
CONVERSATIONS_DIR = os.path.join(BASE_DIR, "conversations")
LIBRARY_DIR = os.path.join(BASE_DIR, "library")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

class HttpxChatLLM(BaseLLM):
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    _client: Any = PrivateAttr()
    def __init__(self, api_key, base_url, model="gpt-4o-mini", temperature=0.0, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model, temperature=temperature, **kwargs)
        self._client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=httpx.Timeout(120.0, connect=30.0))
    @property
    def _llm_type(self): return "httpx-openai-compatible"
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        generations = []; usage = {}
        for prompt in prompts:
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if stop: payload["stop"] = stop
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            generations.append([Generation(text=content, generation_info={"finish_reason": data["choices"][0].get("finish_reason"), "usage": usage})])
        llm_output = {"token_usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)}, "model_name": self.model}
        return LLMResult(generations=generations, llm_output=llm_output)
    def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature, "stream": True, "stream_options": {"include_usage": True}}
        if stop: payload["stop"] = stop
        with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line: continue
                if isinstance(line, bytes): line = line.decode("utf-8", errors="ignore")
                if not line.startswith("data: "): continue
                data = line[6:]
                if data.strip() == "[DONE]": break
                try: obj = json.loads(data)
                except Exception: continue
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    yield GenerationChunk(text=delta)
    @property
    def _identifying_params(self): return {"model": self.model, "base_url": self.base_url}

llm_model = HttpxChatLLM(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

def show_langsmith_trace():
    """Fetch and display the latest LangSmith trace after an LLM call."""
    try:
        ls_key = os.getenv("LANGSMITH_API_KEY", "")
        if not ls_key: return
        client = LangSmithClient(api_key=ls_key)
        runs = list(client.list_runs(project_name=os.getenv("LANGSMITH_PROJECT", "nexusask"), is_root=True, limit=1))
        if runs:
            run = runs[0]
            with st.expander("LangSmith Trace"):
                url = run.url if hasattr(run, "url") else f"https://smith.langchain.com/runs/{run.id}"
                st.markdown(f"**Trace URL:** [{url}]({url})")
                st.markdown(f"**Run ID:** `{run.id}`")
                st.markdown(f"**Model:** {LLM_MODEL}")
                if run.start_time and run.end_time:
                    latency = (run.end_time - run.start_time).total_seconds()
                    st.markdown(f"**Latency:** {latency:.2f}s")
                if hasattr(run, "inputs") and run.inputs:
                    st.markdown("**Input:**")
                    st.text(str(run.inputs)[:500])
                if hasattr(run, "outputs") and run.outputs:
                    st.markdown("**Output:**")
                    st.text(str(run.outputs)[:500])
    except Exception as e:
        st.caption(f"Trace fetch failed: {e}")

st.set_page_config(page_title="NexusAsk", page_icon=None, layout="wide")
from pathlib import Path
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)
if st.session_state.get("user_email"):
    i18n.init(st.session_state["user_email"])           # Settings > Language ("auto" follows the browser)
else:
    # login screen: the saved setting of the email being typed in ("auto" follows the browser)
    i18n.init(st.session_state.get("login_email") or st.session_state.get("reg_email"))
agentic_rag.set_ui_language(i18n.lang())                # the agent's own texts (refusal, disclaimer) follow the UI language
st.markdown(i18n.css_vars(), unsafe_allow_html=True)    # translated texts used by style.css (the disclaimer)
_UNUSED_CSS = """
<style>
    .stApp { background: linear-gradient(135deg, #fce4ec 0%, #f3e5f5 30%, #e1bee7 60%, #ce93d8 100%); }
    .main { background: transparent; }
    /* transparent top header + bottom container so the gradient shows through (chat boxes stay white) */
    header[data-testid="stHeader"] { background: transparent !important; }
    header[data-testid="stHeader"]::before { background: transparent !important; box-shadow: none !important; }
    div[data-testid="stAppViewContainer"] { background: transparent !important; }
    section[data-testid="stMain"] { background: transparent !important; }
    div[data-testid="stBottom"] { background: transparent !important; }
    .stChatMessage { background: rgba(255,255,255,0.85); border-radius: 16px; padding: 12px; margin: 6px 0; box-shadow: 0 2px 12px rgba(156,39,176,0.1); border-left: 3px solid #e91e9c; }
    .upload-box { background: rgba(255,255,255,0.9); border-radius: 20px; padding: 28px; box-shadow: 0 4px 20px rgba(156,39,176,0.15); border: 1px solid rgba(156,39,176,0.1); }
    h1 { background: linear-gradient(90deg, #e91e63, #9c27b0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton > button { background: linear-gradient(90deg, #f06292, #ba68c8); color: white; border: none; border-radius: 20px; padding: 6px 18px; }
    .stButton > button:hover { background: linear-gradient(90deg, #ec407a, #ab47bc); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(156,39,176,0.3); }
    .stSidebar { background: linear-gradient(180deg, #fce4ec 0%, #f3e5f5 100%); }
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 { color: #880e4f; }
    .stSidebar .stButton > button { background: linear-gradient(90deg, #f8bbd0, #ce93d8); color: #4a148c; width: 100%; }
    .stTextInput > div > input { border-radius: 12px; border: 2px solid #e1bee7; }
    .stFileUploader { background: rgba(255,255,255,0.7); border-radius: 12px; border: 2px dashed #ce93d8; }
    .stTextArea > div > textarea { border-radius: 12px; border: 2px solid #e1bee7; }
    .stCaption { color: #6a1b9a; }
    .stExpander { background: rgba(255,255,255,0.7); border-radius: 12px; border: 1px solid rgba(156,39,176,0.1); }
    /* session rows: transparent frame by default, shown on hover; ⋮ hidden by default, shown on hover */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) { border: 1px solid transparent; border-radius: 8px; transition: border-color 0.15s ease, background 0.15s ease; padding: 1px 4px; margin: 0; }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]):hover { border-color: #9c27b0; background: rgba(156,39,176,0.05); }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) .stButton > button { background: transparent !important; color: #4a148c !important; border: none !important; text-align: left !important; box-shadow: none !important; }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) .stButton > button:hover { background: transparent !important; }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stPopover"] { opacity: 0; transition: opacity 0.15s ease; }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]):hover [data-testid="stPopover"] { opacity: 1; }
    /* compact vertical spacing between session cards */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 2px !important; }
    /* ⋮ trigger: transparent, no border, no caret — just the dots */
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button { background: transparent !important; border: none !important; box-shadow: none !important; padding: 2px 4px !important; min-width: 0 !important; }
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button svg { display: none !important; }
    /* popover body: small frame around Delete/Rename */
    section[data-testid="stSidebar"] [data-testid="stPopoverBody"] { min-width: 0 !important; width: 150px !important; padding: 4px !important; gap: 2px !important; }
    /* Delete/Rename options: plain (no purple), frame on hover */
    [data-testid="stPopoverBody"] .stButton > button { background: transparent !important; color: #4a148c !important; border: 1px solid transparent !important; border-radius: 6px !important; text-align: left !important; width: 100% !important; box-shadow: none !important; }
    [data-testid="stPopoverBody"] .stButton > button:hover { border-color: #9c27b0 !important; background: rgba(156,39,176,0.06) !important; }
    /* chat messages: hide avatars */
    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
    /* user message: right-aligned, outer transparent */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { display: flex !important; justify-content: flex-end !important; background: transparent !important; border: none !important; box-shadow: none !important; padding: 4px 0 !important; }
    /* user text bubble: auto-width white */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] { width: fit-content !important; max-width: 80% !important; background: rgba(255,255,255,0.92) !important; border-radius: 16px !important; padding: 8px 14px !important; margin-left: auto !important; }
    /* copy container (inside toolbar): no bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"] { background: transparent !important; width: auto !important; max-width: none !important; padding: 0 !important; border-radius: 0 !important; }
    /* assistant message: transparent (override global white) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { background: transparent !important; border: none !important; box-shadow: none !important; }
    /* toolbar (user messages): always visible, small */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stHorizontalBlock"] { opacity: 1 !important; max-height: 34px !important; }
    /* toolbar buttons: transparent, small */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stHorizontalBlock"] .stButton > button { background: transparent !important; color: #4a148c !important; border: none !important; font-size: 0.8rem !important; padding: 2px 8px !important; box-shadow: none !important; }
    /* compact toolbar: shrink to buttons, close together, right-aligned */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stHorizontalBlock"] { width: fit-content !important; max-width: 240px !important; gap: 2px !important; margin-left: auto !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stHorizontalBlock"] > div { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stHorizontalBlock"] > div + div::before { content: "·"; color: #9c27b0; margin-right: 2px; font-size: 0.8rem; }
    /* sidebar ⋮ trigger: aggressively transparent (no white bg) */
    section[data-testid="stSidebar"] [data-testid="stPopover"],
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button,
    section[data-testid="stSidebar"] [data-testid="stPopoverButton"] { background: transparent !important; background-color: transparent !important; border: none !important; box-shadow: none !important; outline: none !important; }
</style>
"""

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}
def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f: json.dump(users, f, ensure_ascii=False, indent=2)
def is_valid_email(email): return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def show_auth_screen():
    st.title("NexusAsk"); st.markdown("---")
    tab1, tab2 = st.tabs([i18n.t("auth.login_tab"), i18n.t("auth.register_tab")])
    with tab1:
        st.subheader(i18n.t("auth.login_title"))
        le = st.text_input(i18n.t("auth.email"), key="login_email")
        lp = st.text_input(i18n.t("auth.password"), key="login_password", type="password")
        if st.button(i18n.t("auth.login_title"), key="login_btn"):
            if not le or not lp: st.error(i18n.t("auth.enter_both"))
            else:
                users = load_users(); user = users.get(le)
                if user and bcrypt.checkpw(lp.encode(), user["password"].encode()):
                    st.session_state.user_email = le; st.session_state.user_name = user.get("name", le); st.rerun()
                else: st.error(i18n.t("auth.invalid_login"))
    with tab2:
        st.subheader(i18n.t("auth.create_title"))
        rn = st.text_input(i18n.t("auth.name"), key="reg_name")
        re_ = st.text_input(i18n.t("auth.email"), key="reg_email")
        rp = st.text_input(i18n.t("auth.password"), key="reg_password", type="password")
        rp2 = st.text_input(i18n.t("auth.confirm"), key="reg_password2", type="password")
        if st.button(i18n.t("auth.register_tab"), key="register_btn"):
            if not rn or not re_ or not rp: st.error(i18n.t("auth.all_required"))
            elif not is_valid_email(re_): st.error(i18n.t("auth.invalid_email"))
            elif rp != rp2: st.error(i18n.t("auth.mismatch"))
            elif len(rp) < 4: st.error(i18n.t("auth.too_short"))
            else:
                users = load_users()
                if re_ in users: st.error(i18n.t("auth.exists"))
                else:
                    users[re_] = {"name": rn, "password": bcrypt.hashpw(rp.encode(), bcrypt.gensalt()).decode()}
                    save_users(users); st.success(i18n.t("auth.created"))
    st.stop()

def get_conv_file(email): return os.path.join(CONVERSATIONS_DIR, f"{re.sub(r'[^a-zA-Z0-9._-]', '_', email)}.json")
def load_user_convs(email):
    fp = get_conv_file(email)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f: return json.load(f)
        except: return []
    return []
def save_user_convs(email, convs):
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    with open(get_conv_file(email), "w", encoding="utf-8") as f: json.dump(convs, f, ensure_ascii=False, indent=2)


# ----- Persistent per-user file Library (raw bytes on disk) -----
def get_library_dir(email):
    d = os.path.join(LIBRARY_DIR, re.sub(r'[^a-zA-Z0-9._-]', '_', email or "_anon"))
    os.makedirs(d, exist_ok=True)
    return d
def list_library_files(email):
    if not email: return []
    d = get_library_dir(email)
    return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
def save_library_file(email, fname, raw):
    if not email or not fname: return
    with open(os.path.join(get_library_dir(email), os.path.basename(fname)), "wb") as f: f.write(raw)
def read_library_file(email, fname):
    if not email or not fname: return None
    p = os.path.join(get_library_dir(email), os.path.basename(fname))
    return open(p, "rb").read() if os.path.exists(p) else None
def delete_library_file(email, fname):
    if not email or not fname: return
    p = os.path.join(get_library_dir(email), os.path.basename(fname))
    if os.path.exists(p): os.remove(p)

def generate_title(question):
    """LLM: generate a short topic title from the first question (does NOT answer it)."""
    try:
        msg = agentic_rag.llm_chat([
            {"role": "system", "content": """You generate a short topic title (3-8 words, same language as the question) describing what the user's question is about. Do NOT answer the question. Output ONLY the title, no quotes, no trailing punctuation.

Examples:
Q: 法国的首都是哪里？
Title: 法国首都
Q: 计算 sqrt(144)+3*5 等于多少？
Title: 数学计算
Q: 2026年世界杯在哪个国家举办？
Title: 2026世界杯举办国"""},
            {"role": "user", "content": question},
        ])
        t = (msg.get("content") or "").strip().strip('"').strip()
        return t[:40] or "New Conversation"
    except Exception:
        return "New Conversation"


_title_in_progress = set()  # conversation ids whose title is being upgraded in the background


def upgrade_title_async(email, conv_id):
    """Background: generate the real title for a conversation still using a temp title, then save."""
    try:
        convs = load_user_convs(email)
        c = next((x for x in convs if x.get("id") == conv_id), None)
        if not c: return
        _first_q = next((m["content"] for m in c.get("messages", []) if m.get("role") == "user"), "")
        if not _first_q: return
        _title = generate_title(_first_q)
        convs2 = load_user_convs(email)  # re-load in case it changed
        c2 = next((x for x in convs2 if x.get("id") == conv_id), None)
        if c2:
            c2["title"] = _title
            c2["title_is_final"] = True
            save_user_convs(email, convs2)
    except Exception:
        pass
    finally:
        _title_in_progress.discard(conv_id)


def archive_current():
    if "messages" not in st.session_state or not st.session_state.messages: return
    email = st.session_state.get("user_email", "")
    if not email: return
    fns = st.session_state.get("uploaded_filenames", [])
    fn_str = ", ".join(fns) if isinstance(fns, list) else str(fns)
    chunks = st.session_state.get("doc_chunks", [])
    convs = load_user_convs(email)
    cont_idx = st.session_state.get("continuing_conversation_idx")
    if cont_idx is not None and cont_idx < len(convs):
        convs[cont_idx]["messages"] = st.session_state.messages.copy()
        convs[cont_idx]["doc_chunks"] = [c[:MAX_STORED_CHUNK_CHARS] for c in chunks]
        convs[cont_idx]["filename"] = fn_str
        convs[cont_idx]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        _first_q = next((m["content"] for m in st.session_state.messages if m.get("role") == "user"), "")
        _title = (_first_q[:24] + ("…" if len(_first_q) > 24 else "")) if _first_q else (fn_str or "New Conversation")
        convs.append({"id": hashlib.md5(f"{fn_str}{datetime.now().isoformat()}".encode()).hexdigest()[:8], "filename": fn_str, "title": _title, "title_is_final": False, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "messages": st.session_state.messages.copy(), "doc_chunks": [c[:MAX_STORED_CHUNK_CHARS] for c in chunks]})
    if len(convs) > 100: convs = convs[-100:]
    save_user_convs(email, convs)
    threading.Thread(target=memory.extract_memory, args=(email, st.session_state.get("messages", [])), daemon=True).start()

def checkpoint_conversation(email, convs, idx):
    """Checkpoint (save) the current conversation's messages + doc_chunks so it can be resumed later."""
    if idx is None or idx >= len(convs): return
    convs[idx]["messages"] = st.session_state.get("messages", []).copy()
    convs[idx]["doc_chunks"] = [c[:MAX_STORED_CHUNK_CHARS] for c in st.session_state.get("doc_chunks", [])]
    convs[idx]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_user_convs(email, convs)


def process_uploads(files):
    """Process new uploaded files into doc_chunks (no UI). Returns True if new files were processed."""
    if not files:
        return False
    cur_names = [f.name for f in files]
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}; st.session_state.doc_chunks = []
        st.session_state.uploaded_filenames = []; st.session_state.file_raw_bytes = {}
        if "messages" not in st.session_state: st.session_state.messages = []
    new_files = []
    for uf in files:
        raw = uf.getvalue(); fh = hashlib.sha256(raw).hexdigest()
        if st.session_state.processed_files.get(uf.name) != fh: new_files.append(uf)
    old_n = set(st.session_state.processed_files.keys()); cur_n = set(cur_names)
    if old_n - cur_n:
        st.session_state.doc_chunks = []; st.session_state.processed_files = {}
        st.session_state.uploaded_filenames = []; st.session_state.file_raw_bytes = {}
        new_files = list(files)
    if not new_files:
        return False
    for uf in new_files:
        raw = uf.getvalue(); st.session_state.file_raw_bytes[uf.name] = raw
        save_library_file(st.session_state.get("user_email", ""), uf.name, raw)
        docs = load_file(uf)
        if not docs: continue
        sp = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = sp.split_documents(docs)
        st.session_state.doc_chunks.extend([d.page_content for d in chunks])
        st.session_state.processed_files[uf.name] = hashlib.sha256(raw).hexdigest()
        if uf.name not in st.session_state.uploaded_filenames: st.session_state.uploaded_filenames.append(uf.name)
    st.session_state.applied_chunk_size = chunk_size
    st.session_state.applied_chunk_overlap = chunk_overlap
    return True


def _on_initial_submit():
    """on_change for the initial centered text_input: process any pending upload, append the user msg,
    set pending_q + clear, so the layout flips to the bottom chat_input on Enter."""
    _v = st.session_state.get("initial_q", "")
    if _v:
        process_uploads(st.session_state.get("uploader"))  # process the ＋ popover's files before the flip
        st.session_state.setdefault("messages", []).append({"role": "user", "content": _v})
        st.session_state["pending_q"] = _v
        st.session_state["initial_q"] = ""


def upload_popover():
    """＋ popover with a drag & drop file uploader; returns the uploaded files (or None)."""
    with st.popover("+", use_container_width=False, help=i18n.t("upload.help")):
        st.caption(i18n.t("upload.caption"))
        return st.file_uploader("Files", type=["pdf", "docx", "pptx", "xlsx", "xls", "txt", "md", "csv"], label_visibility="collapsed", accept_multiple_files=True, key="uploader")


@st.dialog(i18n.t("dlg.clear_all_title"))
def confirm_clear_all(email):
    st.write(i18n.t("dlg.clear_all_body"))
    c1, c2 = st.columns(2)
    if c1.button(i18n.t("dlg.delete_all"), type="primary"):
        save_user_convs(email, []); st.rerun()
    if c2.button(i18n.t("nav.cancel")):
        st.rerun()


@st.dialog(i18n.t("lib.delete_title"))
def confirm_delete_library_file(email, fname):
    st.write(i18n.t("lib.delete_confirm", name=fname))
    c1, c2 = st.columns(2)
    if c1.button(i18n.t("nav.delete"), type="primary"):
        delete_library_file(email, fname)
        if fname in st.session_state.get("file_raw_bytes", {}): del st.session_state.file_raw_bytes[fname]
        if st.session_state.get("preview_file") == fname: del st.session_state.preview_file
        if fname in st.session_state.get("uploaded_filenames", []): st.session_state.uploaded_filenames.remove(fname)
        if fname in st.session_state.get("processed_files", {}): del st.session_state.processed_files[fname]
        st.rerun()
    if c2.button(i18n.t("nav.cancel")):
        st.rerun()


@st.dialog(i18n.t("dlg.logout_title"))
def confirm_logout(email):
    st.write(i18n.t("dlg.logout_body"))
    c1, c2 = st.columns(2)
    if c1.button(i18n.t("acct.logout"), type="primary"):
        archive_current()
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
    if c2.button(i18n.t("nav.cancel")):
        st.rerun()


@st.dialog(i18n.t("dlg.delete_conv_title"))
def confirm_delete_conv(email, idx):
    st.write(i18n.t("dlg.delete_conv_body"))
    c1, c2 = st.columns(2)
    if c1.button(i18n.t("nav.delete"), type="primary"):
        convs = load_user_convs(email)
        if idx < len(convs): convs.pop(idx); save_user_convs(email, convs)
        st.rerun()
    if c2.button(i18n.t("nav.cancel")):
        st.rerun()


@st.dialog(i18n.t("dlg.clear_memory_title"))
def confirm_clear_memory(email):
    st.write(i18n.t("dlg.clear_memory_body"))
    c1, c2 = st.columns(2)
    if c1.button(i18n.t("dlg.clear_memory_btn"), type="primary"):
        memory.clear_memory(email); st.rerun()
    if c2.button(i18n.t("nav.cancel")):
        st.rerun()


def load_pdf(path): return PyPDFLoader(path).load()
def load_docx(path): return Docx2txtLoader(path).load()
def load_pptx(path):
    from pptx import Presentation
    prs = Presentation(path); docs = []
    for sn, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip(): texts.append(para.text.strip())
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        if cell.text.strip(): texts.append(cell.text.strip())
        if texts: docs.append(Document(page_content="\n".join(texts), metadata={"page": sn, "source": path}))
    return docs
def load_xlsx(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True); docs = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]; rows_text = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None]
            if cells: rows_text.append(" | ".join(cells))
        if rows_text: docs.append(Document(page_content="\n".join(rows_text), metadata={"sheet": sheet, "source": path}))
    return docs
def load_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f: return [Document(page_content=f.read(), metadata={"source": path})]
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f: return [Document(page_content=f.read(), metadata={"source": path})]

def load_file(uf):
    suffix = "." + uf.name.split(".")[-1].lower(); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f: f.write(uf.getvalue()); path = f.name
        if suffix == ".pdf": docs = load_pdf(path)
        elif suffix == ".docx": docs = load_docx(path)
        elif suffix == ".pptx": docs = load_pptx(path)
        elif suffix in [".xlsx", ".xls"]: docs = load_xlsx(path)
        elif suffix in [".txt", ".md", ".csv"]: docs = load_text(path)
        else: st.error(i18n.t("file.unsupported", suffix=suffix)); return []
        if not docs: st.warning(i18n.t("file.no_text", name=uf.name))
        return docs
    except Exception as e:
        st.error(i18n.t("file.read_failed", name=uf.name, error=e)); return []
    finally:
        if path and os.path.exists(path):
            try: os.unlink(path)
            except: pass

def search_chunks(query, chunks, k=4):
    """Document retrieval (BM25 + keyword hybrid, fused with RRF)."""
    return retrieval.hybrid_search(query, chunks, k)

# ---------- Semantic cache (threshold 0.92) + traced Retrieval/LLM pipeline ----------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        with st.spinner(i18n.t("chat.loading_model")):
            _embedder = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder

def build_prompt(q, ctx):
    instr = "Think step by step and provide a thorough, detailed answer." if st.session_state.get("answer_mode") == "Thinking" else "Answer concisely and directly."
    return f"You are a helpful assistant. {instr}\n\nDocument content:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"

def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / n) if n else 0.0

def _cache_fn(x, config: RunnableConfig):
    q = x["query"]
    cache = st.session_state.setdefault("semantic_cache", [])
    best_sim = 0.0; best_idx = -1; q_emb = None
    if cache:
        q_emb = get_embedder().embed_query(q)
        for i, e in enumerate(cache):
            s = _cosine(q_emb, e.get("embedding", []))
            if s > best_sim: best_sim, best_idx = s, i
        if best_idx >= 0 and best_sim >= SEMANTIC_CACHE_THRESHOLD:
            hit = cache[best_idx]
            st.session_state["cache_hits"] = st.session_state.get("cache_hits", 0) + 1
            return {"hit": True, "similarity": best_sim, "answer": hit["answer"], "contexts": hit.get("contexts", []), "q_emb": q_emb}
    return {"hit": False, "similarity": best_sim, "q_emb": q_emb}
_cache_chain = RunnableLambda(_cache_fn).with_config({"run_name": "SemanticCache"})

_retriever = RunnableLambda(lambda x: search_chunks(x["query"], x["chunks"], x["k"])).with_config({"run_name": "Retrieval"})

def _make_streamlit_reporter(status):
    """Map structured retry/fallback events onto a live st.status widget."""
    def report(ev):
        if status is None: return
        t = ev.get("type")
        try:
            if t == "attempt":
                status.update(label=f"Calling model (attempt {ev['n']}/{ev['total']})")
            elif t == "rate_limited":
                st.write(f"Attempt {ev['n']}: rate-limited (429). Retrying in {ev['wait']:.1f}s…")
            elif t == "retrying":
                st.write(f"Attempt {ev['n']}: {ev.get('reason', 'error')}. Retrying in {ev['wait']:.1f}s…")
            elif t == "exhausted":
                status.update(label="Fallback model thinking…", state="running")
            elif t == "fallback":
                st.write("Contacting fallback model…")
            elif t == "fallback_done":
                status.update(label="Fallback model responded", state="complete")
            elif t == "done":
                status.update(label=i18n.t("status.ready"), state="complete")
        except Exception:
            pass
    return report

def _make_timeout_reporter(status):
    """Map timeout events onto the same st.status widget (live waiting ticks)."""
    def report(ev):
        if status is None: return
        t = ev.get("type")
        try:
            if t == "calling":
                status.update(label=f"⏳ Calling model (timeout {ev['timeout']:.0f}s)…")
            elif t == "waiting":
                status.update(label=f"⏳ Waiting for model… {ev['elapsed']:.1f}s / {ev['timeout']:.0f}s")
            elif t == "done":
                status.update(label=f"Responded in {ev['elapsed']:.1f}s", state="complete")
            elif t == "timeout":
                status.update(label=f"⏱ Timed out after {ev['timeout']:.0f}s", state="error")
        except Exception:
            pass
    return report

def _answer_fn(x, config: RunnableConfig):
    q, chunks, k = x["query"], x["chunks"], x["k"]
    status = x.get("status")
    c = _cache_chain.invoke({"query": q}, config=config)
    if c["hit"]:
        _make_streamlit_reporter(status)({"type": "done"})
        return {"answer": c["answer"], "contexts": c["contexts"], "cache_hit": True, "similarity": c["similarity"]}
    retrieved = _retriever.invoke({"query": q, "chunks": chunks, "k": k}, config=config)
    ctx = "\n\n".join(retrieved[:MAX_CONTEXT_CHARS // max(DEFAULT_CHUNK_SIZE, 100) + 1])
    if len(ctx) > MAX_CONTEXT_CHARS: ctx = ctx[:MAX_CONTEXT_CHARS] + "...[truncated]"
    pr = build_prompt(q, ctx)
    def _real_call(prompt):
        # Real model call wrapped with a 15s timeout + live waiting ticks (runs in a daemon
        # thread so the wait can be abandoned). config carries the parent callbacks so the
        # HttpxChatLLM run still traces (with token usage) under answer_question.
        return th.call_with_timeout_sync(prompt, timeout_seconds=REAL_MODEL_TIMEOUT, model_fn=lambda p: llm_model.invoke(p, config=config), on_progress=_make_timeout_reporter(status))
    def traced_primary_call(prompt):
        _attempt["n"] += 1
        try:
            return _real_call(prompt)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise rlf.RateLimitError(f"429 from primary: {e}")
            raise
    def fallback_call(prompt):
        return rlf.call_fallback_model(prompt)
    a = rlf.answer_with_retry_and_fallback(pr, primary_fn=traced_primary_call, fallback_fn=fallback_call, on_progress=_make_streamlit_reporter(status)).strip()
    q_emb = c["q_emb"] or get_embedder().embed_query(q)
    cache = st.session_state.setdefault("semantic_cache", [])
    cache.append({"question": q, "answer": a, "embedding": q_emb, "contexts": retrieved})
    if len(cache) > 200: del cache[:len(cache) - 200]
    st.session_state["cache_misses"] = st.session_state.get("cache_misses", 0) + 1
    return {"answer": a, "contexts": retrieved, "cache_hit": False, "similarity": c["similarity"]}
_answer_chain = RunnableLambda(_answer_fn).with_config({"run_name": "answer_question"})

def answer_question(q, chunks, k, status=None):
    return _answer_chain.invoke({"query": q, "chunks": chunks, "k": k, "status": status})

def _make_agentic_reporter(status):
    """Map agentic step events onto a live one-line status (streaming: label-only; the
    answer itself streams in the main chat via st.write_stream)."""
    def report(ev):
        if status is None: return
        ev_t = ev.get("type")
        try:
            if ev_t == "blocked":
                status.update(label=i18n.t("status.blocked"), state="error")
            elif ev_t == "rewrite":
                status.update(label=i18n.t("status.rewriting"))
            elif ev_t == "thinking":
                status.update(label=i18n.t("status.thinking_step", step=ev['step']))
            elif ev_t == "tool_call":
                status.update(label=f"{i18n.tool_label(ev['name'])}…")
            elif ev_t == "tool_result":
                status.update(label=i18n.t("status.tool_done", tool=i18n.tool_label(ev['name']), n=ev['n']))
            elif ev_t == "rewrite_done":
                status.update(label=f"Rewritten: {str(ev.get('rewritten', ''))[:30]}…")
            elif ev_t == "retry":
                status.update(label=i18n.t("status.retrying", n=ev['n'], total=ev.get('total','?')))
        except Exception: pass
    return report

def agentic_answer_cached(q, chunks, k, status=None, email=None):
    """Agentic RAG with semantic cache. Returns a GENERATOR that yields answer tokens
    (real streaming for cache misses via agentic_answer_stream; chunked for cache hits).
    The semantic cache is used (lookup AND store) ONLY when there is no conversation history,
    because a cached answer to a follow-up like 'how tall is it?' would be wrong in another conversation."""
    blocked, refusal = agentic_rag.guard_input(q)
    if blocked:
        if status:
            try: status.update(label=i18n.t("status.blocked"), state="error")
            except Exception: pass
        yield refusal
        return
    history = agentic_rag.build_history(st.session_state.get("messages", []), q)
    cache = st.session_state.setdefault("semantic_cache", [])
    q_emb = get_embedder().embed_query(q) if (cache and not history) else None
    best_sim, best_idx = 0.0, -1
    if cache and not history:
        for i, e in enumerate(cache):
            s = _cosine(q_emb, e.get("embedding", []))
            if s > best_sim: best_sim, best_idx = s, i
        if best_idx >= 0 and best_sim >= SEMANTIC_CACHE_THRESHOLD:
            st.session_state["cache_hits"] = st.session_state.get("cache_hits", 0) + 1
            st.session_state["_last_stream_sim"] = best_sim
            st.session_state["_last_stream_cached"] = True
            if status:
                try: status.update(label=i18n.t("status.cached"), state="complete")
                except Exception: pass
            _cached_ans = cache[best_idx]["answer"]
            for i in range(0, len(_cached_ans), 3):
                yield _cached_ans[i:i+3]
                time.sleep(0.01)
            return
    _mem = memory.load_memory(email) if email else None
    _rep = _make_agentic_reporter(status)
    _acc = []
    for tok in agentic_rag.agentic_answer_stream(q, chunks, k, on_progress=_rep, memory=_mem, history=history):
        _acc.append(tok)
        yield tok
    if status:
        try: status.update(label=i18n.t("status.ready"), state="complete")
        except Exception: pass
    answer = "".join(_acc)
    if not history:
        if q_emb is None: q_emb = get_embedder().embed_query(q)
        cache.append({"question": q, "answer": answer, "embedding": q_emb, "contexts": []})
        if len(cache) > 200: del cache[:len(cache) - 200]
    st.session_state["cache_misses"] = st.session_state.get("cache_misses", 0) + 1
    st.session_state["_last_stream_sim"] = best_sim
    st.session_state["_last_stream_cached"] = False

def stream_answer(pr, status=None):
    """Real streaming from llm_model.stream() with light 429-retry on stream-start."""
    attempt = 0
    while True:
        attempt += 1
        try:
            for chunk in llm_model.stream(pr):
                txt = getattr(chunk, "text", "") or ""
                if txt:
                    yield txt
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < 10:
                if status:
                    try: status.update(label=f"⏳ Rate-limited, retrying ({attempt}/10)…")
                    except Exception: pass
                time.sleep(min(2 ** attempt, 30))
                continue
            raise

def stream_answer_question(q, chunks, k, status=None):
    """Streaming answer with semantic cache + token-truncated retrieval + content fallback."""
    cache = st.session_state.setdefault("semantic_cache", [])
    st.session_state["_last_stream_sim"] = 0.0
    st.session_state["_last_stream_cached"] = False
    q_emb = None
    if cache:
        q_emb = get_embedder().embed_query(q)
        best_sim, best_idx = 0.0, -1
        for i, e in enumerate(cache):
            s = _cosine(q_emb, e.get("embedding", []))
            if s > best_sim: best_sim, best_idx = s, i
        if best_idx >= 0 and best_sim >= SEMANTIC_CACHE_THRESHOLD:
            st.session_state["cache_hits"] = st.session_state.get("cache_hits", 0) + 1
            st.session_state["_last_stream_sim"] = best_sim
            st.session_state["_last_stream_cached"] = True
            if status:
                try: status.update(label=i18n.t("status.cached"), state="complete")
                except Exception: pass
            yield cache[best_idx]["answer"]
            return
    if q_emb is not None:
        st.session_state["_last_stream_sim"] = best_sim
    retrieved = search_chunks(q, chunks, k)
    ctx_chunks = str_mod.truncate_context_to_fit(retrieved, MAX_CONTEXT_TOKENS)
    pr = build_prompt(q, "\n\n".join(ctx_chunks))
    if status:
        try: status.update(label=f"Thinking…")
        except Exception: pass
    full = []
    for delta in stream_answer(pr, status):
        full.append(delta); yield delta
    a = "".join(full)
    if not str_mod.is_valid_response(a):
        a = str_mod.call_with_content_retry(lambda _p: llm_model.invoke(pr).strip(), pr, max_attempts=3)
        yield a
    if q_emb is None: q_emb = get_embedder().embed_query(q)
    cache.append({"question": q, "answer": a, "embedding": q_emb, "contexts": retrieved})
    if len(cache) > 200: del cache[:len(cache) - 200]
    st.session_state["cache_misses"] = st.session_state.get("cache_misses", 0) + 1
    if status:
        try: status.update(label=i18n.t("status.ready"), state="complete")
        except Exception: pass

def preview_pdf(raw, fname):
    b64 = base64.b64encode(raw).decode()
    st.markdown(f"### {fname}")
    st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800" type="application/pdf"></iframe>', unsafe_allow_html=True)

def preview_docx(raw, fname):
    st.markdown(f"### {fname}"); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f: f.write(raw); path = f.name
        from docx import Document as D
        doc = D(path)
        for p in doc.paragraphs:
            if p.text.strip():
                s = p.style.name if p.style else "Normal"
                if "Heading 1" in s or "Title" in s: st.markdown(f"# {p.text}")
                elif "Heading 2" in s: st.markdown(f"## {p.text}")
                elif "Heading 3" in s: st.markdown(f"### {p.text}")
                elif "List" in s: st.markdown(f"- {p.text}")
                else: st.write(p.text)
        for t in doc.tables:
            rows = [[c.text for c in r.cells] for r in t.rows]
            if rows: st.table(pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows))
    finally:
        if path and os.path.exists(path): os.unlink(path)

def preview_pptx(raw, fname):
    st.markdown(f"### {fname}"); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as f: f.write(raw); path = f.name
        from pptx import Presentation
        prs = Presentation(path)
        for sn, slide in enumerate(prs.slides, 1):
            st.markdown(f"---\n**Slide {sn}:**")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        if p.text.strip(): st.write(p.text.strip())
                if shape.has_table:
                    rows = [[c.text.strip() for c in r.cells] for r in shape.table.rows]
                    if rows: st.table(pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows))
    finally:
        if path and os.path.exists(path): os.unlink(path)

def preview_xlsx(raw, fname):
    st.markdown(f"### {fname}"); path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f: f.write(raw); path = f.name
        from openpyxl import load_workbook
        wb = load_workbook(path, data_only=True)
        for sn in wb.sheetnames:
            ws = wb[sn]; st.markdown(f"**Sheet: {sn}**")
            data = [[str(c) if c is not None else "" for c in r] for r in ws.iter_rows(values_only=True)]
            if data: st.dataframe(pd.DataFrame(data[1:], columns=data[0]) if len(data) > 1 else pd.DataFrame(data), use_container_width=True)
    finally:
        if path and os.path.exists(path): os.unlink(path)

def preview_text(raw, fname):
    st.markdown(f"### {fname}")
    suffix = "." + fname.split(".")[-1].lower()
    try: content = raw.decode("utf-8")
    except: content = raw.decode("latin-1")
    if suffix == ".md": st.markdown(content)
    elif suffix == ".csv":
        import io; st.dataframe(pd.read_csv(io.StringIO(content)), use_container_width=True)
    else: st.text_area("File content", value=content, height=600, disabled=True)

def render_preview(fname, raw):
    suffix = "." + fname.split(".")[-1].lower()
    if suffix == ".pdf": preview_pdf(raw, fname)
    elif suffix == ".docx": preview_docx(raw, fname)
    elif suffix == ".pptx": preview_pptx(raw, fname)
    elif suffix in [".xlsx", ".xls"]: preview_xlsx(raw, fname)
    elif suffix in [".txt", ".md", ".csv"]: preview_text(raw, fname)
    else: st.warning(i18n.t("lib.no_preview", suffix=suffix))

if "user_email" not in st.session_state: show_auth_screen()
user_email = st.session_state.user_email
user_name = st.session_state.get("user_name", user_email)
chunk_size = DEFAULT_CHUNK_SIZE
chunk_overlap = DEFAULT_CHUNK_OVERLAP
k = DEFAULT_K

# ======================== Sidebar ========================
# The search icon opens a centre pop-up (st.dialog) with recent searches, recent chats and
# results. Recent searches are saved on disk per user (search_history/<email>.json) and
# stay until the user removes them (x on a row, or "Clear all").
# Account menu: Settings, Language, Get help (onboarding), Log out. Settings and search
# history are saved per user on disk (user_settings/, search_history/).
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
    st.rerun()   # full rerun also closes the dialog


# ---- per-user settings (saved on disk): answer language + developer info ----
SETTINGS_DIR = os.path.join(BASE_DIR, "user_settings")     # one JSON file per user
DEFAULT_SETTINGS = {"language": "auto", "dev_info": False}
LANGUAGES = {"auto": "lang.auto", "English": "lang.en", "Simplified Chinese": "lang.zh"}   # value -> translation key


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
    st.button(label, key=key, on_click=lambda k=key: st.session_state.update(_clr_pending=k))   # one prompt open at a time
    if st.session_state.get("_clr_pending") == key:
        st.caption(t("set.cannot_undo"))
        c1, c2 = st.columns(2)
        if c1.button(t("set.yes"), key=f"yes_{key}"):           # style.css makes the text red (key starts with yes_)
            action()
            st.session_state.pop("_clr_pending", None)
            st.session_state["_flash"] = message         # shown as a toast after the dialog closes
            st.rerun()                                   # full rerun: closes the dialog and refreshes the sidebar
        c2.button(t("nav.cancel"), key=f"no_{key}",              # callback runs before the rerun, so the prompt just disappears
                  on_click=lambda: st.session_state.pop("_clr_pending", None))


@st.dialog(t("set.title"))
def settings_dialog(email, name):
    s = load_user_settings(email)
    st.markdown(f"**{t('set.account')}**")
    st.caption(f"{name} · {email}")
    st.markdown(f"**{t('set.display')}**")
    dev = st.toggle(t("set.dev_toggle"), value=s["dev_info"], help=t("set.dev_help"))
    if dev != s["dev_info"]:
        s["dev_info"] = dev
        save_user_settings(email, s)
        st.session_state["dev_info"] = dev
    st.markdown(f"**{t('set.data')}**")
    st.caption(t("set.memory_note"))
    _clear_with_confirm(t("set.clear_memory"), "clr_mem", lambda: memory.clear_memory(email), t("set.memory_cleared"))
    _clear_with_confirm(t("set.clear_chats"), "clr_chats", lambda: save_user_convs(email, []), t("set.chats_cleared"))
    _clear_with_confirm(t("set.clear_searches"), "clr_search", lambda: clear_searches(email), t("set.searches_cleared"))


@st.dialog(t("lang.title"))
def language_dialog(email):
    s = load_user_settings(email)
    st.caption(t("lang.note"))
    keys = list(LANGUAGES)
    choice = st.radio(t("lang.title"), keys, index=keys.index(s["language"]),
                      format_func=lambda k: t(LANGUAGES[k]), label_visibility="collapsed")
    if st.button(t("nav.save"), key="lang_save", type="primary"):
        s["language"] = choice
        save_user_settings(email, s)
        st.rerun()


@st.dialog(t("help.title"), width="large")
def help_dialog():
    st.markdown(t("help.body"))
# ---- search history: saved on disk per user, kept until the user removes it ----
SEARCH_DIR = os.path.join(BASE_DIR, "search_history")   # one JSON file per user
MAX_SEARCHES = 100                                      # safety cap so the file cannot grow forever


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
def _on_search_submit(email):          # text_input on_change = the user pressed Enter / left the box
    log_search(email, st.session_state.get("dlg_query", ""))


def _use_search(email, term):          # clicked a recent search: fill the box and move it to the top
    st.session_state["dlg_query"] = term
    log_search(email, term)


@st.dialog(t("search.title"), width="large")
def search_dialog(email):
    convs = load_user_convs(email)
    history = load_search_history(email)          # always read from disk, so removals show at once
    q = st.text_input(t("search.title"), key="dlg_query", placeholder=t("search.placeholder"),
                      label_visibility="collapsed", on_change=_on_search_submit, args=(email,))
    ql = q.strip()

    if ql:
        results = find_chats(convs, ql)
        st.markdown(f'<div class="search-label">{t("search.results")}</div>', unsafe_allow_html=True)
        with st.container(key="search_results"):
            for idx, snip in results:
                if st.button(convs[idx].get("title") or convs[idx].get("filename", "?"), key=f"srch_{idx}", use_container_width=True):
                    _open_chat(idx)
                if snip: st.caption(snip)
            if not results:
                st.caption(t("search.none"))
    else:
        if history:
            _h1, _h2 = st.columns([4, 1])
            _h1.markdown(f'<div class="search-label">{t("search.recent_searches")}</div>', unsafe_allow_html=True)
            _h2.button(t("search.clear_all"), key="srch_clear", on_click=clear_searches, args=(email,))
            with st.container(key="search_recent"):
                for j, h in enumerate(history[:8]):
                    _r1, _r2 = st.columns([12, 1], vertical_alignment="center")
                    _r1.button(h, icon=":material/history:", key=f"srch_h_{j}", use_container_width=True,
                               on_click=_use_search, args=(email, h))
                    _r2.button(t("search.remove"), icon=":material/close:", key=f"srch_x_{j}", help=t("search.remove"),
                               on_click=remove_search, args=(email, h))
        if convs:
            st.markdown(f'<div class="search-label">{t("search.recent_chats")}</div>', unsafe_allow_html=True)
            with st.container(key="search_results"):
                for idx in range(len(convs) - 1, max(len(convs) - 9, -1), -1):
                    if st.button(convs[idx].get("title") or convs[idx].get("filename", "?"), key=f"srch_r_{idx}", use_container_width=True):
                        _open_chat(idx)
        else:
            st.caption(t("nav.no_chats"))
_flash = st.session_state.pop("_flash", None)                  # message left by a dialog action
if _flash:
    st.toast(_flash)

_settings = load_user_settings(user_email)
st.session_state["dev_info"] = _settings["dev_info"]          # app.py footers can read this
agentic_rag.set_response_language(_settings["language"])      # answer language (Settings > Language)

with st.sidebar:
    # search toggle (style.css places this icon next to the collapse arrow)
    with st.container(key="nav_search"):
        if st.button(t("search.title"), icon=":material/search:", key="btn_search", help=t("nav.search_help")):
            search_dialog(user_email)

    with st.container(key="brand_wrap"):
        st.markdown('<div class="brand">NexusAsk</div>', unsafe_allow_html=True)

    if st.button(t("nav.new_chat"), icon=":material/add:", key="nav_new", use_container_width=True):
        with st.spinner(t("nav.saving")):
            archive_current()
        for key in ["messages", "doc_chunks", "uploaded_filenames", "processed_files", "file_raw_bytes", "applied_chunk_size", "applied_chunk_overlap", "preview_file", "review_conversation", "show_uploader", "library_view", "preview_from_library"]:
            if key in st.session_state: del st.session_state[key]
        if "continuing_conversation_idx" in st.session_state: del st.session_state["continuing_conversation_idx"]
        st.rerun()

    if st.button(t("nav.library"), icon=":material/folder_open:", key="nav_library", use_container_width=True):
        st.session_state.library_view = True
        if "review_conversation" in st.session_state: del st.session_state.review_conversation
        if "preview_file" in st.session_state: del st.session_state.preview_file
        st.rerun()

    convs = load_user_convs(user_email)
    # lazy title upgrade: generate the real title in the background for any conversation still using a temp title
    for _c in convs:
        _cid = _c.get("id")
        if _c.get("title_is_final") is False and _cid and _cid not in _title_in_progress:
            _title_in_progress.add(_cid)
            threading.Thread(target=upgrade_title_async, args=(user_email, _cid), daemon=True).start()

    st.markdown(f'<div class="side-label">{t("nav.chats")}</div>', unsafe_allow_html=True)
    if convs:
        _active_idx = st.session_state.get("review_conversation", st.session_state.get("continuing_conversation_idx"))
        for idx in range(len(convs) - 1, -1, -1):
            _name = convs[idx].get("title") or convs[idx].get("filename", "?")
            # the open chat gets a different container key so style.css can highlight it
            _frame_key = f"convframe_active_{idx}" if idx == _active_idx else f"convframe_{idx}"
            with st.container(border=False, key=_frame_key):
                _t1, _t2 = st.columns([8, 1])
                with _t1:
                    if st.button(_name, key=f"conv_{idx}", use_container_width=True):   # full title; CSS adds the ellipsis
                        st.session_state.review_conversation = idx
                        if "preview_file" in st.session_state: del st.session_state.preview_file
                        st.rerun()
                with _t2:
                    with st.popover("⋮", use_container_width=True, key=f"menu_{idx}"):
                        if st.button(t("nav.delete"), icon=":material/delete:", key=f"del_{idx}", use_container_width=True):
                            confirm_delete_conv(user_email, idx)
                        if st.button(t("nav.rename"), icon=":material/edit:", key=f"ren_{idx}", use_container_width=True):
                            st.session_state["renaming_conv"] = idx
                            st.rerun()
                if st.session_state.get("renaming_conv") == idx:
                    _new = st.text_input(t("nav.rename"), value=_name, key=f"rn_{idx}", label_visibility="collapsed")
                    _rc1, _rc2 = st.columns(2)
                    if _rc1.button(t("nav.save"), key=f"rsv_{idx}"):
                        _convs = load_user_convs(user_email)
                        if idx < len(_convs): _convs[idx]["title"] = _new; save_user_convs(user_email, _convs)
                        del st.session_state["renaming_conv"]; st.rerun()
                    if _rc2.button(t("nav.cancel"), key=f"rnc_{idx}"):
                        del st.session_state["renaming_conv"]; st.rerun()
    else:
        st.caption(t("nav.no_chats"))

    # account area: pinned to the very bottom; click for the menu (Settings / Language / Get help / Log out)
    with st.container(key="side_account"):
        _initial = (user_name[:1] or "?").upper()
        if not _initial.isalnum():
            _initial = "?"
        st.markdown(f"<style>.st-key-side_account{{--initial:'{_initial}'}}</style>", unsafe_allow_html=True)
        with st.popover(user_name, use_container_width=True):
            st.caption(user_email)
            if st.button(t("acct.settings"), icon=":material/settings:", key="acct_settings", use_container_width=True):
                st.session_state.pop("_clr_pending", None)
                settings_dialog(user_email, user_name)
            if st.button(t("acct.language"), icon=":material/language:", key="acct_language", use_container_width=True):
                language_dialog(user_email)
            if st.button(t("acct.help"), icon=":material/help:", key="acct_help", use_container_width=True):
                help_dialog()
            st.divider()
            if st.button(t("acct.logout"), icon=":material/logout:", key="acct_logout", use_container_width=True):
                confirm_logout(user_email)

# ======================== Main ========================
# (greeting "Hi! {name}, what's on your mind?" now shows only on the initial get-started screen)

if "library_view" in st.session_state:
    if st.button(i18n.t("lib.back_chat"), icon=":material/arrow_back:"):
        del st.session_state.library_view; st.rerun()
    st.divider()
    st.markdown(f"## {i18n.t('nav.library')}")
    _lib_files = list_library_files(user_email)
    if _lib_files:
        for fname in _lib_files:
            _lr1, _lr2 = st.columns([9, 1])
            with _lr1:
                if st.button(f"{fname}", key=f"lib_{fname}", icon=":material/description:"):
                    _raw = read_library_file(user_email, fname)
                    if _raw is not None:
                        st.session_state.setdefault("file_raw_bytes", {})[fname] = _raw
                        st.session_state.preview_file = fname
                        st.session_state.preview_from_library = True
                        del st.session_state.library_view
                        st.rerun()
            with _lr2:
                if st.button("", icon=":material/delete:", key=f"libdelpage_{fname}", help=i18n.t("nav.delete")):
                    confirm_delete_library_file(user_email, fname)
    else:
        st.caption(i18n.t("lib.empty"))

elif "preview_file" in st.session_state:
    fname = st.session_state.preview_file
    raw_bytes = st.session_state.get("file_raw_bytes", {})
    if st.session_state.get("preview_from_library"):
        if st.button(i18n.t("lib.back_lib"), icon=":material/arrow_back:"):
            del st.session_state.preview_file
            st.session_state.library_view = True
            del st.session_state.preview_from_library
            st.rerun()
    else:
        if st.button(i18n.t("lib.back_chat"), icon=":material/arrow_back:"):
            del st.session_state.preview_file; st.rerun()
    st.divider()
    if fname in raw_bytes: render_preview(fname, raw_bytes[fname])
    else: st.warning(i18n.t("lib.not_found", name=fname))

elif "review_conversation" in st.session_state:
    convs = load_user_convs(user_email)
    idx = st.session_state.review_conversation
    if idx < len(convs):
        c = convs[idx]
        st.markdown(f"### {c.get('filename', '?')} — {c.get('timestamp', '')}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(i18n.t("review.back"), icon=":material/arrow_back:"): del st.session_state.review_conversation; st.rerun()
        with col2:
            if st.button(i18n.t("review.continue"), icon=":material/play_arrow:"):
                st.session_state.messages = c.get("messages", []).copy()
                st.session_state.doc_chunks = c.get("doc_chunks", []).copy()
                st.session_state.uploaded_filenames = c.get("filename", "").split(", ")
                st.session_state.applied_chunk_size = DEFAULT_CHUNK_SIZE
                st.session_state.applied_chunk_overlap = DEFAULT_CHUNK_OVERLAP
                st.session_state.continuing_conversation_idx = idx
                del st.session_state.review_conversation; st.rerun()
        st.divider()
        for m in c.get("messages", []):
            with st.chat_message(m["role"]): st.write(m["content"])
        st.divider(); st.markdown(f"### {i18n.t('chat.continue_asking')}")
        _pending = st.session_state.pop("pending_retry_followup", None)
        if _pending:
            def _footer(text):
                if st.session_state.get("dev_info"):
                    st.caption(f"~{str_mod.count_tokens(text)} tokens · {'Cached' if st.session_state.get('_last_stream_cached') else 'Fresh'} · sim {st.session_state.get('_last_stream_sim', 0.0):.2f}")
                    show_langsmith_trace()
            a, _steps = step_ui.run_assistant_turn(
                lambda status: agentic_answer_cached(_pending, c.get("doc_chunks", []), DEFAULT_K, status, email=user_email),
                footer=_footer)
            if a:
                convs[idx]["messages"] = st.session_state.messages.copy()
                save_user_convs(user_email, convs)
        q = st.chat_input(i18n.t("chat.followup"))
        if q:
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"): st.write(q)
            def _footer(text):
                if st.session_state.get("dev_info"):
                    st.caption(f"~{str_mod.count_tokens(text)} tokens · {'Cached' if st.session_state.get('_last_stream_cached') else 'Fresh'} · sim {st.session_state.get('_last_stream_sim', 0.0):.2f}")
                    show_langsmith_trace()
            a, _steps = step_ui.run_assistant_turn(
                lambda status: agentic_answer_cached(q, c.get("doc_chunks", []), DEFAULT_K, status, email=user_email),
                footer=_footer)
            if a:
                convs[idx]["messages"] = st.session_state.messages.copy()
                save_user_convs(user_email, convs)
            else:
                if st.button(i18n.t("action.retry"), key="retry_followup_err", icon=":material/refresh:"):
                    st.session_state["pending_retry_followup"] = q
                    st.rerun()
    else:
        del st.session_state.review_conversation; st.rerun()

else:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("doc_chunks", [])
    uploaded_files = None
    q = None
    _pending_flip = False
    _pq = st.session_state.pop("pending_q", None)
    _pc = st.session_state.pop("pending_continue", None)
    if _pq is not None:
        q = _pq
        _pending_flip = True
    autoscroll.install(force=_pq is not None or _pc is not None)
    step_ui.settle_interrupted()      # keep the text of an answer cut off by Stop (or a new message)
    # ----- INITIAL state (no messages): centered greeting + centered chat box -----
    if not st.session_state.get("messages"):
        # check for a submitted question BEFORE rendering the centered input, so the flip is instant (no lingering)
        _iq = st.session_state.get("initial_q", "")
        if _iq:
            process_uploads(st.session_state.get("uploader"))
            st.session_state.setdefault("messages", []).append({"role": "user", "content": _iq})
            st.session_state["pending_q"] = _iq
            st.session_state["initial_q"] = ""
            st.rerun()
        st.markdown('<div style="height:24vh"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hello">{i18n.t("chat.greeting", name=user_name)}</div>', unsafe_allow_html=True)
        _gc1, _gc2, _gc3 = st.columns([2, 8, 2])
        with _gc2:
            _ic1, _ic2 = st.columns([10, 1])
            with _ic1:
                st.text_input(i18n.t("chat.ask_anything"), key="initial_q", placeholder=i18n.t("chat.ask_anything"), label_visibility="collapsed", on_change=_on_initial_submit)
            with _ic2:
                uploaded_files = upload_popover()
    else:
        # message history
        for msg_idx, m in enumerate(st.session_state.get("messages", [])):
            with st.chat_message(m["role"]):
                if m["role"] == "user" and st.session_state.get("editing_msg") == msg_idx:
                    edited = st.text_area("Edit", value=m["content"], key=f"et_{msg_idx}", height=100, label_visibility="collapsed")
                    _es1, _es2 = st.columns(2)
                    with _es1:
                        if st.button(i18n.t("nav.save"), key=f"esv_{msg_idx}", icon=":material/check:"):
                            _after = st.session_state.messages[msg_idx + 2:] if msg_idx + 2 < len(st.session_state.messages) else []
                            st.session_state.messages = st.session_state.messages[:msg_idx]
                            st.session_state.messages.append({"role": "user", "content": edited})
                            del st.session_state.editing_msg
                            a, _steps = step_ui.run_assistant_turn(
                                lambda status: agentic_answer_cached(edited, st.session_state.doc_chunks, k, status, email=user_email))
                            if a:
                                st.session_state.messages.extend(_after)
                                if st.session_state.get("dev_info"):
                                    show_langsmith_trace()
                            st.rerun()
                    with _es2:
                        if st.button(i18n.t("nav.cancel"), key=f"ecn_{msg_idx}", icon=":material/close:"):
                            del st.session_state.editing_msg
                            st.rerun()
                else:
                    step_ui.render_history_message(msg_idx, m, msg_idx == len(st.session_state.messages) - 1, user_email)
        # chat_input (bottom; only in the conversation view — the start page uses the centered text_input above)
        _chunks = st.session_state.get("doc_chunks", [])
        _ci = st.chat_input(i18n.t("chat.ph_docs") if _chunks else i18n.t("chat.ph_any"), accept_file='multiple', file_type=['pdf', 'docx', 'pptx', 'xlsx', 'xls', 'txt', 'md', 'csv'])
        if _ci:
            q = _ci.text if hasattr(_ci, 'text') else _ci
            uploaded_files = list(getattr(_ci, 'files', None) or [])
        # surface an unanswered question (failed answer / page refresh) when nothing is being generated
        if not q and _pc is None and "review_conversation" not in st.session_state:
            step_ui.render_unanswered()
    # loaded annotation (one-shot: shows once right after an upload)
    if st.session_state.pop("_loaded_just_now", False):
        _lfns = st.session_state.get("uploaded_filenames", [])
        if _lfns:
            st.caption(i18n.t("chat.loaded", files=", ".join(_lfns), n=len(st.session_state.get('doc_chunks', []))))
    # ----- process uploads captured from the chat_input or the ＋ popover -----
    if uploaded_files:
        with st.spinner(i18n.t("chat.reading_files")):
            _new = process_uploads(uploaded_files)
        if _new:
            st.session_state["_loaded_just_now"] = True
        if _new and not q:
            st.success(i18n.t("chat.uploaded", n=len(st.session_state.uploaded_filenames)))
            st.rerun()
    # ----- generate answer if a question was submitted (real streaming) -----
    if q:
        if not _pending_flip or not st.session_state.get("messages"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state["pending_q"] = q
            st.rerun()
        def _footer(text):
            answer_actions.render(len(st.session_state.messages), {"role": "assistant", "content": text}, user_email, is_last=True)
            if st.session_state.get("dev_info"):
                st.caption(f"~{str_mod.count_tokens(text)} tokens · {'Cached' if st.session_state.get('_last_stream_cached') else 'Fresh'} · sim {st.session_state.get('_last_stream_sim', 0.0):.2f}")
                show_langsmith_trace()
        step_ui.run_assistant_turn(
            lambda status: agentic_answer_cached(q, st.session_state.get("doc_chunks", []), k, status, email=user_email),
            footer=_footer)
    elif _pc is not None:
        # "Click to continue": ask the model to carry on from where the answer was stopped
        step_ui.run_assistant_turn(
            lambda status: agentic_answer_cached(step_ui.CONTINUE_PROMPT, st.session_state.get("doc_chunks", []), k, status, email=user_email),
            continue_idx=_pc)