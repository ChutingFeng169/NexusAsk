# NexusAsk

> A Streamlit-based **Agentic RAG** assistant: native tool-calling, real token-by-token streaming, a bilingual **English / Chinese** interface, a UI with a persistent file Library, chat search, cross-conversation memory, semantic cache, and LangSmith tracing — built on a pinned legacy LangChain 0.1.20 stack (no LangGraph).

NexusAsk has evolved from a simple Chroma-vector-store RAG into a full agentic assistant: it calls tools (search the docs / web / Wikipedia / calculate), streams answers live, remembers you across conversations, keeps every file you've ever uploaded in a personal Library — and the whole interface speaks English or Simplified Chinese, following each user's Settings > Language.

---

## Features

### Agent & answers

- **Native tool-calling agent** over httpx — `search_docs`, `search_web`, `calculator`, `wikipedia_search` (the model emits real `tool_calls`; a plain-Python loop runs them, up to 8 steps).
- **Real token-by-token streaming** — `llm_chat_stream` uses `stream=True` + SSE, yields `delta.content` live and reconstructs `delta.tool_calls` from the stream so the tool-calling loop keeps working mid-stream.
- **One status box per answer** — a single collapsed `st.status` driven by `StepTracker` (step_ui.py) shows translated step labels — "Rewriting your question", "Thinking (step N)", "Searching your documents · 3 result(s)", "Answer ready" — each with its duration; the recorded steps re-render under a "Show steps" expander in the chat history.
- **Answer actions** — small grey copy / retry icons under the newest answer (answer_actions.py): copy uses a tiny HTML iframe so the browser clipboard gets a real click; retry drops the answer, evicts it from the semantic cache and re-asks through the normal pending-question flow.
- **Edit & resend** — edit any of your own messages; the rest of the thread is dropped and the answer regenerated from there.
- **Failed answers** — one friendly red message + a red retry icon (the real traceback goes to the log); a stale unanswered question (failed answer or page refresh) shows the same retry row, so you're never stuck.
- **Semantic cache** — local `paraphrase-multilingual-MiniLM-L12-v2` embeddings + cosine similarity; a repeated/paraphrased question ≥ 0.9 hits the cache (no LLM call) and streams the cached answer. Only used when there is no conversation history.
- **Safety guard + conditional disclaimer** — sensitive words short-circuit the agent before any model call; a (translated) AI-generated disclaimer is appended only to answers that used web search or Wikipedia.
- **Long-term memory** — LLM-extracted durable facts (≤ 30 per user), injected into the agent's system prompt so it remembers you across conversations; clearable from Settings.
- **LangSmith tracing** — `@traceable` bridges the raw-httpx gap; the optional developer info under answers shows token count, cache status and the trace.

### Interface

- **English / Chinese interface (i18n)** — every user-facing string lives in `i18n.py` in both languages. The language follows each user's **Settings > Language** (`"auto"` = the browser's language) and the same setting also controls the answer language. The login screen resolves the email being typed in and defaults to English for `"auto"`/unknown.
- **Claude / Gemini-like design** — one `style.css` theme + a `.streamlit/config.toml` theme: narrow centred reading column, fixed bottom chat bar with a grey disclaimer, sidebar with brand / search / chat list / fixed account bar, right-aligned user bubbles, plain assistant text.
- **Auto-scroll** — the page follows the answer while it streams; scrolling up to read stops the following (scroll back to the bottom to resume); sending a message or pressing retry jumps to the bottom.
- **ChatGPT-style get-started screen** — centered greeting + centered text input (＋ popover for files), then an instant flip to the bottom chat input after the first message.
- **Sidebar** — brand, search (magnifier icon), New chat, Library, the Chats list (rename / delete via the ⋮ menu, active-conversation highlight), and an account bar with Settings / Language / Get help / Log out.
- **Search chats** — a centered search dialog over titles + message contents, with per-user recent searches (clear all / remove individually).
- **Settings dialog** — Display (developer-info toggle), Language, and Your data (clear long-term memory / chat history / search history — each behind a red "cannot be undone" confirm).
- **Get help** — an in-app help dialog explaining everything you can do.
- **Confirm dialogs** (`@st.dialog`, all translated) — delete file, delete conversation, clear all history, clear memory, logout.

### Accounts & data

- **Auth** — email + password login/registration, bcrypt-hashed in `users.json`; logout checkpoints the current conversation first.
- **Persistent per-user Library** — every uploaded file is saved to disk (`library/<user>/`); a dedicated Library page lists them all with preview + delete (confirm dialog).
- **Per-conversation checkpoints** — conversations auto-save (messages + truncated chunks) and can be reviewed, resumed and continued; titles are LLM-generated by a background thread (the first 24 chars serve as a temp title meanwhile).
- **Uploads** — pdf, docx, pptx, xlsx, txt, md, csv, via the ＋ popover or drag-and-drop onto the chat input; file preview in-app for every type.

---

## Crucial trade-offs & decisions

These are the non-obvious design choices that shape the app — each is deliberate.

### 1. Pinned legacy LangChain 0.1.20 (no LangGraph)
The stack is intentionally pinned to `langchain==0.1.20` / `langchain-core==0.1.53`. LangGraph is **incompatible** with the 0.1.x core, so the agent loop is **plain Python + native function-calling** (a hand-rolled `for step in range(max_steps)` loop, `max_steps=8`), not a graph runtime. **Trade-off:** no graph-style orchestration; the loop is simple + debuggable, but a future LangChain upgrade would require migrating to LangGraph.

### 2. Real streaming skips faithfulness self-correction
The streaming path (`agentic_answer_stream`) **skips** the faithfulness self-correction that the non-streaming path (`agentic_answer`) has. Self-correction needs the **full** answer to run the embedding-similarity check, which conflicts with streaming tokens live. **Trade-off:** the streamed answer is final (no re-grounding); live streaming won out over the (rare) self-check.

### 3. Streaming may surface the model's "narration"
When the model calls a tool, it sometimes emits narration content (e.g. "Let me search…") **alongside** `tool_calls`. With live streaming this is yielded as it arrives. **Trade-off:** document Q&A (no tools) streams cleanly; web-search questions may show narration before the final answer. (Could be hidden with a per-step buffer, at the cost of the live token feel.)

### 4. tool_calls parsed from the SSE stream
`llm_chat_stream` parses `delta.tool_calls` (accumulated by index: id / function.name / function.arguments) to drive the tool-calling loop mid-stream. ***Trade-off:** it depends on the DeepSeek cpolar proxy's streaming `tool_calls` format matching the OpenAI SSE delta schema (verified — it does); a provider with a different delta format would need a new parser.

### 5. @traceable bridges the httpx tracing gap
`LANGCHAIN_TRACING_V2` only traces langchain calls, not raw httpx. The `@traceable` decorator on `llm_chat`/tools bridges this. **Trade-off:** the streaming `llm_chat_stream` is **not** `@traceable` (generators + tracing don't mix cleanly), so streamed LLM calls aren't traced.

### 6. Cache-hit streams with a small delay
The cache (local MiniLM embeddings, cosine ≥ 0.9) is consulted and stored only **when there is no conversation history**, because a follow-up question's meaning depends on earlier turns and a cached answer would be stale or wrong. A cache hit yields the cached answer in 3-char chunks with a 0.01s pause (not instantly), so it *feels* like a stream. **Trade-off:** cached answers aren't instant; they stream briefly (~0.7s for 200 chars) for visual consistency with real streams.

### 7. Background threads for titles and memory (eventually consistent)
Two slow LLM jobs are moved off the click path so "New chat" is instant instead of ~2–20s:

* Titles: a new conversation saves a `temp title` (first question, truncated to 24 chars + "…") + `title_is_final=False`; the real title is generated by a background thread when the sidebar next renders the conversation.
* Memory: `memory.extract_memory` runs in a fire-and-forget **background thread**` on "New chat" and logout.

**Trade-off:** both are eventually consistent — the title is temporary for a moment and then "silently" upgrades on the next rerun, and the very next question might not yet see the just-extracted memory facts (it is up to date by the following question).

### 8. Theme via CSS `data-testid` selectors (transparent bars)
The Streamlit header + bottom container are made transparent via CSS so the warm page background shows through behind the fixed chat bar, and the whole Claude-like look (sidebar, dialogs, icon rows, bubbles) is built on Streamlit's `data-testid` selectors in `style.css`. **Trade-off:** a Streamlit upgrade could rename those selectors; the styles are grouped per UI area so fixes stay local.

### 9. i18n is a plain dict + `t()` — called module-qualified in the big files
`i18n.py` keeps both languages in one `STRINGS` dict; `t(key)` falls back current-language → English → the key itself. `app.py` and `sidebar_ui.py` call it **module-qualified** (`i18n.t(...)`) because both files already use `t` as a local variable (event types, text snippets) that would shadow a `from i18n import t` and cause `TypeError`s. **Trade-off:** slightly more verbose call sites; zero risk of shadowing.

### 10. Dialog titles are re-decorated on every call
`@st.dialog` freezes its title at decoration time. `app.py` re-executes on every run, so `@st.dialog(i18n.t(...))` there is always current — but an **imported** module (`sidebar_ui.py`) is decorated once per process, which would freeze its dialog titles in the very first run's language. sidebar_ui's `_t_dialog(title_key)` therefore re-applies `st.dialog(i18n.t(key))` on every call. **Trade-off:** one extra wrapper layer for the 4 sidebar dialogs; titles always follow the language.

### 11. Auto-scroll is injected JS (no Streamlit primitive)
Streamlit has no "scroll to bottom" API, so `autoscroll.py` injects a small idempotent script (`st.html(..., unsafe_allow_javascript=True)` on Streamlit ≥ 1.55, iframe fallback on older versions). A `MutationObserver` follows new/growing content; a capture-phase scroll listener stops following when you're more than 160 px from the bottom and resumes at the bottom; `force=True` (new question / retry) jumps to the bottom for ~1.8 s while the page settles. Its container is collapsed by `.st-key-autoscroll` in `style.css`. **Trade-off:** the script relies on the page's DOM structure (`section[data-testid="stMain"]`, …).

### 12. The login screen resolves the email being typed in
Before login there is no session identity, so the saved language can't be known — except from the email field: on every rerun the login screen looks up the setting saved for `login_email` / `reg_email`, and `"auto"`, unknown and empty emails default to **English** (`i18n.init(email, auto="en")`). After login, `"auto"` follows the browser again (`st.context.locale`). **Trade-off:** the login screen may switch languages mid-typing, once a full email with a saved setting is recognized.

---

## Architecture

- **Pinned legacy stack:** `langchain==0.1.20`, `langchain-core==0.1.53`, `langsmith==0.1.147`.
- **LLM transport:** all chat calls go to an OpenAI-compatible endpoint (the DeepSeek cpolar proxy) over **httpx** (streaming + non-streaming).
- **Agent loop:** plain Python + native function-calling (`agentic_rag.py`; `max_steps=8`, `max_self_correct=2`). No LangGraph.
- **UI modules:** `app.py` (auth, chat, Library/preview/review pages, confirm dialogs) + `sidebar_ui.py` (sidebar, Settings/Language/Get-help/Search dialogs, per-user settings) + `step_ui.py` (one status box per turn) + `answer_actions.py` (copy/retry) + `autoscroll.py` (auto-scroll) + `i18n.py` (English / Simplified Chinese).
- **Theme:** `style.css` injected once per run by `app.py`, plus `.streamlit/config.toml`; texts that live inside CSS (the disclaimer) are fed per-run by `i18n.css_vars()` as CSS variables (`--disclaimer`).
- **Storage:** JSON / file-backed (no external DB) — `users.json`, `conversations/`, `library/`, `memory/`, `user_settings/`, `search_history/`.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app: auth, i18n init + theme injection, chat UI, Library / preview / review pages, confirm dialogs, semantic cache, checkpoints, streaming wiring |
| `agentic_rag.py` | Agent loop + 4 tools + streaming (`llm_chat`, `llm_chat_stream`, `agentic_answer`, `agentic_answer_stream`), guard, disclaimer, answer language |
| `sidebar_ui.py` | Sidebar (brand, search, New chat, Library, chat list, account bar) + Settings / Language / Get help / Search dialogs + per-user settings on disk |
| `step_ui.py` | One status box per assistant turn (`StepTracker`), "Show steps" history, failed-answer row + retry, stale-question retry |
| `answer_actions.py` | The copy / retry icon row under the newest answer |
| `autoscroll.py` | Keeps the chat scrolled to the newest text (follows streaming, respects scroll-up, force-jump on send/retry) |
| `i18n.py` | English / Simplified Chinese interface strings + `init` / `t` / `resolve` / `tool_label` / `css_vars` |
| `memory.py` | Long-term memory: `extract_memory` / `load_memory` / `format_memory_prompt` / `clear_memory` |
| `retrieval.py` | Hybrid document retrieval (BM25 + keyword, fused with RRF) |
| `streaming_tokens_retry.py` | httpx SSE streaming (`stream_chat`), token counting (`count_tokens`), content retry, context truncation |
| `rate_limit_fallback.py` | tenacity retry (429/5xx/timeout/network) + jitter backoff + fallback model (non-agentic path) |
| `retry_helper.py` | retry + logging helpers shared across paths |
| `timeout_handling.py` | timeout for model calls (non-agentic path) |
| `async_pipelines.py` | async/concurrent multi-file helpers |
| `simple_eval.py` | faithfulness (embedding similarity) |
| `style.css` | the whole Claude-like theme (page frame, sidebar, search dialog, icon rows, confirms, bubbles, autoscroll container) |
| `.streamlit/config.toml` | Streamlit theme (light, `#c15f3c` accent) + minimal toolbar |
| `test_i18n.py` | i18n tests: en/zh key parity, placeholder parity, format safety, resolve + login-screen defaults, `t()`/`tool_label()` fallbacks |
| `demo/` | standalone demo scripts (429 recovery, streaming, agentic run, timeout, async pipelines) |
| `experiment/` | offline evaluation experiments (RAGAS, chunking) |
| `requirements.txt` | pinned dependencies |
| `.env` | runtime config (keys, endpoint, tracing) |
| `users.json` | accounts (bcrypt-hashed passwords) |
| `conversations/` | per-user conversation JSON |
| `library/` | per-user uploaded files (raw bytes) |
| `memory/` | per-user memory JSON |
| `user_settings/` | per-user settings JSON (`language`, `dev_info`) |
| `search_history/` | per-user recent search terms |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then edit .env (see Configuration)
streamlit run app.py              # -> http://localhost:8501
```

The first run downloads the local embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB) for the semantic cache.

## Configuration (`.env`)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for the OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` | Endpoint URL (e.g. the DeepSeek cpolar proxy, `.../v1`) |
| `LLM_MODEL` | Model name (e.g. `deepseek-v4-flash`) |
| `LANGCHAIN_TRACING_V2` | `true` to enable tracing |
| `LANGSMITH_API_KEY` | LangSmith API key |
| `LANGSMITH_PROJECT` | LangSmith project name (e.g. `nexusask`) |

Key constants: `DEFAULT_CHUNK_SIZE=500`, `DEFAULT_CHUNK_OVERLAP=50`, `DEFAULT_K=4`, `SEMANTIC_CACHE_THRESHOLD=0.9`, `REAL_MODEL_TIMEOUT=15.0`, `MAX_CONTEXT_CHARS=12000`, `MAX_CONTEXT_TOKENS=3000`, `MAX_STORED_CHUNK_CHARS=500` (app.py); `MAX_STEPS=8`, `MAX_SELF_CORRECT=2` (agentic_rag.py); `MAX_MEMORIES=30` (memory.py); `MAX_SEARCHES=100` (sidebar_ui.py).

## How it works — the agent pipeline

```
guard_input -> rewrite_query -> [tool-calling loop, ≤ 8 steps] -> answer -> disclaimer -> [wiki source]
```

1. **guard_input** — before-model safety: sensitive words short-circuit (refusal, no model call; status "Blocked (sensitive word)").
2. **rewrite_query** — LLM rewrites the question for retrieval (falls back to the original question on error).
3. **Tool-calling loop** — `llm_chat_stream` streams content + accumulates `tool_calls`. If `tool_calls` → run the tool (progress label via `i18n.tool_label`), append the result, loop. If no `tool_calls` → the streamed content is the final answer.
4. **(stream mode)** faithfulness self-correction skipped.
5. **Disclaimer** — appended (translated) if `search_web`/`wikipedia_search` was used, reminding that the answer is AI-generated.
6. **`[Source: Wikipedia](url)`** — appended if `wikipedia_search` returned a URL.

### Tools

| Tool | Purpose | Source |
|------|---------|--------|
| `search_docs(query, chunks, k)` | keyword-overlap retrieval over uploaded doc chunks | local |
| `search_web(query, n)` | DuckDuckGo web search (used first for current events / when you mention a website) | `duckduckgo-search` |
| `calculator(expression)` | safe math evaluation (sandboxed) | stdlib `math` |
| `wikipedia_search(query, sentences)` | Wikipedia summary + page URL | `wikipedia` |

## Streaming & the step UI

- **`llm_chat_stream(messages, tools, _out)`** — httpx `stream=True` + SSE; yields `delta.content`; accumulates `delta.tool_calls` into `_out`.
- **`agentic_answer_stream(...)`** — generator; drives the loop; yields the final answer token-by-token; appends disclaimer/wiki.
- **`agentic_answer_cached(...)`** — generator; cache hit → chunked stream; cache miss → real stream. The cache is consulted (and stored) only when there is no conversation history.
- **`step_ui.run_assistant_turn`** renders ONE assistant message: a single **collapsed** status box ("Nexus is thinking…" / "Nexus 正在思考…"), then the answer streamed via `st.write_stream` into an `st.empty()` created *before* any slow call (so stale content clears at once), then the copy/retry row. `StepTracker` records `(label, seconds)` per step, shows "Done · Ns" / "Failed · Ns", and the saved steps re-render under a "Show steps" / "查看步骤" expander in the history.
- **Failures** — one friendly red error + retry icon (`step_ui._failure_row`); the real traceback only goes to the log. If the last message is an unanswered question (failed answer or page refresh), `render_unanswered` shows the same retry row.
- **Auto-scroll** — `autoscroll.install(force=pending_question is not None)` runs once per run at the top of the chat view.

## Interface language (English / 中文)

- Every user-facing string lives in `i18n.STRINGS` **in both languages** — `test_i18n.py` enforces that both tables have the same keys with identical `{placeholders}` and that every string formats cleanly.
- `i18n.init()` runs once per run at the top of `app.py`:
  - **logged in** → the account's Settings > Language from `user_settings/<email>.json`: `English`, `Simplified Chinese`, or `auto` (follows the browser via `st.context.locale`);
  - **login screen** → the setting saved for the email being typed in; `auto`/unknown/empty → **English**.
- The same setting also drives the **answer language** (`agentic_rag.set_response_language`): `auto` lets the model match the language of your question; an explicit choice appends "Always write your final answer in …" to the system prompt.
- `i18n.tool_label()` names the agent tools in the status line ("Searching your documents", "正在联网搜索", …); `i18n.css_vars()` feeds the disclaimer text to `style.css` as `--disclaimer`.
- Not translated on purpose: the brand "NexusAsk", log lines, developer info (token counts, LangSmith trace), and the prompts sent to the model.
- **Adding a string:** add the key to *both* `en` and `zh` (identical `{placeholders}`), call `t("key")` (or `i18n.t("key")` in app.py / sidebar_ui.py) — `pytest test_i18n.py` keeps the tables honest.

## Auto-scroll

`autoscroll.py` keeps the chat pinned to the newest text, like Claude / ChatGPT: while an answer streams in the page follows it; if you scroll up to read, the following stops (> 160 px from the bottom counts as "reading"); scrolling back to the bottom resumes it; sending a message or pressing retry jumps straight to the bottom and follows for ~1.8 s while the page settles (`force=True`). The script is idempotent (listeners registered once per window) and its container takes no space (`.st-key-autoscroll` in style.css).

## UI theme

`style.css` (injected once per run by `app.py`) + `.streamlit/config.toml` style the whole app from a handful of CSS variables (`--accent: #c15f3c`, warm off-white background): narrow centred reading column, a fixed bottom chat bar with the grey disclaimer (its text arrives per-run from `i18n.css_vars()` as `--disclaimer`), sidebar layout (Georgia-serif brand, search icon, chat rows with circle bullets, active highlight, fixed account bar), the centred search dialog, small grey copy/retry icon rows, red destructive confirms, right-aligned user bubbles and plain assistant text. Change `--accent` (and the matching `primaryColor` in `config.toml`) to restyle everything.

## Long-term memory

- **`extract_memory(email, messages)`** — LLM extracts 1-5 durable facts per conversation (per-user JSON, cap 30). Runs in a background thread on new-chat/logout.
- **`load_memory` / `format_memory_prompt`** — facts injected into the agent's system prompt under `## Remembered about this user`.
- Clearable anytime from Settings > Your data (with a red confirm).

## Semantic cache

The question is embedded (local MiniLM); cosine vs cached questions. If similarity ≥ 0.9 (and there is no conversation history), the cached answer is returned (chunked stream, no LLM call). Otherwise: agentic stream + cache the result. The status line shows "Answer from a similar earlier question" on a hit.

## Per-user data on disk

| Path | Content |
|------|---------|
| `users.json` | accounts (bcrypt-hashed passwords) |
| `conversations/<email>.json` | saved conversations (messages, truncated chunks, title, timestamp) |
| `library/<email>/` | every uploaded file (raw bytes) |
| `memory/<email>.json` | long-term memory facts (≤ 30) |
| `user_settings/<email>.json` | settings: `language` (`auto`/`English`/`Simplified Chinese`), `dev_info` |
| `search_history/<email>.json` | recent chat-search terms (≤ 100) |

## Testing

```bash
python -m pytest test_i18n.py -q
```

8 tests: en/zh key parity, no empty strings, placeholder parity, every string formats with its own placeholders, `resolve()` mapping, the login-screen English default, and `t()`/`tool_label()` translation + fallbacks.

## Notes / limitations

- **Image / vision upload** is not supported (the current model can't see images).
- **Streaming narration** — see trade-off #3.
- **Pinned legacy stack** — see trade-off #1.
- `bcrypt` (auth) and `langsmith` (tracing) are installed in this project's venv but are not pinned in `requirements.txt` — add them there before deploying elsewhere.




