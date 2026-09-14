"""memory.py — long-term memory store (cross-conversation), legacy-stack-safe.

The langgraph `Store` equivalent: per-user durable facts extracted from each
conversation and injected into the agent's system prompt so the model
"remembers" the user across conversations. Backed by JSON files (no DB).
"""
import json
import os
import re

import agentic_rag  # reuse llm_chat for extraction (no circular import)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
MAX_MEMORIES = 30


def _safe_name(email):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", email or "anon")


def get_memory_file(email):
    return os.path.join(MEMORY_DIR, f"{_safe_name(email)}.json")


def load_memory(email):
    """Return the list of durable memory strings for a user."""
    fp = get_memory_file(email)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f) or []
        except Exception:
            return []
    return []


def save_memory(email, items):
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(get_memory_file(email), "w", encoding="utf-8") as f:
        json.dump(items[:MAX_MEMORIES], f, ensure_ascii=False, indent=2)


def _dedup(items):
    seen = []
    for it in items:
        s = (it or "").strip()
        if s and s not in seen:
            seen.append(s)
    return seen


def extract_memory(email, messages):
    """LLM-extract durable facts from a conversation, merge into the store.
    Returns the merged memory list. No-op (returns existing) if messages empty."""
    if not messages:
        return load_memory(email)
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in messages[-10:])[:3000]
    new = []
    try:
        msg = agentic_rag.llm_chat([
            {"role": "system", "content": "Extract 1-5 durable facts/preferences about the user or their domain from this conversation (NOT question-specific trivia). Output JSON: {\"memories\": [\"short fact\", ...]}. If nothing durable, return {\"memories\": []}."},
            {"role": "user", "content": convo},
        ])
        txt = (msg.get("content") or "").strip()
        blob = txt[txt.find("{"):txt.rfind("}") + 1]
        obj = json.loads(blob) if blob else {}
        new = obj.get("memories", []) or []
    except Exception:
        new = []
    merged = _dedup(load_memory(email) + new)
    save_memory(email, merged)
    return merged


def clear_memory(email):
    save_memory(email, [])


def format_memory_prompt(items):
    """Return a system-prompt block, or '' if empty."""
    items = [m for m in (items or []) if m]
    if not items:
        return ""
    return "\n\n## Remembered about this user (from past conversations)\n" + "\n".join(f"- {m}" for m in items)
