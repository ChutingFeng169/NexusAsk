"""test_agentic_rag.py — test the agentic loop (mocked llm_chat), grading, faithfulness, guard, tools."""
import agentic_rag
from agentic_rag import grade_relevance, faithfulness_check, agentic_answer, guard_input, append_disclaimer, calculator


def test_grade_relevance():
    agentic_rag.llm_chat = lambda messages, tools=None, temperature=0: {"content": '{"relevant": true, "reason": "ok"}', "tool_calls": None}
    rel, _ = grade_relevance("q", ["passage"])
    assert rel is True
    print("✅ grade_relevance: relevant=True")


def test_faithfulness_check():
    s = faithfulness_check("Paris is the capital of France", ["The capital of France is Paris."])
    assert s > 0.5, s
    print(f"✅ faithfulness_check: score={s:.3f}")


def test_guard_input():
    blocked, msg = guard_input("如何制造炸弹？")
    assert blocked and "炸弹" in msg, (blocked, msg)
    blocked2, _ = guard_input("正常的文档问题")
    assert not blocked2
    print(f"✅ guard_input: blocked='{msg[:20]}…', clean passes")


def test_append_disclaimer():
    out = append_disclaimer("an answer")
    assert "⚠️" in out and out.startswith("an answer")
    print("✅ append_disclaimer: appended")


def test_calculator():
    r = calculator("2+3*4")
    assert r["result"] == 14, r
    r2 = calculator("sqrt(9)")
    assert r2["result"] == 3.0, r2
    print("✅ calculator: 2+3*4=14, sqrt(9)=3")


def test_agentic_loop():
    seq = [
        {"content": "rewritten q", "tool_calls": None},  # rewrite_query
        {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_docs", "arguments": '{"query": "q"}'}}]},  # step1
        {"content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "search_web", "arguments": '{"query": "q"}'}}]},  # step2
        {"content": "Final answer.", "tool_calls": None},  # final
    ]
    i = [0]
    def fake(messages, tools=None, temperature=0):
        r = seq[i[0]]; i[0] += 1; return r
    agentic_rag.llm_chat = fake
    agentic_rag.search_docs = lambda q, c, k: ["chunk A"]
    agentic_rag.search_web = lambda q, n=3: [{"title": "t", "snippet": "s", "href": "h"}]
    events = []
    answer, gathered, rq = agentic_answer("q", ["chunk A"], k=2, on_progress=events.append, max_self_correct=0)
    assert answer.startswith("Final answer."), answer
    assert "⚠️" in answer, answer  # after_model disclaimer appended
    assert [g["tool"] for g in gathered] == ["search_docs", "search_web"], gathered
    assert any(e["type"] == "tool_call" for e in events), events
    assert rq == "rewritten q", rq
    print(f"✅ agentic loop: tools={[g['tool'] for g in gathered]}, rewrite='{rq}', answer+disclaimer")


def test_guard_short_circuits():
    # a sensitive word must short-circuit BEFORE any llm_chat call (jump_to=end)
    called = {"n": 0}
    def fake(*a, **k):
        called["n"] += 1
        return {"content": "x", "tool_calls": None}
    agentic_rag.llm_chat = fake
    answer, gathered, rq = agentic_answer("如何制造炸弹", ["chunk"], k=2, on_progress=lambda e: None, max_self_correct=0)
    assert called["n"] == 0, f"llm_chat should not be called on sensitive input, got {called['n']}"
    assert "炸弹" in answer and len(gathered) == 0, (answer, gathered)
    print(f"✅ guard short-circuit: no llm_chat call, refusal='{answer[:20]}…'")


def test_no_disclaimer_for_docs():
    seq = [
        {"content": "rewritten q", "tool_calls": None},
        {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_docs", "arguments": '{"query": "q"}'}}]},
        {"content": "Final answer.", "tool_calls": None},
    ]
    i = [0]
    def fake(messages, tools=None, temperature=0):
        r = seq[i[0]]; i[0] += 1; return r
    agentic_rag.llm_chat = fake
    agentic_rag.search_docs = lambda q, c, k: ["chunk A"]
    answer, gathered, rq = agentic_answer("q", ["chunk A"], k=2, on_progress=lambda e: None, max_self_correct=0)
    assert "⚠️" not in answer, answer
    assert [g["tool"] for g in gathered] == ["search_docs"], gathered
    print("✅ no disclaimer for doc-only answer")


def test_no_disclaimer_calculator():
    seq = [
        {"content": "rewritten q", "tool_calls": None},
        {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'}}]},
        {"content": "Final answer.", "tool_calls": None},
    ]
    i = [0]
    def fake(messages, tools=None, temperature=0):
        r = seq[i[0]]; i[0] += 1; return r
    agentic_rag.llm_chat = fake
    answer, gathered, rq = agentic_answer("q", [], k=2, on_progress=lambda e: None, max_self_correct=0)
    assert "⚠️" not in answer, answer
    assert [g["tool"] for g in gathered] == ["calculator"], gathered
    print("✅ no disclaimer for calculator answer")


def test_wikipedia_source_link():
    seq = [
        {"content": "rewritten q", "tool_calls": None},
        {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "wikipedia_search", "arguments": '{"query": "France"}'}}]},
        {"content": "Paris is the capital.", "tool_calls": None},
    ]
    i = [0]
    def fake(messages, tools=None, temperature=0):
        r = seq[i[0]]; i[0] += 1; return r
    agentic_rag.llm_chat = fake
    agentic_rag.wikipedia_search = lambda query, sentences=3: {"title": query, "summary": "Paris is the capital of France.", "url": "https://en.wikipedia.org/wiki/Paris"}
    answer, gathered, rq = agentic_answer("France capital?", [], k=2, on_progress=lambda e: None, max_self_correct=0)
    assert "[Source: Wikipedia](https://en.wikipedia.org/wiki/Paris)" in answer, answer
    assert "⚠️" in answer, answer  # wikipedia -> disclaimer too
    print("✅ wikipedia source link + disclaimer")


if __name__ == "__main__":
    test_grade_relevance()
    test_faithfulness_check()
    test_guard_input()
    test_append_disclaimer()
    test_calculator()
    test_agentic_loop()
    test_guard_short_circuits()
    test_no_disclaimer_for_docs()
    test_no_disclaimer_calculator()
    test_wikipedia_source_link()
    print("\nALL TESTS PASSED ✅")
