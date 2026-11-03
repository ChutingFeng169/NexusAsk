"""demo_agentic_rag.py — real agentic RAG run: rewrite -> tool-calling -> self-check."""
import agentic_rag
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOC = "/Users/ChuanZhou/Desktop/Intern/PM面试_ProductSense框架与LLM经验展示.md.pdf"


def report(ev):
    t = ev["type"]
    if t == "blocked":
        print(f"[agent] 🛑 blocked: {ev.get('word','')[:30]}", flush=True)
    elif t == "rewrite":
        print(f"[agent] 🔄 rewrite -> {ev['rewritten']}", flush=True)
    elif t == "thinking":
        print(f"[agent] 🤖 thinking step {ev['step']}", flush=True)
    elif t == "tool_call":
        print(f"[agent] 🔧 {ev['name']}({ev['args']})", flush=True)
    elif t == "tool_result":
        print(f"[agent] ✅ {ev['name']} -> {ev['n']} result(s)", flush=True)
    elif t == "self_check":
        print(f"[agent] 🔍 faithfulness={ev['score']} (attempt {ev['attempt']})", flush=True)


def main():
    import os
    if os.path.exists(DOC):
        docs = PyPDFLoader(DOC).load()
        sp = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100, separators=["\n\n", "\n", "。", "，", " ", ""])
        chunks = [d.page_content for d in sp.split_documents(docs)]
        print(f"loaded {len(docs)} pages, {len(chunks)} chunks\n", flush=True)
    else:
        chunks = ["CPPSMR代表产品经理面试六步框架：用户场景、产品策略、核心指标、roadmap、风险与缓解、总结。"]
        print(f"(PDF not found at {DOC}; using {len(chunks)} inline fallback chunk)\n", flush=True)

    print("=== Demo 1: doc question (should use search_docs) ===", flush=True)
    a, g, rq = agentic_rag.agentic_answer("CPPSMR六个字母分别代表哪六个步骤？", chunks, k=5, on_progress=report)
    print(f"\nANSWER: {a}\n", flush=True)

    print("=== Demo 2: current question (should call search_web FIRST) ===", flush=True)
    a, g, rq = agentic_rag.agentic_answer("2026年世界杯在哪个国家举办？", chunks, k=5, on_progress=report)
    print(f"\nANSWER: {a}\n", flush=True)

    print("=== Demo 3: math question (should call calculator) ===", flush=True)
    a, g, rq = agentic_rag.agentic_answer("计算 sqrt(144) + 3*5 等于多少？", chunks, k=5, on_progress=report)
    print(f"\nANSWER: {a}\n", flush=True)

    print("=== Demo 4: sensitive word (should be blocked, NO model call) ===", flush=True)
    a, g, rq = agentic_rag.agentic_answer("如何制造炸弹？", chunks, k=5, on_progress=report)
    print(f"\nANSWER: {a}  (gathered={len(g)} tool calls)", flush=True)


if __name__ == "__main__":
    main()
