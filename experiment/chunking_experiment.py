"""
chunking_experiment.py — compare chunking strategies over the 陶行知 15-question testset.

Configs: {256,50,5}, {512,100,5}, {1024,100,5}.
For each: split_with_config (Chinese-aware separators) -> run real pipeline (retrieval+LLM)
-> score with simple_eval (local embeddings, fast/deterministic) -> comparison table + best.

Reuses: eval_ragas.load_pptx / qa_testset / run_docchat_pipeline / search_chunks,
        simple_eval.evaluate_one (multilingual cosine-similarity metrics).
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
import eval_ragas
from simple_eval import evaluate_one

DOC_PATH = "/Users/ChuanZhou/Desktop/弟弟的东西/陶行知电子小报.pptx"
CONFIGS_TO_TEST = [
    {"chunk_size": 256, "chunk_overlap": 50, "k": 5},
    {"chunk_size": 512, "chunk_overlap": 100, "k": 5},
    {"chunk_size": 1024, "chunk_overlap": 100, "k": 5},
]
METRIC_NAMES = ["answer_relevancy", "faithfulness", "correctness", "context_recall"]


def split_with_config(documents, chunk_size, chunk_overlap):
    """RecursiveCharacterTextSplitter with Chinese-aware separators (。 ， etc.)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def inspect_retrieval(question, ground_truth, chunks, k=5):
    """Manual spot-check: print the chunks retrieved for a question."""
    retrieved = eval_ragas.search_chunks(question, chunks, k)
    print(f"问题: {question}")
    print(f"标准答案: {ground_truth}")
    print("检索到的内容:")
    for i, c in enumerate(retrieved, 1):
        print(f"  [{i}] {c[:100]}...")
    print("---")
    return retrieved


def run_chunking_experiment(documents, qa_testset, config):
    chunks = split_with_config(documents, config["chunk_size"], config["chunk_overlap"])
    doc_chunks = [d.page_content for d in chunks]
    label = f"{config['chunk_size']}/{config['chunk_overlap']}/k{config['k']}"
    print(f"\n##### config {label}: {len(doc_chunks)} chunks #####", flush=True)
    qa_results = []
    for i, item in enumerate(qa_testset, 1):
        q = item["question"]
        answer, retrieved = eval_ragas.run_docchat_pipeline(q, doc_chunks, config["k"])
        print(f"  [{i}/{len(qa_testset)}] {q} -> {answer[:50]}", flush=True)
        qa_results.append({"question": q, "generated_answer": answer, "ground_truth": item["ground_truth"], "retrieved_contexts": retrieved})
    rows = [evaluate_one(r["question"], r["generated_answer"], r["ground_truth"], r["retrieved_contexts"]) for r in qa_results]
    means = {m: sum(r[m] for r in rows) / len(rows) for m in METRIC_NAMES}
    means["overall"] = sum(means.values()) / len(means)
    return {"label": label, "config": config, "n_chunks": len(doc_chunks), "means": means, "rows": rows}


def main():
    print("loading doc...", flush=True)
    docs = eval_ragas.load_pptx(DOC_PATH)
    print(f"  {len(docs)} slides", flush=True)
    results = []
    for cfg in CONFIGS_TO_TEST:
        results.append(run_chunking_experiment(docs, eval_ragas.qa_testset, cfg))
    print("\n\n========== COMPARISON ==========", flush=True)
    hdr = f"{'config':20s} {'n_chunks':>8} {'faith':>7} {'rel':>7} {'corr':>7} {'c_rec':>7} {'overall':>8}"
    print(hdr, flush=True)
    for r in results:
        m = r["means"]
        print(f"{r['label']:20s} {r['n_chunks']:>8} {m['faithfulness']:7.3f} {m['answer_relevancy']:7.3f} {m['correctness']:7.3f} {m['context_recall']:7.3f} {m['overall']:8.3f}", flush=True)
    best = max(results, key=lambda r: r["means"]["overall"])
    print(f"\nBEST: {best['label']} (overall {best['means']['overall']:.3f})", flush=True)
    print("\n=== inspect_retrieval (best config, Q1) ===", flush=True)
    best_chunks = [d.page_content for d in split_with_config(docs, best["config"]["chunk_size"], best["config"]["chunk_overlap"])]
    inspect_retrieval(eval_ragas.qa_testset[0]["question"], eval_ragas.qa_testset[0]["ground_truth"], best_chunks, best["config"]["k"])


if __name__ == "__main__":
    main()
