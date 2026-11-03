"""Compare two chunk strategies with REAL ragas 0.4.3 over the 陶行知 15-question testset.

Config A: chunk_size=256, overlap=50, k=5
Config B: chunk_size=1024, overlap=100, k=5

For each config: chunk the 陶行知 pptx, run the real DocChat pipeline (retrieval+DeepSeek)
for all 15 questions, then score with ragas (faithfulness, answer_relevancy,
context_precision, context_recall) using DeepSeek as judge + local multilingual embeddings.
Prints a comparison table and recommends the best config by overall mean.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, traceback, json
from dotenv import load_dotenv
load_dotenv("/Users/ChuanZhou/rag-project/rag-project/.env")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
import eval_ragas

DOC_PATH = "/Users/ChuanZhou/Desktop/弟弟的东西/陶行知电子小报.pptx"
CONFIGS = [
    {"name": "chunk256/overlap50/k5", "chunk_size": 256, "chunk_overlap": 50, "k": 5},
    {"name": "chunk1024/overlap100/k5", "chunk_size": 1024, "chunk_overlap": 100, "k": 5},
]
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

print("loading doc...", flush=True)
docs = eval_ragas.load_pptx(DOC_PATH)
print(f"  {len(docs)} slides", flush=True)

judge = LangchainLLMWrapper(ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("LLM_MODEL"), temperature=0))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2"))
rc = RunConfig(max_workers=2, timeout=180, max_retries=6, max_wait=90)

results = {}
for cfg in CONFIGS:
    print(f"\n##### CONFIG: {cfg['name']} #####", flush=True)
    sp = RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    chunks = [d.page_content for d in sp.split_documents(docs)]
    print(f"  {len(chunks)} chunks", flush=True)
    samples = []
    for i, item in enumerate(eval_ragas.qa_testset, 1):
        q = item["question"]
        try:
            answer, retrieved = eval_ragas.run_docchat_pipeline(q, chunks, cfg["k"])
        except Exception as e:
            print(f"  [{i}/{len(eval_ragas.qa_testset)}] pipeline error: {e}", flush=True)
            answer, retrieved = "", []
        print(f"  [{i}/{len(eval_ragas.qa_testset)}] {q} -> {answer[:60]}", flush=True)
        samples.append(SingleTurnSample(user_input=q, response=answer, retrieved_contexts=retrieved, reference=item["ground_truth"]))
    ds = EvaluationDataset(samples)
    print(f"  running ragas ({len(samples)} samples, 4 metrics)...", flush=True)
    t0 = time.time()
    means = {n: float("nan") for n in METRIC_NAMES}
    try:
        res = evaluate(ds, metrics=METRICS, llm=judge, embeddings=emb, run_config=rc)
        df = res.to_pandas()
        means = {n: float(df[n].mean()) for n in METRIC_NAMES}
    except Exception as e:
        print(f"  ragas error: {e}", flush=True); traceback.print_exc()
    means["overall"] = sum(means.values()) / len(means)
    results[cfg["name"]] = means
    print(f"  means: { {k: round(v,3) for k,v in means.items()} }  ({time.time()-t0:.0f}s)", flush=True)
    with open("/tmp/chunk_strategy_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print("\n\n========== COMPARISON ==========", flush=True)
print(f"{'config':28s} {'faith':>7} {'rel':>7} {'c_prec':>7} {'c_rec':>7} {'overall':>8}", flush=True)
for name, m in results.items():
    print(f"{name:28s} {m['faithfulness']:7.3f} {m['answer_relevancy']:7.3f} {m['context_precision']:7.3f} {m['context_recall']:7.3f} {m['overall']:8.3f}", flush=True)
best = max(results, key=lambda n: results[n]["overall"])
print(f"\nBEST chunk strategy: {best}  (overall {results[best]['overall']:.3f})", flush=True)
