#!/usr/bin/env python
"""RAGAS Evaluation for DocChat."""
import os, warnings, httpx, json
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv("/Users/ChuanZhou/rag-project/rag-project/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 4
MAX_CONTEXT_CHARS = 12000

from pptx import Presentation
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pptx(path):
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

def search_chunks(query, chunks, k=4):
    """Document retrieval (BM25 + keyword hybrid, fused with RRF)."""
    import retrieval
    return retrieval.hybrid_search(query, chunks, k)

def call_llm(prompt):
    client = httpx.Client(base_url=OPENAI_BASE_URL, headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, timeout=httpx.Timeout(120.0, connect=30.0))
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    resp = client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    client.close()
    return resp.json()["choices"][0]["message"]["content"].strip()


qa_testset = [
    {"question": "陶行知的教育思想被称为什么理论？", "ground_truth": "生活教育论。"},
    {"question": "陶行知的生卒年份是哪一年到哪一年？", "ground_truth": "1891年到1946年。"},
    {"question": "陶行知是哪里人？", "ground_truth": "安徽省歙县人。"},
    {"question": "陶行知曾在哪所大学攻读教育学博士学位？", "ground_truth": "美国哥伦比亚大学。"},
    {"question": "陶行知被谁誉为'万世师表'？", "ground_truth": "被宋庆龄誉为'万世师表'。"},
    {"question": "陶行知生活教育论的三大原理是什么？", "ground_truth": "生活即教育，社会即学校，教学做合一。"},
    {"question": "陶行知提倡的四种精神是什么？", "ground_truth": "大爱，奉献，创造，求真。"},
    {"question": "陶行知的五大主张分别是什么？", "ground_truth": "行是知之始、在劳力上劳心、以教人者教己、即知即传、六大解放。"},
    {"question": "陶行知提出的'六大解放'具体解放的是哪六个方面？", "ground_truth": "解放儿童的头脑、双手、嘴、眼睛、时间和空间。"},
    {"question": "陶行知归国后曾担任过哪些职务？", "ground_truth": "曾任南京高等师范学校、国立东南大学教授、教务主任等职。"},
    {"question": "陶行知先后创办了哪些工学团？", "ground_truth": "先后创办'山海工学团'、'报童工学团'、'晨更工学团'、'流浪儿工学团'。"},
    {"question": "陶行知关于'重师'的名言是怎么说的？", "ground_truth": "农不重师，则农必破产；工不重师，则工必粗陋；国民不重师，则国必不能富强；人类不重师，则世界不得太平。"},
    {"question": "陶行知关于'活的书'和'死的书'的名言是什么？", "ground_truth": "我们要活的书，不要死的书；要真的书，不要假的书；要动的书，不要静的书；要用的书，不要读的书。总起来说，我们要以生活为中心的教学做指导，不要以文字为中心的教科书。"},
    {"question": "陶行知获得过诺贝尔奖吗？", "ground_truth": "文档中没有提到陶行知获得诺贝尔奖的信息，不应回答获得过。"},
    {"question": "陶行知在美国期间创办了哪些学校？", "ground_truth": "文档中只提到他在美国哥伦比亚大学攻读博士学位，并未提到他在美国创办学校，工学团都是归国后在国内创办的。"},
]

def run_docchat_pipeline(question, doc_chunks, k=DEFAULT_K):
    rc = search_chunks(question, doc_chunks, k)
    ctx = "\n\n".join(rc[:MAX_CONTEXT_CHARS // max(DEFAULT_CHUNK_SIZE, 100) + 1])
    if len(ctx) > MAX_CONTEXT_CHARS: ctx = ctx[:MAX_CONTEXT_CHARS] + "...[truncated]"
    pr = f"You are a helpful assistant. Answer the question based on the document content below.\n\nDocument content:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    r = call_llm(pr)
    answer = r
    return answer, rc

if __name__ == "__main__":
    DOC_PATH = "/Users/ChuanZhou/Desktop/弟弟的东西/陶行知电子小报.pptx"
    print(f"Loading: {DOC_PATH}")
    docs = load_pptx(DOC_PATH)
    print(f"  {len(docs)} slides loaded")
    sp = RecursiveCharacterTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    chunks = sp.split_documents(docs)
    doc_chunks = [d.page_content for d in chunks]
    print(f"  {len(doc_chunks)} chunks\n")

    from datasets import Dataset
    questions, answers, contexts_list, ground_truths = [], [], [], []
    print("Running pipeline...\n")
    for i, item in enumerate(qa_testset, 1):
        q = item["question"]
        print(f"  [{i}/{len(qa_testset)}] {q}")
        answer, retrieved = run_docchat_pipeline(q, doc_chunks)
        print(f"    -> Answer: {answer[:120]}...")
        for j, ch in enumerate(retrieved, 1):
            print(f"       [Chunk {j}] {ch[:100]}...")
        print()
        questions.append(q); answers.append(answer)
        contexts_list.append(retrieved); ground_truths.append(item["ground_truth"])

    # Save full results (answer + retrieved content) to file
    with open("/tmp/docchat_results.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(qa_testset, 1):
            f.write(f"{'='*60}\n")
            f.write(f"Q{i}: {item['question']}\n")
            f.write(f"Ground Truth: {item['ground_truth']}\n")
            f.write(f"Answer: {answers[i-1]}\n")
            f.write(f"Retrieved Chunks ({len(contexts_list[i-1])}):\n")
            for j, ch in enumerate(contexts_list[i-1], 1):
                f.write(f"  [Chunk {j}] {ch}\n")
            f.write("\n")
    print("Full results saved to /tmp/docchat_results.txt\n")

    dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts_list, "ground_truth": ground_truths})

    print("\nRunning RAGAS...\n")
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper
    chat_llm = ChatOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=LLM_MODEL, temperature=0)
    # bypass_n=True forces n=1 requests (avoids "n must be 1" API error)
    evaluator_llm = LangchainLLMWrapper(chat_llm, bypass_n=True, bypass_temperature=True)
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=evaluator_llm)

    print("\n" + "=" * 50)
    print("         RAGAS SCORES")
    print("=" * 50)
    # RAGAS 0.4.x: result is EvaluationResult, use to_dataframe() or scores
    try:
        df = result.to_pandas()
        for col in [c for c in df.columns if c not in ["question", "answer", "contexts", "ground_truth"]]:
            mean_score = df[col].mean()
            print(f"  {col:25s}: {mean_score:.4f}")
    except Exception:
        # Fallback: try direct dict access
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if hasattr(result, metric):
                print(f"  {metric:25s}: {getattr(result, metric):.4f}")
            elif hasattr(result, "scores"):
                scores = [s.get(metric, 0) for s in result.scores if s.get(metric) is not None]
                if scores: print(f"  {metric:25s}: {sum(scores)/len(scores):.4f}")
    print("=" * 50)
    with open("/tmp/ragas_results.txt", "w") as f:
        try:
            df = result.to_pandas()
            for col in [c for c in df.columns if c not in ["question", "answer", "contexts", "ground_truth"]]:
                f.write(f"{col}: {df[col].mean()}\n")
        except Exception as e:
            f.write(f"Error: {e}\n")
    print("\nSaved to /tmp/ragas_results.txt")
