"""
chunking_experiment_pm.py — chunking experiment over the (longer) PM Product-Sense PDF,
15-question CPPSMR testset, 3 configs (256/50, 512/100, 1024/100, k=5), scored with simple_eval.
Reuses split_with_config / run_chunking_experiment / inspect_retrieval from chunking_experiment.
"""
from langchain_community.document_loaders import PyPDFLoader
from chunking_experiment import run_chunking_experiment, split_with_config, inspect_retrieval

DOC_PATH = "/Users/ChuanZhou/Desktop/Intern/PM面试_ProductSense框架与LLM经验展示.md.pdf"
CONFIGS = [
    {"chunk_size": 256, "chunk_overlap": 50, "k": 5},
    {"chunk_size": 512, "chunk_overlap": 100, "k": 5},
    {"chunk_size": 1024, "chunk_overlap": 100, "k": 5},
]

qa_testset = [
    {"question": "PM面试万能框架的六步骤缩写叫什么？", "ground_truth": "CPPSMR。"},
    {"question": "CPPSMR六个字母分别代表哪六个步骤？", "ground_truth": "Clarify（明确边界）、Persona（定用户）、Problem（诊断痛点）、Prioritize（取舍）、Solution + Metric（方案+度量）、Risk（风险预判）。"},
    {"question": "Problem这一步建议花多长时间？", "ground_truth": "2分钟。"},
    {"question": "Clarify这一步建议花多长时间？", "ground_truth": "30秒。"},
    {"question": "Risk这一步建议花多长时间？", "ground_truth": "30秒。"},
    {"question": "文档里认为六步中最关键、最能拉开差距的是哪一步？", "ground_truth": "Problem（诊断痛点）这一步，文档里特别标注了星号强调其重要性。"},
    {"question": "Clarify这一步必须问的3类问题是什么？", "ground_truth": "目标用户是谁、目标平台/场景是什么、有没有特定的业务目标。"},
    {"question": "在Gemini Memory案例研究中，诊断出的根因是什么？", "ground_truth": "session-based vs project-based mental model mismatch（基于会话的模式与基于项目的心智模型不匹配）。"},
    {"question": "Gemini Memory案例中用来衡量成功的具体指标是什么？", "ground_truth": "context re-explanation rate（用户需要重新解释上下文的频率）。"},
    {"question": "如果面试中被问'有没有用过LLM'，文档建议怎么回答？", "ground_truth": "建议回答自己开发过DocChat，一个基于RAG的文档问答应用，使用Python、LangChain、Streamlit和Chroma作为向量数据库，用户可以上传文档并用自然语言提问，系统检索相关片段并交给LLM生成有依据的回答。"},
    {"question": "文档建议如何把'我没有正式LLM工作经验'这句话转换成更好的表达？", "ground_truth": "把它转换成强调动手实践经验的表达，比如'我有从零构建LLM应用的实际经验——包括数据摄取、向量搜索和提示词设计——通过我的DocChat项目'，核心是把'缺乏经验'的框架换成'已有具体、可验证的动手成果'的框架。"},
    {"question": "Persona这一步的话术模板是怎么说的？", "ground_truth": "\"I'll focus on [具体人群]，because [选择这个群体的理由].\""},
    {"question": "如果面试中卡壳了，文档建议的应急锚点话术之一是什么？", "ground_truth": "\"Let me think through the core problem here for a second...\" 或 \"I want to make sure I'm prioritizing the right user segment before jumping to solutions...\" 或 \"That's a good push — let me reconsider the trade-off here...\""},
    {"question": "文档里提到CPPSMR框架适合用在系统设计面试题上吗？", "ground_truth": "文档中没有提到CPPSMR适用于系统设计题，它明确列出的适用场景是产品设计类问题，比如'Design a product for...'、'How would you improve...'这类问题。"},
    {"question": "文档里有没有给出具体的CPPSMR框架发明者或者出处？", "ground_truth": "文档中没有提到这个框架的发明者或出处信息。"},
]


def main():
    print(f"loading PDF: {DOC_PATH}", flush=True)
    docs = PyPDFLoader(DOC_PATH).load()
    print(f"  {len(docs)} pages, {sum(len(d.page_content) for d in docs)} chars", flush=True)
    results = []
    for cfg in CONFIGS:
        results.append(run_chunking_experiment(docs, qa_testset, cfg))
    print("\n\n========== COMPARISON (PM PDF) ==========", flush=True)
    print(f"{'config':20s} {'n_chunks':>8} {'faith':>7} {'rel':>7} {'corr':>7} {'c_rec':>7} {'overall':>8}", flush=True)
    for r in results:
        m = r["means"]
        print(f"{r['label']:20s} {r['n_chunks']:>8} {m['faithfulness']:7.3f} {m['answer_relevancy']:7.3f} {m['correctness']:7.3f} {m['context_recall']:7.3f} {m['overall']:8.3f}", flush=True)
    best = max(results, key=lambda r: r["means"]["overall"])
    print(f"\nBEST: {best['label']} (overall {best['means']['overall']:.3f})", flush=True)
    print("\n=== inspect_retrieval (best config, Q1) ===", flush=True)
    best_chunks = [d.page_content for d in split_with_config(docs, best["config"]["chunk_size"], best["config"]["chunk_overlap"])]
    inspect_retrieval(qa_testset[0]["question"], qa_testset[0]["ground_truth"], best_chunks, best["config"]["k"])


if __name__ == "__main__":
    main()
