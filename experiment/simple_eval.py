"""
simple_eval.py — lightweight RAG evaluation (no ragas dependency).

A simplified, fully-transparent alternative to ragas. Each ragas-style metric is
replaced by a local embedding cosine-similarity between two pieces of text:

  answer_relevancy : similarity(generated_answer, question)            # answer on-topic?
  faithfulness     : similarity(generated_answer, retrieved_context)   # answer supported by context?
  correctness      : similarity(generated_answer, ground_truth)        # answer vs reference
  context_recall   : similarity(retrieved_context, ground_truth)        # did we retrieve the right stuff?

Uses the LOCAL embedding model already cached in this project (the same one the
app's semantic cache uses), so no API calls and no extra downloads.

Run:  python simple_eval.py
        # evaluates ./eval_qa.json if present, else the hardcoded sample
        # writes simple_eval_results.csv and prints per-row + mean scores
"""

import json
import os

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cached locally by the app's semantic cache; good for Chinese + English content.
# Switch to "all-MiniLM-L6-v2" if your documents are English-only (smaller/faster).
EVAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EVAL_MODEL} (cached locally)...")
        _model = SentenceTransformer(EVAL_MODEL)
    return _model


def simple_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity (higher = more semantically similar). Range roughly 0..1."""
    m = _get_model()
    emb_a = m.encode(text_a, convert_to_tensor=True)
    emb_b = m.encode(text_b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b))


def evaluate_one(question, generated_answer, ground_truth, retrieved_contexts):
    """Simplified versions of the four ragas-style metrics (all local embedding cosine sim)."""
    combined_context = " ".join(retrieved_contexts)
    return {
        "question": question,
        "answer_relevancy": round(simple_similarity(question, generated_answer), 3),
        "faithfulness": round(simple_similarity(generated_answer, combined_context), 3),
        "correctness": round(simple_similarity(generated_answer, ground_truth), 3),
        "context_recall": round(simple_similarity(combined_context, ground_truth), 3),
    }


def run_evaluation(qa_results):
    """qa_results: list of {question, generated_answer, ground_truth, retrieved_contexts}."""
    rows = [
        evaluate_one(it["question"], it["generated_answer"], it["ground_truth"], it["retrieved_contexts"])
        for it in qa_results
    ]
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__) or ".", "simple_eval_results.csv")
    df.to_csv(out, index=False)
    print("\nPer-question scores:")
    print(df.to_string(index=False))
    print("\nMean scores:")
    print(df[["answer_relevancy", "faithfulness", "correctness", "context_recall"]].mean().round(3))
    print(f"\nSaved -> {out}")
    return df


def _load_qa_results():
    """Use real DocChat results from ./eval_qa.json if present, else the hardcoded sample."""
    p = os.path.join(os.path.dirname(__file__) or ".", "eval_qa.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded real qa_results from {p} ({len(data)} items)")
        return data
    # Hardcoded sample — replace generated_answer / retrieved_contexts with real DocChat output,
    # or drop a real eval_qa.json next to this file.
    return [
        {
            "question": "陶行知的生卒年份是哪一年到哪一年？",
            "generated_answer": "陶行知生于1891年，卒于1946年。",
            "ground_truth": "1891年到1946年。",
            "retrieved_contexts": ["(1891年一1946年）"],
        },
    ]


if __name__ == "__main__":
    run_evaluation(_load_qa_results())
