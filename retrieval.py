"""
retrieval.py — BM25 + keyword hybrid search with Reciprocal Rank Fusion (RRF).

Two retrievers:
  1. keyword_search  — set-intersection overlap (the original keyword search)
  2. bm25_search     — BM25 (TF-IDF + document-length normalization, k1=1.5, b=0.75)

Fused with Reciprocal Rank Fusion (RRF, k=60) to combine the best of both
without needing tuned weights or an embedding model.
"""
import math
from collections import Counter


def _tokenize(text):
    """Whitespace tokenization (lowercased) — consistent with the original keyword search."""
    return text.lower().split()


def keyword_search(query, chunks, k=5):
    """Keyword-overlap retrieval (set-intersection count). Returns [(score, index, chunk), ...]."""
    qw = set(_tokenize(query))
    scored = []
    for i, c in enumerate(chunks):
        s = len(qw & set(_tokenize(c)))
        if s > 0:
            scored.append((s, i, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def bm25_search(query, chunks, k=5, k1=1.5, b=0.75):
    """BM25 retrieval. Returns [(score, index, chunk), ...].

    score = Σ_t IDF(t) * (f*(k1+1)) / (f + k1*(1 - b + b*|d|/avgdl))
    IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)   # +1 avoids negative IDF
    """
    if not chunks:
        return []
    q_tokens = _tokenize(query)
    N = len(chunks)
    doc_tokens = [_tokenize(c) for c in chunks]
    doc_lens = [len(dt) for dt in doc_tokens]
    avgdl = sum(doc_lens) / N if N else 0.0

    # document frequency + IDF for each unique query term
    unique_q = set(q_tokens)
    df = {}
    for t in unique_q:
        df[t] = sum(1 for dt in doc_tokens if t in set(dt))
    idf = {}
    for t in unique_q:
        idf[t] = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)

    scored = []
    for i, dt in enumerate(doc_tokens):
        tf = Counter(dt)
        score = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if f > 0:
                dl_norm = doc_lens[i] / avgdl if avgdl else 0.0
                score += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl_norm))
        if score > 0:
            scored.append((score, i, chunks[i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def rrf_fuse(rankings, k=5, rrf_k=60):
    """Reciprocal Rank Fusion. rankings = list of [(score, index, chunk), ...].

    score(idx) = Σ 1/(rrf_k + rank_i) across all retrievers (1-based rank).
    """
    scores = {}
    chunk_map = {}
    for ranking in rankings:
        for rank, (_s, idx, c) in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map[idx] = c
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[idx] for idx, _ in fused[:k]]


def hybrid_search(query, chunks, k=5):
    """BM25 + keyword overlap, fused with RRF. Returns [chunk, ...] top-k."""
    return rrf_fuse([bm25_search(query, chunks, k), keyword_search(query, chunks, k)], k=k)
