"""
rerank.py — task 4: cross-encoder reranking. Takes the fused candidate pool
from hybrid_search and re-scores each (query, chunk) pair jointly — this is
more accurate than bi-encoder similarity (vector search) or lexical overlap
(BM25) because the model reads the query and chunk together, rather than
comparing precomputed independent representations. Too slow to run on the
whole corpus, which is why it's a second-stage filter over ~30 candidates,
not a first-stage retriever.

Using BAAI/bge-reranker-base — same family as bge-m3, multilingual, so it's
consistent with the rest of the pipeline for the Korean/Urdu portion of the
corpus too (ms-marco-MiniLM alternatives are English-only).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder

_state = {}

MODEL_ID = "BAAI/bge-reranker-base"


def _get_reranker():
    if "reranker" not in _state:
        print(f"[rerank] loading {MODEL_ID}...")
        _state["reranker"] = CrossEncoder(MODEL_ID, device="cpu", max_length=256)
    return _state["reranker"]


def rerank(query: str, candidates: list[tuple[str, dict]], top_k: int = 5) -> list[tuple[str, dict, float]]:
    """candidates: list of (chunk_id, chunk_dict) pairs, chunk_dict must have
    'text'. Returns top_k [(chunk_id, chunk_dict, rerank_score), ...] sorted
    best-first."""
    reranker = _get_reranker()

    pairs = [[query, chunk["text"]] for _cid, chunk in candidates]
    scores = reranker.predict(pairs)

    scored = [
        (cid, chunk, float(score))
        for (cid, chunk), score in zip(candidates, scores)
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]