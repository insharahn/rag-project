"""
hybrid_search.py — orchestrates BM25 + vector search + RRF fusion into one callable
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import bm25_index, vector_search
from retrieval.fusion import reciprocal_rank_fusion


def hybrid_search(query: str, k: int = 10, candidate_pool: int = 30) -> list[tuple[str, float]]:
    """Runs BM25 and vector search independently, fuses via RRF, returns
    top-k fused results. candidate_pool controls how deep each retriever
    searches before fusion (deeper pool = fusion has more to work with)."""
    bm25_results = bm25_index.search(query, k=candidate_pool)
    vector_results = vector_search.search(query, k=candidate_pool)

    fused = reciprocal_rank_fusion([bm25_results, vector_results])
    return fused[:k]