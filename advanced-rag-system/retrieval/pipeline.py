"""
pipeline.py — full retrieval pipeline: rewrite -> multi-query -> hybrid
search (per variant) -> merge -> rerank. This is what task 5 (citation
generation) will call to get final context chunks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_rewrite import rewrite_query
from retrieval.multi_query import generate_query_variants
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank
from retrieval.bootstrap import get_index


def retrieve(raw_query: str, top_k: int = 5, candidate_pool: int = 10) -> list[tuple[str, dict, float]]:
    _, text_by_id = get_index()

    rewritten = rewrite_query(raw_query)
    variants = generate_query_variants(raw_query, rewritten, n=3)

    seen = {}
    for variant in variants:
        results = hybrid_search(variant, k=candidate_pool)
        for cid, score in results:
            if cid not in seen:
                seen[cid] = score

    top_candidates_sorted = sorted(seen.items(), key=lambda x: x[1], reverse=True)

    if top_candidates_sorted:
        top_score = top_candidates_sorted[0][1]
        MIN_RELATIVE_SCORE = 0.5  # keep only candidates scoring at least 50% of the top RRF score
        top_candidates_sorted = [
            (cid, score) for cid, score in top_candidates_sorted
            if score >= top_score * MIN_RELATIVE_SCORE
        ]

    MAX_RERANK_CANDIDATES = 40
    top_candidates = top_candidates_sorted[:MAX_RERANK_CANDIDATES]

    candidates = [(cid, text_by_id[cid]) for cid, _score in top_candidates]
    final = rerank(raw_query, candidates, top_k=top_k)
    return final