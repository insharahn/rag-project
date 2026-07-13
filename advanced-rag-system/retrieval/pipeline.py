"""
pipeline.py — full retrieval pipeline: rewrite -> multi-query -> hybrid
search (per variant) -> merge -> rerank.
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

    seen: dict[str, float] = {}
    for variant in variants:
        for cid, score in hybrid_search(variant, k=candidate_pool):
            if cid not in seen or score > seen[cid]:
                seen[cid] = score
                
    """ # --- graph search on the raw query ---
    # Graph hits are added to the same pool before filtering.
    # Scores are scaled to the same order of magnitude as RRF scores so
    # the relative-score filter treats them fairly.
    if _GRAPH_AVAILABLE and seen:
        top_hybrid_score = max(seen.values())
        for cid, gscore in graph_search(raw_query, top_k=candidate_pool):
            # A graph score of 1.0 (perfect entity match) maps to the top hybrid score.
            # This keeps high-entity-match chunks competitive without dominating.
            scaled = gscore * top_hybrid_score
            if cid not in seen or scaled > seen[cid]:
                seen[cid] = scaled
    """

    top_candidates_sorted = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    if top_candidates_sorted:
        threshold = top_candidates_sorted[0][1] * 0.5
        top_candidates_sorted = [
            (cid, score) for cid, score in top_candidates_sorted
            if score >= threshold
        ]

    MAX_RERANK_CANDIDATES = 40
    top_candidates = top_candidates_sorted[:MAX_RERANK_CANDIDATES]

    candidates = [(cid, text_by_id[cid]) for cid, _ in top_candidates if cid in text_by_id]
    return rerank(raw_query, candidates, top_k=top_k)