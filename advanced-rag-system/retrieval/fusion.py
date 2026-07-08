"""
fusion.py — task 3 (final step): Reciprocal Rank Fusion to merge BM25 and
vector search result lists into a single ranked list. RRF works on rank
position, not raw scores, which sidesteps the fact that BM25 scores (0-30+)
and cosine similarity scores (0-1) aren't comparable on the same scale.
"""

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """ranked_lists: list of [(chunk_id, score), ...] lists, each already
    sorted best-first (as returned by bm25_index.search / vector_search.search).
    Returns a single fused [(chunk_id, rrf_score), ...] list, sorted best-first.

    k=60 is the standard RRF constant from the original paper — dampens the
    impact of any single list's exact rank position, works well across
    retriever types without tuning."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused