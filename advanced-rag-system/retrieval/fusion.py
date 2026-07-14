"""
fusion.py: Reciprocal Rank Fusion to merge BM25 and
vector search result lists into a single ranked list.
"""

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60, #standard rrf constant
) -> list[tuple[str, float]]:
    """ranked_lists: list of [(chunk_id, score), ...] lists, each already
    sorted 
    Returns a single fused [(chunk_id, rrf_score), ...] list, sorted best-first.
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused