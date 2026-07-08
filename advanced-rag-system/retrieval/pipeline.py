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


def retrieve(raw_query: str, top_k: int = 5, candidate_pool: int = 30) -> list[tuple[str, dict, float]]:
    _, text_by_id = get_index()

    rewritten = rewrite_query(raw_query)
    variants = generate_query_variants(raw_query, rewritten, n=3)

    # run hybrid search per variant, merge all candidates (dedup by chunk_id,
    # keep first-seen since variants are already ordered rewritten-first)
    seen = {}
    for variant in variants:
        results = hybrid_search(variant, k=candidate_pool)
        for cid, score in results:
            if cid not in seen:
                seen[cid] = score

    candidates = [(cid, text_by_id[cid]) for cid in seen.keys()]
    final = rerank(raw_query, candidates, top_k=top_k)  # rerank against ORIGINAL query intent
    return final