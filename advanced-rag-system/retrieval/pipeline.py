"""
pipeline.py — full retrieval pipeline:
  rewrite -> multi-query -> hybrid search -> graph search -> merge -> rerank
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_rewrite   import rewrite_query
from retrieval.multi_query     import generate_query_variants
from retrieval.hybrid_search   import hybrid_search
from retrieval.rerank          import rerank
from retrieval.bootstrap       import get_index

try:
    from retrieval.graph_search import graph_search
    _GRAPH_AVAILABLE = True
except Exception:
    _GRAPH_AVAILABLE = False

def retrieve(raw_query: str, top_k: int = 5, candidate_pool: int = 10) -> list[tuple[str, dict, float]]:
    _, text_by_id = get_index()

    rewritten = rewrite_query(raw_query)
    variants  = generate_query_variants(raw_query, rewritten, n=3)

    # 1. hybrid search across all variants
    seen: dict[str, float] = {}
    for variant in variants:
        for cid, score in hybrid_search(variant, k=candidate_pool):
            if cid not in seen or score > seen[cid]:
                seen[cid] = score

    # 2. direct graph search on raw + rewritten + all variants
    if _GRAPH_AVAILABLE and seen:
        top_hybrid_score = max(seen.values())
        graph_queries = list({raw_query, rewritten} | set(variants))
        for gq in graph_queries:
            for cid, gscore in graph_search(gq, top_k=candidate_pool):
                scaled = gscore * top_hybrid_score
                if cid not in seen or scaled > seen[cid]:
                    seen[cid] = scaled

    # 3. graph expansion from top hits — runs last so it seeds from the
    #    largest possible pool (hybrid + direct graph combined)
    if _GRAPH_AVAILABLE and seen:
        from graphs.build_graph import extract_entities, detect_script
        from graphs.graph_index import get_chunks_for_entities

        top_score = max(seen.values())
        seed_ids  = [cid for cid, _ in sorted(seen.items(), key=lambda x: x[1], reverse=True)[:5]]
        seed_entities = set()
        for cid in seed_ids:
            if cid in text_by_id:
                chunk_text = text_by_id[cid] if isinstance(text_by_id[cid], str) else text_by_id[cid].get("text", "")
                seed_entities |= extract_entities(chunk_text, detect_script(chunk_text))

        if seed_entities:
            for cid, gscore in get_chunks_for_entities(list(seed_entities), hop=1).items():
                if cid not in seen:
                    seen[cid] = gscore * top_score * 0.5

    # 4. filter + rerank
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