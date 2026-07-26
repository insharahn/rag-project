# agents/research_agent.py
"""
Research agent: decides whether the current retrieval result is
sufficient, or whether the query needs decomposition + additional
retrieval passes. 
Builds on retrieval.query_decompose and retrieval_agent.retrieval_node.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_decompose import decompose_query
from agents.retrieval_agent import retrieval_node
from retrieval.hybrid_search import hybrid_search
from retrieval.bootstrap import get_index

LOW_CONFIDENCE_THRESHOLD = 0.5 

def needs_deeper_research(state: dict) -> bool:
    """Decide whether retrieval was sufficient or more digging is needed."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return True
    top_score = chunks[0][2]
    return top_score < LOW_CONFIDENCE_THRESHOLD

def retrieve_raw(query: str, top_k: int) -> list:
    """Retrieval without the rewrite step — for use with sub-queries that
    are already clean/declarative (e.g. from decomposition)."""
    _, text_by_id = get_index()
    results = hybrid_search(query, k=top_k * 2)

    candidates = [(cid, text_by_id[cid]) for cid, _ in results if cid in text_by_id]

    from retrieval.rerank import rerank
    return rerank(query, candidates, top_k=top_k)

def research_node(state: dict) -> dict:
    """If retrieval looks insufficient, decompose the query into
    sub-questions and retrieve for each, merging results with the
    original retrieval."""
    if not needs_deeper_research(state):
        return {**state, "research_expanded": False}

    sub_queries = decompose_query(state["query"])

    if len(sub_queries) <= 1:
        # decomposition found nothing to split — genuinely just a hard query
        return {**state, "research_expanded": False}

    print(f"[research_agent] low confidence ({state['retrieved_chunks'][0][2]:.3f}), "
          f"decomposing into {len(sub_queries)} sub-queries")

    all_chunks = list(state.get("retrieved_chunks", []))
    seen_ids = {c[0] for c in all_chunks}

    for sub_q in sub_queries:
        sub_chunks = retrieve_raw(sub_q, top_k=state.get("top_k", 5))
        for chunk_tuple in sub_chunks:
            if chunk_tuple[0] not in seen_ids:
                all_chunks.append(chunk_tuple)
                seen_ids.add(chunk_tuple[0])

    all_chunks.sort(key=lambda c: c[2], reverse=True)

    return {
        **state,
        "retrieved_chunks": all_chunks[:state.get("top_k", 5) * 2],  # widen slightly since multiple sub-queries contributed
        "research_expanded": True,
        "sub_queries_used": sub_queries,
    }