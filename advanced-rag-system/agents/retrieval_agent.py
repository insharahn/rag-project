# agents/retrieval_agent.py
"""
Retrieval agent — LangGraph node wrapper around the existing retrieve()
pipeline (rewrite -> multi-query -> hybrid search -> graph search -> 
RRF fusion -> rerank). 
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pipeline import retrieve


def retrieval_node(state: dict) -> dict:
    """Runs retrieval against the current query. Stores retrieved
    chunks in state for downstream agents (research, summarization)."""
    chunks = retrieve(
        state["query"],
        top_k=state.get("top_k", 5),
        candidate_pool=10,
        history=state.get("history", []),
    )
    return {
        **state,
        "retrieved_chunks": chunks,
        "top_score": chunks[0][2] if chunks else 0.0,
    }