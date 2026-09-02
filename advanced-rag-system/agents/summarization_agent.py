# agents/summarization_agent.py
"""
Summarization agent: synthesizes retrieved chunks (possibly expanded by
the research agent) into a draft answer. Reuses the existing citation
generation logic, but treats its output as a draft for the validation
agent to check, not a final answer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.citation_generator import generate_answer

def summarization_node(state: dict) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {**state, "draft_answer": "No relevant information was found to answer this question.",
                "draft_sources": {}, "draft_confidence": "low", "draft_top_score": 0.0}

    feedback = state.get("validation_issues") if state.get("_retry_pass") else None
    result = generate_answer(state["query"], chunks, feedback=feedback)

    return {
        **state,
        "draft_answer": result["answer"],
        "draft_sources": result["sources"],
        "draft_confidence": result["confidence"],
        "draft_top_score": result["top_score"],
        "draft_followups": result.get("followup_questions", []),
    }

"""
def summarization_node(state: dict) -> dict:
    #Synthesize retrieved chunks into a draft answer with citations.
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            **state,
            "draft_answer": "No relevant information was found to answer this question.",
            "draft_sources": {},
            "draft_confidence": "low",
            "draft_top_score": 0.0,
        }

    result = generate_answer(state["query"], chunks)

    return {
        **state,
        "draft_answer": result["answer"],
        "draft_sources": result["sources"],
        "draft_confidence": result["confidence"],
        "draft_top_score": result["top_score"],
        "draft_followups": result.get("followup_questions", []),
    }
"""