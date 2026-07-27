# agents/workflow.py
"""
Multi-agent workflow — wires all five agents into a LangGraph StateGraph.

Flow:
  security_input -> [blocked -> END]
                  -> retrieval -> research -> summarization -> validation
                                                                    -> security_output -> [blocked -> END]
                                                                                        -> END (return answer)

research conditionally expands retrieval (query decomposition) based on
low confidence; validation only reports pass/fail as metadata
"""
import sys
from typing import TypedDict, Optional
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END

from agents.security_agent import (
    security_input_node, security_output_node,
    route_after_input_security, route_after_output_security,
)
from agents.retrieval_agent import retrieval_node
from agents.research_agent import research_node
from agents.summarization_agent import summarization_node
from agents.validation_agent import validation_node
from generation.citation_generator import generate_answer

class WorkflowState(TypedDict):
    query: str
    language: str
    top_k: int
    history: list
    input_blocked: bool
    input_block_reasons: list
    retrieved_chunks: list
    top_score: float
    research_expanded: bool
    sub_queries_used: Optional[list]
    draft_answer: str
    draft_sources: dict
    draft_confidence: str
    draft_top_score: float
    draft_followups: list
    validation_passed: bool
    validation_grounded: bool
    validation_cited_correctly: bool
    validation_addresses_query: bool
    validation_issues: str
    output_blocked: bool
    output_block_reasons: list
    final_answer: str


def finalize_node(state: dict) -> dict:
    return {
        **state,
        "final_answer": state.get("draft_answer", ""),
        "final_sources": state.get("draft_sources", {}),
        "final_confidence": state.get("draft_confidence", "low"),
        "final_top_score": state.get("draft_top_score", 0.0),
        "final_followups": state.get("draft_followups", []),
        "final_validated": state.get("validation_passed", False),
        "final_validation_issues": state.get("validation_issues", ""),
        "final_research_expanded": state.get("research_expanded", False),
        "blocked": False,
    }


def blocked_input_node(state: dict) -> dict:
    return {**state, "final_answer": "This request cannot be processed.", "blocked": True}


def blocked_output_node(state: dict) -> dict:
    return {**state, "final_answer": "This request cannot be processed.", "blocked": True}

def route_after_validation(state: dict) -> str:
    if state.get("validation_passed", True):
        return "passed"
    if state.get("_retry_pass"):
        # already retried once and still failed — stop looping, hedge instead
        return "give_up"
    return "retry"


def mark_retry_node(state: dict) -> dict:
    return {**state, "_retry_pass": True}


def hedge_node(state: dict) -> dict:
    """Validation failed even after one retry — don't serve a
    known-flawed answer; degrade to an honest hedge instead."""
    return {
        **state,
        "draft_answer": (
            "I found some relevant information, but I'm not confident enough "
            "in the accuracy of a direct answer to state one. "
            f"({state.get('validation_issues', 'unspecified validation concern')})"
        ),
        "draft_confidence": "low",
    }
    
def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("security_input", security_input_node)
    graph.add_node("blocked_input", blocked_input_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("research", research_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("validation", validation_node)
    graph.add_node("security_output", security_output_node)
    graph.add_node("blocked_output", blocked_output_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("mark_retry", mark_retry_node)
    graph.add_node("hedge", hedge_node)

    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {"passed": "security_output", "retry": "mark_retry", "give_up": "hedge"},
    )
    graph.add_edge("mark_retry", "summarization")  # loop back, feedback now attached
    graph.add_edge("hedge", "security_output")

    graph.set_entry_point("security_input")

    graph.add_conditional_edges(
        "security_input",
        route_after_input_security,
        {"blocked": "blocked_input", "continue": "retrieval"},
    )
    graph.add_edge("blocked_input", END)

    graph.add_edge("retrieval", "research")
    graph.add_edge("research", "summarization")
    graph.add_edge("summarization", "validation")

    graph.add_conditional_edges(
        "security_output",
        route_after_output_security,
        {"blocked": "blocked_output", "continue": "finalize"},
    )
    graph.add_edge("blocked_output", END)
    graph.add_edge("finalize", END)

    return graph.compile()


workflow = build_workflow()