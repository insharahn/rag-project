# agents/security_agent.py
"""
Security agent:LangGraph node wrapper around the existing Week 4
guardrail module. 
Adapts check_input_deep and check_output to the graph's state/node 
interface.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.guardrail import check_input_deep, check_output


def security_input_node(state: dict) -> dict:
    """Runs before retrieval. Blocks the whole pipeline if the query
    itself is flagged."""
    result = check_input_deep(state["query"], language=state.get("language", "en"))
    return {
        **state,
        "input_blocked": result.blocked,
        "input_block_reasons": result.reasons,
    }


def security_output_node(state: dict) -> dict:
    """Runs after generation. Blocks the final answer if it leaks
    exfiltration/PII content."""
    answer = state.get("answer", "")
    result = check_output(answer, language=state.get("language", "en"))
    return {
        **state,
        "output_blocked": result.blocked,
        "output_block_reasons": result.reasons,
    }


def route_after_input_security(state: dict) -> str:
    return "blocked" if state["input_blocked"] else "continue"


def route_after_output_security(state: dict) -> str:
    return "blocked" if state["output_blocked"] else "continue"