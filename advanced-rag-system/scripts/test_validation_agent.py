# scripts/test_validation_agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval_agent import retrieval_node
from agents.research_agent import research_node
from agents.summarization_agent import summarization_node
from agents.validation_agent import validation_node

query = "What is a class in OOP?"

state = {"query": query, "top_k": 5, "history": []}
state = retrieval_node(state)
state = research_node(state)
state = summarization_node(state)
state = validation_node(state)

print(f"Validation passed: {state['validation_passed']}")
print(f"  Grounded: {state['validation_grounded']}")
print(f"  Cited correctly: {state['validation_cited_correctly']}")
print(f"  Addresses query: {state['validation_addresses_query']}")
print(f"  Issues: {state['validation_issues']}")