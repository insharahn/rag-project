# scripts/test_summarization_agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval_agent import retrieval_node
from agents.research_agent import research_node
from agents.summarization_agent import summarization_node

query = "What is a class in OOP?"

state = {"query": query, "top_k": 5, "history": []}
state = retrieval_node(state)
state = research_node(state)
state = summarization_node(state)

print(f"Confidence: {state['draft_confidence']}")
print(f"Top score: {state['draft_top_score']:.3f}")
print(f"\nDraft answer:\n{state['draft_answer']}")
print(f"\nSources cited: {list(state['draft_sources'].keys())}")