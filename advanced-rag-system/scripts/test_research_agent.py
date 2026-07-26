# scripts/test_research_agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval_agent import retrieval_node
from agents.research_agent import research_node

query = "What is a class in OOP?"

state = {"query": query, "top_k": 5, "history": []}
state = retrieval_node(state)
print(f"Initial retrieval top score: {state['top_score']:.3f}")

state = research_node(state)
print(f"Research expanded: {state['research_expanded']}")
print(f"Final chunk count: {len(state['retrieved_chunks'])}")
for cid, chunk, score in state["retrieved_chunks"][:5]:
    print(f"  [{score:.3f}] {chunk.get('source_doc', cid)}")