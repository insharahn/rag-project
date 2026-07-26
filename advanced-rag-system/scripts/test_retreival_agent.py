# scripts/test_retrieval_agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval_agent import retrieval_node

state = {
    "query": "What are the themes in Animal Farm?",
    "top_k": 5,
    "history": [],
}

result = retrieval_node(state)
print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
print(f"Top score: {result['top_score']:.3f}")
for cid, chunk, score in result["retrieved_chunks"][:3]:
    print(f"  [{score:.3f}] {chunk.get('source_doc', cid)}")