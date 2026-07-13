# scripts/test_missed_both.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.graph_search import graph_search
from retrieval.bootstrap import PROJ2_SRC

EVAL_PATH = PROJ2_SRC.parent / "eval" / "eval_queries.json"
PREV_RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results" / "retrieval_results.json"

queries = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]
previous = json.loads(PREV_RESULTS.read_text(encoding="utf-8"))

for q in queries:
    query, answer = q["query"], q["answer_chunk_id"]
    graph_ids = [cid for cid, _ in graph_search(query, top_k=10)]
    prev_ids = previous.get(query, {}).get("retrieved_ids", [])
    if answer not in graph_ids and answer not in prev_ids:
        print(f"Q: {query}")
        print(f"   answer: {answer}")
        print(f"   graph found: {graph_ids[:3]}...")
        print()