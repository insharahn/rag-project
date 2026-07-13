"""
scripts/test_graph_contribution.py

Tests graph_search in isolation against the 45 eval queries.
No LLM calls — compares graph retrieval directly against your 
existing retrieval_results.json to see what the graph adds.

Run: python scripts/test_graph_contribution.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.graph_search import graph_search
from retrieval.bootstrap import PROJ2_SRC

EVAL_PATH    = PROJ2_SRC.parent / "eval" / "eval_queries.json"
PREV_RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results" / "retrieval_results.json"

queries  = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]
previous = json.loads(PREV_RESULTS.read_text(encoding="utf-8")) if PREV_RESULTS.exists() else {}

K = 10

hits_graph    = 0   # graph alone finds it
hits_prev     = 0   # previous pipeline found it
hits_new      = 0   # graph finds it AND previous didn't (net new recoveries)
missed_both   = 0   # neither finds it

lang_stats = {"en": {"graph": 0, "prev": 0, "new": 0, "n": 0},
              "ko": {"graph": 0, "prev": 0, "new": 0, "n": 0},
              "ur": {"graph": 0, "prev": 0, "new": 0, "n": 0}}

for q in queries:
    query   = q["query"]
    answer  = q["answer_chunk_id"]
    lang    = q["language"]

    # graph search result (no LLM)
    graph_results = graph_search(query, top_k=K)
    graph_ids     = [cid for cid, _ in graph_results]
    graph_hit     = answer in graph_ids

    # what the previous pipeline got
    prev_entry = previous.get(query, {})
    prev_ids   = prev_entry.get("retrieved_ids", [])
    prev_hit   = answer in prev_ids

    if graph_hit:
        hits_graph += 1
    if prev_hit:
        hits_prev += 1
    if graph_hit and not prev_hit:
        hits_new += 1
        print(f"  [NEW] graph recovers missed answer | lang={lang}")
        print(f"        Q: {query[:70]}")
        print(f"        answer: {answer}")
        if graph_results:
            rank = next((i+1 for i, (cid,_) in enumerate(graph_results) if cid == answer), None)
            print(f"        graph rank: {rank}/{K}")
    if not graph_hit and not prev_hit:
        missed_both += 1

    if lang in lang_stats:
        lang_stats[lang]["n"]    += 1
        lang_stats[lang]["prev"] += int(prev_hit)
        lang_stats[lang]["graph"]+= int(graph_hit)
        lang_stats[lang]["new"]  += int(graph_hit and not prev_hit)

print(f"\n{'='*55}")
print(f"GRAPH SEARCH CONTRIBUTION  (k={K}, {len(queries)} queries)")
print(f"{'='*55}")
print(f"Previous pipeline recall@{K}: {hits_prev}/{len(queries)} = {hits_prev/len(queries):.3f}")
print(f"Graph alone     recall@{K}: {hits_graph}/{len(queries)} = {hits_graph/len(queries):.3f}")
print(f"Net new recoveries (graph finds, prev missed): {hits_new}")
print(f"Missed by both: {missed_both}")
print()
print("By language:")
for lang, s in lang_stats.items():
    n = s["n"]
    print(f"  {lang}: prev={s['prev']}/{n} ({s['prev']/n:.2f})  "
          f"graph={s['graph']}/{n} ({s['graph']/n:.2f})  "
          f"net new={s['new']}")