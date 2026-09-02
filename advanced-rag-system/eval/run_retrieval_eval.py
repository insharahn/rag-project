"""
run_retrieval_eval.py — runs retrieve() over all 45 ground-truth queries,
checkpointing after every single query so progress survives a crash/stop.
Retrieval-only (no generation call) to keep runtime manageable — generation
quality is evaluated separately on a smaller subset.
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pipeline import retrieve
from retrieval.bootstrap import PROJ2_SRC

EVAL_QUERIES_PATH = PROJ2_SRC.parent / "eval" / "eval_queries.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results_with_graph_search" / "retrieval_results.json"
RESULTS_PATH.parent.mkdir(exist_ok=True)


def load_queries():
    data = json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
    return data["queries"]


def load_checkpoint():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(results):
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run_eval(top_k=10):
    queries = load_queries()
    results = load_checkpoint()  # resumable — already-done queries are skipped

    for i, q in enumerate(queries, 1):
        key = q["query"]
        if key in results:
            print(f"[{i}/{len(queries)}] SKIP (already done): {key[:50]!r}")
            continue

        print(f"[{i}/{len(queries)}] Running: {key[:50]!r}")
        t0 = time.time()
        try:
            retrieved = retrieve(key, top_k=top_k)
            retrieved_ids = [cid for cid, _chunk, _score in retrieved]
            elapsed = time.time() - t0

            results[key] = {
                "answer_chunk_id": q["answer_chunk_id"],
                "language": q["language"],
                "retrieved_ids": retrieved_ids,
                "elapsed_seconds": round(elapsed, 2),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[key] = {
                "answer_chunk_id": q["answer_chunk_id"],
                "language": q["language"],
                "retrieved_ids": [],
                "elapsed_seconds": round(time.time() - t0, 2),
                "error": str(e),
            }

        save_checkpoint(results)  # checkpoint after EVERY query, not batched
        print(f"  done in {results[key]['elapsed_seconds']}s")

    print(f"\nCompleted {len(results)}/{len(queries)} queries.")
    return results


if __name__ == "__main__":
    run_eval(top_k=10)