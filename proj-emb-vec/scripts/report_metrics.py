"""
scripts/report_metrics.py

Reads results/benchmark_raw.json and derives the two comparison axes:

  MODEL AXIS  (fix DB=faiss exact, vary model): recall@k, MRR, precision@k, PER LANGUAGE
              -> "which embedding model retrieves best, and where does English-only collapse?"
  DB AXIS     (fix one model, vary DB): latency p50/p95/p99, build time, memory,
              recall-vs-FAISS (ANN approximation loss)
              -> "which database, and what does the approximation cost?"

Writes results/metrics.json and prints readable tables.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
raw = json.loads((ROOT / "results" / "benchmark_raw.json").read_text(encoding="utf-8"))

K_VALUES      = [1, 3, 5, 10]
MODEL_AXIS_DB = "faiss"     # exact search -> clean model comparison
DB_AXIS_MODEL = "bge-m3"    # representative model for the DB comparison

runs = {(r["model"], r["db"]): r for r in raw["runs"]}
LANGS = sorted({q["language"] for r in raw["runs"] for q in r["per_query"]})


def quality(per_query):
    """recall@k, precision@k, MRR — overall and per language. Single relevant chunk per query."""
    def block(rows):
        n = len(rows)
        if n == 0:
            return None
        out = {"n": n, "mrr": round(np.mean([1.0 / r["rank"] if r["rank"] else 0.0 for r in rows]), 4)}
        for k in K_VALUES:
            hits = sum(1 for r in rows if r["rank"] and r["rank"] <= k)
            out[f"recall@{k}"]    = round(hits / n, 4)
            out[f"precision@{k}"] = round(hits / (n * k), 4)  # 1 relevant/query -> recall@k / k
        return out
    res = {"overall": block(per_query)}
    for lang in LANGS:
        res[lang] = block([r for r in per_query if r["language"] == lang])
    return res


def db_recall_vs_faiss(model):
    """Per-query top-K set overlap of each DB against FAISS exact (ANN approximation loss)."""
    faiss_pq = {q["query_idx"]: set(q["retrieved"]) for q in runs[(model, "faiss")]["per_query"]}
    out = {}
    for db in raw["config"]["dbs"]:
        pq = runs[(model, db)]["per_query"]
        overlaps = [len(set(q["retrieved"]) & faiss_pq[q["query_idx"]]) / max(len(faiss_pq[q["query_idx"]]), 1)
                    for q in pq]
        out[db] = round(float(np.mean(overlaps)), 4)
    return out


metrics = {"model_axis": {}, "db_axis": {}}

# ---------- MODEL AXIS ----------
print(f"\n=== MODEL QUALITY  (search=FAISS exact, k={raw['config']['k']}) ===")
for model in raw["config"]["models"]:
    q = quality(runs[(model, MODEL_AXIS_DB)]["per_query"])
    metrics["model_axis"][model] = q
    o = q["overall"]
    print(f"\n{model}")
    print(f"  overall   recall@1={o['recall@1']:.3f}  recall@5={o['recall@5']:.3f}  "
          f"recall@10={o['recall@10']:.3f}  MRR={o['mrr']:.3f}")
    for lang in LANGS:
        b = q[lang]
        if b:
            print(f"  {lang:<9} recall@1={b['recall@1']:.3f}  recall@5={b['recall@5']:.3f}  "
                  f"MRR={b['mrr']:.3f}  (n={b['n']})")

# ---------- DB AXIS ----------
print(f"\n\n=== DATABASE  (model={DB_AXIS_MODEL}) ===")
recall_vs_faiss = db_recall_vs_faiss(DB_AXIS_MODEL)
print(f"{'db':<10}{'p50':>8}{'p95':>8}{'p99':>8}{'build_s':>9}{'mem_MB':>9}{'recall_vs_faiss':>17}")
for db in raw["config"]["dbs"]:
    r = runs[(DB_AXIS_MODEL, db)]
    lat = np.array(r["search_latencies_ms"])
    row = {
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "p99_ms": round(float(np.percentile(lat, 99)), 3),
        "build_seconds": r["build_seconds"],
        "index_memory_mb": r["index_memory_mb"],
        "recall_vs_faiss": recall_vs_faiss[db],
    }
    metrics["db_axis"][db] = row
    mem = f"{row['index_memory_mb']:.0f}" if row["index_memory_mb"] else "n/a"
    print(f"{db:<10}{row['p50_ms']:>8.2f}{row['p95_ms']:>8.2f}{row['p99_ms']:>8.2f}"
          f"{row['build_seconds']:>9.2f}{mem:>9}{row['recall_vs_faiss']:>17.3f}")

print(f"\n  note: {raw['config']['note_milvus']}")

metrics["_meta"] = {
    "model_axis_search": MODEL_AXIS_DB, "db_axis_model": DB_AXIS_MODEL,
    "k": raw["config"]["k"], "languages": LANGS,
    "caveat_precision": "1 relevant chunk/query, so precision@k = recall@k / k.",
    "caveat_korean": "Korean queries are held-out sentences (near-exact substrings); compare "
                     "models WITHIN Korean, not Korean vs English absolute scores.",
}
(ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
print("\n[done] -> results/metrics.json")
