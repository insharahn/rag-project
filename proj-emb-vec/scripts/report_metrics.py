"""
scripts/report_metrics.py

Reads results/benchmark_raw.json (and fixed/recursive variants) and derives metrics.
Writes results/metrics.json for each strategy.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

K_VALUES      = [1, 3, 5, 10]
MODEL_AXIS_DB = "faiss"
DB_AXIS_MODEL = "bge-m3"

STRATEGY_RESULTS = {
    "semantic":  ROOT / "results",
    "fixed":     ROOT / "results_fixed",
    "recursive": ROOT / "results_recursive",
}

STRATEGIES_TO_REPORT = ["recursive"] #remove any, add any


def quality(per_query):
    langs = sorted({r["language"] for r in per_query})
    def block(rows):
        n = len(rows)
        if n == 0:
            return None
        out = {"n": n, "mrr": round(np.mean([1.0 / r["rank"] if r["rank"] else 0.0 for r in rows]), 4)}
        for k in K_VALUES:
            hits = sum(1 for r in rows if r["rank"] and r["rank"] <= k)
            out[f"recall@{k}"]    = round(hits / n, 4)
            out[f"precision@{k}"] = round(hits / (n * k), 4)
        return out
    res = {"overall": block(per_query)}
    for lang in langs:
        res[lang] = block([r for r in per_query if r["language"] == lang])
    return res


def db_recall_vs_faiss(model, runs):
    faiss_pq = {q["query_idx"]: set(q["retrieved"]) for q in runs[(model, "faiss")]["per_query"]}
    out = {}
    for db in set(k[1] for k in runs):
        pq = runs[(model, db)]["per_query"]
        overlaps = [
            len(set(q["retrieved"]) & faiss_pq[q["query_idx"]]) / max(len(faiss_pq[q["query_idx"]]), 1)
            for q in pq
        ]
        out[db] = round(float(np.mean(overlaps)), 4)
    return out


def crosscut_recall(raw, runs):
    out = {}
    for model in raw["config"]["models"]:
        out[model] = {}
        for db in raw["config"]["dbs"]:
            q = quality(runs[(model, db)]["per_query"])["overall"]
            out[model][db] = {
                "recall@1": q["recall@1"], "recall@5": q["recall@5"],
                "recall@10": q["recall@10"], "mrr": q["mrr"],
            }
    return out


def report_one(strategy: str, res_dir: Path):
    raw_path = res_dir / "benchmark_raw.json"
    if not raw_path.exists():
        print(f"[{strategy}] no benchmark_raw.json in {res_dir}, skipping.")
        return

    raw  = json.loads(raw_path.read_text(encoding="utf-8"))
    runs = {(r["model"], r["db"]): r for r in raw["runs"]}
    LANGS = sorted({q["language"] for r in raw["runs"] for q in r["per_query"]})

    metrics = {"strategy": strategy, "model_axis": {}, "db_axis": {}}

    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy}")
    print(f"=== MODEL QUALITY  (search={MODEL_AXIS_DB}, k={raw['config']['k']}) ===")

    for model in raw["config"]["models"]:
        q = quality(runs[(model, MODEL_AXIS_DB)]["per_query"])
        metrics["model_axis"][model] = q
        o = q["overall"]
        print(f"\n{model}")
        print(f"  overall   recall@1={o['recall@1']:.3f}  recall@5={o['recall@5']:.3f}  "
              f"recall@10={o['recall@10']:.3f}  MRR={o['mrr']:.3f}")
        for lang in LANGS:
            b = q.get(lang)
            if b:
                print(f"  {lang:<9} recall@1={b['recall@1']:.3f}  recall@5={b['recall@5']:.3f}  "
                      f"MRR={b['mrr']:.3f}  (n={b['n']})")

    print(f"\n=== DATABASE  (model={DB_AXIS_MODEL}) ===")
    recall_vs_faiss = db_recall_vs_faiss(DB_AXIS_MODEL, runs)
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
            "search_latencies_ms": r["search_latencies_ms"],
        }
        metrics["db_axis"][db] = row
        mem = f"{row['index_memory_mb']:.0f}" if row["index_memory_mb"] else "n/a"
        print(f"{db:<10}{row['p50_ms']:>8.2f}{row['p95_ms']:>8.2f}{row['p99_ms']:>8.2f}"
              f"{row['build_seconds']:>9.2f}{mem:>9}{row['recall_vs_faiss']:>17.3f}")

    print(f"\n=== CROSS-CUT: recall@1 by model x db ===")
    crosscut = crosscut_recall(raw, runs)
    metrics["crosscut"] = crosscut
    header = f"{'model':<24}" + "".join(f"{db:>10}" for db in raw["config"]["dbs"])
    print(header)
    for model in raw["config"]["models"]:
        row = f"{model:<24}"
        for db in raw["config"]["dbs"]:
            row += f"{crosscut[model][db]['recall@1']:>10.3f}"
        print(row)

    metrics["_meta"] = {
        "strategy": strategy,
        "model_axis_search": MODEL_AXIS_DB, "db_axis_model": DB_AXIS_MODEL,
        "k": raw["config"]["k"], "languages": LANGS,
        "caveat_precision": "1 relevant chunk/query, so precision@k = recall@k / k.",
    }

    out_path = res_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{strategy}] -> {out_path}")


def run():
    for strategy in STRATEGIES_TO_REPORT:
        report_one(strategy, STRATEGY_RESULTS[strategy])


if __name__ == "__main__":
    run()