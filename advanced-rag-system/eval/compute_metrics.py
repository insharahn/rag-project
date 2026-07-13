"""
compute_metrics.py — reads eval/results/retrieval_results.json (produced by
run_retrieval_eval.py) and computes recall@k, precision@k, and MRR at
k=1,3,5,10, both overall and broken down by language (en/ur/ko).

Since each query has exactly ONE relevant chunk (answer_chunk_id), recall@k
is binary (1 if found in top-k, else 0), and precision@k = 1/k on a hit,
0 otherwise. MRR = 1/rank if found anywhere in the retrieved list, else 0.
"""
import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "retrieval_results.json"
K_VALUES = [1, 3, 5, 10]


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found — run eval/run_retrieval_eval.py first."
        )
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def compute_query_metrics(retrieved_ids: list[str], answer_id: str) -> dict:
    """Metrics for a single query, across all k values."""
    metrics = {}

    # find rank of the correct answer (1-indexed), or None if not found
    rank = None
    if answer_id in retrieved_ids:
        rank = retrieved_ids.index(answer_id) + 1

    metrics["mrr"] = (1.0 / rank) if rank else 0.0

    for k in K_VALUES:
        hit = rank is not None and rank <= k
        metrics[f"recall@{k}"] = 1.0 if hit else 0.0
        metrics[f"precision@{k}"] = (1.0 / k) if hit else 0.0

    return metrics


def aggregate(per_query_metrics: list[dict]) -> dict:
    """Averages a list of per-query metric dicts. Skips non-numeric fields
    like 'language' that get carried along for grouping but aren't metrics."""
    if not per_query_metrics:
        return {}
    keys = [k for k in per_query_metrics[0].keys() if k not in ("language", "elapsed_seconds")]
    return {
        key: round(sum(m[key] for m in per_query_metrics) / len(per_query_metrics), 4)
        for key in keys
    }


def compute_metrics():
    results = load_results()

    completed = {q: r for q, r in results.items() if "error" not in r}
    errored = {q: r for q, r in results.items() if "error" in r}

    if errored:
        print(f"WARNING: {len(errored)} queries had errors and are excluded from metrics:")
        for q in errored:
            print(f"  - {q[:60]!r}: {errored[q]['error']}")
        print()

    print(f"Computing metrics over {len(completed)} completed queries.\n")

    # per-query metrics, tagged with language for breakdown
    per_query = []
    for query_text, r in completed.items():
        m = compute_query_metrics(r["retrieved_ids"], r["answer_chunk_id"])
        m["language"] = r["language"]
        m["elapsed_seconds"] = r["elapsed_seconds"]
        per_query.append(m)

    # overall aggregate
    overall = aggregate(per_query)
    avg_latency = round(sum(m["elapsed_seconds"] for m in per_query) / len(per_query), 2)

    print("=== OVERALL ===")
    for k in K_VALUES:
        print(f"  recall@{k}:    {overall[f'recall@{k}']:.4f}")
        print(f"  precision@{k}: {overall[f'precision@{k}']:.4f}")
    print(f"  MRR:         {overall['mrr']:.4f}")
    print(f"  avg latency: {avg_latency}s/query")
    print()

    # per-language breakdown
    languages = sorted(set(m["language"] for m in per_query))
    for lang in languages:
        lang_metrics = [m for m in per_query if m["language"] == lang]
        lang_agg = aggregate(lang_metrics)
        lang_latency = round(sum(m["elapsed_seconds"] for m in lang_metrics) / len(lang_metrics), 2)
        print(f"=== LANGUAGE: {lang} (n={len(lang_metrics)}) ===")
        for k in K_VALUES:
            print(f"  recall@{k}:    {lang_agg[f'recall@{k}']:.4f}")
            print(f"  precision@{k}: {lang_agg[f'precision@{k}']:.4f}")
        print(f"  MRR:         {lang_agg['mrr']:.4f}")
        print(f"  avg latency: {lang_latency}s/query")
        print()

    # save a machine-readable summary alongside the raw results
    summary = {
        "overall": overall,
        "avg_latency_seconds": avg_latency,
        "n_queries": len(per_query),
        "n_errored": len(errored),
        "by_language": {
            lang: {
                **aggregate([m for m in per_query if m["language"] == lang]),
                "avg_latency_seconds": round(
                    sum(m["elapsed_seconds"] for m in per_query if m["language"] == lang)
                    / len([m for m in per_query if m["language"] == lang]),
                    2,
                ),
                "n": len([m for m in per_query if m["language"] == lang]),
            }
            for lang in languages
        },
    }
    summary_path = RESULTS_PATH.parent / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    compute_metrics()