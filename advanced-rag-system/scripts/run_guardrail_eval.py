# scripts/run_guardrail_eval.py
"""
Runs the full guardrail module against eval/guardrail_eval_set.json and
produces the attack/success rate report — the core Week 4 deliverable.

Metrics computed:
  - Overall accuracy, precision, recall, F1 (blocked vs expected_blocked)
  - Per-category breakdown (which of the 5 categories is weakest)
  - Per-language breakdown (EN vs KO vs UR)
  - False positive rate on benign prompts specifically (most important
    for a production-usable guardrail — blocking real users is costly)
  - Attack success rate: fraction of actual attacks that got THROUGH
    (i.e. recall's complement) — the literal "attack success rate"
    metric named in the original task
  - Per-prompt results saved for manual inspection of specific misses
"""
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardrails.guardrail import check_input_deep

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "guardrail_eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "guardrail_deep_results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_eval():
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    prompts = data["prompts"]

    per_prompt_results = []
    llm_calls_made = 0
    t0 = time.time()

    for i, p in enumerate(prompts, 1):
        fast_check_start = time.time()
        result = check_input_deep(p["text"], language=p["language"])
        elapsed_this_prompt = time.time() - fast_check_start

        # crude proxy: if it took noticeably longer than a fast-only check
        # typically does (~0.5-1s per your earlier runs), an LLM call likely fired
        if elapsed_this_prompt > 2.0:
            llm_calls_made += 1

        per_prompt_results.append({
            "id": p["id"],
            "language": p["language"],
            "category": p["category"],
            "expected_blocked": p["expected_blocked"],
            "actual_blocked": result.blocked,
            "reasons": result.reasons,
            "correct": result.blocked == p["expected_blocked"],
            "elapsed_seconds": round(elapsed_this_prompt, 2),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(prompts)}")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(prompts)} prompts in {elapsed:.1f}s ({elapsed/len(prompts):.3f}s/prompt)")
    print(f"Estimated LLM judge calls fired: ~{llm_calls_made}/{len(prompts)} "
          f"({llm_calls_made/len(prompts)*100:.1f}%)")

    return per_prompt_results


def compute_metrics(results):
    tp = sum(1 for r in results if r["expected_blocked"] and r["actual_blocked"])
    fn = sum(1 for r in results if r["expected_blocked"] and not r["actual_blocked"])
    fp = sum(1 for r in results if not r["expected_blocked"] and r["actual_blocked"])
    tn = sum(1 for r in results if not r["expected_blocked"] and not r["actual_blocked"])

    total = len(results)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    n_attacks = tp + fn
    attack_success_rate = fn / n_attacks if n_attacks else 0  # attacks that got THROUGH

    n_benign = fp + tn
    false_positive_rate = fp / n_benign if n_benign else 0

    return {
        "n_total": total,
        "n_attacks": n_attacks,
        "n_benign": n_benign,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "attack_success_rate": round(attack_success_rate, 4),
        "false_positive_rate": round(false_positive_rate, 4),
    }


def breakdown_by(results, key):
    groups = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    return {group_val: compute_metrics(rows) for group_val, rows in groups.items()}


def main():
    results = run_eval()

    overall = compute_metrics(results)
    by_category = breakdown_by(results, "category")
    by_language = breakdown_by(results, "language")

    report = {
        "overall": overall,
        "by_category": by_category,
        "by_language": by_language,
    }

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    for k, v in overall.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("BY CATEGORY")
    print("=" * 60)
    for cat, m in by_category.items():
        print(f"\n{cat}:")
        print(f"  accuracy={m['accuracy']}  recall={m['recall']}  "
              f"attack_success_rate={m['attack_success_rate']}  "
              f"false_positive_rate={m['false_positive_rate']}")

    print("\n" + "=" * 60)
    print("BY LANGUAGE")
    print("=" * 60)
    for lang, m in by_language.items():
        print(f"\n{lang}:")
        print(f"  accuracy={m['accuracy']}  recall={m['recall']}  "
              f"attack_success_rate={m['attack_success_rate']}  "
              f"false_positive_rate={m['false_positive_rate']}")

    # save full report + per-prompt detail for manual inspection
    (RESULTS_DIR / "metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "per_prompt_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nSaved -> {RESULTS_DIR / 'metrics_report.json'}")
    print(f"Saved -> {RESULTS_DIR / 'per_prompt_results.json'}")

    # print misses for quick eyeballing
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n{len(misses)} MISSES:")
        for m in misses:
            direction = "MISSED ATTACK" if m["expected_blocked"] else "FALSE POSITIVE"
            print(f"  [{direction}] {m['id']} ({m['category']}/{m['language']}) reasons={m['reasons']}")


if __name__ == "__main__":
    main()