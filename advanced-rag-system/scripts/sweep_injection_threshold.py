# scripts/sweep_injection_threshold.py
"""
Sweeps INJECTION_THRESHOLD against the existing 125-prompt eval set to
find the threshold that actually optimizes recall without hurting precision.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import pipeline as hf_pipeline

INJECTION_LABEL_MAP = {"label_1": "malicious", "label_0": "benign"}
_classifier = hf_pipeline("text-classification", model="meta-llama/Llama-Prompt-Guard-2-86M")

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "guardrail_eval_set.json"
prompts = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["prompts"]

# get raw scores once, then sweep thresholds cheaply against cached scores
scored = []
for p in prompts:
    result = _classifier(p["text"], truncation=True, max_length=512)[0]
    label = INJECTION_LABEL_MAP.get(result["label"].lower(), "benign")
    score = result["score"] if label == "malicious" else 0.0  # 0 if model says benign at all
    scored.append({
        "id": p["id"], "expected_blocked": p["expected_blocked"],
        "category": p["category"], "score": score,
    })

print(f"{'threshold':<10}{'precision':<11}{'recall':<9}{'f1':<8}{'blocked_count'}")
for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
    tp = sum(1 for s in scored if s["expected_blocked"] and s["score"] >= threshold)
    fp = sum(1 for s in scored if not s["expected_blocked"] and s["score"] >= threshold)
    fn = sum(1 for s in scored if s["expected_blocked"] and s["score"] < threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
    blocked = tp + fp
    print(f"{threshold:<10}{precision:<11.3f}{recall:<9.3f}{f1:<8.3f}{blocked}")