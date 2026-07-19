# scripts/build_english_eval_set.py
"""
Samples English attack + benign prompts from Necent's aggregated dataset
for the guardrail eval set. 
Korean/Urdu are hand-written separately.

Maps Necent's fields onto our 5-category schema:
  injection    <- prompt_type='prompt_injection', prompt_adversarial=1
  jailbreak    <- prompt_type='jailbreak', prompt_adversarial=1
  exfiltration <- source='TensorTrust' or attack_technique in ('extraction','hijacking')
  toxicity     <- prompt_type='toxicity', is_dangerous=1
  pii          <- not present in Necent, hand-written separately
  benign       <- category in ('benign','safe'), prompt_harmful=0
"""
import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)  # reproducible sampling

N_PER_CATEGORY = 8  # target count per English category before hand-review

ds = load_dataset("Necent/llm-jailbreak-prompt-injection-dataset")["train"]
en_rows = ds.filter(lambda x: x["language"] == "en")


def sample_rows(filtered_ds, n, seed_offset=0):
    indices = list(range(len(filtered_ds)))
    random.Random(42 + seed_offset).shuffle(indices)
    picked = indices[:n * 3]  # oversample 3x, we'll hand-trim after manual review
    return filtered_ds.select(picked)


print("Filtering injection...")
injection = en_rows.filter(lambda x: x["prompt_type"] == "prompt_injection" and x["prompt_adversarial"] == 1)
print(f"  {len(injection)} candidates")

print("Filtering jailbreak...")
jailbreak = en_rows.filter(lambda x: x["prompt_type"] == "jailbreak" and x["prompt_adversarial"] == 1)
print(f"  {len(jailbreak)} candidates")

print("Filtering exfiltration...")
exfiltration = en_rows.filter(lambda x: x["source"] == "TensorTrust" or x["attack_technique"] in ("extraction", "hijacking"))
print(f"  {len(exfiltration)} candidates")

print("Filtering toxicity...")
toxicity = en_rows.filter(lambda x: x["prompt_type"] == "toxicity" and x["is_dangerous"] == 1)
print(f"  {len(toxicity)} candidates")

print("Filtering benign...")
benign = en_rows.filter(lambda x: x["category"] in ("benign", "safe") and x["prompt_harmful"] == 0)
print(f"  {len(benign)} candidates")

categories = {
    "injection": (injection, True),
    "jailbreak": (jailbreak, True),
    "exfiltration": (exfiltration, True),
    "toxicity": (toxicity, True),
    "benign": (benign, False),
}

output = {"prompts": []}
review_log = []  # separate file for manual review before finalizing

for cat_name, (rows, expected_blocked) in categories.items():
    sampled = sample_rows(rows, N_PER_CATEGORY, seed_offset=hash(cat_name) % 1000)
    for i in range(len(sampled)):
        entry = {
            "id": f"{cat_name}_en_{i+1:03d}",
            "text": sampled[i]["prompt"],
            "language": "en",
            "category": cat_name,
            "subcategory": sampled[i].get("attack_technique") or sampled[i].get("source") or None,
            "expected_blocked": expected_blocked,
            "source": sampled[i]["source"],
            "needs_review": True,  # flip to False once manually confirmed
        }
        review_log.append(entry)

OUT_DIR = Path(__file__).resolve().parent.parent / "eval"
OUT_DIR.mkdir(exist_ok=True)

with open(OUT_DIR / "english_candidates_for_review.json", "w", encoding="utf-8") as f:
    json.dump({"prompts": review_log}, f, ensure_ascii=False, indent=2)

print(f"\n{len(review_log)} candidate prompts written to eval/english_candidates_for_review.json")
print("Review manually before promoting to the final eval set — oversample was 3x target,")
print("so trim each category down to ~5-8 clean, representative, non-redundant examples.")