# scripts/smoke_test_toxicity.py
"""
Sanity check on textdetox/bert-multilingual-toxicity-classifier.
Llama-Guard-3-8B was tested and disqualified: 91-minute download, then
hung indefinitely at 0% weight loading — impractical for this hardware.
This lightweight alternative (~178M params) is the practical path forward.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "0"  # need network for first download
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="textdetox/bert-multilingual-toxicity-classifier",
)

test_cases = [
    ("You're all idiots and should just disappear.", "en", "toxic", "harassment"),
    ("What's the weather like today?", "en", "neutral", "obviously benign"),
    ("You are amazing and so helpful!", "en", "neutral", "obviously benign, positive"),

    ("너희 나라 사람들은 다 멍청하고 없어져야 해.", "ko", "toxic", "harassment, Korean"),
    ("오늘 날씨가 어때요?", "ko", "neutral", "obviously benign, Korean"),

    ("تم سب بیوقوف ہو اور تمہیں ختم ہو جانا چاہیے۔", "ur", "toxic", "harassment, Urdu"),
    ("آج موسم کیسا ہے؟", "ur", "neutral", "obviously benign, Urdu"),
]

print(f"{'lang':<5}{'expected':<9}{'predicted':<12}{'score':<8}{'match':<7}text")
print("-" * 100)

correct = 0
lang_stats = {"en": [0, 0], "ko": [0, 0], "ur": [0, 0]}

for text, lang, expected, note in test_cases:
    result = classifier(text)[0]
    raw_label = result["label"].lower()
    predicted = "toxic" if "toxic" in raw_label or raw_label in ("1", "label_1") else "neutral"
    score = result["score"]
    match = "OK" if predicted == expected else "X"
    if match == "OK":
        correct += 1
        lang_stats[lang][0] += 1
    lang_stats[lang][1] += 1
    print(f"{lang:<5}{expected:<9}{predicted:<12}{score:<8.3f}{match:<7}{text[:45]}  [raw: {raw_label}]")

print(f"\nOverall: {correct}/{len(test_cases)}")
for lang, (c, t) in lang_stats.items():
    print(f"  {lang}: {c}/{t}")