# scripts/test_toxicity_detector.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.toxicity_detector import detect_toxicity

test_cases = [
    ("You're all idiots and should just disappear.", "en", True),
    ("What's the weather like today?", "en", False),
    ("너희 나라 사람들은 다 멍청하고 없어져야 해.", "ko", True),   # the case the model missed
    ("오늘 날씨가 어때요?", "ko", False),
    ("تم سب بیوقوف ہو اور تمہیں ختم ہو جانا چاہیے۔", "ur", True),  # the case the model missed
    ("آج موسم کیسا ہے؟", "ur", False),
]

correct = 0
for text, lang, expected in test_cases:
    result = detect_toxicity(text, language=lang)
    match = "OK" if result.is_toxic == expected else "X"
    if match == "OK":
        correct += 1
    print(f"{match}  expected={expected!s:<6} got={result.is_toxic!s:<6} source={result.source:<10} "
          f"score={result.model_score}  {text[:45]}")

print(f"\n{correct}/{len(test_cases)}")