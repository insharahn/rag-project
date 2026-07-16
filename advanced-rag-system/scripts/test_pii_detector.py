# scripts/test_pii_detector.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.pii_detector import detect_pii

test_cases = [
    ("My name is John Smith and my email is john.smith@email.com", "en", True),
    ("My social security number is 123-45-6789", "en", True),
    ("Call me at 555-123-4567 tomorrow.", "en", True),
    ("What is the capital of France?", "en", False),
    ("제 주민등록번호는 901231-1234567 입니다.", "ko", True),
    ("오늘 날씨가 어때요?", "ko", False),
    ("میرا فون نمبر 0300-1234567 ہے۔", "ur", True),
    ("آج موسم کیسا ہے؟", "ur", False),
]

correct = 0
for text, lang, expected in test_cases:
    result = detect_pii(text, language=lang)
    match = "OK" if result.has_pii == expected else "X"
    if match == "OK":
        correct += 1
    print(f"{match}  expected={expected!s:<6} got={result.has_pii!s:<6} entities={result.entities}  {text[:40]}")

print(f"\n{correct}/{len(test_cases)}")