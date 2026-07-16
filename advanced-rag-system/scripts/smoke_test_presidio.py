# scripts/smoke_test_presidio.py
"""
Sanity check on Microsoft Presidio for PII detection.
Confirmed going in: English NER support is strong (default spaCy engine).
Korean gets a built-in RRN (national ID) regex recognizer regardless of
NER language. Urdu has NO built-in recognizers — testing to confirm the
gap, not hoping it's covered.
"""
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

test_cases = [
    # (text, language, expected_entities_present)
    ("My name is John Smith and my email is john.smith@email.com", "en", True),
    ("Call me at 555-123-4567 tomorrow.", "en", True),
    ("What is the capital of France?", "en", False),
    ("My social security number is 123-45-6789.", "en", True),

    # Korean RRN format: 6 digits - 7 digits (e.g. 901231-1234567)
    ("제 주민등록번호는 901231-1234567 입니다.", "ko", True),
    ("오늘 날씨가 어때요?", "ko", False),

    # Urdu — expecting these to NOT be caught (confirming the known gap)
    ("میرا نام احمد ہے اور میرا فون نمبر 0300-1234567 ہے۔", "ur", True),  # contains real PII, likely MISSED
    ("آج موسم کیسا ہے؟", "ur", False),
]

print(f"{'lang':<5}{'expect_pii':<11}{'found_entities':<40}text")
print("-" * 110)

for text, lang, expected in test_cases:
    # Presidio requires a supported language code registered with its NLP engine;
    # default engine only has 'en' configured unless we add more — so we pass
    # 'en' universally here just to see what it does with non-English text
    # through the English-configured pipeline (this IS the actual limitation
    # we're testing for)
    try:
        results = analyzer.analyze(text=text, language="en")
        entities = [(r.entity_type, r.score) for r in results]
    except Exception as e:
        entities = [f"ERROR: {e}"]

    found_any = bool(entities) and not (len(entities) == 1 and isinstance(entities[0], str))
    print(f"{lang:<5}{str(expected):<11}{str(entities)[:60]:<40}{text[:40]}")

print("\nNote: analyzer was called with language='en' universally since Presidio's")
print("default engine has no ko/ur language config. This test shows what happens")
print("when non-English text is run through the English pipeline as-is.")