# guardrails/pii_detector.py
"""
PII detection: hybrid approach given Presidio's demonstrated limitations
(broken US_SSN recognizer even in isolation; zero Korean/Urdu language
support; some categories like LOCATION are too broad to be useful).

Strategy:
  - Presidio (English only) handles PERSON/EMAIL via spaCy NER — verified
    working in initial testing.
  - Custom regex handles structured identifiers (SSN, phone numbers) for
    all three languages, since these are fixed formats regex handles well
    and doesn't depend on NER quality per language.
"""
import re
from dataclasses import dataclass
from presidio_analyzer import AnalyzerEngine

_analyzer = AnalyzerEngine()

# Entity types worth trusting from Presidio's English NER — excludes
# LOCATION (too broad, flags country names) and relies on regex below
# for structured IDs Presidio's own recognizers failed to catch reliably.
TRUSTED_PRESIDIO_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]


@dataclass
class PIIResult:
    has_pii: bool
    entities: list[str]
    language: str


# Custom SSN regex (Presidio's own recognizer confirmed broken via direct testing)
SSN_PATTERN = re.compile(r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b")

# Phone patterns per language/region-agnostic formats
PHONE_PATTERNS = {
    "en": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ko": re.compile(r"\b01[0-9][-.\s]?\d{3,4}[-.\s]?\d{4}\b"),   # Korean mobile format
    "ur": re.compile(r"\b0\d{3}[-.\s]?\d{7}\b"),                  # Pakistani mobile format
}

# Korean RRN — Presidio has a built-in one, but given the SSN failure, don't
# trust it blindly; hand-write it since it's a fixed, well-documented format
KOREAN_RRN_PATTERN = re.compile(r"\b\d{6}[-.\s]?[1-4]\d{6}\b")

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")  # script-agnostic


def detect_pii(text: str, language: str = "en") -> PIIResult:
    entities = []

    # Presidio pass — English only, trusted subset
    if language == "en":
        try:
            results = _analyzer.analyze(text=text, language="en", score_threshold=0.5)
            entities += [r.entity_type for r in results if r.entity_type in TRUSTED_PRESIDIO_ENTITIES]
        except Exception:
            pass  # fail open on Presidio errors, rely on regex below

    # Custom regex pass — all languages
    if SSN_PATTERN.search(text):
        entities.append("US_SSN")
    if KOREAN_RRN_PATTERN.search(text) and language == "ko":
        entities.append("KOREAN_RRN")
    if EMAIL_PATTERN.search(text):
        entities.append("EMAIL_ADDRESS")
    phone_pattern = PHONE_PATTERNS.get(language)
    if phone_pattern and phone_pattern.search(text):
        entities.append("PHONE_NUMBER")

    entities = list(set(entities))  # dedupe (Presidio + regex might both catch email)

    return PIIResult(
        has_pii=bool(entities),
        entities=entities,
        language=language,
    )