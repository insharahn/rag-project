# guardrails/pii_detector.py
"""
PII detection: hybrid approach

Strategy:
  - Presidio (English only) handles EMAIL/PHONE.
  - Custom regex handles structured identifiers (SSN, phone numbers,
    Korean RRN) for all three languages.
"""
import re
from dataclasses import dataclass
from presidio_analyzer import AnalyzerEngine

_analyzer = AnalyzerEngine()

TRUSTED_PRESIDIO_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER"]


@dataclass
class PIIResult:
    has_pii: bool
    entities: list[str]
    language: str


# Custom SSN regex (Presidio's own recognizer confirmed broken via direct testing)
SSN_PATTERN = re.compile(r"(?<!\d)\d{3}[-.\s]\d{2}[-.\s]\d{4}(?!\d)") #no digits before or after, 3 digits, then a dash/space/dot, then 2 digits, then a dash/space/dot, then 4 digits

# Phone patterns per language/region-agnostic formats
PHONE_PATTERNS = {
    "en": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
    "ko": re.compile(r"(?<!\d)01[0-9][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"), # Korean mobile format
    "ur": re.compile(r"(?<!\d)0\d{3}[-.\s]?\d{7}(?!\d)"), # Pakistani mobile format
}

# Korean RRN — Presidio has a built-in one, but it fails
KOREAN_RRN_PATTERN = re.compile(r"(?<!\d)\d{6}[-.\s]?[1-4]\d{6}(?!\d)") #no digits before or after, 6 digits, then a dash/space/dot, then a digit 1-4, then 6 digits

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}(?![a-zA-Z])") #must have a word, then a @, then a word, then a dot, then 2+ letters, but not followed by a letter

CREDIT_CARD_PATTERN = re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)") #no digits before or after, 4 digits, then a dash/space (optional), then 4 digits, then a dash/space (optional), then 4 digits, then a dash/space (optional), then 4 digits
#for when someone hides the rest but gives the last 4 digits, e.g. "my credit card ends with 1234"
CREDIT_CARD_LAST4_PATTERN = re.compile(
    r"(마지막\s*네\s*자리|last\s*four\s*digits|last\s*4\s*digits|آخری\s*چار\s*ہندسے).{0,15}(?<!\d)\d{4}(?!\d)"
)

CNIC_PATTERN = re.compile(r"(?<!\d)\d{5}-\d{7}-\d(?!\d)") #no digits before or after, 5 digits, then a dash, then 7 digits, then a dash, then 1 digit

ADDRESS_PATTERNS = {
    "en": re.compile(r"\b(street address|home address|my address)\b.{0,40}", re.IGNORECASE),
    "ko": re.compile(r"(집\s*주소|저희\s*집).{0,10}(는|은).{0,40}"),
    "ur": re.compile(r"(گھر\s*کا\s*پتہ|میرا\s*پتہ).{0,40}"),
}

def detect_pii(text: str, language: str = "en") -> PIIResult:
    entities = []

    if language == "en": #presidio only supports English, so we only call it for English text
        try:
            results = _analyzer.analyze(text=text, language="en", score_threshold=0.5)
            entities += [r.entity_type for r in results if r.entity_type in TRUSTED_PRESIDIO_ENTITIES]
        except Exception:
            pass

    if SSN_PATTERN.search(text):
        entities.append("US_SSN")
    if KOREAN_RRN_PATTERN.search(text) and language == "ko":
        entities.append("KOREAN_RRN")
    if CNIC_PATTERN.search(text) and language == "ur":
        entities.append("PAKISTANI_CNIC")
    if EMAIL_PATTERN.search(text):
        entities.append("EMAIL_ADDRESS")
    if CREDIT_CARD_PATTERN.search(text) or CREDIT_CARD_LAST4_PATTERN.search(text):
        entities.append("CREDIT_CARD")
    phone_pattern = PHONE_PATTERNS.get(language)
    if phone_pattern and phone_pattern.search(text):
        entities.append("PHONE_NUMBER")
    address_pattern = ADDRESS_PATTERNS.get(language)
    if address_pattern and address_pattern.search(text):
        entities.append("ADDRESS")

    entities = list(set(entities)) #remove duplicates, e.g. if Presidio and regex both detect the same email address

    return PIIResult(
        has_pii=bool(entities),
        entities=entities,
        language=language,
    )