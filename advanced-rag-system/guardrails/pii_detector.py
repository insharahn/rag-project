# guardrails/pii_detector.py
r"""
PII detection: hybrid approach given Presidio's demonstrated limitations
(broken US_SSN recognizer even in isolation; zero Korean/Urdu language
support; some categories like LOCATION are too broad to be useful).

Strategy:
  - Presidio (English only) handles EMAIL/PHONE via spaCy NER. PERSON was
    removed from the trusted set — a bare name is not sensitive on its own
    (character names, public figures, "how do I reach X" are all normal
    text) and blocking on PERSON alone produced false positives on benign
    creative-writing and contact queries once PII started blocking on input.
  - Custom regex handles structured identifiers (SSN, phone numbers,
    Korean RRN) for all three languages.

CONFIDENCE NOTE ON REGEX BOUNDARIES: all digit-pattern regexes use
(?<!\d)...(?!\d) instead of \b. Python's \b treats any \w-adjacent
character as a boundary, but Hangul (and other non-Latin scripts) count
as \w in Python's re module — so a digit sequence immediately followed
by Korean text (no space) silently fails to match with \b, even though
it matches fine when followed by a space. Confirmed directly: a Korean
RRN followed by "이고" (no space) was invisible to \b-based patterns
while the same RRN followed by a space matched normally. Explicit
digit-only boundary checks fix this regardless of what script follows.
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
SSN_PATTERN = re.compile(r"(?<!\d)\d{3}[-.\s]\d{2}[-.\s]\d{4}(?!\d)")

# Phone patterns per language/region-agnostic formats
PHONE_PATTERNS = {
    "en": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
    "ko": re.compile(r"(?<!\d)01[0-9][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"),   # Korean mobile format
    "ur": re.compile(r"(?<!\d)0\d{3}[-.\s]?\d{7}(?!\d)"),                 # Pakistani mobile format
}

# Korean RRN — Presidio has a built-in one, but given the SSN failure, don't
# trust it blindly; hand-write it since it's a fixed, well-documented format
KOREAN_RRN_PATTERN = re.compile(r"(?<!\d)\d{6}[-.\s]?[1-4]\d{6}(?!\d)")

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}(?![a-zA-Z])")

CREDIT_CARD_PATTERN = re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)")
CREDIT_CARD_LAST4_PATTERN = re.compile(
    r"(마지막\s*네\s*자리|last\s*four\s*digits|last\s*4\s*digits|آخری\s*چار\s*ہندسے).{0,15}(?<!\d)\d{4}(?!\d)"
)

CNIC_PATTERN = re.compile(r"(?<!\d)\d{5}-\d{7}-\d(?!\d)")

ADDRESS_PATTERNS = {
    "en": re.compile(r"\b(street address|home address|my address)\b.{0,40}", re.IGNORECASE),
    "ko": re.compile(r"(집\s*주소|저희\s*집).{0,10}(는|은).{0,40}"),
    "ur": re.compile(r"(گھر\s*کا\s*پتہ|میرا\s*پتہ).{0,40}"),
}

def detect_pii(text: str, language: str = "en") -> PIIResult:
    entities = []

    if language == "en":
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

    entities = list(set(entities))

    return PIIResult(
        has_pii=bool(entities),
        entities=entities,
        language=language,
    )