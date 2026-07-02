"""Detect the language(s) present in a document's text.

Uses langdetect's probability distribution across candidate languages,
rather than forcing a single verdict — so a genuinely multilingual
document (e.g. a bilingual report, or a PDF that mixes two scripts)
shows up as multiple languages with their relative confidence, instead
of one language silently swallowing the other.
"""

from langdetect import detect_langs, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

MIN_TEXT_LENGTH = 50
# Languages below this probability are treated as detection noise rather
# than a real second language actually present in the document.
DEFAULT_THRESHOLD = 0.10

#the languages this system will support; eventually, queries outside these will be rejected
SUPPORTED_LANGUAGES = {"en", "ur", "ar", "fr", "ko"}


def detect_languages(text: str, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Return [{"language": code, "probability": float}, ...], sorted by
    probability descending. Empty list if text is too short or detection
    fails outright.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return []
    try:
        results = detect_langs(text)
    except LangDetectException:
        return []
    return [
        {"language": r.lang, "probability": round(r.prob, 4)}
        for r in results
        if r.prob >= threshold
    ]

def detect_language(text: str) -> str | None:
    """Single-language convenience wrapper: just the most likely code,
    or None. Kept for callers (like metadata.py) that want one answer
    rather than a full distribution.
    """
    languages = detect_languages(text, threshold=0.0)
    return languages[0]["language"] if languages else None


def is_supported(languages: list[dict]) -> bool:
    """Return True if the document's primary language is in the supported
    set. Used as a metadata flag, not as a hard rejection gate — the
    pipeline ingests all documents regardless; enforcement is handled
    downstream.
    """
    if not languages:
        return False
    return languages[0]["language"] in SUPPORTED_LANGUAGES

if __name__ == "__main__":
    import sys
    from pipeline.ingest import ingest_document
    from pipeline.clean import clean_text

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.detect_language <path-to-document>")
        sys.exit(1)

    cleaned = clean_text(ingest_document(sys.argv[1])["full_text"])
    languages = detect_languages(cleaned)

    if not languages:
        print("Detected language: None (text too short or detection failed)")
    else:
        print("Detected languages:")
        for entry in languages:
            print(f"  {entry['language']}: {entry['probability']:.1%}")