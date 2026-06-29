"""Detect the dominant language of a document's text.

Uses langdetect (a Python port of Google's language-detection library).
Works well on a decent amount of text; short snippets are unreliable,
which is why a minimum-length threshold is applied before trusting a
result.
"""

from langdetect import detect, DetectorFactory, LangDetectException

# langdetect samples the text internally to make its guess. Without a
# fixed seed, that sampling means results can vary slightly between runs
# on short or ambiguous text. Fixing the seed makes detection
# reproducible — worth having now since this pipeline gets benchmarked
# later, and you don't want flaky results muddying that.
DetectorFactory.seed = 0

MIN_TEXT_LENGTH = 50


def detect_language(text: str) -> str | None:
    """Return an ISO 639-1 language code (e.g. 'en', 'fr'), or None if
    the text is too short or detection fails outright.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


if __name__ == "__main__":
    import sys
    from pipeline.ingest import ingest_document
    from pipeline.clean import clean_text

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.detect_language <path-to-document>")
        sys.exit(1)

    cleaned = clean_text(ingest_document(sys.argv[1])["full_text"])
    language = detect_language(cleaned)

    print(f"Detected language: {language}")