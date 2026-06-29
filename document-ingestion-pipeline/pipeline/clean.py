"""Clean extracted text: normalize whitespace/unicode, strip control
characters, and fix the line-wrap hyphenation PDFs commonly introduce.

This is type-agnostic — it works on full_text from any document type.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize unicode (e.g. combine accented characters consistently)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (keep newlines and tabs)
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )

    # Fix hyphenated line breaks: "informa-\ntion" -> "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse 3+ blank lines down to a max of one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


if __name__ == "__main__":
    import sys
    from pipeline.ingest import ingest_document

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.clean <path-to-document>")
        sys.exit(1)

    result = ingest_document(sys.argv[1])
    raw = result["full_text"] or ""
    cleaned = clean_text(raw)

    print(f"Raw length: {len(raw)} characters")
    print(f"Cleaned length: {len(cleaned)} characters")
    print("\n--- First 500 characters of cleaned text ---\n")
    print(cleaned[:500])