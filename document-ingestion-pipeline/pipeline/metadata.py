"""Extract document metadata: file-level facts (size, type), content-derived
facts (character/word counts, detected languages), and document properties
(title, author, creation date) where the format provides them.

Title resolution: many files never had a real title set in their document
properties — some are empty, and Word in particular auto-fills a generic
placeholder ("Word Document") rather than leaving it blank. Both cases are
treated as "no real title" and fall back to the filename instead, so the
metadata record always has something useful to show rather than a
technically-correct but unhelpful null or placeholder.
"""

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document
import trafilatura

from pipeline.ingest import ingest_document, detect_file_type
from pipeline.clean import clean_text
from pipeline.detect_language import detect_languages

# Known placeholder values that count as "no real title" rather than a
# genuine one -- compared case-insensitively.
GENERIC_TITLE_PLACEHOLDERS = {"word document", "untitled", ""}


def _resolve_title(native_title: str | None, filename: str) -> str:
    """Return the native title if it's a real one, otherwise derive a
    readable title from the filename."""
    if native_title and native_title.strip().lower() not in GENERIC_TITLE_PLACEHOLDERS:
        return native_title.strip()
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").strip()


def _native_properties(path: str, file_type: str) -> dict:
    """Pull format-specific document properties (title, author, created date)."""
    if file_type == "pdf":
        doc = fitz.open(path)
        meta = doc.metadata or {}
        doc.close()
        return {
            "title": meta.get("title") or None,
            "author": meta.get("author") or None,
            "created": meta.get("creationDate") or None,
        }

    if file_type == "docx":
        props = Document(path).core_properties
        return {
            "title": props.title or None,
            "author": props.author or None,
            "created": props.created.isoformat() if props.created else None,
        }

    # file_type == "html"
    with open(path, "rb") as f:
        html_content = f.read()
    meta = trafilatura.extract_metadata(html_content)
    if meta is None:
        return {"title": None, "author": None, "created": None}
    return {
        "title": meta.title or None,
        "author": meta.author or None,
        "created": meta.date or None,
    }


def extract_metadata(path: str) -> dict:
    """Build a full metadata record for a single document."""
    file_type = detect_file_type(path)
    ingest_result = ingest_document(path)
    cleaned = clean_text(ingest_result["full_text"])
    file_path = Path(path)

    native = _native_properties(path, file_type)
    languages = detect_languages(cleaned)

    return {
        "path": str(file_path),
        "filename": file_path.name,
        "file_type": file_type,
        "file_size_bytes": file_path.stat().st_size,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()),
        "languages": languages,
        "primary_language": languages[0]["language"] if languages else None,
        "title": _resolve_title(native["title"], file_path.name),
        "author": native["author"],
        "created": native["created"],
        "extraction_info": ingest_result["extraction_info"],
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.metadata <path-to-document>")
        sys.exit(1)

    metadata = extract_metadata(sys.argv[1])
    print(json.dumps(metadata, indent=2, ensure_ascii=False))