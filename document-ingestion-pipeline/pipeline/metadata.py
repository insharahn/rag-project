"""Extract document metadata: file-level facts (size, type), content-derived
facts (character/word counts, detected language), and native document
properties (title, author, creation date) where the format provides them.

Native properties are often empty — many PDFs and DOCX files were never
given a title/author in their document properties, even when the content
clearly has one printed on the page. That's expected, not a bug: this
reports what the file's metadata fields actually contain, not what a
human would guess from reading the content.
"""

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document
import trafilatura

from pipeline.ingest import ingest_document, detect_file_type
from pipeline.clean import clean_text
from pipeline.detect_language import detect_language


def _native_properties(path: str, file_type: str) -> dict:
    """Pull format-specific document properties (title, author, created date)."""
    if file_type == "pdf":
        doc = fitz.open(path)
        meta = doc.metadata or {}
        doc.close()
        return {
            "title": meta.get("title") or None,
            "author": meta.get("author") or None,
            # PDF dates come in their own native format (e.g. "D:20250125000000+00'00'"),
            # left unparsed here — fine for a metadata report, revisit only if you
            # need to sort/filter by date later.
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

    return {
        "path": str(file_path),
        "filename": file_path.name,
        "file_type": file_type,
        "file_size_bytes": file_path.stat().st_size,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()),
        "language": detect_language(cleaned),
        **_native_properties(path, file_type),
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