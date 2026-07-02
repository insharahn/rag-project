"""Unified entry point: given any file path, detect its type and
extract text using the right extractor.
"""

import sys
from pathlib import Path

from pipeline.extract_pdf import extract_pdf_text
from pipeline.extract_docx import extract_docx_text
from pipeline.extract_html import extract_html_text


def detect_file_type(path: str) -> str:
    """Return 'pdf', 'docx', or 'html' based on file extension.

    Raises ValueError if the extension isn't one we support yet.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in (".html", ".htm"):
        return "html"
    raise ValueError(f"Unsupported file extension: '{suffix}' for file {path}")


def ingest_document(path: str) -> dict:
    """Detect file type and extract text, in one unified shape.

    The top-level fields (path, file_type, full_text) are the same
    no matter what kind of document this was. Everything that's
    specific to that document type (page count, OCR pages, table
    count, etc.) lives under extraction_info instead.
    """
    file_type = detect_file_type(path)

    if file_type == "pdf":
        result = extract_pdf_text(path)
        return {
            "path": path,
            "file_type": "pdf",
            "full_text": result["full_text"],
            "extraction_info": {
                "page_count": result["page_count"],
                "ocr_pages": result["ocr_pages"],
                "ocr_confidence": result["ocr_confidence"],
                "low_confidence_pages": result["low_confidence_pages"],
                "failed_pages": result["failed_pages"],
            },
        }

    if file_type == "docx":
        result = extract_docx_text(path)
        return {
            "path": path,
            "file_type": "docx",
            "full_text": result["full_text"],
            "extraction_info": {
                "paragraph_count": result["paragraph_count"],
                "table_count": result["table_count"],
            },
        }

    # file_type == "html"
    result = extract_html_text(path)
    return {
        "path": path,
        "file_type": "html",
        "full_text": result["full_text"],
        "extraction_info": {
            "word_count": result["word_count"],
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.ingest <path-to-document>")
        sys.exit(1)

    result = ingest_document(sys.argv[1])

    print(f"File type: {result['file_type']}")
    print(f"Extraction info: {result['extraction_info']}")
    text = result["full_text"] or ""
    print("\n--- First 500 characters of extracted text ---\n")
    print(text[:500])