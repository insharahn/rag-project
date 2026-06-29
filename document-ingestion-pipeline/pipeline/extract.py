"""Extract text from PDF documents.

For now this only handles PDFs with an existing text layer.
OCR for scanned pages comes in the next step.
"""

import sys
import fitz  # PyMuPDF


def extract_pdf_text(path: str) -> dict:
    """Extract text from a PDF, page by page.

    Returns a dict with:
      - pages: list of per-page text strings
      - full_text: all pages joined together
      - likely_scanned_pages: page numbers (0-indexed) where almost no
        text was found, meaning they're probably scanned images and will
        need OCR (handled in the next step).
      - page_count: total number of pages
    """
    doc = fitz.open(path)
    pages = []
    likely_scanned_pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text().strip()
        pages.append(text)
        if len(text) < 20:
            likely_scanned_pages.append(page_number)

    doc.close()

    return {
        "pages": pages,
        "full_text": "\n\n".join(pages),
        "likely_scanned_pages": likely_scanned_pages,
        "page_count": len(pages),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline/extract.py <path-to-pdf>")
        sys.exit(1)

    result = extract_pdf_text(sys.argv[1])

    print(f"Pages: {result['page_count']}")
    print(f"Likely scanned pages: {result['likely_scanned_pages']}")
    print("\n--- First 500 characters of extracted text ---\n")
    print(result["full_text"][:500])