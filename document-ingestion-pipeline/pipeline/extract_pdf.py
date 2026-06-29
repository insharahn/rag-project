"""Extract text from PDF documents, falling back to OCR for scanned pages."""

import sys

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def _ocr_page(page: "fitz.Page", dpi: int = 300) -> str:
    """Render a PDF page to an image and run OCR on it."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img).strip()


def extract_pdf_text(path: str) -> dict:
    """Extract text from a PDF, page by page.

    Tries the normal text layer first for each page. If a page has
    almost no extractable text, it's treated as scanned and OCR'd instead.

    Returns a dict with:
      - pages: list of per-page text strings
      - full_text: all pages joined together
      - ocr_pages: page numbers (0-indexed) that needed OCR
      - page_count: total number of pages
    """
    doc = fitz.open(path)
    pages = []
    ocr_pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text().strip()

        if len(text) < 20:
            text = _ocr_page(page)
            ocr_pages.append(page_number)

        pages.append(text)

    doc.close()

    return {
        "pages": pages,
        "full_text": "\n\n".join(pages),
        "ocr_pages": ocr_pages,
        "page_count": len(pages),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline/extract.py <path-to-pdf>")
        sys.exit(1)

    result = extract_pdf_text(sys.argv[1])

    print(f"Pages: {result['page_count']}")
    print(f"OCR'd pages: {result['ocr_pages']}")
    print("\n--- First 500 characters of extracted text ---\n")
    print(result["full_text"][:500])