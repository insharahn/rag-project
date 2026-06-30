"""Extract text from PDF documents, falling back to OCR for scanned pages.
"""

import sys

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

#keep only the languages we actually expect to see in our documents, to avoid
# false positives from Tesseract's language models for other scripts
OCR_LANGUAGES = "eng+kor+urd"

MIN_TEXT_LENGTH_FOR_NATIVE_TEXT = 20


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Lightweight preprocessing to help OCR on poor-quality scans (old
    paper, faint ink, low contrast): grayscale plus autocontrast. This
    won't fix a fundamentally degraded source scan, but it measurably
    helps borderline cases.
    """
    return ImageOps.autocontrast(img.convert("L"))


def _reconstruct_text_from_ocr_data(data: dict) -> str:
    """Rebuild text with real line breaks from Tesseract's word-level
    output, grouped by its own block/paragraph/line numbering -- rather
    than space-joining every word on the page into one run-on string.
    Keeps OCR'd pages structurally closer to native-text pages, which
    matters once cleaning/chunking treat line breaks as meaningful.
    """
    lines: dict[tuple, list[str]] = {}
    order = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
    return "\n".join(" ".join(lines[key]) for key in order)


def _ocr_page(page: "fitz.Page", dpi: int = 300) -> dict:
    """Render a PDF page to an image and run OCR on it, once.

    Returns text plus a mean confidence score (0-100, Tesseract's own
    per-word confidence averaged across the page) so low-quality OCR can
    be flagged downstream instead of trusted at face value. Important:
    this is a measure of Tesseract's own certainty, not ground-truth
    accuracy -- a confident misread (e.g. a library stamp or decorative
    border read as if it were body text) can still score high.
    """
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = _preprocess_for_ocr(img)

    # One Tesseract call, not two -- image_to_data already contains the
    # recognized text per word, so a separate image_to_string call was
    # pure duplicate work (this used to roughly double OCR time).
    data = pytesseract.image_to_data(img, lang=OCR_LANGUAGES, output_type=pytesseract.Output.DICT)
    text = _reconstruct_text_from_ocr_data(data)

    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {"text": text, "confidence": round(mean_confidence, 1)}


def extract_pdf_text(path: str, ocr_dpi: int = 300, low_confidence_threshold: float = 60.0) -> dict:
    """Extract text from a PDF, page by page.

    Tries the normal text layer first for each page. If a page has
    almost no extractable text, it's treated as scanned and OCR'd instead.

    Returns a dict with:
      - pages: list of per-page text strings
      - full_text: all pages joined together
      - ocr_pages: page numbers (0-indexed) that needed OCR
      - ocr_confidence: {page_number: confidence_score} for OCR'd pages
      - low_confidence_pages: OCR'd pages below low_confidence_threshold,
        worth a human spot-check rather than trusting blindly
      - page_count: total number of pages
      - failed_pages: [{"page": n, "error": str}] for pages that raised
        an error during extraction/OCR -- isolated so one bad page
        doesn't take down the whole document
    """
    doc = fitz.open(path)
    pages = []
    ocr_pages = []
    ocr_confidence = {}
    low_confidence_pages = []
    failed_pages = []

    for page_number, page in enumerate(doc):
        try:
            text = page.get_text().strip()

            if len(text) < MIN_TEXT_LENGTH_FOR_NATIVE_TEXT:
                ocr_result = _ocr_page(page, dpi=ocr_dpi)
                text = ocr_result["text"]
                ocr_pages.append(page_number)
                ocr_confidence[page_number] = ocr_result["confidence"]
                if ocr_result["confidence"] < low_confidence_threshold:
                    low_confidence_pages.append(page_number)

            pages.append(text)

        except Exception as e:
            pages.append("")
            failed_pages.append({"page": page_number, "error": str(e)})

    doc.close()

    return {
        "pages": pages,
        "full_text": "\n\n".join(p for p in pages if p),
        "ocr_pages": ocr_pages,
        "ocr_confidence": ocr_confidence,
        "low_confidence_pages": low_confidence_pages,
        "page_count": len(pages),
        "failed_pages": failed_pages,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline/extract_pdf.py <path-to-pdf>")
        sys.exit(1)

    result = extract_pdf_text(sys.argv[1])

    print(f"Pages: {result['page_count']}")
    print(f"OCR'd pages: {result['ocr_pages']}")
    if result["ocr_confidence"]:
        print(f"OCR confidence by page: {result['ocr_confidence']}")
    if result["low_confidence_pages"]:
        print(f"Low-confidence OCR pages (worth a spot-check): {result['low_confidence_pages']}")
    if result["failed_pages"]:
        print(f"Failed pages: {result['failed_pages']}")
    print("\n--- First 500 characters of extracted text ---\n")
    print(result["full_text"][:500])