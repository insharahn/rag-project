"""Extract text from PDF documents, falling back to OCR for scanned pages.

OCR uses Tesseract via pytesseract. OCR_LANGUAGES lists every language
pack actually installed.
Currently: eng + kr + ur
Tesseract's accuracy on Urdu specifically is not great;
published OCR benchmarks show Tesseract handles far worse than Latin or
Naskh-style Arabic text -- this is a known limitation of the tool.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

OCR_LANGUAGES = "eng+kor+urd"

MIN_TEXT_LENGTH_FOR_NATIVE_TEXT = 20

# Pages needing OCR are run through Tesseract concurrently via threads.
# This is safe specifically because, by this point, all PyMuPDF
# rendering is already done -- only plain PIL images and Tesseract
# subprocess calls remain, neither of which touch fitz. PyMuPDF itself
# explicitly does not support being used from multiple threads, so all
# fitz/PDF work below happens sequentially in the main thread first.
MAX_OCR_WORKERS = 4


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Grayscale plus autocontrast -- cheap, helps borderline scans."""
    return ImageOps.autocontrast(img.convert("L"))


def _reconstruct_text_from_ocr_data(data: dict) -> str:
    """Rebuild text with real line breaks from Tesseract's word-level
    output, grouped by its own block/paragraph/line numbering -- rather
    than space-joining every word on the page into one run-on string.
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


def _ocr_image(page_number: int, img: Image.Image) -> dict:
    """Run OCR on one already-rendered page image, once. Pure Tesseract
    work -- no PyMuPDF/fitz objects involved, safe to call from a
    worker thread.
    """
    img = _preprocess_for_ocr(img)
    data = pytesseract.image_to_data(img, lang=OCR_LANGUAGES, output_type=pytesseract.Output.DICT)
    text = _reconstruct_text_from_ocr_data(data)

    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {"page": page_number, "text": text, "confidence": round(mean_confidence, 1)}


def extract_pdf_text(path: str, ocr_dpi: int = 300, low_confidence_threshold: float = 60.0) -> dict:
    """Extract text from a PDF, page by page.

    Tries the normal text layer first for each page. If a page has
    almost no extractable text, it's rendered to an image and queued
    for OCR.

    Returns a dict with:
      - pages: list of per-page text strings
      - full_text: all pages joined together
      - ocr_pages: page numbers (0-indexed) that needed OCR
      - ocr_confidence: {page_number: confidence_score} for OCR'd pages
      - low_confidence_pages: OCR'd pages below low_confidence_threshold,
        worth a human spot-check rather than trusting blindly
      - page_count: total number of pages
      - failed_pages: [{"page": n, "error": str}] for pages that raised
        an error during extraction/OCR
    """
    doc = fitz.open(path)

    pages = [None] * len(doc)
    failed_pages = []
    ocr_targets = []  # (page_number, PIL.Image) -- only for pages needing OCR

    # Sequential pass: all fitz/PDF work happens here, in the main
    # thread, because PyMuPDF does not support concurrent access.
    for page_number, page in enumerate(doc):
        try:
            text = (page.get_text() or "").strip()
            if len(text) < MIN_TEXT_LENGTH_FOR_NATIVE_TEXT:
                pix = page.get_pixmap(dpi=ocr_dpi, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                ocr_targets.append((page_number, img))
            else:
                pages[page_number] = text
        except Exception as e:
            pages[page_number] = ""
            failed_pages.append({"page": page_number, "error": str(e)})

    doc.close()  # done with fitz entirely from this point on

    ocr_pages = []
    ocr_confidence = {}
    low_confidence_pages = []

    if ocr_targets:
        with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
            futures = {
                executor.submit(_ocr_image, page_number, img): page_number
                for page_number, img in ocr_targets
            }
            for future in futures:
                page_number = futures[future]
                try:
                    result = future.result()
                    pages[page_number] = result["text"]
                    ocr_pages.append(page_number)
                    ocr_confidence[page_number] = result["confidence"]
                    if result["confidence"] < low_confidence_threshold:
                        low_confidence_pages.append(page_number)
                except Exception as e:
                    pages[page_number] = ""
                    failed_pages.append({"page": page_number, "error": str(e)})

    return {
        "pages": pages,
        "full_text": "\n\n".join(p for p in pages if p),
        "ocr_pages": sorted(ocr_pages),
        "ocr_confidence": ocr_confidence,
        "low_confidence_pages": sorted(low_confidence_pages),
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