"""Extract text from PDF documents, falling back to OCR for scanned pages.

Two OCR engines, chosen once per document:
  - Tesseract for English and Korean documents
  - EasyOCR for Urdu documents, since Tesseract's accuracy on Urdu's
    Nastaliq script is a documented weak point of the engine itself.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8")

OCR_LANGUAGES = "eng+kor+urd"
MIN_TEXT_LENGTH_FOR_NATIVE_TEXT = 20
MAX_OCR_WORKERS = 4
SCRIPT_SAMPLE_DPI = 150  # cheap, just for routing -- not the real OCR pass

_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("Loading EasyOCR Urdu model (first use only)...")
        _easyocr_reader = easyocr.Reader(["ur"], gpu=False)
    return _easyocr_reader


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(img.convert("L"))


def _reconstruct_text_from_ocr_data(data: dict) -> str:
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


def _dominant_script(text: str) -> str:
    counts = {"urd": 0, "kor": 0, "eng": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            counts["urd"] += 1
        elif 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF:
            counts["kor"] += 1
        elif cp < 128 and ch.isalpha():
            counts["eng"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "eng"

def _detect_document_script(ocr_pages: list, sample_size: int = 3) -> str:
    """Sample up to `sample_size` OCR-needing pages, not just the first,
    and pool script counts across all of them before deciding. A single
    page -- especially the first one, which is often a cover/title page
    with unrepresentative content -- can misroute an entire document.
    """
    combined_counts = {"urd": 0, "kor": 0, "eng": 0}
    for _, page in ocr_pages[:sample_size]:
        img = _preprocess_for_ocr(_render_page(page, SCRIPT_SAMPLE_DPI))
        data = pytesseract.image_to_data(img, lang=OCR_LANGUAGES, output_type=pytesseract.Output.DICT)
        sample_text = " ".join(w for w in data["text"] if w.strip())
        for ch in sample_text:
            cp = ord(ch)
            if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
                combined_counts["urd"] += 1
            elif 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF:
                combined_counts["kor"] += 1
            elif cp < 128 and ch.isalpha():
                combined_counts["eng"] += 1
    return max(combined_counts, key=combined_counts.get) if any(combined_counts.values()) else "eng"

def _render_page(page: "fitz.Page", dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _ocr_with_tesseract(img: Image.Image) -> dict:
    img = _preprocess_for_ocr(img)
    data = pytesseract.image_to_data(img, lang=OCR_LANGUAGES, output_type=pytesseract.Output.DICT)
    text = _reconstruct_text_from_ocr_data(data)
    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {"text": text, "confidence": round(mean_confidence, 1)}


def _ocr_with_easyocr(img: Image.Image) -> dict:
    reader = _get_easyocr_reader()
    img_array = np.array(img.convert("RGB"))

    # paragraph=True groups text boxes into reading-order paragraphs --
    # critical for Urdu Nastaliq. Trade-off: paragraph mode returns
    # (bbox, text) tuples only, no per-box confidence scores. We run a
    # second pass without paragraph=True on the same image to get
    # confidence scores for the overall quality signal, without using
    # those results for the actual text content.
    results = reader.readtext(img_array, detail=1, paragraph=True)
    if not results:
        return {"text": "", "confidence": 0.0}

    lines = [text for (_, text) in results]

    # Confidence pass -- detail=1 without paragraph gives (bbox, text, conf)
    conf_results = reader.readtext(img_array, detail=1, paragraph=False)
    confidences = [conf for (_, _, conf) in conf_results] if conf_results else []

    return {
        "text": "\n".join(lines),
        "confidence": round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0.0,
    }


def extract_pdf_text(path: str, ocr_dpi: int = 300, low_confidence_threshold: float = 60.0) -> dict:
    """Extract text from a PDF, page by page.

    Returns a dict with:
      - pages, full_text, page_count, failed_pages -- as before
      - ocr_pages: page numbers that needed OCR
      - ocr_confidence: {page_number: confidence_score}
      - ocr_engine_used: "tesseract" | "easyocr" | "none" -- one engine
        for the whole document
      - low_confidence_pages: OCR'd pages below low_confidence_threshold
    """
    doc = fitz.open(path)
    
    pages = [None] * len(doc)
    failed_pages = []
    ocr_targets = []  # (page_number, fitz.Page)

    for page_number, page in enumerate(doc):
        try:
            text = (page.get_text() or "").strip()
            if len(text) < MIN_TEXT_LENGTH_FOR_NATIVE_TEXT:
                ocr_targets.append((page_number, page))
            else:
                pages[page_number] = text
        except Exception as e:
            pages[page_number] = ""
            failed_pages.append({"page": page_number, "error": str(e)})

    ocr_pages, ocr_confidence, low_confidence_pages = [], {}, []
    engine_used = "none"

    if ocr_targets:
        # decide the engine for the whole document.
        script = _detect_document_script(ocr_targets)
        engine_used = "easyocr" if script == "urd" else "tesseract"

        # Render every OCR-needing page at full quality now, while doc
        # is still open and we're still in the main thread (PyMuPDF
        # rules: no concurrent access).
        rendered = [(pn, _render_page(p, ocr_dpi)) for pn, p in ocr_targets]
        doc.close()

        if engine_used == "tesseract":
            with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
                futures = {executor.submit(_ocr_with_tesseract, img): pn for pn, img in rendered}
                for future in futures:
                    pn = futures[future]
                    try:
                        result = future.result()
                        pages[pn] = result["text"]
                        ocr_pages.append(pn)
                        ocr_confidence[pn] = result["confidence"]
                        if result["confidence"] < low_confidence_threshold:
                            low_confidence_pages.append(pn)
                    except Exception as e:
                        pages[pn] = ""
                        failed_pages.append({"page": pn, "error": str(e)})
        else:
            # EasyOCR not run in a thread pool to avoid concurrency issues
            for i, (pn, img) in enumerate(rendered, start=1):
                print(f"  EasyOCR: page {pn} ({i}/{len(rendered)})...", flush=True)
                try:
                    result = _ocr_with_easyocr(img)
                    pages[pn] = result["text"]
                    ocr_pages.append(pn)
                    ocr_confidence[pn] = result["confidence"]
                    if result["confidence"] < low_confidence_threshold:
                        low_confidence_pages.append(pn)
                except Exception as e:
                    pages[pn] = ""
                    failed_pages.append({"page": pn, "error": str(e)})
    else:
        doc.close()

    return {
        "pages": pages,
        "full_text": "\n\n".join(p for p in pages if p),
        "ocr_pages": sorted(ocr_pages),
        "ocr_confidence": ocr_confidence,
        "ocr_engine_used": engine_used,
        "low_confidence_pages": sorted(low_confidence_pages),
        "page_count": len(pages),
        "failed_pages": failed_pages,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        print("Usage: python pipeline/extract_pdf.py <path-to-pdf>")
        sys.exit(1)

    result = extract_pdf_text(sys.argv[1])

    print(f"Pages: {result['page_count']}")
    print(f"OCR engine used: {result['ocr_engine_used']}")
    print(f"OCR'd pages: {result['ocr_pages']}")
    if result["ocr_confidence"]:
        print(f"OCR confidence by page: {result['ocr_confidence']}")
    if result["low_confidence_pages"]:
        print(f"Low-confidence OCR pages: {result['low_confidence_pages']}")
    if result["failed_pages"]:
        print(f"Failed pages: {result['failed_pages']}")
    print("\n--- First 1000 characters of extracted text ---\n")
    print(result["full_text"][:1000])