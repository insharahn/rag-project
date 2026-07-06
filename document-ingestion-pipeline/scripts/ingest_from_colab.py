"""Ingest pre-extracted OCR text files from Colab into the pipeline.

Usage:
  python -m scripts.ingest_from_colab_ocr <path-to-unzipped-folder>

Expects the folder to contain:
  <stem>.txt              -- extracted text from EasyOCR
  <stem>_ocr_meta.json   -- OCR metadata (confidence, page count, etc.)

Runs the rest of the pipeline (cleaning, language detection, chunking,
dedup, metadata, persist) exactly as the API upload endpoint does, so
the output lands in processed_documents/ in the same format.
"""

import json
import sys
from pathlib import Path

from pipeline.clean import clean_text
from pipeline.detect_language import detect_languages, is_supported
from pipeline.dedup import exact_hash
from pipeline.chunk_fixed import chunk_fixed_size
from pipeline.chunk_recursive import chunk_recursive
from pipeline.chunk_semantic import chunk_semantic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import numpy as np

PROCESSED_DIR = Path("processed_documents")

LANGUAGE_STRATEGY_MAP = {
    "en": "recursive",
    "fr": "semantic",
    "ur": "semantic",
    "ko": "semantic",
    "ar": "semantic",
}

CHUNKERS = {
    "fixed": chunk_fixed_size,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
}


def ingest_one(txt_path: Path, ocr_meta_path: Path | None):
    display_name = txt_path.stem + ".pdf"  # reconstruct original filename
    display_stem = txt_path.stem

    print(f"  Ingesting: {display_name}")

    raw_text = txt_path.read_text(encoding="utf-8")
    cleaned  = clean_text(raw_text)

    languages        = detect_languages(cleaned)
    primary_language = languages[0]["language"] if languages else None
    strategy         = LANGUAGE_STRATEGY_MAP.get(primary_language, "recursive")

    ocr_meta = {}
    if ocr_meta_path and ocr_meta_path.exists():
        try:
            ocr_meta = json.loads(ocr_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Dedup against already-processed documents
    is_duplicate = False
    duplicate_of = None
    PROCESSED_DIR.mkdir(exist_ok=True)
    metadata_path = PROCESSED_DIR / "metadata.json"
    existing_texts = {}

    if metadata_path.exists():
        try:
            master = json.loads(metadata_path.read_text(encoding="utf-8"))
            for fname, _meta in master.items():
                t = PROCESSED_DIR / f"{Path(fname).stem}.txt"
                if t.exists():
                    text = t.read_text(encoding="utf-8")
                    if text:
                        existing_texts[fname] = text
        except Exception:
            pass

    if cleaned and existing_texts:
        incoming_hash = exact_hash(cleaned)
        for fname, stored_text in existing_texts.items():
            if exact_hash(stored_text) == incoming_hash:
                is_duplicate = True
                duplicate_of = {"filename": fname, "similarity": 1.0}
                break
        if not is_duplicate:
            try:
                names = list(existing_texts.keys())
                texts = [cleaned] + [existing_texts[n] for n in names]
                vec = TfidfVectorizer(stop_words="english", max_features=20000)
                matrix = vec.fit_transform(texts)
                sims = sk_cosine(matrix[0:1], matrix[1:])[0]
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim >= 0.85:
                    is_duplicate = True
                    duplicate_of = {"filename": names[best_idx], "similarity": round(best_sim, 4)}
            except Exception:
                pass

    # Chunk
    chunks = CHUNKERS[strategy](cleaned)
    chunks_result = {
        "strategy": strategy,
        "num_chunks": len(chunks),
        "avg_token_count": round(sum(c["token_count"] for c in chunks) / len(chunks), 1) if chunks else 0,
        "chunks": chunks,
    }

    # Extraction info from OCR meta
    extraction_info = {
        "page_count": ocr_meta.get("page_count"),
        "ocr_engine": ocr_meta.get("ocr_engine", "easyocr"),
        "ocr_pages": list(range(ocr_meta.get("page_count", 0))),
        "ocr_confidence": ocr_meta.get("per_page_confidence", {}),
        "low_confidence_pages": ocr_meta.get("low_confidence_pages", []),
        "avg_confidence": ocr_meta.get("avg_confidence"),
    }

    meta_entry = {
        "filename": display_name,
        "file_type": "pdf",
        "title": display_stem.replace("_", " ").replace("-", " ").strip(),
        "author": None,
        "created": None,
        "file_size_bytes": None,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()),
        "languages": languages,
        "primary_language": primary_language,
        "is_supported_language": is_supported(languages),
        "chunking_strategy_used": strategy,
        "extraction_info": extraction_info,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "ingested_via": "colab_easyocr",
    }

    # Persist
    (PROCESSED_DIR / f"{display_stem}.json").write_text(
        json.dumps({"chunking_strategy_used": strategy, "chunks": chunks_result},
                   indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (PROCESSED_DIR / f"{display_stem}.txt").write_text(cleaned or "", encoding="utf-8")

    try:
        master = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    except Exception:
        master = {}
    master[display_name] = meta_entry
    metadata_path.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"    lang={primary_language}, strategy={strategy}, chunks={len(chunks)}, duplicate={is_duplicate}")


def main(folder: str):
    folder_path = Path(folder)
    txt_files = sorted(folder_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {folder}")
        sys.exit(1)

    print(f"Found {len(txt_files)} file(s) to ingest.\n")
    for txt_path in txt_files:
        ocr_meta_path = folder_path / f"{txt_path.stem}_ocr_meta.json"
        ingest_one(txt_path, ocr_meta_path)

    print(f"\nDone. Check processed_documents/ and http://127.0.0.1:8000/ui")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.ingest_from_colab_ocr <path-to-unzipped-folder>")
        sys.exit(1)
    main(sys.argv[1])