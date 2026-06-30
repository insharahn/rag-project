"""Validate every document in the corpus folders (pdfs/, docxs/, htmls/)
by running it through the real ingestion + cleaning + language detection
pipeline, and write a report of what happened.

Catches documents that crash outright, documents that "succeed" 
but produce suspiciously little text (a sign extraction silently failed), 
and gives one place to see language/OCR stats across the whole corpus at a glance.
"""

import csv
import time
from pathlib import Path

from pipeline.ingest import ingest_document
from pipeline.clean import clean_text
from pipeline.detect_language import detect_languages

CORPUS_DIRS = {
    "pdfs": ["*.pdf"],
    "docxs": ["*.docx"],
    "htmls": ["*.html", "*.htm"],
}

REPORT_PATH = Path("corpus_validation_report.csv")


def find_corpus_files() -> list[Path]:
    files = []
    for folder, patterns in CORPUS_DIRS.items():
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: folder '{folder}' not found, skipping.")
            continue
        for pattern in patterns:
            files.extend(sorted(folder_path.glob(pattern)))
    return files


def validate_one(path: Path) -> dict:
    row = {
        "path": str(path),
        "filename": path.name,
        "status": "ok",
        "error": "",
        "char_count": 0,
        "word_count": 0,
        "languages": "",
        "ocr_pages": "",
        "low_confidence_pages": "",
        "failed_pages": "",
        "seconds": 0.0,
    }

    start = time.time()
    try:
        ingested = ingest_document(str(path))
        cleaned = clean_text(ingested["full_text"]) or ""
        languages = detect_languages(cleaned)

        row["char_count"] = len(cleaned)
        row["word_count"] = len(cleaned.split())
        row["languages"] = "; ".join(f"{l['language']}:{l['probability']}" for l in languages)

        info = ingested.get("extraction_info", {})
        row["ocr_pages"] = len(info.get("ocr_pages", []))
        row["low_confidence_pages"] = len(info.get("low_confidence_pages", []))
        row["failed_pages"] = len(info.get("failed_pages", []))

        if row["char_count"] == 0:
            row["status"] = "warning_empty_text"

    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"

    row["seconds"] = round(time.time() - start, 2)
    return row


def main():
    files = find_corpus_files()
    print(f"Found {len(files)} files to validate.\n")

    results = []
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path} ...", end=" ", flush=True)
        row = validate_one(path)
        print(f"{row['status']} ({row['seconds']}s)")
        results.append(row)

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["status"] == "ok")
    warnings = sum(1 for r in results if r["status"] == "warning_empty_text")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'='*50}")
    print(f"Done. {ok} ok, {warnings} empty-text warnings, {errors} errors.")
    print(f"Full report written to {REPORT_PATH}")

    if warnings or errors:
        print("\nFiles needing a look:")
        for r in results:
            if r["status"] != "ok":
                print(f"  [{r['status']}] {r['path']}  {r['error']}")


if __name__ == "__main__":
    main()