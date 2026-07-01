"""Document ingestion API.

Three endpoints:
  POST /upload         — ingest document(s), run the full pipeline,
                         save results to processed_documents/<name>.json
  GET  /documents      — list all processed documents (metadata only)
  GET  /documents/{filename} — retrieve full results for one document
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from api.pipeline_runner import run_pipeline, PROCESSED_DIR

sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(
    title="Document Ingestion Pipeline API",
    description=(
        "Uploads documents through a full ingestion pipeline: extraction, "
        "OCR (where needed), cleaning, language detection, metadata extraction, "
        "duplicate detection, and three chunking strategies. "
        "Results are persisted as JSON files."
    ),
    version="1.0.0",
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}


@app.get("/")
def root():
    return {
        "message": "Document Ingestion Pipeline API",
        "endpoints": {
            "POST /upload": "Upload and process a document",
            "GET /documents": "List all processed documents",
            "GET /documents/{filename}": "Retrieve results for a specific document",
        },
    }


@app.post("/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    """Upload one or more documents and run the full ingestion pipeline
    on each. Accepts PDF, DOCX, HTML/HTM files.

    Saves two output files per document:
      - processed_documents/<filename>.json  (full structured results)
      - processed_documents/<filename>.txt   (cleaned plain text only)

    Returns a summary per file. Use GET /documents/{filename} for full
    chunk data.
    """
    results = []
    
    # If the batch is large, override per-language strategy selection
    # and use recursive for all documents -- semantic chunking at 18s/doc
    # makes large batches impractical (20 docs = ~6 mins). Recursive is
    # the benchmark-recommended strategy for high-throughput scenarios.
    BATCH_STRATEGY_OVERRIDE_THRESHOLD = 10
    batch_strategy_override = "recursive" if len(files) >= BATCH_STRATEGY_OVERRIDE_THRESHOLD else None

    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": f"Unsupported file type '{file_ext}'. Accepted: {sorted(SUPPORTED_EXTENSIONS)}",
            })
            continue

        with tempfile.NamedTemporaryFile(
            suffix=file_ext, delete=False, dir="."
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            result = run_pipeline(str(tmp_path), original_filename=file.filename, strategy_override=batch_strategy_override)

            original_stem = Path(file.filename).stem

            # Rename both output files to match the original filename
            for ext in (".json", ".txt"):
                auto_named = PROCESSED_DIR / f"{tmp_path.stem}{ext}"
                final_named = PROCESSED_DIR / f"{original_stem}{ext}"
                if auto_named.exists():
                    auto_named.rename(final_named)

            results.append({
                "filename": file.filename,
                "status": "success",
                "saved_to": {
                    "json": str(PROCESSED_DIR / f"{Path(file.filename).stem}.json"),
                    "txt": str(PROCESSED_DIR / f"{Path(file.filename).stem}.txt"),
                },
                "summary": {
                    "file_type": result["file_type"],
                    "title": result["title"],
                    "primary_language": result["primary_language"],
                    "languages": result["languages"],
                    "word_count": result["word_count"],
                    "char_count": result["char_count"],
                    "is_duplicate": result["is_duplicate"],
                    "duplicate_of": result["duplicate_of"],
                    "extraction_info": result["extraction_info"],
                    "chunking_strategy_used": result["chunking_strategy_used"],
                    "strategy_selection_reason": "batch_threshold_override" if batch_strategy_override else f"language_based ({result['primary_language']})",
                    "chunks": {
                        "strategy": result["chunks"].get("strategy"),
                        "num_chunks": result["chunks"].get("num_chunks", 0),
                        "avg_token_count": result["chunks"].get("avg_token_count", 0),
                    },
                },
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": str(e),
            })

        finally:
            tmp_path.unlink(missing_ok=True)

    total = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = total - succeeded

    return JSONResponse({
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    })


@app.get("/documents")
def list_documents():
    """List all processed documents with their metadata.

    Returns lightweight records -- filename, language, word count,
    duplicate status -- without the full chunk data.
    """
    PROCESSED_DIR.mkdir(exist_ok=True)
    records = []
    for json_file in sorted(PROCESSED_DIR.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            records.append({
                "filename": data.get("filename"),
                "file_type": data.get("file_type"),
                "title": data.get("title"),
                "primary_language": data.get("primary_language"),
                "languages": data.get("languages"),
                "word_count": data.get("word_count"),
                "is_duplicate": data.get("is_duplicate"),
                "duplicate_of": data.get("duplicate_of"),
            })
        except Exception:
            continue
    return {"total": len(records), "documents": records}


@app.get("/documents/{filename}")
def get_document(filename: str):
    """Retrieve the full pipeline results for a processed document.

    {filename} should be the original filename without extension,
    e.g. /documents/my_report for my_report.pdf.
    Includes full chunk text for all three strategies.
    """
    json_path = PROCESSED_DIR / f"{filename}.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No processed document found for '{filename}'. "
                   f"Upload it first via POST /upload.",
        )
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {e}")