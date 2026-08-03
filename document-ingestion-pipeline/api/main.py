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


#ui (week 1)
from fastapi.staticfiles import StaticFiles
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

#week 2
app.mount("/week2", StaticFiles(directory="ui_week2", html=True), name="week2")

#week 3
app.mount("/week3", StaticFiles(directory="ui_week3", html=True), name="week3")

WEEK2_RESULTS = Path("../proj-emb-vec") 

@app.get("/week2-data/{strategy}")
def get_week2_metrics(strategy: str):
    """Serve project 2's pre-computed metrics.json for a given chunking strategy."""
    path_map = {
        "semantic": WEEK2_RESULTS / "results" / "metrics.json",
        "fixed": WEEK2_RESULTS / "results_fixed" / "metrics.json",
        "recursive": WEEK2_RESULTS / "results_recursive" / "metrics.json",
    }
    p = path_map.get(strategy)
    if not p or not p.exists():
        raise HTTPException(404, f"No metrics found for strategy '{strategy}'")
    return json.loads(p.read_text(encoding="utf-8"))

#week 4
app.mount("/week4", StaticFiles(directory="ui_week4", html=True), name="week4")

WEEK4_RESULTS = Path("../advanced-rag-system/eval/guardrail_deep_results")

@app.get("/week4-data")
def get_week4_metrics():
    """Serve project 4's guardrail eval results (fast+LLM configuration)."""
    metrics_path = WEEK4_RESULTS / "metrics_report.json"
    per_prompt_path = WEEK4_RESULTS / "per_prompt_results.json"

    if not metrics_path.exists():
        raise HTTPException(404, "No guardrail metrics found. Run scripts/run_guardrail_eval.py first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_prompt = json.loads(per_prompt_path.read_text(encoding="utf-8")) if per_prompt_path.exists() else []

    return {"metrics": metrics, "per_prompt": per_prompt}

#week 5
app.mount("/week5", StaticFiles(directory="ui_week5", html=True), name="week5")

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

#week 6
app.mount("/week6", StaticFiles(directory="ui_week6", html=True), name="week6")

WEEK6_RESULTS = Path("../advanced-rag-system/notebooks/finetuning")

@app.get("/week6-data")
def get_week6_metrics():
    """Serve project 6's fine-tuning comparison results."""
    path = WEEK6_RESULTS / "final_comparison_summary.json"
    if not path.exists():
        raise HTTPException(404, "No fine-tuning comparison results found. Run the evaluation notebook first.")
    return json.loads(path.read_text(encoding="utf-8"))

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
    BATCH_STRATEGY_OVERRIDE_THRESHOLD = 100
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
    """List all processed documents from the master metadata file.
    Auto-removes any entry whose per-document JSON no longer exists on disk.
    """
    PROCESSED_DIR.mkdir(exist_ok=True)
    metadata_path = PROCESSED_DIR / "metadata.json"

    if not metadata_path.exists():
        return {"total": 0, "documents": []}

    try:
        master = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {"total": 0, "documents": []}

    # Sync: prune entries where the per-doc JSON was deleted from disk
    stale = [
        fname for fname in master
        if not (PROCESSED_DIR / f"{Path(fname).stem}.json").exists()
    ]
    if stale:
        for fname in stale:
            del master[fname]
        metadata_path.write_text(
            json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {"total": len(master), "documents": list(master.values())}


@app.get("/documents/{filename}")
def get_document(filename: str):
    """Retrieve full results for a processed document.
    Combines metadata (from metadata.json), chunks (from <filename>.json),
    and full text (from <filename>.txt).
    """
    doc_json = PROCESSED_DIR / f"{filename}.json"
    doc_txt  = PROCESSED_DIR / f"{filename}.txt"

    if not doc_json.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No processed document found for '{filename}'. Upload it first via POST /upload.",
        )

    try:
        chunks_data = json.loads(doc_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read chunks: {e}")

    full_text = doc_txt.read_text(encoding="utf-8") if doc_txt.exists() else ""

    # Find metadata by stem match in the master file
    meta = {}
    metadata_path = PROCESSED_DIR / "metadata.json"
    if metadata_path.exists():
        try:
            master = json.loads(metadata_path.read_text(encoding="utf-8"))
            for fname, m in master.items():
                if Path(fname).stem == filename:
                    meta = m
                    break
        except Exception:
            pass

    return {**meta, "full_text": full_text, **chunks_data}