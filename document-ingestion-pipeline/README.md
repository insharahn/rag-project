# Document Ingestion Pipeline API

A production-ready document processing pipeline for ingesting, extracting, and structuring documents from **PDF, DOCX, and HTML** sources.

It supports OCR for scanned PDFs, language-aware processing, duplicate detection, and multiple chunking strategies exposed through a FastAPI backend and a lightweight web UI.

The same FastAPI gateway also mounts the Week 2 through Week 5 platform UIs for the broader multi-phase demo.

---

## Features

- **Multi-format ingestion**: PDF, DOCX, HTML/HTM
- **OCR with smart routing**:
  - **Tesseract** for English/Korean
  - **EasyOCR** for Urdu (Nastaliq)
- **Text cleaning**: Unicode normalization, whitespace cleanup, hyphenation repair
- **Language detection**: 50+ languages with confidence scores
- **Metadata extraction**: title, author, dates, word/character counts
- **Duplicate detection**:
  - Exact duplicate via SHA-256
  - Near-duplicate via TF-IDF cosine similarity
- **Chunking strategies**:
  - Fixed-size
  - Recursive
  - Semantic (language-optimized)

---

## Architecture

```text
Upload → Extraction → Cleaning → Language Detection →
Metadata → Deduplication → Chunking → Persistence
```

### Design highlights

- **Language-based chunking strategy**
  - English → Recursive
  - Urdu/Korean/French/Arabic → Semantic
  - Others → Recursive fallback
- **OCR routing by script**
  - Latin/Korean scripts → Tesseract
  - Urdu → EasyOCR
- **Batch optimization**
  - Large batches (10+ files) can auto-switch to recursive chunking for throughput
- **Thread-safe PDF handling**
  - Pages are rendered before threading to avoid PyMuPDF conflicts
- **Original filename preservation**
  - Works correctly even when uploads use temporary file paths

---

## Project Structure

```text
.
├── api/                    # FastAPI endpoints
│   └── pipeline_runner.py  # Pipeline orchestrator
├── pipeline/               # Core processing modules
│   ├── extract_pdf.py      # PDF extraction + OCR routing w/ Tessaract + EasyOCR
│   ├── extract_docx.py     # DOCX extraction
│   ├── extract_html.py     # HTML extraction
│   ├── clean.py            # Text cleaning
│   ├── detect_language.py  # Language detection
│   ├── metadata.py         # Metadata extraction
│   ├── dedup.py            # Exact + near-duplicate detection
│   ├── chunk_fixed.py      # Fixed-size chunking
│   ├── chunk_recursive.py  # Recursive chunking
│   └── chunk_semantic.py   # Semantic chunking
├── benchmark/
│   └── run_benchmark.py    # Chunking strategy benchmarks
├── processed_documents/    # Output storage
├── ui/                     # Static web UI
└── scripts/                # Utility scripts
```

---

## API Endpoints

- `POST /upload` — Upload and process one or more documents
- `GET /documents` — List processed documents (metadata)
- `GET /documents/{filename}` — Retrieve full processed output
- `GET /ui` — Simple web interface for testing/demo
- `GET /week2` — Retrieval benchmark dashboard UI
- `GET /week3` — Phase 3 RAG chat UI
- `GET /week4` — Phase 4 guardrails metrics dashboard
- `GET /week5` — Phase 5 multi-agent RAG chat UI

---

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet present, install core packages from the dependency list in this repo.

### 2) Run the API

```bash
uvicorn api.pipeline_runner:app --reload
```

Default local URL: `http://127.0.0.1:8000`

### 3) Try the pipeline

- Open API docs: `http://127.0.0.1:8000/docs`
- Or open the web UI (if mounted): `http://127.0.0.1:8000/ui`
- Upload a document via `POST /upload`

The same gateway also exposes the other platform UIs at `http://127.0.0.1:8000/week2`, `http://127.0.0.1:8000/week3`, `http://127.0.0.1:8000/week4`, and `http://127.0.0.1:8000/week5`.

---

## OCR Setup Notes

### Tesseract (required for scanned PDF OCR)

Install Tesseract and language packs:

- `eng.traineddata`
- `kor.traineddata`
- `urd.traineddata`

> On Windows, ensure the Tesseract install path is available to `pytesseract` (PATH or explicit config).

### EasyOCR

- EasyOCR downloads model weights on first run (internet required initially).
- GPU is recommended for speed, but CPU fallback works.

---

## Performance (typical)

- **Recursive chunking**: ~2s/document
- **Semantic chunking**: ~18s/document
- **English OCR confidence**: ~90–95%
- **Urdu OCR confidence**: improved via EasyOCR vs Tesseract baseline

---

## Known Limitations

- Urdu Nastaliq OCR quality remains challenging.
- Semantic chunking is significantly slower than recursive.
- Mixed-script PDFs (page-to-page) may occasionally be routed suboptimally.

---

## Future Improvements

- Better mixed-script OCR routing at page/block level
- Faster semantic chunking (model/runtime optimization)
- Enhanced Urdu OCR post-processing and correction
- Stronger persistence/indexing for large-scale corpora

---

## Core Dependencies

- Python 3.12+
- FastAPI, Uvicorn
- PyMuPDF, python-docx, trafilatura, BeautifulSoup4
- pytesseract, Pillow, EasyOCR
- sentence-transformers, scikit-learn