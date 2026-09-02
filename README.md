# Enterprise Multi-Agent RAG Platform

An end-to-end, six-phase platform covering document ingestion, retrieval
benchmarking, advanced Retrieval-Augmented Generation (RAG), guardrail
evaluation, multi-agent RAG, and fine-tuning/model optimization.

---

## Repository Structure

```text
.
├── document-ingestion-pipeline/   # Phase 1 + unified web/API gateway (port 8000)
├── proj-emb-vec/                  # Phase 2: embedding + retrieval benchmarking
└── advanced-rag-system/           # Phase 3-6: RAG, guardrails, multi-agent, fine-tuning (port 8001)
```

Each subproject has its own `docs/`, `requirements.txt`, and README with
setup details specific to that piece — see the links in each phase below.

---

## Setup

Each subproject is installed and run independently.

```bash
# from repo root
cd document-ingestion-pipeline
pip install -r requirements.txt

cd ../proj-emb-vec
pip install -r requirements.txt

cd ../advanced-rag-system
pip install -r requirements.txt
```

You'll need API keys for the LLM providers used across the pipeline — set
these in a `.env` file inside `advanced-rag-system/` (see
`advanced-rag-system/config.py` for the full list of expected variables).
At minimum, an OpenRouter and/or Groq key is required for query rewriting,
generation, and the multi-agent workflow. Note: on OpenRouter, only the
`openrouter/free` auto-router model has proven reliable for this pipeline —
other free-tier models have hung in testing.

Groq's free-tier token-per-day limit is a known, documented constraint —
multi-agent queries (`/agent-query`) consume meaningfully more tokens per
query than the single-pass `/query` pipeline, so expect to hit rate limits
faster when testing the Week 5 flow.

---

## Phase 1 — Document Ingestion Pipeline

Processes PDF, DOCX, and HTML/HTM documents through:

- text extraction + OCR routing (Tesseract/EasyOCR, with language-aware
  routing and per-document script detection for Urdu)
- cleaning/normalization
- language detection
- metadata extraction
- duplicate/near-duplicate detection
- chunking (fixed, recursive, semantic)

Outputs structured per-document JSON and cleaned TXT artifacts.

See [document-ingestion-pipeline/README.md](document-ingestion-pipeline/README.md)
for setup and full API details.

---

## Phase 2 — Retrieval Benchmark

Evaluates retrieval performance across embedding models, vector databases,
and chunking strategies using multilingual (English/Korean/Urdu) ground-truth
queries. Benchmarked 5 embedding models × 3 chunking strategies × 3 vector
databases; best-performing combination: semantic chunking + BGE-M3 + FAISS.

Tracks key IR/system metrics including recall@k, precision@k, MRR, latency,
build time, and memory.

See [proj-emb-vec/README.md](proj-emb-vec/README.md) for methodology and
benchmark workflow.

---

## Phase 3 — Advanced RAG System

A production-style RAG API that runs:

1. query rewrite
2. multi-query expansion
3. BM25 + FAISS hybrid retrieval (RRF fusion)
4. graph retrieval
5. candidate merge/filter
6. cross-encoder reranking
7. cited answer generation with confidence gating
8. input/output guardrails
9. optional speech input/output for Urdu

Corpus is live-updatable: FAISS/BM25/entity graph persist to disk, with
`/reindex/partial` (diff-based incremental reindex) and `/reindex/full`
endpoints to keep indexes synced with newly uploaded documents.

Key capabilities:

- multilingual query support (English, Korean, Urdu)
- startup preloading of heavy models/indexes for faster first query
- chat-history-aware querying
- a unified guardrails module screening both queries and generated answers
  for prompt injection, roleplay/jailbreak attempts, data exfiltration,
  toxicity, and PII leakage
- speech-to-text and text-to-speech for Urdu via `/stt/urdu` and
  `/tts/urdu`, with browser-side speech input/output in the Week 3 UI for
  English and Korean

Core endpoints:

- `POST /query`
- `POST /agent-query`
- `POST /tts/urdu`
- `POST /stt/urdu`
- `POST /reindex/partial`
- `POST /reindex/full`
- `GET /reindex/status`
- `GET /health`
- `GET /stats`

See [advanced-rag-system/README.md](advanced-rag-system/README.md) for full
architecture and run instructions.

---

## Phase 4 — Guardrails for RAG

Guardrails are integrated directly into the Phase 3 chat experience,
screening both user input and generated output inside the Week 3 UI. A
separate Week 4 dashboard surfaces guardrail evaluation metrics and deeper
analysis.

Key behaviors:

- input screening for prompt injection, jailbreak attempts, exfiltration,
  toxicity, and PII
- output screening for exfiltration and PII leakage
- fast rule/model-based checks, backed by Prompt Guard 2 and regex
  detectors, with a deeper LLM judge for ambiguous or narrative-style
  jailbreak attempts
- guardrail evaluation results surfaced through the Week 4 metrics dashboard

See [advanced-rag-system/README.md](advanced-rag-system/README.md) for the
guardrail implementation.

---

## Phase 5 — Multi-Agent RAG System

A separate LangGraph-based multi-agent query workflow, exposed through the
Week 5 UI on the same local gateway as the other platform UIs, but backed by
the multi-agent query pipeline instead of the single-pass `/query` flow.

Five agents: security, retrieval, research, summarization, validation.

1. **Security Agent** blocks unsafe requests before retrieval.
2. **Retrieval Agent** runs the core rewrite, multi-query, hybrid search,
   graph search, fusion, and reranking pipeline.
3. **Research Agent** expands low-confidence queries into sub-queries and
   merges additional evidence.
4. **Summarization Agent** drafts a cited answer from retrieved chunks.
5. **Validation Agent** checks grounding, citation correctness, and query
   coverage, then either accepts, retries, or returns a hedge.

Key capabilities:

- separate `/agent-query` and `/agent-query/stream` endpoints alongside the
  original single-pipeline `/query`
- security-first execution with pre-retrieval blocking
- low-confidence query expansion through research sub-queries
- citation-grounded summarization with validation retries and hedge
  responses when needed
- chat history support and the same language-aware behavior as the Phase 3
  chat UI
- feature parity with the Week 3 UI (chat history, Urdu STT/TTS, expandable
  cited-source chunk text)

See [advanced-rag-system/README.md](advanced-rag-system/README.md) for full
multi-agent architecture and implementation details.

---

## Phase 6 — Fine-Tuning and Model Optimization

Fine-tunes open-source models (Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct)
with QLoRA on a RAG-specific objective — improving retrieval faithfulness
and citation quality rather than general instruction-tuning — and compares
base vs. fine-tuned performance on accuracy, hallucination rate, response
quality, faithfulness to retrieved context, and citation correctness.

Training runs on Colab (T4 GPU, free tier); notebooks live under
`advanced-rag-system/notebooks/`.

Key findings:

- Both fine-tuned models improved citation-targeting accuracy over their
  base models, but manual review surfaced real generation defects beyond
  what automated citation-marker checks alone would catch — including
  fabricated facts/statistics and, on one shared query, an empty or
  off-topic answer from both fine-tunes independently.
- Results and a base-vs-fine-tuned comparison dashboard are surfaced through
  the Week 6 dashboard, mounted on the same unified gateway as the other
  weeks.

See `advanced-rag-system/notebooks/finetuning/` for training data, eval
sets, and the fine-tuning report in `advanced-rag-system/docs/`.

---

## Single Unified Interface (Document Ingestion API)

The FastAPI app in
[document-ingestion-pipeline/api/main.py](document-ingestion-pipeline/api/main.py#L1)
acts as the single local web gateway for the full platform UI experience on
port 8000.

It serves:

- `GET /ui` — Week 1 ingestion interface
- `GET /week2` — Week 2 benchmark dashboard UI
- `GET /week3` — Week 3 advanced RAG chat UI
- `GET /week4` — Week 4 guardrails evaluation UI
- `GET /week5` — Week 5 multi-agent RAG chat UI
- `GET /week6` — Week 6 fine-tuning comparison dashboard

Important: the Week 3 and Week 5 pages are hosted by this unified interface,
but call the Advanced RAG backend running separately on port 8001. The Week
4 and Week 6 pages stay on the same gateway and read static results
(guardrail eval JSON, fine-tuning comparison summary JSON) rather than
calling the backend live.

---

## API Services Overview

- `document-ingestion-pipeline` API (default `127.0.0.1:8000`): ingestion
  endpoints + unified UI hosting for all six weeks
- `advanced-rag-system` API (default `127.0.0.1:8001`): the RAG pipeline,
  LangGraph multi-agent `/agent-query` workflow, guardrails, and Urdu
  STT/TTS endpoints

Run both services together for the complete end-to-end demo.

---

## Reports

Project reports for each phase are in the respective `docs/` directory.
