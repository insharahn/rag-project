# Retrieval Benchmark Project

This repository contains the second phase of a document ingestion pipeline: a retrieval benchmarking suite for comparing chunking strategies, embedding models, and vector databases over a multilingual corpus.

The project evaluates pre-processed, chunked text from 63 documents in English, Korean, and Urdu against 45 ground-truth retrieval queries. It measures retrieval quality, latency, index build time, and memory use, then summarizes the results in JSON files and a Streamlit dashboard.

## What This Benchmarks

The benchmark sweeps across:

- 3 chunking strategies: fixed-size, recursive, and semantic
- 5 embedding models: BGE-large-en, BGE-M3, E5-large-v2, multilingual-e5-large, and Instructor-XL
- 3 vector databases: FAISS, Chroma, and Milvus
- 45 evaluation queries with ground-truth answer chunk IDs

It collects:

- recall@k for k = 1, 3, 5, 10
- precision@k for k = 1, 3, 5, 10
- MRR
- search latency percentiles: p50, p95, and p99
- index build time
- memory usage during index construction

## Repository Layout

- `dashboard.py` - Streamlit dashboard for exploring benchmark results
- `scripts/` - benchmark, reporting, validation, and utility scripts
- `src/` - shared loaders and vector database adapters
- `embeddings/` - semantic-strategy corpus embeddings and query embeddings
- `embeddings_fixed/` - fixed-size strategy embeddings and query embeddings
- `embeddings_recursive/` - recursive strategy embeddings and query embeddings
- `eval/`, `eval_fixed/`, `eval_recursive/` - ground-truth evaluation sets per strategy
- `results/`, `results_fixed/`, `results_recursive/` - raw benchmark output and derived metrics
- `notebooks/` - Colab/notebook workflows for embedding generation
- `docs/` - project report assets

## Data And Inputs

The benchmark is built around a multilingual corpus of 63 documents and 45 retrieval queries:

- 22 English queries across 11 documents
- 12 Korean queries across 7 documents
- 11 Urdu queries across 3 documents

Embeddings are generated from an existing corpus and stored as `.npy` files with matching `.meta.json` files. Query embeddings live under each strategy folder in `embeddings*/queries/`.

## End-to-End Workflow

1. Prepare the chunked corpus in the upstream document ingestion pipeline.
2. Generate corpus embeddings for each strategy and model, typically on GPU in Colab for speed and efficiency.
3. Generate query embeddings for each strategy.
4. Remap evaluation ground truth for fixed and recursive chunking strategies.
5. Run the benchmark against FAISS, Chroma, and Milvus.
6. Derive summary metrics from the raw benchmark output.
7. Open the Streamlit dashboard to inspect the results.

## Setup

This project targets Python 3.11 or newer.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running The Dashboard

The dashboard reads the generated `metrics.json` files and visualizes the benchmark results.

```bash
streamlit run dashboard.py
```

If the dashboard reports missing metrics, run the reporting step first.

## Running The Benchmark

Milvus runs in Docker, so start the containers before benchmarking:

```bash
docker compose up -d
python scripts/run_benchmark.py
```

By default, the runner scripts can be narrowed to a subset of strategies through the `STRATEGIES_TO_RUN` and `STRATEGIES_TO_REPORT` lists in the script files. Expand those lists to execute the full sweep.

## Generating Metrics

After `benchmark_raw.json` files exist, derive the summary metrics:

```bash
python scripts/report_metrics.py
```

This writes `metrics.json` into each results folder and powers the dashboard views.

## Useful Scripts

- `scripts/remap_eval.py` - remap evaluation targets for alternate chunking strategies
- `scripts/embed_queries.py` - generate query embeddings for the main models
- `scripts/embed_queries_instructor.py` - generate Instructor-XL query embeddings
- `scripts/check_corpus.py` - validate corpus inputs
- `scripts/check_embeddings.py` - validate embedding artifacts
- `scripts/verify_eval.py` - verify evaluation remapping
- `scripts/test_faiss.py`, `scripts/test_chroma.py`, `scripts/test_milvus.py` - backend-specific checks

## Vector Database Notes

- FAISS is the baseline exact-search backend and is the most reliable reference for retrieval comparisons.
- Chroma is included as a second local vector store backend.
- Milvus runs in Docker and is the most operationally demanding backend; the benchmark intentionally checkpoints after each model to survive long runs.

## Output Files

Each strategy writes two main artifacts:

- `results*/benchmark_raw.json` - raw per-query benchmark data, latencies, build times, and memory usage
- `results*/metrics.json` - aggregated quality and performance metrics consumed by the dashboard

## Implementation Notes

- The shared corpus loader lives in `src/loader.py`.
- Vector database adapters live in `src/db/`.
- The dashboard expects the metrics files already to be present and will stop with an error if they are missing.
- Instructor-XL uses a separate notebook workflow and pinned Colab stack.

## Project Goal

The benchmark findings are used to decide which chunking strategy, embedding model, and vector database should be adopted in production.
