# Advanced RAG System

An end-to-end Retrieval-Augmented Generation (RAG) API that combines BM25 lexical search, FAISS vector search, optional graph search, reranking, and cited answer generation with language detection and chat history.

## What It Does

- Loads the heavy retrieval models and indexes once at startup.
- Accepts a query, rewrites it, expands it into multiple variants, retrieves candidates, reranks them, and generates a cited answer.
- Supports English, Korean, and Urdu queries.
- Includes a security layer in `guardrails/` that screens for prompt injection, jailbreak attempts, PII leakage, data exfiltration, and toxicity.
- Records chat history. 
- Exposes partial and full reindex endpoints for keeping the indexes in sync with new corpus chunks.

## Quick Start

Create and activate the virtual environment, then install dependencies:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the API:

```powershell
uvicorn api.main:app --reload --port 8001
```

## Main Endpoints

- `POST /query` - runs retrieval and returns a grounded answer with citations.
- `POST /reindex/partial` - indexes only new chunks that are not already in FAISS, BM25, or the graph.
- `POST /reindex/full` - rebuilds all indexes from scratch.
- `GET /reindex/status` - shows how many corpus chunks are still pending indexing.
- `GET /health` - simple health check.
- `GET /stats` - returns the current indexed chunk count.

## Architecture

The retrieval flow is:

1. Query rewrite
2. Multi-query generation
3. BM25 + vector hybrid search
4. Optional graph search
5. Candidate merge and filtering
6. Cross-encoder reranking
7. Citation-based answer generation

The request flow also passes through guardrails that block prompt injection, jailbreak attempts, PII leakage, data exfiltration, and toxicity.

## Index Files

Persisted artifacts live in `index_data/`:

- `faiss.index` - vector index
- `faiss_ids.pkl` - FAISS ID order
- `text_by_id.pkl` - chunk metadata
- BM25 and graph state are also persisted by their respective modules

## Notes

- The API expects the upstream corpus and embeddings used by the bootstrap code.
- Graph retrieval is optional and will be skipped if the graph file has not been built yet.
- The server is configured for local development on `http://127.0.0.1:8001`.