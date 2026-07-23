# Advanced RAG System

An end-to-end Retrieval-Augmented Generation (RAG) API that combines BM25 lexical search, FAISS vector search, optional graph search, reranking, and cited answer generation with language detection and chat history.

## What It Does

- Loads the heavy retrieval models and indexes once at startup.
- Accepts a query, rewrites it, expands it into multiple variants, retrieves candidates, reranks them, and generates a cited answer.
- Includes a security layer in `guardrails/` that screens for prompt injection, jailbreak attempts, PII leakage, data exfiltration, and toxicity before and after answer generation.
- Supports Urdu TTS and STT endpoints.
- Supports English and Korean at the client layer, while the API's language-aware pipeline also handles Urdu.
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
- `POST /tts/urdu` - synthesizes Urdu speech from text and returns `audio/mpeg`.
- `POST /stt/urdu` - transcribes Urdu audio uploads with Whisper.
- `POST /reindex/partial` - indexes only new chunks that are not already in FAISS, BM25, or the graph.
- `POST /reindex/full` - rebuilds all indexes from scratch.
- `GET /reindex/status` - shows how many corpus chunks are still pending indexing.
- `GET /health` - simple health check.
- `GET /stats` - returns the current indexed chunk count.

## Architecture

The request flow now looks like this:

```text
User Query
	↓
Language Detection (langdetect)
	↓
Input Guardrail (check_input_deep)
	├── Injection (Prompt Guard 2)
	├── Jailbreak Roleplay (regex)
	├── Exfiltration (regex)
	├── Toxicity (model + regex supplement)
	└── PII (Presidio + custom regex)
		↓ (if blocked)
		✗ Refusal Message
	↓ (if passes)
Retrieval Pipeline
	├── Multi-query expansion
	├── Hybrid search (FAISS + BM25)
	└── Reranking
	↓
Answer Generation (LLM with citations)
	↓
Output Guardrail (check_output)
	├── Exfiltration
	└── PII
	↓ (if blocked)
	✗ Refusal Message
	↓ (if passes)
Return Response
```

The retrieval pipeline itself is:

1. Query rewrite
2. Multi-query generation
3. BM25 + vector hybrid search
4. Optional graph search
5. Candidate merge and filtering
6. Cross-encoder reranking
7. Citation-based answer generation

The guardrails module wraps the request on both the input and output sides, blocking prompt injection, jailbreak attempts, PII leakage, data exfiltration, and toxicity.

The TTS and STT routes are Urdu-specific on the server side; English and Korean language support is handled by the client application.

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