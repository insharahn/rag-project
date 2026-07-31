# Advanced RAG System

An end-to-end Retrieval-Augmented Generation (RAG) API that combines BM25 lexical search, FAISS vector search, optional graph search, reranking, cited answer generation, language detection, chat history, and a separate multi-agent workflow built with LangGraph.

## What It Does

- Loads the heavy retrieval models and indexes once at startup.
- Accepts a query, rewrites it, expands it into multiple variants, retrieves candidates, reranks them, and generates a cited answer.
- Includes a security layer in `guardrails/` that screens for prompt injection, jailbreak attempts, PII leakage, data exfiltration, and toxicity before and after answer generation.
- Supports Urdu TTS and STT endpoints.
- Supports English and Korean at the client layer, while the API's language-aware pipeline also handles Urdu.
- Records chat history.
- Exposes partial and full reindex endpoints for keeping the indexes in sync with new corpus chunks.
- Includes a separate **multi-agent RAG workflow** with security, retrieval, research expansion, summarization, and validation agents.

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

- `POST /query` - runs the original Phase 3 retrieval pipeline and returns a grounded answer with citations.
- `POST /agent-query` - runs the new LangGraph multi-agent workflow with research expansion and validation.
- `POST /tts/urdu` - synthesizes Urdu speech from text and returns `audio/mpeg`.
- `POST /stt/urdu` - transcribes Urdu audio uploads with Whisper.
- `POST /reindex/partial` - indexes only new chunks that are not already in FAISS, BM25, or the graph.
- `POST /reindex/full` - rebuilds all indexes from scratch.
- `GET /reindex/status` - shows how many corpus chunks are still pending indexing.
- `GET /health` - simple health check.
- `GET /stats` - returns the current indexed chunk count.

## Architecture

The system now has two query paths:

### 1. Original RAG Pipeline
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
    ├── Query rewrite
    ├── Multi-query generation
    ├── Hybrid search (FAISS + BM25)
    ├── Optional graph search
    ├── RRF fusion
    └── Cross-encoder reranking
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

### 2. Multi-Agent Workflow
```text
Start
    ↓
Security Agent
    ↓
Input Blocked?
    ├── Yes → Blocked Input Node
    └── No
        ↓
        Retrieval Agent
            ↓
        Research Agent
            ↓
        Need Expansion?
            ├── Yes → Query Decomposition → Retrieve Sub-queries → Merge Results
            └── No
                ↓
        Summarization Agent
            ↓
        Validation Agent
            ↓
        Validation Passed?
            ├── Yes → Security Output Agent → Finalize Node
            ├── No - First Failure → Mark Retry → Summarization Agent
            └── No - Second Failure → Hedge Node → Security Output Agent
```

## Multi-Agent Details

The `/agent-query` endpoint uses a five-agent LangGraph pipeline:

1. **Security Agent**
   - Reuses the existing guardrail logic.
   - Screens both input and output.
   - Blocks unsafe requests before retrieval begins.

2. **Retrieval Agent**
   - Wraps the existing retrieval pipeline.
   - Runs query rewrite, multi-query generation, hybrid search, graph search, RRF fusion, and reranking.

3. **Research Agent**
   - Triggered only when retrieval confidence is low.
   - Decomposes the query into declarative sub-queries.
   - Re-retrieves and merges results.

4. **Summarization Agent**
   - Generates a draft answer from retrieved chunks.
   - Reuses the citation-grounded answer generation logic.
   - Can incorporate validator feedback on retry.

5. **Validation Agent**
   - Checks whether the draft is grounded in the retrieved sources.
   - Verifies citation correctness.
   - Confirms that the answer actually addresses the query.
   - If validation fails once, the workflow retries summarization with feedback.
   - If validation fails again, the workflow returns a hedge instead of a flawed answer.

## Implemented Agents

### `agents/`
The new agent modules are organized under `agents/` and are responsible for:
- security routing
- retrieval orchestration
- research expansion
- answer summarization
- answer validation

The multi-agent workflow is built for explicit, inspectable control flow rather than hidden router behavior.

## Retrieval Pipeline

The retrieval pipeline itself is:

1. Query rewrite
2. Multi-query generation
3. BM25 + vector hybrid search
4. Optional graph search
5. Candidate merge and filtering
6. Cross-encoder reranking
7. Citation-based answer generation

## Guardrails

The guardrails module wraps the request on both the input and output sides, blocking prompt injection, jailbreak attempts, PII leakage, data exfiltration, and toxicity.

## Index Files

Persisted artifacts live in `index_data/`:

- `faiss.index` - vector index
- `faiss_ids.pkl` - FAISS ID order
- `text_by_id.pkl` - chunk metadata
- BM25 and graph state are also persisted by their respective modules

## Notes

- The API expects the upstream corpus and embeddings used by the bootstrap code.
- Graph retrieval is optional and will be skipped if the graph file has not been built yet.
- `POST /query` and `POST /agent-query` are separate and independently demoable.
- The server is configured for local development on `http://127.0.0.1:8001`.