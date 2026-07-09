"""
main.py — end-to-end RAG API. Loads all models/indices ONCE at startup
(not per-request), exposes a single /query endpoint that runs the full
pipeline: rewrite -> multi-query -> hybrid search -> rerank -> cited answer.
"""
import sys
import time
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.bootstrap import get_index
from retrieval.bm25_index import get_bm25_index
from retrieval.rerank import _get_reranker
from retrieval.vector_search import _get_model as _get_embed_model
from retrieval.pipeline import retrieve
from generation.citation_generator import generate_answer

app = FastAPI(title="Advanced RAG System", version="1.0")


@app.on_event("startup")
def load_everything():
    """Warm every component once, so no request pays index-build or
    model-load cost. This is the fix validated in test_warm_latency.py —
    warm local components are fast; only LLM calls remain variable."""
    print("[startup] loading corpus + FAISS index...")
    get_index()
    print("[startup] loading BM25 index...")
    get_bm25_index()
    print("[startup] loading embedding model...")
    _get_embed_model()
    print("[startup] loading reranker...")
    _get_reranker()
    print("[startup] ready.")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: dict
    confidence: str
    top_score: float
    latency_seconds: float


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    t0 = time.time()
    try:
        chunks = await asyncio.wait_for(
            asyncio.to_thread(retrieve, req.query, req.top_k),
            timeout=120,
        )
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_answer, req.query, chunks),
            timeout=60,
        )
    except asyncio.TimeoutError:
        return QueryResponse(
            answer="Request timed out — the LLM provider may be experiencing high load. Please try again.",
            sources={}, confidence="low", top_score=0.0,
            latency_seconds=round(time.time() - t0, 2),
        )
    t1 = time.time()

    return QueryResponse(
        answer=result["answer"], sources=result["sources"],
        confidence=result["confidence"], top_score=result["top_score"],
        latency_seconds=round(t1 - t0, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    _, text_by_id = get_index()
    return {"total_chunks": len(text_by_id), "status": "ready"}