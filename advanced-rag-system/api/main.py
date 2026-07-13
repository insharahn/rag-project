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
from langdetect import detect, LangDetectException


from retrieval.bootstrap import get_index
from retrieval.bm25_index import get_bm25_index
from retrieval.rerank import _get_reranker
from retrieval.vector_search import _get_model as _get_embed_model
from retrieval.pipeline import retrieve
from generation.citation_generator import generate_answer

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Advanced RAG System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_QUERY_LANGUAGES = {"en", "ko", "ur"}


@app.on_event("startup")
def load_everything():
    print("[startup] loading corpus + FAISS index...")
    get_index()
    print("[startup] loading BM25 index...")
    get_bm25_index()
    print("[startup] loading embedding model...")
    _get_embed_model()
    print("[startup] loading reranker...")
    _get_reranker()
    # graph (skipped if not built yet)
    try:
        from graphs.graph_index import get_graph
        get_graph()
        print("[startup] graph loaded.")
    except FileNotFoundError:
        print("[startup] graph not found — run python graphs/build_graph.py to enable graph retrieval.")
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
     # Language gate — only English, Korean, Urdu are supported
    try:
        if len(req.query.strip()) >= 20:  # langdetect is unreliable on very short strings
            detected = detect(req.query)
            if detected not in SUPPORTED_QUERY_LANGUAGES:
                return QueryResponse(
                    answer=f"This system only supports queries in English, Korean, or Urdu. Detected language: {detected}.",
                    sources={}, confidence="low", top_score=0.0,
                    latency_seconds=0.0,
                )
    except LangDetectException:
        pass  # detection failed, let the query through rather than blocking valid input

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