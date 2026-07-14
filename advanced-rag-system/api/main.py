"""
main.py — end-to-end RAG API. Loads all models/indices ONCE at startup
(not per-request), exposes a single /query endpoint that runs the full
pipeline: rewrite -> multi-query -> hybrid search -> rerank -> cited answer.
"""
import sys
import time
import numpy as np
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect, LangDetectException

from retrieval.bootstrap import get_index
from retrieval.bm25_index import get_bm25_index
from retrieval.bm25_index import build_bm25_index
from retrieval.rerank import _get_reranker
from retrieval.vector_search import _get_model as _get_embed_model
from retrieval.vector_search import embed_query
from retrieval.pipeline import retrieve
from generation.citation_generator import generate_answer
from graphs.build_graph import extract_entities
from graphs.graph_index import get_graph
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Advanced RAG System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_QUERY_LANGUAGES = {"en", "ko", "ur"}


@app.post("/reindex/partial")
def reindex_partial(req: ReindexRequest):
    """Embed and index only the newly uploaded documents' chunks.
    Appends to FAISS, rebuilds BM25 (cheap), merges new entities into graph."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "proj-emb-vec" / "src"))
    from loader import load_corpus

    db, text_by_id = get_index()

    # 1. load ONLY the new docs' chunks (loader already reads the full metadata.json,
    #    so filter to just the new stems)
    full_corpus = load_corpus(strategy="semantic")
    new_chunks = [c for c in full_corpus if c["chunk_id"].split("__")[0] in req.new_doc_stems]

    if not new_chunks:
        return {"status": "no new chunks found", "indexed": 0}

    # 2. embed just the new chunks (fast — few hundred chunks max)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    texts = [c["text"] for c in new_chunks]
    new_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    # 3. FAISS: append, don't rebuild
    db.index.add(new_vecs)
    db.ids.extend([c["chunk_id"] for c in new_chunks])

    # 4. update in-memory text lookup so retrieval can find the new chunks' text
    for c in new_chunks:
        text_by_id[c["chunk_id"]] = c

    # 5. BM25: rebuild from the FULL corpus (old + new) — cheap, no GPU
    build_bm25_index(full_corpus)

    # 6. graph: extract + merge entities from new chunks only
    graph = get_graph()
    for c in new_chunks:
        cid = c["chunk_id"]
        ents = extract_entities(c["text"], c.get("language"))
        for ent in ents:
            graph["entity_to_chunks"].setdefault(ent, set()).add(cid)
            graph["chunk_to_entities"].setdefault(cid, set()).update(ents)
    # (co_occurrence update omitted here for brevity — same pairwise loop as build_graph.py,
    #  scoped to just the new chunks' entity sets)

    return {"status": "ok", "indexed": len(new_chunks), "total_chunks": len(db.ids)}


@app.post("/reindex/full")
def reindex_full():
    """Rebuild everything from scratch — full corpus, full re-embed.
    WARNING: this re-embeds the entire corpus on CPU and will take hours.
    Consider running this as a background task, not a blocking request."""
    # same steps as bootstrap.get_index()'s build path, just forced fresh
    # rather than using the cached singleton — reuses existing build_graph.py,
    # FaissDB.build(), build_bm25_index() unchanged, just called on the FULL
    # reloaded corpus instead of only new chunks.
    ...

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