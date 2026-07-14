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
def reindex_partial():
    """Detect and index any chunks in the corpus not yet in FAISS/BM25/graph.
    Self-detecting — no need to specify which document is new; diffs the
    full corpus against what's currently indexed."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from retrieval.bootstrap import save_current_state as save_faiss
    from retrieval.bm25_index import _build_fresh, save_current_state as save_bm25
    from retrieval import bm25_index as bm25_module
    from graphs.build_graph import extract_entities
    from graphs.graph_index import get_graph, save_graph

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "proj-emb-vec" / "src"))
    from loader import load_corpus

    db, text_by_id = get_index()
    current_ids = set(db.ids)
    full_corpus = load_corpus(strategy="semantic")
    new_chunks = [c for c in full_corpus if c["chunk_id"] not in current_ids]

    if not new_chunks:
        return {"status": "up to date", "new_chunks_indexed": 0, "total_chunks": len(db.ids)}

    # FAISS
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    new_vecs = model.encode([c["text"] for c in new_chunks], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    db.index.add(new_vecs)
    db.ids.extend([c["chunk_id"] for c in new_chunks])
    for c in new_chunks:
        text_by_id[c["chunk_id"]] = c
    save_faiss()

    # BM25
    new_bm25, new_ids = _build_fresh(full_corpus)
    bm25_module._state["bm25"] = new_bm25
    bm25_module._state["ids"] = new_ids
    save_bm25()

    # Graph
    graph = get_graph()
    for c in new_chunks:
        cid = c["chunk_id"]
        ents = extract_entities(c["text"], c.get("language"))
        for ent in ents:
            graph["entity_to_chunks"].setdefault(ent, set()).add(cid)
        graph["chunk_to_entities"][cid] = ents
        ent_list = list(ents)
        for i, e1 in enumerate(ent_list):
            for e2 in ent_list[i+1:]:
                graph["co_occurrence"].setdefault(e1, set()).add(e2)
                graph["co_occurrence"].setdefault(e2, set()).add(e1)
    save_graph(graph)

    return {
        "status": "ok",
        "new_chunks_indexed": len(new_chunks),
        "new_chunk_ids": [c["chunk_id"] for c in new_chunks],
        "total_chunks": len(db.ids),
    }


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