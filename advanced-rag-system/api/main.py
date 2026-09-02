"""
main.py — end-to-end RAG API
runs the full pipeline: 
guardrail check -> rewrite -> multi-query -> hybrid search -> rerank ->
cited answer -> guardrail check.
"""
import json
import sys
import time
import numpy as np
import asyncio
import io
import queue
import threading

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
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
from guardrails.guardrail import check_input_deep, check_output
from fastapi.middleware.cors import CORSMiddleware
from agents.workflow import workflow

from config import ALLOWED_ORIGINS, RETRIEVAL_TIMEOUT, GENERATION_TIMEOUT, AGENT_QUERY_TIMEOUT

app = FastAPI(title="Advanced RAG System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_QUERY_LANGUAGES = {"en", "ko", "ur"}

# never reveal which detector fired or why, since that teaches an attacker what to avoid next time.
REFUSAL_MESSAGE = "This request cannot be processed."

#tts/stt for urdu
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


class TTSRequest(BaseModel):
    text: str


@app.post("/tts/urdu")
async def synthesize_urdu(req: TTSRequest):
    from gtts import gTTS
    tts = gTTS(text=req.text, lang='ur')
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


@app.post("/stt/urdu")
async def transcribe_urdu(audio: UploadFile = File(...)):
    import tempfile
    model = get_whisper_model()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="ur")
        return {"text": result["text"].strip()}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        
@app.post("/reindex/partial")
def reindex_partial():
    """Detect and index any chunks in the corpus not yet in FAISS/BM25/graph.
    Diffs the full corpus against what's currently indexed."""
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

    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    new_vecs = model.encode([c["text"] for c in new_chunks], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    db.index.add(new_vecs)
    db.ids.extend([c["chunk_id"] for c in new_chunks])
    for c in new_chunks:
        text_by_id[c["chunk_id"]] = c
    save_faiss()

    new_bm25, new_ids = _build_fresh(full_corpus)
    bm25_module._state["bm25"] = new_bm25
    bm25_module._state["ids"] = new_ids
    save_bm25()

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
    """Rebuild FAISS, BM25, and the graph from scratch using the FULL
    current corpus (all documents, old and new)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from retrieval.bootstrap import _build_fresh as faiss_build_fresh, _save_state as faiss_save_state, _state as faiss_state
    from retrieval.bm25_index import _build_fresh as bm25_build_fresh, _save_state as bm25_save_state, _state as bm25_state
    from graphs.build_graph import build_graph
    from graphs.graph_index import save_graph, get_graph

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "proj-emb-vec" / "src"))
    from loader import load_corpus

    t0 = time.time()
    full_corpus = load_corpus(strategy="semantic")

    from db.faiss_db import FaissDB
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    texts = [c["text"] for c in full_corpus]
    ids = [c["chunk_id"] for c in full_corpus]
    all_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    db = FaissDB()
    db.build(all_vecs, ids)
    text_by_id = {c["chunk_id"]: c for c in full_corpus}

    faiss_state["db"] = db
    faiss_state["ids"] = ids
    faiss_state["text_by_id"] = text_by_id
    faiss_save_state(db, ids, text_by_id)

    new_bm25, bm25_ids = bm25_build_fresh(full_corpus)
    bm25_state["bm25"] = new_bm25
    bm25_state["ids"] = bm25_ids
    bm25_save_state(new_bm25, bm25_ids)

    new_graph = build_graph(full_corpus)
    save_graph(new_graph)

    elapsed = time.time() - t0
    return {
        "status": "ok",
        "total_chunks": len(full_corpus),
        "elapsed_seconds": round(elapsed, 2),
        "note": "Full rebuild complete. All three indices (FAISS, BM25, graph) were reconstructed from the entire current corpus.",
    }


@app.get("/reindex/status")
def reindex_status():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "proj-emb-vec" / "src"))
    from loader import load_corpus

    db, _ = get_index()
    current_ids = set(db.ids)
    full_corpus = load_corpus(strategy="semantic")
    new_ids = [c["chunk_id"] for c in full_corpus if c["chunk_id"] not in current_ids]

    return {"pending_new_chunks": len(new_ids), "total_indexed": len(current_ids)}


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
    try:
        from graphs.graph_index import get_graph
        get_graph()
        print("[startup] graph loaded.")
    except FileNotFoundError:
        print("[startup] graph not found: run python graphs/build_graph.py to enable graph retrieval.")
    # guardrail models (Prompt Guard 2, toxicity classifier)
    print("[startup] loading guardrail models...")
    from guardrails.guardrail import check_input_deep as _warm
    print("[startup] guardrail ready.")
    print("[startup] loading Whisper for Urdu STT...")
    get_whisper_model()
    print("[startup] ready.")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    history: list[dict] = []


class QueryResponse(BaseModel):
    answer: str
    sources: dict
    confidence: str
    top_score: float
    latency_seconds: float
    followup_questions: list[str] = []


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    t0 = time.time()

    # --- language detection ---
    detected_lang = "en"  # default fallback
    try:
        if len(req.query.strip()) >= 20:
            detected = detect(req.query)
            if detected not in SUPPORTED_QUERY_LANGUAGES:
                return QueryResponse(
                    answer=f"This system only supports queries in English, Korean, or Urdu. Detected language: {detected}.",
                    sources={}, confidence="low", top_score=0.0,
                    latency_seconds=round(time.time() - t0, 2),
                )
            detected_lang = detected
    except LangDetectException:
        pass  # detection failed, let the query through rather than blocking valid input

    # --- guardrail: input check, before any retrieval/generation ---
    guard_result = await asyncio.to_thread(check_input_deep, req.query, detected_lang)
    if guard_result.blocked:
        print(f"[guardrail] blocked input — reasons={guard_result.reasons}")
        return QueryResponse(
            answer=REFUSAL_MESSAGE,
            sources={}, confidence="low", top_score=0.0,
            latency_seconds=round(time.time() - t0, 2),
        )

    try:
        chunks = await asyncio.wait_for(
            asyncio.to_thread(retrieve, req.query, req.top_k, 10, req.history),
            timeout=RETRIEVAL_TIMEOUT,
        )
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_answer, req.query, chunks),
            timeout=GENERATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return QueryResponse(
            answer="Request timed out — the LLM provider may be experiencing high load. Please try again.",
            sources={}, confidence="low", top_score=0.0,
            latency_seconds=round(time.time() - t0, 2),
        )

    # --- guardrail: output check, before the answer is returned ---
    output_guard = await asyncio.to_thread(check_output, result["answer"], detected_lang)
    if output_guard.blocked:
        print(f"[guardrail] blocked output — reasons={output_guard.reasons}")
        return QueryResponse(
            answer=REFUSAL_MESSAGE,
            sources={}, confidence="low", top_score=0.0,
            latency_seconds=round(time.time() - t0, 2),
        )

    t1 = time.time()

    return QueryResponse(
        answer=result["answer"], sources=result["sources"],
        confidence=result["confidence"], top_score=result["top_score"],
        latency_seconds=round(t1 - t0, 2),
        followup_questions=result.get("followup_questions", []),
    )


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    _, text_by_id = get_index()
    return {"total_chunks": len(text_by_id), "status": "ready"}

#agents
class AgentQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    history: list[dict] = []


class AgentQueryResponse(BaseModel):
    answer: str
    sources: dict
    confidence: str
    top_score: float
    latency_seconds: float
    followup_questions: list[str] = []
    blocked: bool = False
    research_expanded: bool = False
    sub_queries_used: list[str] | None = None
    validated: bool = False
    validation_issues: str = ""
    retried: bool = False

def _summarize_node_output(node_name: str, output: dict) -> dict:
    """Trim each node's full state dict down to just what the UI trace
    needs — avoids pushing full chunk text over SSE."""
    if node_name == "security_input":
        return {"blocked": output.get("input_blocked", False)}
    if node_name in ("blocked_input", "blocked_output"):
        return {"blocked": True, "final_answer": output.get("final_answer", "")}
    if node_name == "retrieval":
        return {"top_score": output.get("top_score", 0.0)}
    if node_name == "research":
        return {
            "research_expanded": output.get("research_expanded", False),
            "sub_queries_used": output.get("sub_queries_used"),
        }
    if node_name == "summarization":
        return {"retried": output.get("_retry_pass", False)}
    if node_name == "validation":
        return {
            "validation_passed": output.get("validation_passed", False),
            "validation_issues": output.get("validation_issues", ""),
        }
    if node_name == "mark_retry":
        return {"retrying": True}
    if node_name == "security_output":
        return {"blocked": output.get("output_blocked", False)}
    if node_name == "finalize":
        return {
            "final_answer": output.get("final_answer", ""),
            "final_sources": output.get("final_sources", {}),
            "final_confidence": output.get("final_confidence", "low"),
            "final_top_score": output.get("final_top_score", 0.0),
            "final_followups": output.get("final_followups", []),
            "final_validated": output.get("final_validated", False),
            "final_validation_issues": output.get("final_validation_issues", ""),
            "final_research_expanded": output.get("final_research_expanded", False),
        }
    return {}


def _stream_workflow(initial_state: dict, q: queue.Queue):
    try:
        t_prev = time.time()
        for update in workflow.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in update.items():
                now = time.time()
                q.put({
                    "node": node_name,
                    "node_elapsed_seconds": round(now - t_prev, 2),
                    **_summarize_node_output(node_name, node_output),
                })
                t_prev = now
    except Exception as e:
        import traceback
        traceback.print_exc()  
        q.put({"node": "error", "error": str(e)})
    finally:
        q.put(None)
        
@app.post("/agent-query/stream")
async def agent_query_stream(req: AgentQueryRequest):
    t0 = time.time()

    detected_lang = "en"
    try:
        if len(req.query.strip()) >= 20:
            detected = detect(req.query)
            if detected not in SUPPORTED_QUERY_LANGUAGES:
                async def _lang_error():
                    payload = json.dumps({
                        "node": "error",
                        "error": f"This system only supports English, Korean, or Urdu. Detected: {detected}.",
                    })
                    yield f"data: {payload}\n\n"
                return StreamingResponse(_lang_error(), media_type="text/event-stream")
            detected_lang = detected
    except LangDetectException:
        pass

    initial_state = {
        "query": req.query,
        "language": detected_lang,
        "top_k": req.top_k,
        "history": req.history,
    }

    q: queue.Queue = queue.Queue()
    threading.Thread(target=_stream_workflow, args=(initial_state, q), daemon=True).start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                payload = json.dumps({"node": "done", "elapsed_seconds": round(time.time() - t0, 2)})
                yield f"data: {payload}\n\n"
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/agent-query", response_model=AgentQueryResponse)
async def agent_query(req: AgentQueryRequest):
    t0 = time.time()

    # same language gate as /query, since the security nodes read
    # state["language"] and everything downstream assumes it's set
    detected_lang = "en"
    try:
        if len(req.query.strip()) >= 20:
            detected = detect(req.query)
            if detected not in SUPPORTED_QUERY_LANGUAGES:
                return AgentQueryResponse(
                    answer=f"This system only supports queries in English, Korean, or Urdu. Detected language: {detected}.",
                    sources={}, confidence="low", top_score=0.0,
                    latency_seconds=round(time.time() - t0, 2),
                )
            detected_lang = detected
    except LangDetectException:
        pass  # let it through rather than block on a failed detection

    initial_state = {
        "query": req.query,
        "language": detected_lang,
        "top_k": req.top_k,
        "history": req.history,
    }

    try:
        # wider timeout than /query — a research-expansion + validation-retry
        # run can hit 6-7 LLM calls, some with large context
        final_state = await asyncio.wait_for(
            asyncio.to_thread(workflow.invoke, initial_state),
            timeout=AGENT_QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return AgentQueryResponse(
            answer="Request timed out: the LLM provider may be experiencing high load. Please try again.",
            sources={}, confidence="low", top_score=0.0,
            latency_seconds=round(time.time() - t0, 2),
        )

    t1 = time.time()
    print(json.dumps({k: v for k, v in final_state.items() if k not in ("retrieved_chunks", "draft_sources")}, default=str, indent=2))

    return AgentQueryResponse(
        answer=final_state.get("final_answer", ""),
        sources=final_state.get("final_sources", {}),
        confidence=final_state.get("final_confidence", "low"),
        top_score=final_state.get("final_top_score", 0.0),
        latency_seconds=round(t1 - t0, 2),
        followup_questions=final_state.get("final_followups", []),
        blocked=final_state.get("blocked", False),
        research_expanded=final_state.get("final_research_expanded", False),
        sub_queries_used=final_state.get("sub_queries_used"),
        validated=final_state.get("final_validated", False),
        validation_issues=final_state.get("final_validation_issues", ""),
        retried=final_state.get("_retry_pass", False),
    )