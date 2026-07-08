"""
bootstrap.py — loads project 2's corpus + bge-m3 embeddings, builds a FAISS
index in-memory. This is the single source of truth for retrieval state;
call get_index() once at API startup, not per-request.
"""
import sys
from pathlib import Path

# Make proj-emb-vec's src/ importable, same way their own scripts do it
PROJ2_SRC = Path(__file__).resolve().parent.parent.parent / "proj-emb-vec" / "src"
if str(PROJ2_SRC) not in sys.path:
    sys.path.insert(0, str(PROJ2_SRC))

from loader import load_corpus          # project 2's corpus loader
from emb_store import load_embeddings    # project 2's embedding loader
from db.faiss_db import FaissDB          # project 2's FAISS adapter

_state = {}

def get_index():
    """Lazy singleton: builds once, reused across requests."""
    if "db" not in _state:
        corpus = load_corpus(strategy="semantic")   # your winning strategy
        vecs, ids = load_embeddings("bge-m3")        # your winning model

        db = FaissDB()
        db.build(vecs, ids)

        text_by_id = {c["chunk_id"]: c for c in corpus}

        _state["db"] = db
        _state["text_by_id"] = text_by_id
        _state["vecs"] = vecs
        _state["ids"] = ids

    return _state["db"], _state["text_by_id"]