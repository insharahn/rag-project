"""
bootstrap.py — loads project 2's corpus + bge-m3 embeddings, builds/loads FAISS index
"""
import sys
import json
import pickle
from pathlib import Path

PROJ2_SRC = Path(__file__).resolve().parent.parent.parent / "proj-emb-vec" / "src"
if str(PROJ2_SRC) not in sys.path:
    sys.path.insert(0, str(PROJ2_SRC))

from loader import load_corpus
from emb_store import load_embeddings
from db.faiss_db import FaissDB

# persisted index lives inside advanced-rag-system itself, not proj-emb-vec (week 2)
INDEX_DIR = Path(__file__).resolve().parent.parent / "index_data"
INDEX_DIR.mkdir(exist_ok=True)
FAISS_PATH = INDEX_DIR / "faiss.index"
IDS_PATH = INDEX_DIR / "faiss_ids.pkl"
TEXT_BY_ID_PATH = INDEX_DIR / "text_by_id.pkl"

_state = {}


def _build_fresh():
    """Original behavior: build from phase 2's corpus + bge-m3.npy."""
    corpus = load_corpus(strategy="semantic")
    vecs, ids = load_embeddings("bge-m3")

    db = FaissDB()
    db.build(vecs, ids)

    text_by_id = {c["chunk_id"]: c for c in corpus}
    return db, ids, text_by_id


def _save_state(db, ids, text_by_id):
    import faiss
    faiss.write_index(db.index, str(FAISS_PATH))
    with open(IDS_PATH, "wb") as f:
        pickle.dump(ids, f)
    with open(TEXT_BY_ID_PATH, "wb") as f:
        pickle.dump(text_by_id, f)
    print(f"[bootstrap] persisted index -> {FAISS_PATH} ({len(ids)} vectors)")


def _load_persisted():
    import faiss
    index = faiss.read_index(str(FAISS_PATH))
    with open(IDS_PATH, "rb") as f:
        ids = pickle.load(f)
    with open(TEXT_BY_ID_PATH, "rb") as f:
        text_by_id = pickle.load(f)

    db = FaissDB()
    db.index = index
    db.ids = ids
    return db, ids, text_by_id


def get_index():
    """Lazy singleton: loads persisted index if present, otherwise builds
    fresh from phase 2's corpus and persists it for next time."""
    if "db" not in _state:
        if FAISS_PATH.exists() and IDS_PATH.exists() and TEXT_BY_ID_PATH.exists():
            print("[bootstrap] loading persisted FAISS index...")
            db, ids, text_by_id = _load_persisted()
        else:
            print("[bootstrap] no persisted index found, building fresh...")
            db, ids, text_by_id = _build_fresh()
            _save_state(db, ids, text_by_id)

        _state["db"] = db
        _state["ids"] = ids
        _state["text_by_id"] = text_by_id

    return _state["db"], _state["text_by_id"]


def save_current_state():
    """Call this after any mutation (e.g. partial reindex) to persist the
    updated in-memory index back to disk."""
    if "db" not in _state:
        raise RuntimeError("get_index() must be called before save_current_state()")
    _save_state(_state["db"], _state["ids"], _state["text_by_id"])