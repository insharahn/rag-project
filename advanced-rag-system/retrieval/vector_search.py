"""
vector_search.py: embeds a query with bge-m3 (same
model/config as week 2's corpus embeddings) and searches the FAISS index
from bootstrap.py. Returns results in the same (chunk_id, score) convention
as bm25_index.py, so fusion can treat both identically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from retrieval.bootstrap import get_index

_state = {}

MODEL_ID = "BAAI/bge-m3"
QUERY_PREFIX = ""  # bge-m3 query prefix is empty


def _get_model():
    if "model" not in _state:
        print(f"[vector_search] loading {MODEL_ID}...")
        _state["model"] = SentenceTransformer(MODEL_ID, device="cpu")
    return _state["model"]


def embed_query(query: str):
    model = _get_model()
    vec = model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    return vec[0]  # shape [dim]


def search(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Returns top-k [(chunk_id, score), ...] sorted."""
    db, _ = get_index()
    query_vec = embed_query(query)
    return db.search(query_vec, k=k)