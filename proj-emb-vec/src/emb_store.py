"""
    shared vector loader
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # src/ -> root
EMB = ROOT / "embeddings"


def load_embeddings(model_name: str):
    """Load one model's cached vectors + chunk_ids.
    Returns (vectors [n, dim] float32 contiguous, ids list[str])."""
    vecs = np.load(EMB / f"{model_name}.npy")
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)   # FAISS needs this
    meta = json.loads((EMB / f"{model_name}.meta.json").read_text())
    return vecs, meta["chunk_ids"]