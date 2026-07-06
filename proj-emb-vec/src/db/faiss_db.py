#src/db/faiss_db.py
import faiss
import numpy as np
from db.base import VectorDB


class FaissDB(VectorDB):
    """FAISS flat (exact) index using inner product.

    Because our vectors are L2-normalized, inner product == cosine similarity,
    and a flat index does exhaustive brute-force search — no approximation.
    
    RECALL GROUND TRUTH: whatever FAISS-flat returns are the true nearest neighbors 
    that Chroma/Milvus get scored against.
    """

    name = "faiss-flat"

    def __init__(self):
        self.index = None
        self.ids = None

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)     # IP on normalized = cosine
        self.index.add(vectors)                  # exact, no training needed
        self.ids = ids

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        scores, positions = self.index.search(q, k)
        # FAISS returns positions (row indices); map back to chunk_ids.
        # IndexFlatIP scores: higher = more similar (already our convention).
        out = []
        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:        # FAISS uses -1 to pad if fewer than k exist
                continue
            out.append((self.ids[pos], float(score)))
        return out

    def teardown(self) -> None:
        self.index = None
        self.ids = None