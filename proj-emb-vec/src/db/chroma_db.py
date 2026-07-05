import chromadb
import numpy as np
from db.base import VectorDB


class ChromaDB(VectorDB):
    """Chroma with cosine space, in-memory client.

    Two things the adapter normalizes to match the contract:
      - Chroma wants python lists, not numpy arrays.
      - Chroma returns cosine *distance* (lower = closer). We convert to
        similarity (1 - distance) so higher = better, matching FAISS.
    """

    name = "chroma"

    def __init__(self):
        self.client = None
        self.collection = None

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        self.client = chromadb.Client()          # ephemeral, in-memory
        self.collection = self.client.create_collection(
            name="bench",
            metadata={"hnsw:space": "cosine"},    # cosine distance
        )
        # Chroma needs list-of-lists + string ids. Add in batches;
        # it rejects very large single adds.
        B = 5000
        vlist = vectors.tolist()
        for i in range(0, len(ids), B):
            self.collection.add(
                embeddings=vlist[i:i+B],
                ids=ids[i:i+B],
            )

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32).reshape(1, -1).tolist()
        res = self.collection.query(query_embeddings=q, n_results=k)
        out_ids = res["ids"][0]
        dists = res["distances"][0]              # cosine distance
        return [(cid, 1.0 - float(d)) for cid, d in zip(out_ids, dists)]

    def teardown(self) -> None:
        if self.client is not None:
            try:
                self.client.delete_collection("bench")
            except Exception:
                pass
        self.client = None
        self.collection = None