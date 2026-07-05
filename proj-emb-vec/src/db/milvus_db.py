import numpy as np
from pymilvus import MilvusClient
from db.base import VectorDB



class MilvusDB(VectorDB):
    name = "milvus"

    def __init__(self):
        self.client = None
        self.dim = None

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        self.dim = vectors.shape[1]
        self.client = MilvusClient(uri="http://localhost:19530") #docker server

        if self.client.has_collection("bench"):
            self.client.drop_collection("bench")

        self.client.create_collection(
            collection_name="bench",
            dimension=self.dim,
            metric_type="COSINE",
            auto_id=False,
        )
        # Milvus needs an integer primary key; keep a position->chunk_id map
        # and store the position as the id.
        self.ids = ids
        B = 2000
        for i in range(0, len(ids), B):
            rows = [
                {"id": i + j, "vector": vectors[i + j].tolist()}
                for j in range(min(B, len(ids) - i))
            ]
            self.client.insert(collection_name="bench", data=rows)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32).reshape(1, -1).tolist()
        res = self.client.search(
            collection_name="bench",
            data=q,
            limit=k,
            output_fields=["id"],
        )
        out = []
        for hit in res[0]:
            pos = hit["id"]
            score = float(hit["distance"])   # COSINE: higher = better
            out.append((self.ids[pos], score))
        return out

    def teardown(self) -> None:
        if self.client is not None:
            try:
                self.client.drop_collection("bench")
            except Exception:
                pass
        self.client = None