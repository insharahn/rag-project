#src/db/milvus_db.py
import numpy as np
import time
from pymilvus import MilvusClient
from db.base import VectorDB


class MilvusDB(VectorDB):
    name = "milvus"

    def __init__(self):
        self.client = None
        self.dim = None
        
    def _flush_with_retry(self, name="bench", retries=3, wait=2.0):
        for attempt in range(retries):
            try:
                self.client.flush(collection_name=name)
                return
            except Exception as e:
                if attempt == retries - 1:
                    raise
                print(f"[milvus] flush timeout, retrying ({attempt+1}/{retries})...")
                time.sleep(wait)

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
            index_params=self.client.prepare_index_params(
                field_name="vector",
                index_type="FLAT",      # exact brute-force — matches FAISS ground truth
                metric_type="COSINE",
            ),
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
        self._flush_with_retry()
        self.client.load_collection(collection_name="bench")   # ensure searchable before returning

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