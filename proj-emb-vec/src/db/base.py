#src/db/base.py
from abc import ABC, abstractmethod
import numpy as np


class VectorDB(ABC):
    """Common contract every database adapter implements.

    All three databases (FAISS, Chroma, Milvus) expose exactly these three
    methods so the benchmark harness can treat them identically. Each adapter
    is fully independent (can build/search/teardown any one on its own).

    Score convention (enforced by every adapter): search() returns
    (chunk_id, score) where HIGHER score = MORE similar. Since all my vectors
    are L2-normalized, that score is cosine similarity. Each adapter is
    responsible for converting its native distance/similarity output into this
    shared convention, so numbers are comparable across databases.
    """

    name: str = "base"

    @abstractmethod
    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Insert vectors (shape [n, dim], float32) with their chunk_ids,
        and build the index. This is the operation we time for build cost."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return top-k [(chunk_id, score), ...] for a single query vector
        (shape [dim] or [1, dim]), sorted best-first (highest score first)."""

    @abstractmethod
    def teardown(self) -> None:
        """Release memory / drop the collection so the next run starts clean."""