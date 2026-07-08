"""
bm25_index.py — task 3 (BM25 half): builds a BM25 index over the full corpus
using language-aware tokenization, exposes a search() function that returns
results in the same (chunk_id, score) convention as FaissDB, so fusion can
treat both retrievers identically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bm25 import BM25Okapi
from retrieval.bm25_tokenizer import tokenize
from retrieval.bootstrap import PROJ2_SRC

if str(PROJ2_SRC) not in sys.path:
    sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

_state = {}


def build_bm25_index(corpus: list[dict]):
    tokenized_chunks = []
    ids = []

    for chunk in corpus:
        tokens = tokenize(chunk["text"], language=chunk.get("language"))
        tokenized_chunks.append(tokens)
        ids.append(chunk["chunk_id"])

    bm25 = BM25Okapi(tokenized_chunks)

    _state["bm25"] = bm25
    _state["ids"] = ids
    print(f"[bm25] indexed {len(ids)} chunks")
    return bm25, ids


def get_bm25_index():
    """Lazy singleton — loads corpus itself if not already built, same
    pattern as bootstrap.get_index(). No caller ever needs to pass a corpus."""
    if "bm25" not in _state:
        corpus = load_corpus(strategy="semantic")
        build_bm25_index(corpus)
    return _state["bm25"], _state["ids"]


def search(query: str, k: int = 10, query_language: str = None) -> list[tuple[str, float]]:
    bm25, ids = get_bm25_index()
    tokens = tokenize(query, language=query_language)
    scores = bm25.get_scores(tokens)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [(cid, float(score)) for cid, score in ranked[:k]]