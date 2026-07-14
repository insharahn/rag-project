"""
bm25_index.py — task 3 (BM25 half): builds (or loads a persisted) BM25
index over the full corpus using language-aware tokenization.
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bm25 import BM25Okapi
from retrieval.bm25_tokenizer import tokenize
from retrieval.bootstrap import PROJ2_SRC, INDEX_DIR

if str(PROJ2_SRC) not in sys.path:
    sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

BM25_PATH = INDEX_DIR / "bm25.pkl"

_state = {}


def _build_fresh(corpus):
    tokenized_chunks = []
    ids = []
    for chunk in corpus:
        tokens = tokenize(chunk["text"], language=chunk.get("language"))
        tokenized_chunks.append(tokens)
        ids.append(chunk["chunk_id"])

    bm25 = BM25Okapi(tokenized_chunks)
    print(f"[bm25] indexed {len(ids)} chunks")
    return bm25, ids


def _save_state(bm25, ids):
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"[bm25] persisted index -> {BM25_PATH} ({len(ids)} chunks)")


def get_bm25_index():
    """Lazy singleton — loads persisted index if present, otherwise builds
    fresh from the corpus and persists it."""
    if "bm25" not in _state:
        if BM25_PATH.exists():
            print("[bm25] loading persisted index...")
            with open(BM25_PATH, "rb") as f:
                saved = pickle.load(f)
            _state["bm25"] = saved["bm25"]
            _state["ids"] = saved["ids"]
        else:
            print("[bm25] no persisted index found, building fresh...")
            corpus = load_corpus(strategy="semantic")
            bm25, ids = _build_fresh(corpus)
            _state["bm25"] = bm25
            _state["ids"] = ids
            _save_state(bm25, ids)

    return _state["bm25"], _state["ids"]


def save_current_state():
    """Call after any mutation (e.g. partial reindex adds new chunks)."""
    if "bm25" not in _state:
        raise RuntimeError("get_bm25_index() must be called before save_current_state()")
    _save_state(_state["bm25"], _state["ids"])


def search(query: str, k: int = 10, query_language: str = None) -> list[tuple[str, float]]:
    bm25, ids = get_bm25_index()
    tokens = tokenize(query, language=query_language)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [(cid, float(score)) for cid, score in ranked[:k]]