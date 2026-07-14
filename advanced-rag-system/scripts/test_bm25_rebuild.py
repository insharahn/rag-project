# scripts/test_bm25_rebuild.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import PROJ2_SRC
sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

from retrieval.bm25_index import _build_fresh, search, get_bm25_index

# confirm current (stale) BM25 doesn't know about the new doc yet
bm25, ids = get_bm25_index()
print(f"BM25 currently indexed: {len(ids)} chunks")

full_corpus = load_corpus(strategy="semantic")
print(f"Full corpus now: {len(full_corpus)} chunks")

# rebuild BM25 from the FULL corpus (old + new)
new_bm25, new_ids = _build_fresh(full_corpus)

# replace in-memory state (not persisted to disk yet)
from retrieval import bm25_index as bm25_module
bm25_module._state["bm25"] = new_bm25
bm25_module._state["ids"] = new_ids

# now search for something specific to the new doc's real content
results = search("decision tree hair height classifier", k=5)
print("\nSearch results after rebuild:")
for cid, score in results:
    print(f"  {score:.4f}  {cid}")