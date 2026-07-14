# scripts/test_bm25_persistence.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_index import get_bm25_index, search

t0 = time.time()
bm25, ids = get_bm25_index()
print(f"Loaded {len(ids)} chunks in {time.time()-t0:.2f}s")

results = search("Madame Meursault entered the Home three years ago", k=3)
print(f"Top result: {results[0]}")