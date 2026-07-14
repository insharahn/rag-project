# scripts/test_persistence.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import get_index

db, text_by_id = get_index()
print(f"Loaded {len(text_by_id)} chunks, FAISS index has {db.index.ntotal} vectors")

# sanity: run one real search to confirm the loaded index actually works
sample_id = next(iter(text_by_id))
import numpy as np
sample_vec = db.index.reconstruct(0)  # first vector back out of the index
results = db.search(sample_vec, k=3)
print(f"Self-search top result: {results[0]}")