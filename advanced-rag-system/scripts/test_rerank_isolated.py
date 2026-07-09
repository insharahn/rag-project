# scripts/test_rerank_isolated.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.rerank import _get_reranker, rerank
from retrieval.bootstrap import get_index

t0 = time.time()
_get_reranker()   # forces model load only
t1 = time.time()
print(f"Model load: {t1-t0:.2f}s")

_, text_by_id = get_index()
sample_ids = list(text_by_id.keys())[:100]
candidates = [(cid, text_by_id[cid]) for cid in sample_ids]

t2 = time.time()
rerank("Albert Camus death philosophy", candidates, top_k=5)
t3 = time.time()
print(f"Rerank 100 real candidates: {t3-t2:.2f}s")