import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_rewrite import rewrite_query
from retrieval.multi_query import generate_query_variants
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank
from retrieval.bootstrap import get_index

query = "Albert Camus death philosophy"
_, text_by_id = get_index()

t0 = time.time()
rewritten = rewrite_query(query)
t1 = time.time()
print(f"rewrite_query:          {t1-t0:.2f}s")

variants = generate_query_variants(query, rewritten, n=3)
t2 = time.time()
print(f"generate_query_variants: {t2-t1:.2f}s")

seen = {}
for variant in variants:
    results = hybrid_search(variant, k=30)
    for cid, score in results:
        if cid not in seen:
            seen[cid] = score
t3 = time.time()
print(f"hybrid_search x{len(variants)} variants: {t3-t2:.2f}s")

candidates = [(cid, text_by_id[cid]) for cid in seen.keys()]
final = rerank(query, candidates, top_k=5)
t4 = time.time()
print(f"rerank:                  {t4-t3:.2f}s")

print(f"\nTOTAL: {t4-t0:.2f}s")