# scripts/test_warm_latency.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pipeline import retrieve
from generation.citation_generator import generate_answer

query = "Albert Camus death philosophy"

# warm-up call — absorbs all index building + model loading, not timed
_ = retrieve(query, top_k=5)

# now time a second, fully warm call — this is the real per-query cost
t0 = time.time()
chunks = retrieve(query, top_k=5)
t1 = time.time()
result = generate_answer(query, chunks)
t2 = time.time()

print(f"\nWARM retrieval: {t1-t0:.2f}s")
print(f"WARM generation: {t2-t1:.2f}s")
print(f"WARM TOTAL: {t2-t0:.2f}s")