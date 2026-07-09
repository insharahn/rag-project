import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pipeline import retrieve
from generation.citation_generator import generate_answer

query = "Albert Camus death philosophy"

t0 = time.time()
chunks = retrieve(query, top_k=5)
t1 = time.time()
result = generate_answer(query, chunks)
t2 = time.time()

print(f"Retrieval (rewrite+multiquery+hybrid+rerank): {t1-t0:.2f}s")
print(f"Generation (citation answer):                  {t2-t1:.2f}s")
print(f"TOTAL:                                          {t2-t0:.2f}s")