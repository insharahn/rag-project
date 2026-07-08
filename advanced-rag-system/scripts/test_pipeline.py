import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pipeline import retrieve

queries = [
    "the outsider guy's job",
    "Albert Camus death philosophy",
]

for q in queries:
    print(f"QUERY: {q}")
    results = retrieve(q, top_k=5)
    for rank, (cid, chunk, score) in enumerate(results, 1):
        print(f"  {rank}. {score:.4f}  {cid}")
        print(f"       {chunk['text'][:120]!r}")
    print()