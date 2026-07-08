import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import get_index
from retrieval.vector_search import search

_, text_by_id = get_index()

queries = [
    "Madame Meursault entered the Home three years ago",
    "Albert Camus death philosophy",
]

for q in queries:
    print(f"QUERY: {q}")
    results = search(q, k=5)
    for rank, (cid, score) in enumerate(results, 1):
        print(f"  {rank}. {score:.4f}  {cid}")
        print(f"       {text_by_id[cid]['text'][:120]!r}")
    print()