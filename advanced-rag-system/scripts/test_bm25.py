import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import PROJ2_SRC  # reuse the sys.path setup
sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

from retrieval.bm25_index import build_bm25_index, search

corpus = load_corpus(strategy="semantic")
build_bm25_index(corpus)

text_by_id = {c["chunk_id"]: c["text"] for c in corpus}

queries = [
    "Madame Meursault entered the Home three years ago",
    "Albert Camus death philosophy",
]

for q in queries:
    print(f"QUERY: {q}")
    results = search(q, k=5)
    for rank, (cid, score) in enumerate(results, 1):
        print(f"  {rank}. {score:.4f}  {cid}")
        print(f"       {text_by_id[cid][:120]!r}")
    print()