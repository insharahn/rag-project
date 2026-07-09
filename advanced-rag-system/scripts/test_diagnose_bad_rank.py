import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_rewrite import rewrite_query
from retrieval.multi_query import generate_query_variants
from retrieval.hybrid_search import hybrid_search
from retrieval.bootstrap import get_index

raw_query = "Albert Camus death philosophy"
_, text_by_id = get_index()

rewritten = rewrite_query(raw_query)
variants = generate_query_variants(raw_query, rewritten, n=3)

target_id = "The Project Gutenberg eBook of The Great Events by Famous Historians, Vol. 11.__225"

print(f"Variants generated:")
for v in variants:
    print(f"  - {v}")
print()

for variant in variants:
    results = hybrid_search(variant, k=15)
    ids_only = [cid for cid, _ in results]
    if target_id in ids_only:
        rank = ids_only.index(target_id) + 1
        score = dict(results)[target_id]
        print(f"FOUND in variant: {variant!r}")
        print(f"  rank={rank}  rrf_score={score:.5f}")
    else:
        print(f"NOT found in: {variant!r}")

print(f"\nFull text of target chunk:")
print(text_by_id[target_id]["text"])