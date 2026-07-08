import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_rewrite import rewrite_query
from retrieval.multi_query import generate_query_variants

raw_queries = [
    "whats camus say about death and stuff",
    "the outsider guy's job",
]

for raw in raw_queries:
    rewritten = rewrite_query(raw)
    variants = generate_query_variants(raw, rewritten, n=3)
    print(f"RAW:       {raw}")
    print(f"REWRITTEN: {rewritten}")
    print("VARIANTS:")
    for v in variants:
        print(f"  - {v}")
    print()