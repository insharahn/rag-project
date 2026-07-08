import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_rewrite import rewrite_query

test_queries = [
    "whats camus say about death and stuff",
    "the outsider guy's job",
]

for q in test_queries:
    rewritten = rewrite_query(q)
    print(f"RAW:       {q}")
    print(f"REWRITTEN: {rewritten}\n")