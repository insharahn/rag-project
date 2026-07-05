# scripts/eval_stats.py
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
q = json.loads((ROOT / "eval" / "eval_queries.json").read_text(encoding="utf-8"))["queries"]

by_lang = Counter(x["language"] for x in q)
docs_by_lang = {}
for x in q:
    doc = x["answer_chunk_id"].rsplit("__", 1)[0]
    docs_by_lang.setdefault(x["language"], set()).add(doc)

print(f"total queries: {len(q)}")
for lang in by_lang:
    print(f"  {lang}: {by_lang[lang]} queries across {len(docs_by_lang[lang])} documents")
print(f"unique documents total: {len(set(x['answer_chunk_id'].rsplit('__',1)[0] for x in q))}")