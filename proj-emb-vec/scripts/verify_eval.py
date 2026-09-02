'''
scripts/verify_eval.py
    verify that the eval_queries.json file is consistent with the corpus: 
    every answer_chunk_id must exist in the corpus, and the text of that chunk should actually answer the question.
'''
import json
from pathlib import Path
from loader import load_corpus

ROOT = Path(__file__).resolve().parent.parent
corpus = load_corpus("fixed") #change to "semantic" or "recursive" if you want to verify those instead
text_by_id = {c["chunk_id"]: c["text"] for c in corpus}
valid_ids = set(text_by_id)

#change here too to match the loaded corpus
queries = json.loads((ROOT / "eval_fixed" / "eval_queries.json").read_text(encoding="utf-8"))["queries"]

missing, present = [], []
for q in queries:
    cid = q["answer_chunk_id"]
    (present if cid in valid_ids else missing).append(q)

print(f"{len(queries)} queries | {len(present)} valid IDs | {len(missing)} BROKEN\n")

if missing:
    print("=== BROKEN (chunk_id does not exist — these are fabricated) ===")
    for q in missing:
        print(f"  [{q['language']}] {q['answer_chunk_id']}")
        print(f"      Q: {q['query'][:70]}")

print("\n=== VERIFY THESE (does the text actually answer the question?) ===")
for q in present:
    print(f"\n[{q['language']}] {q['answer_chunk_id']}")
    print(f"  Q: {q['query']}")
    print(f"  chunk: {text_by_id[q['answer_chunk_id']][:250]}")