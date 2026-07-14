# scripts/test_diff_detection.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import get_index, PROJ2_SRC
sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

db, text_by_id = get_index()
current_ids = set(db.ids)

full_corpus = load_corpus(strategy="semantic")
full_ids = {c["chunk_id"] for c in full_corpus}

new_chunk_ids = full_ids - current_ids
missing_from_corpus = current_ids - full_ids  # sanity check: should be empty

print(f"Currently indexed: {len(current_ids)}")
print(f"Full corpus now:   {len(full_ids)}")
print(f"New chunks found:  {len(new_chunk_ids)}")
print(f"Indexed-but-missing-from-corpus (should be 0): {len(missing_from_corpus)}")

if new_chunk_ids:
    sample = list(new_chunk_ids)[:5]
    print(f"Sample new chunk_ids: {sample}")