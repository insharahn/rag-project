import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import get_index

db, text_by_id = get_index()
print(f"Index built. {len(text_by_id)} chunks available.")

# grab one real chunk id and re-search it as a smoke test
sample_id = next(iter(text_by_id))
print(f"Sample chunk_id: {sample_id}")
print(f"Text preview: {text_by_id[sample_id]['text'][:100]!r}")