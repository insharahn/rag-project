# scripts/test_faiss_append.py
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import get_index, PROJ2_SRC
sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

db, text_by_id = get_index()
current_ids = set(db.ids)

full_corpus = load_corpus(strategy="semantic")
new_chunks = [c for c in full_corpus if c["chunk_id"] not in current_ids]

print(f"New chunks to embed: {len(new_chunks)}")
for c in new_chunks:
    print(f"  {c['chunk_id']}: {c['text'][:80]!r}")

if not new_chunks:
    print("Nothing new to test — upload a doc first.")
    sys.exit()

# embed with bge-m3, no prefix — same config as corpus-side embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
texts = [c["text"] for c in new_chunks]
new_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

print(f"\nEmbedded shape: {new_vecs.shape}")
print(f"FAISS index size BEFORE append: {db.index.ntotal}")

# append to FAISS (in-memory only, not persisted yet)
db.index.add(new_vecs)
db.ids.extend([c["chunk_id"] for c in new_chunks])
for c in new_chunks:
    text_by_id[c["chunk_id"]] = c

print(f"FAISS index size AFTER append:  {db.index.ntotal}")
print(f"db.ids length AFTER append:     {len(db.ids)}")

# sanity: search using the new chunk's OWN vector — should return itself as rank 1
probe_vec = new_vecs[0]
results = db.search(probe_vec, k=3)
print(f"\nSelf-search on new chunk:")
for cid, score in results:
    print(f"  {score:.4f}  {cid}")