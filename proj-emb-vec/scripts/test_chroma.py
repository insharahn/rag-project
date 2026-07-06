#scripts/test_chroma.py
# tests the ChromaDB implementation against the FaissDB implementation, using the same embeddings and queries.
import numpy as np
from emb_store import load_embeddings
from loader import load_corpus
from db.faiss_db import FaissDB
from db.chroma_db import ChromaDB

corpus = load_corpus()
vecs, ids = load_embeddings("bge-large")
text_by_id = {c["chunk_id"]: c["text"] for c in corpus}

faiss_db = FaissDB(); faiss_db.build(vecs, ids)
chroma_db = ChromaDB(); chroma_db.build(vecs, ids)

probe = 500
query = vecs[probe]
f = faiss_db.search(query, k=5)
c = chroma_db.search(query, k=5)

print(f"QUERY: {ids[probe]}\n")
print("FAISS (exact)          | CHROMA (approx)")
for (fid, fs), (cid, cs) in zip(f, c):
    print(f"{fs:.4f} {fid[:32]:<34}| {cs:.4f} {cid[:32]}")

# agreement: overlap of the two top-5 sets
overlap = len({i for i,_ in f} & {i for i,_ in c})
print(f"\ntop-5 overlap with ground truth: {overlap}/5")

faiss_db.teardown(); chroma_db.teardown()