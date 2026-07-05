import numpy as np
from emb_store import load_embeddings
from loader import load_corpus
from db.faiss_db import FaissDB
from db.milvus_db import MilvusDB

corpus = load_corpus()
vecs, ids = load_embeddings("bge-large")

faiss_db = FaissDB(); faiss_db.build(vecs, ids)
milvus_db = MilvusDB(); milvus_db.build(vecs, ids)

probe = 500
query = vecs[probe]
f = faiss_db.search(query, k=5)
m = milvus_db.search(query, k=5)

print(f"QUERY: {ids[probe]}\n")
print("FAISS (exact)          | MILVUS (approx)")
for (fid, fs), (mid, ms) in zip(f, m):
    print(f"{fs:.4f} {fid[:32]:<34}| {ms:.4f} {mid[:32]}")

overlap = len({i for i,_ in f} & {i for i,_ in m})
print(f"\ntop-5 overlap with ground truth: {overlap}/5")

faiss_db.teardown(); milvus_db.teardown()