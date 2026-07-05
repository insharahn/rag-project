import numpy as np
from emb_store import load_embeddings
from loader import load_corpus
from db.faiss_db import FaissDB

corpus = load_corpus()
vecs, ids = load_embeddings("bge-large")
print(f"loaded {vecs.shape} vectors\n")

db = FaissDB()
db.build(vecs, ids)

# id -> chunk text, for eyeballing results
text_by_id = {c["chunk_id"]: c["text"] for c in corpus}

# Use chunk #500's OWN vector as the query. It MUST come back rank 1.
probe = 500
query = vecs[probe]
results = db.search(query, k=5)

print(f"QUERY chunk: {ids[probe]}")
print(f"  {text_by_id[ids[probe]][:150]!r}\n")
print("TOP 5 NEIGHBORS:")
for rank, (cid, score) in enumerate(results, 1):
    tag = "  <-- itself (must be rank 1, score~1.0)" if cid == ids[probe] else ""
    print(f"  {rank}. {score:.4f}  {cid}{tag}")
    print(f"       {text_by_id[cid][:120]!r}")

db.teardown()