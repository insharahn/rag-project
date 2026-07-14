# scripts/test_full_reindex_persist.py
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import get_index, save_current_state as save_faiss, PROJ2_SRC
sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

from retrieval.bm25_index import get_bm25_index, save_current_state as save_bm25
from retrieval import bm25_index as bm25_module
from graphs.build_graph import extract_entities
from graphs.graph_index import get_graph, save_graph

# --- FAISS ---
db, text_by_id = get_index()
current_ids = set(db.ids)
full_corpus = load_corpus(strategy="semantic")
new_chunks = [c for c in full_corpus if c["chunk_id"] not in current_ids]

if not new_chunks:
    print("No new chunks — nothing to test. Upload a new doc first.")
    sys.exit()

print(f"Persisting {len(new_chunks)} new chunks across FAISS, BM25, graph...")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
new_vecs = model.encode([c["text"] for c in new_chunks], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
db.index.add(new_vecs)
db.ids.extend([c["chunk_id"] for c in new_chunks])
for c in new_chunks:
    text_by_id[c["chunk_id"]] = c
save_faiss()

# --- BM25 ---
from retrieval.bm25_index import _build_fresh
new_bm25, new_ids = _build_fresh(full_corpus)
bm25_module._state["bm25"] = new_bm25
bm25_module._state["ids"] = new_ids
save_bm25()

# --- Graph ---
graph = get_graph()
for c in new_chunks:
    cid = c["chunk_id"]
    ents = extract_entities(c["text"], c.get("language"))
    for ent in ents:
        graph["entity_to_chunks"].setdefault(ent, set()).add(cid)
    graph["chunk_to_entities"][cid] = ents
    ent_list = list(ents)
    for i, e1 in enumerate(ent_list):
        for e2 in ent_list[i+1:]:
            graph["co_occurrence"].setdefault(e1, set()).add(e2)
            graph["co_occurrence"].setdefault(e2, set()).add(e1)
save_graph(graph)

print("\nAll three persisted. Reload check (fresh process would show this):")
print(f"  FAISS index.ntotal: {db.index.ntotal}")
print(f"  BM25 ids count:     {len(new_ids)}")
print(f"  Graph entities:     {len(graph['entity_to_chunks'])}")