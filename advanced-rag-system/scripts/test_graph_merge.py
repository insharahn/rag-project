# scripts/test_graph_merge.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bootstrap import PROJ2_SRC
sys.path.insert(0, str(PROJ2_SRC))
from loader import load_corpus

from graphs.build_graph import extract_entities
from graphs.graph_index import get_graph
from retrieval.graph_search import graph_search

graph = get_graph()
print(f"Graph currently has {len(graph['entity_to_chunks'])} entities")

full_corpus = load_corpus(strategy="semantic")
existing_ids = set()
for ents in graph["chunk_to_entities"].values():
    pass  # chunk_to_entities keys ARE chunk_ids, use that directly
existing_ids = set(graph["chunk_to_entities"].keys())

new_chunks = [c for c in full_corpus if c["chunk_id"] not in existing_ids]
print(f"New chunks to extract entities from: {len(new_chunks)}")

# merge new chunks' entities into the existing graph structures
for c in new_chunks:
    cid = c["chunk_id"]
    ents = extract_entities(c["text"], c.get("language"))
    print(f"  {cid}: {ents}")

    for ent in ents:
        graph["entity_to_chunks"].setdefault(ent, set()).add(cid)
    graph["chunk_to_entities"][cid] = ents

    # co-occurrence: pairwise links among this chunk's own entities
    ent_list = list(ents)
    for i, e1 in enumerate(ent_list):
        for e2 in ent_list[i+1:]:
            graph["co_occurrence"].setdefault(e1, set()).add(e2)
            graph["co_occurrence"].setdefault(e2, set()).add(e1)

print(f"\nGraph now has {len(graph['entity_to_chunks'])} entities")

# sanity: search for something from the new doc's content via graph search
results = graph_search("decision tree classifier hair height", top_k=5)
print(f"\nGraph search results:")
for cid, score in results:
    print(f"  {score:.4f}  {cid}")