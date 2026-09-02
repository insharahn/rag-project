"""
retrieval/graph_search.py

Query-time graph retrieval: extract entities from the query, look them up
in the knowledge graph, return scored (chunk_id, score) pairs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphs.build_graph import extract_entities, detect_script
from graphs.graph_index import get_chunks_for_entities


def graph_search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    lang     = detect_script(query)
    entities = list(extract_entities(query, lang))

    if not entities:
        return []

    scored  = get_chunks_for_entities(entities, hop=1)
    # break ties deterministically by chunk_id, not arbitrary set/dict order
    results = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
    return results[:top_k]