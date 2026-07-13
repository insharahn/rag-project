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
    """
    Entity-based retrieval from the knowledge graph.

    Returns a list of (chunk_id, score) sorted by score descending,
    or [] if the graph is not built or no entities are found.
    Score is in (0, 1].
    """
    lang     = detect_script(query)
    entities = list(extract_entities(query, lang))

    if not entities:
        return []

    scored  = get_chunks_for_entities(entities, hop=1)
    results = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return results[:top_k]