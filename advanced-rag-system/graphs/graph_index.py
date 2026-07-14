"""
graphs/graph_index.py

Loads the pickled graph once and exposes a scored chunk-lookup interface.
"""
import pickle
from functools import lru_cache
from pathlib import Path

GRAPH_PATH = Path(__file__).resolve().parent / "graph.pkl"


@lru_cache(maxsize=1)
def get_graph() -> dict:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Graph not found at {GRAPH_PATH}. "
            "Run:  python graphs/build_graph.py"
        )
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)

def save_graph(graph: dict):
    """Persist the (possibly mutated) graph back to disk. Call after any
    merge operation (e.g. partial reindex adding new chunks' entities)."""
    import pickle
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)
    get_graph.cache_clear()  # force next get_graph() to reload from disk, not the stale lru_cache
    print(f"[graph] persisted -> {GRAPH_PATH} ({len(graph['entity_to_chunks'])} entities)")

def get_chunks_for_entities(entities: list[str], hop: int = 1) -> dict[str, float]:
    """
    Look up chunks that mention any of the given entities.

    Score:
      direct hit  — fraction of query entities present in that chunk (0–1)
      one-hop hit — same but multiplied by 0.5 (co-occurring entity, not direct)

    Returns {} gracefully if graph is not built yet.
    """
    try:
        graph = get_graph()
    except FileNotFoundError:
        return {}

    entity_to_chunks = graph["entity_to_chunks"]
    co_occurrence    = graph["co_occurrence"]
    n                = max(len(entities), 1)

    # --- direct hits ---
    scores: dict[str, float] = {}
    matched: set[str] = set()
    for ent in entities:
        if ent in entity_to_chunks:
            matched.add(ent)
            for cid in entity_to_chunks[ent]:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / n

    if hop < 1 or not matched:
        return scores

    # --- one-hop expansion ---
    hop_ents: set[str] = set()
    for ent in matched:
        hop_ents |= co_occurrence.get(ent, set())
    hop_ents -= matched

    for ent in hop_ents:
        for cid in entity_to_chunks.get(ent, set()):
            if cid not in scores:
                scores[cid] = 0.5 / n  # half-credit for indirect link

    return scores