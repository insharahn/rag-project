"""
graphs/graph_index.py

Loads the pickled graph once and exposes a scored chunk-lookup interface.
"""
import pickle
from functools import lru_cache
from pathlib import Path
from collections import defaultdict

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


@lru_cache(maxsize=1)
def get_word_index() -> dict:
    """Word -> set of canonical entities containing that word, built once
    from the existing entity_to_chunks keys. Lets query-side single words
    (e.g. 'car') match multi-word canonical entities (e.g. 'class car')
    without requiring exact string equality or a graph rebuild."""
    graph = get_graph()
    word_index = defaultdict(set)
    for entity in graph["entity_to_chunks"]:
        for word in entity.split():
            word_index[word].add(entity)
    return dict(word_index)


def _resolve_entity(query_entity: str, entity_to_chunks: dict, word_index: dict) -> list[tuple[str, float]]:
    """Returns [(matched_canonical_entity, match_quality), ...] for a single
    query entity. match_quality is 1.0 for exact match, or word-overlap
    fraction (matched words / canonical entity's total words) for a partial
    substring-style match — so 'car' matching 'class car' scores 0.5, not
    1.0, reflecting it's a weaker signal than a full match."""
    if query_entity in entity_to_chunks:
        return [(query_entity, 1.0)]

    matches = []
    query_words = set(query_entity.split())
    for word in query_words:
        for candidate in word_index.get(word, ()):
            candidate_words = set(candidate.split())
            overlap = len(query_words & candidate_words) / len(candidate_words)
            matches.append((candidate, overlap))
    return matches


def get_chunks_for_entities(entities: list[str], hop: int = 1) -> dict[str, float]:
    try:
        graph = get_graph()
        word_index = get_word_index()
    except FileNotFoundError:
        return {}

    entity_to_chunks = graph["entity_to_chunks"]
    co_occurrence    = graph["co_occurrence"]
    n                = max(len(entities), 1)

    scores: dict[str, float] = {}
    matched: set[str] = set()

    for ent in entities:
        for canonical, quality in _resolve_entity(ent, entity_to_chunks, word_index):
            matched.add(canonical)
            for cid in entity_to_chunks[canonical]:
                scores[cid] = scores.get(cid, 0.0) + (quality / n)

    if hop < 1 or not matched:
        return scores

    hop_ents: set[str] = set()
    for ent in matched:
        hop_ents |= co_occurrence.get(ent, set())
    hop_ents -= matched

    for ent in hop_ents:
        for cid in entity_to_chunks.get(ent, set()):
            if cid not in scores:
                scores[cid] = 0.5 / n

    return scores