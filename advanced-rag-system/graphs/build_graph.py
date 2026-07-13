"""
graphs/build_graph.py

Offline graph construction. Run once:
    python graphs/build_graph.py

Reads the corpus from the ingestion pipeline, extracts entities per chunk
using language-appropriate heuristics, builds a co-occurrence graph, and
pickles it to graphs/graph.pkl.

No new dependencies — uses kiwipiepy (already installed) for Korean,
stdlib re/unicodedata for English and Urdu.
"""
import json
import pickle
import re
import sys
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GRAPH_PATH = Path(__file__).resolve().parent / "graph.pkl"
PROCESSED   = (
    Path(__file__).resolve().parent.parent.parent
    / "document-ingestion-pipeline"
    / "processed_documents"
)

# ---------------------------------------------------------------------------
# English: capitalized-sequence heuristic (no spacy needed)
# ---------------------------------------------------------------------------
_EN_STOP = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","has","have","had","do",
    "does","did","will","would","could","should","may","might","shall","can",
    "that","this","these","those","it","its","he","she","they","we","i","you",
    "his","her","their","our","your","my","not","no","so","as","if","then",
    "than","when","where","who","which","what","how","all","also","into","out",
    "up","down","only","about","after","before","between","through","during",
    "without","within","along","following","across","behind","beyond","plus",
    "except","around","however","therefore","thus","hence","first","second",
    "third","one","two","said","such","some","any","each","every","both","few",
    "more","most","other","another","same","different","like","just","even",
    "still","already","yet","again","once","very","quite","rather","almost",
    "always","never","often","while","although","though","because","since",
    "until","unless","whether","either","neither","nor","upon","above","below",
    "over","under","further","there","here","now","much","many","new","old",
    "great","good","long","little","own","right","high","small","large","next",
    "last","well","way","back","mr","mrs","dr","st","chapter","section",
}

_CAPS_SEQ = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b')
_ACRONYM   = re.compile(r'\b[A-Z]{2,6}\b')


def _extract_en(text: str) -> set[str]:
    entities = set()
    for sent in re.split(r'(?<=[.!?])\s+', text):
        prev_end = -1
        for m in _CAPS_SEQ.finditer(sent):
            phrase = m.group(1).strip()
            words  = phrase.split()
            # skip single word at sentence start
            if len(words) == 1 and m.start() < 4:
                prev_end = m.end()
                continue
            if all(w.lower() in _EN_STOP for w in words):
                prev_end = m.end()
                continue
            entities.add(phrase.lower())
            prev_end = m.end()
    for m in _ACRONYM.finditer(text):
        entities.add(m.group().lower())
    return entities


# ---------------------------------------------------------------------------
# Korean: kiwipiepy NNP/NNG
# ---------------------------------------------------------------------------
try:
    from kiwipiepy import Kiwi
    _KIWI = Kiwi()
except Exception:
    _KIWI = None

def _extract_ko(text: str) -> set[str]:
    if _KIWI is None:
        return {w for w in re.findall(r"[\uAC00-\uD7A3]+", text) if len(w) >= 2}
    entities = set()
    for tok in _KIWI.tokenize(text):
        if tok.tag in ("NNP", "NNG") and len(tok.form) >= 2:
            entities.add(tok.form)
    return entities

# ---------------------------------------------------------------------------
# Urdu: frequency + length heuristics over Arabic-script tokens
# ---------------------------------------------------------------------------
def _extract_ur(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text)
    words = re.findall(r"[\u0600-\u06FF\u0750-\u077F]+", text)
    counts = Counter(words)
    entities: set[str] = set()
    # words appearing 2+ times in the chunk are likely named entities / key terms
    entities |= {w for w, c in counts.items() if len(w) >= 3 and c >= 2}
    # longer words even if rare
    entities |= {w for w in words if len(w) >= 5}
    return entities


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------
def detect_script(text: str) -> str:
    hangul = len(re.findall(r"[\uAC00-\uD7A3]", text))
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    latin  = len(re.findall(r"[a-zA-Z]", text))
    if hangul > arabic and hangul > latin:
        return "ko"
    if arabic > hangul and arabic > latin:
        return "ur"
    return "en"


def normalize_entity(ent: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", ent)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())

def extract_entities(text: str, lang: str | None = None) -> set[str]:
    if lang is None:
        lang = detect_script(text)
    if lang == "ko":
        raw = _extract_ko(text)
    elif lang == "ur":
        raw = _extract_ur(text)
    else:
        raw = _extract_en(text)
    return {normalize_entity(e) for e in raw if normalize_entity(e)}



# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(corpus: list[dict]) -> dict:
    """
    Returns:
        entity_to_chunks : entity_str -> set of chunk_ids
        chunk_to_entities: chunk_id  -> set of entity_strs
        co_occurrence    : entity    -> set of entities co-occurring in same chunk
    """
    entity_to_chunks  = defaultdict(set)
    chunk_to_entities = defaultdict(set)

    print(f"[build_graph] extracting entities from {len(corpus)} chunks…")
    for i, chunk in enumerate(corpus):
        if i % 500 == 0:
            print(f"  {i}/{len(corpus)}")
        cid  = chunk["chunk_id"]
        lang = chunk.get("language")
        for ent in extract_entities(chunk["text"], lang):
            entity_to_chunks[ent].add(cid)
            chunk_to_entities[cid].add(ent)

    print("[build_graph] computing co-occurrence edges…")
    co_occurrence = defaultdict(set)
    for cid, ents in chunk_to_entities.items():
        ent_list = list(ents)
        for i, e1 in enumerate(ent_list):
            for e2 in ent_list[i + 1:]:
                co_occurrence[e1].add(e2)
                co_occurrence[e2].add(e1)

    print(
        f"[build_graph] done — "
        f"{len(entity_to_chunks)} entities, "
        f"{len(chunk_to_entities)} indexed chunks, "
        f"{sum(len(v) for v in co_occurrence.values()) // 2} co-occurrence edges"
    )
    return {
        "entity_to_chunks":  dict(entity_to_chunks),
        "chunk_to_entities": dict(chunk_to_entities),
        "co_occurrence":     dict(co_occurrence),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _load_corpus() -> list[dict]:
    master = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    corpus = []
    for fname, meta in master.items():
        if meta.get("is_duplicate"):
            continue
        stem    = Path(fname).stem
        doc_json = PROCESSED / f"{stem}.json"
        if not doc_json.exists():
            continue
        data = json.loads(doc_json.read_text(encoding="utf-8"))
        for c in data["chunks"]["chunks"]:
            text = c["text"].strip()
            if text:
                corpus.append({
                    "chunk_id": f"{stem}__{c['chunk_index']}",
                    "text":     text,
                    "language": meta.get("primary_language"),
                })
    return corpus


if __name__ == "__main__":
    corpus = _load_corpus()
    graph  = build_graph(corpus)
    GRAPH_PATH.parent.mkdir(exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)
    print(f"[build_graph] saved -> {GRAPH_PATH}")