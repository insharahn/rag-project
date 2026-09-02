"""
scripts/remap_eval.py

Reads eval/eval_queries.json (ground truth against semantic chunks),
then for each query finds the corresponding chunk in recursive
corpora by substring-matching a key excerpt from the original semantic
chunk text.

Writes:
  eval_recursive/eval_queries.json

Usage:
  python -m scripts.remap_eval
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_corpus

EXCERPT_LEN = 120   # chars to use as the search key from the original chunk


def extract_key(text: str) -> str:
    """Pull a stable substring from the middle of a chunk.
    Middle is safer than the start/end because overlap windows
    from fixed/recursive chunking are most likely to contaminate
    the edges, not the middle.
    """
    text = text.strip()
    if len(text) <= EXCERPT_LEN:
        return text
    mid = len(text) // 2
    half = EXCERPT_LEN // 2
    return text[mid - half: mid + half]


def find_chunk(original_text: str, chunks_for_doc: list[dict]) -> dict | None:
    """Try excerpts from multiple positions at decreasing lengths."""
    text = original_text.strip()
    if not text:
        return None

    def candidates(length: int) -> list[str]:
        if len(text) <= length:
            return [text]
        mid   = len(text) // 2
        quart = len(text) // 4
        tquart = (3 * len(text)) // 4
        half  = length // 2
        keys  = [
            text[mid - half : mid + half],               # middle
            text[quart : quart + length],                 # 1/4 in
            text[tquart - length : tquart],               # 3/4 in
        ]
        # start/end but skip the first/last 80 chars (overlap zone)
        if len(text) > length + 80:
            keys.append(text[80 : 80 + length])
            keys.append(text[-(80 + length) : -80])
        return [k.strip() for k in keys if k.strip()]

    for length in [120, 80, 50, 35]:
        for key in candidates(length):
            for chunk in chunks_for_doc:
                if key in chunk["text"]:
                    return chunk

    return None

def build_doc_index(corpus: list[dict]) -> dict[str, list[dict]]:
    """Group chunks by their source document stem."""
    index: dict[str, list[dict]] = {}
    for chunk in corpus:
        # chunk_id format: "<doc_stem>__<index>"
        doc_stem = chunk["chunk_id"].rsplit("__", 1)[0]
        index.setdefault(doc_stem, []).append(chunk)
    return index


def remap(strategy: str, semantic_text_by_id: dict[str, str], queries: list[dict]) -> list[dict]:
    # Load the target corpus with the specified strategy
    corpus = load_corpus(strategy)
    doc_index = build_doc_index(corpus)
    
    # Build a set of all valid chunk IDs for verification
    valid_chunk_ids = set(c["chunk_id"] for c in corpus)

    remapped = []
    broken = []

    for q in queries:
        original_id = q["answer_chunk_id"]
        doc_stem = original_id.rsplit("__", 1)[0]

        original_text = semantic_text_by_id.get(original_id, "")
        if not original_text:
            broken.append((original_id, "not found in semantic corpus"))
            remapped.append({**q, "answer_chunk_id": None, "remap_status": "semantic_chunk_missing"})
            continue

        doc_chunks = doc_index.get(doc_stem, [])

        if not doc_chunks:
            broken.append((original_id, f"doc '{doc_stem}' missing from {strategy} corpus"))
            remapped.append({**q, "answer_chunk_id": None, "remap_status": "doc_missing"})
            continue

        match = find_chunk(original_text, doc_chunks)

        if match:
            matched_id = match["chunk_id"]
            # Verify the matched ID actually exists in the corpus
            if matched_id in valid_chunk_ids:
                remapped.append({**q, "answer_chunk_id": matched_id, "remap_status": "ok"})
            else:
                # This shouldn't happen, but just in case
                broken.append((original_id, f"matched chunk '{matched_id}' not in corpus"))
                remapped.append({**q, "answer_chunk_id": None, "remap_status": "invalid_match"})
        else:
            broken.append((original_id, "no chunk contains the key excerpt"))
            remapped.append({**q, "answer_chunk_id": None, "remap_status": "no_match"})

    # After processing, check for any remaining issues
    print(f"\n{len(remapped)} queries | "
          f"{sum(1 for r in remapped if r['remap_status'] == 'ok')} mapped | "
          f"{len(broken)} BROKEN")
    
    # Debug: Show all chunk IDs for the problematic document
    if "the_fall_of_the_house_of_usher" in doc_index:
        print(f"\nDEBUG: All chunk IDs for the_fall_of_the_house_of_usher in {strategy} corpus:")
        all_chunks = sorted([c["chunk_id"] for c in doc_index["the_fall_of_the_house_of_usher"]])
        for cid in all_chunks:
            print(f"  {cid}")
    
    for chunk_id, reason in broken:
        print(f"  BROKEN: {chunk_id}\n    reason: {reason}")

    return remapped


def main():
    eval_path = ROOT / "eval" / "eval_queries.json"
    queries = json.loads(eval_path.read_text(encoding="utf-8"))["queries"]

    # Load semantic corpus to get original answer chunk texts
    semantic_corpus = load_corpus("semantic")  # Explicitly load semantic
    semantic_text_by_id = {c["chunk_id"]: c["text"] for c in semantic_corpus}

    # Check which semantic IDs are missing
    all_original_ids = set(q["answer_chunk_id"] for q in queries if q["answer_chunk_id"])
    missing_semantic_ids = all_original_ids - set(semantic_text_by_id.keys())
    
    if missing_semantic_ids:
        print(f"\nWARNING: These chunk_ids don't exist in the semantic corpus:")
        for cid in sorted(missing_semantic_ids):
            print(f"  - {cid}")
        print()

    # Process the target strategy ----------------------------------------------------
    remapped = remap("fixed", semantic_text_by_id, queries)  #change recursive/fixed here

    out_dir = ROOT / "eval_fixed"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eval_queries.json"
    out_path.write_text(
        json.dumps({"queries": remapped}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  written → {out_path}")


if __name__ == "__main__":
    main()