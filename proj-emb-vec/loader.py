import json
from pathlib import Path

PROCESSED = Path("data/processed_documents")

def load_corpus():
    """Read all non-duplicate chunks from project 1's output.
    Returns a flat list of chunk dicts."""
    master = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    corpus, skipped_dupes, missing = [], 0, 0

    for fname, meta in master.items():
        if meta.get("is_duplicate"):
            skipped_dupes += 1
            continue
        stem = Path(fname).stem
        doc_json = PROCESSED / f"{stem}.json"
        if not doc_json.exists():
            missing += 1
            continue
        data = json.loads(doc_json.read_text(encoding="utf-8"))
        for c in data["chunks"]["chunks"]:
            text = c["text"].strip()
            if not text:
                continue
            corpus.append({
                "chunk_id":    f"{stem}__{c['chunk_index']}",
                "text":        text,
                "source_doc":  fname,
                "language":    meta.get("primary_language"),
                "strategy":    data.get("chunking_strategy_used"),
                "token_count": c.get("token_count"),
            })

    print(f"[loader] {len(corpus)} chunks | {skipped_dupes} dupe docs skipped | {missing} missing json")
    return corpus