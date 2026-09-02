import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_corpus(strategy="semantic"):
    """Read all non-duplicate chunks from project 1's output.
    strategy: "semantic", "fixed", or "recursive"
    Returns a flat list of chunk dicts."""
    
    # Map strategy to folder name
    folder_map = {
        "semantic": "processed_documents",
        "fixed": "processed_documents_fixed",
        "recursive": "processed_documents_recursive"
    }
    
    folder_name = folder_map.get(strategy, "processed_documents")
    PROCESSED = ROOT.parent / "document-ingestion-pipeline" / folder_name
    
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

    print(f"[loader] {len(corpus)} chunks | {skipped_dupes} dupe docs skipped | {missing} missing json | strategy: {strategy}")
    return corpus