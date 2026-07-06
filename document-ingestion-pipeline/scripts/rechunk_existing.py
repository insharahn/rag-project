"""Re-chunk existing cleaned .txt files with fixed and recursive strategies,
for comparison against the existing semantic-chunked output.

Reuses:
  - cleaned text from processed_documents_semantic/<stem>.txt
  - metadata (language, title, etc.) from processed_documents_semantic/metadata.json
Does NOT re-run extraction, cleaning, language detection, or dedup --
those are chunking-strategy-agnostic and already computed.

Usage:
  python -m scripts.rechunk_existing
"""

import json
from pathlib import Path

from pipeline.chunk_fixed import chunk_fixed_size
from pipeline.chunk_recursive import chunk_recursive

SOURCE_DIR = Path("processed_documents_semantic")

STRATEGIES = {
    "fixed": (chunk_fixed_size, Path("processed_documents_fixed")),
    "recursive": (chunk_recursive, Path("processed_documents_recursive")),
}


def load_source_metadata() -> dict:
    meta_path = SOURCE_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {SOURCE_DIR}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def find_meta_entry(master: dict, stem: str) -> dict:
    """Match a .txt file's stem back to its metadata.json entry
    (keyed by original display filename, e.g. 'urdu1.pdf')."""
    for fname, entry in master.items():
        if Path(fname).stem == stem:
            return fname, entry
    return None, {}


def rechunk_one(txt_path: Path, source_master: dict, chunker_fn, strategy_name: str, out_dir: Path):
    stem = txt_path.stem
    display_name, src_meta = find_meta_entry(source_master, stem)

    if not src_meta:
        print(f"  [skip] No metadata found for '{stem}', skipping.")
        return

    cleaned = txt_path.read_text(encoding="utf-8")

    # --- Chunk with the target strategy ---
    chunks = chunker_fn(cleaned)
    chunks_result = {
        "strategy": strategy_name,
        "num_chunks": len(chunks),
        "avg_token_count": round(
            sum(c["token_count"] for c in chunks) / len(chunks), 1
        ) if chunks else 0,
        "chunks": chunks,
    }

    # --- Persist per-doc json/txt, same shape as run_pipeline ---
    out_dir.mkdir(exist_ok=True, parents=True)

    (out_dir / f"{stem}.json").write_text(
        json.dumps({
            "chunking_strategy_used": strategy_name,
            "chunks": chunks_result,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (out_dir / f"{stem}.txt").write_text(cleaned, encoding="utf-8")

    # --- Update this strategy's own metadata.json ---
    meta_entry = {**src_meta, "chunking_strategy_used": strategy_name}

    meta_path = out_dir / "metadata.json"
    try:
        master = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        master = {}
    master[display_name] = meta_entry
    meta_path.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  {stem}: {len(chunks)} chunks (avg {chunks_result['avg_token_count']} tokens)")


def main():
    source_master = load_source_metadata()
    txt_files = sorted(SOURCE_DIR.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "metadata.json"]

    if not txt_files:
        print(f"No .txt files found in {SOURCE_DIR}")
        return

    print(f"Found {len(txt_files)} source .txt file(s).\n")

    for strategy_name, (chunker_fn, out_dir) in STRATEGIES.items():
        print(f"=== Strategy: {strategy_name} -> {out_dir} ===")
        for txt_path in txt_files:
            rechunk_one(txt_path, source_master, chunker_fn, strategy_name, out_dir)
        print()

    print("Done. Check processed_documents_fixed/ and processed_documents_recursive/")


if __name__ == "__main__":
    main()