"""Ties the full pipeline together into one function call: given a file
path, runs extraction → cleaning → language detection → metadata →
dedup → chunking (strategy selected by detected language) → persist.

Strategy selection implements the benchmark findings: semantic chunking
produces the best boundary quality for French, Urdu, Korean, and Arabic.
English defaults to recursive -- a good quality/speed balance that suits
both literary and technical English without knowing which type is being
ingested. Fixed-size is not selected automatically; it remains available
as an explicit override for throughput-critical batch scenarios.
"""

import json
from pathlib import Path

from pipeline.ingest import ingest_document
from pipeline.clean import clean_text
from pipeline.detect_language import detect_languages
from pipeline.metadata import extract_metadata
from pipeline.dedup import find_duplicates

PROCESSED_DIR = Path("processed_documents")

# Strategy selection by primary language, derived from benchmark findings.
# Languages not in this map fall back to recursive as a safe default.
LANGUAGE_STRATEGY_MAP = {
    "en": "recursive",   # practical default -- works for both literary and
                         # technical English; 7x faster than semantic with
                         # reasonable boundary quality (38.6%)
    "fr": "semantic",    # semantic wins decisively (0.933 vs 0.164 on the
                         # bilingual PDF; 0.929 vs 0.4 on Wikipedia)
    "ur": "semantic",    # semantic: 0.966 (native text), 0.905 (OCR);
                         # recursive: 0.055 -- not viable for Urdu
    "ko": "semantic",    # semantic: 0.986 vs recursive: 0.537
    "ar": "semantic",    # semantic: 0.65–0.97 vs recursive: 0.12–0.53
}

DEFAULT_STRATEGY = "recursive"


def select_strategy(primary_language: str | None) -> str:
    """Return the recommended chunking strategy for a detected language.
    Falls back to recursive for unknown or None languages.
    """
    if not primary_language:
        return DEFAULT_STRATEGY
    return LANGUAGE_STRATEGY_MAP.get(primary_language, DEFAULT_STRATEGY)


def _load_existing_paths() -> list[str]:
    """Return paths of all previously processed documents for dedup."""
    paths = []
    for json_file in PROCESSED_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if "path" in data:
                paths.append(data["path"])
        except Exception:
            continue
    return paths


def run_pipeline(file_path: str, original_filename: str | None = None, strategy_override: str | None = None) -> dict:
    """Run the full pipeline on one document and return all results.

    Saves output to processed_documents/<filename>.json and .txt.
    """
    # Heavy imports deferred -- sentence_transformers pulls in PyTorch
    # which is slow to load, so we pay that cost only when a request
    # actually arrives rather than at server startup.
    from pipeline.chunk_fixed import chunk_fixed_size
    from pipeline.chunk_recursive import chunk_recursive
    from pipeline.chunk_semantic import chunk_semantic

    CHUNKERS = {
        "fixed": chunk_fixed_size,
        "recursive": chunk_recursive,
        "semantic": chunk_semantic,
    }

    file_path_obj = Path(file_path)
    display_name = original_filename or file_path_obj.name
    display_stem = Path(display_name).stem

    # --- Extraction + cleaning ---
    ingest_result = ingest_document(file_path)
    full_text = ingest_result["full_text"] or ""
    cleaned = clean_text(full_text)

    # --- Language detection ---
    languages = detect_languages(cleaned)
    primary_language = languages[0]["language"] if languages else None

    # --- Metadata (pass display_filename so title fallback uses the
    # real name, not whatever temp name the file has on disk) ---
    metadata = extract_metadata(file_path, display_filename=display_name)

    # --- Strategy selection based on detected language ---
    strategy = strategy_override or select_strategy(primary_language)
    chunker_fn = CHUNKERS[strategy]

    # --- Dedup ---
    existing_paths = _load_existing_paths()
    dedup_result = {"exact_duplicate_groups": [], "near_duplicate_pairs": []}
    if existing_paths:
        try:
            dedup_result = find_duplicates([file_path] + existing_paths)
        except Exception as e:
            dedup_result["error"] = str(e)

    is_duplicate = bool(
        dedup_result["exact_duplicate_groups"] or
        dedup_result["near_duplicate_pairs"]
    )
    duplicate_of = None
    if dedup_result["near_duplicate_pairs"]:
        _, other_path, similarity = dedup_result["near_duplicate_pairs"][0]
        duplicate_of = {"path": other_path, "similarity": similarity}
    elif dedup_result["exact_duplicate_groups"]:
        group = dedup_result["exact_duplicate_groups"][0]
        other = [p for p in group if p != file_path]
        if other:
            duplicate_of = {"path": other[0], "similarity": 1.0}

    # --- Chunking (single selected strategy only) ---
    chunks_result = {}
    if cleaned:
        try:
            chunks = chunker_fn(cleaned)
            chunks_result = {
                "strategy": strategy,
                "num_chunks": len(chunks),
                "avg_token_count": round(
                    sum(c["token_count"] for c in chunks) / len(chunks), 1
                ) if chunks else 0,
                "chunks": chunks,
            }
        except Exception as e:
            chunks_result = {"strategy": strategy, "error": str(e)}
    else:
        chunks_result = {"strategy": strategy, "num_chunks": 0, "chunks": []}

    # --- Assemble result ---
    result = {
        "path": str(file_path_obj),
        "filename": display_name,
        "file_type": ingest_result["file_type"],
        "title": metadata.get("title"),
        "author": metadata.get("author"),
        "created": metadata.get("created"),
        "file_size_bytes": metadata.get("file_size_bytes"),
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()),
        "languages": languages,
        "primary_language": primary_language,
        "chunking_strategy_used": strategy,
        "extraction_info": ingest_result["extraction_info"],
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "full_text": cleaned,
        "chunks": chunks_result,
    }

    # --- Persist ---
    PROCESSED_DIR.mkdir(exist_ok=True)
    (PROCESSED_DIR / f"{display_stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (PROCESSED_DIR / f"{display_stem}.txt").write_text(
        cleaned or "", encoding="utf-8"
    )

    return result