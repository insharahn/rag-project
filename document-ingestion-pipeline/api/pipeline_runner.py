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

    # --- Dedup: read existing text from TXT files, using metadata.json
    # as the source of truth for what documents actually exist ---
    from pipeline.dedup import exact_hash
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    import numpy as np

    is_duplicate = False
    duplicate_of = None

    existing_texts = {}
    metadata_path = PROCESSED_DIR / "metadata.json"
    PROCESSED_DIR.mkdir(exist_ok=True)

    if metadata_path.exists():
        try:
            master = json.loads(metadata_path.read_text(encoding="utf-8"))
            for display_fname, _meta in master.items():
                stem = Path(display_fname).stem
                txt_path = PROCESSED_DIR / f"{stem}.txt"
                if txt_path.exists():
                    text = txt_path.read_text(encoding="utf-8")
                    if text:
                        existing_texts[display_fname] = text
        except Exception:
            pass

    if cleaned and existing_texts:
        incoming_hash = exact_hash(cleaned)
        for fname, stored_text in existing_texts.items():
            if exact_hash(stored_text) == incoming_hash:
                is_duplicate = True
                duplicate_of = {"filename": fname, "similarity": 1.0}
                break

        if not is_duplicate and len(existing_texts) >= 1:
            try:
                names = list(existing_texts.keys())
                texts = [cleaned] + [existing_texts[n] for n in names]
                vec = TfidfVectorizer(stop_words="english", max_features=20000)
                matrix = vec.fit_transform(texts)
                sims = sk_cosine(matrix[0:1], matrix[1:])[0]
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim >= 0.85:
                    is_duplicate = True
                    duplicate_of = {"filename": names[best_idx], "similarity": round(best_sim, 4)}
            except Exception:
                pass

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
    
    # --- Persist ---
    PROCESSED_DIR.mkdir(exist_ok=True)

    # Per-document JSON: chunks only (no metadata — that lives in metadata.json)
    (PROCESSED_DIR / f"{display_stem}.json").write_text(
        json.dumps({
            "chunking_strategy_used": strategy,
            "chunks": chunks_result,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Per-document TXT: cleaned readable text
    (PROCESSED_DIR / f"{display_stem}.txt").write_text(
        cleaned or "", encoding="utf-8"
    )

    # Master metadata file: one entry per document, updated on every upload.
    # This is the single source of truth for what's been processed.
    meta_entry = {
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
    }

    try:
        master = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    except Exception:
        master = {}

    master[display_name] = meta_entry
    metadata_path.write_text(
        json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Assemble result ---
    # result returned to the API (full_text included for dedup on next upload;
    # not written to any file)
    result = {**meta_entry, "full_text": cleaned, "chunks": chunks_result}
    return result