"""Duplicate detection across a set of documents.

Unlike the other pipeline modules (which take one document and return
one result), duplicate detection only makes sense across a *group* of
documents — so everything here takes a list of paths, not a single path.

Two layers:
  - Exact duplicates: SHA-256 hash of normalized text. Catches identical
    content even if filenames differ.
  - Near duplicates: cosine similarity between embedding fingerprints.
    Catches the same content saved in a different format (e.g. the same
    report as both a PDF and a DOCX), where whitespace/formatting
    differences mean the exact hash won't match even though a human
    would call them duplicates.
"""

import hashlib
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.ingest import ingest_document
from pipeline.clean import clean_text

NEAR_DUP_THRESHOLD = 0.92

# Loaded once per process — the first run downloads the model (~80MB,
# cached locally afterward), so the very first call will be slow.
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _normalize_for_hash(text: str) -> str:
    """Collapse whitespace and lowercase, so two documents that differ
    only in spacing/capitalization still hash identically."""
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_hash(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def fingerprint_embedding(text: str) -> np.ndarray:
    """A single vector representing the document, for near-duplicate
    comparison. The model truncates long input internally (around its
    max sequence length), so this is really a fingerprint of the
    document's beginning, not a deep read of the whole thing — good
    enough for catching whole-document duplicates, not for spotting two
    documents that only happen to share one paragraph.
    """
    return _get_model().encode(text)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_duplicates(paths: list[str]) -> dict:
    """Check a list of documents for exact and near duplicates.

    Returns:
      - exact_duplicate_groups: lists of paths that hash identically
      - near_duplicate_pairs: (path_a, path_b, similarity) tuples above
        the threshold, excluding pairs already caught as exact duplicates
    """
    records = []
    for path in paths:
        cleaned = clean_text(ingest_document(path)["full_text"])
        records.append({
            "path": path,
            "hash": exact_hash(cleaned),
            "embedding": fingerprint_embedding(cleaned),
        })

    hash_groups: dict[str, list[str]] = {}
    for r in records:
        hash_groups.setdefault(r["hash"], []).append(r["path"])
    exact_duplicate_groups = [g for g in hash_groups.values() if len(g) > 1]
    exact_pairs = {
        (g[i], g[j])
        for g in exact_duplicate_groups
        for i in range(len(g)) for j in range(i + 1, len(g))
    }

    near_duplicate_pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if (a["path"], b["path"]) in exact_pairs or (b["path"], a["path"]) in exact_pairs:
                continue
            sim = _cosine_sim(a["embedding"], b["embedding"])
            if sim >= NEAR_DUP_THRESHOLD:
                near_duplicate_pairs.append((a["path"], b["path"], sim))

    return {
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_pairs": near_duplicate_pairs,
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.dedup <path1> <path2> [path3 ...]")
        sys.exit(1)

    result = find_duplicates(sys.argv[1:])
    print(json.dumps(result, indent=2, ensure_ascii=False))