"""Duplicate detection across a set of documents.

Two layers:
  - Exact duplicates: SHA-256 hash of normalized text. Catches identical
    content even if filenames differ.
  - Near duplicates: TF-IDF cosine similarity between documents. Catches
    the same content saved in a different format (e.g. the same report
    as both a PDF and a DOCX), or with minor edits.

This is deliberately lexical (word-overlap based), not semantic
(meaning-based).
"""

import hashlib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.ingest import ingest_document
from pipeline.clean import clean_text

NEAR_DUP_THRESHOLD = 0.85


def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_hash(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def find_duplicates(paths: list[str]) -> dict:
    """Check a list of documents for exact and near duplicates.

    Returns:
      - exact_duplicate_groups: lists of paths that hash identically
      - near_duplicate_pairs: (path_a, path_b, similarity) tuples above
        the threshold, excluding pairs already caught as exact duplicates
    """
    cleaned_texts = []
    hashes = []
    for path in paths:
        cleaned = clean_text(ingest_document(path)["full_text"]) or ""
        cleaned_texts.append(cleaned)
        hashes.append(exact_hash(cleaned))

    hash_groups: dict[str, list[str]] = {}
    for path, h in zip(paths, hashes):
        hash_groups.setdefault(h, []).append(path)
    exact_duplicate_groups = [g for g in hash_groups.values() if len(g) > 1]
    exact_pairs = {
        (g[i], g[j])
        for g in exact_duplicate_groups
        for i in range(len(g)) for j in range(i + 1, len(g))
    }

    near_duplicate_pairs = []
    non_empty = [(p, t) for p, t in zip(paths, cleaned_texts) if t.strip()]
    if len(non_empty) >= 2:
        vec_paths = [p for p, _ in non_empty]
        vec_texts = [t for _, t in non_empty]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        matrix = vectorizer.fit_transform(vec_texts)
        sims = cosine_similarity(matrix)

        for i in range(len(vec_paths)):
            for j in range(i + 1, len(vec_paths)):
                a, b = vec_paths[i], vec_paths[j]
                if (a, b) in exact_pairs or (b, a) in exact_pairs:
                    continue
                sim = float(sims[i, j])
                if sim >= NEAR_DUP_THRESHOLD:
                    near_duplicate_pairs.append((a, b, round(sim, 4)))

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