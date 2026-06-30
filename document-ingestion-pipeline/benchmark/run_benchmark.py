"""Benchmark comparing the three chunking strategies (fixed-size,
recursive, semantic) across the real document corpus.

Metrics, per chunker:
  - Speed: total and average time to chunk a document.
  - Chunk size distribution: avg/median/std token count per chunk.
    Fixed-size should show near-zero variance by design; the other two
    should vary more, since they prioritize respecting structure/meaning
    over hitting an exact size.
  - Boundary quality: fraction of chunks whose text ends at a real
    sentence boundary (., !, ?), as a proxy for "respects structure."
    Fixed-size is expected to score worst here -- it has no idea where
    sentences end.
  - Semantic coherence: average cosine similarity between consecutive
    sentences *within* a chunk, versus average similarity *across* a
    chunk boundary (last sentence of one chunk vs. first of the next).
    A chunker doing a good job should show a clear gap -- sentences
    inside a chunk are more related to each other than to whatever
    comes right after the cut. Computed on a subset of documents (not
    the full corpus), since it requires re-embedding every sentence on
    top of what semantic chunking already does, which adds real time.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just saving files
import matplotlib.pyplot as plt

from pipeline.ingest import ingest_document
from pipeline.clean import clean_text
from pipeline.chunk_fixed import chunk_fixed_size
from pipeline.chunk_recursive import chunk_recursive
from pipeline.chunk_semantic import chunk_semantic, _get_model, _split_sentences
from scripts.validate_corpus import find_corpus_files

CHUNKERS = {
    "fixed": chunk_fixed_size,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
}

SENTENCE_END_CHARS = (".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'")
COHERENCE_SAMPLE_SIZE = 10  # keep the heavier metric bounded in runtime

RESULTS_DIR = Path("benchmark/results")


def ends_at_sentence_boundary(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(SENTENCE_END_CHARS)


def boundary_quality(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    return sum(ends_at_sentence_boundary(c["text"]) for c in chunks) / len(chunks)


def semantic_coherence(chunks: list[dict]) -> dict:
    """Average intra-chunk and cross-boundary sentence similarity for
    one document's chunk list, from one chunker.
    """
    model = _get_model()
    intra_sims = []
    boundary_sims = []
    prev_last_embedding = None

    for chunk in chunks:
        sentences = _split_sentences(chunk["text"])
        if not sentences:
            continue
        embeddings = model.encode(sentences, show_progress_bar=False)

        if len(sentences) > 1:
            a, b = embeddings[:-1], embeddings[1:]
            norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            norms[norms == 0] = 1e-10
            intra_sims.extend((np.sum(a * b, axis=1) / norms).tolist())

        if prev_last_embedding is not None:
            first = embeddings[0]
            denom = np.linalg.norm(prev_last_embedding) * np.linalg.norm(first)
            boundary_sims.append(float(np.dot(prev_last_embedding, first) / (denom + 1e-10)))

        prev_last_embedding = embeddings[-1]

    return {
        "avg_intra_chunk_similarity": float(np.mean(intra_sims)) if intra_sims else None,
        "avg_boundary_similarity": float(np.mean(boundary_sims)) if boundary_sims else None,
    }


def run_benchmark():
    files = find_corpus_files()
    print(f"Running benchmark across {len(files)} documents.\n")

    per_doc_rows = []
    coherence_results = {name: [] for name in CHUNKERS}

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path}", end=" ", flush=True)
        try:
            cleaned = clean_text(ingest_document(str(path))["full_text"]) or ""
        except Exception as e:
            print(f"-- skipped (extraction error: {e})")
            continue

        if not cleaned.strip():
            print("-- skipped (empty text)")
            continue

        for name, fn in CHUNKERS.items():
            start = time.time()
            chunks = fn(cleaned)
            elapsed = time.time() - start

            token_counts = [c["token_count"] for c in chunks]
            per_doc_rows.append({
                "filename": path.name,
                "chunker": name,
                "elapsed_seconds": round(elapsed, 4),
                "num_chunks": len(chunks),
                "avg_chunk_tokens": round(float(np.mean(token_counts)), 1) if token_counts else 0,
                "median_chunk_tokens": round(float(np.median(token_counts)), 1) if token_counts else 0,
                "std_chunk_tokens": round(float(np.std(token_counts)), 1) if token_counts else 0,
                "boundary_quality": round(boundary_quality(chunks), 3),
            })

            if i <= COHERENCE_SAMPLE_SIZE:
                coherence_results[name].append(semantic_coherence(chunks))

        print("done")

    # --- Aggregate per chunker ---
    summary = {}
    for name in CHUNKERS:
        rows = [r for r in per_doc_rows if r["chunker"] == name]
        coh = [c for c in coherence_results[name] if c["avg_intra_chunk_similarity"] is not None]

        summary[name] = {
            "documents_processed": len(rows),
            "total_time_seconds": round(sum(r["elapsed_seconds"] for r in rows), 3),
            "avg_time_per_doc_seconds": round(float(np.mean([r["elapsed_seconds"] for r in rows])), 4),
            "total_chunks": sum(r["num_chunks"] for r in rows),
            "avg_chunk_tokens": round(float(np.mean([r["avg_chunk_tokens"] for r in rows])), 1),
            "avg_std_chunk_tokens": round(float(np.mean([r["std_chunk_tokens"] for r in rows])), 1),
            "avg_boundary_quality": round(float(np.mean([r["boundary_quality"] for r in rows])), 3),
            "avg_intra_chunk_similarity": round(float(np.mean([c["avg_intra_chunk_similarity"] for c in coh])), 4) if coh else None,
            "avg_boundary_similarity": round(float(np.mean([c["avg_boundary_similarity"] for c in coh if c["avg_boundary_similarity"] is not None])), 4) if any(c["avg_boundary_similarity"] is not None for c in coh) else None,
            "coherence_sample_size": len(coh),
        }

    # --- Save raw + summary ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "per_document_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_doc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_doc_rows)

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    _make_chart(summary)

    print(f"\nFull results: {RESULTS_DIR / 'per_document_results.csv'}")
    print(f"Summary: {RESULTS_DIR / 'summary.json'}")
    print(f"Chart: {RESULTS_DIR / 'comparison_chart.png'}")


def _print_summary(summary: dict):
    print(f"\n{'='*70}")
    print(f"{'Metric':<30}{'Fixed':<14}{'Recursive':<14}{'Semantic':<14}")
    print("-" * 70)
    rows = [
        ("Total time (s)", "total_time_seconds"),
        ("Avg time/doc (s)", "avg_time_per_doc_seconds"),
        ("Total chunks", "total_chunks"),
        ("Avg chunk tokens", "avg_chunk_tokens"),
        ("Avg std within doc", "avg_std_chunk_tokens"),
        ("Boundary quality", "avg_boundary_quality"),
        ("Intra-chunk similarity", "avg_intra_chunk_similarity"),
        ("Cross-boundary similarity", "avg_boundary_similarity"),
    ]
    for label, key in rows:
        vals = [summary[name].get(key) for name in ("fixed", "recursive", "semantic")]
        print(f"{label:<30}" + "".join(f"{str(v):<14}" for v in vals))


def _make_chart(summary: dict):
    names = list(CHUNKERS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].bar(names, [summary[n]["avg_time_per_doc_seconds"] for n in names])
    axes[0, 0].set_title("Avg time per document (s)")

    axes[0, 1].bar(names, [summary[n]["avg_chunk_tokens"] for n in names])
    axes[0, 1].set_title("Avg chunk size (tokens)")

    axes[1, 0].bar(names, [summary[n]["avg_boundary_quality"] for n in names])
    axes[1, 0].set_title("Boundary quality (fraction ending at sentence end)")
    axes[1, 0].set_ylim(0, 1)

    intra = [summary[n]["avg_intra_chunk_similarity"] or 0 for n in names]
    cross = [summary[n]["avg_boundary_similarity"] or 0 for n in names]
    x = np.arange(len(names))
    width = 0.35
    axes[1, 1].bar(x - width/2, intra, width, label="intra-chunk")
    axes[1, 1].bar(x + width/2, cross, width, label="cross-boundary")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_title("Semantic coherence")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_chart.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_benchmark()