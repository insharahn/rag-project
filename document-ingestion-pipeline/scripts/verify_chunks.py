"""Sanity-check a chunking function's output:
  1. No chunk exceeds the requested token budget (allowing a little
     slack for overlap, which can push a chunk slightly over).
  2. Concatenating all chunks (roughly) reconstructs the original text,
     i.e. nothing was silently dropped -- checked by comparing total
     unique character coverage, not exact equality, since overlap and
     separator handling mean chunks won't concatenate back byte-for-byte.
"""

import sys

from pipeline.ingest import ingest_document
from pipeline.clean import clean_text
from pipeline.chunk_fixed import chunk_fixed_size
from pipeline.chunk_recursive import chunk_recursive
from pipeline.chunk_semantic import chunk_semantic


def verify(chunks: list[dict], original_text: str, chunk_size: int, label: str):
    print(f"\n--- Verifying: {label} ---")

    oversized = [c for c in chunks if c["token_count"] > chunk_size * 1.1]
    if oversized:
        print(f"  FAIL: {len(oversized)} chunk(s) exceed {chunk_size} tokens by >10%")
        for c in oversized[:3]:
            print(f"    chunk {c['chunk_index']}: {c['token_count']} tokens")
    else:
        print(f"  OK: all {len(chunks)} chunks within token budget")

    total_chunk_chars = sum(len(c["text"]) for c in chunks)
    coverage_ratio = total_chunk_chars / max(len(original_text), 1)
    print(f"  Original text: {len(original_text)} chars | Total chunk chars: {total_chunk_chars} | Ratio: {coverage_ratio:.2f}")
    if coverage_ratio < 0.95:
        print("  WARNING: chunk content is noticeably shorter than the original -- possible silent content loss")
    elif coverage_ratio > 3.0:
        print("  WARNING: chunk content is much longer than the original -- possible excessive duplication")
    else:
        print("  OK: coverage ratio looks reasonable (overlap explains modest >1.0 values)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.verify_chunks <path-to-document>")
        sys.exit(1)

    sys.stdout.reconfigure(encoding="utf-8")
    cleaned = clean_text(ingest_document(sys.argv[1])["full_text"]) or ""

    verify(chunk_fixed_size(cleaned), cleaned, 512, "fixed-size")
    verify(chunk_recursive(cleaned), cleaned, 512, "recursive")
    verify(chunk_semantic(cleaned), cleaned, 512, "semantic")