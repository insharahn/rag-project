"""Fixed-size chunking: split text into chunks of a fixed token count,
with a configurable overlap between consecutive chunks so context isn't
lost right at the boundary.

This is the simplest of the three chunking strategies: it
has no idea where sentences or paragraphs end, and will cut text
wherever the token count happens to land, mid-sentence or not.
"""

import tiktoken

ENCODING_NAME = "cl100k_base"  # GPT-3.5/4-era tokenizer; a reasonable generic token-count proxy
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(ENCODING_NAME)
    return _encoding


def chunk_fixed_size(text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """Split text into fixed-size token chunks with overlap.

    Returns a list of dicts, each with:
      - chunk_index: position in the sequence (0-indexed)
      - text: the chunk's text
      - token_count: number of tokens in this chunk
      - start_token / end_token: token offsets into the original text
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    encoding = _get_encoding()
    tokens = encoding.encode(text or "")

    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "start_token": start,
            "end_token": end,
        })

        chunk_index += 1
        if end == len(tokens):
            break
        start += step

    return chunks


if __name__ == "__main__":
    import sys
    from pipeline.ingest import ingest_document
    from pipeline.clean import clean_text

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.chunk_fixed <path-to-document>")
        sys.exit(1)

    sys.stdout.reconfigure(encoding="utf-8")
    cleaned = clean_text(ingest_document(sys.argv[1])["full_text"]) or ""
    chunks = chunk_fixed_size(cleaned)

    print(f"Total tokens: {len(_get_encoding().encode(cleaned))}")
    print(f"Number of chunks: {len(chunks)}")
    if chunks:
        print(f"\n--- First chunk ---\n{chunks[0]['text'][:300]}")
        if len(chunks) > 1:
            print(f"\n--- Second chunk (note overlap with end of first) ---\n{chunks[1]['text'][:300]}")