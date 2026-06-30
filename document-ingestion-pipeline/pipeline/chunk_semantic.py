"""Semantic chunking: split text into sentences, embed each one, and
place chunk boundaries where sentence-to-sentence similarity drops --
i.e. where the topic actually shifts -- rather than at a fixed token
count or structural separator.

Guardrails:
  - A hard token ceiling (chunk_size): a chunk is force-flushed once it
    hits this limit, regardless of similarity.
  - Oversized single sentences (rare, but possible with run-on prose or
    garbled OCR/extraction text missing punctuation) are split further
    using token-window slicing before being added -- without this, one
    huge "sentence" could produce a chunk that violates chunk_size
    entirely, bypassing the ceiling check above it.
  - A minimum token floor before a semantic breakpoint is allowed to
    trigger a flush, so noisy short sentences can't fragment into many
    tiny, near-useless chunks.

Known limitation, not fixed here: similarity is computed only between
*adjacent* sentence pairs. A more robust version would compare rolling
windows of several sentences instead of single pairs, which is less
sensitive to one oddly-phrased sentence creating a false breakpoint.
Worth noting as future work rather than implementing given the time
available -- the adjacent-pair approach is the standard, simplest
version of this technique.

Sentence splitting is a lightweight regex, not a trained segmenter --
it will occasionally mis-split on abbreviations (e.g. "Mr." or "U.S.").
Documented, not chased further given the time available.

Token offsets (start_token/end_token) describe each chunk's own text,
not a position in the original document. With overlap, "position in
the original" is genuinely ambiguous -- overlapping content exists in
two chunks at once, so it has two valid positions. Reporting them as
chunk-relative is honest about that, rather than implying a precision
that overlap makes impossible to guarantee.
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.chunk_fixed import _get_encoding

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _token_count(text: str) -> int:
    return len(_get_encoding().encode(text))


def _split_sentences(text: str) -> list[str]:
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s for s in sentences if s.strip()]


def _split_oversized_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Fall back to token-window slicing for a single sentence that's
    already bigger than chunk_size on its own -- without this, such a
    sentence would produce a chunk violating the size ceiling entirely,
    since the normal flush-before-adding check only catches oversized
    *additions*, not an oversized single piece.
    """
    encoding = _get_encoding()
    tokens = encoding.encode(sentence)
    return [encoding.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]


def _cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between each consecutive pair of embeddings.
    Returns an array of length len(embeddings) - 1.
    """
    a = embeddings[:-1]
    b = embeddings[1:]
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    norms[norms == 0] = 1e-10
    return np.sum(a * b, axis=1) / norms


def _get_token_tail(text: str, n: int) -> str:
    if n <= 0 or not text:
        return ""
    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= n:
        return text
    return encoding.decode(tokens[-n:])


def chunk_semantic(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    std_multiplier: float = 1.0,
    min_chunk_tokens: int = 128,
) -> list[dict]:
    """Split text into chunks at semantic (topic-shift) boundaries.

    Returns the same shape as chunk_fixed_size and chunk_recursive: a
    list of dicts with chunk_index, text, token_count, start_token,
    end_token (chunk-relative -- see module docstring).
    """
    if not text:
        return []

    raw_sentences = _split_sentences(text)
    if len(raw_sentences) <= 1:
        token_count = _token_count(text)
        return [{"chunk_index": 0, "text": text, "token_count": token_count, "start_token": 0, "end_token": token_count}]

    # Pre-split any sentence that's already too big on its own. This
    # changes sentence count, so similarity is computed on this final
    # list, not the raw one -- a sentence sliced into windows no longer
    # has one meaningful embedding anyway.
    sentences = []
    for s in raw_sentences:
        if _token_count(s) > chunk_size:
            sentences.extend(_split_oversized_sentence(s, chunk_size))
        else:
            sentences.append(s)

    embeddings = _get_model().encode(sentences, show_progress_bar=False)
    similarities = _cosine_similarities(embeddings)

    threshold = similarities.mean() - std_multiplier * similarities.std()
    is_breakpoint = similarities < threshold

    chunks = []
    buffer = []
    buffer_tokens = 0
    buffer_token_overlap_seed = ""
    overlap_seed_tokens = 0  # token count of the seed, reserved against chunk_size

    def flush():
        nonlocal buffer, buffer_tokens, buffer_token_overlap_seed, overlap_seed_tokens
        body = " ".join(buffer)
        chunk_text = f"{buffer_token_overlap_seed} {body}" if buffer_token_overlap_seed else body
        chunks.append(chunk_text)
        buffer_token_overlap_seed = _get_token_tail(chunk_text, overlap)
        overlap_seed_tokens = _token_count(buffer_token_overlap_seed)
        buffer = []
        buffer_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = _token_count(sentence)

        # Reserve room for the overlap seed that flush() will prepend --
        # without this, a buffer could fill all the way to chunk_size on
        # sentences alone, then silently exceed it once the previous
        # chunk's overlap tail gets glued on top.
        if buffer_tokens + sentence_tokens + overlap_seed_tokens > chunk_size and buffer:
            flush()

        buffer.append(sentence)
        buffer_tokens += sentence_tokens

        is_last_sentence = i == len(sentences) - 1
        if not is_last_sentence and is_breakpoint[i] and buffer_tokens + overlap_seed_tokens >= min_chunk_tokens:
            flush()

    if buffer:
        flush()

    result = []
    for i, chunk_text in enumerate(chunks):
        token_count = _token_count(chunk_text)
        result.append({
            "chunk_index": i,
            "text": chunk_text,
            "token_count": token_count,
            "start_token": 0,
            "end_token": token_count,
        })

    return result


if __name__ == "__main__":
    import sys
    from pipeline.ingest import ingest_document
    from pipeline.clean import clean_text

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.chunk_semantic <path-to-document>")
        sys.exit(1)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cleaned = clean_text(ingest_document(sys.argv[1])["full_text"]) or ""
    chunks = chunk_semantic(cleaned)

    print(f"Number of chunks: {len(chunks)}")
    if chunks:
        print(f"\n--- First chunk ---")
        print(f"Token count: {chunks[0]['token_count']}")
        print(f"Text preview: {chunks[0]['text'][:300]}...")
        if len(chunks) > 1:
            print(f"\n--- Second chunk ---")
            print(f"Token count: {chunks[1]['token_count']}")
            print(f"Text preview: {chunks[1]['text'][:300]}...")

            end_tokens = _get_encoding().encode(chunks[0]["text"])[-50:]
            start_tokens = _get_encoding().encode(chunks[1]["text"])[:50]
            print(f"\nOverlap check (token-level): {'match' if end_tokens == start_tokens else 'no overlap at boundary (may be expected -- see BPE re-tokenization note in chat)'}")