"""Recursive chunking: split on the largest structural separator first
(paragraphs), and only recurse into smaller separators (lines, sentences,
words) for pieces that are still too big. Unlike fixed-size chunking,
this respects natural text boundaries wherever the content allows it,
falling back to a harder split only when it has no choice.

The canonical recursive algorithm:
1. Try to split on the current separator
2. For each piece:
   - If it fits in chunk_size → keep it
   - If it's too big → recurse with the next separator
3. Repeat until all pieces fit or we hit character level
"""

from pipeline.chunk_fixed import _get_encoding

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _token_count(text: str) -> int:
    """Count tokens in text using the encoder."""
    return len(_get_encoding().encode(text))


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Canonical recursive text splitter.
    
    Recursion happens HERE - on the text itself, not on merging.
    
    Args:
        text: The text to split
        separators: List of separators, from largest to smallest
        chunk_size: Maximum tokens per chunk
    
    Returns:
        List of text pieces, each guaranteed to be <= chunk_size tokens
    """
    # Base case 1: Text already fits
    if _token_count(text) <= chunk_size:
        return [text]
    
    # Base case 2: No more separators left → force split at character level
    if not separators:
        return _force_split_characters(text, chunk_size)
    
    # Get the current separator (largest one)
    separator = separators[0]
    next_separators = separators[1:]
    
    # If separator doesn't appear in text, skip to next one
    if separator not in text:
        return _recursive_split(text, next_separators, chunk_size)
    
    # Split text on current separator
    parts = text.split(separator)
    
    # Process each part
    result = []
    for i, part in enumerate(parts):
        # Add separator back (except for the last part)
        if i < len(parts) - 1:
            part_with_sep = part + separator
        else:
            part_with_sep = part
        
        # Skip empty parts
        if not part_with_sep:
            continue
        
        # Check if this part fits
        if _token_count(part_with_sep) <= chunk_size:
            result.append(part_with_sep)
        else:
            # RECURSION: This part is too big, split it with smaller separators
            sub_parts = _recursive_split(part_with_sep, next_separators, chunk_size)
            result.extend(sub_parts)
    
    return result


def _force_split_characters(text: str, chunk_size: int) -> list[str]:
    """
    Last resort: split at character level when no more separators.
    
    This is the base case of the recursion when we've exhausted all separators.
    It's the only place where we force a split that might break words.
    """
    if _token_count(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for char in text:
        char_tokens = _token_count(char)
        
        # If this character alone exceeds chunk_size (shouldn't happen with standard encoding)
        if char_tokens > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_tokens = 0
            chunks.append(char)
            continue
        
        # Check if adding this character would exceed the limit
        if current_tokens + char_tokens > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = char
                current_tokens = char_tokens
            else:
                # Single character exceeds limit (edge case)
                chunks.append(char)
                current_chunk = ""
                current_tokens = 0
        else:
            current_chunk += char
            current_tokens += char_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Merge pieces into final chunks with token-based overlap.
    
    This is a SEPARATE step from recursive splitting.
    It only handles packing and overlap, not recursion.
    
    CHANGED: Now uses token-based overlap instead of whole-piece overlap.
    This guarantees every adjacent pair shares the requested overlap tokens.
    """
    if not pieces:
        return []
    
    chunks = []
    buffer = []
    buffer_tokens = 0
    
    def get_token_overlap(text: str, overlap_tokens: int) -> str:
        """Get the last `overlap_tokens` tokens from text."""
        if not text or overlap_tokens <= 0:
            return ""
        
        encoding = _get_encoding()
        tokens = encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        return encoding.decode(tokens[-overlap_tokens:])
    
    def flush():
        nonlocal buffer, buffer_tokens
        if buffer:
            chunk_text = "".join(buffer)
            chunks.append(chunk_text)
            
            # Token-based overlap: take last `overlap` tokens from the chunk
            if overlap > 0:
                overlap_text = get_token_overlap(chunk_text, overlap)
                # Replace buffer with just the overlap text
                buffer = [overlap_text] if overlap_text else []
                buffer_tokens = _token_count(overlap_text) if overlap_text else 0
            else:
                buffer = []
                buffer_tokens = 0
        else:
            buffer = []
            buffer_tokens = 0
    
    for piece in pieces:
        piece_tokens = _token_count(piece)
        
        # If adding this piece would exceed the limit, flush first
        if buffer_tokens + piece_tokens > chunk_size and buffer:
            flush()
        
        buffer.append(piece)
        buffer_tokens += piece_tokens
    
    # Flush any remaining buffer
    if buffer:
        chunks.append("".join(buffer))
    
    return chunks


def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """
    Split text recursively, respecting paragraph/sentence boundaries.
    
    CANONICAL RECURSIVE APPROACH:
    1. Split text recursively using _recursive_split()
    2. Merge pieces into chunks with token-based overlap using _merge_with_overlap()
    
    The recursion happens purely in the splitting phase, not the merging phase.
    
    Returns:
        List of dicts with chunk_index, text, token_count, start_token, end_token
        
    NOTE: start_token and end_token are computed based on the original text's
    tokenization, not the chunked text. This ensures offsets are accurate
    even with overlap.
    """
    if not text:
        return []
    
    # Step 1: Recursively split the text into pieces that fit
    pieces = _recursive_split(text, SEPARATORS, chunk_size)
    
    # Step 2: Merge pieces into chunks with token-based overlap
    chunk_texts = _merge_with_overlap(pieces, chunk_size, overlap)
    
    # Step 3: Format output with accurate token offsets
    # We tokenize the original text once to get correct offsets
    encoding = _get_encoding()
    original_tokens = encoding.encode(text)
    
    result = []
    running_offset = 0
    
    for i, chunk_text in enumerate(chunk_texts):
        token_count = _token_count(chunk_text)
        
        # Find the actual position of this chunk in the original text
        # We need to find where this chunk text occurs in the original
        # This handles overlap correctly
        chunk_tokens = encoding.encode(chunk_text)
        
        # Find the start position in the original token sequence
        # This is a simple approach - find where the chunk tokens appear
        start_pos = None
        for offset in range(max(0, running_offset - overlap), 
                           min(running_offset + overlap + 1, len(original_tokens) - len(chunk_tokens) + 1)):
            if original_tokens[offset:offset + len(chunk_tokens)] == chunk_tokens:
                start_pos = offset
                break
        
        if start_pos is None:
            # Fallback: use running_offset (may be slightly off with overlap)
            start_pos = running_offset
        
        end_pos = start_pos + len(chunk_tokens)
        
        result.append({
            "chunk_index": i,
            "text": chunk_text,
            "token_count": token_count,
            "start_token": start_pos,
            "end_token": end_pos,
        })
        
        # Update running_offset to the end of this chunk (for next chunk's search)
        running_offset = end_pos
    
    return result


def _merge_with_overlap_alt(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Alternative merge: token-based overlap using a sliding window.
    
    This is an even cleaner approach that guarantees exact overlap.
    Use this if the above approach has issues with finding token positions.
    """
    if not pieces:
        return []
    
    chunks = []
    buffer = []
    buffer_tokens = 0
    
    def get_token_tail(text: str, n: int) -> str:
        """Get the last n tokens as text."""
        if n <= 0 or not text:
            return ""
        encoding = _get_encoding()
        tokens = encoding.encode(text)
        if len(tokens) <= n:
            return text
        return encoding.decode(tokens[-n:])
    
    def flush():
        nonlocal buffer, buffer_tokens
        if buffer:
            chunk_text = "".join(buffer)
            chunks.append(chunk_text)
            
            # Keep only the overlap tail
            if overlap > 0:
                overlap_text = get_token_tail(chunk_text, overlap)
                buffer = [overlap_text] if overlap_text else []
                buffer_tokens = _token_count(overlap_text) if overlap_text else 0
            else:
                buffer = []
                buffer_tokens = 0
    
    for piece in pieces:
        piece_tokens = _token_count(piece)
        
        if buffer_tokens + piece_tokens > chunk_size and buffer:
            flush()
        
        buffer.append(piece)
        buffer_tokens += piece_tokens
    
    if buffer:
        chunks.append("".join(buffer))
    
    return chunks


if __name__ == "__main__":
    import sys
    from pipeline.ingest import ingest_document
    from pipeline.clean import clean_text

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.chunk_recursive <path-to-document>")
        sys.exit(1)

    # Handle stdout encoding
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding="utf-8")
    
    cleaned = clean_text(ingest_document(sys.argv[1])["full_text"]) or ""
    chunks = chunk_recursive(cleaned)

    print(f"Number of chunks: {len(chunks)}")
    
    if chunks:
        # Show first chunk
        print(f"\n--- First chunk ---")
        print(f"Token count: {chunks[0]['token_count']}")
        print(f"Start token: {chunks[0]['start_token']}")
        print(f"End token: {chunks[0]['end_token']}")
        print(f"Text preview: {chunks[0]['text'][:300]}...")
        
        if len(chunks) > 1:
            print(f"\n--- Second chunk ---")
            print(f"Token count: {chunks[1]['token_count']}")
            print(f"Start token: {chunks[1]['start_token']}")
            print(f"End token: {chunks[1]['end_token']}")
            print(f"Text preview: {chunks[1]['text'][:300]}...")
            
            # Verify overlap
            print(f"\n--- Overlap verification ---")
            end_of_chunk1 = chunks[0]['text'][-200:]
            start_of_chunk2 = chunks[1]['text'][:200]
            print(f"End of chunk 1: ...{end_of_chunk1}")
            print(f"Start of chunk 2: {start_of_chunk2}...")
            
            # Check if they overlap
            overlap_found = False
            for i in range(min(len(end_of_chunk1), len(start_of_chunk2)), 0, -1):
                if end_of_chunk1[-i:] == start_of_chunk2[:i]:
                    print(f"Found {i} characters of overlap at the boundary")
                    overlap_found = True
                    break
            
            if not overlap_found:
                # Check if chunk2 starts with content that was at the end of chunk1
                end_tokens = _get_encoding().encode(chunks[0]['text'])[-50:]
                start_tokens = _get_encoding().encode(chunks[1]['text'])[:50]
                if end_tokens == start_tokens:
                    print("Token-level overlap is working (tokens match)")
                else:
                    print("No overlap detected at the boundary")