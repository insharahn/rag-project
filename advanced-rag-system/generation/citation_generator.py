"""
citation_generator.py — task 5: generates an answer with inline citations
[1], [2], etc. mapped back to source chunks. Uses the reranker's top score
as a confidence gate — below threshold, the system hedges/refuses rather
than let the LLM confidently answer off weak grounding (validated need for
this: see "the outsider guy's job" query, which retrieved nothing above 0.35
rerank score because no chunk actually states the answer).
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm

CONFIDENCE_THRESHOLD = 0.5  # below this top rerank score, hedge instead of answer

CITATION_SYSTEM_PROMPT = """You are a question-answering assistant that MUST
ground every claim in the provided context chunks.

Rules:
- Answer using ONLY information in the context chunks below.
- Cite every claim inline using the chunk's bracketed number, e.g. [1], [2].
- If multiple chunks support a claim, cite all of them, e.g. [1][3].
- Do NOT state anything not directly supported by the context.
- If the context does not contain enough information to answer the question,
  say so plainly instead of guessing or filling gaps with outside knowledge.
- Keep the answer concise and directly responsive to the question."""


def _format_context(chunks: list[tuple[str, dict, float]]) -> str:
    lines = []
    for i, (cid, chunk, score) in enumerate(chunks, 1):
        lines.append(f"[{i}] (source: {chunk.get('source_doc', cid)})\n{chunk['text']}")
    return "\n\n".join(lines)


def _build_source_map(chunks: list[tuple[str, dict, float]]) -> dict:
    return {
        i: {"chunk_id": cid, "source_doc": chunk.get("source_doc", cid), "rerank_score": score}
        for i, (cid, chunk, score) in enumerate(chunks, 1)
    }


def _extract_cited_indices(answer: str) -> set:
    """Finds all [n] citation markers actually used in the answer text."""
    matches = re.findall(r"\[(\d+)\]", answer)
    return {int(m) for m in matches}


def generate_answer(raw_query: str, retrieved_chunks: list[tuple[str, dict, float]]) -> dict:
    if not retrieved_chunks:
        return {
            "answer": "The corpus doesn't contain information relevant to this question.",
            "sources": {},
            "confidence": "low",
            "top_score": 0.0,
        }

    top_score = retrieved_chunks[0][2]

    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": (
                "The corpus doesn't clearly address this question — the most "
                "relevant passages found don't directly answer it. Try rephrasing, "
                "or this information may not be present in the corpus."
            ),
            "sources": _build_source_map(retrieved_chunks),
            "confidence": "low",
            "top_score": top_score,
        }

    context = _format_context(retrieved_chunks)
    messages = [
        {"role": "system", "content": CITATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {raw_query}"},
    ]
    answer = call_llm(messages, temperature=0.2)

    full_sources = _build_source_map(retrieved_chunks)
    cited_indices = _extract_cited_indices(answer)
    cited_sources = {i: src for i, src in full_sources.items() if i in cited_indices}

    return {
        "answer": answer,
        "sources": cited_sources,
        "confidence": "high",
        "top_score": top_score,
    }