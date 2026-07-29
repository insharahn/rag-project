"""
citation_generator.py — generates an answer with inline citations
[1], [2], etc. mapped back to source chunks.
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm
from config import CITATION_CONFIDENCE_THRESHOLD, CITATION_TEMPERATURE

CONFIDENCE_THRESHOLD = CITATION_CONFIDENCE_THRESHOLD  # chance to hallucinate below this

CITATION_SYSTEM_PROMPT = """You are a question-answering assistant that MUST
ground every claim in the provided context chunks.

Rules:
- Answer using ONLY information in the context chunks below.
- Cite every claim inline using the chunk's bracketed number, e.g. [1], [2].
- If multiple chunks support a claim, cite all of them, e.g. [1][3].
- Do NOT state anything not directly supported by the context — do not invent
  facts, names, or events absent from the chunks.
- You MAY synthesize, summarize, or draw reasonable conclusions (e.g. themes,
  causes, comparisons) by connecting multiple pieces of context together, as
  long as each underlying fact you rely on is grounded in a cited chunk. For
  example, if chunks show a character being punished unfairly and rules being
  changed to benefit those already in power, you may identify this as
  illustrating a theme of "corruption of power" — this is legitimate synthesis,
  not hallucination, and citing the chunks that support each piece of the
  synthesis is sufficient grounding.
- Attempt synthesis before concluding the context is insufficient. Only say
  the context doesn't address the question if reasonable synthesis genuinely
  isn't possible from what's given — do not default to this just because the
  answer requires connecting facts rather than quoting one chunk directly.
- If the user asks you to elaborate, interpret, or "come up with" something
  based on the context already discussed, treat this as a request to attempt
  deeper synthesis on the existing topic — not as a question about your own
  capabilities.
- Keep the answer concise and directly responsive to the question.

After your answer, on a new line, output exactly:
FOLLOWUPS:
Then list exactly 3 short follow-up questions a curious reader might ask
next, based ONLY on topics actually present in the context chunks above
(not general knowledge). One per line, no numbering, no extra text."""


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

def _split_answer_and_followups(raw_response: str) -> tuple[str, list[str]]:
    """Splits the LLM's raw output into the answer text and follow-up
    questions, based on the FOLLOWUPS: marker."""
    if "FOLLOWUPS:" not in raw_response:
        return raw_response.strip(), []

    answer_part, followups_part = raw_response.split("FOLLOWUPS:", 1)
    followups = [line.strip("- ").strip() for line in followups_part.strip().split("\n") if line.strip()]
    return answer_part.strip(), followups[:3]

def generate_answer(raw_query: str, retrieved_chunks: list[tuple[str, dict, float]], feedback: str = None) -> dict:
    if not retrieved_chunks:
        return {
            "answer": "The corpus doesn't contain information relevant to this question.",
            "sources": {},
            "confidence": "low",
            "top_score": 0.0,
            "followup_questions": [],
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
            "followup_questions": [],
        }

    context = _format_context(retrieved_chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {raw_query}"
    if feedback:
        user_content += (
            f"\n\nNote: a previous attempt at answering this question had the "
            f"following problem, which you must fix this time: {feedback}"
        )

    messages = [
        {"role": "system", "content": CITATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    answer_raw = call_llm(messages, temperature=CITATION_TEMPERATURE)
    answer, followups = _split_answer_and_followups(answer_raw)

    full_sources = _build_source_map(retrieved_chunks)
    cited_indices = _extract_cited_indices(answer)
    cited_sources = {i: src for i, src in full_sources.items() if i in cited_indices}

    return {
        "answer": answer,
        "sources": cited_sources,
        "confidence": "high",
        "top_score": top_score,
        "followup_questions": followups,
    }