"""
citation_generator.py — generates an answer with inline citations
[1], [2], etc. mapped back to source chunks.
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm
from config import CITATION_SYSTEM_PROMPT, CITATION_CONFIDENCE_THRESHOLD, CITATION_TEMPERATURE

CONFIDENCE_THRESHOLD = CITATION_CONFIDENCE_THRESHOLD  # chance to hallucinate below this

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