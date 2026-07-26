# agents/validation_agent.py
"""
Validation agent — independently checks the summarization agent's draft
answer against the retrieved chunks it was supposedly grounded in.
Distinct from generate_answer()'s confidence score (which reflects
retrieval quality) — this checks whether the answer text itself is
actually supported by what was cited.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm

VALIDATION_SYSTEM_PROMPT = """You are a fact-checking validator for a RAG
system. You will be given a user's question, a draft answer, and the
source chunks the answer was supposedly based on.

Check three things:
1. GROUNDED: Does every factual claim in the answer actually appear in
   or follow directly from the source chunks? Flag any claim that seems
   invented or not supported.
2. CITED_CORRECTLY: Do the citation numbers in the answer point to chunks
   that actually support the claim next to them?
3. ADDRESSES_QUERY: Does the answer actually respond to what was asked,
   or does it dodge, partially answer, or answer a different question?

Output in this exact format, nothing else:
GROUNDED: yes/no
CITED_CORRECTLY: yes/no
ADDRESSES_QUERY: yes/no
ISSUES: <brief description of any problems found, or "none">"""


def validation_node(state: dict) -> dict:
    draft = state.get("draft_answer", "")
    sources = state.get("draft_sources", {})
    chunks = state.get("retrieved_chunks", [])

    if not draft or not sources:
        return {
            **state,
            "validation_passed": False,
            "validation_issues": "No draft answer or sources to validate.",
        }

    # build cid -> text lookup from the actual retrieved chunks, since
    # draft_sources only carries display metadata, not full chunk text
    text_by_cid = {}
    for cid, chunk, _score in chunks:
        text_by_cid[cid] = chunk["text"] if isinstance(chunk, dict) else chunk

    source_text_parts = []
    for n, src in sources.items():
        cid = src.get("chunk_id", "")
        text = text_by_cid.get(cid, "")
        source_text_parts.append(f"[{n}] {src.get('source_doc', '')}: {text[:500]}")
    source_text = "\n\n".join(source_text_parts)

    user_content = (
        f"Question: {state['query']}\n\n"
        f"Draft answer: {draft}\n\n"
        f"Source chunks:\n{source_text}"
    )

    messages = [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = call_llm(messages, temperature=0.0)

    grounded = "GROUNDED: yes" in response
    cited_correctly = "CITED_CORRECTLY: yes" in response
    addresses_query = "ADDRESSES_QUERY: yes" in response
    issues_line = next((l for l in response.split("\n") if l.startswith("ISSUES:")), "ISSUES: none")

    passed = grounded and cited_correctly and addresses_query

    return {
        **state,
        "validation_passed": passed,
        "validation_grounded": grounded,
        "validation_cited_correctly": cited_correctly,
        "validation_addresses_query": addresses_query,
        "validation_issues": issues_line.replace("ISSUES:", "").strip(),
    }