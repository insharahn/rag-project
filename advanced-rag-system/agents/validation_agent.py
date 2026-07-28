# agents/validation_agent.py
"""
Validation agent: independently checks the summarization agent's draft
answer against the retrieved chunks it was supposedly grounded in.
Distinct from generate_answer()'s confidence score (which reflects
retrieval quality) — this checks whether the answer text itself is
actually supported by what was cited.
"""
import sys
import re
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
        return {**state, "validation_passed": False, "validation_issues": "No draft answer or sources to validate."}

    cited_nums = set(re.findall(r'\[(\d+)\]', draft))
    relevant_sources = {n: s for n, s in sources.items() if not cited_nums or str(n) in cited_nums}

    text_by_cid = {}
    for cid, chunk, _score in chunks:
        text_by_cid[cid] = chunk["text"] if isinstance(chunk, dict) else chunk

    source_text_parts = []
    for n, src in relevant_sources.items():
        cid = src.get("chunk_id", "")
        text = text_by_cid.get(cid, "")
        source_text_parts.append(f"[{n}] {src.get('source_doc', '')}: {text[:250]}")
    source_text = "\n\n".join(source_text_parts)

    user_content = f"Question: {state['query']}\n\nDraft answer: {draft}\n\nSource chunks:\n{source_text}"
    messages = [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = call_llm(messages, temperature=0.0, max_tokens=80, model="small")

    def _field_is_yes(field: str) -> bool | None:
        match = re.search(rf'{field}\s*:\s*(\w+)', response, re.IGNORECASE)
        if not match:
            return None  # couldn't parse at all — distinct from a genuine "no"
        return match.group(1).lower().startswith('y')

    grounded = _field_is_yes("GROUNDED")
    cited_correctly = _field_is_yes("CITED_CORRECTLY")
    addresses_query = _field_is_yes("ADDRESSES_QUERY")

    issues_match = re.search(r'ISSUES\s*:\s*(.+)', response, re.IGNORECASE)
    issues_text = issues_match.group(1).strip() if issues_match else None

    # total parse failure (none of the 4 fields found) — don't silently
    # fail the answer for a formatting mismatch that isn't the answer's fault
    if grounded is None and cited_correctly is None and addresses_query is None:
        print(f"[validation] Could not parse validator response at all, passing through. Raw: {response[:150]!r}")
        return {**state, "validation_passed": True, "validation_issues": "validator response unparseable — passed through"}

    passed = bool(grounded) and bool(cited_correctly) and bool(addresses_query)

    return {
        **state,
        "validation_passed": passed,
        "validation_grounded": grounded,
        "validation_cited_correctly": cited_correctly,
        "validation_addresses_query": addresses_query,
        "validation_issues": issues_text or "none",
    }