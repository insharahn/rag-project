"""
query_rewrite.py: rewrite a raw user query into a cleaner search
query before retrieval. 
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm
from config import REWRITE_SYSTEM_PROMPT, REWRITE_TEMPERATURE

def rewrite_query(raw_query: str, history: list[dict] | None = None) -> str:
    history_context = ""
    if history:
        recent = history[-4:]
        history_context = "\n".join(f"{h['role']}: {h['content']}" for h in recent)
        history_context = f"Recent conversation:\n{history_context}\n\n"

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT + (
            "\n\nIf the query references something from the conversation history "
            "using a pronoun or implicit reference, resolve it into an explicit, "
            "self-contained search query using the conversation context provided. "
            "If the query asks to re-explain or elaborate on something already "
            "discussed, rewrite it as a query for that same specific topic, reusing "
            "concrete terms from the prior answer. If the query is already "
            "self-contained, ignore the history."
            if history else ""
        )},
        {"role": "user", "content": f"{history_context}Query: {raw_query}"},
    ]
    rewritten = call_llm(messages, temperature=REWRITE_TEMPERATURE)

    # Guard against the rewriter answering/hedging instead of rewriting.
    # A real search query should not contain first-person refusal language
    # or claims about corpus contents.
    BAD_PATTERNS = ["cannot", "corpus doesn't", "corpus does not", "i don't have",
                     "i'm unable", "no information", "not contain"]
    if any(p in rewritten.lower() for p in BAD_PATTERNS) or len(rewritten) > 300:
        print(f"[WARN] rewrite looked like a refusal, falling back to raw query: {rewritten!r}")
        return raw_query  # fall back to the original query untouched

    return rewritten