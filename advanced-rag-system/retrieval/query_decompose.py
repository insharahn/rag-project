# retrieval/query_decompose.py
"""
Detects whether a query asks about multiple distinct facts (compound),
and if so, splits it into separate, declarative sub-queries phrased close
to how the fact would appear in source text — not as formal questions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm
from config import DECOMPOSE_SYSTEM_PROMPT, DECOMPOSE_TEMPERATURE, DECOMPOSE_MAX_TOKENS, DECOMPOSE_MODEL_TIER

def decompose_query(query: str) -> list[str]:
    """Returns [query] unchanged if single-fact, or a list of declarative
    sub-queries if compound. Always returns at least one query."""
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = call_llm(messages, temperature=DECOMPOSE_TEMPERATURE, max_tokens=DECOMPOSE_MAX_TOKENS, model=DECOMPOSE_MODEL_TIER)

    if response.strip().upper() == "SINGLE":
        return [query]

    sub_queries = [line.strip() for line in response.split("\n") if line.strip()]
    return sub_queries if sub_queries else [query]