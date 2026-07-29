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
from config import DECOMPOSE_TEMPERATURE, DECOMPOSE_MAX_TOKENS, DECOMPOSE_MODEL_TIER

DECOMPOSE_SYSTEM_PROMPT = """You determine whether a search query asks about
ONE fact or MULTIPLE distinct facts about the same or related subjects.

If the query asks about only one fact or topic, output exactly:
SINGLE

If the query asks about two or more distinct facts, split it into separate,
self-contained sub-queries — one per fact. Each sub-query must be phrased as
a concrete, DECLARATIVE STATEMENT anchored in specific nouns from the
original query, NOT as a formal question. Declarative statements retrieve
better because they are closer to how facts are actually phrased in source
text, whereas formal questions ("did X happen?") are further from typical
prose phrasing and retrieve worse even when factually equivalent.

Example:
Query: "What was Meursault's job and did his employer offer him a new position?"
Output:
Meursault's job and occupation
Meursault's employer offering him a new position in Paris

Output ONLY "SINGLE", or the sub-queries one per line with no numbering,
no preamble, no extra text."""


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