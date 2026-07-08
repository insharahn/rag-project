"""
multi_query.py — task 2: generate multiple query variants from a rewritten
query to cast a wider net during retrieval. Variants explore different
angles/phrasings/possible resolutions of ambiguous references, so retrieval
isn't betting everything on one phrasing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm

MULTI_QUERY_SYSTEM_PROMPT = """You generate query variants for a retrieval system
over a specific document corpus.

Given a search query, produce {n} different variants that explore different
angles of the same underlying question. Variants should:
- Preserve the original intent — do not change what is being asked.
- Vary in phrasing, specificity, or possible interpretation of ambiguous terms
  (e.g. if a query references an unnamed character/entity by description,
  one variant may attempt a plausible specific resolution while another stays
  general).
- NOT invent specific facts, titles, or names with high confidence — plausible
  guesses are fine as ONE variant among several, not asserted as certain.
- Aim for a natural, specific sentence or phrase, not a 2-3 word fragment.
- Each variant must be a single, self-contained search query.
- Ground variants in the original question's intent — if the rewritten query dropped
or changed a specific detail/noun, restore it.

Output ONLY the {n} variants, one per line, no numbering, no preamble, no extra text."""


def generate_query_variants(raw_query: str, rewritten_query: str, n: int = 3) -> list[str]:
    user_content = f"Original question: {raw_query}\nRewritten query: {rewritten_query}"
    messages = [
        {"role": "system", "content": MULTI_QUERY_SYSTEM_PROMPT.format(n=n)},
        {"role": "user", "content": user_content},
    ]
    response = call_llm(messages)
    variants = [line.strip() for line in response.split("\n") if line.strip()]
    if rewritten_query not in variants:
        variants.insert(0, rewritten_query)
    return variants[:n + 1]