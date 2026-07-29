"""
multi_query.py: generate multiple query variants from a rewritten
query to cast a wider net during retrieval.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm
from config import MULTI_QUERY_SYSTEM_PROMPT

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