"""
query_rewrite.py — task 1: rewrite a raw user query into a cleaner search
query before retrieval. 
Handles vague/colloquial phrasing, expands implicit
context, keeps it as a single focused query (multi-query variants come later).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import call_llm

REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant for a retrieval system
over a specific document corpus.

Rules:
- Preserve the original intent exactly, do not answer the question.
- Remove filler words and fix ambiguity using only what's implied by the query itself.
- Do NOT invent specific titles, names, or facts not stated or clearly implied by the query.
- Keep enough detail for the query to be useful for semantic search — do not
  over-shorten into a bare keyword phrase. Aim for a natural, specific sentence
  or phrase, not a 2-3 word fragment.
- Output ONLY the rewritten query, nothing else. No preamble, no quotes."""


def rewrite_query(raw_query: str) -> str:
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": raw_query},
    ]
    rewritten = call_llm(messages)
    return rewritten