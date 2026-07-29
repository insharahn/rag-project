# guardrails/llm_judge.py
"""
LLM-based semantic security judge
slower, more expensive but semantically capable check for attack patterns 
that structural regex/ classifier tools cannot reliably catch: n
arrative/roleplay-framed jailbreaks and dialogue-format toxicity
"""
from llm.client import call_llm
from config import JUDGE_SYSTEM_PROMPT, JUDGE_TEMPERATURE, JUDGE_RETRIES

def judge_query(query: str) -> bool:
    """Returns True if the query is judged unsafe (jailbreak attempt)."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = call_llm(messages, temperature=JUDGE_TEMPERATURE, max_retries=JUDGE_RETRIES)
    return response.strip().upper() == "UNSAFE"