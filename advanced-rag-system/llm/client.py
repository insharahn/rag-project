"""
client.py — shared OpenRouter LLM client with retry logic.
Every LLM call in this project (query rewriting, multi-query, citation
generation) goes through call_llm() so retry/fallback behavior is consistent.
"""
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

DEFAULT_MODEL = "openrouter/free"   # auto-router, avoids single-model congestion


def call_llm(messages, model=DEFAULT_MODEL, max_retries=3, temperature=0.3):
    """Call the LLM with retry on rate limits / timeouts.
    Returns the response text (str)."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            last_error = e
            wait = 25 * (attempt + 1)
            print(f"[llm] Rate limited, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except APITimeoutError as e:
            last_error = e
            print(f"[llm] Timeout, retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(5)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")