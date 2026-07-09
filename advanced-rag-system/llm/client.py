"""
client.py — shared OpenRouter LLM client with retry logic + basic output validation.
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

DEFAULT_MODEL = "openrouter/free"
#DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
#DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"


def _looks_valid(response_text: str) -> bool:
    """Cheap sanity check, not a quality check. Rejects empty/near-empty
    responses and known-bad boilerplate patterns seen in practice."""
    if not response_text or len(response_text.strip()) < 3:
        return False
    bad_patterns = ["user safety:", "i cannot assist", "i'm unable to help with that"]
    lowered = response_text.lower()
    if any(p in lowered for p in bad_patterns) and len(response_text) < 60:
        # short + matches a known bad-output pattern = likely a misroute,
        # not a genuine refusal of a real harmful request
        return False
    return True


def call_llm(messages, model=DEFAULT_MODEL, max_retries=3, temperature=0.3):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()

            if not _looks_valid(text):
                print(f"[llm] Suspicious output, retrying... (attempt {attempt+1}/{max_retries}): {text[:80]!r}")
                last_error = f"invalid output: {text[:80]!r}"
                continue

            return text

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