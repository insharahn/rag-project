"""
client.py — shared LLM client (OpenRouter or Groq, switchable via env var)
with retry logic + basic output validation.
"""
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError

load_dotenv()

# switch providers with LLM_PROVIDER=groq or LLM_PROVIDER=openrouter in .env
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

_PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "models": {
            "large": "openai/gpt-oss-120b", #"large": "llama-3.3-70b-versatile", "large": "openai/gpt-oss-120b", "large": "qwen/qwen3.6-27b"
            "small": "llama-3.1-8b-instant",
        },
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": {
            "large": "openrouter/free", #same for both :>
            "small": "openrouter/free",  
        },
    },
}

_config = _PROVIDER_CONFIG[LLM_PROVIDER]

client = OpenAI(
    base_url=_config["base_url"],
    api_key=os.environ.get(_config["api_key_env"]),
)

MODELS = _config["models"]
DEFAULT_MODEL = MODELS["large"]  # kept for compatibility if anything still imports this directly


def _resolve_model(model):
    """Accepts a logical tier ("large"/"small"), a raw model string, or None.
    Lets call sites say model="small" instead of a provider-specific string
    that breaks the moment you switch providers."""
    if model is None:
        return MODELS["large"]
    if model in MODELS:
        return MODELS[model]
    return model  # assume it's already a valid model string for the active provider


def _looks_valid(response_text: str) -> bool:
    if not response_text or len(response_text.strip()) < 3:
        return False
    bad_patterns = ["user safety:", "i cannot assist", "i'm unable to help with that"]
    lowered = response_text.lower()
    if any(p in lowered for p in bad_patterns) and len(response_text) < 60:
        return False
    return True


def call_llm(messages, model=None, max_retries=3, temperature=0.3, timeout=20, max_tokens=None):
    resolved_model = _resolve_model(model)
    last_error = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=resolved_model, messages=messages, temperature=temperature, timeout=timeout)
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            raw_content = response.choices[0].message.content
            text = raw_content.strip() if raw_content is not None else ""

            if not _looks_valid(text):
                wait = 5 * (attempt + 1)
                print(f"[llm] Suspicious/empty output, retrying in {wait}s... (attempt {attempt+1}/{max_retries}): {text[:80]!r}")
                last_error = f"invalid output: {text[:80]!r}"
                time.sleep(wait)
                continue

            return text

        except RateLimitError as e:
            last_error = e
            wait = 8 * (attempt + 1)
            print(f"[llm] Rate limited, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except APITimeoutError as e:
            last_error = e
            print(f"[llm] Timeout, retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(3)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")