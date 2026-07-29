# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def _float(key, default):
    return float(os.environ.get(key, default))

def _int(key, default):
    return int(os.environ.get(key, default))

def _load_prompt(env_key, default_path):
    path = Path(os.environ.get(env_key, default_path))
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.read_text(encoding="utf-8").strip()

# thresholds
RESEARCH_LOW_CONFIDENCE_THRESHOLD = _float("RESEARCH_LOW_CONFIDENCE_THRESHOLD", 0.5)
CITATION_CONFIDENCE_THRESHOLD = _float("CITATION_CONFIDENCE_THRESHOLD", 0.5)

# validation agent LLM params
VALIDATION_TEMPERATURE = _float("VALIDATION_TEMPERATURE", 0.0)
VALIDATION_MAX_TOKENS = _int("VALIDATION_MAX_TOKENS", 80)
VALIDATION_MODEL_TIER = os.environ.get("VALIDATION_MODEL_TIER", "small")

# decompose LLM params
DECOMPOSE_TEMPERATURE = _float("DECOMPOSE_TEMPERATURE", 0.1)
DECOMPOSE_MAX_TOKENS = _int("DECOMPOSE_MAX_TOKENS", 120)
DECOMPOSE_MODEL_TIER = os.environ.get("DECOMPOSE_MODEL_TIER", "small")

# citation generation
CITATION_TEMPERATURE = _float("CITATION_TEMPERATURE", 0.2)

#rewrite llm call
REWRITE_TEMPERATURE = _float("REWRITE_TEMPERATURE", 0.1)

#multi query llm call
MULTI_QUERY_TEMPERATURE = _float("MULTI_QUERY_TEMPERATURE", 0.3)

#guardrails llm judge call
JUDGE_TEMPERATURE = _float("JUDGE_TEMPERATURE", 0.0)
JUDGE_RETRIES = _int("JUDGE_RETRIES", 2)

#default llm call 
MAX_RETRIES = _int("MAX_RETRIES", 3)
DEFAULT_TEMPERATURE = _float("DEFAULT_TEMPERATURE", 0.3)
DEFAULT_TIMEOUT = _int("DEFAULT_TIMEOUT", 20)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

# request timeouts (seconds)
RETRIEVAL_TIMEOUT = _int("RETRIEVAL_TIMEOUT", 120)
GENERATION_TIMEOUT = _int("GENERATION_TIMEOUT", 60)
AGENT_QUERY_TIMEOUT = _int("AGENT_QUERY_TIMEOUT", 180)

#guardrail injection threshold
INJECTION_THRESHOLD = _float("INJECTION_THRESHOLD", 0.5)

#system prompts
VALIDATION_SYSTEM_PROMPT = _load_prompt("VALIDATION_PROMPT_PATH", "prompts/validation_system.txt")
DECOMPOSE_SYSTEM_PROMPT = _load_prompt("DECOMPOSE_PROMPT_PATH", "prompts/decompose_system.txt")
CITATION_SYSTEM_PROMPT = _load_prompt("CITATION_PROMPT_PATH", "prompts/citation_system.txt")
JUDGE_SYSTEM_PROMPT= _load_prompt("JUDGE_PROMPT_PATH", "prompts/llm_judge_system.txt")
REWRITE_SYSTEM_PROMPT= _load_prompt("QUERY_REWRITE_PROMPT_PATH", "prompts/query_rewrite_system.txt")
MULTI_QUERY_SYSTEM_PROMPT = _load_prompt("MULTI_QUERY_PROMPT_PATH", "prompts/multi_query_system.txt")

#origin strings
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
).split(",")