# config.py
import os
from dotenv import load_dotenv
load_dotenv()

def _float(key, default):
    return float(os.environ.get(key, default))

def _int(key, default):
    return int(os.environ.get(key, default))

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