# guardrails/guardrail.py
"""
Unified guardrail: runs all input-side checks on a query before retrieval,
and output-side checks on the generated answer before it's returned.

Categories covered:
  - injection / direct jailbreak -> Prompt Guard 2 (meta-llama/Llama-Prompt-Guard-2-86M)
  - jailbreak (roleplay/persona-override) -> custom regex, supplementing a
    confirmed gap where Prompt Guard 2 missed this specific pattern in KO/UR
  - toxicity                -> textdetox classifier + KO/UR regex supplement
  - data exfiltration       -> custom regex (guardrails/exfiltration_detector.py)
  - PII                     -> Presidio (EN) + custom regex (guardrails/pii_detector.py),
    blocked on both input and output — even a user's own PII is not
    passed through to retrieval/generation.

All checks run once per request, not per query-variant/chunk — kept
lightweight and fast relative to the LLM-bound stages of the pipeline.
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import pipeline as hf_pipeline
from guardrails.exfiltration_detector import detect_exfiltration
from guardrails.pii_detector import detect_pii
from guardrails.toxicity_detector import detect_toxicity
from guardrails.jailbreak_detector import detect_jailbreak_roleplay

_injection_classifier = hf_pipeline(
    "text-classification", model="meta-llama/Llama-Prompt-Guard-2-86M"
)
INJECTION_LABEL_MAP = {"label_1": "malicious", "label_0": "benign"}

# NOTE: this threshold is a placeholder, not yet tuned against real data.
# Revisit once the 100+ prompt eval set gives real precision/recall
# numbers at different thresholds.
INJECTION_THRESHOLD = 0.9


@dataclass
class GuardrailResult:
    blocked: bool
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def check_input(text: str, language: str = "en") -> GuardrailResult:
    reasons = []
    details = {}

    # 1. Injection / direct jailbreak (classifier)
    inj_result = _injection_classifier(text, truncation=True, max_length=512)[0]
    inj_label = INJECTION_LABEL_MAP.get(inj_result["label"].lower(), "benign")
    if inj_label == "malicious" and inj_result["score"] >= INJECTION_THRESHOLD:
        reasons.append("injection_or_jailbreak")
        details["injection_score"] = round(inj_result["score"], 3)

    # 1b. Jailbreak roleplay/persona-override (regex supplement)
    roleplay = detect_jailbreak_roleplay(text, language=language)
    if roleplay.is_jailbreak_attempt:
        reasons.append("jailbreak_roleplay")
        details["jailbreak_pattern_matched"] = True

    # 2. Data exfiltration
    exfil = detect_exfiltration(text, language=language)
    if exfil.is_exfiltration_attempt:
        reasons.append(f"exfiltration:{exfil.category}")
        details["exfiltration_category"] = exfil.category

    # 3. Toxicity
    tox = detect_toxicity(text, language=language)
    if tox.is_toxic:
        reasons.append("toxicity")
        details["toxicity_source"] = tox.source
        details["toxicity_score"] = tox.model_score

    # 4. PII —  blocking on input as well as output 
    # even a user's own PII shouldn't flow into retrieval/generation
    # unnecessarily
    pii = detect_pii(text, language=language)
    if pii.has_pii:
        reasons.append("pii_detected")
        details["pii_entities"] = pii.entities

    return GuardrailResult(blocked=bool(reasons), reasons=reasons, details=details)


def check_output(text: str, language: str = "en") -> GuardrailResult:
    """Checks the generated answer before it's returned to the user —
    catches cases where retrieved chunk content successfully injected
    something into the answer, or the answer leaks PII/secrets."""
    reasons = []
    details = {}

    exfil = detect_exfiltration(text, language=language)
    if exfil.is_exfiltration_attempt:
        reasons.append(f"exfiltration:{exfil.category}")
        details["exfiltration_category"] = exfil.category

    pii = detect_pii(text, language=language)
    if pii.has_pii:
        reasons.append("pii_leak")
        details["pii_entities"] = pii.entities

    return GuardrailResult(blocked=bool(reasons), reasons=reasons, details=details)