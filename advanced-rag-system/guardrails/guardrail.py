# guardrails/guardrail.py
"""
Unified guardrail: runs all input-side checks on a query before retrieval,
and output-side checks on the generated answer before it's returned.

Categories covered:
  - injection / direct jailbreak -> Prompt Guard 2 (meta-llama/Llama-Prompt-Guard-2-86M)
  - jailbreak (roleplay/persona-override) -> custom regex
  - toxicity -> textdetox classifier + KO/UR regex
  - data exfiltration -> custom regex (guardrails/exfiltration_detector.py)
  - PII -> Presidio (EN) + custom regex (guardrails/pii_detector.py)
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
from config import INJECTION_THRESHOLD
_injection_classifier = hf_pipeline(
    "text-classification", model="meta-llama/Llama-Prompt-Guard-2-86M"
)
INJECTION_LABEL_MAP = {"label_1": "malicious", "label_0": "benign"}

"""
tuned via scripts/sweep_injection_threshold.py against
the 125-prompt eval set — swept 0.5 to 0.99; recall was
early flat across the whole range (0.245–0.277), meaning
the classifier's score distribution is bimodal rather than
threshold-sensitive. 0.5 gives the best recall at
essentially no precision cost (0.963 vs 0.958 at 0.9).
"""

INJECTION_THRESHOLD = INJECTION_THRESHOLD 

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

def check_input_deep(text: str, language: str = "en") -> GuardrailResult:
    """
    Fast checks first, then an LLM semantic judge as a second pass —
    only reached if the fast layer found nothing. Catches narrative/
    roleplay jailbreaks the fast layer structurally cannot.
    """
    fast_result = check_input(text, language=language)
    if fast_result.blocked:
        return fast_result  # fast layer already caught it, skip the LLM call

    from guardrails.llm_judge import judge_query
    if judge_query(text):
        return GuardrailResult(
            blocked=True,
            reasons=["jailbreak_narrative_llm"],
            details={"detection_method": "llm_judge"},
        )

    return fast_result  # confirmed clean by both layers

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