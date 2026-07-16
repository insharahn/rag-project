# guardrails/toxicity_detector.py
"""
Toxicity detection: hybrid approach.
- English: textdetox/bert-multilingual-toxicity-classifier alone (tested 3/3).
- Korean/Urdu: same classifier PLUS a regex supplement, since smoke testing
  confirmed the model misses real toxic content in these languages
  (defaults to 'neutral' rather than false-flagging — the worse failure
  direction for a safety layer). Same reasoning as the exfiltration
  detector's design.

CONFIDENCE NOTE: Korean patterns are reasonably confident. Urdu patterns
should be reviewed by a native speaker before production use — same
caveat applied throughout this project's Urdu regex work.
"""
import re
from dataclasses import dataclass
from transformers import pipeline

_classifier = pipeline(
    "text-classification",
    model="textdetox/bert-multilingual-toxicity-classifier",
)


@dataclass
class ToxicityResult:
    is_toxic: bool
    source: str        # 'model' | 'regex' | 'model+regex'
    model_score: float
    language: str


# Harassment/dehumanizing phrase patterns — supplementing the model's
# confirmed gap on KO/UR toxic content specifically (not benign content,
# which the model handled correctly in testing).
TOXIC_PATTERNS = {
    "ko": [
        r"(멍청|바보|병신).{0,10}(해야|되어야|없어져)",       # "should be [insult]/disappear" pattern
        r"(죽어|없어져).{0,10}(버려|야)",                        # "should die/disappear"
        r"(너희|니들|저것들).{0,10}(다|모두).{0,10}(멍청|쓸모없)", # "you're all [insult]"
    ],
    "ur": [
        # NOTE: needs native-speaker verification before production use
        r"(بیوقوف|احمق).{0,15}(ہو|ہیں)",                        # "you are stupid/foolish"
        r"(ختم|مر).{0,10}(ہو جانا|جانی).{0,10}چاہیے",           # "should die/end"
        r"(تم سب|تم لوگ).{0,15}(بیوقوف|احمق|بے کار)",           # "you all are [insult]"
    ],
}

MODEL_TOXIC_THRESHOLD = 0.5


def _regex_check(text: str, language: str) -> bool:
    patterns = TOXIC_PATTERNS.get(language, [])
    return any(re.search(p, text) for p in patterns)


def detect_toxicity(text: str, language: str = "en") -> ToxicityResult:
    result = _classifier(text)[0]
    raw_label = result["label"].lower()
    model_says_toxic = "toxic" in raw_label or raw_label in ("1", "label_1")
    model_score = result["score"] if model_says_toxic else 1 - result["score"]

    regex_hit = False
    if language in ("ko", "ur"):
        regex_hit = _regex_check(text, language)

    is_toxic = model_says_toxic or regex_hit

    if model_says_toxic and regex_hit:
        source = "model+regex"
    elif regex_hit:
        source = "regex"
    elif model_says_toxic:
        source = "model"
    else:
        source = "none"

    return ToxicityResult(
        is_toxic=is_toxic,
        source=source,
        model_score=round(model_score, 3),
        language=language,
    )