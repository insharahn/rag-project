# guardrails/toxicity_detector.py
"""
Toxicity detection: hybrid approach.
- English: textdetox/bert-multilingual-toxicity-classifier.
- Korean/Urdu: same classifier PLUS a regex supplement
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
    source: str # 'model' | 'regex' | 'model+regex'
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
        # discriminatory generalization: "[group] should be [excluded]
        # because [stereotype]" — hate speech stated as factual claim,
        # not direct insult
        r"(장애인|여자|남자|노인|이 나라 사람들|그 사람들).{0,20}(다|전부|원래).{0,20}(못하|안 되|게으르|멍청|짐이)",
        r"(격리|배제|없애|제거).{0,10}(시켜야|해야)",
        r"(태어나지|살 가치가).{0,10}(말았어야|없)",
        r"(장애인|여자|남자|노인|이 나라 사람들|그 사람들).{0,30}(다|전부|원래).{0,20}(못하|안 되|게으르|멍청|바보|짐이|미신)",
    ],
    "ur": [
        r"(بیوقوف|احمق).{0,15}(ہو|ہیں)",                        # "you are stupid/foolish"
        r"(ختم|مر).{0,10}(ہو جانا|جانی).{0,10}چاہیے",           # "should die/end"
        r"(تم سب|تم لوگ).{0,15}(بیوقوف|احمق|بے کار)",           # "you all are [insult]"
        # discriminatory generalization: "[group] ... [incapable/burden] ...
        # should [be separated/excluded]" — same pattern class as Korean
        r"(معذور|عورتیں|بوڑھے|تم جیسے).{0,60}(کام کے نہیں|نہیں چلا سکتیں|بوجھ ہیں|پیدا ہی نہیں).{0,20}(چاہیے|جاؤ)",
        r"(الگ کر دینا|الگ کرنا).{0,15}چاہیے",
        r"(چاہیے تھا|ہونا چاہیے تھا).{0,15}(ختم|بس)",
    ],
}

MODEL_TOXIC_THRESHOLD = 0.5


def _regex_check(text: str, language: str) -> bool:
    patterns = TOXIC_PATTERNS.get(language, [])
    return any(re.search(p, text) for p in patterns)


def detect_toxicity(text: str, language: str = "en") -> ToxicityResult:
    result = _classifier(text, truncation=True, max_length=512)[0]
    raw_label = result["label"].lower()
    is_toxic_label = "toxic" in raw_label or raw_label in ("1", "label_1")

    # Use the threshold explicitly: only trust the "toxic" label if the
    # model's confidence actually clears the threshold, rather than
    # accepting any confidence level just because the label says toxic.
    model_says_toxic = is_toxic_label and result["score"] >= MODEL_TOXIC_THRESHOLD
    model_score = result["score"] if is_toxic_label else 1 - result["score"]

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