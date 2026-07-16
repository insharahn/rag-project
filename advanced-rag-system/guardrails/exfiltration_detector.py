# guardrails/exfiltration_detector.py
"""
Data exfiltration detector: regex/keyword-based detection for attempts to
extract system prompt content, credentials/secrets, or bulk raw corpus
content rather than a normal answer.

No model required — this is a fast, deterministic first-pass filter,
designed to run on every incoming query before retrieval.
"""
import re
from dataclasses import dataclass, field


@dataclass
class ExfiltrationResult:
    is_exfiltration_attempt: bool
    category: str | None       # 'system_prompt' | 'credentials' | 'indirection' | 'bulk_corpus' | None
    matched_pattern: str | None
    language: str | None


# Category 1: System prompt extraction
SYSTEM_PROMPT_PATTERNS = {
    "en": [
        r"\b(reveal|show|print|repeat|output|tell me|give me)\b.{0,20}\b(your|the)\b.{0,20}\b(system prompt|instructions|initial prompt|prompt above)\b",
        r"\brepeat (everything|all text|the text)\b.{0,20}\b(above|before)\b",
        r"\bwhat (are|were) your (instructions|rules|guidelines)\b",
        r"\bignore (previous|all|the above)\b.{0,20}\binstructions\b",
        r"\bwhat is (written |contained )?in your (system prompt|configuration)\b",
        r"\bwhat (instructions|were you (told|given))\b.{0,20}\b(before|prior to)\b.{0,20}\b(this conversation|talking|started)\b",
        r"\bwhat.{0,15}\b(your )?(original |initial )?(setup|config|configuration)\b.{0,15}\blook(s)? like\b",
        r"\bforget (everything|all)\b.{0,15}\b(before|prior)\b.{0,15}\b(act like|pretend|become)\b",
    ],
    "ko": [
        r"(시스템\s*프롬프트|초기\s*프롬프트).{0,15}(알려|보여|출력|말해)",
        r"(당신의|너의)\s*(지시사항|규칙|설정).{0,15}(알려|보여|뭐)",
        r"이전\s*지시.{0,10}(무시|출력|보여)",
    ],
    "ur": [
        r"(سسٹم\s*پرامپٹ|ابتدائی\s*ہدایات).{0,15}(بتائیں|دکھائیں)",
        r"(آپ\s*کی|تمہاری)\s*(ہدایات|قواعد).{0,15}(بتائیں|کیا ہیں)",
        r"پچھلی\s*ہدایات.{0,10}(نظر انداز|بتائیں)",
    ],
}

# Category 2: Credential / secret extraction
CREDENTIAL_PATTERNS = {
    "en": [
        r"\b(what|show|print|reveal|give me)\b.{0,15}\b(api key|api token|secret key|access token|env(ironment)? variable)\b",
        r"\bwhat (model|llm|provider) (are you|is this) (running on|using)\b",
        r"\b(print|dump|show) (your |the )?(config|configuration|credentials)\b",
    ],
    "ko": [
        r"(API\s*키|비밀\s*키|엑세스\s*토큰).{0,15}(뭐|알려|보여)",
        r"(어떤|무슨)\s*(모델|LLM).{0,10}(사용|돌아가)",
    ],
    "ur": [
        r"(اے پی آئی\s*کی|رازداری\s*کی چابی).{0,15}(بتائیں|دکھائیں)",
    ],
}

# Category 3: Indirection / bypass attempts (disguised requests for the same info)
INDIRECTION_PATTERNS = {
    "en": [
        r"\btranslate\b.{0,20}\b(your instructions|system prompt|the above)\b",
        r"\b(summarize|paraphrase|rewrite)\b.{0,20}\b(your instructions|system prompt|your own rules)\b",
        r"\boutput\b.{0,20}\b(your instructions|system prompt)\b.{0,20}\b(base64|rot13|reversed|backwards)\b",
        r"\bpretend (you are|to be)\b.{0,30}\b(no restrictions|without rules|unrestricted)\b",
    ],
    "ko": [
        r"(지시사항|시스템\s*프롬프트).{0,15}(번역|요약|바꿔)",
    ],
    "ur": [
        r"(ہدایات).{0,15}(ترجمہ|خلاصہ)",
    ],
}

# Category 4: Bulk corpus extraction (RAG-specific)
BULK_CORPUS_PATTERNS = {
    "en": [
        r"\boutput\b.{0,20}\b(all|every|the full|raw)\b.{0,20}\b(chunk|document|context|text)\b",
        r"\b(dump|export|list)\b.{0,20}\b(all|every)\b.{0,20}\b(document|chunk|file)\b.{0,20}\bcorpus\b",
        r"\bgive me\b.{0,15}\b(full|complete|raw)\b.{0,15}\btext of\b.{0,15}\b(every|all)\b",
    ],
    "ko": [
        r"(모든|전체)\s*(청크|문서|텍스트).{0,15}(출력|보여|다)",
    ],
    "ur": [
        r"(تمام|سارے)\s*(دستاویزات|متن).{0,15}(دکھائیں|دیں)",
    ],
}

CATEGORY_PATTERNS = {
    "system_prompt": SYSTEM_PROMPT_PATTERNS,
    "credentials": CREDENTIAL_PATTERNS,
    "indirection": INDIRECTION_PATTERNS,
    "bulk_corpus": BULK_CORPUS_PATTERNS,
}


def detect_exfiltration(text: str, language: str | None = None) -> ExfiltrationResult:
    """
    Check text against known exfiltration attack patterns.

    If `language` is given, only that language's patterns are checked
    (faster, more precise). 
    If None, all languages are checked 
    """
    text_lower = text.lower() if language == "en" or language is None else text
    langs_to_check = [language] if language else ["en", "ko", "ur"]

    for category, lang_patterns in CATEGORY_PATTERNS.items():
        for lang in langs_to_check:
            patterns = lang_patterns.get(lang, [])
            for pattern in patterns:
                search_text = text_lower if lang == "en" else text
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    return ExfiltrationResult(
                        is_exfiltration_attempt=True,
                        category=category,
                        matched_pattern=pattern,
                        language=lang,
                    )

    return ExfiltrationResult(
        is_exfiltration_attempt=False,
        category=None,
        matched_pattern=None,
        language=None,
    )