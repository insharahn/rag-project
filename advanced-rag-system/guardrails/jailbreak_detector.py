# guardrails/jailbreak_detector.py
"""
Jailbreak roleplay/persona-override detector: regex supplement to
Prompt Guard 2 cause it fails on urdu/korean roleplay jailbreaks
"""
import re
from dataclasses import dataclass

@dataclass
class JailbreakResult:
    is_jailbreak_attempt: bool
    matched_pattern: str | None
    language: str


ROLEPLAY_JAILBREAK_PATTERNS = {
    "en": [
        r"\b(pretend|act as if|imagine)\b.{0,20}\byou (are|have)\b.{0,25}\b(no restrictions|no rules|unrestricted|no limitations|no filters)\b",
        r"\byou are now\b.{0,20}\b(DAN|an AI (with|that has) no|a different AI|unrestricted)\b",
        r"\b(from now on|starting now)\b.{0,20}\byou (are|will be|have)\b.{0,25}\b(no rules|unrestricted|free from)\b",
        r"\bignore your (safety|ethical) (guidelines|restrictions|training)\b",
        r"\bin this (roleplay|scenario|hypothetical)\b.{0,25}\b(no rules|anything goes|no restrictions)\b",
    ],
    "ko": [
        r"(제약|규칙|제한)이?\s*없는\s*(AI|인공지능).{0,15}(척|되어|해줘)",
        r"(AI|인공지능).{0,15}(척|되어).{0,15}(제약|규칙|제한)이?\s*없",  # reverse order too
        r"이제\s*(부터|는).{0,15}(자유|규칙\s*없이|제한\s*없이)",
        r"(안전|윤리)\s*(지침|가이드라인).{0,15}(무시|없애)",
    ],
    "ur": [
        # split into two directions since "become X" and "no restrictions"
        # can appear in either order depending on sentence structure
        r"(بن جائیں|ہو جائیں).{0,20}(کوئی\s*پابندی|قواعد).{0,15}(نہیں|بغیر)",
        r"(کوئی\s*پابندی|قواعد).{0,15}(نہیں|بغیر).{0,20}(بن جائیں|ہو جائیں)",
        r"اب\s*(سے|آپ).{0,15}(آزاد|بغیر\s*قواعد)",
        r"(حفاظتی|اخلاقی)\s*(ہدایات|رہنما اصول).{0,15}(نظر انداز|ہٹا)",
    ],
}


def detect_jailbreak_roleplay(text: str, language: str | None = None) -> JailbreakResult:
    """
    Check text against known roleplay/persona-override jailbreak patterns.
    Supplements Prompt Guard 2
    """
    langs_to_check = [language] if language else ["en", "ko", "ur"]

    for lang in langs_to_check:
        patterns = ROLEPLAY_JAILBREAK_PATTERNS.get(lang, [])
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return JailbreakResult(
                    is_jailbreak_attempt=True,
                    matched_pattern=pattern,
                    language=lang,
                )

    return JailbreakResult(is_jailbreak_attempt=False, matched_pattern=None, language=None)