"""
bm25_tokenizer.py — language-aware tokenization for BM25.
BM25 has no language logic of its own; it just scores pre-tokenized token
lists. This module is where "language-awareness" actually lives.
"""
import re
import unicodedata
from kiwipiepy import Kiwi

_kiwi = Kiwi()

HANGUL_RE = re.compile(r'[\uac00-\ud7a3]')
ARABIC_RE = re.compile(r'[\u0600-\u06ff\u0750-\u077f]')  # covers Urdu's Arabic-script range


def detect_script(text: str) -> str:
    """Rough script detection for queries (no precomputed language label)."""
    if HANGUL_RE.search(text):
        return "ko"
    if ARABIC_RE.search(text):
        return "ur"
    return "en"


def char_ngrams(text: str, n: int = 3) -> list[str]:
    text = text.replace(" ", "")
    return [text[i:i+n] for i in range(len(text) - n + 1)] if len(text) >= n else [text]


def tokenize_en(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def tokenize_ko(text: str) -> list[str]:
    morphs = [token.form for token in _kiwi.tokenize(text)]
    return morphs + char_ngrams(text)


def tokenize_ur(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    words = re.findall(r"[\w]+", normalized, re.UNICODE)
    return words + char_ngrams(normalized)


def tokenize(text: str, language: str = None) -> list[str]:
    """Route to the right tokenizer. `language` is the corpus metadata field
    for docs; if None (e.g. for queries), script is auto-detected."""
    lang = (language or detect_script(text)).lower()
    if lang.startswith("ko"):
        return tokenize_ko(text)
    if lang.startswith("ur"):
        return tokenize_ur(text)
    return tokenize_en(text)