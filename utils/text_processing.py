"""Text helpers: SQuAD-style normalisation, EM, F1, sentence splitting."""
import re
import string
from collections import Counter

NOT_FOUND = "Answer not found"
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = set(string.punctuation)


def normalize_answer(text) -> str:
    """Lowercase, remove punctuation, remove articles, collapse spaces."""
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def get_tokens(text) -> list:
    norm = normalize_answer(text)
    return norm.split() if norm else []


def exact_match(prediction, reference) -> int:
    """1 if normalised strings are identical, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction, reference) -> float:
    """Token-level F1 (precision/recall on overlapping tokens)."""
    pred, ref = get_tokens(prediction), get_tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def is_not_found(answer) -> bool:
    norm = normalize_answer(answer)
    return norm == "" or norm.startswith("answer not found")


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    return len((text or "").split())


def clean_text(text: str) -> str:
    return " ".join((text or "").split())
