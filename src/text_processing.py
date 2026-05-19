from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.tokenize import wordpunct_tokenize
except Exception:  # pragma: no cover
    wordpunct_tokenize = None


STOPWORDS = set(ENGLISH_STOP_WORDS)
STOPWORDS -= {"no", "not", "nor", "never", "none", "cannot", "can't", "won't", "don't", "didn't", "isn't", "wasn't", "aren't", "couldn't", "shouldn't", "wouldn't"}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s']+")
SPACE_PATTERN = re.compile(r"\s+")
REPEATED_CHARS = re.compile(r"([a-z]){2,}")


def _normalize_repeated_letters(text: str) -> str:
    return REPEATED_CHARS.sub(r"", text)


def tokenize(text: str):
    text = text or ""
    if wordpunct_tokenize is not None:
        return wordpunct_tokenize(text)
    return re.findall(r"[A-Za-z']+", text)


def preprocess_text(text: str) -> str:
    """Lowercase, remove URLs/mentions/punctuation, tokenize, and remove stopwords."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = _normalize_repeated_letters(text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = SPACE_PATTERN.sub(" ", text).strip()
    tokens = tokenize(text)
    cleaned = [tok.strip("'") for tok in tokens if tok.strip("'") and tok not in STOPWORDS and len(tok.strip("'")) > 1]
    return " ".join(cleaned)


def preprocess_series(texts: Iterable[str]):
    return [preprocess_text(t) for t in texts]


def get_top_n_words(texts, n=15):
    from collections import Counter

    counter = Counter()
    for txt in texts:
        counter.update(preprocess_text(txt).split())
    return counter.most_common(n)
