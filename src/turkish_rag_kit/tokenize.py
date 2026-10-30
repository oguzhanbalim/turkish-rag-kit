"""Tokenisers: the naive baseline, the Turkish analyser, and the mode switch."""

from __future__ import annotations

import re
from collections.abc import Callable

from .folding import ascii_fold, fold
from .stem import stem

_WORD = re.compile(r"\w+(?:['’ʼ]\w+)*", re.UNICODE)
_NAIVE_WORD = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    fold(word)
    for word in """
    ve veya ile de da ki bu şu o bir için ama fakat ancak gibi kadar daha en çok az mi mı mu mü
    ne nasıl nerede neden kim hangi ben sen biz siz onlar var yok olan olarak ise eğer her
    """.split()
)

Analyzer = Callable[[str], list[str]]


def naive_tokens(text: str) -> list[str]:
    """The default most code reaches for: str.lower() and \\w+."""
    return _NAIVE_WORD.findall(text.lower())


def folded_tokens(text: str) -> list[str]:
    """Turkish case folding only. No stemming, no stopwords."""
    return _WORD.findall(fold(text))


def turkish_tokens(text: str, *, use_ascii_fold: bool = False) -> list[str]:
    """Turkish folding, stopword removal and the light stemmer."""
    folded = fold(text)
    tokens = [stem(t) for t in _WORD.findall(folded) if t not in STOPWORDS]
    tokens = [t for t in tokens if t]
    return [ascii_fold(t) for t in tokens] if use_ascii_fold else tokens


ANALYZERS: dict[str, Analyzer] = {
    "naive": naive_tokens,
    "folded": folded_tokens,
    "turkish": turkish_tokens,
}
