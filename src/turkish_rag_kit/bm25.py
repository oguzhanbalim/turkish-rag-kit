"""In-memory Okapi BM25 with no dependencies. Exhaustive only over documents with a query term."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from .tokenize import Analyzer, turkish_tokens


@dataclass(frozen=True)
class Hit:
    doc_id: Hashable
    score: float
    text: str
    matched_terms: tuple[str, ...]


class BM25Index:
    def __init__(
        self,
        analyzer: Analyzer = turkish_tokens,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._analyzer = analyzer
        self._k1 = k1
        self._b = b
        self._postings: dict[str, dict[int, int]] = {}
        self._lengths: list[int] = []
        self._ids: list[Hashable] = []
        self._texts: list[str] = []
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, doc_id: Hashable, text: str) -> None:
        tokens = self._analyzer(text)
        position = len(self._ids)
        self._ids.append(doc_id)
        self._texts.append(text)
        self._lengths.append(len(tokens))
        self._total_length += len(tokens)
        for term, count in Counter(tokens).items():
            self._postings.setdefault(term, {})[position] = count

    def add_all(self, documents: Iterable[tuple[Hashable, str]]) -> None:
        for doc_id, text in documents:
            self.add(doc_id, text)

    def _idf(self, term: str) -> float:
        document_frequency = len(self._postings.get(term, ()))
        n = len(self._ids)
        return math.log((n - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self._ids:
            return []
        average_length = self._total_length / len(self._ids) or 1.0
        scores: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        for term in dict.fromkeys(self._analyzer(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for position, tf in postings.items():
                length_norm = 1 - self._b + self._b * self._lengths[position] / average_length
                gain = idf * tf * (self._k1 + 1) / (tf + self._k1 * length_norm)
                scores[position] = scores.get(position, 0.0) + gain
                matched.setdefault(position, []).append(term)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            Hit(self._ids[pos], score, self._texts[pos], tuple(matched[pos]))
            for pos, score in ranked
        ]
