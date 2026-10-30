"""Sentence-aware chunking. Chunks keep character offsets into the source for citation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .folding import fold

# Abbreviations whose period does not end a sentence. Compared in folded form.
_ABBREVIATIONS = frozenset(
    fold(a)
    for a in """
    dr prof doç yrd av uzm op müh sn bkz vb vs örn yy no nr tel faks mah cad sok apt blv
    ltd şti a.ş t.c s.a st m.ö m.s hz
    """.split()
)
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
_UPPER_START = re.compile(r"[\"'“‘(\[]*[A-ZÇĞİÖŞÜ0-9]")
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int
    index: int


def _is_false_boundary(text: str, match: re.Match[str]) -> bool:
    """True when this punctuation does not end a sentence."""
    if match.group() != ".":
        return False
    before = text[: match.start()]
    word = before.rsplit(maxsplit=1)[-1] if before.split() else ""
    word = word.lstrip("(\"'“‘[")
    if not word:
        return False
    # Ordinals and list numbers: "15. madde", "3. Bölüm".
    if word.isdigit():
        return True
    # A single capital is an initial: "A. Yılmaz".
    if len(word) == 1 and word.isalpha() and word == word.upper():
        return True
    if fold(word) in _ABBREVIATIONS:
        return True
    following = text[match.end() :].lstrip()
    # A lowercase continuation is not a new sentence.
    return bool(following) and _UPPER_START.match(following) is None


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentences. Paragraph breaks always end a sentence."""
    spans: list[tuple[int, int]] = []
    paragraph_start = 0
    paragraphs = [m.start() for m in _PARAGRAPH.finditer(text)] + [len(text)]
    for paragraph_end in paragraphs:
        segment_start = paragraph_start
        for match in _BOUNDARY.finditer(text, paragraph_start, paragraph_end):
            if _is_false_boundary(text[: paragraph_end], match):
                continue
            spans.append((segment_start, match.end()))
            segment_start = match.end()
        spans.append((segment_start, paragraph_end))
        paragraph_start = paragraph_end
    return [_strip_span(text, s, e) for s, e in spans if text[s:e].strip()]


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def _hard_split(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split one oversized sentence at whitespace, or mid-word if there is none."""
    pieces = []
    while end - start > max_chars:
        cut = text.rfind(" ", start + 1, start + max_chars + 1)
        if cut <= start:
            cut = start + max_chars
        pieces.append(_strip_span(text, start, cut))
        start = cut
    pieces.append(_strip_span(text, start, end))
    return [p for p in pieces if p[0] < p[1]]


def chunk_text(text: str, max_chars: int = 800, overlap_sentences: int = 1) -> list[Chunk]:
    """Pack whole sentences into chunks of at most max_chars.

    Consecutive chunks share the last `overlap_sentences` sentences when that still fits.
    A sentence longer than max_chars is the only thing that gets split.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must not be negative")

    sentences: list[tuple[int, int]] = []
    for start, end in sentence_spans(text):
        sentences.extend(_hard_split(text, start, end, max_chars))

    chunks: list[Chunk] = []
    i = 0
    while i < len(sentences):
        first = i
        chunk_start = sentences[first][0]
        last = first
        while last + 1 < len(sentences) and sentences[last + 1][1] - chunk_start <= max_chars:
            last += 1
        chunk_end = sentences[last][1]
        chunks.append(Chunk(text[chunk_start:chunk_end], chunk_start, chunk_end, len(chunks)))
        if last + 1 >= len(sentences):
            break
        # Carry the overlap only if the next chunk still has room for a new sentence;
        # otherwise it would be a subset of this one. Always advance at least one sentence.
        next_new = sentences[last + 1][1]
        i = max(first + 1, last + 1 - overlap_sentences)
        while i <= last and next_new - sentences[i][0] > max_chars:
            i += 1
    return chunks
