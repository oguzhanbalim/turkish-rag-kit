"""Command line: index a folder of .txt files in memory and run one query."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bm25 import BM25Index
from .chunk import chunk_text
from .tokenize import ANALYZERS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turkish-rag", description=__doc__)
    parser.add_argument("corpus", type=Path, help="folder of UTF-8 .txt files")
    parser.add_argument("query")
    parser.add_argument("--analyzer", choices=sorted(ANALYZERS), default="turkish")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=400)
    args = parser.parse_args(argv)

    index = BM25Index(ANALYZERS[args.analyzer])
    for path in sorted(args.corpus.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(text, max_chars=args.max_chars):
            index.add(f"{path.name}#{chunk.index}", chunk.text)

    hits = index.search(args.query, top_k=args.top)
    if not hits:
        print("no match")
    for hit in hits:
        snippet = " ".join(hit.text.split())[:110]
        print(f"{hit.score:6.2f}  {hit.doc_id}  [{', '.join(hit.matched_terms)}]")
        print(f"        {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
