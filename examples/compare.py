"""Compare three analyzers on the toy corpus. Prints real retrieval output.

An expected value of "-" means the corpus has nothing relevant, so returning no result is
the correct behaviour.

Usage: python examples/compare.py
"""

from __future__ import annotations

from pathlib import Path

from turkish_rag_kit import ANALYZERS, BM25Index

HERE = Path(__file__).resolve().parent
MODES = ["naive", "folded", "turkish"]


def build(mode: str) -> BM25Index:
    index = BM25Index(ANALYZERS[mode])
    for path in sorted((HERE / "corpus").glob("*.txt")):
        index.add(path.name, path.read_text(encoding="utf-8"))
    return index


def main() -> None:
    indexes = {mode: build(mode) for mode in MODES}
    queries = [
        line.split("\t")
        for line in (HERE / "queries.tsv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wins = dict.fromkeys(MODES, 0)
    for query, expected in queries:
        print(f"query: {query!r}  (expected {expected})")
        for mode in MODES:
            hits = indexes[mode].search(query, top_k=1)
            if hits:
                top = hits[0]
                ok = top.doc_id == expected
                wins[mode] += ok
                terms = ", ".join(top.matched_terms)
                print(
                    f"  {mode:<8}{'ok ' if ok else 'BAD'} {top.doc_id:<22}"
                    f"{top.score:5.2f}  [{terms}]"
                )
            else:
                ok = expected == "-"
                wins[mode] += ok
                print(f"  {mode:<8}{'ok ' if ok else 'BAD'} (no result)")
    print()
    print(f"correct top result, {len(queries)} queries, {len(indexes['naive'])} documents:")
    for mode in MODES:
        print(f"  {mode:<8}{wins[mode]:>3}/{len(queries)}")


if __name__ == "__main__":
    main()
