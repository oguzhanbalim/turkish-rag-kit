import math

from examples.compare import build
from turkish_rag_kit import BM25Index, folded_tokens, naive_tokens, turkish_tokens


def test_score_matches_hand_calculation():
    # Two documents, one term, k1=1.5, b=0.75.
    index = BM25Index(str.split)
    index.add("a", "x y")
    index.add("b", "z z z z")
    hit = index.search("x")[0]
    idf = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1)
    norm = 1 - 0.75 + 0.75 * 2 / 3
    assert hit.doc_id == "a"
    assert math.isclose(hit.score, idf * 1 * 2.5 / (1 + 1.5 * norm))


def test_more_occurrences_score_higher_and_length_penalised():
    index = BM25Index(str.split)
    index.add("short", "x")
    index.add("repeat", "x x a b")
    index.add("long", "x a b c d e f g h i j k")
    ranked = [h.doc_id for h in index.search("x")]
    assert ranked[-1] == "long"


def test_empty_cases():
    index = BM25Index()
    assert index.search("herhangi") == []
    index.add("a", "kitap")
    assert index.search("") == []
    assert index.search("elma") == []


def test_ranking_is_deterministic_on_ties():
    index = BM25Index(str.split)
    index.add("first", "x")
    index.add("second", "x")
    assert [h.doc_id for h in index.search("x")] == ["first", "second"]


def test_top_k_and_duplicate_query_terms():
    index = BM25Index(str.split)
    for i in range(5):
        index.add(i, "x")
    assert len(index.search("x", top_k=2)) == 2
    assert index.search("x x")[0].score == index.search("x")[0].score


def test_naive_lower_breaks_dotted_capital_i():
    assert naive_tokens("İSTANBUL") == ["i", "stanbul"]
    assert folded_tokens("İSTANBUL") == ["istanbul"]
    assert turkish_tokens("İstanbul'da") == ["istanbul"]


def test_naive_retrieves_nerve_document_for_border_query():
    # The headline failure from the README, kept as a regression test.
    query = "SINIR KAPISI NE ZAMAN AÇIK"
    assert build("naive").search(query)[0].doc_id == "sinir-sistemi.txt"
    assert build("turkish").search(query)[0].doc_id == "sinir-kapisi.txt"


def test_stemmer_finds_inflected_forms_lowercasing_cannot():
    assert build("folded").search("faturaya itiraz") == []
    assert build("turkish").search("faturaya itiraz")[0].doc_id == "fatura.txt"


def test_known_overstemming_false_match():
    # "kapları" (containers) matches "kapısı" (door) in the Turkish analyser. Documented limit.
    assert build("folded").search("saklama kapları") == []
    assert build("turkish").search("saklama kapları")[0].doc_id == "sinir-kapisi.txt"


def test_cli_prints_hits(capsys):
    from pathlib import Path

    from turkish_rag_kit.cli import main

    corpus = Path(__file__).resolve().parent.parent / "examples" / "corpus"
    assert main([str(corpus), "faturaya itiraz", "--top", "1"]) == 0
    assert "fatura.txt#0" in capsys.readouterr().out
    assert main([str(corpus), "zzzz"]) == 0
    assert capsys.readouterr().out.strip() == "no match"
