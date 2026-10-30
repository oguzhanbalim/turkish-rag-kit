import unicodedata

import pytest

from turkish_rag_kit import ascii_fold, fold, upper


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("İSTANBUL", "istanbul"),
        ("ISPARTA", "ısparta"),
        ("IŞIK", "ışık"),
        ("SINIR", "sınır"),
        ("SİNİR", "sinir"),
        ("Çığ", "çığ"),
        ("kâr", "kar"),
        ("Âlim", "alim"),
    ],
)
def test_fold(raw, expected):
    assert fold(raw) == expected


def test_python_lower_is_actually_wrong_here():
    # The behaviour this package exists to avoid; if this ever changes, the README is stale.
    assert "İSTANBUL".lower() == "i̇stanbul"
    assert "İSTANBUL".lower() != "istanbul"
    assert "SINIR".lower() == "sinir"


def test_fold_repairs_damaged_and_decomposed_input():
    assert fold("i̇stanbul") == "istanbul"
    assert fold(unicodedata.normalize("NFD", "İSTANBUL")) == "istanbul"
    assert fold(unicodedata.normalize("NFD", "şöğüç")) == "şöğüç"


def test_fold_is_idempotent():
    for word in ["İSTANBUL", "IŞIK", "Çığ", "kâr", "i̇"]:
        assert fold(fold(word)) == fold(word)


def test_upper():
    assert upper("istanbul") == "İSTANBUL"
    assert upper("ışık") == "IŞIK"
    assert fold(upper("sınır sinir")) == "sınır sinir"


def test_ascii_fold_conflates_on_purpose():
    assert ascii_fold("ŞİRKET") == "sirket"
    assert ascii_fold("sınır") == ascii_fold("sinir") == "sinir"
