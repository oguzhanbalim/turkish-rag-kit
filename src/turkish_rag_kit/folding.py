"""Turkish case folding.

Python's str.lower() is locale-free: "I" becomes "i" (Turkish wants "ı") and "İ" becomes
"i" plus a combining dot above, U+0307, which is not a word character. So "İSTANBUL"
lowers to a string that matches neither "istanbul" nor even a single \\w+ token.
"""

from __future__ import annotations

import unicodedata

_COMBINING_DOT = "̇"
_ASCII_TABLE = str.maketrans("çğıöşüâîû", "cgiosuaiu")
_CIRCUMFLEX_TABLE = str.maketrans("âîû", "aiu")
_UPPER_TABLE = str.maketrans({"i": "İ", "ı": "I"})


def fold(text: str) -> str:
    """Lowercase with Turkish rules: I -> ı, İ -> i. Also NFC-normalises and removes circumflex.

    Decomposed input is handled: "I" + U+0307 composes to "İ" under NFC before lowering, and
    an already-damaged "i" + U+0307 (the output of str.lower) is repaired.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("İ", "i").replace("I", "ı").lower().replace(_COMBINING_DOT, "")
    return text.translate(_CIRCUMFLEX_TABLE)


def upper(text: str) -> str:
    """Uppercase with Turkish rules: i -> İ, ı -> I."""
    return text.translate(_UPPER_TABLE).upper()


def ascii_fold(text: str) -> str:
    """fold() and then strip Turkish letters: for queries typed on a non-Turkish keyboard.

    This conflates real word pairs (sınır/sinir, kırk/kirk), so it is off by default.
    """
    return fold(text).translate(_ASCII_TABLE)
