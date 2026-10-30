"""A light rule-based suffix stripper for Turkish nouns.

This is not a morphological analyser. It peels common inflectional suffixes (plural,
possessive, case) from the end of a word. The result is a retrieval key, not a dictionary
form: "masa" and "masalarda" both become "mas". Verbs are not analysed, and derivational
suffixes are left in place on purpose.

Precision is protected by a short word list rather than by analysis, so it will be wrong
on plenty of words outside that list. See the README's Limits and tests/test_stem.py.
"""

from __future__ import annotations

from .folding import fold

VOWELS = frozenset("aeıioöuü")
_APOSTROPHES = "'’ʼ`"
_MAX_ROUNDS = 6

# Longest suffix first within a group, and each entry says what must precede it:
#   "V" a vowel (buffer-consonant forms), "H" a high vowel (the -n- that only follows a
#   possessive: evi-nde, but zaman-da), "C" a consonant, "*" anything.
_SUFFIXES: tuple[tuple[str, str], ...] = (
    # Relative -ki, usually on a locative.
    ("daki", "*"), ("deki", "*"), ("taki", "*"), ("teki", "*"), ("ndaki", "H"), ("ndeki", "H"),
    # Ablative and locative, with the -n- buffer after a possessive.
    ("ndan", "H"), ("nden", "H"), ("nda", "H"), ("nde", "H"),
    ("dan", "*"), ("den", "*"), ("tan", "*"), ("ten", "*"),
    ("da", "*"), ("de", "*"), ("ta", "*"), ("te", "*"),
    # Genitive.
    ("nın", "V"), ("nin", "V"), ("nun", "V"), ("nün", "V"),
    ("ın", "C"), ("in", "C"), ("un", "C"), ("ün", "C"),
    # Instrumental.
    ("yla", "V"), ("yle", "V"), ("la", "C"), ("le", "C"),
    # Dative and accusative with buffer consonants.
    ("na", "H"), ("ne", "H"), ("ya", "V"), ("ye", "V"),
    ("nı", "H"), ("ni", "H"), ("nu", "H"), ("nü", "H"),
    ("yı", "V"), ("yi", "V"), ("yu", "V"), ("yü", "V"),
    # Possessive plural and third person.
    ("ımız", "C"), ("imiz", "C"), ("umuz", "C"), ("ümüz", "C"),
    ("ınız", "C"), ("iniz", "C"), ("unuz", "C"), ("ünüz", "C"),
    ("mız", "V"), ("miz", "V"), ("muz", "V"), ("müz", "V"),
    ("nız", "V"), ("niz", "V"), ("nuz", "V"), ("nüz", "V"),
    ("ları", "*"), ("leri", "*"),
    ("sı", "V"), ("si", "V"), ("su", "V"), ("sü", "V"),
    # First person singular possessive (common in support queries: "siparişim").
    ("ım", "C"), ("im", "C"), ("um", "C"), ("üm", "C"), ("m", "V"),
    # Plural.
    ("lar", "*"), ("ler", "*"),
    # A bare final vowel: dative, accusative or third-person possessive.
    ("a", "C"), ("e", "C"), ("ı", "C"), ("i", "C"), ("u", "C"), ("ü", "C"),
)

# Words whose ending only looks like a suffix. Stripping stops when one of these is reached.
_PROTECTED = frozenset(
    """
    altın yakın kadın akın ekin çekin bilim belediye durum kurum konum bölüm ürün üzüm
    uzun kısım ölüm yüzüm tüm hizmet ada ana ata ara oda ama ise ile yine daha gibi
    için hem bütün önem etki kanun hukuk resim isim şehir burun oğul ağız vakit şekil
    fikir akıl kayıp sabır ömür karın boyun alın gönül emir nakit ayrım toplum türkiye
    kanada dede nine yeni temel kira kredi bedel neden sonra önce şimdi hangi kendi
    kişi sene saat vergi iade ücret süre kargo bilgi zaman ekran insan meydan liman vatan
    kalem adam akşam sistem program problem
    """.split()
)

# Two-letter stems that are allowed when a bare vowel or short suffix is removed.
_SHORT_STEMS = frozenset({"ev", "el", "iş", "ad", "at", "ok", "ot", "ip", "ay", "su", "et"})

# Vowel-dropping nouns: the inflected base maps back to the dictionary form.
_VOWEL_DROP = {
    "şehr": "şehir", "burn": "burun", "ağz": "ağız", "oğl": "oğul", "resm": "resim",
    "ism": "isim", "vakt": "vakit", "şekl": "şekil", "fikr": "fikir", "akl": "akıl",
    "kayb": "kayıp", "sabr": "sabır", "ömr": "ömür", "karn": "karın", "boyn": "boyun",
    "aln": "alın", "gönl": "gönül", "emr": "emir", "nakd": "nakit",
}

_SOFTENING = {"b": "p", "c": "ç", "d": "t", "ğ": "k"}
_HIGH_VOWELS = frozenset("ıiuü")


def _has_vowel(word: str) -> bool:
    return any(c in VOWELS for c in word)


def _acceptable_stem(stem: str) -> bool:
    if not _has_vowel(stem):
        return stem in _VOWEL_DROP
    return len(stem) >= 3 or stem in _SHORT_STEMS


def _context_allows(stem: str, before: str) -> bool:
    last = stem[-1]
    if before == "V":
        return last in VOWELS
    if before == "H":
        return last in _HIGH_VOWELS
    if before == "C":
        return last not in VOWELS
    return True


def _strip_once(word: str) -> str | None:
    candidates = [word[: -len(suffix)] for suffix, _ in _SUFFIXES if word.endswith(suffix)]
    # A known word beats any suffix reading: "kadını" is kadın+ı, not kadı+nı.
    for candidate in candidates:
        if candidate in _PROTECTED:
            return candidate
    for suffix, before in _SUFFIXES:
        if not word.endswith(suffix):
            continue
        stem = word[: -len(suffix)]
        if stem and _acceptable_stem(stem) and _context_allows(stem, before):
            return stem
    return None


def _unsoften(stem: str) -> str:
    """Undo final-consonant softening: kitab -> kitap, çocuğ -> çocuk, reng -> renk."""
    if len(stem) < 4:
        return stem
    if stem.endswith("ng"):
        return stem[:-1] + "k"
    replacement = _SOFTENING.get(stem[-1])
    return stem[:-1] + replacement if replacement else stem


def stem(token: str) -> str:
    """Return the retrieval key for one token. Input is folded first."""
    word = fold(token)
    # Everything after an apostrophe is a suffix on a proper name: Ankara'da -> ankara.
    for mark in _APOSTROPHES:
        head, sep, _ = word.partition(mark)
        if sep:
            word = head
    if not word:
        return word

    for _ in range(_MAX_ROUNDS):
        if word in _PROTECTED:
            return word
        stripped = _strip_once(word)
        if stripped is None:
            break
        word = stripped

    if word in _VOWEL_DROP:
        return _VOWEL_DROP[word]
    return _unsoften(word)
