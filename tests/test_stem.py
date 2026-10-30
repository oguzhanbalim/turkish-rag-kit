import pytest

from turkish_rag_kit import stem

SAME_STEM = [
    "kitap kitaplar kitabı kitabım kitaplarım kitabın kitaba kitapta kitaptan kitapla "
    "kitapların kitaplarımızdan",
    "masa masalar masada masadan masaya masayı masanın masası masasında masalarda masalarımız",
    "ev evler evi evim evde evden eve evin evlerimiz evdeki evimde evinde evini evine",
    "çocuk çocuklar çocuğu çocuğa çocuğun çocukta çocuklarla",
    "şehir şehri şehre şehirde şehrin şehirler şehirlerde şehrimiz",
    "iade iadesi iadem iadeyi iadeler iadenin iadesini",
    "fatura faturam faturası faturayı faturalar faturanın faturada faturalarım",
    "sipariş siparişim siparişi siparişler siparişimi siparişimin siparişleri",
    "kurum kurumu kurumlar kurumda kurumun",
    "renk rengi renkler renge rengin",
    "hesap hesabım hesabı hesaba hesaplar hesapta",
    "Ankara Ankara'da Ankara'nın ANKARA Ankara'ya Ankara’dan",
    "ışık ışığı IŞIK ışıklar ışığın",
    "sınır SINIR sınırı sınırlar sınırda",
    "iş işler işi işe işte işin işim",
    "kadın kadınlar kadını kadının",
    "zaman zamanı zamanda zamanın",
    "ekran ekranı ekranda",
    "şirket şirketi şirketin şirkette şirketten şirketini",
    "gün günü gününü günde",
]


@pytest.mark.parametrize("group", SAME_STEM, ids=lambda g: g.split()[0])
def test_inflections_share_a_key(group):
    assert len({stem(word) for word in group.split()}) == 1, {w: stem(w) for w in group.split()}


@pytest.mark.parametrize(("a", "b"), [("sınır", "sinir"), ("kalem", "kale"), ("zaman", "zam")])
def test_distinct_words_stay_distinct(a, b):
    assert stem(a) != stem(b)


@pytest.mark.parametrize("word", ["ev", "at", "su", "o", "ve", ""])
def test_short_words_survive(word):
    assert stem(word) == word


def test_stem_is_stable_when_reapplied_to_protected_forms():
    assert stem(stem("kadınlar")) == stem("kadınlar")


# Documented limits: these assert today's wrong behaviour so the README stays honest.
# If an improvement breaks one of these, update the README's Limits section with it.
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("kaybettim", "kayıp"),  # verbs are not analysed
        ("hesaplama", "hesaplanır"),  # derivation and voice
        ("gidiyorum", "gitti"),  # verb stem change
        ("bahri", "bahir"),  # vowel-dropping noun outside the short list
    ],
)
def test_known_misses(a, b):
    assert stem(a) != stem(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("kapı", "kap"),  # door / container
        ("kale", "kal"),  # castle / stay
        ("hasta", "has"),  # ill / pure
    ],
)
def test_known_false_conflations(a, b):
    assert stem(a) == stem(b)
