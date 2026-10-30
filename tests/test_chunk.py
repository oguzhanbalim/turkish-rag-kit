import pytest

from turkish_rag_kit import chunk_text, split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dr. Ayşe geldi. Sonra gitti.", ["Dr. Ayşe geldi.", "Sonra gitti."]),
        ("Prof. Dr. Kaya 15. maddeyi okudu.", ["Prof. Dr. Kaya 15. maddeyi okudu."]),
        ("Tutar 3.5 TL oldu! Neden? Bilmiyorum.", ["Tutar 3.5 TL oldu!", "Neden?", "Bilmiyorum."]),
        ("T.C. vatandaşları başvurabilir.", ["T.C. vatandaşları başvurabilir."]),
        ("Kalem, defter vb. şeyler alındı. Bitti.", ["Kalem, defter vb. şeyler alındı.", "Bitti."]),
        ("Başlık\n\nİlk cümle.", ["Başlık", "İlk cümle."]),
        ("A. Yılmaz imzaladı. İşlem tamam.", ["A. Yılmaz imzaladı.", "İşlem tamam."]),
        ("Dedi ki: \"Geldim.\" Sonra çıktı.", ["Dedi ki: \"Geldim.\"", "Sonra çıktı."]),
        ("İşler bitti... Eve gittik.", ["İşler bitti...", "Eve gittik."]),
        ("Son cümlenin noktası yok", ["Son cümlenin noktası yok"]),
        ("", []),
        ("   \n\n  ", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


TEXT = (
    "Birinci cümle kısadır. İkinci cümle biraz daha uzundur ve devam eder. "
    "Üçüncü cümle de burada yer alır. Dördüncü cümle son derece sıradan bir cümledir. "
    "Beşinci cümle metni bitirir."
)


def test_offsets_point_back_into_source():
    for chunk in chunk_text(TEXT, max_chars=80):
        assert TEXT[chunk.start : chunk.end] == chunk.text


def test_chunks_never_cut_inside_a_sentence():
    sentences = set(split_sentences(TEXT))
    for chunk in chunk_text(TEXT, max_chars=80, overlap_sentences=0):
        assert all(s in sentences for s in split_sentences(chunk.text))


def test_every_sentence_is_covered_and_limit_respected():
    chunks = chunk_text(TEXT, max_chars=80)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in split_sentences(TEXT))
    assert all(len(c.text) <= 80 for c in chunks)


def test_overlap_repeats_a_sentence():
    without = chunk_text(TEXT, max_chars=80, overlap_sentences=0)
    with_overlap = chunk_text(TEXT, max_chars=80, overlap_sentences=1)
    assert len(with_overlap) >= len(without)
    first = split_sentences(with_overlap[0].text)[-1]
    assert with_overlap[1].text.startswith(first)


def test_oversized_sentence_is_split_at_whitespace():
    long_sentence = " ".join(["kelime"] * 100) + "."
    chunks = chunk_text(long_sentence, max_chars=50, overlap_sentences=0)
    assert all(len(c.text) <= 50 for c in chunks)
    assert all(not c.text.startswith(" ") for c in chunks)
    assert long_sentence.split() == " ".join(c.text for c in chunks).split()


def test_overlap_larger_than_chunk_still_terminates():
    chunks = chunk_text(TEXT, max_chars=40, overlap_sentences=5)
    assert chunks[-1].end == len(TEXT)
    assert len(chunks) < 20


def test_no_chunk_is_a_subset_of_another():
    text = (
        "Dr. Ayşe Kurgu 15. maddeyi okudu. Tutar 3.5 TL oldu. Prof. Dr. Örnek itiraz etti! "
        "Karar, örn. bugünkü oturumda, yeniden görüşülecek."
    )
    chunks = chunk_text(text, max_chars=70)
    for a in chunks:
        for b in chunks:
            assert a is b or not (b.start <= a.start and a.end <= b.end)
    assert chunks[-1].end == len(text)


def test_bad_arguments_rejected():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", overlap_sentences=-1)
