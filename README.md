# turkish-rag-kit

`"SINIR".lower()` is `"sinir"`. One of those words means border and the other means nerve, and your retriever can't tell.

Most RAG stacks treat Turkish as English with odd letters, so case folding corrupts words and inflected forms miss. This is the small piece that fixes those two things at the retrieval layer: correct folding, a light suffix stripper, sentence-aware chunking and BM25, with no dependencies.

## The problem, in one Python session

```
>>> "İSTANBUL".lower(), "İSTANBUL".lower() == "istanbul"
('i̇stanbul', False)
>>> "IŞIK".lower(), "IŞIK".lower() == "ışık"
('işik', False)
>>> "SINIR".lower(), "SINIR".lower() == "sınır"
('sinir', False)
>>> re.findall(r"\w+", "İSTANBUL".lower())
['i', 'stanbul']
```

`İ` lowers to `i` plus a combining dot (U+0307), which isn't a word character, so a standard tokeniser cuts the city in half.

## Before and after

```
pip install -e .
turkish-rag examples/corpus "SINIR KAPISI NE ZAMAN AÇIK" --analyzer naive --top 2
```

Naive `str.lower()` and `\w+` looks for "nerve" and returns the anatomy note:

```
  2.68  sinir-sistemi.txt#0  [sinir]
        Sinir sistemi beyin, omurilik ve çevresel sinirlerden oluşur. Sinirler vücuda sinyal taşır. Bu metin sağlık ok
```

The default analyzer returns the border crossing:

```
$ turkish-rag examples/corpus "SINIR KAPISI NE ZAMAN AÇIK" --top 2
  5.18  sinir-kapisi.txt#0  [sınır, kap]
        Kurgusal Doğu Sınır Kapısı yaz aylarında 24 saat açıktır. Sınırdan geçişlerde pasaport kontrolü yapılır. Araçl
  2.08  su-kesintisi.txt#0  [açık]
        Planlı su kesintileri bir gün önceden duyurulur. Kesintinin süresi bakım işinin kapsamına bağlıdır. Suyun geri
```

`python examples/compare.py` runs 20 queries through three analyzers over 10 documents. `naive` is `str.lower()`, `folded` adds correct case folding, and `turkish` adds the stemmer and stopwords. Selected output:

```
query: 'ISI YALITIMI DESTEĞİ'  (expected isi-yalitimi.txt)
  naive   BAD (no result)
  folded  ok  isi-yalitimi.txt       4.64  [ısı, yalıtımı]
  turkish ok  isi-yalitimi.txt       7.34  [ısı, yalıt, destek]
query: 'SINIR KAPISI NE ZAMAN AÇIK'  (expected sinir-kapisi.txt)
  naive   BAD sinir-sistemi.txt      2.68  [sinir]
  folded  ok  sinir-kapisi.txt       4.57  [sınır, kapısı]
  turkish ok  sinir-kapisi.txt       5.18  [sınır, kap]
query: 'faturaya itiraz'  (expected fatura.txt)
  naive   BAD (no result)
  folded  BAD (no result)
  turkish ok  fatura.txt             5.66  [fatur, itiraz]
query: 'saklama kapları'  (expected -)
  naive   ok  (no result)
  folded  ok  (no result)
  turkish BAD sinir-kapisi.txt       2.00  [kap]

correct top result, 20 queries, 10 documents:
  naive     7/20
  folded   10/20
  turkish  16/20
```

Don't read 16/20 as a benchmark. I wrote the corpus and the queries, and I shaped both knowing what the stemmer handles. The set exists to show the failure modes with real output, including the last query above, where the stemmer makes things worse: `kapları` (containers) matches `kapısı` (door).

## Use

```python
from turkish_rag_kit import BM25Index, chunk_text, fold, stem

fold("IŞIK")                  # 'ışık'
stem("kitabımızdan")          # 'kitap'
stem("şehrinde")              # 'şehir'
stem("masalarda")             # 'mas'   a retrieval key, not a word

index = BM25Index()           # Turkish analyzer by default
for chunk in chunk_text(document, max_chars=800, overlap_sentences=1):
    index.add(chunk.index, chunk.text)
index.search("faturalarımı nereden indiririm", top_k=5)
```

The sentence splitter knows Turkish abbreviations, ordinals and decimals, and chunks keep offsets back into the source for citations:

```
>>> for c in chunk_text(text, max_chars=70): print(c.start, c.end, repr(c.text))
0 52 'Dr. Ayşe Kurgu 15. maddeyi okudu. Tutar 3.5 TL oldu.'
53 81 'Prof. Dr. Örnek itiraz etti!'
82 132 'Karar, örn. bugünkü oturumda, yeniden görüşülecek.'
```

## Limits

**The stemmer is lightweight and not morphologically complete.** It is about 150 lines of suffix rules and a short word list.

- **Verbs are not analysed.** `kaybettim` doesn't reach `kayıp`, and `gidiyorum` doesn't reach `gitti`. `elim uyuştu` doesn't find `uyuşma` in the comparison above.
- **Derivation is untouched.** `mantolamak` and `mantolama` are different keys, as are `hesaplama` and `hesaplanır`.
- **It conflates unrelated words:** `kapı` / `kap`, `kale` / `kal`, `hasta` / `has`. Stripping a final vowel gets recall and pays for it here. Tests pin these so they stay documented.
- **Words outside the short lists go wrong.** Vowel-dropping nouns work only for a list of 19 (`şehir`/`şehri`), and suffix look-alikes are protected only for a list of about 80 (`altın`, `kurum`). `n`-final nouns outside the list can split between `-ı` and `-nı` readings.
- **The result is a key, not a lemma.** Don't show `mas` to a person.

Other limits:

- Folding is Turkish-specific. It lowers an English `I` to `ı`, so `INDEX` and `index` get different keys. Mixed-language text needs a decision this library doesn't make.
- Circumflexes are removed, which merges `kâr` (profit) and `kar` (snow).
- `ascii_fold` for queries typed without Turkish letters exists but is off by default, because it merges `sınır` and `sinir` again.
- The sentence splitter is rule-based. A sentence ending in a number or an abbreviation won't split.
- BM25 is in-memory, keyword-only, with no persistence. It doesn't help with synonyms; pair it with embeddings if you need them.
- Tested locally on Python 3.12 only. CI is configured for 3.11 and 3.12 but this repository has not been pushed.

## License

MIT
