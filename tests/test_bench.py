"""
CodSpeed benchmarks for `lev.distance` and `lev.ratio`.

Each benchmark times only a batch of calls through the `benchmark` fixture.
Timing a whole test instead measured about 185 us of pytest and
instrumentation overhead per test against 0.03-1 us of lev work, which hid
every change to lev below CodSpeed's noise floor. The batch is driven from
C (`map` drained into a zero-length `deque`), so the interpreter adds little
per call, and its size is fixed per case so that every batch does a comparable amount
of work. Correctness lives in `test_lev.py`.

There are two kinds of cases:

- Single-pair cases repeat one pair. They track peak kernel speed and keep
  CodSpeed's history, but the branch predictor learns the pair's branch
  sequence, which a real workload of distinct pairs never allows.
- Varied cases (`varied-*`) cycle through a seeded pool of distinct,
  realistic pairs of one class (e.g. CJK strings of 17-40 chars, names with
  3 typos, 65-704-char sentences just past the band's certification limit).

Judge every gate or threshold change (which path a pair takes) on the varied
cases. On Apple M2, three gates flipped sign between the two kinds: the
64-slot single-word hash, TINY_M = 16 for wide strings and mbleven on short
Latin-1 names. A win only on single-pair cases is a branch predictor artefact.
"""

from __future__ import annotations

import os
import random
import sys
import unicodedata
from array import array
from collections import deque
from itertools import accumulate, repeat
from typing import TYPE_CHECKING

import pytest

import lev

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_codspeed import BenchmarkFixture

# Cells of the DP matrix (len(s1) * len(s2)) one batch should cover, roughly.
_BATCH_CELLS = 4_000_000
# Calls per batch for the smallest pairs, where per-call overhead dominates.
_MAX_BATCH = 20_000


def _batch_size(s1: str | list[str], s2: str | list[str]) -> int:
    """
    Return the calls per batch, inversely proportional to the DP size.

    For a pool (two lists of strings) the mean DP size of its pairs counts,
    and the batch makes whole passes over the pool, so that every pair
    weighs the same.

    Returns:
        int: number of calls, at least 1 (at least the pool size for a pool).

    """
    if isinstance(s1, str):
        return max(1, min(_MAX_BATCH, _BATCH_CELLS // max(1, len(s1) * len(s2))))
    cells = sum(len(a) * len(b) for a, b in zip(s1, s2, strict=True))
    passes = max(1, min(_MAX_BATCH // len(s1), round(_BATCH_CELLS / max(1, cells))))
    return passes * len(s1)


def _run_batch(fn: Callable[[str, str], object], s1: str | list[str], s2: str | list[str], n: int) -> None:
    """
    Call `fn` n times with as little interpreter work as possible.

    Two strings form one pair that is repeated n times; two lists of n
    strings (a pool cycled to n entries) give one call per entry.
    """
    if isinstance(s1, str):
        deque(map(fn, repeat(s1, n), repeat(s2, n)), maxlen=0)
    else:
        deque(map(fn, s1, s2), maxlen=0)


def _rand_str(alphabet: str, n: int, seed: int) -> str:
    """
    Build a deterministic random string over the given alphabet.

    Returns:
        str: random string of length n.

    """
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(n))


def _mutate(s: str, alphabet: str, edits: int, seed: int) -> str:
    """
    Apply `edits` random substitutions, insertions, and deletions to s.

    Returns:
        str: mutated copy of s.

    """
    rng = random.Random(seed)
    b = list(s)
    for _ in range(edits):
        i = rng.randrange(len(b))
        op = rng.randrange(3)
        if op == 0:
            b[i] = rng.choice(alphabet)
        elif op == 1:
            b.insert(i, rng.choice(alphabet))
        else:
            del b[i]
    return "".join(b)


def _pair(alphabet: str, n: int, edits: int | None, seed: int, id: str) -> object:  # ruff: ignore[builtin-argument-shadowing]
    """
    Build a mutated pair (`edits` edits) or an unrelated pair (`edits=None`).

    Returns:
        object: pytest param of (s1, s2).

    """
    s = _rand_str(alphabet, n, seed=seed)
    t = _rand_str(alphabet, n, seed=seed + 1) if edits is None else _mutate(s, alphabet, edits, seed + 1)
    return pytest.param(s, t, id=id)


def _quote(s: str) -> str:
    """
    Replace the middle char by a curly apostrophe, which makes s UCS-2.

    Returns:
        str: s with one U+2019.

    """
    return s[: len(s) // 2] + chr(0x2019) + s[len(s) // 2 + 1 :]


_ASCII = "abcdefghijklmnopqrstuvwxyz "
_LATIN1 = "àáâãäåæçèé abcdef"
_UCS2 = "ぁあぃいぅうぇえぉお"
_UCS4 = "".join(chr(0x1F600 + i) for i in range(10))

_SHORT = [
    pytest.param("", "a" * 100, id="zero-length-empty-vs-100"),
    pytest.param(
        "The quick brown fox",
        "The quick brown fox jumps over the lazy dog",
        id="prefix-contained-fox",
    ),
    pytest.param("a", "b", id="ascii-1"),
    pytest.param("kitten", "sitting", id="ascii-6"),
    pytest.param("hello", "hello", id="equal-5"),
    pytest.param("résumé", "resume", id="latin1-6"),
    pytest.param("日本語", "日本", id="ucs2-3"),
    pytest.param("🦀🐍", "🐍🦀", id="ucs4-2"),
    pytest.param("abc", "abcЀ", id="mixed-3"),
    _pair(_ASCII, 12, None, seed=1, id="ascii-12-dis"),
    _pair(_ASCII, 32, 3, seed=3, id="ascii-32-d3"),
    _pair(_ASCII, 32, None, seed=5, id="ascii-32-dis"),
    _pair(_UCS2, 32, 8, seed=7, id="ucs2-32"),
    _pair(_UCS4, 64, 16, seed=9, id="ucs4-64"),
    # Byte patterns of 17-64 chars after affix stripping (single-word path).
    _pair(_LATIN1, 24, 3, seed=103, id="latin1-24-d3"),
    _pair(_LATIN1, 32, None, seed=13, id="latin1-32-dis"),
    _pair(_ASCII, 48, 6, seed=31, id="ascii-48-d6"),
    _pair(_LATIN1, 64, None, seed=17, id="latin1-64-dis"),
    # A 24-char pattern against a 4x and a 16x longer text.
    pytest.param(_rand_str(_ASCII, 24, seed=19), _rand_str(_ASCII, 96, seed=21), id="ascii-24-vs-96"),
    pytest.param(_rand_str(_ASCII, 24, seed=23), _rand_str(_ASCII, 384, seed=25), id="ascii-24-vs-384"),
]

_MEDIUM = [
    _pair(_ASCII, 100, 25, seed=20, id="ascii-100"),
    pytest.param(
        "fnwnujp gzpzqfqqhzaalsncezyhhwbnxyntobkrpwdulaex cdzaofrxkmpefvbp hcyggvzvyysxcdrpr mgkernjloqmkpmia cfwwpapcrtxqhstkrtd",
        "fnwnujp gzpzqfqXhzaalsncezyhhwbnxyntobkrpwdulaex cdzaofrxkmpefvbp hcyggvzvyysxcdrpr mZkernjloqmkpmia cfwwpapcrtxqhstkrtd",
        id="ascii-120-sim2",
    ),
    _pair(_ASCII, 300, 75, seed=22, id="ascii-300"),
    pytest.param(
        "q" * 20
        + "fnwnujp gzpzqfqXhzaalsncezyhhwbnxyntobkrpwdulaex cdzaofrxkmpefvbp hcyggvzvyysxcdrpr mZkernjloqmkpmia cfwwpapcrtxqhstkrtd"
        + "q" * 20,
        "q" * 20
        + "fnwnujp gzpzqfqqhzaalsncezyhhwbnxyntobkrpwdulaex cdzaofrxkmpefvbp hcyggvzvyysxcdrpr mgkernjloqmkpmia cfwwpapcrtxqhstkrtd"
        + "q" * 20,
        id="ascii-160-sim2",
    ),
    _pair(_LATIN1, 100, 25, seed=24, id="latin1-100"),
    _pair(_UCS2, 100, 25, seed=26, id="ucs2-100"),
    _pair(_UCS2, 300, None, seed=28, id="ucs2-300-dis"),
    _pair(_UCS4, 200, 50, seed=30, id="ucs4-200"),
    _pair(_LATIN1, 360, 90, seed=32, id="latin1-360"),
    _pair(_ASCII, 420, 105, seed=34, id="ascii-420"),
    _pair(_UCS2, 500, None, seed=36, id="ucs2-500-dis"),
    _pair(_ASCII, 640, 160, seed=38, id="ascii-640"),
]

# One string UCS-1, the other UCS-2 or UCS-4 (a curly quote is enough).
_MIXED = [
    pytest.param(_rand_str(_ASCII, 100, seed=40), _quote(_rand_str(_ASCII, 100, seed=41)), id="quote-100-dis"),
    pytest.param(
        _rand_str(_ASCII, 300, seed=42),
        _quote(_mutate(_rand_str(_ASCII, 300, seed=42), _ASCII, 30, seed=43)),
        id="quote-300-sim",
    ),
    pytest.param(
        _rand_str(_ASCII, 100, seed=44),
        _mutate(_rand_str(_ASCII, 100, seed=44), _UCS2, 25, seed=45),
        id="ascii-ucs2-100",
    ),
    pytest.param(_rand_str(_ASCII, 200, seed=46), _rand_str(_UCS4 + "ab", 200, seed=47), id="ascii-ucs4-200"),
]

# Long shared prefix and suffix around a short differing middle.
_AFFIX = [
    pytest.param(f"{'x' * 400}kitten{'x' * 400}", f"{'x' * 400}sitting{'x' * 400}", id="ascii-800"),
    pytest.param(f"{'ж' * 400}kitten{'ж' * 400}", f"{'ж' * 400}sitting{'ж' * 400}", id="ucs2-800"),
    pytest.param(f"{'🦀' * 400}kitten{'🦀' * 400}", f"{'🦀' * 400}sitting{'🦀' * 400}", id="ucs4-800"),
    pytest.param(f"{'x' * 400}kitten{'x' * 400}", f"{'x' * 400}sittiłg{'x' * 400}", id="mixed-800"),
]

_LONG = [
    _pair(_ASCII, 2048, 4, seed=60, id="ascii-2048-similar"),
    _pair(_ASCII, 2048, None, seed=62, id="ascii-2048-dissimilar"),
    _pair(_ASCII, 8192, 8, seed=64, id="ascii-8192-similar"),
    _pair(_UCS2, 2048, 4, seed=66, id="ucs2-2048-similar"),
    _pair(_UCS2, 1024, None, seed=68, id="ucs2-1024-dissimilar"),
]

# ---------------------------------------------------------------------------
# Varied-pair pools
# ---------------------------------------------------------------------------
# Distinct pairs per pool, measured on M2 (in-process A/B of builds that
# differ in one gate, pools of 1 to 16384 pairs): the ratios of the hash,
# TINY_M and mbleven gates settle once a pool holds about 100k chars of short
# strings. Wide (UCS-2/4) strings of up to 40 chars need 4096 pairs (the
# TINY_M ratio went 0.83 at 4 pairs, 1.51 at 1024, 1.95 at 4096, 2.02 at
# 8192), byte strings 1024, 65-704-char pairs 256. Long pairs run thousands
# of branches per call, and their pools stay small to bound the batch time.
_POOL_WIDE = 4096  # UCS-2/4 strings of up to 64 chars
_POOL_BYTES = 1024  # byte strings of up to 64 chars, and short-vs-long pairs
_POOL_MEDIUM = 256  # 65-704 chars
_POOL_LONG = 32  # > 512 chars

_LOWER = "abcdefghijklmnopqrstuvwxyz"

# Word lists, most frequent first (weights follow Zipf's law by rank).
_EN = """
the of and to in a is that for it as was with be by on not he this are or his from at which but have an they
you were her she there one been we their has would all when more will if no out so said what up its about into
than them can only other new some could time these two may then do first any my now such like our over man me
even most made after also did many before must through back years where much your way well down should because
each just those people how too little state good very make world still own see men work long get here between
both life being under never day same another know while last might us great old year off come since against go
came right used take three city school music season family river house water history name game album film team
party league power court church army center book story market island station road group child woman company
government system program number area line point report public field order level war body case part form place
end word fact night price street table window garden letter answer question problem service product customer
account address phone email delivery invoice payment record update review result search support
""".split()  # ruff: ignore[split-static-string]
_RU = """
и в не на я быть он с что а по это она этот к но они мы как из у который то за свой весь год от так о для ты же
все тот мочь вы человек такой его сказать только или ещё бы себя один уже до время если сам когда другой вот
говорить наш мой знать стать при чтобы дело жизнь кто первый очень два день её новый рука даже во со раз где там
под можно ну какой после их работа без самый потом надо хотеть ли слово идти большой должен место иметь ничто
город улица дом страна вопрос сторона глаз голова друг дорога система власть деньги история книга
""".split()  # ruff: ignore[split-static-string, ambiguous-unicode-character-string]
_FIRST = """
James Mary John Patricia Robert Jennifer Michael Linda William Elizabeth David Barbara Richard Susan Joseph
Jessica Thomas Sarah Charles Karen Christopher Lisa Daniel Nancy Matthew Betty Anthony Margaret Mark Sandra
Donald Ashley Steven Kimberly Paul Emily Andrew Donna Joshua Michelle Kenneth Carol Kevin Amanda Brian Melissa
George Deborah Timothy Stephanie Ronald Rebecca Edward Sharon Jason Laura Jeffrey Cynthia Ryan Kathleen Jacob
Amy Gary Angela Nicholas Shirley Eric Anna Jonathan Brenda Stephen Pamela Larry Emma Justin Nicole Scott Helen
""".split()  # ruff: ignore[split-static-string]
_LAST = """
Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson
Anderson Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson
Walker Young Allen King Wright Scott Torres Nguyen Hill Flores Green Adams Nelson Baker Hall Rivera Campbell
Mitchell Carter Roberts Gomez Phillips Evans Turner Diaz Parker Cruz Edwards Collins Reyes Stewart Morris
Morales Murphy Cook Rogers Gutierrez Ortiz Morgan Cooper Peterson Bailey Reed Kelly Howard Ramos Kim Cox Ward
""".split()  # ruff: ignore[split-static-string]
_FIRST_LATIN1 = """
José André François Hélène Renée Zoë Chloé Inês Mónica Ángel Jürgen Jörg Björn Søren Åsa Françoise Gaël Noël
Raphaël Léa Clément Jérôme Dörte Günter Íñigo Begoña Joël Agnès Benoît Óscar
""".split()  # ruff: ignore[split-static-string]
_LAST_LATIN1 = """
Martínez García Muñoz Peña Núñez Gómez Lefèvre Dupré Gauthier Bérénger Müller Schäfer Köhler Groß Jäger Åström
Ångström Sørensen Bjørnstad Brontë Conceição Gonçalves Fernández Rodríguez Hernández Löwe Weiß Böhm Méndez Côté
""".split()  # ruff: ignore[split-static-string]
_BRANDS = "Apple Samsung Sony LG Dell HP Lenovo Asus Acer Microsoft Logitech Bose Canon Nikon Philips Xiaomi".split()  # ruff: ignore[split-static-string]
_TYPES = "Laptop Smartphone Headphones Monitor Keyboard Mouse Camera Tablet Smartwatch Speaker Router Printer".split()  # ruff: ignore[split-static-string]
_ATTRS = """
Wireless|Bluetooth|Noise Cancelling|4K|UHD|OLED|Gaming|Ultra|Pro|Max|Mini|Plus|Lite|Black|White|Silver|Blue|Red
|Titanium|128GB|256GB|512GB|1TB|8GB RAM|16GB RAM|15.6 inch|27 inch|USB-C|Fast Charging|2024 Model|Refurbished
|with Case|Dual SIM
""".replace("\n", "").split("|")
# Accents dropped the way a plain-ASCII copy of a name drops them.
_UNACCENT = {c: unicodedata.normalize("NFKD", chr(c))[0] for c in range(0xC0, 0x100)} | {
    0xDF: "ss",
    0xD8: "O",
    0xF8: "o",
    0xC6: "AE",
    0xE6: "ae",
}


def _zipf(symbols: list[str]) -> list[str]:
    """
    Build a sampling table in which symbol r (most frequent first) fills a share of 1 / r.

    Drawing uniform entries of the 65536-entry table follows Zipf's law, and
    can be done at C speed (see `_draw`), which keeps the module import fast.

    Returns:
        list[str]: 65536 entries.

    """
    cum = list(accumulate(1 / rank for rank in range(1, len(symbols) + 1)))
    ends = [round(c * 65536 / cum[-1]) for c in cum]
    return [s for s, lo, hi in zip(symbols, [0, *ends], ends, strict=False) for _ in range(hi - lo)]


def _sample(seed: str, lo: int, hi: int, k: int) -> list[str]:
    """
    Draw k distinct chars from the code points [lo, hi) in a seeded random order.

    Returns:
        list[str]: the chars.

    """
    return [chr(c) for c in random.Random(seed).sample(range(lo, hi), k)]


# 3000 ideographs spread over the CJK block, as real text is (common
# characters are not contiguous code points), plus two punctuation marks.
_HANZI = _sample("hanzi", 0x4E00, 0xA000, 3000)
_CJK_SYMBOLS = [*_HANZI[:2], "，", *_HANZI[2:4], "。", *_HANZI[4:]]  # ruff: ignore[ambiguous-unicode-character-string]
_CJK = _zipf(_CJK_SYMBOLS)
_HIRAGANA = _sample("hiragana", 0x3041, 0x3097, 0x3097 - 0x3041)
_KATAKANA = _sample("katakana", 0x30A1, 0x30FB, 0x30FB - 0x30A1)
# Japanese: hiragana (particles, endings) most frequent, kanji, katakana loan words.
_JA = _zipf([*_HIRAGANA[:20], *_HANZI[:50], *_HIRAGANA[20:], *_HANZI[50:600], *_KATAKANA, *_HANZI[600:2000]])
# 400 emoji from the emoticon, symbol, transport and supplemental blocks.
_EMOJI = _zipf([
    chr(c)
    for c in random.Random("emoji").sample(
        [*range(0x1F300, 0x1F650), *range(0x1F680, 0x1F700), *range(0x1F900, 0x1FA00)], 400
    )
])
_EN_Z = _zipf(_EN)
_RU_Z = _zipf(_RU)


def _draw(rng: random.Random, table: list[str], k: int) -> map[str]:
    """
    Draw k entries of a `_zipf` table.

    Returns:
        map[str]: drawn symbols or words.

    """
    return map(table.__getitem__, array("H", rng.randbytes(2 * k)))


# Corpora of Zipf-distributed text; every varied string is a window of one,
# which keeps the module import fast. Dissimilar pairs take their two strings
# from different halves (suffix A and B), so that windows never overlap.
_EN_A, _EN_B = (" ".join(_draw(random.Random(f"en-{h}"), _EN_Z, 60_000)) for h in "ab")
_RU_A, _RU_B = (" ".join(_draw(random.Random(f"ru-{h}"), _RU_Z, 8_000)) for h in "ab")
_CJK_A, _CJK_B = ("".join(_draw(random.Random(f"cjk-{h}"), _CJK, 100_000)) for h in "ab")
_JA_A, _JA_B = ("".join(_draw(random.Random(f"ja-{h}"), _JA, 30_000)) for h in "ab")
_EMOJI_A, _EMOJI_B = ("".join(_draw(random.Random(f"emoji-{h}"), _EMOJI, 30_000)) for h in "ab")


def _text(rng: random.Random, corpus: str, lo: int, hi: int) -> str:
    """
    Cut a window with a length drawn from [lo, hi] out of a corpus.

    Returns:
        str: the window.

    """
    n = lo + int(rng.random() * (hi - lo + 1))
    i = int(rng.random() * (len(corpus) - n))
    return corpus[i : i + n]


def _loguniform(rng: random.Random, lo: int, hi: int) -> int:
    """
    Draw a length from [lo, hi] with uniform log, so short and long lengths both get weight.

    Returns:
        int: the length.

    """
    return min(hi, int(lo * (hi / lo) ** rng.random()))


def _edits(s: str, k: int, rng: random.Random, alphabet: str | list[str] = _LOWER, swaps: bool = True) -> str:
    """
    Apply k typos at random positions: substitution, insertion, deletion or (with `swaps`) transposition.

    Returns:
        str: edited copy of s.

    """
    if len(s) < 3 or k <= 0:
        return s
    b, last, ops, rand = list(s), len(s) - 1, 4 if swaps else 3, rng.random
    # Right to left, so that each position still refers to s.
    for pos, op, c in sorted(((int(rand() * last), int(rand() * ops), rand()) for _ in range(k)), reverse=True):
        if len(b) < 3:
            break
        i = min(pos, len(b) - 2)
        if op == 0:
            b[i] = alphabet[int(c * len(alphabet))]
        elif op == 1:
            b.insert(i, alphabet[int(c * len(alphabet))])
        elif op == 2:
            del b[i]
        else:
            b[i], b[i + 1] = b[i + 1], b[i]
    return "".join(b)


def _other(c: str, rng: random.Random, alphabet: str = _LOWER) -> str:
    """
    Draw a char of the alphabet other than c.

    Returns:
        str: the char.

    """
    while (x := rng.choice(alphabet)) == c:
        pass
    return x


def _ub3(s: str, rng: random.Random) -> str:
    """
    Edit s so that its first and last chars differ (affix stripping keeps all of s) at an upper bound of 3.

    Length difference 0 (3 spread substitutions), 1 (2 plus an appended char)
    or 2 (1 plus two appended chars), a third each.

    Returns:
        str: edited copy of s.

    """
    b, m, diff = list(s), len(s), rng.randrange(3)
    for p in ((0, m // 2, m - 1), (0, m - 1), (0,))[diff]:
        b[p] = _other(b[p], rng)
    return "".join(b) + "".join(_other(s[-1], rng) for _ in range(diff))


def _title(rng: random.Random) -> str:
    """
    Build a product title.

    Returns:
        str: the title.

    """
    attrs = " ".join(rng.sample(_ATTRS, rng.randint(1, 4)))
    return f"{rng.choice(_BRANDS)} {rng.choice(_LOWER).upper()}{rng.randint(1, 99)} {rng.choice(_TYPES)} {attrs}"


def _sized(rng: random.Random, make: Callable[[random.Random], str], lo: int, hi: int) -> str:
    """
    Draw texts from `make` until one has a length in [lo, hi].

    Returns:
        str: the text.

    """
    while not lo <= len(s := make(rng)) <= hi:
        pass
    return s


def _accents(s: str, rng: random.Random) -> str:
    """
    Accent a random subset of the letters a, e, i, o, u, n and c in s, which makes it Latin-1.

    Returns:
        str: accented copy of s.

    """
    return s.translate({ord(k): v for k, v in zip("aeiounc", "áéíöüñç", strict=True) if rng.random() < 0.4})


def _with_emoji(s: str, rng: random.Random) -> str:
    """
    Insert 1-3 emoji into s, which makes it UCS-4.

    Returns:
        str: s with emoji.

    """
    b = list(s)
    for e in _draw(rng, _EMOJI, rng.randint(1, 3)):
        b.insert(rng.randint(0, len(b)), e)
    return "".join(b)


def _query_in_text(
    rng: random.Random,
    q: str,
    n: int,
    text: Callable[[random.Random, int], str],
    alphabet: str | list[str] = _LOWER,
) -> str:
    """
    Build a text of about n chars that contains q (with up to one typo) in half the cases.

    Returns:
        str: the text.

    """
    if rng.random() < 0.5:
        return text(rng, n)
    t = text(rng, max(0, n - len(q)))
    i = rng.randint(0, len(t))
    return t[:i] + _edits(q, rng.randint(0, 1), rng, alphabet) + t[i:]


def _band_limit(m: int, n: int) -> int:
    """
    Largest distance the small bands (#260) certify on an m-by-n pair, n >= m.

    The one-word band certifies up to 63, the two-word band (from 257 chars)
    up to 127, and both give up beyond about one edit per 4 text chars.

    Returns:
        int: the distance.

    """
    return min(63 if m <= 256 else 127, n - m + 2 + n // 4)


def _pairs(name: str, size: int, make: Callable[[random.Random], tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Build a seeded pool of distinct pairs.

    Returns:
        list[tuple[str, str]]: `size` distinct pairs.

    """
    rng = random.Random(name)
    pool: dict[tuple[str, str], None] = {}
    while len(pool) < size:
        pool[make(rng)] = None
    return list(pool)


def _en(rng: random.Random, n: int, corpus: str = _EN_A) -> str:
    """
    Cut English-like text of n chars.

    Returns:
        str: the text.

    """
    return _text(rng, corpus, n, n)


def _en_b(rng: random.Random, n: int) -> str:
    """
    Cut English-like text of n chars from the other half of the corpus than `_en`.

    Returns:
        str: the text.

    """
    return _en(rng, n, _EN_B)


def _sim(rng: random.Random, s: str, lo: float, hi: float, alphabet: str | list[str] = _LOWER) -> str:
    """
    Apply typos (no transpositions) to a fraction of s's chars drawn from [lo, hi].

    Returns:
        str: edited copy of s.

    """
    return _edits(s, max(1, round(len(s) * rng.uniform(lo, hi))), rng, alphabet, swaps=False)


def _near_miss(rng: random.Random, s: str) -> str:
    """
    Apply 1.4-1.75 times as many typos as the small bands certify, so that both bands give up late.

    Returns:
        str: edited copy of s.

    """
    k = _band_limit(len(s), len(s))
    return _edits(s, round(k * rng.uniform(1.4, 1.75)), rng, swaps=False)


def _insert_words(rng: random.Random, s: str, lo: int, hi: int) -> str:
    """
    Insert words totalling lo-hi chars near the middle of s (an insertion-heavy similar pair).

    Returns:
        str: s with the inserted words.

    """
    ins = _text(rng, _EN_B, lo, hi)
    i = rng.randint(len(s) // 4, 3 * len(s) // 4)
    return s[:i] + ins + s[i:]


_CJK_TOP = _CJK_SYMBOLS[:300]
_NAME = (
    lambda r: f"{r.choice(_FIRST)} {r.choice(_LAST)}",
    lambda r: f"{r.choice(_FIRST)} {r.choice(_FIRST)} {r.choice(_LAST)}",
)

# name: (functions, pool size, pair generator). "dr" runs distance and ratio,
# "d" distance only (mbleven and the small bands are Levenshtein-only).
# fmt: off
_VARIED_SPECS: dict[str, tuple[str, int, Callable[[random.Random], tuple[str, str]]]] = {
    # Single-word hash for UCS-2/4 patterns of 17-64 chars (64 or 128 slots).
    "cjk-17-40-dis": ("dr", _POOL_WIDE, lambda r: (_text(r, _CJK_A, 17, 40), _text(r, _CJK_B, 17, 40))),
    "cjk-17-40-rel": ("dr", _POOL_WIDE, lambda r: (s := _text(r, _CJK_A, 17, 40), _edits(s, len(s) // 4, r, _CJK_TOP, swaps=False))),
    "ja-17-40-dis": ("dr", _POOL_WIDE, lambda r: (_text(r, _JA_A, 17, 40), _text(r, _JA_B, 17, 40))),
    "ja-17-40-rel": ("dr", _POOL_WIDE, lambda r: (s := _text(r, _JA_A, 17, 40), _edits(s, len(s) // 4, r, _HIRAGANA, swaps=False))),
    "cyrillic-17-40-dis": ("dr", _POOL_WIDE, lambda r: (_text(r, _RU_A, 17, 40), _text(r, _RU_B, 17, 40))),
    "emoji-17-40-dis": ("dr", _POOL_WIDE, lambda r: (_text(r, _EMOJI_A, 17, 40), _text(r, _EMOJI_B, 17, 40))),
    "ucs4-text-17-40-dis": ("dr", _POOL_WIDE, lambda r: (_with_emoji(_text(r, _EN_A, 16, 37), r), _with_emoji(_text(r, _EN_B, 16, 37), r))),
    # Tiny path for wide strings (TINY_M).
    "cjk-9-16-dis": ("dr", _POOL_WIDE, lambda r: (_text(r, _CJK_A, 9, 16), _text(r, _CJK_B, 9, 16))),
    "cjk-9-16-rel": ("dr", _POOL_WIDE, lambda r: (s := _text(r, _CJK_A, 9, 16), _edits(s, r.randint(1, 3), r, _CJK_TOP, swaps=False))),
    "emoji-9-16-dis": ("dr", _POOL_WIDE, lambda r: (_text(r, _EMOJI_A, 9, 16), _text(r, _EMOJI_B, 9, 16))),
    # Short names and words with typos: tiny path, small_ub gate, mbleven.
    "words-typo": ("dr", _POOL_BYTES, lambda r: (s := _text(r, _EN_A, 4, 24).strip(), _edits(s, r.randint(1, 2), r))),
    "names-typo": ("dr", _POOL_BYTES, lambda r: (s := r.choice(_NAME)(r), _edits(s, r.randint(1, 3), r))),
    "latin1-names": ("dr", _POOL_BYTES, lambda r: (
        s := f"{r.choice(_FIRST_LATIN1)} {r.choice(['', *_FIRST])} {r.choice(_LAST_LATIN1)}",
        _edits(s.translate(_UNACCENT), r.randint(0, 1), r),
    )),
    "names-ub3": ("d", _POOL_BYTES, lambda r: (s := _sized(r, _NAME[1], 17, 32), _ub3(s, r))),
    "titles-ub3": ("d", _POOL_BYTES, lambda r: (s := _sized(r, _title, 33, 64), _ub3(s, r))),
    # Byte patterns of 17-64 chars: table-free masks up to 8x longer texts.
    "titles-dis": ("dr", _POOL_BYTES, lambda r: (_sized(r, _title, 17, 64), _sized(r, _title, 17, 64))),
    "latin1-17-64-dis": ("dr", _POOL_BYTES, lambda r: (_accents(_text(r, _EN_A, 17, 64), r), _accents(_text(r, _EN_B, 17, 64), r))),
    # Short query against a longer text (n >= 2m).
    "query-4-16-vs-2-16x": ("dr", _POOL_BYTES, lambda r: (q := _text(r, _EN_A, 4, 16), _query_in_text(r, q, len(q) * r.randint(2, 16), _en_b))),
    "query-17-64-vs-2-8x": ("dr", _POOL_BYTES, lambda r: (q := _text(r, _EN_A, 17, 64), _query_in_text(r, q, len(q) * r.randint(2, 8), _en_b))),
    "query-17-32-vs-9-12x": ("dr", _POOL_BYTES, lambda r: (q := _text(r, _EN_A, 17, 32), _query_in_text(r, q, len(q) * r.randint(9, 12), _en_b))),
    "cjk-query-9-40-vs-2-8x": ("dr", _POOL_BYTES, lambda r: (
        q := _text(r, _CJK_A, 9, 40),
        _query_in_text(r, q, len(q) * r.randint(2, 8), lambda r, n: _text(r, _CJK_B, n, n), _CJK_TOP),
    )),
    # Sentences of 65-704 chars: speculative small bands (#260).
    "band-sim": ("dr", _POOL_MEDIUM, lambda r: (s := _en(r, _loguniform(r, 65, 704)), _sim(r, s, 0.01, 0.08))),
    "band-near-miss": ("d", _POOL_MEDIUM, lambda r: (s := _en(r, _loguniform(r, 80, 704)), _near_miss(r, s))),
    "band-dis": ("dr", _POOL_MEDIUM, lambda r: (s := _en(r, _loguniform(r, 65, 704)), _en_b(r, max(65, len(s) + r.randint(-29, 29))))),
    "band-difflen-dis": ("d", _POOL_MEDIUM, lambda r: (s := _en(r, _loguniform(r, 65, 644)), _en_b(r, len(s) + r.randint(30, 60)))),
    "band-difflen-sim": ("d", _POOL_MEDIUM, lambda r: (s := _en(r, _loguniform(r, 65, 644)), _edits(_insert_words(r, s, 30, 60), r.randint(0, 2), r))),
    # Long texts: LargeCtx bands and the full matrix.
    "large-cjk-513-704-dis": ("dr", _POOL_LONG, lambda r: (_text(r, _CJK_A, 513, 704), _text(r, _CJK_B, 513, 704))),
    "long-sim": ("dr", _POOL_LONG, lambda r: (s := _en(r, _loguniform(r, 705, 2048)), _sim(r, s, 0.01, 0.10))),
    "long-near-miss": ("dr", _POOL_LONG, lambda r: (s := _en(r, _loguniform(r, 705, 2048)), _sim(r, s, 0.10, 0.20))),
    "long-dis": ("dr", _POOL_LONG, lambda r: (s := _en(r, _loguniform(r, 705, 2048)), _en_b(r, round(len(s) * r.uniform(0.95, 1.05))))),
    "long-cjk-sim": ("dr", _POOL_LONG, lambda r: (s := _text(r, _CJK_A, 705, 1500), _sim(r, s, 0.01, 0.10, _CJK_TOP))),
    "long-cjk-dis": ("dr", _POOL_LONG, lambda r: (_text(r, _CJK_A, 705, 1500), _text(r, _CJK_B, 705, 1500))),
}
# fmt: on

# Fixed calls per batch, so that base and head always run the same work: the
# DP-size rule for pairs whose cost follows their size, explicit counts where
# affix stripping or banding makes a pair far cheaper than its size.
# CodSpeed rejects the same benchmark name from two jobs of one workflow, so the
# jobs that measure another architecture prefix their names (e.g. "aarch64-"),
# and the running interpreter version is tagged to keep matrix jobs distinct.
_PY_TAG = f"py{sys.version_info.major}.{sys.version_info.minor}"
_PREFIX = os.environ.get("LEV_BENCH_PREFIX", "")


def _case_id(name: str) -> str:
    """
    Format benchmark case ID with architecture prefix and Python version tag.

    Returns:
        str: benchmark case ID.

    """
    return f"{_PREFIX}{name}][{_PY_TAG}"


def _varied(name: str) -> object:
    """
    Build the varied case `name` of `_VARIED_SPECS`: its pool cycled to the batch size.

    Returns:
        object: pytest param of (s1 list, s2 list, n).

    """
    _, size, make = _VARIED_SPECS[name]
    s1, s2 = map(list, zip(*_pairs(name, size, make), strict=True))
    passes = _batch_size(s1, s2) // size
    return pytest.param(s1 * passes, s2 * passes, passes * size, id=_case_id(f"varied-{name}"))


# Single-pair cases, run for both functions.
_CASES = [
    *[pytest.param(*p.values, _batch_size(*p.values), id=_case_id(f"short-{p.id}")) for p in _SHORT],
    *[pytest.param(*p.values, _batch_size(*p.values), id=_case_id(f"medium-{p.id}")) for p in _MEDIUM],
    *[pytest.param(*p.values, _batch_size(*p.values), id=_case_id(f"mixed-{p.id}")) for p in _MIXED],
    *[pytest.param(*p.values, 2_000, id=_case_id(f"affix-{p.id}")) for p in _AFFIX],
    *[pytest.param(*p.values, 4, id=_case_id(f"long-{p.id}")) for p in _LONG],
]
_VARIED = {name: _varied(name) for name in _VARIED_SPECS}
_DISTANCE_CASES = [*_CASES, *_VARIED.values()]
_RATIO_CASES = [*_CASES, *(p for name, p in _VARIED.items() if _VARIED_SPECS[name][0] == "dr")]


@pytest.mark.parametrize(("s1", "s2", "n"), _DISTANCE_CASES)
def test_bench_distance(benchmark: BenchmarkFixture, s1: str | list[str], s2: str | list[str], n: int) -> None:
    """
    Benchmark a batch of lev.distance calls.

    Args:
        benchmark (BenchmarkFixture): CodSpeed fixture timing the batch.
        s1 (str | list[str]): First input string, or the first strings of a pool.
        s2 (str | list[str]): Second input string, or the second strings of a pool.
        n (int): Calls per batch.

    """
    benchmark(_run_batch, lev.distance, s1, s2, n)


@pytest.mark.parametrize(("s1", "s2", "n"), _RATIO_CASES)
def test_bench_ratio(benchmark: BenchmarkFixture, s1: str | list[str], s2: str | list[str], n: int) -> None:
    """
    Benchmark a batch of lev.ratio calls.

    Args:
        benchmark (BenchmarkFixture): CodSpeed fixture timing the batch.
        s1 (str | list[str]): First input string, or the first strings of a pool.
        s2 (str | list[str]): Second input string, or the second strings of a pool.
        n (int): Calls per batch.

    """
    benchmark(_run_batch, lev.ratio, s1, s2, n)
