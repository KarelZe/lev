"""
Integration tests for the `lev` extension module.

Run after building the extension with `maturin develop` (see README).
Expected distances come from rapidfuzz, which serves as the reference
implementation.
"""

from __future__ import annotations

import math
import random
import threading

import pytest
from rapidfuzz.distance import Levenshtein

import lev


@pytest.fixture
def expected(s1: str, s2: str) -> int:
    """
    Return the rapidfuzz distance of the parametrized pair.

    Fixtures run during test setup, which CodSpeed does not measure.

    Returns:
        int: Levenshtein distance between s1 and s2.

    """
    return Levenshtein.distance(s1, s2)


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------

_DISTANCE_PAIRS = [
    ("", ""),
    ("abc", ""),
    ("", "abc"),
    ("hello", "hello"),
    ("kitten", "sitting"),
    ("saturday", "sunday"),
    ("flaw", "lawn"),
    ("gumbo", "gambol"),
    ("intention", "execution"),
    ("a", "b"),
    ("aaaa", "bbbb"),
    # Common-affix stripping must not change the result.
    ("xxx_kitten_yyy", "xxx_sitting_yyy"),
    # Unicode: counted in code points, not bytes.
    ("résumé", "resume"),
    ("café", "cafe"),
    ("日本語", "日本"),
    ("🦀🐍", "🐍🦀"),
    # pylev (duplicates from above removed)
    # https://github.com/toastdriven/pylev/blob/700700ec1b3f637ef1a59bb46f1b2176def2886d/tests.py#L7
    ("meilenstein", "levenshtein"),
    ("levenshtein", "frankenstein"),
    ("confide", "deceit"),
    ("CUNsperrICY", "conspiracy"),
    # Long strings: > 64 chars exercises the multi-word kernels.
    ("abc" * 40, "x" + "abc" * 40),
    ("abc" * 40, "abc" * 40 + "xyz"),
    # 64-char boundary: a pattern of exactly one full word.
    ("a" * 64, "a" * 64),
    ("a" * 64, "a" * 65),
    ("a" * 64, "b" + "a" * 63),
    # Mixed internal encodings (ASCII, Latin-1, UCS-2, UCS-4).
    ("abc", "abc\xff"),  # ASCII vs Latin-1
    ("abc", "abc\u0400"),  # ASCII vs UCS-2
    ("abc", "abc\U0001f400"),  # ASCII vs UCS-4
    ("abc\xff", "abc\u0400"),  # Latin-1 vs UCS-2
    ("abc\u0400", "abc\U0001f400"),  # UCS-2 vs UCS-4
    # Mixed types with common affixes.
    ("prefix_abc", "prefix_abc\xff"),
    ("abc_suffix", "abc\xff_suffix"),
    # Mixed types with multi-word patterns.
    ("a" * 70, ("a" * 70)[:-1] + "\xff"),
    ("a" * 70, ("a" * 70)[:-1] + "\u0400"),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("s1", "s2"), _DISTANCE_PAIRS)
def test_distance(s1: str, s2: str, expected: int) -> None:
    """
    Test and benchmark lev.distance.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.
        expected (int): Expected Levenshtein distance.

    """
    assert lev.distance(s1, s2) == expected
    assert lev.distance(s2, s1) == expected  # symmetric


# ---------------------------------------------------------------------------
# distance: mixed internal encodings at realistic lengths
# ---------------------------------------------------------------------------

# The mixed cases in `test_distance` are tiny or affix-dominated; these pairs
# keep the mixed-kind kernels (single-word, multiword, small-distance, and
# affix stripping) busy at realistic lengths. One side of each pair contains
# a character of a wider kind, so the two strings use different CPython
# internal representations.


def _substitute(s: str, positions: list[int], ch: str) -> str:
    """
    Replace the characters of s at the given positions with ch.

    Returns:
        str: copy of s with the substitutions applied.

    """
    b = list(s)
    for p in positions:
        b[p] = ch
    return "".join(b)


_ASCII_100 = ("The quick brown fox jumps over the lazy dog. " * 3)[:100]
_LATIN1_100 = ("café résumé naïve façade jalapeño Zürich smörgåsbord " * 2)[:100]
_CJK_100 = ("日本語のテスト文字列を長くするために繰り返します。" * 5)[:100]

_MIXED_KIND_CASES = [
    pytest.param(
        _ASCII_100,
        _substitute(_ASCII_100, [0, 33, 66, 99], "😀"),
        id="ascii-vs-emoji-100",
    ),
    pytest.param(
        _ASCII_100,
        _substitute(_ASCII_100, [0, 33, 66, 99], "日"),
        id="ascii-vs-cjk-100",
    ),
    pytest.param(
        _ASCII_100,
        _substitute(_ASCII_100, [0, 33, 66, 99], "ÿ"),
        id="ascii-vs-latin1-100",
    ),
    pytest.param(
        _LATIN1_100,
        _substitute(_LATIN1_100, [0, 33, 66, 99], "😀"),
        id="latin1-vs-emoji-100",
    ),
    pytest.param(
        _CJK_100,
        _substitute(_CJK_100, [0, 33, 66, 99], "😀"),
        id="cjk-vs-emoji-100",
    ),
    # Single-word mixed kernel (pattern <= 64 chars).
    pytest.param(
        _ASCII_100[:48],
        _substitute(_ASCII_100[:48], [0, 47], "😀"),
        id="ascii-vs-emoji-48",
    ),
    # Near-identical mixed pair (small-distance fast path).
    pytest.param(
        _ASCII_100,
        _substitute(_ASCII_100, [50], "😀"),
        id="ascii-vs-emoji-d1",
    ),
    # Long shared affixes around a mixed-kind difference.
    pytest.param(
        "x" * 80 + "middle" + "y" * 80,
        "x" * 80 + "m😀ddle" + "y" * 80,
        id="mixed-affix-heavy",
    ),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("s1", "s2"), _MIXED_KIND_CASES)
def test_distance_mixed_kind(s1: str, s2: str, expected: int) -> None:
    """
    Test and benchmark lev.distance on pairs with different internal encodings.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string (wider CPython string kind than s1).
        expected (int): Expected Levenshtein distance.

    """
    assert lev.distance(s1, s2) == expected
    assert lev.distance(s2, s1) == expected  # symmetric


# ---------------------------------------------------------------------------
# distance: long strings (> 512 chars, banded multiword kernel)
# ---------------------------------------------------------------------------


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


_ASCII = "abcdefghij"
_UCS2 = "ぁあぃいぅうぇえぉお"

# Similar pairs resolve inside a narrow Ukkonen band; dissimilar pairs measure the banded passes' overhead
# on top of the full-matrix fallback; moderate sits in between.
_LONG_CASES = [
    pytest.param(
        _rand_str(_ASCII, 2048, seed=1),
        _mutate(_rand_str(_ASCII, 2048, seed=1), _ASCII, edits=4, seed=2),
        id="ascii-2048-similar",
    ),
    pytest.param(
        _rand_str(_ASCII, 8192, seed=3),
        _mutate(_rand_str(_ASCII, 8192, seed=3), _ASCII, edits=8, seed=4),
        id="ascii-8192-similar",
    ),
    pytest.param(
        _rand_str(_ASCII, 2048, seed=5),
        _mutate(_rand_str(_ASCII, 2048, seed=5), _ASCII, edits=205, seed=6),
        id="ascii-2048-moderate",
    ),
    pytest.param(
        _rand_str(_ASCII, 2048, seed=7),
        _rand_str(_ASCII, 2048, seed=8),
        id="ascii-2048-dissimilar",
    ),
    pytest.param(
        _rand_str(_ASCII, 8192, seed=9),
        _rand_str(_ASCII, 8192, seed=10),
        id="ascii-8192-dissimilar",
    ),
    pytest.param(
        _rand_str(_UCS2, 2048, seed=11),
        _mutate(_rand_str(_UCS2, 2048, seed=11), _UCS2, edits=4, seed=12),
        id="ucs2-2048-similar",
    ),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("s1", "s2"), _LONG_CASES)
def test_distance_long(s1: str, s2: str, expected: int) -> None:
    """
    Test and benchmark lev.distance on strings beyond the 512-char gate.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.
        expected (int): Expected Levenshtein distance.

    """
    assert lev.distance(s1, s2) == expected
    assert lev.distance(s2, s1) == expected  # symmetric


# ---------------------------------------------------------------------------
# distance: medium strings (9-512 chars, single- and multi-word kernels)
# ---------------------------------------------------------------------------

_LATIN1 = "àáâãäåæçèé"
_UCS4 = "".join(chr(0x1F600 + i) for i in range(10))


def _medium(alphabet: str, n: int, edits: int, seed: int, id: str) -> object:  # noqa: A002
    """
    Build a mutated pair far enough apart that the mbleven path cannot fire.

    Returns:
        object: pytest param of (s1, s2, expected).

    """
    s = _rand_str(alphabet, n, seed=seed)
    return pytest.param(s, _mutate(s, alphabet, edits=edits, seed=seed + 1), id=id)


# 32/64-char UCS-2/4 pairs run
# the single-word hash kernel; 100/300-char pairs run the multi-word kernels
# (stack peq for UCS-1, hash-indexed peq for UCS-2 and mixed kinds).
_MEDIUM_CASES = [
    _medium(_UCS2, 32, 8, seed=100, id="ucs2-32"),
    _medium(_UCS2, 64, 16, seed=102, id="ucs2-64"),
    _medium(_UCS4, 32, 8, seed=104, id="ucs4-32"),
    _medium(_UCS4, 64, 16, seed=106, id="ucs4-64"),
    _medium(_UCS2, 100, 25, seed=108, id="ucs2-100"),
    _medium(_ASCII, 100, 25, seed=110, id="ascii-100"),
    _medium(_ASCII, 300, 75, seed=112, id="ascii-300"),
    _medium(_LATIN1, 100, 25, seed=114, id="latin1-100"),
    _medium(_LATIN1, 300, 75, seed=116, id="latin1-300"),
    pytest.param(
        _rand_str(_ASCII, 100, seed=130),
        _mutate(_rand_str(_ASCII, 100, seed=130), _UCS2, edits=25, seed=131),
        id="mixed-100",
    ),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("s1", "s2"), _MEDIUM_CASES)
def test_distance_medium(s1: str, s2: str, expected: int) -> None:
    """
    Test and benchmark lev.distance on strings between the tiny and banded paths.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.
        expected (int): Expected Levenshtein distance.

    """
    assert lev.distance(s1, s2) == expected
    assert lev.distance(s2, s1) == expected  # symmetric


# ---------------------------------------------------------------------------
# ratio
# ---------------------------------------------------------------------------


def _ratio(s1: str, s2: str) -> float:
    """
    Compute the expected `lev.ratio` from the rapidfuzz distance.

    Returns:
        float: `1 - distance / (len(s1) + len(s2))`, or 1.0 for two empty strings.

    """
    total = len(s1) + len(s2)
    return 1.0 if total == 0 else 1.0 - Levenshtein.distance(s1, s2) / total


@pytest.fixture
def expected_ratio(s1: str, s2: str) -> float:
    """
    Return the expected ratio of the parametrized pair during test setup.

    Returns:
        float: expected `lev.ratio(s1, s2)`.

    """
    return _ratio(s1, s2)


_RATIO_PAIRS = [
    ("", ""),
    ("abc", "abc"),
    ("abc", "xyz"),
    ("kitten", "sitting"),
    ("a", "b"),
    # Long strings.
    ("abc" * 40, "x" + "abc" * 40),
    ("a" * 64, "a" * 65),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("s1", "s2"), _RATIO_PAIRS)
def test_ratio(s1: str, s2: str, expected_ratio: float) -> None:
    """
    Test and benchmark lev.ratio.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.
        expected_ratio (float): Expected ratio.

    """
    assert math.isclose(lev.ratio(s1, s2), expected_ratio, abs_tol=1e-12)


def test_ratio_unicode_uses_code_points() -> None:
    """Test ratio implementation with unicode strings."""
    # 6 + 6 code points; distance 2 => 1 - 2/12.
    assert math.isclose(lev.ratio("résumé", "resume"), _ratio("résumé", "resume"), abs_tol=1e-12)


# ---------------------------------------------------------------------------
# randomized oracle tests: lev.distance vs rapidfuzz
# ---------------------------------------------------------------------------

# Alphabets chosen to exercise every CPython string kind (PEP 393) plus
# mixed-kind pairs; small sizes force shared affixes and the tiny-pattern path.
ALPHABETS = {
    "ascii": "abcd",
    "latin1": "\xe0\xe1\xe2\xe3",
    "ucs2": "ぁあぃい",
    "ucs4": "\U0001f600\U0001f601\U0001f602\U0001f603",
    "mixed": "abぁ\U0001f600",
}


@pytest.mark.parametrize("kind", list(ALPHABETS))
def test_small_strings_match_oracle(kind: str) -> None:
    """Random small strings (length 0-13) across all string kinds."""
    rng = random.Random(42)
    alpha = ALPHABETS[kind]
    for _ in range(1000):
        a = "".join(rng.choice(alpha) for _ in range(rng.randrange(0, 14)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randrange(0, 14)))
        assert lev.distance(a, b) == Levenshtein.distance(a, b), (a, b)


def test_long_strings_with_affixes_match_oracle() -> None:
    """Random strings crossing the 64-char word boundary with shared affixes."""
    rng = random.Random(7)
    for _ in range(200):
        n1, n2 = rng.randrange(50, 140), rng.randrange(50, 140)
        core = "".join(rng.choice("abcdef") for _ in range(n1))
        mutated = "".join(c if rng.random() > 0.15 else rng.choice("abcdef") for c in core)[:n2]
        pre = "prefix" * rng.randrange(0, 4)
        suf = "suffix" * rng.randrange(0, 4)
        a, b = pre + core + suf, pre + mutated + suf
        assert lev.distance(a, b) == Levenshtein.distance(a, b), (a, b)


def test_affix_stripping_partial_element() -> None:
    """UCS-2 code units whose raw bytes match past an element boundary."""
    a = chr(0x0101) + chr(0x0102) + chr(0x0201)
    b = chr(0x0101) + chr(0x0202) + chr(0x0201)
    assert lev.distance(a, b) == 1


@pytest.mark.parametrize("n", range(1, 18))
def test_affix_stripping_unaligned_lengths(n: int) -> None:
    """One string a prefix of the other, crossing the 8-byte chunk boundary."""
    assert lev.distance("a" * n, "a" * (n + 1)) == 1
    assert lev.distance("a" * n + "b", "a" * n) == 1


@pytest.mark.parametrize(("n", "edits"), [(550, 2), (700, 40), (650, 200)])
def test_banded_long_strings_match_oracle(n: int, edits: int) -> None:
    """Strings > 512 chars route through the banded multiword kernel."""
    a = _rand_str(_ASCII, n, seed=n + edits)
    b = _mutate(a, _ASCII, edits, seed=n + edits)
    # A leading shift additionally defeats affix stripping and the Hamming bound.
    b = "x" + b[:-1]
    assert lev.distance(a, b) == Levenshtein.distance(a, b)
    assert lev.distance(b, a) == Levenshtein.distance(a, b)  # symmetric


# ---------------------------------------------------------------------------
# Very different lengths
# ---------------------------------------------------------------------------

# Short lengths hit every kernel regime (empty, tiny <= 8, single word,
# multi-word, heap-backed > 512); long lengths are several times longer.
_DISPARATE_LENGTHS = [
    (0, 50),
    (1, 200),
    (5, 300),
    (8, 120),
    (9, 400),
    (40, 800),
    (64, 1300),
    (65, 650),
    (130, 1500),
    (520, 1600),
]


def _embed(short: str, long_len: int, alpha: str, where: str, seed: int) -> str:
    """
    Pad `short` with random characters to `long_len`, placing it at `where`.

    Returns:
        str: the padded string, with `short` as a contiguous substring.

    """
    rng = random.Random(seed)
    pad = "".join(rng.choice(alpha) for _ in range(long_len - len(short)))
    cut = {"start": 0, "middle": len(pad) // 2, "end": len(pad)}[where]
    return pad[:cut] + short + pad[cut:]


@pytest.mark.parametrize(("m", "n"), _DISPARATE_LENGTHS)
@pytest.mark.parametrize("where", ["start", "middle", "end"])
def test_substring_of_much_longer_string(m: int, n: int, where: str) -> None:
    """
    A string inside a much longer one: exactly `n - m` deletions apart.

    The ratio is then `2m / (m + n)`, whether it is normalized by the
    Levenshtein or by the indel distance.

    """
    alpha = "abcdefgh"
    short = _rand_str(alpha, m, seed=m)
    long = _embed(short, n, alpha, where, seed=m + n)
    assert lev.distance(short, long) == n - m
    assert lev.distance(long, short) == n - m
    expected = 1.0 if m + n == 0 else 2 * m / (m + n)
    assert math.isclose(lev.ratio(short, long), expected, abs_tol=1e-12)
    assert math.isclose(lev.ratio(long, short), expected, abs_tol=1e-12)


@pytest.mark.parametrize(("m", "n"), _DISPARATE_LENGTHS)
@pytest.mark.parametrize("kind", list(ALPHABETS))
def test_very_different_lengths_match_oracle(m: int, n: int, kind: str) -> None:
    """Distance of unrelated strings and of a mutated copy inside padding."""
    alpha = ALPHABETS[kind]
    short = _rand_str(alpha, m, seed=m + 1)
    pairs = [
        (short, _rand_str(alpha, n, seed=n + 2)),
        (short, _embed(_mutate(short, alpha, max(1, m // 10), seed=m) if m else "", n, alpha, "middle", seed=n)),
    ]
    for a, b in pairs:
        d = Levenshtein.distance(a, b)
        assert lev.distance(a, b) == d, (kind, m, n)
        assert lev.distance(b, a) == d, (kind, m, n)


# ---------------------------------------------------------------------------
# Concurrency (meaningful on free-threaded builds, harmless with the GIL)
# ---------------------------------------------------------------------------


def test_concurrent_calls_from_threads() -> None:
    """Many threads share the same string objects; every result must stay exact."""
    pairs = [
        (_rand_str(ALPHABETS[kind], n, seed=n), _rand_str(ALPHABETS[kind], n + 7, seed=n + 1))
        for kind in ALPHABETS
        for n in (12, 70, 600)
    ]
    expected = [(Levenshtein.distance(a, b), lev.ratio(a, b)) for a, b in pairs]
    barrier = threading.Barrier(8)
    errors: list[tuple[str, str]] = []

    def worker() -> None:
        barrier.wait()
        for _ in range(50):
            for (a, b), (d, r) in zip(pairs, expected, strict=True):
                if lev.distance(a, b) != d or lev.ratio(b, a) != r:
                    errors.append((a, b))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
