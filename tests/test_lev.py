"""
Integration tests for the `lev` extension module.

Run after building the extension with `maturin develop` (see README).
"""

from __future__ import annotations

import math
import random

import pytest

import lev

# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("s1", "s2", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("hello", "hello", 0),
        ("kitten", "sitting", 3),
        ("saturday", "sunday", 3),
        ("flaw", "lawn", 2),
        ("gumbo", "gambol", 2),
        ("intention", "execution", 5),
        ("a", "b", 1),
        ("aaaa", "bbbb", 4),
        # Common-affix stripping must not change the result.
        ("xxx_kitten_yyy", "xxx_sitting_yyy", 3),
        # Unicode: counted in code points, not bytes.
        ("résumé", "resume", 2),
        ("café", "cafe", 1),
        ("日本語", "日本", 1),
        ("🦀🐍", "🐍🦀", 2),
        # pylev (duplicates from above removed)
        # https://github.com/toastdriven/pylev/blob/700700ec1b3f637ef1a59bb46f1b2176def2886d/tests.py#L7
        ("meilenstein", "levenshtein", 4),
        ("levenshtein", "frankenstein", 6),
        ("confide", "deceit", 6),
        ("CUNsperrICY", "conspiracy", 8),
        # Long strings: > 64 chars exercises the fallback algorithm.
        ("abc" * 40, "x" + "abc" * 40, 1),
        ("abc" * 40, "abc" * 40 + "xyz", 3),
        # 64-char boundary: pattern length exactly 64 hits the `m == 64` branch.
        ("a" * 64, "a" * 64, 0),
        ("a" * 64, "a" * 65, 1),
        ("a" * 64, "b" + "a" * 63, 1),
        # Mixed internal encodings (ASCII, Latin-1, UCS-2, UCS-4).
        ("abc", "abc\xff", 1),  # ASCII vs Latin-1
        ("abc", "abc\u0400", 1),  # ASCII vs UCS-2
        ("abc", "abc\U0001f400", 1),  # ASCII vs UCS-4
        ("abc\xff", "abc\u0400", 1),  # Latin-1 vs UCS-2
        ("abc\u0400", "abc\U0001f400", 1),  # UCS-2 vs UCS-4
        # Mixed types with common affixes.
        ("prefix_abc", "prefix_abc\xff", 1),
        ("abc_suffix", "abc\xff_suffix", 1),
        # Mixed types with multi-word patterns.
        ("a" * 70, ("a" * 70)[:-1] + "\xff", 1),
        ("a" * 70, ("a" * 70)[:-1] + "\u0400", 1),
    ],
)
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
# internal representations. Expected distances verified against rapidfuzz.


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
        4,
        id="ascii-vs-emoji-100",
    ),
    pytest.param(
        _ASCII_100,
        _substitute(_ASCII_100, [0, 33, 66, 99], "日"),
        4,
        id="ascii-vs-cjk-100",
    ),
    pytest.param(
        _ASCII_100,
        _substitute(_ASCII_100, [0, 33, 66, 99], "ÿ"),
        4,
        id="ascii-vs-latin1-100",
    ),
    pytest.param(
        _LATIN1_100,
        _substitute(_LATIN1_100, [0, 33, 66, 99], "😀"),
        4,
        id="latin1-vs-emoji-100",
    ),
    pytest.param(
        _CJK_100,
        _substitute(_CJK_100, [0, 33, 66, 99], "😀"),
        4,
        id="cjk-vs-emoji-100",
    ),
    # Single-word mixed kernel (pattern <= 64 chars).
    pytest.param(
        _ASCII_100[:48],
        _substitute(_ASCII_100[:48], [0, 47], "😀"),
        2,
        id="ascii-vs-emoji-48",
    ),
    # Near-identical mixed pair (small-distance fast path).
    pytest.param(
        _ASCII_100,
        _substitute(_ASCII_100, [50], "😀"),
        1,
        id="ascii-vs-emoji-d1",
    ),
    # Long shared affixes around a mixed-kind difference.
    pytest.param(
        "x" * 80 + "middle" + "y" * 80,
        "x" * 80 + "m😀ddle" + "y" * 80,
        1,
        id="mixed-affix-heavy",
    ),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("s1", "s2", "expected"), _MIXED_KIND_CASES)
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

# Expected distances verified against rapidfuzz. Similar pairs resolve inside
# a narrow Ukkonen band; dissimilar pairs measure the banded passes' overhead
# on top of the full-matrix fallback; moderate sits in between.
_LONG_CASES = [
    pytest.param(
        _rand_str(_ASCII, 2048, seed=1),
        _mutate(_rand_str(_ASCII, 2048, seed=1), _ASCII, edits=4, seed=2),
        4,
        id="ascii-2048-similar",
    ),
    pytest.param(
        _rand_str(_ASCII, 8192, seed=3),
        _mutate(_rand_str(_ASCII, 8192, seed=3), _ASCII, edits=8, seed=4),
        7,
        id="ascii-8192-similar",
    ),
    pytest.param(
        _rand_str(_ASCII, 2048, seed=5),
        _mutate(_rand_str(_ASCII, 2048, seed=5), _ASCII, edits=205, seed=6),
        186,
        id="ascii-2048-moderate",
    ),
    pytest.param(
        _rand_str(_ASCII, 2048, seed=7),
        _rand_str(_ASCII, 2048, seed=8),
        1522,
        id="ascii-2048-dissimilar",
    ),
    pytest.param(
        _rand_str(_ASCII, 8192, seed=9),
        _rand_str(_ASCII, 8192, seed=10),
        6069,
        id="ascii-8192-dissimilar",
    ),
    pytest.param(
        _rand_str(_UCS2, 2048, seed=11),
        _mutate(_rand_str(_UCS2, 2048, seed=11), _UCS2, edits=4, seed=12),
        4,
        id="ucs2-2048-similar",
    ),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("s1", "s2", "expected"), _LONG_CASES)
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
# ratio
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("s1", "s2", "expected"),
    [
        ("", "", 1.0),
        ("abc", "abc", 1.0),
        # No common subsequence at all: the ratio bottoms out at exactly 0.0.
        ("abc", "xyz", 0.0),
        ("a", "b", 0.0),
        # LCS "ittn" (4) => indel distance 6 + 7 - 8 = 5.
        ("kitten", "sitting", 1.0 - 5.0 / 13.0),
        # Long strings.
        ("abc" * 40, "x" + "abc" * 40, 1.0 - 1.0 / 241.0),
        ("a" * 64, "a" * 65, 1.0 - 1.0 / 129.0),
        # A substitution costs 2 under the indel metric, an indel costs 1.
        ("abcdef", "abcXef", 1.0 - 2.0 / 12.0),
        ("abcdef", "abcef", 1.0 - 1.0 / 11.0),
    ],
)
def test_ratio(s1: str, s2: str, expected: float) -> None:
    """
    Test and benchmark lev.ratio.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.
        expected (float): Expected ratio.

    """
    assert math.isclose(lev.ratio(s1, s2), expected, abs_tol=1e-12)
    assert math.isclose(lev.ratio(s2, s1), expected, abs_tol=1e-12)  # symmetric


def test_ratio_unicode_uses_code_points() -> None:
    """Test ratio implementation with unicode strings."""
    # 6 + 6 code points; two substitutions => indel distance 4 => 1 - 4/12.
    assert math.isclose(lev.ratio("résumé", "resume"), 1.0 - 4.0 / 12.0, abs_tol=1e-12)


@pytest.mark.parametrize(
    ("s1", "s2"),
    [
        ("kitten", "sitting"),
        ("日本語のテスト", "日本語のテスト文字列"),
        ("😀🎉🚀✨🐍", "😀🎉🚀✨🦀"),
        ("café résumé", "cafe resume"),
        ("a" * 100, "b" * 100),
        ("abcdefghij" * 60, "abcdefghij" * 30 + "x" + "abcdefghij" * 30),
    ],
)
def test_ratio_matches_indel_definition(s1: str, s2: str) -> None:
    """
    Test lev.ratio against the indel distance derived from a reference LCS.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.

    """
    expected = 1.0 - naive_indel(s1, s2) / (len(s1) + len(s2))
    assert math.isclose(lev.ratio(s1, s2), expected, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# randomized oracle tests: lev.distance vs a reference DP implementation
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


def naive_indel(a: str, b: str) -> int:
    """
    Compute the indel distance (insertions and deletions only) with a DP.

    Returns:
        int: indel distance between a and b, i.e. len(a) + len(b) - 2 * LCS.

    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i - 1] == b[j - 1] else min(dp[j], dp[j - 1]) + 1
    return dp[n]


def naive(a: str, b: str) -> int:
    """
    Compute the Levenshtein distance with the textbook DP recurrence.

    Returns:
        int: edit distance between a and b.

    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return dp[n]


@pytest.mark.parametrize("kind", list(ALPHABETS))
def test_small_strings_match_oracle(kind: str) -> None:
    """Random small strings (length 0-13) across all string kinds."""
    rng = random.Random(42)
    alpha = ALPHABETS[kind]
    for _ in range(1000):
        a = "".join(rng.choice(alpha) for _ in range(rng.randrange(0, 14)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randrange(0, 14)))
        assert lev.distance(a, b) == naive(a, b), (a, b)


@pytest.mark.parametrize("kind", list(ALPHABETS))
def test_ratio_small_strings_match_oracle(kind: str) -> None:
    """Random small strings (length 0-13) across all string kinds."""
    rng = random.Random(43)
    alpha = ALPHABETS[kind]
    for _ in range(1000):
        a = "".join(rng.choice(alpha) for _ in range(rng.randrange(0, 14)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randrange(0, 14)))
        total = len(a) + len(b)
        expected = 1.0 if total == 0 else 1.0 - naive_indel(a, b) / total
        assert math.isclose(lev.ratio(a, b), expected, abs_tol=1e-12), (a, b)


@pytest.mark.parametrize("n", [63, 64, 65, 128, 300, 513, 700])
def test_ratio_long_strings_match_oracle(n: int) -> None:
    """Random long strings crossing the word, multi-word, and heap boundaries."""
    alpha = "abcdefgh"
    for edits in (1, 5, n // 4):
        a = _rand_str(alpha, n, seed=n + edits)
        b = _mutate(a, alpha, edits, seed=n * 7 + edits)
        expected = 1.0 - naive_indel(a, b) / (len(a) + len(b))
        assert math.isclose(lev.ratio(a, b), expected, abs_tol=1e-12), (n, edits)
    # Disjoint alphabets: nothing in common, so the ratio bottoms out at 0.0.
    assert math.isclose(lev.ratio("x" * n, "y" * n), 0.0, abs_tol=1e-12)


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
        assert lev.distance(a, b) == naive(a, b), (a, b)


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
    assert lev.distance(a, b) == naive(a, b)
    assert lev.distance(b, a) == naive(a, b)  # symmetric
