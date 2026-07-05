"""Randomized oracle tests: `lev.distance` vs a reference DP implementation."""

from __future__ import annotations

import random

import pytest

import lev

# Alphabets chosen to exercise every CPython string kind (PEP 393) plus
# mixed-kind pairs; small sizes force shared affixes and the tiny-pattern path.
ALPHABETS = {
    "ascii": "abcd",
    "latin1": "\xe0\xe1\xe2\xe3",
    "ucs2": "ぁあぃい",
    "ucs4": "\U0001f600\U0001f601\U0001f602\U0001f603",
    "mixed": "abぁ\U0001f600",
}


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
