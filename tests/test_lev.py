"""
Integration tests for the `lev` extension module.

Run after building the extension with `maturin develop` (see README).
"""

from __future__ import annotations

import math

import pytest

import lev

# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


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
        ("kitten", "sitting", 3),
        ("meilenstein", "levenshtein", 4),
        ("levenshtein", "frankenstein", 6),
        ("confide", "deceit", 6),
        ("CUNsperrICY", "conspiracy", 8),
    ],
)
def test_distance(s1: str, s2: str, expected: int) -> None:
    """
    Test levenshtein distance.

    Args:
        s1 (str): left string
        s2 (str): right string
        expected (int): expected distance

    """
    assert lev.distance(s1, s2) == expected
    assert lev.distance(s2, s1) == expected  # symmetric


def test_distance_long_inputs() -> None:
    """Test levenshtein distance."""
    # > 64 chars on the shorter side exercises the Wagner-Fischer fallback.
    a = "abc" * 40  # 120 chars
    b = "x" + a
    assert lev.distance(a, b) == 1
    assert lev.distance(a, a + "xyz") == 3


def test_distance_64_boundary() -> None:
    """Test levenshtein distance at 64 char boundary."""
    a64 = "a" * 64
    a65 = "a" * 65
    assert lev.distance(a64, a64) == 0
    assert lev.distance(a64, a65) == 1
    # Pattern length exactly 64 hits the `m == 64` branch.
    assert lev.distance(a64, "b" + "a" * 63) == 1


# ---------------------------------------------------------------------------
# ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("s1", "s2", "expected"),
    [
        ("", "", 1.0),
        ("abc", "abc", 1.0),
        ("abc", "xyz", 1.0 - 3.0 / 6.0),
        ("kitten", "sitting", 1.0 - 3.0 / 13.0),
        ("a", "b", 1.0 - 1.0 / 2.0),
    ],
)
def test_ratio(s1: str, s2: str, expected: float) -> None:
    """
    Test ratio implementation.

    Args:
        s1 (str): left string
        s2 (str): right string
        expected (float): expected ratio

    """
    assert math.isclose(lev.ratio(s1, s2), expected, abs_tol=1e-12)


def test_ratio_unicode_uses_code_points() -> None:
    """Test ratio implementation with unicode strings."""
    # 6 + 6 code points; distance 2 => 1 - 2/12.
    assert math.isclose(lev.ratio("résumé", "resume"), 1.0 - 2.0 / 12.0, abs_tol=1e-12)
