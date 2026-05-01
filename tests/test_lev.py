"""Integration tests for the `lev` extension module.

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
    ],
)
def test_distance(s1: str, s2: str, expected: int) -> None:
    assert lev.distance(s1, s2) == expected
    assert lev.distance(s2, s1) == expected  # symmetric


def test_distance_long_inputs() -> None:
    # > 64 chars on the shorter side exercises the Wagner-Fischer fallback.
    a = "abc" * 40  # 120 chars
    b = "x" + a
    assert lev.distance(a, b) == 1
    assert lev.distance(a, a + "xyz") == 3


def test_distance_64_boundary() -> None:
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
    assert math.isclose(lev.ratio(s1, s2), expected, abs_tol=1e-12)


def test_ratio_unicode_uses_code_points() -> None:
    # 6 + 6 code points; distance 2 => 1 - 2/12.
    assert math.isclose(lev.ratio("résumé", "resume"), 1.0 - 2.0 / 12.0, abs_tol=1e-12)
