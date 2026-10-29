"""
CodSpeed benchmarks for `lev.distance` and `lev.ratio`.

Each benchmark times only a batch of calls through the `benchmark` fixture.
Timing a whole test instead measured about 185 us of pytest and
instrumentation overhead per test against 0.03-1 us of lev work, which hid
every change to lev below CodSpeed's noise floor. The batch is driven from
C (`map` drained into a zero-length `deque`), so the interpreter adds little
per call, and its size is fixed per pair so that every batch does a comparable amount
of work. Correctness lives in `test_lev.py`.
"""

from __future__ import annotations

import os
import random
import sys
from collections import deque
from itertools import repeat
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


def _batch_size(s1: str, s2: str) -> int:
    """
    Return the calls per batch for a pair, inversely proportional to its DP size.

    Returns:
        int: number of calls, at least 1.

    """
    return max(1, min(_MAX_BATCH, _BATCH_CELLS // max(1, len(s1) * len(s2))))


def _run_batch(fn: Callable[[str, str], object], s1: str, s2: str, n: int) -> None:
    """Call `fn(s1, s2)` n times with as little interpreter work as possible."""
    deque(map(fn, repeat(s1, n), repeat(s2, n)), maxlen=0)


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


_CASES = [
    *[pytest.param(*p.values, _batch_size(*p.values), id=_case_id(f"short-{p.id}")) for p in _SHORT],
    *[pytest.param(*p.values, _batch_size(*p.values), id=_case_id(f"medium-{p.id}")) for p in _MEDIUM],
    *[pytest.param(*p.values, _batch_size(*p.values), id=_case_id(f"mixed-{p.id}")) for p in _MIXED],
    *[pytest.param(*p.values, 2_000, id=_case_id(f"affix-{p.id}")) for p in _AFFIX],
    *[pytest.param(*p.values, 4, id=_case_id(f"long-{p.id}")) for p in _LONG],
]


@pytest.mark.parametrize(("s1", "s2", "n"), _CASES)
def test_bench_distance(benchmark: BenchmarkFixture, s1: str, s2: str, n: int) -> None:
    """
    Benchmark a batch of lev.distance calls.

    Args:
        benchmark (BenchmarkFixture): CodSpeed fixture timing the batch.
        s1 (str): First input string.
        s2 (str): Second input string.
        n (int): Calls per batch.

    """
    benchmark(_run_batch, lev.distance, s1, s2, n)


@pytest.mark.parametrize(("s1", "s2", "n"), _CASES)
def test_bench_ratio(benchmark: BenchmarkFixture, s1: str, s2: str, n: int) -> None:
    """
    Benchmark a batch of lev.ratio calls.

    Args:
        benchmark (BenchmarkFixture): CodSpeed fixture timing the batch.
        s1 (str): First input string.
        s2 (str): Second input string.
        n (int): Calls per batch.

    """
    benchmark(_run_batch, lev.ratio, s1, s2, n)
