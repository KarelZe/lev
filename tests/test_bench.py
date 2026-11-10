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
- Varied cases (`varied-*`) cycle through a seeded pool of random pairs of
  one class (e.g. CJK strings of 17-40 chars, short words with typos,
  65-704-char strings just past the band's certification limit).

Judge every gate or threshold change (which path a pair takes) on the varied
cases. On Apple M2, three gates flipped sign between the two kinds: the
64-slot single-word hash, TINY_M = 16 for wide strings and mbleven at an
upper bound of 3. A win only on single-pair cases is a branch predictor
artefact.
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


def _mutate(s: str, alphabet: str, edits: int, seed: int, ops: int = 3) -> str:
    """
    Apply `edits` random substitutions, insertions, and deletions to s.

    `ops=1` applies substitutions only.

    Returns:
        str: mutated copy of s.

    """
    rng = random.Random(seed)
    b = list(s)
    for _ in range(edits):
        i = rng.randrange(len(b))
        op = rng.randrange(ops)
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

# Varied cases: each cycles through a seeded pool of random pairs of one
# class, so that the branch predictor cannot learn one pair's branches. On M2,
# gate ratios settle once a pool holds about 100k chars of short strings (the
# TINY_M distance ratio was 0.94 with 4 pairs, 1.51 with 1024 and 1.95 with
# 4096 pairs of 9-16 CJK chars) and with 256 pairs of 65-704 chars. Long pools
# stay small to bound the batch time.
_CJK = "".join(chr(c) for c in range(0x4E00, 0xA000, 82))  # 256 ideographs
_CYRILLIC = "".join(chr(c) for c in range(0x430, 0x450)) + " "
_EMOJI = "".join(chr(c) for c in range(0x1F300, 0x1F400))

# (name, ratio too, alphabet, length range, edits, appended chars, pool size).
# The second string is an unrelated one of the same length (edits None), the
# first with that fraction of random substitutions, insertions and deletions
# (float, +-25%), or with that many substitutions (int; mbleven's upper bound
# counts aligned mismatches). Then a random tail with a length from the
# appended-chars range follows. mbleven and the small bands are
# Levenshtein-only, so their cases run distance only.
# fmt: off
_VARIED_SPECS = [
    # Single-word hash for UCS-2/4 patterns of 17-64 chars (64 or 128 slots).
    ("cjk-17-40-dis", True, _CJK, (17, 40), None, (0, 0), 4096),
    ("cjk-17-40-rel", True, _CJK, (17, 40), 0.25, (0, 0), 4096),
    ("cyrillic-17-40-dis", True, _CYRILLIC, (17, 40), None, (0, 0), 4096),
    ("emoji-17-40-dis", True, _EMOJI, (17, 40), None, (0, 0), 4096),
    # Tiny path for wide strings (TINY_M).
    ("cjk-9-16-dis", True, _CJK, (9, 16), None, (0, 0), 4096),
    ("cjk-9-16-rel", True, _CJK, (9, 16), 0.15, (0, 0), 4096),
    ("emoji-9-16-dis", True, _EMOJI, (9, 16), None, (0, 0), 4096),
    # Short words with typos and mbleven at its upper bound of 3.
    ("ascii-4-24-typo", True, _ASCII, (4, 24), 0.08, (0, 0), 1024),
    ("latin1-4-24-typo", True, _LATIN1, (4, 24), 0.08, (0, 0), 1024),
    ("ascii-17-32-ub3", False, _ASCII, (17, 32), 3, (0, 0), 1024),
    ("ascii-33-64-ub3", False, _ASCII, (33, 64), 3, (0, 0), 1024),
    # Byte patterns of 17-64 chars (table-free masks), also against longer texts.
    ("latin1-17-64-dis", True, _LATIN1, (17, 64), None, (0, 0), 1024),
    ("ascii-17-64-rel", True, _ASCII, (17, 64), 0.1, (0, 0), 1024),
    ("ascii-4-16-vs-long", True, _ASCII, (4, 16), None, (8, 240), 1024),
    ("ascii-17-64-vs-long", True, _ASCII, (17, 64), None, (17, 450), 1024),
    ("cjk-9-40-vs-long", True, _CJK, (9, 40), None, (9, 280), 1024),
    # 65-704 chars: speculative small bands (#260).
    ("band-sim", True, _ASCII, (65, 704), 0.04, (0, 0), 256),
    ("band-near-miss", False, _ASCII, (80, 250), 0.5, (0, 0), 256),
    ("band-dis", True, _ASCII, (65, 704), None, (0, 29), 256),
    ("band-difflen-sim", False, _ASCII, (65, 644), 0.005, (30, 60), 256),
    ("band-difflen-dis", False, _ASCII, (65, 644), None, (30, 60), 256),
    # Long texts: LargeCtx bands and the full matrix.
    ("large-cjk-513-704-dis", True, _CJK, (513, 704), None, (0, 0), 32),
    ("long-sim", True, _ASCII, (705, 2048), 0.05, (0, 0), 32),
    ("long-near-miss", True, _ASCII, (705, 2048), 0.15, (0, 0), 32),
    ("long-dis", True, _ASCII, (705, 2048), None, (0, 0), 32),
    ("long-cjk-sim", True, _CJK, (705, 1500), 0.05, (0, 0), 32),
    ("long-cjk-dis", True, _CJK, (705, 1500), None, (0, 0), 32),
]
# fmt: on


def _pool(
    name: str,
    alphabet: str,
    lengths: tuple[int, int],
    edits: float | int | None,
    extra: tuple[int, int],
    size: int,
) -> tuple[list[str], list[str]]:
    """
    Build the seeded pool of one varied case (see `_VARIED_SPECS`).

    Returns:
        tuple[list[str], list[str]]: first and second strings of `size` pairs.

    """
    rng = random.Random(name)
    # Random bytes mapped to the alphabet (at most 256 chars), which is far
    # faster than one `rng.choice` per char and keeps the import fast.
    table = {i: alphabet[i % len(alphabet)] for i in range(256)}

    def chars(k: int) -> str:
        return rng.randbytes(k).decode("latin-1").translate(table)

    s1, s2 = [], []
    for _ in range(size):
        s = chars(rng.randint(*lengths))
        if edits is None:
            t = chars(len(s))
        elif isinstance(edits, int):
            t = _mutate(s, alphabet, edits, seed=rng.getrandbits(32), ops=1)
        else:
            k = max(1, round(len(s) * edits * rng.uniform(0.75, 1.25)))
            t = _mutate(s, alphabet, k, seed=rng.getrandbits(32))
        s1.append(s)
        s2.append(t + chars(rng.randint(*extra)))
    return s1, s2


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


def _varied(
    name: str, alphabet: str, lengths: tuple[int, int], edits: float | int | None, extra: tuple[int, int], size: int
) -> object:
    """
    Build a varied case: its pool, cycled to the batch size.

    Returns:
        object: pytest param of (s1 list, s2 list, n).

    """
    s1, s2 = _pool(name, alphabet, lengths, edits, extra, size)
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
_VARIED = [(_varied(name, *spec), ratio) for name, ratio, *spec in _VARIED_SPECS]
_DISTANCE_CASES = [*_CASES, *(p for p, _ in _VARIED)]
_RATIO_CASES = [*_CASES, *(p for p, ratio in _VARIED if ratio)]


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
