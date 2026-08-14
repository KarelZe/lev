#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "editdistance==0.8.1",
#     "edlib==1.3.9.post1",
#     "polyleven==0.11.0",
#     "rapidfuzz==3.14.5",
#     "lev-rs",
# ]
#
# [tool.uv.sources]
# lev-rs = { path = "../", editable = true }
# ///
"""
Benchmark lev against rapidfuzz, polyleven, editdistance, and edlib.

Each pair is 100 characters long. Each measurement runs `--repetitions`
calls `--timeit-repeat` times with `timeit.repeat` and reports the
minimum total wall time (the most noise-resistant estimate: background
load can only slow a run down, never speed it up).

Examples:
    # default: all kinds, write to docs/assets/benchmark_results.json
    uv run scripts/benchmark.py

    # single kind, custom output (used by the performance workflow)
    uv run scripts/benchmark.py --kind ascii --save baselines/pre.json

"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import timeit
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import editdistance
import edlib
import polyleven
import rapidfuzz.distance.Levenshtein as rf_lev  # noqa: N813

import lev

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "benchmark_results.json"

STRING_LEN = 100
REPETITIONS_DEFAULT = 1_000
TIMEIT_REPEAT_DEFAULT = 5

# One representative pair per CPython string kind. Length is exactly
# STRING_LEN code points in each case, and the pair differs by a handful
# of edits so we exercise the DP rather than an early-exit path.
KINDS: dict[str, tuple[str, str]] = {
    # ASCII kind: max codepoint < 128
    "ascii": (
        ("The quick brown fox jumps over the lazy dog. " * 3)[:STRING_LEN],
        ("The quick brown cat jumps over the lazy dog! " * 3)[:STRING_LEN],
    ),
    # Latin-1 / UCS-1 kind: max codepoint < 256, non-ASCII present
    "latin1": (
        ("café résumé naïve façade jalapeño Zürich smörgåsbord " * 3)[:STRING_LEN],
        ("cafe  resume  naive facade jalapeno Zurich  smorgasbord " * 3)[:STRING_LEN],
    ),
    # UCS-2 kind: BMP, max codepoint < 65536 (CJK ideographs live here)
    "cjk": (
        ("日本語のテスト文字列を長くするために繰り返します。" * 5)[:STRING_LEN],
        ("日本語のテスト文字列を短くするために繰り返します。" * 5)[:STRING_LEN],
    ),
    # UCS-4 kind: astral plane, max codepoint >= 65536 (most emoji)
    "emoji": (
        ("😀🎉🚀✨🐍🦀📦🔥💡🌟" * 20)[:STRING_LEN],
        ("😀🎉🚀✨🐍🦀📦🔥💡⭐" * 20)[:STRING_LEN],
    ),
}

Contender = tuple[str, Callable[[str, str], int]]

CONTENDERS: list[Contender] = [
    ("lev", lev.distance),
    ("rapidfuzz", rf_lev.distance),
    ("polyleven", polyleven.levenshtein),
    ("editdistance", editdistance.eval),
    ("edlib", lambda a, b: edlib.align(a, b, task="distance")["editDistance"]),
]

# Distribution names used to look up installed versions for the meta block.
_DISTRIBUTIONS = (
    "lev",
    "rapidfuzz",
    "polyleven",
    "editdistance",
    "edlib",
    "levenshtein",
)


def _measure(fn: Callable[[str, str], int], a: str, b: str, reps: int, repeat: int) -> float:
    """
    Time `reps` calls to fn(a, b), `repeat` times over.

    The statement is a plain string evaluated against `globals`, so the
    timed loop pays only for the fn(a, b) call itself — no extra
    lambda/closure indirection that would bias results against fast C
    implementations.

    Returns:
        Minimum total wall time in seconds across the `repeat` runs.

    """
    timer = timeit.Timer("fn(a, b)", globals={"fn": fn, "a": a, "b": b})
    return min(timer.repeat(repeat=repeat, number=reps))


def _library_versions() -> dict[str, str]:
    """
    Look up installed versions of every benchmarked distribution.

    Returns:
        Mapping of distribution name to version string, or "unknown"
        when the distribution metadata is unavailable (e.g. a plain
        local module that was never pip-installed).

    """
    versions: dict[str, str] = {}
    for dist in _DISTRIBUTIONS:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = "unknown"
    return versions


def run(kinds: list[str], reps: int, repeat: int) -> dict:
    """
    Benchmark every contender on each string kind.

    Sanity-checks that all libraries agree with lev on the distance
    before timing them.

    Returns:
        Payload dict with a `meta` section (including environment and
        library versions) and per-kind, per-library minimum total wall
        times in seconds.

    """
    results: dict[str, dict[str, float]] = {}
    for kind in kinds:
        a, b = KINDS[kind]
        assert len(a) == STRING_LEN and len(b) == STRING_LEN, f"{kind} pair not exactly {STRING_LEN} code points"
        expected = lev.distance(a, b)
        per_lib: dict[str, float] = {}
        for name, fn in CONTENDERS:
            # sanity check: everyone should agree on the distance
            got = fn(a, b)
            if got != expected:
                print(
                    f"warning: {name} disagrees on {kind}: {got} != {expected}",
                    file=sys.stderr,
                )
            per_lib[name] = _measure(fn, a, b, reps, repeat)
        results[kind] = per_lib
    return {
        "meta": {
            "string_len": STRING_LEN,
            "repetitions": reps,
            "timeit_repeat": repeat,
            "measurement": "min total seconds across timeit.repeat runs",
            "kinds": kinds,
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "library_versions": _library_versions(),
        },
        "results": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed namespace with `kind`, `save`, `repetitions`, and
        `timeit_repeat`.

    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--kind",
        choices=[*KINDS.keys(), "all"],
        default="all",
        help="Which CPython string kind to benchmark (default: all).",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON results (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    p.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS_DEFAULT,
        help=f"timeit repetitions per measurement (default: {REPETITIONS_DEFAULT}).",
    )
    p.add_argument(
        "--timeit-repeat",
        type=int,
        default=TIMEIT_REPEAT_DEFAULT,
        help=(
            f"How many times to repeat each measurement; the minimum is reported (default: {TIMEIT_REPEAT_DEFAULT})."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Run the benchmark, write JSON results, and print a summary.

    Returns:
        Process exit code (0 on success).

    """
    args = parse_args(argv)
    kinds = list(KINDS.keys()) if args.kind == "all" else [args.kind]

    payload = run(kinds, args.repetitions, args.timeit_repeat)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {args.save}")

    # Human-readable summary
    reps = args.repetitions
    for kind, per_lib in payload["results"].items():
        print(f"\n{kind}:")
        for name, secs in sorted(per_lib.items(), key=lambda kv: kv[1]):
            per_call_us = secs / reps * 1e6
            print(f"  {name:14s} {secs * 1000:8.2f} ms total  {per_call_us:10.2f} µs/call")  # noqa: RUF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
