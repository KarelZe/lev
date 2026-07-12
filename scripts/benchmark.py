#!/usr/bin/env python3
"""
Benchmark lev against rapidfuzz, editdistance, and edlib.

Each pair is 100 characters long. Each measurement is the total wall time
of `--repetitions` calls, measured with `timeit`.

Examples:
    # default: all kinds, write to docs/assets/benchmark_results.json
    uv run python scripts/benchmark.py

    # single kind, custom output (used by the performance workflow)
    uv run python scripts/benchmark.py --kind ascii --save baselines/pre.json

"""

from __future__ import annotations

import argparse
import json
import sys
import timeit
from collections.abc import Callable
from pathlib import Path

import editdistance
import edlib
import rapidfuzz.distance.Levenshtein as rf_lev  # noqa: N813

import lev

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "benchmark_results.json"

STRING_LEN = 100
REPETITIONS_DEFAULT = 1_000

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
    ("editdistance", editdistance.eval),
    ("edlib", lambda a, b: edlib.align(a, b, task="distance")["editDistance"]),
]


def _measure(fn: Callable[[str, str], int], a: str, b: str, reps: int) -> float:
    """
    Time `reps` calls to fn(a, b).

    Returns:
        Total wall time in seconds.

    """
    return timeit.timeit(lambda: fn(a, b), number=reps)


def run(kinds: list[str], reps: int) -> dict:
    """
    Benchmark every contender on each string kind.

    Sanity-checks that all libraries agree with lev on the distance
    before timing them.

    Returns:
        Payload dict with a `meta` section and per-kind, per-library
        total wall times in seconds.

    """
    results: dict[str, dict[str, float]] = {}
    for kind in kinds:
        a, b = KINDS[kind]
        assert len(a) == STRING_LEN and len(b) == STRING_LEN, f"{kind} pair not exactly {STRING_LEN} code points"
        per_lib: dict[str, float] = {}
        for name, fn in CONTENDERS:
            # sanity check: everyone should agree on the distance
            expected = lev.distance(a, b)
            got = fn(a, b)
            if got != expected:
                print(
                    f"warning: {name} disagrees on {kind}: {got} != {expected}",
                    file=sys.stderr,
                )
            per_lib[name] = _measure(fn, a, b, reps)
        results[kind] = per_lib
    return {
        "meta": {
            "string_len": STRING_LEN,
            "repetitions": reps,
            "kinds": kinds,
        },
        "results": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed namespace with `kind`, `save`, and `repetitions`.

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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Run the benchmark, write JSON results, and print a summary.

    Returns:
        Process exit code (0 on success).

    """
    args = parse_args(argv)
    kinds = list(KINDS.keys()) if args.kind == "all" else [args.kind]

    payload = run(kinds, args.repetitions)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {args.save}")

    # Human-readable summary
    for kind, per_lib in payload["results"].items():
        print(f"\n{kind}:")
        for name, secs in sorted(per_lib.items(), key=lambda kv: kv[1]):
            print(f"  {name:14s} {secs * 1000:8.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
