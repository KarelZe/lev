"""
A/B wall-clock benchmark of two builds of lev on the same machine.

Each round starts one subprocess per build, in alternating order, which
times every case of `tests/test_bench.py` (best of `--repeat` batches).
Alternating builds on one machine cancels most of the drift of shared CI
runners; the report gives the median ns per call over rounds, its IQR, and
the B/A ratio of medians.

Usage:
    python scripts/ab_benchmark.py --a <dir> --b <dir> [--rounds 7] [--summary out.md]

`<dir>` holds an installed `lev` package, e.g. from
`pip install --no-deps --target <dir> <wheel>`.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import timeit
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ratios outside this band are flagged; AGENTS.md treats < 3% as noise.
_NOISE = 0.03


def _child(build_dir: str, repeat: int) -> dict[str, float]:
    """
    Time every benchmark case with the lev found in `build_dir`.

    Returns:
        dict[str, float]: best ns per call, keyed by "function:case".

    """
    sys.path.insert(0, build_dir)
    sys.path.insert(1, str(REPO_ROOT / "tests"))
    import test_bench

    import lev

    assert Path(lev.__file__).resolve().is_relative_to(Path(build_dir).resolve()), lev.__file__
    out = {}
    for fn in (lev.distance, lev.ratio):
        for case in test_bench._CASES:
            s1, s2, n = case.values
            best = min(timeit.repeat(lambda: test_bench._run_batch(fn, s1, s2, n), number=1, repeat=repeat))
            out[f"{fn.__name__}:{case.id}"] = best / n * 1e9
    return out


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    """
    Return the quartiles of xs (all equal to the value for a single sample).

    Returns:
        tuple[float, float, float]: first quartile, median, third quartile.

    """
    if len(xs) < 2:
        return xs[0], xs[0], xs[0]
    q1, q2, q3 = statistics.quantiles(xs, n=4)
    return q1, q2, q3


def main() -> int:
    """
    Run the A/B comparison and print a markdown report.

    Returns:
        int: process exit code.

    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="directory with the baseline lev build")
    parser.add_argument("--b", required=True, help="directory with the candidate lev build")
    parser.add_argument("--rounds", type=int, default=7, help="subprocess pairs (default: 7)")
    parser.add_argument("--repeat", type=int, default=15, help="batches per case and process (default: 15)")
    parser.add_argument("--summary", type=Path, help="also append the report to this file")
    parser.add_argument("--json", type=Path, help="write raw per-round timings to this file")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(_child(args.child, args.repeat)))
        return 0

    runs: dict[str, list[dict[str, float]]] = {"a": [], "b": []}
    for r in range(args.rounds):
        order = ("a", "b") if r % 2 == 0 else ("b", "a")
        for side in order:
            cmd = [sys.executable, __file__, "--a", args.a, "--b", args.b, "--repeat", str(args.repeat)]
            out = subprocess.run([*cmd, "--child", getattr(args, side)], capture_output=True, text=True, check=True)
            runs[side].append(json.loads(out.stdout))

    lines = [
        f"### A/B benchmark: {platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "",
        f"{args.rounds} alternating rounds, best of {args.repeat} batches each; median ns per call (IQR).",
        f"Flagged: B/A outside 1 ± {_NOISE:.0%}.",
        "",
        "| case | A | B | B/A | |",
        "|---|---:|---:|---:|---|",
    ]
    ratios = []
    for key in runs["a"][0]:
        qa = _quartiles([d[key] for d in runs["a"]])
        qb = _quartiles([d[key] for d in runs["b"]])
        ratio = qb[1] / qa[1]
        ratios.append(ratio)
        flag = "faster" if ratio < 1 - _NOISE else ("**slower**" if ratio > 1 + _NOISE else "")
        lines.append(
            f"| `{key}` | {qa[1]:.1f} ({qa[2] - qa[0]:.1f}) | {qb[1]:.1f} ({qb[2] - qb[0]:.1f}) | {ratio:.3f} | {flag} |"
        )
    geo = statistics.geometric_mean(ratios)
    lines[3:3] = [f"Geometric mean B/A over {len(ratios)} cases: **{geo:.3f}**.", ""]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(report)
    if args.json:
        args.json.write_text(json.dumps(runs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
