#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
r"""
Diff two scripts/benchmark_by_length.py --save JSON payloads and print a go/no-go verdict.

Compares the `lev [ours]` series only (rapidfuzz/polyleven are reported as a
noise-floor diagnostic: they're identical binaries across both runs, so a
large drift there means the machine wasn't quiet and the run should be
redone, independent of what lev's own delta shows).

Per AGENTS.md's performance-workflow rule: PASS requires a median
improvement >= --threshold on every kind at the given length ceiling;
anything else is REVERT.

Usage:
    uv run scripts/compare_lev_builds.py \\
        --baseline baseline.json --candidate pgo.json \\
        --threshold 3.0 --max-length 64
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LEV_KEY = "lev [ours]"
NOISE_KEYS = ("rapidfuzz", "polyleven")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _pct_delta(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return (candidate - baseline) / baseline * 100.0


def compare(baseline: dict, candidate: dict, max_length: float | None) -> tuple[list[dict], dict[str, float]]:
    """
    Compute per-(kind, length) % deltas for lev, plus a noise-floor summary.

    Returns:
        A tuple of (rows, noise), where `rows` is a list of per-(kind,
        length) delta dicts for the `lev [ours]` series, and `noise` maps
        each library in NOISE_KEYS to its mean absolute % delta between the
        two runs (a proxy for measurement noise on the same binary).

    """
    lengths = baseline["meta"]["lengths"]
    rows = []
    noise_samples: dict[str, list[float]] = {k: [] for k in NOISE_KEYS}

    for kind in baseline["meta"]["kinds"]:
        base_lev = baseline["results"][kind][LEV_KEY]["median_us"]
        cand_lev = candidate["results"][kind][LEV_KEY]["median_us"]
        for i, length in enumerate(lengths):
            if max_length is not None and length > max_length:
                continue
            rows.append({
                "kind": kind,
                "length": length,
                "baseline_us": base_lev[i],
                "candidate_us": cand_lev[i],
                "delta_pct": _pct_delta(base_lev[i], cand_lev[i]),
            })
        for lib in NOISE_KEYS:
            base_lib = baseline["results"][kind][lib]["median_us"]
            cand_lib = candidate["results"][kind][lib]["median_us"]
            for i, length in enumerate(lengths):
                if max_length is not None and length > max_length:
                    continue
                noise_samples[lib].append(abs(_pct_delta(base_lib[i], cand_lib[i])))

    noise = {lib: statistics.mean(vals) if vals else 0.0 for lib, vals in noise_samples.items()}
    return rows, noise


def verdict(rows: list[dict], threshold: float) -> tuple[bool, dict[str, float]]:
    """
    Decide PASS/REVERT: every kind's median delta must be <= -threshold%.

    Returns:
        A tuple of (passed, worst_delta_per_kind), where a negative delta
        means the candidate is faster than the baseline.

    """
    worst: dict[str, float] = {}
    for row in rows:
        kind = row["kind"]
        worst[kind] = max(worst.get(kind, float("-inf")), row["delta_pct"])
    passed = bool(worst) and all(delta <= -threshold for delta in worst.values())
    return passed, worst


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed namespace with `baseline`, `candidate`, `threshold`, and `max_length`.

    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True, help="JSON from the non-PGO build.")
    p.add_argument("--candidate", type=Path, required=True, help="JSON from the PGO build.")
    p.add_argument("--threshold", type=float, default=3.0, help="Required %% improvement to PASS (default: 3.0).")
    p.add_argument(
        "--max-length",
        type=float,
        default=64,
        help="Only consider lengths <= this value in the verdict (default: 64, the realistic ceiling).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Print the delta table, noise-floor diagnostic, and PASS/REVERT verdict.

    Returns:
        Process exit code: 0 on PASS, 1 on REVERT.

    """
    args = parse_args(argv)
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)

    rows, noise = compare(baseline, candidate, args.max_length)

    print(f"{'kind':10s} {'len':>6s} {'baseline (us)':>14s} {'candidate (us)':>15s} {'delta':>8s}")
    for row in rows:
        print(
            f"{row['kind']:10s} {row['length']:6.0f} {row['baseline_us']:14.3f} "
            f"{row['candidate_us']:15.3f} {row['delta_pct']:7.2f}%"
        )

    print("\nnoise floor (identical binaries across runs, mean |delta|):")
    for lib, pct in noise.items():
        print(f"  {lib:12s} {pct:6.2f}%")
        if pct > 2.0:
            print(f"    warning: {lib} drifted >2% between runs on an unchanged binary; consider a quieter rerun")

    passed, worst = verdict(rows, args.threshold)
    print(f"\nworst (least-improved) delta per kind, at lengths <= {args.max_length:.0f}:")
    for kind, delta in worst.items():
        print(f"  {kind:10s} {delta:7.2f}%")

    print(f"\nverdict: {'PASS' if passed else 'REVERT'} (threshold: -{args.threshold:.1f}% on every kind)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
