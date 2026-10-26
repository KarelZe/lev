#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.10.9",
#   "matplotx>=0.3.10",
#   "numpy>=2.2.6",
# ]
# ///
"""
Benchmark lev across released versions up to the current unreleased build.

Measures average runtime (and spread) on realistic natural-language text.
Produces GitHub-themed light and dark SVG and PNG line plots using dufte and matplotx.

Examples:
    # Run default benchmark across all versions and generate line plots:
    uv run scripts/benchmark_by_version.py

    # Select specific versions via CLI:
    uv run scripts/benchmark_by_version.py --versions 0.4.0 0.4.1 0.5.0 unreleased

    # Plot only the average metric:
    uv run scripts/benchmark_by_version.py --metrics average

    # Save JSON results to a custom file:
    uv run scripts/benchmark_by_version.py --save baselines/versions.json

"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "benchmark_by_version.json"
ASSETS_DIR = DEFAULT_OUTPUT.parent

# Default tuple of versions to benchmark: all released versions on PyPI
# plus the current unreleased development version in the local repository.
DEFAULT_VERSIONS: tuple[str, ...] = (
    "0.1.0",
    "0.2.0",
    "0.2.1",
    "0.2.2",
    "0.2.3",
    "0.2.4",
    "0.2.5",
    "0.2.6",
    "0.3.0",
    "0.4.0",
    "0.4.1",
    "0.5.0",
    "unreleased",
)

# Natural-length, real-world prose with realistic typos (identical to benchmark.py).
REALISTIC_TEXT: tuple[str, str] = (
    "Customer support was fantastic today. The technician arrived on time, "
    "diagnosed the issue within minutes, and had everything working again "
    "before lunch. I would definitely recommend this service to a friend "
    "or colleague.",
    "Customer suport was fantastic today. The technician arived on time, "
    "diagnosed the issue within mintues, and had everything working again "
    "before lunch. I would definately recommend this service to a freind "
    "or colleague.",
)

REPETITIONS_DEFAULT = 5_000
REPEAT_DEFAULT = 10
WARMUP_DEFAULT = 1_000

UNRELEASED_ALIASES = frozenset({"unreleased", "current", "dev", "head", "local", "."})

CHILD_RUNNER_CODE = """
import json
import sys
import timeit
import lev

s1 = sys.argv[1]
s2 = sys.argv[2]
repetitions = int(sys.argv[3])
repeat = int(sys.argv[4])
warmup = int(sys.argv[5])

# Sanity check
dist = lev.distance(s1, s2)

# Warmup to prime CPU caches and branch predictors
if warmup > 0:
    timeit.repeat(lambda: lev.distance(s1, s2), number=warmup, repeat=3)

# Measurement
raw_times = timeit.repeat(lambda: lev.distance(s1, s2), number=repetitions, repeat=repeat)
times_us = [(t / repetitions) * 1e6 for t in raw_times]

print(json.dumps({
    "dist": dist,
    "times_us": times_us,
    "version": getattr(lev, "__version__", None),
}))
"""


def _cpu_model() -> str:
    """
    Best-effort CPU model name, e.g. `Apple M1 Pro` or `AMD EPYC 7763`.

    Returns:
        The CPU model, or `platform.processor()` if it cannot be read.

    """
    try:
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def machine_info() -> dict[str, str]:
    """
    Describe the machine and interpreter the benchmark runs on.

    Returns:
        CPU model, OS, architecture, and Python version.

    """
    return {
        "cpu": _cpu_model(),
        "os": platform.system(),
        "arch": platform.machine(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
    }


def is_unreleased_version(ver: str) -> bool:
    """
    Check if a version string refers to the local unreleased build.

    Args:
        ver: Version string to check.

    Returns:
        True if the version refers to the unreleased/local build.

    """
    return ver.lower() in UNRELEASED_ALIASES


def measure_version(
    version_label: str,
    s1: str,
    s2: str,
    repetitions: int,
    repeat: int,
    warmup: int,
    uv_bin: str = "uv",
) -> dict:
    """
    Run benchmark for a single version in an isolated Python subprocess via uv.

    Args:
        version_label: Released version string (e.g. '0.5.0') or 'unreleased'.
        s1: First input string.
        s2: Second input string.
        repetitions: Calls per batch.
        repeat: Number of batches.
        warmup: Number of warmup calls.
        uv_bin: Path to the uv executable.

    Returns:
        Dictionary with statistics (mean_us, min_us, etc.) and distance check.

    Raises:
        RuntimeError: If the uv runner fails or exits with a non-zero exit code.
        ValueError: If no output is received from the runner subprocess.

    """
    if is_unreleased_version(version_label):
        pkg_spec = str(REPO_ROOT)
    elif Path(version_label).exists():
        pkg_spec = str(Path(version_label).resolve())
    else:
        pkg_spec = f"lev-rs=={version_label}"

    cmd = [
        uv_bin,
        "run",
        "--no-config",
        "--isolated",
        "--no-project",
        f"--with={pkg_spec}",
        "python",
        "-c",
        CHILD_RUNNER_CODE,
        s1,
        s2,
        str(repetitions),
        str(repeat),
        str(warmup),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Benchmark failed for version '{version_label}': {err_msg}")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No output received from benchmark runner for version '{version_label}'")

    data = json.loads(lines[-1])
    times_us = np.asarray(data["times_us"], dtype=float)

    return {
        "label": version_label,
        "is_unreleased": is_unreleased_version(version_label),
        "mean_us": float(np.mean(times_us)),
        "median_us": float(np.median(times_us)),
        "std_us": float(np.std(times_us)),
        "min_us": float(np.min(times_us)),
        "max_us": float(np.max(times_us)),
        "p25_us": float(np.percentile(times_us, 25)),
        "p75_us": float(np.percentile(times_us, 75)),
        "times_us": times_us.tolist(),
        "distance": data["dist"],
        "reported_version": data.get("version"),
    }


def run_benchmarks(
    versions: Sequence[str] = DEFAULT_VERSIONS,
    s1: str = REALISTIC_TEXT[0],
    s2: str = REALISTIC_TEXT[1],
    repetitions: int = REPETITIONS_DEFAULT,
    repeat: int = REPEAT_DEFAULT,
    warmup: int = WARMUP_DEFAULT,
    uv_bin: str | None = None,
) -> dict:
    """
    Benchmark each specified version of lev and return comprehensive metrics.

    Args:
        versions: Sequence of version strings to benchmark.
        s1: First string.
        s2: Second string.
        repetitions: Number of calls per measurement run.
        repeat: Number of measurement repetitions.
        warmup: Number of warmup calls before timing.
        uv_bin: Path to the uv binary (detected automatically if None).

    Returns:
        Payload dictionary with metadata and per-version results.

    """
    resolved_uv = uv_bin or shutil.which("uv") or "uv"
    results: dict[str, dict] = {}

    print(f"Benchmarking {len(versions)} version(s) on realistic text...")
    for ver in versions:
        sys.stdout.write(f"  • {ver:14s} ... ")
        sys.stdout.flush()
        try:
            res = measure_version(ver, s1, s2, repetitions, repeat, warmup, uv_bin=resolved_uv)
            results[ver] = res
            print(f"done  [avg: {res['mean_us']:.3f} μs, min: {res['min_us']:.3f} μs, dist: {res['distance']}]")
        except Exception as e:
            print(f"FAILED ({e})")
            raise

    return {
        "meta": {
            "repetitions": repetitions,
            "repeat": repeat,
            "warmup": warmup,
            "string_lengths": [len(s1), len(s2)],
            "versions": list(versions),
            "machine": machine_info(),
        },
        "results": results,
    }


def plot_versions(
    payload: dict,
    output_dir: Path = ASSETS_DIR,
    chart_type: str = "line",
    metrics: Sequence[str] = ("average", "min"),
    formats: tuple[str, ...] = ("svg", "png"),
) -> list[Path]:
    """
    Render and save light and dark benchmark plots for all tested versions.

    Args:
        payload: Benchmark results payload.
        output_dir: Directory where plot images should be saved.
        chart_type: 'line' for line plot with dufte styling, 'bar' for bar chart.
        metrics: Which metric series to plot (e.g. 'average', 'min').
        formats: File formats to export ('svg', 'png', etc.).

    Returns:
        List of generated plot paths.

    """
    results: dict[str, dict] = payload["results"]
    meta: dict = payload["meta"]
    labels_versions = list(results.keys())
    x0 = np.arange(len(labels_versions))

    # Assemble series to plot based on metrics
    series: list[tuple[str, np.ndarray]] = []
    if "average" in metrics:
        means = np.array([results[ver]["mean_us"] for ver in labels_versions])
        series.append(("average", means))
    if "min" in metrics:
        mins = np.array([results[ver]["min_us"] for ver in labels_versions])
        series.append(("min", mins))
    if "median" in metrics:
        medians = np.array([results[ver]["median_us"] for ver in labels_versions])
        series.append(("median", medians))

    if not series:
        means = np.array([results[ver]["mean_us"] for ver in labels_versions])
        series.append(("average", means))

    y = [s[1] for s in series]
    labels = [s[0] for s in series]

    machine = meta["machine"]
    subtitle = (
        f"{machine['cpu']}, {machine['os']} {machine['arch']}, {machine['python']}\n"
        f"Realistic text ({meta['string_lengths'][0]} chars), n={meta['repetitions']:,} calls × {meta['repeat']} runs"  # ruff: ignore[ambiguous-unicode-character-string]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for theme in ("light", "dark"):
        style = matplotx.styles.dufte if theme == "light" else matplotx.styles.duftify(matplotx.styles.github["dark"])

        with plt.style.context(style):
            plt.figure(figsize=(10.5, 5.0))

            if chart_type == "line":
                for yy, label in zip(y, labels):
                    plt.plot(x0, yy, label=label, marker="o", markersize=4)

                _, tick_labels = plt.xticks(x0, labels_versions, rotation=30, ha="right")
                for tick in tick_labels:
                    if is_unreleased_version(tick.get_text()):
                        tick.set_fontweight("bold")

                plt.xlabel(f"version\n{subtitle}")
                matplotx.ylabel_top("time [µs]")  # ruff: ignore[ambiguous-unicode-character-string]
                matplotx.line_labels()
            else:
                means = np.array([results[ver]["mean_us"] for ver in labels_versions])
                released_color = "#58a6ff" if theme == "dark" else "#0969da"
                unreleased_color = "#3fb950" if theme == "dark" else "#1a7f37"
                grid_color = "#30363d" if theme == "dark" else "#d0d7de"
                text_color = "white" if theme == "dark" else "black"

                colors = [
                    unreleased_color if results[ver]["is_unreleased"] else released_color for ver in labels_versions
                ]
                bars = plt.bar(x0, means, color=colors, width=0.62, edgecolor="none", zorder=3)
                plt.bar_label(bars, fmt="%.2f μs", padding=4, fontsize=8.5, color=text_color)
                plt.ylim(0, max(means) * 1.20)
                plt.grid(True, axis="y", ls="--", alpha=0.5, color=grid_color, zorder=0)
                plt.grid(False, axis="x")

                _, tick_labels = plt.xticks(x0, labels_versions, rotation=30, ha="right")
                for tick in tick_labels:
                    if is_unreleased_version(tick.get_text()):
                        tick.set_fontweight("bold")
                        tick.set_color(unreleased_color)

                plt.title("Levenshtein Distance Performance Across lev Versions", fontsize=13, pad=12)
                plt.xlabel(f"version\n{subtitle}", fontsize=9.5, labelpad=8)
                matplotx.ylabel_top("time [µs]")  # ruff: ignore[ambiguous-unicode-character-string]

            for fmt in formats:
                out_path = output_dir / f"benchmark_by_version_{theme}.{fmt}"
                plt.savefig(out_path, bbox_inches="tight")
                generated_files.append(out_path)

            plt.close()

    return generated_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.

    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--versions",
        nargs="*",
        default=None,
        help=(
            "Versions of lev to benchmark (space- or comma-separated, e.g. "
            "'0.4.0 0.4.1 0.5.0 unreleased'). Defaults to DEFAULT_VERSIONS."
        ),
    )
    p.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to save JSON benchmark results (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    p.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS_DEFAULT,
        help=f"Number of distance calls per measurement batch (default: {REPETITIONS_DEFAULT:,}).",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=REPEAT_DEFAULT,
        help=f"Number of independent measurement batches (default: {REPEAT_DEFAULT}).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_DEFAULT,
        help=f"Number of warmup calls prior to measurement (default: {WARMUP_DEFAULT:,}).",
    )
    p.add_argument(
        "--chart-type",
        choices=["line", "bar"],
        default="line",
        help="Visualization chart type (default: line).",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        choices=["average", "min", "median"],
        default=["average", "min"],
        help="Metrics to plot as lines (default: average min).",
    )
    p.add_argument(
        "--format",
        choices=["svg", "png", "both"],
        default="both",
        help="Plot image format to produce (default: both).",
    )
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate and save plot images (default: True).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help=f"Directory for plot output files (default: {ASSETS_DIR.relative_to(REPO_ROOT)}).",
    )
    p.add_argument(
        "--uv",
        dest="uv_bin",
        type=str,
        default=None,
        help="Custom path to the uv binary (detected automatically if omitted).",
    )
    return p.parse_args(argv)


def resolve_versions(arg_versions: list[str] | None) -> tuple[str, ...]:
    """
    Resolve version arguments into a normalized tuple of version strings.

    Args:
        arg_versions: Raw CLI arguments list, or None.

    Returns:
        Tuple of version strings.

    """
    if not arg_versions:
        return DEFAULT_VERSIONS

    resolved: list[str] = []
    for item in arg_versions:
        # Handle comma-separated list e.g. "0.4.0,0.5.0,unreleased"
        parts = [p.strip() for p in item.split(",") if p.strip()]
        resolved.extend(parts)

    return tuple(resolved) if resolved else DEFAULT_VERSIONS


def main(argv: list[str] | None = None) -> int:
    """
    Execute benchmark across versions and generate summary and plots.

    Returns:
        0 on success.

    """
    args = parse_args(argv)
    versions_tuple = resolve_versions(args.versions)

    payload = run_benchmarks(
        versions=versions_tuple,
        repetitions=args.repetitions,
        repeat=args.repeat,
        warmup=args.warmup,
        uv_bin=args.uv_bin,
    )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote benchmark results JSON to {args.save}")

    if args.plot:
        fmts = ("svg", "png") if args.format == "both" else (args.format,)
        generated = plot_versions(
            payload,
            output_dir=args.output_dir,
            chart_type=args.chart_type,
            metrics=args.metrics,
            formats=fmts,
        )
        print(f"Generated {len(generated)} plot files in {args.output_dir}:")
        for p in generated:
            print(f"  • {p.name}")

    # Summary table
    print("\nBenchmark Summary (Realistic Text):")
    print(f"{'Version':<14} {'Average [μs]':>14} {'Median [μs]':>14} {'Min [µs]':>14} {'Std [μs]':>12}")  # ruff: ignore[ambiguous-unicode-character-string]
    print("-" * 72)
    for ver, data in payload["results"].items():
        label = f"{ver} *" if data["is_unreleased"] else ver
        print(
            f"{label:<14} {data['mean_us']:>14.3f} {data['median_us']:>14.3f} "
            f"{data['min_us']:>14.3f} {data['std_us']:>12.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
