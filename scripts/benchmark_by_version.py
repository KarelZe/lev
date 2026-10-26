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
Produces GitHub-themed light and dark SVG and PNG line plots with filled uncertainty bands.

Examples:
    # Run default benchmark across all versions and generate line plots:
    uv run scripts/benchmark_by_version.py

    # Select specific versions via CLI:
    uv run scripts/benchmark_by_version.py --versions 0.4.0 0.4.1 0.5.0 main

    # Save JSON results to a custom file:
    uv run scripts/benchmark_by_version.py --save baselines/versions.json

"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    "main",
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

UNRELEASED_ALIASES = frozenset({"main", "unreleased", "current", "dev", "head", "local", "."})
SEMVER_REGEX = re.compile(r"^\d+(\.\d+)*$")

CHILD_RUNNER_CODE = """
import json
import sys
import timeit
import lev

payload = json.loads(sys.stdin.read())
s1 = payload["s1"]
s2 = payload["s2"]
repetitions = int(payload["repetitions"])
repeat = int(payload["repeat"])
warmup = int(payload["warmup"])

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


def _validate_safe_path(path: Path) -> Path:
    """
    Ensure the path is safely bounded within the repository to prevent path traversal.

    Args:
        path: Path to validate.

    Returns:
        Canonicalized Path safely inside the repository.

    Raises:
        ValueError: If path resolves outside the repository root.

    """
    resolved = path.resolve()
    base = REPO_ROOT.resolve()
    if not resolved.is_relative_to(base) or os.path.commonpath([str(base), str(resolved)]) != str(base):
        raise ValueError(f"Path traversal detected: {path} must be inside {base}")
    return resolved


def _safe_write_json(destination: Path, data: dict) -> None:
    """
    Write JSON safely to destination inside the repository.

    Args:
        destination: Path to write to.
        data: Data dictionary to serialize.

    """
    safe_path = _validate_safe_path(destination)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def _validate_version_label(label: str) -> str:
    """
    Validate that the version label is safe and matches expected semver format.

    Args:
        label: Raw version string.

    Returns:
        Validated version label.

    Raises:
        ValueError: If the version label does not match allowed characters.

    """
    cleaned = label.strip()
    if is_unreleased_version(cleaned):
        return cleaned
    if not SEMVER_REGEX.match(cleaned):
        raise ValueError(f"Invalid version format '{cleaned}'. Expected semver like '0.5.0' or 'unreleased'.")
    return cleaned


def measure_version(
    version_label: str,
    s1: str,
    s2: str,
    repetitions: int,
    repeat: int,
    warmup: int,
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

    Returns:
        Dictionary with statistics (mean_us, min_us, etc.) and distance check.

    Raises:
        RuntimeError: If the uv runner fails or exits with a non-zero exit code.
        ValueError: If no output is received from the runner subprocess.

    """
    validated_label = _validate_version_label(version_label)
    pkg_spec = str(REPO_ROOT) if is_unreleased_version(validated_label) else f"lev-rs=={validated_label}"

    uv_bin = shutil.which("uv") or "uv"
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
    ]

    input_data = json.dumps({
        "s1": s1,
        "s2": s2,
        "repetitions": int(repetitions),
        "repeat": int(repeat),
        "warmup": int(warmup),
    })

    result = subprocess.run(cmd, input=input_data, capture_output=True, text=True)
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

    Returns:
        Payload dictionary with metadata and per-version results.

    """
    results: dict[str, dict] = {}

    print(f"Benchmarking {len(versions)} version(s) on realistic text...")
    for ver in versions:
        sys.stdout.write(f"  • {ver:14s} ... ")
        sys.stdout.flush()
        try:
            res = measure_version(ver, s1, s2, repetitions, repeat, warmup)
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


def _render_line_chart(
    x0: np.ndarray,
    means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    version_labels: list[str],
    subtitle: str,
) -> None:
    """
    Render dufte line chart with a filled uncertainty band and end-of-line label.

    Args:
        x0: Array of x-axis indices.
        means: Average runtime values per version.
        lower: Lower bound of the uncertainty band (e.g. min runtime).
        upper: Upper bound of the uncertainty band (e.g. max runtime).
        version_labels: Display labels for versions.
        subtitle: Descriptive metadata subtitle.

    """
    (line,) = plt.plot(x0, means, label="average", marker="o", markersize=4)
    plt.fill_between(x0, lower, upper, alpha=0.2, color=line.get_color())

    _, tick_labels = plt.xticks(x0, version_labels, rotation=90, ha="right", va="center")
    for tick in tick_labels:
        if is_unreleased_version(tick.get_text()):
            tick.set_fontweight("bold")

    plt.xlabel(f"version\n{subtitle}")
    matplotx.ylabel_top("time [µs]")  # ruff: ignore[ambiguous-unicode-character-string]
    matplotx.line_labels()


def _render_bar_chart(
    x0: np.ndarray,
    means: np.ndarray,
    version_labels: list[str],
    results: dict[str, dict],
    theme: str,
    subtitle: str,
) -> None:
    """
    Render horizontal or vertical bar chart visualization.

    Args:
        x0: Array of x-axis indices.
        means: Average runtime values per version.
        version_labels: Display labels for versions.
        results: Raw benchmark results mapping.
        theme: 'light' or 'dark' theme string.
        subtitle: Descriptive metadata subtitle.

    """
    released_color = "#58a6ff" if theme == "dark" else "#0969da"
    unreleased_color = "#3fb950" if theme == "dark" else "#1a7f37"
    grid_color = "#30363d" if theme == "dark" else "#d0d7de"
    text_color = "white" if theme == "dark" else "black"

    colors = [unreleased_color if results[v]["is_unreleased"] else released_color for v in version_labels]
    bars = plt.bar(x0, means, color=colors, width=0.62, edgecolor="none", zorder=3)
    plt.bar_label(bars, fmt="%.2f μs", padding=4, fontsize=8.5, color=text_color)
    plt.ylim(0, max(means) * 1.20)
    plt.grid(True, axis="y", ls="--", alpha=0.5, color=grid_color, zorder=0)
    plt.grid(False, axis="x")

    _, tick_labels = plt.xticks(x0, version_labels, rotation=90, ha="right", va="center")
    for tick in tick_labels:
        if is_unreleased_version(tick.get_text()):
            tick.set_fontweight("bold")
            tick.set_color(unreleased_color)

    plt.title("Levenshtein Distance Performance Across lev Versions", fontsize=13, pad=12)
    plt.xlabel(f"version\n{subtitle}", fontsize=9.5, labelpad=8)
    matplotx.ylabel_top("time [µs]")  # ruff: ignore[ambiguous-unicode-character-string]


def _get_theme_style(theme: str) -> Any:
    """
    Return the matplotlib style context for light or dark theme.

    Args:
        theme: Theme name ('light' or 'dark').

    Returns:
        Matplotlib style object.

    """
    if theme == "light":
        return matplotx.styles.dufte
    return matplotx.styles.duftify(matplotx.styles.github["dark"])


def plot_versions(
    payload: dict,
    output_dir: Path = ASSETS_DIR,
    chart_type: str = "line",
    formats: tuple[str, ...] = ("svg", "png"),
) -> list[Path]:
    """
    Render and save light and dark benchmark plots for all tested versions.

    Args:
        payload: Benchmark results payload.
        output_dir: Directory where plot images should be saved.
        chart_type: 'line' for line plot with filled uncertainty band, 'bar' for bar chart.
        formats: File formats to export ('svg', 'png', etc.).

    Returns:
        List of generated plot paths.

    """
    results: dict[str, dict] = payload["results"]
    meta: dict = payload["meta"]
    labels_versions = list(results.keys())
    version_display_labels = ["main" if is_unreleased_version(v) else v for v in labels_versions]
    x0 = np.arange(len(labels_versions))

    means = np.array([results[v]["mean_us"] for v in labels_versions])
    mins = np.array([results[v]["min_us"] for v in labels_versions])
    maxs = np.array([results[v]["max_us"] for v in labels_versions])

    machine = meta["machine"]
    subtitle = (
        f"{machine['cpu']}, {machine['os']} {machine['arch']}, {machine['python']}\n"
        f"Realistic text ({meta['string_lengths'][0]} chars), n={meta['repetitions']:,} calls × {meta['repeat']} runs"  # ruff: ignore[ambiguous-unicode-character-string]
    )

    safe_dir = _validate_safe_path(output_dir)
    safe_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for theme in ("light", "dark"):
        style = _get_theme_style(theme)
        with plt.style.context(style):
            plt.figure(figsize=(10.5, 5.0))
            if chart_type == "line":
                _render_line_chart(x0, means, mins, maxs, version_display_labels, subtitle)
            else:
                _render_bar_chart(x0, means, version_display_labels, results, theme, subtitle)

            for fmt in formats:
                out_path = safe_dir / f"benchmark_by_version_{theme}.{fmt}"
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
            "'0.4.0 0.4.1 0.5.0 main'). Defaults to DEFAULT_VERSIONS."
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
    )

    _safe_write_json(args.save, payload)
    print(f"\nWrote benchmark results JSON to {args.save}")

    if args.plot:
        fmts = ("svg", "png") if args.format == "both" else (args.format,)
        generated = plot_versions(
            payload,
            chart_type=args.chart_type,
            formats=fmts,
        )
        print(f"Generated {len(generated)} plot files in {ASSETS_DIR}:")
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
