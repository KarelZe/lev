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

Measures average runtime (and spread) across representative architectural regimes:
- Short: small strings (m <= 16) testing tiny SIMD and mbleven fast paths (available via `--regimes short`).
- Medium: realistic natural-language text testing multi-word direct tables + affix stripping.
- Long Similar: 2,048-char near-duplicate documents testing banded Ukkonen Myers.
- Long Dissimilar: 2,048-char unrelated documents testing full-matrix DP.

Produces GitHub-themed light and dark SVG and PNG line plots with filled uncertainty bands
and end-of-line labels on a single unified plot.

Examples:
    # Run default benchmark across curated versions and regimes:
    uv run scripts/benchmark_by_version.py

    # Select specific regimes or versions:
    uv run scripts/benchmark_by_version.py --regimes medium --versions 0.2.6 0.5.0 main

    # Benchmark including short strings:
    uv run scripts/benchmark_by_version.py --regimes all

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

# Default tuple of versions to benchmark: key releases that introduced
# meaningful algorithmic or performance changes, plus the current unreleased
# development version in the local repository.
DEFAULT_VERSIONS: tuple[str, ...] = (
    "0.1.0",
    "0.2.4",
    "0.2.6",
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

# Long near-duplicate prose: 2,048 characters with 4 edits.
LONG_SIMILAR_TEXT: tuple[str, str] = (
    ("The quick brown fox jumps over the lazy dog. " * 45)[:2048],
    ("The quick brown cat jumps over the lazy dog! " * 45)[:2048],
)

# Long dissimilar text: 2,048 characters over disjoint alphabets.
LONG_DISSIMILAR_TEXT: tuple[str, str] = (
    ("abcdefghijklmnopqrstuvwxyz" * 80)[:2048],
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 80)[:2048],
)

REGIMES_CATALOG: dict[str, dict[str, Any]] = {
    "short": {
        "label": "short",
        "s1": "kitten",
        "s2": "sitting",
        "repetitions": 20_000,
        "repeat": 10,
        "warmup": 1_000,
    },
    "medium": {
        "label": "medium",
        "s1": REALISTIC_TEXT[0],
        "s2": REALISTIC_TEXT[1],
        "repetitions": 5_000,
        "repeat": 10,
        "warmup": 1_000,
    },
    "long-similar": {
        "label": "long similar",
        "s1": LONG_SIMILAR_TEXT[0],
        "s2": LONG_SIMILAR_TEXT[1],
        "repetitions": 500,
        "repeat": 10,
        "warmup": 100,
    },
    "long-dissimilar": {
        "label": "long dissimilar",
        "s1": LONG_DISSIMILAR_TEXT[0],
        "s2": LONG_DISSIMILAR_TEXT[1],
        "repetitions": 100,
        "repeat": 10,
        "warmup": 50,
    },
}

ALL_REGIMES: tuple[str, ...] = ("short", "medium", "long-similar", "long-dissimilar")
DEFAULT_REGIMES: tuple[str, ...] = ("medium", "long-similar", "long-dissimilar")

UNRELEASED_ALIASES = frozenset({"main", "unreleased", "current", "dev", "head", "local", "."})
SEMVER_REGEX = re.compile(r"^\d+(\.\d+)*$")

CHILD_RUNNER_CODE = """
import json
import sys
import timeit
import lev

payload = json.loads(sys.stdin.read())
regimes_input = payload["regimes"]
results = {}

for name, cfg in regimes_input.items():
    s1 = cfg["s1"]
    s2 = cfg["s2"]
    repetitions = int(cfg["repetitions"])
    repeat = int(cfg["repeat"])
    warmup = int(cfg["warmup"])

    dist = lev.distance(s1, s2)
    if warmup > 0:
        timeit.repeat(lambda: lev.distance(s1, s2), number=warmup, repeat=2)
    raw_times = timeit.repeat(lambda: lev.distance(s1, s2), number=repetitions, repeat=repeat)
    times_us = [(t / repetitions) * 1e6 for t in raw_times]

    results[name] = {
        "dist": dist,
        "times_us": times_us,
    }

print(json.dumps({
    "version": getattr(lev, "__version__", None),
    "regimes": results,
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
        Dictionary with cpu, os, arch, and python runtime information.

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
    regimes_cfg: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Run benchmark for all requested regimes in an isolated Python subprocess via uv.

    Args:
        version_label: Released version string (e.g. '0.5.0') or 'main'.
        regimes_cfg: Mapping of regime name to input parameters.

    Returns:
        Dictionary mapping regime name to statistical metrics.

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

    input_data = json.dumps({"regimes": regimes_cfg})
    result = subprocess.run(cmd, input=input_data, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Benchmark failed for version '{version_label}': {err_msg}")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No output received from benchmark runner for version '{version_label}'")

    data = json.loads(lines[-1])
    reported_version = data.get("version")
    out: dict[str, dict[str, Any]] = {}

    for regime_name, reg_res in data["regimes"].items():
        times_us = np.asarray(reg_res["times_us"], dtype=float)
        out[regime_name] = {
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
            "distance": reg_res["dist"],
            "reported_version": reported_version,
        }

    return out


def run_benchmarks(
    versions: Sequence[str] = DEFAULT_VERSIONS,
    regimes: Sequence[str] = DEFAULT_REGIMES,
) -> dict:
    """
    Benchmark each specified version of lev across requested regimes.

    Args:
        versions: Sequence of version strings to benchmark.
        regimes: Sequence of regime names to benchmark.

    Returns:
        Payload dictionary with metadata and per-regime per-version results.

    """
    regimes_cfg = {r: REGIMES_CATALOG[r] for r in regimes}
    results: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in regimes}

    print(f"Benchmarking {len(versions)} version(s) across {len(regimes)} regime(s) ({', '.join(regimes)})...")
    for ver in versions:
        sys.stdout.write(f"  • {ver:14s} ... ")
        sys.stdout.flush()
        try:
            ver_results = measure_version(ver, regimes_cfg)
            summary_parts = []
            for r in regimes:
                results[r][ver] = ver_results[r]
                summary_parts.append(f"{r}: {ver_results[r]['mean_us']:.3f} μs")
            print(f"done  [{', '.join(summary_parts)}]")
        except Exception as e:
            print(f"FAILED ({e})")
            raise

    regimes_meta = {
        r: {
            "label": REGIMES_CATALOG[r]["label"],
            "string_lengths": [len(REGIMES_CATALOG[r]["s1"]), len(REGIMES_CATALOG[r]["s2"])],
            "repetitions": REGIMES_CATALOG[r]["repetitions"],
            "repeat": REGIMES_CATALOG[r]["repeat"],
            "warmup": REGIMES_CATALOG[r]["warmup"],
        }
        for r in regimes
    }

    return {
        "meta": {
            "versions": list(versions),
            "regimes": list(regimes),
            "regimes_meta": regimes_meta,
            "machine": machine_info(),
        },
        "results": results,
    }


def _render_multi_series(
    ax: plt.Axes,
    x0: np.ndarray,
    regimes: Sequence[str],
    results: dict[str, dict],
    versions: Sequence[str],
    display_labels: Sequence[str],
    scale: str,
) -> None:
    """
    Render multiple regime series with confidence bands and end-of-line labels on a single axis.

    Args:
        ax: Matplotlib axes object.
        x0: X-axis indices.
        regimes: Sequence of regime names.
        results: Benchmark results mapping.
        versions: Sequence of version identifiers.
        display_labels: Formatted labels for x-axis ticks.
        scale: Scaling mode ('auto', 'linear', or 'log').

    """
    all_lowers: list[float] = []
    all_uppers: list[float] = []

    for r in regimes:
        means = np.array([results[r][v]["mean_us"] for v in versions])
        lowers = np.array([results[r][v]["min_us"] for v in versions])
        uppers = np.array([results[r][v]["max_us"] for v in versions])
        all_lowers.extend(lowers.tolist())
        all_uppers.extend(uppers.tolist())

        label = REGIMES_CATALOG[r]["label"]
        (line,) = ax.plot(x0, means, label=label, marker="o", markersize=3.5)
        ax.fill_between(x0, lowers, uppers, alpha=0.18, color=line.get_color())

    y_min = float(min(all_lowers))
    y_max = float(max(all_uppers))
    if scale == "log":
        ax.set_yscale("log")
        pad_factor = (y_max / max(y_min, 1e-9)) ** 0.05
        ax.set_ylim(max(y_min / pad_factor, 1e-9), y_max * pad_factor)
    else:
        pad = (y_max - y_min) * 0.05
        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    ax.set_xticks(x0)
    tick_labels = ax.set_xticklabels(display_labels, rotation=90, ha="right", va="top")
    for tick in tick_labels:
        if is_unreleased_version(tick.get_text()):
            tick.set_fontweight("bold")

    matplotx.ylabel_top("time [µs]", ax=ax)  # ruff: ignore[ambiguous-unicode-character-string]
    matplotx.line_labels(ax=ax)


def _render_bar_chart(
    ax: plt.Axes,
    x0: np.ndarray,
    means: np.ndarray,
    version_labels: list[str],
    results: dict[str, dict],
    theme: str,
    subtitle: str,
) -> None:
    """
    Render bar chart visualization for a single regime.

    Args:
        ax: Matplotlib axes object.
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
    bars = ax.bar(x0, means, color=colors, width=0.62, edgecolor="none", zorder=3)
    ax.bar_label(bars, fmt="%.2f μs", padding=4, fontsize=8.5, color=text_color)
    ax.set_ylim(0, max(means) * 1.20)
    ax.grid(True, axis="y", ls="--", alpha=0.5, color=grid_color, zorder=0)
    ax.grid(False, axis="x")

    ax.set_xticks(x0)
    tick_labels = ax.set_xticklabels(version_labels, rotation=90, ha="right", va="top")
    for tick in tick_labels:
        if is_unreleased_version(tick.get_text()):
            tick.set_fontweight("bold")
            tick.set_color(unreleased_color)

    ax.set_title("Levenshtein Distance Performance Across lev Versions", fontsize=13, pad=12)
    ax.set_xlabel(f"version\n{subtitle}", fontsize=9.5, labelpad=8)
    matplotx.ylabel_top("time [µs]", ax=ax)  # ruff: ignore[ambiguous-unicode-character-string]


def _get_theme_style(theme: str) -> Any:
    """
    Return the matplotlib style context for light or dark theme.

    Args:
        theme: Theme name ('light' or 'dark').

    Returns:
        Matplotlib style object.

    """
    if theme == "light":
        return matplotx.styles.duftify(matplotx.styles.github["light"])
    return matplotx.styles.duftify(matplotx.styles.github["dark"])


def _render_theme_figure(
    theme: str,
    payload: dict,
    chart_type: str,
    scale: str,
) -> plt.Figure:
    """
    Construct the figure for a single theme.

    Args:
        theme: Theme identifier ('light' or 'dark').
        payload: Full benchmark payload.
        chart_type: Chart type ('line' or 'bar').
        scale: Scaling mode ('auto', 'linear', or 'log').

    Returns:
        Rendered Matplotlib Figure.

    """
    results: dict[str, dict] = payload["results"]
    meta: dict = payload["meta"]
    regimes: list[str] = meta["regimes"]
    versions: list[str] = meta["versions"]
    display_labels = ["main" if is_unreleased_version(v) else v for v in versions]
    x0 = np.arange(len(versions))

    machine = meta["machine"]
    machine_desc = f"{machine['cpu']}, {machine['os']} {machine['arch']}, {machine['python']}"

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    if chart_type == "bar" and len(regimes) == 1:
        r = regimes[0]
        means = np.array([results[r][v]["mean_us"] for v in versions])
        _render_bar_chart(ax, x0, means, display_labels, results[r], theme, machine_desc)
    else:
        _render_multi_series(ax, x0, regimes, results, versions, display_labels, scale)
        ax.set_title("Levenshtein Distance Performance Across lev Versions", fontsize=12, pad=12)
        ax.set_xlabel(f"version\n{machine_desc}", fontsize=9.5, labelpad=8)

    return fig


def plot_versions(
    payload: dict,
    output_dir: Path = ASSETS_DIR,
    chart_type: str = "line",
    formats: tuple[str, ...] = ("svg", "png"),
    scale: str = "log",
) -> list[Path]:
    """
    Render and save light and dark benchmark plots for all tested versions.

    Args:
        payload: Benchmark results payload.
        output_dir: Directory where plot images should be saved.
        chart_type: 'line' for line plot with uncertainty band, 'bar' for bar chart.
        formats: File formats to export ('svg', 'png', etc.).
        scale: Scaling mode ('linear' or 'log').

    Returns:
        List of generated plot paths.

    """
    safe_dir = _validate_safe_path(output_dir)
    safe_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for theme in ("light", "dark"):
        style = _get_theme_style(theme)
        with plt.style.context(style):
            fig = _render_theme_figure(theme, payload, chart_type, scale)
            for fmt in formats:
                out_path = safe_dir / f"benchmark_by_version_{theme}.{fmt}"
                fig.savefig(out_path, bbox_inches="tight")
                generated_files.append(out_path)
            plt.close(fig)

    return generated_files


def _normalize_regime_token(token: str) -> str:
    """
    Normalize user-supplied regime token to canonical regime identifier.

    Args:
        token: Raw CLI token string.

    Returns:
        Canonical regime string.

    """
    cleaned = token.strip().lower()
    return "medium" if cleaned == "realistic" else cleaned


def resolve_regimes(arg_regimes: list[str] | None) -> tuple[str, ...]:
    """
    Resolve regime arguments into a normalized tuple of regime names.

    Args:
        arg_regimes: Raw CLI arguments list, or None.

    Returns:
        Tuple of regime names.

    """
    if not arg_regimes:
        return DEFAULT_REGIMES

    raw_tokens = [tok for item in arg_regimes for tok in item.split(",") if tok.strip()]
    if any(t.strip().lower() == "all" for t in raw_tokens):
        return ALL_REGIMES

    seen: set[str] = set()
    result: list[str] = []
    for t in raw_tokens:
        canon = _normalize_regime_token(t)
        if canon in ALL_REGIMES and canon not in seen:
            seen.add(canon)
            result.append(canon)

    return tuple(result) if result else DEFAULT_REGIMES


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
            "'0.2.4 0.2.6 0.5.0 main'). Defaults to DEFAULT_VERSIONS."
        ),
    )
    p.add_argument(
        "--regimes",
        nargs="*",
        default=None,
        help=(
            "Regimes to benchmark: 'medium', 'long-similar', 'long-dissimilar' "
            "(default), or 'all' (includes 'short'). Can also select specific regimes."
        ),
    )
    p.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to save JSON benchmark results (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    p.add_argument(
        "--chart-type",
        choices=["line", "bar"],
        default="line",
        help="Visualization chart type for single regime (default: line).",
    )
    p.add_argument(
        "--scale",
        choices=["linear", "log"],
        default="log",
        help="Y-axis scale mode for multi-series line plot (default: log).",
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


def main(argv: list[str] | None = None) -> int:
    """
    Execute benchmark across versions and generate summary and plots.

    Returns:
        0 on success.

    """
    args = parse_args(argv)
    versions_tuple = resolve_versions(args.versions)
    regimes_tuple = resolve_regimes(args.regimes)

    payload = run_benchmarks(
        versions=versions_tuple,
        regimes=regimes_tuple,
    )

    _safe_write_json(args.save, payload)
    print(f"\nWrote benchmark results JSON to {args.save}")

    if args.plot:
        fmts = ("svg", "png") if args.format == "both" else (args.format,)
        generated = plot_versions(
            payload,
            chart_type=args.chart_type,
            formats=fmts,
            scale=args.scale,
        )
        print(f"Generated {len(generated)} plot files in {ASSETS_DIR}:")
        for p in generated:
            print(f"  • {p.name}")

    # Summary table
    regimes = payload["meta"]["regimes"]
    versions = payload["meta"]["versions"]

    print("\nBenchmark Summary (Average Runtimes [μs]):")
    header = f"{'Version':<12}" + "".join(f"{r:>18}" for r in regimes)
    print(header)
    print("-" * len(header))
    for ver in versions:
        is_unrel = payload["results"][regimes[0]][ver]["is_unreleased"]
        label = f"{ver} *" if is_unrel else ver
        row = f"{label:<12}"
        for r in regimes:
            mean = payload["results"][r][ver]["mean_us"]
            row += f"{mean:>18.3f}"
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
