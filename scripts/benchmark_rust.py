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
Summarize and plot the criterion comparison of lev against other Rust crates.

Run the benchmark first, then this script:

    (cd benches/compare && cargo bench)
    uv run scripts/benchmark_rust.py

It reads criterion's raw samples from `benches/compare/target/criterion`,
prints the `kind/*` group as a Markdown table of median time per call,
writes the summary to `docs/assets/benchmark_rust_results.json`, and plots
the `length` group as light/dark SVGs to `docs/assets/`.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benches" / "compare"
CRITERION_DIR = BENCH_DIR / "target" / "criterion"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"

KIND_LABELS = {
    "ascii": "ASCII",
    "latin1": "Latin-1",
    "cjk": "CJK",
    "emoji": "Emoji",
    "realistic": "Realistic",
}


def load(criterion_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Read every benchmark's latest samples.

    Returns:
        `results[group][id]` with the median and 25th/75th percentile time
        per call in nanoseconds (per pair for `mixed`). `id` is the crate,
        or `<crate>/<length>` for the `length` group.

    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    for meta_path in sorted(criterion_dir.glob("**/new/benchmark.json")):
        meta = json.loads(meta_path.read_text())
        sample = json.loads((meta_path.parent / "sample.json").read_text())
        per_call = np.asarray(sample["times"]) / np.asarray(sample["iters"])
        if meta["group_id"] == "mixed":
            # One iteration runs every pair; report the time per pair.
            per_call /= meta["throughput"]["Elements"]
        # `BenchmarkId::from_parameter` leaves `function_id` empty.
        key = "/".join(part for part in (meta["function_id"], meta["value_str"]) if part)
        results.setdefault(meta["group_id"], {})[key] = {
            "median_ns": float(np.median(per_call)),
            "p25_ns": float(np.percentile(per_call, 25)),
            "p75_ns": float(np.percentile(per_call, 75)),
        }
    return results


def crate_versions() -> dict[str, str]:
    """
    Read the benchmarked crates' versions from the lock file.

    Returns:
        Crate name to version.

    """
    lock = (BENCH_DIR / "Cargo.lock").read_text()
    manifest = (BENCH_DIR / "Cargo.toml").read_text()
    wanted = set(re.findall(r"^([\w-]+) = ", manifest, flags=re.MULTILINE))
    pkgs = re.findall(r'name = "([^"]+)"\nversion = "([^"]+)"', lock)
    return {name: ver for name, ver in pkgs if name in wanted}


def machine_info() -> dict[str, str]:
    """
    Describe the machine and toolchain.

    Returns:
        CPU model, OS, architecture, and rustc version.

    """
    cpu = platform.processor() or "unknown CPU"
    try:
        if sys.platform == "linux":
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        elif sys.platform == "darwin":
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
    except (OSError, StopIteration, subprocess.CalledProcessError):
        pass
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=False).stdout.strip()
    return {"cpu": cpu, "os": platform.system(), "arch": platform.machine(), "rustc": rustc}


def _fmt(ns: float) -> str:
    return f"{ns:,.0f} ns" if ns < 10_000 else f"{ns / 1000:,.1f} µs"  # ruff: ignore[ambiguous-unicode-character-string]


def kind_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """
    Format the `kind/*` and `mixed` groups as a Markdown table, fastest crate first.

    Returns:
        The table, with `lev` in bold and `n/a` where a crate was skipped.

    """
    columns = {f"kind/{k}": label for k, label in KIND_LABELS.items()} | {"mixed": "Mixed (per pair)"}
    columns = {g: label for g, label in columns.items() if g in results}
    crates = sorted(
        {c for g in columns for c in results[g]},
        key=lambda c: results["kind/ascii"].get(c, {"median_ns": float("inf")})["median_ns"],
    )
    head = "| crate | " + " | ".join(columns.values()) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in columns) + " |"
    rows = []
    for crate in crates:
        cells = []
        for g in columns:
            r = results[g].get(crate)
            cell = _fmt(r["median_ns"]) if r else "n/a"
            cells.append(f"**{cell}**" if crate == "lev" and r else cell)
        name = f"**`{crate}`**" if crate == "lev" else f"`{crate}`"
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def plot_lengths(per_id: dict[str, dict[str, float]], machine: dict[str, str]) -> None:
    """Plot median time per call against string length, log-log, as light and dark SVGs."""
    series: dict[str, list[tuple[int, dict[str, float]]]] = {}
    for key, stats in per_id.items():
        crate, n = key.rsplit("/", 1)
        series.setdefault(crate, []).append((int(n), stats))
    # Legend order: slowest at the longest length first, so it matches the lines top to bottom.
    order = sorted(series, key=lambda c: -max(series[c])[1]["median_ns"])
    # Fixed colors per crate (the theme's own cycle repeats after seven), lev first.
    palette = plt.get_cmap("tab10").colors
    colors = {c: palette[i] for i, c in enumerate(sorted(series, key=lambda c: (c != "lev", c)))}
    lengths = sorted({n for pts in series.values() for n, _ in pts})
    subtitle = f"{machine['cpu']}, {machine['os']} {machine['arch']}, {machine['rustc']}"

    for theme in ("light", "dark"):
        text_color = "white" if theme == "dark" else "black"
        with plt.style.context(matplotx.styles.duftify(matplotx.styles.github[theme])):
            plt.rcParams.update({
                "text.color": text_color,
                "axes.labelcolor": text_color,
                "xtick.color": text_color,
                "ytick.color": text_color,
                "axes.edgecolor": text_color,
                "legend.edgecolor": text_color,
            })
            fig, ax = plt.subplots(figsize=(10, 5))
            for crate in order:
                pts = sorted(series[crate])
                xs = [n for n, _ in pts]
                ax.plot(
                    xs,
                    [s["median_ns"] / 1000 for _, s in pts],
                    label=crate,
                    color=colors[crate],
                    marker="o",
                    markersize=3,
                    linewidth=2.6 if crate == "lev" else 1.4,
                )
                ax.fill_between(
                    xs,
                    [s["p25_ns"] / 1000 for _, s in pts],
                    [s["p75_ns"] / 1000 for _, s in pts],
                    alpha=0.2,
                    color=colors[crate],
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(lengths, labels=[str(n) for n in lengths])
            ax.minorticks_off()
            ax.grid(True, which="major", ls="-")
            ax.set_title("ASCII, 20% substitutions [median, 25th-75th percentile band]")
            ax.set_xlabel(f"string length [chars]\n{subtitle}")
            ax.set_ylabel("time per call [µs]")  # ruff: ignore[ambiguous-unicode-character-string]
            legend = ax.legend(loc="upper left")
            for text in legend.get_texts():
                if text.get_text() == "lev":
                    text.set_fontweight("bold")
            fig.savefig(ASSETS_DIR / f"benchmark_rust_length_{theme}.svg", bbox_inches="tight")
            plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """
    Summarize the criterion results, write JSON, and plot.

    Returns:
        Process exit code (0 on success).

    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--criterion-dir", type=Path, default=CRITERION_DIR, help="criterion output directory.")
    p.add_argument("--save", type=Path, default=ASSETS_DIR / "benchmark_rust_results.json", help="JSON output.")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Write the SVG plots.")
    args = p.parse_args(argv)

    results = load(args.criterion_dir)
    if not results:
        print(f"no criterion results in {args.criterion_dir}; run `cargo bench` in {BENCH_DIR}", file=sys.stderr)
        return 1
    machine = machine_info()
    payload = {"meta": {"machine": machine, "versions": crate_versions()}, "results": results}
    args.save.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.save}\n")
    print(kind_table(results))

    if args.plot and "length" in results:
        plot_lengths(results["length"], machine)
        print(f"\nwrote benchmark_rust_length_{{light,dark}}.svg to {ASSETS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
