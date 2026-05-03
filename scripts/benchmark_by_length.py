"""
Benchmark lev vs rapidfuzz across string lengths for each CPython encoding kind.

Covers UCS-1 ASCII, UCS-1 Latin-1, UCS-2 CJK, and UCS-4 Emoji. Each subplot
is a line-plot of median runtime (μs) vs. string length. Shaded bands show the
25th-75th percentile over N_REPEAT independent timeit runs.
"""

import timeit

import matplotlib.pyplot as plt
import matplotx
import numpy as np
from rapidfuzz.distance import Levenshtein

import lev as _lev

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_REPEAT = 20  # independent timing runs per (kind, length, library)
N_NUMBER = 10_000  # calls per run (determines timing resolution)

# Dense sampling around the 64-char Hyyrö word-size boundary.
LENGTHS = [4, 8, 16, 32, 48, 56, 64, 72, 96, 128, 192, 256, 384, 512]

# Each entry: (pool_a, pool_b).
# s1 is built by cycling pool_a; s2 swaps every 5th char to pool_b,
# giving a stable ~20 % edit distance regardless of length.
KINDS: dict[str, tuple[str, str]] = {
    "UCS-1 — ASCII": (
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ),
    "UCS-1 — Latin-1": (
        "àáâãäåæçèéêëìíîïðñòóôõöøùúûüý",
        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ",
    ),
    "UCS-2 — CJK": (
        "".join(chr(0x3041 + i) for i in range(80)),  # Hiragana U+3041..
        "".join(chr(0x30A1 + i) for i in range(80)),  # Katakana U+30A1..
    ),
    "UCS-4 — Emoji": (
        "".join(chr(0x1F600 + i) for i in range(80)),
        "".join(chr(0x1F610 + i) for i in range(80)),
    ),
}

LIBRARIES: dict[str, object] = {
    "lev [ours]": _lev.distance,
    "rapidfuzz": Levenshtein.distance,
}

# ---------------------------------------------------------------------------
# String generation
# ---------------------------------------------------------------------------


def make_pair(pool_a: str, pool_b: str, n: int) -> tuple[str, str]:
    """
    Return (s1, s2) of length n with ~20 % substitutions.

    Returns:
        tuple[str, str]: A pair of strings (s1, s2) of length n.

    """
    s1 = "".join(pool_a[i % len(pool_a)] for i in range(n))
    s2 = "".join(pool_a[i % len(pool_a)] if i % 5 != 0 else pool_b[i % len(pool_b)] for i in range(n))
    return s1, s2


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def measure(fn, s1: str, s2: str) -> np.ndarray:
    """
    Return an array of N_REPEAT per-call times in microseconds.

    Returns:
        np.ndarray: Array of N_REPEAT per-call execution times in microseconds.

    """
    timer = timeit.Timer("fn(s1, s2)", globals={"fn": fn, "s1": s1, "s2": s2})
    raw = timer.repeat(repeat=N_REPEAT, number=N_NUMBER)
    return np.asarray(raw) / N_NUMBER * 1e6


def run_all() -> dict[str, dict[str, np.ndarray]]:
    """
    Return data[kind][lib] with shape (len(LENGTHS), N_REPEAT), values in µs per call.

    Returns:
        dict[str, dict[str, np.ndarray]]: Data for each kind and library,
            shape (len(LENGTHS), N_REPEAT), values in µs per call.

    """  # noqa: RUF002
    data: dict[str, dict[str, np.ndarray]] = {
        k: {lib: np.empty((len(LENGTHS), N_REPEAT)) for lib in LIBRARIES} for k in KINDS
    }
    for kind, (pa, pb) in KINDS.items():
        print(f"  {kind}", end="", flush=True)
        for li, n in enumerate(LENGTHS):
            s1, s2 = make_pair(pa, pb, n)
            for lib, fn in LIBRARIES.items():
                data[kind][lib][li] = measure(fn, s1, s2)
            print(f" {n}", end="", flush=True)
        print()
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(
    data: dict[str, dict[str, np.ndarray]],
    out: str = "benchmark_by_length.png",
) -> None:
    """
    Plot timings.

    Args:
        data (dict[str, dict[str, np.ndarray]]): timings
        out (str, optional): file name. Defaults to "benchmark_by_length.png".

    """
    xs = np.asarray(LENGTHS, dtype=float)
    style = matplotx.styles.duftify(matplotx.styles.github["dark"])

    with plt.style.context(style):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(
            "Levenshtein distance — runtime vs. string length\n"
            f"median  +  25th-75th percentile band  "
            f"({N_REPEAT} runs × {N_NUMBER:,} calls)",  # noqa: RUF001
            fontsize=14,
        )
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        lib_colors = {lib: (colors[2] if lib == "lev [ours]" else colors[0]) for lib in LIBRARIES}

        for ax, kind in zip(axes.flat, KINDS):
            for lib, _ in LIBRARIES.items():
                color = lib_colors[lib]
                ys = data[kind][lib]  # (n_lengths, N_REPEAT)
                med = np.median(ys, axis=1)
                p25 = np.percentile(ys, 25, axis=1)
                p75 = np.percentile(ys, 75, axis=1)
                ax.plot(xs, med, label=lib, color=color, linewidth=1.8)
                ax.fill_between(xs, p25, p75, alpha=0.22, color=color)

            ax.set_title(kind, fontsize=13)
            ax.set_xlabel("string length (chars)", fontsize=11)
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(bottom=0)
            matplotx.ylabel_top("time (µs)", ax=ax)  # noqa: RUF001
            ax.legend(fontsize=11)

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Benchmarking ({N_REPEAT} runs × {N_NUMBER:,} calls per data point) …")  # noqa: RUF001
    data = run_all()
    print("Plotting …")
    plot(data)
