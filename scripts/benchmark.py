"""Simple benchmarking script for popular levenshtein implementations."""

import json
from pathlib import Path
from timeit import timeit

import matplotlib.pyplot as plt
import matplotx

# All output files are written here so `uv run scripts/benchmark.py` works from any cwd.
OUT_DIR = Path(__file__).parent.parent / "docs" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Iterations for each library comparison.
N_LIBS = 1_000


def make_len(s: str, n: int) -> str:
    """
    Repeat *s* until it is exactly *n* characters long.

    Returns:
        str: s repeated to length n.

    """
    return (s * (n // len(s) + 1))[:n]


# Representative 100-character string pairs keyed by CPython internal encoding kind.
KINDS: dict[str, tuple[str, str]] = {
    "ASCII": (
        make_len("Lets pretend Marshall Mathers never picked up a pen", 100),
        make_len("Lets pretend things woulda been no different", 100),
    ),
    "Latin-1": (
        make_len("H\xe9llo w\xf6rld, wi\xe9 geht \xe8s \xcdhnen?", 100),
        make_len("H\xe9llo w\xf6rld, wi\xe9 geht es Ihnen?", 100),
    ),
    "CJK": (
        make_len("\u65e5\u672c\u8a9e\u306e\u30c6\u30b9\u30c8\u6587\u5b57\u5217", 100),
        make_len("\u65e5\u672c\u8a9e\u306e\u30c6\u30b9\u30c4\u6587\u5b57\u5217", 100),
    ),
    "Emoji": (
        make_len("\U0001f980\U0001f40d\U0001f389\U0001f38a\U0001f388", 100),
        make_len("\U0001f40d\U0001f980\U0001f389\U0001f38a\U0001f388", 100),
    ),
}


def measure_libraries(s1: str, s2: str) -> dict[str, float]:
    """
    Measure wall time for each library on the given string pair.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        dict[str, float]: Library name to total time for N_LIBS repetitions.

    """
    return {
        "editdistance": timeit(
            f"editdistance.eval({s1!r}, {s2!r})",
            "import editdistance",
            number=N_LIBS,
        ),
        "edlib": timeit(
            f"edlib.align({s1!r}, {s2!r})['editDistance']",
            "import edlib",
            number=N_LIBS,
        ),
        "rapidfuzz": timeit(
            f"Levenshtein.distance({s1!r}, {s2!r})",
            "from rapidfuzz.distance import Levenshtein",
            number=N_LIBS,
        ),
        "lev": timeit(
            f"lev.distance({s1!r}, {s2!r})",
            "import lev",
            number=N_LIBS,
        ),
    }


def plot_libraries(measures: dict[str, float], kind: str) -> None:
    """
    Plot a horizontal bar chart comparing all libraries for the given encoding kind.

    Saves a light and a dark variant for use with MkDocs light/dark theme switching.

    Args:
        measures: Library name to total time in seconds.
        kind: Encoding kind label shown in the chart title and used in the filename.

    """
    # Convert to milliseconds and sort slowest → fastest.
    measures_ms = {k: v * 1000 for k, v in measures.items()}
    sorted_measures = dict(sorted(measures_ms.items(), key=lambda item: item[1], reverse=True))

    # Slug for the filename: "Latin-1" → "latin_1", "CJK" → "cjk", etc.
    kind_slug = kind.lower().replace("-", "_").replace(" ", "_")

    def _render(theme: str) -> None:
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
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            bar_colors = [colors[2] if k == "lev" else colors[0] for k in sorted_measures]
            fig, ax = plt.subplots(figsize=(10, 2))
            bars = ax.barh(list(sorted_measures.keys()), list(sorted_measures.values()), color=bar_colors)
            ax.bar_label(bars, padding=5, fmt="%.1f ms")
            ax.grid(True, axis="x", ls="-")
            ax.grid(False, axis="y")
            ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.15)
            ax.set_title(f"{kind} [100 chars, n={N_LIBS}]")
            ax.set_xlabel("time [ms]")
            fig.savefig(OUT_DIR / f"benchmark_{kind_slug}_{theme}.svg", bbox_inches="tight")
            plt.close(fig)

    _render("light")
    _render("dark")


if __name__ == "__main__":
    all_results: dict[str, dict[str, float]] = {}

    for kind, (s1, s2) in KINDS.items():
        print(f"Benchmarking {kind}...")
        times = measure_libraries(s1, s2)
        plot_libraries(times, kind)
        all_results[kind] = times

        lev_ms = times["lev"] / N_LIBS * 1000
        rf_ms = times["rapidfuzz"] / N_LIBS * 1000
        print(f"  lev {lev_ms:.3f} ms  rapidfuzz {rf_ms:.3f} ms  speedup {rf_ms / lev_ms:.2f}x")

    with open(OUT_DIR / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults written to {OUT_DIR}")
