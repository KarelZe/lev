# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "editdistance>=0.8.1",
#     "levenshtein>=0.27.3",
#     "matplotlib>=3.10.9",
#     "matplotx>=0.3.10",
#     "pylev>=1.4.0",
#     "rapidfuzz>=3.14.5",
# ]
# ///
"""Simple benchmarking script for popular levenshtein implementations."""

from timeit import timeit

import matplotlib.pyplot as plt
import matplotx

# Iterations for the slow multi-library comparison (pylev is ~1000x slower).
N_LIBS = 1_000
# Iterations for the fast lev-vs-rapidfuzz kind comparison.
N_KINDS = 100_000

# Representative string pairs keyed by their CPython internal encoding kind.
KINDS: dict[str, tuple[str, str]] = {
    "ASCII": (
        "Lets pretend Marshall Mathers never picked up a pen",
        "Lets pretend things woulda been no different",
    ),
    "Latin-1": (
        "H\xe9llo w\xf6rld, wi\xe9 geht \xe8s \xcdhnen?",
        "H\xe9llo w\xf6rld, wi\xe9 geht es Ihnen?",
    ),
    "CJK": ("日本語のテスト文字列", "日本語のテスツ文字列"),
    "Emoji": (
        "\U0001f980\U0001f40d\U0001f389\U0001f38a\U0001f388",
        "\U0001f40d\U0001f980\U0001f389\U0001f38a\U0001f388",
    ),
}

_ASCII_S1, _ASCII_S2 = KINDS["ASCII"]


def measure_libraries() -> dict[str, float]:
    """
    Measure wall time for each library on the ASCII pair.

    Returns:
        dict[str, float]: Library name to total time for N_LIBS repetitions.

    """
    return {
        "editdistance": timeit(
            f"editdistance.eval({_ASCII_S1!r}, {_ASCII_S2!r})",
            "import editdistance",
            number=N_LIBS,
        ),
        "levenshtein": timeit(
            f"Levenshtein.distance({_ASCII_S1!r}, {_ASCII_S2!r})",
            "import Levenshtein",
            number=N_LIBS,
        ),
        "pylev": timeit(
            f"pylev.levenshtein({_ASCII_S1!r}, {_ASCII_S2!r})",
            "import pylev",
            number=N_LIBS,
        ),
        "rapidfuzz": timeit(
            f"Levenshtein.distance({_ASCII_S1!r}, {_ASCII_S2!r})",
            "from rapidfuzz.distance import Levenshtein",
            number=N_LIBS,
        ),
        "lev [ours]": timeit(
            f"lev.distance({_ASCII_S1!r}, {_ASCII_S2!r})",
            "import lev",
            number=N_LIBS,
        ),
    }


def measure_by_kind() -> dict[str, dict[str, float]]:
    """
    Measure lev vs rapidfuzz across all four CPython string-encoding kinds.

    Returns:
        dict[str, dict[str, float]]: Kind to library-to-time mapping for N_KINDS repetitions.

    """
    results: dict[str, dict[str, float]] = {}
    for kind, (s1, s2) in KINDS.items():
        results[kind] = {
            "lev [ours]": timeit(
                f"lev.distance({s1!r}, {s2!r})",
                "import lev",
                number=N_KINDS,
            ),
            "rapidfuzz": timeit(
                f"Levenshtein.distance({s1!r}, {s2!r})",
                "from rapidfuzz.distance import Levenshtein",
                number=N_KINDS,
            ),
        }
    return results


def plot_libraries(measures: dict[str, float]) -> None:
    """
    Plot a bar chart comparing all libraries on the ASCII pair.

    Args:
        measures: Library name to total time in seconds.

    """
    with plt.style.context(matplotx.styles.duftify(matplotx.styles.github["dark"])):
        fig, ax = plt.subplots()
        ax.bar(measures.keys(), measures.values())
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_yscale("log")
        ax.set_title(f"Average runtime for Levenshtein distance on ASCII strings with $n={N_LIBS:,}$ repeats")
        matplotx.ylabel_top("time [ms]")
        fig.savefig("benchmark_results.png", bbox_inches="tight")
        plt.close(fig)


def plot_by_kind(by_kind: dict[str, dict[str, float]]) -> None:
    """
    Plot a grouped bar chart comparing lev vs rapidfuzz per string-encoding kind.

    Args:
        by_kind: Kind to library-to-time mapping in seconds.

    """
    kinds = list(by_kind)
    libraries = list(next(iter(by_kind.values())))
    width = 0.35
    x = list(range(len(kinds)))

    with plt.style.context(matplotx.styles.duftify(matplotx.styles.github["dark"])):
        fig, ax = plt.subplots()
        for i, lib in enumerate(libraries):
            times_us = [by_kind[k][lib] / N_KINDS * 1e6 for k in kinds]
            offset = (i - (len(libraries) - 1) / 2) * width
            ax.bar([xi + offset for xi in x], times_us, width, label=lib)
        ax.set_xticks(x)
        ax.set_xticklabels(kinds)
        ax.legend()
        ax.set_title(f"runtime by string-encoding kind ($n={N_KINDS:,}$ repeats)")
        matplotx.ylabel_top("time [μs]")
        fig.savefig("benchmark_results_by_kind.png", bbox_inches="tight")
        plt.close(fig)


def print_kind_table(by_kind: dict[str, dict[str, float]]) -> None:
    """
    Print a per-kind timing table to stdout.

    Args:
        by_kind: Kind to library-to-time mapping in seconds.

    """
    header = f"{'Kind':8s}  {'lev [ours]':>12s}  {'rapidfuzz':>12s}  {'speedup':>8s}"
    print(header)
    print("-" * len(header))
    for kind, times in by_kind.items():
        lev_us = times["lev [ours]"] / N_KINDS * 1e6
        rf_us = times["rapidfuzz"] / N_KINDS * 1e6
        print(f"{kind:8s}  {lev_us:>11.3f}μs  {rf_us:>11.3f}μs  {rf_us / lev_us:>7.2f}x")


if __name__ == "__main__":
    lib_times = measure_libraries()
    plot_libraries(lib_times)

    kind_times = measure_by_kind()
    print_kind_table(kind_times)
    plot_by_kind(kind_times)
