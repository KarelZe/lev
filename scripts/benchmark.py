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

n = 1_000


def measure() -> dict[str, float]:
    """
    Measure times for popular levenshtein implementations.

    Returns:
        dict[str, float]: dict with measures, library name as key and measure as value.

    """
    pylev_time = timeit(
        "pylev.levenshtein('Lets pretend Marshall Mathers never picked up a pen', 'Lets pretend things woulda been no different')",
        "import pylev",
        number=n,
    )
    rapidfuzz_time = timeit(
        "Levenshtein.distance('Lets pretend Marshall Mathers never picked up a pen', 'Lets pretend things woulda been no different')",
        "from rapidfuzz.distance import Levenshtein",
        number=n,
    )
    editdistance_time = timeit(
        "editdistance.eval('Lets pretend Marshall Mathers never picked up a pen', 'Lets pretend things woulda been no different')",
        "import editdistance",
        number=n,
    )
    levenshtein_time = timeit(
        "Levenshtein.distance('Lets pretend Marshall Mathers never picked up a pen', 'Lets pretend things woulda been no different')",
        "import Levenshtein",
        number=n,
    )
    lev_time = timeit(
        "lev.distance('Lets pretend Marshall Mathers never picked up a pen', 'Lets pretend things woulda been no different')",
        "import lev",
        number=n,
    )

    return {
        "editdistance": editdistance_time,
        "levenshtein": levenshtein_time,
        "pylev": pylev_time,
        "rapidfuzz": rapidfuzz_time,
        "lev [ours]": lev_time,
    }


def plot(measures: dict[str, float]) -> None:
    """
    Plot measures.

    Args:
        measures (dict[str, float]): Dictionary with library name and measure.

    """
    with plt.style.context(matplotx.styles.duftify(matplotx.styles.github["dark"])):
        # Generate Figure and Axes objects
        fig, ax = plt.subplots()

        # Plotting logic on the Axes object
        ax.bar(list(measures.keys()), list(measures.values()))

        # Formatting using Axes methods
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_yscale("log")
        ax.set_title(f"Average runtime for Levenshtein distance on ASCII strings with $n={n:,}$ repeats")

        matplotx.ylabel_top("time [ms]")

        # Save and close using the Figure object
        fig.savefig("benchmark_results.png", transparent=True, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    measures = measure()
    plot(measures)
