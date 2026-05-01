"""Simple benchmarking script for popular levenshtein implementations."""
from timeit import timeit

import matplotlib.pyplot as plt

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
        "lev": lev_time,
    }


def plot(measures: dict[str, float]) -> None:
    """
    Plot measures.

    Args:
        measures (dict[str, float]): dict with library name and measure

    """
    _fig, ax = plt.subplots()
    ax.bar(measures.keys(), measures.values())
    ax.set_xlabel(f"Time in ms for n={n:,}")
    ax.set_title("Levenshtein distance (ascii)")
    plt.yscale("log")
    plt.savefig("benchmark_results.png")


if __name__ == "__main__":
    measures = measure()
    plot(measures)
