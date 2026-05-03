"""Simple benchmarking script to write numbers to disk."""

import argparse
import json

import numpy as np
from benchmark_by_length import LENGTHS, LIBRARIES, run_all


def main():
    """Calculate numbers."""
    parser = argparse.ArgumentParser(description="Dump lev benchmark to JSON.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="bench_numbers.json",
        help="Output JSON file name (e.g. bench_before.json or bench_after.json)",
    )
    args = parser.parse_args()

    data = run_all()
    summary = {}
    for kind in data:
        summary[kind] = {}
        for lib in LIBRARIES:
            ys = data[kind][lib]
            med = np.median(ys, axis=1)
            summary[kind][lib] = med.tolist()

    with open(args.output, "w") as f:
        json.dump({"lengths": LENGTHS, "data": summary}, f, indent=2)


if __name__ == "__main__":
    main()
