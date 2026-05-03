"""Dump benchmark numbers after optimization."""

import json

import numpy as np
from benchmark_by_length import LENGTHS, LIBRARIES, run_all

data = run_all()
summary = {}
for kind in data:
    summary[kind] = {}
    for lib in LIBRARIES:
        ys = data[kind][lib]
        med = np.median(ys, axis=1)
        summary[kind][lib] = {str(n): float(m) for n, m in zip(LENGTHS, med)}

with open("bench_after.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Dumped bench_after.json")
