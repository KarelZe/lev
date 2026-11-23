# ruff: file-ignore[undocumented-public-function, missing-blank-line-after-summary]
"""
Summarize a run.py results file: per case, median ns (IQR %) of the reference build and
median ratio variant/reference per build, flagged when outside +-3% and outside the IQR noise.

Usage: python report.py results.json [--ref base] [--filter substr] [--only v1,v2] [--md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def quart(xs):
    if len(xs) < 2:
        return xs[0], xs[0], xs[0]
    q = statistics.quantiles(xs, n=4)
    return q[0], q[1], q[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--ref", default="base")
    ap.add_argument("--filter", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--min-dev", type=float, default=0.0, help="only rows where some build deviates by more")
    args = ap.parse_args()
    d = json.loads(args.path.read_text())
    runs = d["runs"]
    names = [n for n in d["builds"] if n != args.ref]
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    ref = runs[args.ref]
    keys = [k for k in ref[0] if args.filter in k]
    print(f"rounds={d['rounds']} repeat={d['repeat']} mismatch={ {n: v for n, v in d['mismatch'].items() if v} }")
    for n, m in d["meta"].items():
        print(f"  {n}: sha {m['sha']} {m['so']}")
    hdr = f"{'case':58s} {'ref ns':>9s} {'iqr%':>5s} " + " ".join(f"{n[:11]:>11s}" for n in names)
    print(hdr)
    geo = {n: [] for n in names}
    for k in keys:
        q = quart([r[k] for r in ref])
        cells = []
        dev = 0.0
        for n in names:
            qv = quart([r[k] for r in runs[n]])
            ratio = qv[1] / q[1]
            geo[n].append(ratio)
            noise = max(0.03, (q[2] - q[0]) / q[1], (qv[2] - qv[0]) / qv[1])
            flag = "*" if abs(ratio - 1) > noise else " "
            dev = max(dev, abs(ratio - 1))
            cells.append(f"{ratio:10.3f}{flag}")
        if dev >= args.min_dev:
            print(f"{k[:58]:58s} {q[1]:9.1f} {100 * (q[2] - q[0]) / q[1]:5.1f} " + " ".join(cells))
    print(f"{'geomean':58s} {'':9s} {'':5s} " + " ".join(f"{statistics.geometric_mean(geo[n]):10.3f} " for n in names))


if __name__ == "__main__":
    main()
