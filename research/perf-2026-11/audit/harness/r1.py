# ruff: file-ignore[undocumented-public-function]
"""
R1 fixed-overhead decomposition: ns per call of tiny inputs, interleaved builds in subprocesses.

Usage: python r1.py --builds base=DIR probe=DIR --rounds 7 [--out file.json]
Runs with whichever interpreter executes it (children use the same one).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import operator
import statistics
import subprocess
import sys
import timeit
from collections import deque
from itertools import repeat
from pathlib import Path

N = 200_000


def child(build_dir: str, rep: int) -> dict:
    sys.path.insert(0, build_dir)
    import lev

    assert Path(lev.__file__).resolve().is_relative_to(Path(build_dir).resolve())
    so = next(Path(lev.__file__).parent.glob("_lev*.so"))
    gc.disable()
    hello2 = "".join(["hel", "lo"])
    cases = {
        "operator.is_": (operator.is_, "kitten", "sitting"),
        "d:same-object": (lev.distance, "kitten", "kitten"),
        "d:empty": (lev.distance, "", ""),
        "d:equal-5": (lev.distance, "hello", hello2),
        "d:a-b": (lev.distance, "a", "b"),
        "d:kitten": (lev.distance, "kitten", "sitting"),
        "d:ascii-12-dis": (lev.distance, "qwertyuiopas", "zxcvbnmlkjhg"),
        "d:cjk-3": (lev.distance, "日本語", "日本"),
        "r:same-object": (lev.ratio, "kitten", "kitten"),
        "r:empty": (lev.ratio, "", ""),
        "r:a-b": (lev.ratio, "a", "b"),
        "r:kitten": (lev.ratio, "kitten", "sitting"),
        "r:ascii-12-dis": (lev.ratio, "qwertyuiopas", "zxcvbnmlkjhg"),
    }
    out = {}
    for k, (fn, a, b) in cases.items():
        t = min(timeit.repeat(lambda: deque(map(fn, repeat(a, N), repeat(b, N)), maxlen=0), number=1, repeat=rep))
        out[k] = t / N * 1e9
    return {"times": out, "sha": hashlib.sha256(so.read_bytes()).hexdigest()[:16], "so": str(so)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=[])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--child")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child, a.repeat)))
        return 0
    builds = dict(b.split("=", 1) for b in a.builds)
    names = list(builds)
    runs = {n: [] for n in names}
    meta = {}
    for r in range(a.rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            res = subprocess.run(
                [sys.executable, "-I", __file__, "--child", builds[n], "--repeat", str(a.repeat)],
                capture_output=True,
                text=True,
                check=True,
            )
            d = json.loads(res.stdout)
            runs[n].append(d["times"])
            meta[n] = {"sha": d["sha"], "so": d["so"]}
    print(f"Python {sys.version.split()[0]}; {a.rounds} rounds; median ns/call (IQR)")
    for n, m in meta.items():
        print(f"  {n}: {m['sha']} {m['so']}")
    keys = list(runs[names[0]][0])
    print(f"{'case':18s} " + " ".join(f"{n:>16s}" for n in names))
    for k in keys:
        cells = []
        for n in names:
            xs = [x[k] for x in runs[n]]
            q = statistics.quantiles(xs, n=4) if len(xs) > 1 else [xs[0]] * 3
            cells.append(f"{q[1]:8.1f} ({q[2] - q[0]:4.1f})")
        print(f"{k:18s} " + " ".join(f"{c:>16s}" for c in cells))
    if a.out:
        a.out.write_text(json.dumps({"runs": runs, "meta": meta, "python": sys.version}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
