# ruff: file-ignore[undocumented-public-function, missing-blank-line-after-summary, unused-variable]
"""
In-process interleaved A/B: every build's extension is loaded into one process (distinct .so files,
distinct module statics) and, per case, batches alternate between builds in rotating order. Paired
timing within a few ms cancels the P-cluster frequency steps (~8%) that other load on the machine
causes between processes. Each round is a fresh subprocess (new ASLR/layout); per round the ratio of
best batch times build/ref is taken; the report gives the median ratio over rounds and its IQR.

Usage: python ab2.py --builds base=DIR v=DIR ... --set 1|2 [--filter substr,substr] --rounds 5 --out f.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from collections import deque
from itertools import repeat
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_NS = 1_000_000


def load(tag: str, build_dir: str):
    so = next((Path(build_dir) / "lev").glob("_lev*.so"))
    spec = importlib.util.spec_from_file_location(f"{tag}._lev", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, str(so), hashlib.sha256(so.read_bytes()).hexdigest()[:16]


def child(builds: dict[str, str], pkl: str, reps: int, seed: int) -> dict:
    mods = {}
    meta = {}
    for tag, d in builds.items():
        m, so, sha = load(tag, d)
        mods[tag] = m
        meta[tag] = {"so": so, "sha": sha}
    cases = pickle.loads(Path(pkl).read_bytes())
    names = list(builds)
    gc.disable()
    out: dict[str, dict[str, float]] = {}
    mism: list[str] = []
    for ci, c in enumerate(cases):
        for fn_name in c["fns"]:
            key = f"{fn_name}:{c['name']}"
            fns = {t: getattr(mods[t], fn_name) for t in names}
            if c["kind"] == "pair":
                s1, s2 = c["pair"]
                ref = fns[names[0]](s1, s2)
                if any(f(s1, s2) != ref for f in fns.values()):
                    mism.append(key)
                f0 = fns[names[0]]
                t0 = time.perf_counter_ns()
                f0(s1, s2)
                one = max(20, time.perf_counter_ns() - t0)
                n = 64
                while True:
                    t0 = time.perf_counter_ns()
                    deque(map(f0, repeat(s1, n), repeat(s2, n)), maxlen=0)
                    t = time.perf_counter_ns() - t0
                    if t > 200_000 or n >= 200_000:
                        break
                    n *= 4
                n = max(1, min(200_000, round(n * TARGET_NS / max(t, 1))))

                def run(f, n=n, s1=s1, s2=s2):
                    t0 = time.perf_counter_ns()
                    deque(map(f, repeat(s1, n), repeat(s2, n)), maxlen=0)
                    return (time.perf_counter_ns() - t0) / n

            else:
                a, b = c["batch"]
                ref = list(map(fns[names[0]], a, b))
                if any(list(map(f, a, b)) != ref for f in fns.values()):
                    mism.append(key)

                def run(f, a=a, b=b):
                    t0 = time.perf_counter_ns()
                    deque(map(f, a, b), maxlen=0)
                    return (time.perf_counter_ns() - t0) / len(a)

            best = {t: float("inf") for t in names}
            for r in range(reps):
                k = (r + ci + seed) % len(names)
                order = names[k:] + names[:k]
                if r % 2:
                    order = order[::-1]
                for t in order:
                    best[t] = min(best[t], run(fns[t]))
            out[key] = best
    return {"times": out, "meta": meta, "mismatch": mism}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=[])
    ap.add_argument("--set", default="1")
    ap.add_argument("--filter", default="")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--pkl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    builds = dict(b.split("=", 1) for b in a.builds)
    if a.child:
        print(json.dumps(child(builds, a.pkl, a.reps, a.seed)))
        return 0
    sys.path.insert(0, str(HERE))
    import cases as cm

    cases = {"1": cm.build, "2": cm.build2, "3": cm.build3}[a.set]()
    if a.filter:
        subs = a.filter.split(",")
        cases = [c for c in cases if any(s in c["name"] for s in subs)]
    pkl = a.out.with_suffix(".cases.pkl")
    pkl.write_bytes(pickle.dumps(cases))
    rounds = []
    t0 = time.time()
    for r in range(a.rounds):
        for _ in range(600):  # wait out other agents' compiles
            if subprocess.run(["pgrep", "-x", "rustc"], capture_output=True).returncode != 0 and os.getloadavg()[0] < 6:
                break
            time.sleep(2)
        cmd = [
            sys.executable,
            "-I",
            str(HERE / "ab2.py"),
            "--child",
            "--pkl",
            str(pkl),
            "--reps",
            str(a.reps),
            "--seed",
            str(r),
            "--builds",
            *a.builds,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode:
            raise SystemExit(res.stderr)
        d = json.loads(res.stdout)
        rounds.append(d)
        print(
            f"round {r + 1}/{a.rounds} at {time.time() - t0:.0f}s load {os.getloadavg()[0]:.2f} mismatches {d['mismatch']}",
            file=sys.stderr,
        )
    a.out.write_text(json.dumps({"builds": builds, "rounds": rounds}))
    names = list(builds)
    ref = names[0]
    print(
        f"in-process interleaved, {a.rounds} rounds x {a.reps} reps; cells: median ratio build/{ref} over rounds (IQR of ratio)"
    )
    for t, m in rounds[0]["meta"].items():
        print(f"  {t}: {m['sha']} {m['so']}")
    print(f"{'case':52s} {'ref ns':>9s} " + " ".join(f"{n[:13]:>15s}" for n in names[1:]))
    geo = {n: [] for n in names[1:]}
    for key in rounds[0]["times"]:
        refs = [rd["times"][key][ref] for rd in rounds]
        cells = []
        for n in names[1:]:
            rs = [rd["times"][key][n] / rd["times"][key][ref] for rd in rounds]
            q = statistics.quantiles(rs, n=4) if len(rs) > 1 else [rs[0]] * 3
            geo[n].append(q[1])
            flag = "*" if abs(q[1] - 1) > max(0.03, q[2] - q[0]) else " "
            cells.append(f"{q[1]:6.3f}({q[2] - q[0]:5.3f}){flag}")
        print(f"{key[:52]:52s} {statistics.median(refs):9.1f} " + " ".join(f"{c:>15s}" for c in cells))
    print(f"{'geomean':52s} {'':9s} " + " ".join(f"{statistics.geometric_mean(geo[n]):15.3f}" for n in names[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
