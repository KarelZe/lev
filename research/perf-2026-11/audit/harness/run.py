# ruff: file-ignore[undocumented-public-function]
"""
Interleaved multi-build wall-clock harness for the lev layer audit.

Each round starts one subprocess per build, in a rotating order, which times
every case (best of --repeat batches, fixed call counts calibrated once on the
first build) and returns ns per call plus a digest of all results, so every
build is also checked to compute identical outputs.

Usage:
  python run.py --builds base=DIR v1=DIR ... --rounds 7 --out results.json [--tags a,b]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
import timeit
from collections import deque
from itertools import repeat
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_NS = 1_500_000  # ns per batch


def load_lev(build_dir: str):
    sys.path.insert(0, build_dir)
    import lev

    assert Path(lev.__file__).resolve().is_relative_to(Path(build_dir).resolve()), lev.__file__
    d = Path(lev.__file__).parent
    so = next(d.glob("_lev*.so"))
    return lev, str(so), hashlib.sha256(so.read_bytes()).hexdigest()[:16], so.stat().st_mtime


def child(build_dir: str, cases_pkl: str, counts_json: str | None, repeat_n: int, calibrate: bool) -> dict:
    lev, so, sha, mtime = load_lev(build_dir)
    cases = pickle.loads(Path(cases_pkl).read_bytes())
    counts = json.loads(Path(counts_json).read_text()) if counts_json else {}
    gc.disable()
    times: dict[str, float] = {}
    digests: dict[str, str] = {}
    for c in cases:
        for fn_name in c["fns"]:
            fn = getattr(lev, fn_name)
            key = f"{fn_name}:{c['name']}"
            if c["kind"] == "pair":
                s1, s2 = c["pair"]
                out = repr(fn(s1, s2))
                if calibrate:
                    n = 1
                    while True:
                        t = min(
                            timeit.repeat(
                                lambda: deque(map(fn, repeat(s1, n), repeat(s2, n)), maxlen=0), number=1, repeat=3
                            )
                        )
                        if t * 1e9 > 200_000 or n > 100_000:
                            break
                        n *= 4
                    counts[key] = max(1, min(200_000, round(n * TARGET_NS / (t * 1e9))))
                    continue
                n = counts[key]
                best = min(
                    timeit.repeat(
                        lambda: deque(map(fn, repeat(s1, n), repeat(s2, n)), maxlen=0), number=1, repeat=repeat_n
                    )
                )
                times[key] = best / n * 1e9
            else:
                a, b = c["batch"]
                out = repr(list(map(fn, a, b)))
                if calibrate:
                    counts[key] = len(a)
                    continue
                best = min(timeit.repeat(lambda: deque(map(fn, a, b), maxlen=0), number=1, repeat=repeat_n))
                times[key] = best / len(a) * 1e9
            digests[key] = hashlib.sha256(out.encode()).hexdigest()[:12]
    return {"so": so, "sha": sha, "mtime": mtime, "times": times, "digests": digests, "counts": counts}


def quart(xs: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2:
        return xs[0], xs[0], xs[0]
    q = statistics.quantiles(xs, n=4)
    return q[0], q[1], q[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=[])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tags", default="", help="comma-separated tags; a case runs if it has any")
    ap.add_argument("--child")
    ap.add_argument("--cases")
    ap.add_argument("--counts")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--set", default="1")
    args = ap.parse_args()

    if args.child:
        print(json.dumps(child(args.child, args.cases, args.counts, args.repeat, args.calibrate)))
        return 0

    sys.path.insert(0, str(HERE))
    import cases as cases_mod

    cases = cases_mod.build() if args.set == "1" else cases_mod.build2()
    if args.tags:
        want = set(args.tags.split(","))
        cases = [c for c in cases if c["tags"] & want]
    out = args.out
    pkl = out.with_suffix(".cases.pkl")
    pkl.write_bytes(pickle.dumps(cases))
    builds = dict(b.split("=", 1) for b in args.builds)
    names = list(builds)
    py = sys.executable

    def quiet() -> None:
        # Another agent's rustc/cargo build on this machine wrecks timings: wait it out.
        for _ in range(600):
            busy = subprocess.run(["pgrep", "-x", "rustc"], capture_output=True).returncode == 0
            if not busy and os.getloadavg()[0] < 6.0:
                return
            time.sleep(2)

    def run_child(name: str, extra: list[str]) -> dict:
        quiet()
        cmd = [
            py,
            "-I",
            str(HERE / "run.py"),
            "--child",
            builds[name],
            "--cases",
            str(pkl),
            "--repeat",
            str(args.repeat),
            *extra,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode:
            raise SystemExit(f"child {name} failed:\n{r.stderr}")
        return json.loads(r.stdout)

    t0 = time.time()
    cal = run_child(names[0], ["--calibrate"])
    counts_path = out.with_suffix(".counts.json")
    counts_path.write_text(json.dumps(cal["counts"]))
    print(f"calibrated {len(cal['counts'])} keys in {time.time() - t0:.1f}s", file=sys.stderr)

    runs: dict[str, list[dict]] = {n: [] for n in names}
    meta: dict[str, dict] = {}
    for r in range(args.rounds):
        k = r % len(names)
        order = names[k:] + names[:k]
        if r % 2:
            order = order[::-1]
        for n in order:
            res = run_child(n, ["--counts", str(counts_path)])
            runs[n].append(res["times"])
            meta[n] = {"so": res["so"], "sha": res["sha"], "mtime": res["mtime"], "digests": res["digests"]}
        load = os.getloadavg()[0]
        print(f"round {r + 1}/{args.rounds} done at {time.time() - t0:.0f}s, load {load:.2f}", file=sys.stderr)

    # Correctness: every build must produce identical outputs.
    ref = meta[names[0]]["digests"]
    mism = {n: [k for k in ref if meta[n]["digests"].get(k) != ref[k]] for n in names[1:]}
    out.write_text(
        json.dumps({
            "builds": builds,
            "meta": meta,
            "runs": runs,
            "mismatch": mism,
            "rounds": args.rounds,
            "repeat": args.repeat,
        })
    )
    print(f"wrote {out}; output mismatches: { {n: len(v) for n, v in mism.items()} }", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
