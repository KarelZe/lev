#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["lev-rs"]
#
# [tool.uv.sources]
# lev-rs = { path = "..", editable = true }
# ///
"""
PGO training workload: exercise `lev.distance`/`lev.ratio` over a realistic corpus.

Run against an instrumented (`-Cprofile-generate`) build so the process exit
flushes `.profraw` profile data. Intentionally has no dependencies beyond
`lev` itself, so the training run stays fast and the resulting profile isn't
diluted by unrelated code paths pulled in by test/plotting libraries.

Usage:
    RUSTFLAGS="-Cprofile-generate=$PGO_DIR" uv run maturin develop --release
    uv run scripts/pgo_train.py
"""

from __future__ import annotations

from pgo_corpus import generate_pairs

import lev


def main() -> int:
    """
    Call `lev.distance`/`lev.ratio` once per corpus pair.

    Returns:
        Process exit code (0 on success).

    """
    pairs = generate_pairs()
    for s1, s2 in pairs:
        lev.distance(s1, s2)
        lev.ratio(s1, s2)
    print(f"trained on {len(pairs)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
