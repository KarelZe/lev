r"""
PGO training workload: exercise `lev.distance`/`lev.ratio` over a realistic corpus.

Run against an instrumented (`-Cprofile-generate`) build so the process exit
flushes `.profraw` profile data. Intentionally has no dependencies beyond
`lev` itself, so the training run stays fast and the resulting profile isn't
diluted by unrelated code paths pulled in by test/plotting libraries.

This module deliberately has **no** `uv run --script` shebang or PEP 723
header. Such a header would resolve `lev-rs` from the repo root and build a
fresh, *non-instrumented* extension into a throwaway environment, so the run
would silently collect zero samples. Instead, install the instrumented wheel
into an environment first and invoke this with that interpreter:

    RUSTFLAGS="-Cprofile-generate=$PGO_DIR" maturin build --release \\
        --strip=false --out dist-pgo-instrumented
    pip install --force-reinstall --no-deps dist-pgo-instrumented/*.whl
    LLVM_PROFILE_FILE="$PGO_DIR/train/lev-%p.profraw" python scripts/pgo_train.py

Setting `LLVM_PROFILE_FILE` is what keeps the training profile separate from
the `.profraw` files that build scripts and proc macros emit into `$PGO_DIR`
while they run during compilation. Merge only the training subdirectory.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from pgo_corpus import generate_pairs

import lev

# LLVM emits these sections into any binary built with `-Cprofile-generate`.
# The profiling runtime's own symbols are local rather than exported, so they
# are not reachable via `dlsym`; the section names are, and they are spelled
# identically in Mach-O and ELF.
_INSTRUMENTATION_MARKERS = (b"__llvm_prf_cnts", b"__llvm_prf_names")


def _extension_path() -> Path:
    """
    Locate the compiled `lev` extension module on disk.

    Returns:
        Filesystem path to the `lev.lev` shared object.

    """
    return Path(importlib.import_module("lev.lev").__file__)


def require_instrumented() -> Path:
    """
    Abort unless the imported `lev` was built with `-Cprofile-generate`.

    Without this guard a training run against an ordinary release build
    succeeds, prints a plausible pair count, and writes no profile data at
    all -- leaving the subsequent `-Cprofile-use` build to consume whatever
    unrelated `.profraw` files happen to be lying around.

    Returns:
        Filesystem path to the verified, instrumented extension module.

    Raises:
        SystemExit: If the extension carries no LLVM instrumentation.

    """
    path = _extension_path()
    blob = path.read_bytes()
    if not all(marker in blob for marker in _INSTRUMENTATION_MARKERS):
        raise SystemExit(
            f"error: {path} is not instrumented for PGO.\n"
            "  Build it with RUSTFLAGS='-Cprofile-generate=<dir>' and "
            "`maturin build --release --strip=false`, then install that wheel\n"
            "  into this environment before training. Running this script via "
            "`uv run` will NOT work: it rebuilds lev-rs without instrumentation."
        )
    return path


def main() -> int:
    """
    Call `lev.distance`/`lev.ratio` once per corpus pair.

    Returns:
        Process exit code (0 on success).

    """
    path = require_instrumented()
    destination = os.environ.get("LLVM_PROFILE_FILE", "<-Cprofile-generate default>")
    print(f"instrumented extension: {path}")
    print(f"profile destination:    {destination}")

    pairs = generate_pairs()
    for s1, s2 in pairs:
        lev.distance(s1, s2)
        lev.ratio(s1, s2)
    print(f"trained on {len(pairs)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
