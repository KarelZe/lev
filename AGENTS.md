# AGENTS.md

## Project Overview
`lev` is a Python library exposing Levenshtein distance and similarity ratio.
The hot path is implemented in Rust via PyO3; Python bindings live in `src/lib.rs`
and type stubs in `lev.pyi`. Performance is the primary design constraint —
prefer Rust-side optimizations over Python wrappers.

## Environment Setup
- Python >= 3.10
- Install: `uv sync --all-extras`
- After editing Rust, rebuild the extension: `uv run maturin develop --release`

## Performance workflow

All performance changes follow this loop. One change per experiment,
measured on the same hardware, kept only if it wins.

1. Pin a baseline before any change:
   - Python-level:
     `uv run scripts/benchmark.py --save baselines/<name>.json`
     Add `--kind ascii` (or `latin1`, `cjk`, `emoji`) to baseline a single
     CPython string kind instead of all four.
   - Rust-level (if criterion benches exist for the change surface):
     `cargo bench -- --save-baseline <name>`
2. State a single, testable hypothesis before editing (e.g. "prefix/suffix
   trim removes ~30% of DP work on ASCII-100"). One change per experiment.
3. Re-run the same benchmark on identical hardware, quiet system, `taskset`
   or `sudo nice -n -20` where reasonable. Report median + IQR, not mean.
4. If the change is under 3% or within noise on every kind, REVERT.
   Do not commit tuning that only wins in one microbenchmark.
5. Verify correctness against the property tests and a diff test vs.
   rapidfuzz / edlib on random inputs before keeping any win.
6. Check codegen for the hot loop with `cargo asm` (or `--emit=asm`) when
   claiming SIMD or auto-vectorization wins.
7. Never edit the release profile in `Cargo.toml` (LTO, codegen-units,
   panic) as part of an algorithmic experiment. Separate PR.

Profiling is a read-only step and is delegated to the `profile-runner`
sub-agent (`.claude/agents/profile-runner.md`). It uses samply for the
Python-level benchmark and cargo flamegraph for the criterion benches;
neither needs `--profile-time`.

## Testing
- Python: `pytest tests/`
- Rust: `cargo test`
- Rust changes require a rebuild before pytest will pick them up. You can trigger a rebuild with `uv run maturin develop --release`.

## Benchmarking
- Scripts are self-contained (PEP 723 inline deps); no install step needed.
- Compare against other libraries: `uv run scripts/benchmark.py`
- Compare across string lengths: `uv run scripts/benchmark_by_length.py`
- For performance-critical changes, run `benchmark.py` before and after,
  then diff `benchmark_results.json`.

## Documentation
- Serve locally: `uv run --extra docs zensical serve`, then access http://localhost:8000

## Code Style
- Python: `ruff format .` and `ruff check . --fix`
- Rust: `cargo fmt`
- Pre-commit hooks (via prek): `prek install` once, then `prek run --all-files`

## Before Committing
- All Python and Rust tests pass
- `prek run --all-files` passes
- Commit message follows Conventional Commits with a trailing emoji:
  - `feat: … ✨`  `fix: … 🐛`  `perf: … ⚡`
  - `docs: … 📝`  `test: … ✅`  `style: … 💄`  `build: … 🔧`

## Project Structure
- `src/` — Rust crate (core library + PyO3 bindings)
- `lev.pyi` — Python type stubs
- `tests/` — pytest suite
- `scripts/` — build and benchmark scripts
- `docs/` — Zensical documentation source

## PR Guidelines
- Keep diffs small and focused
- Add or update tests for changed code paths
- Update `docs/` and `README.md` when behavior or API changes

## Permissions

**Allowed without asking**
- Read files, run tests, run linters and formatters, run benchmarks
- Build the extension locally

**Ask first**
- Installing new packages (Python or Rust)
- Pushing to `main`
- Deleting files
- Modifying CI workflows or release configuration
