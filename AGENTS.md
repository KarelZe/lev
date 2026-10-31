# AGENTS.md

## Project Overview
`lev` is a Python library exposing Levenshtein distance and similarity ratio.
The hot path is implemented in Rust and also published as the `lev-rs` crate.
The Rust API lives in `src/lib.rs`; the PyO3 bindings live in `src/python.rs`
behind the `python` Cargo feature (enabled by maturin via `pyproject.toml`), and
type stubs in `python/lev/_lev.pyi`. Performance is the primary design constraint —
prefer Rust-side optimizations over Python wrappers.

## Environment Setup
- Python >= 3.10
- Install: `uv sync`
- After editing Rust, rebuild the extension: `uv run maturin develop --release`

## Performance workflow

All performance changes follow this loop. One change per experiment,
measured on the same hardware, kept only if it wins.

1. Pin a baseline before any change:
   - Python-level:
     `uv run scripts/benchmark.py --save baselines/<name>.json`
     Add `--kind ascii` (or `latin1`, `cjk`, `emoji`, `realistic`) to baseline a
     single string kind instead of all five.
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

## Architecture-specific kernels
Kernels with target-specific forms live in `src/kernels/`: the reference form
in `portable.rs`, ISA-feature forms in `<target_arch>/<feature>.rs` (named after
the `target_feature` they enable, e.g. `x86_64/bmi1.rs`, `x86_64/avx2.rs`) and
core-tuned forms in `<target_arch>/<uarch>.rs` (named after the LLVM
`target-cpu` family, e.g. `aarch64/apple.rs`, `aarch64/neoverse.rs`). Each form
keeps the kernel's name and signature (`portable::multiword`,
`x86_64::bmi1::multiword`). Selection lives only in the dispatchers in
`kernels/mod.rs`: runtime detection (`is_x86_feature_detected!`) for ISA
features, `cfg` for core tuning. Every plain-Rust form is compiled on every
target (gate only `asm!`, `#[target_feature]` and intrinsics with `cfg`) and gets
a test in `kernels/tests.rs` against the portable form or the DP oracle.

## Testing
- Python: `pytest tests/`
- Rust: `cargo test` (pure-Rust build; the bindings are tested through pytest)
- Rust changes require a rebuild before pytest will pick them up. You can trigger a rebuild with `uv run maturin develop --release`.

## Benchmarking
- Compare against other libraries: `uv run scripts/benchmark.py`
- Compare across string lengths: `uv run scripts/benchmark_by_length.py`
- Compare across released versions: `uv run scripts/benchmark_by_version.py`
- Compare against other Rust crates (criterion, separate crate in `benches/compare`):
  `(cd benches/compare && cargo bench)`, then `uv run scripts/benchmark_rust.py`
  for the Markdown table and plots
- For performance-critical changes, run `benchmark.py` before and after,
  then diff `benchmark_results.json`.

## Documentation
- Serve locally: `uv run zensical serve`, then access http://localhost:8000

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

## Releasing
- The version lives only in `Cargo.toml`; maturin passes it to the Python package.
- PRs are squash-merged, so the PR title becomes the commit on `main` and must be
  a conventional commit (checked by `.github/workflows/pr-title.yml`).
- release-please keeps a release PR open that bumps `Cargo.toml` and `CHANGELOG.md`.
  Merging it tags the release, creates the GitHub Release and publishes to PyPI
  and crates.io.
- Never bump versions or push tags by hand. To rebuild an existing release, run
  the `release` workflow manually with its tag.

## Project Structure
- `src/` — Rust crate `lev-rs` (core library in `lib.rs`, PyO3 bindings in `python.rs`)
- `python/lev/` — Python package: `__init__.py` re-exports the private `_lev`
  extension, `_lev.pyi` holds its type stubs
- `tests/` — pytest suite
- `scripts/` — build and benchmark scripts
- `docs/` — Zensical documentation source
- `README.md` — GitHub landing page; `python/README.md` is the PyPI page and
  `src/README.md` the crates.io page (its examples run as doctests). Keep
  Python content out of the crate README and Rust content out of the PyPI one.

## PR Guidelines
- Keep diffs small and focused
- Add or update tests for changed code paths
- Update `docs/` and the READMEs when behavior or API changes

## Permissions

**Allowed without asking**
- Read files, run tests, run linters and formatters, run benchmarks
- Build the extension locally

**Ask first**
- Installing new packages (Python or Rust)
- Pushing to `main`
- Deleting files
- Modifying CI workflows or release configuration
