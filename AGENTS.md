# AGENTS.md

## Project Overview
`lev` is a Python library exposing Levenshtein distance and similarity ratio.
The hot path is implemented in Rust via PyO3; Python bindings live in `src/lib.rs`
and type stubs in `lev.pyi`. Performance is the primary design constraint —
prefer Rust-side optimizations over Python wrappers.

## Environment Setup
- Python >= 3.10
- Install: `uv sync --all-extras`
- After editing Rust, rebuild the extension: `bash scripts/install_dev.sh`

## Testing
- Python: `pytest tests/`
- Rust: `cargo test`
- Rust changes require a rebuild before pytest will pick them up. You can trigger a rebuild with `bash scripts/install_dev.sh`.

## Benchmarking
- Install deps: `uv sync --extra benchmark`
- Compare against other libraries: `uv run python scripts/benchmark.py`
- Compare across string lengths: `uv run python scripts/benchmark_by_length.py`
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
