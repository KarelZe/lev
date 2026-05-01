# lev

High-performance Levenshtein distance and similarity ratio for Python, implemented in Rust 🦀.

## Usage

```python
import lev

lev.distance("kitten", "sitting")   # 3
lev.distance("résumé", "resume")    # 2
lev.distance("日本語", "日本")       # 1

lev.ratio("kitten", "sitting")      # 0.769...
lev.ratio("", "")                   # 1.0
```

Both functions accept any Python `str` regardless of script or encoding.

## Development build

Dependencies: Rust toolchain, [uv](https://docs.astral.sh/uv/), and [maturin](https://www.maturin.rs/).

```bash
# Install Python dependencies (includes maturin)
uv sync

# Build the release wheel and hot-swap the .so into the local venv.
# Use this instead of `maturin develop` — uv's version-based wheel cache
# skips reinstalls when the version string has not changed.
bash scripts/install_dev.sh
```

Run the test suite:

```bash
cargo test
```

## Benchmarks

```bash
uv run python scripts/benchmark.py
```

Produces two plots and a printed timing table:

- `benchmark_results.png` — all libraries compared on ASCII strings
- `benchmark_results_by_kind.png` — `lev` vs `rapidfuzz` across ASCII, Latin-1, CJK, and Emoji strings
