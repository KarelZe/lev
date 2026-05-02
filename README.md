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

## Code formatting

Rust sources are formatted with `rustfmt`. Run it manually with:

```bash
cargo fmt
```

The formatter is also enforced automatically via [prek](https://prek.j178.dev/) on every commit (pre-commit hook) and in CI via the `prek checks` GitHub Actions workflow.

## Docs

Build and serve the documentation locally (requires the extension to be compiled first):

```bash
uv sync
bash scripts/install_dev.sh
uv run --extra docs zensical serve
```

Open <http://localhost:8000> in your browser. The server reloads automatically on changes to `docs/` or `zensical.toml`.

## Benchmarks

```bash
uv run python scripts/benchmark.py
```

Produces two plots and a printed timing table:

- `benchmark_results.png` — all libraries compared on ASCII strings
- `benchmark_results_by_kind.png` — `lev` vs `rapidfuzz` across ASCII, Latin-1, CJK, and Emoji strings

## Releasing

Releases are driven by [commitizen](https://commitizen-tools.github.io/commitizen/) and automated via
[`.github/workflows/release.yml`](.github/workflows/release.yml).

### One-time setup

**Trusted publishing** — configure OIDC publishers on both registries so no API tokens are needed.
Go to the publishing settings page and add a pending publisher for each row:

| Registry | Project | Owner | Repo | Workflow | Environment |
|---|---|---|---|---|---|
| [PyPI](https://pypi.org/manage/account/publishing/) | `lev` | `KarelZe` | `lev` | `release.yml` | `pypi` |
| [Test PyPI](https://test.pypi.org/manage/account/publishing/) | `lev` | `KarelZe` | `lev` | `release.yml` | `test-pypi` |

**GitHub environments** — go to Settings → Environments and create two environments:

- `test-pypi`: no protection rules (publishes automatically on every tag)
- `pypi`: add yourself as a required reviewer so you can verify Test PyPI first

### Cutting a release

```bash
# 1. Bump version, update CHANGELOG.md, and create an annotated tag
cz bump

# 2. Push the bump commit and the new tag
git push && git push --tags
```

The tag push triggers the release workflow, which:

1. Builds wheels for Linux (x86_64 + aarch64, manylinux), Windows (x64), and macOS (universal2)
   across Python 3.10–3.13, plus a source distribution
2. Publishes automatically to **Test PyPI**
3. Waits for manual approval, then publishes to **PyPI**
4. Creates a **GitHub Release** with all wheels attached and auto-generated release notes
