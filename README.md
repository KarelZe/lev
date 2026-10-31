# lev

`lev` is an extremely fast library for comparing strings, written in Rust. 🦀 It computes the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and a similarity ratio based on the [longest common subsequence](https://en.wikipedia.org/wiki/Longest_common_subsequence), for Python and Rust.

## Python

```bash
uv add lev-rs
```

```python
import lev

lev.distance("kitten", "sitting")  # 3
lev.distance("résumé", "resume")  # 2
lev.distance("日本語", "日本")  # 1

lev.ratio("kitten", "sitting")  # 0.615...
lev.ratio("abc", "xyz")  # 0.0
lev.ratio("", "")  # 1.0
```

Wheels are available for Python 3.10 through 3.15, including free-threaded Python (3.13t+), on Linux, macOS, and Windows. See the [package README](python/README.md) and the [API Reference](https://karelze.github.io/lev/api/) for details.

## Rust

```bash
cargo add lev-rs
```

```rust
assert_eq!(lev::distance("kitten", "sitting"), 3);
assert_eq!(lev::distance("résumé", "resume"), 2);
assert_eq!(lev::distance("日本語", "日本"), 1);

assert_eq!(lev::ratio("kitten", "sitting"), 0.6153846153846154);
assert_eq!(lev::ratio("abc", "xyz"), 0.0);
assert_eq!(lev::ratio("", ""), 1.0);
```

The crate requires Rust 1.91 or newer. See the [crate README](src/README.md) and [docs.rs](https://docs.rs/lev-rs) for details.

## Benchmarks

Against the fastest Python libraries, on a short customer support message with a few typos (total time for 1,000 calls, lower is better):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_realistic_dark.svg">
  <img alt="Python benchmark on realistic text" src="https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_realistic_light.svg">
</picture>

Against common Rust crates, across string lengths (lower is better):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_rust_length_dark.svg">
  <img alt="Rust benchmark across string lengths" src="https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_rust_length_light.svg">
</picture>

More string kinds and the method are in the docs, for [Python](https://karelze.github.io/lev/#benchmarks) and [Rust](https://karelze.github.io/lev/rust/#benchmarks). To reproduce them:

```bash
# Python: against other libraries, across string lengths, across released versions
uv run scripts/benchmark.py
uv run scripts/benchmark_by_length.py
uv run scripts/benchmark_by_version.py

# Rust: against other crates
(cd benches/compare && cargo bench) && uv run scripts/benchmark_rust.py
```

## Contact

Open an issue or email `github@markusbilz.com`.
