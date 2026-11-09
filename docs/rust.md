# Rust

`lev` is published as the [`lev-rs`](https://crates.io/crates/lev-rs) crate. It has no dependencies and is imported as `lev`.

## Installation

```bash
cargo add lev-rs
```

The crate requires Rust 1.91 or newer.

## Usage

```rust
assert_eq!(lev::distance("kitten", "sitting"), 3);
assert_eq!(lev::distance("résumé", "resume"), 2);
assert_eq!(lev::distance("日本語", "日本"), 1);

assert_eq!(lev::ratio("kitten", "sitting"), 0.6153846153846154);
assert_eq!(lev::ratio("abc", "xyz"), 0.0);
assert_eq!(lev::ratio("", ""), 1.0);
```

Both functions take `&str` and count Unicode scalar values (`char`s). See [docs.rs](https://docs.rs/lev-rs) for the API.

## Benchmarks

`lev` is compared against common Levenshtein crates from crates.io:
[rapidfuzz](https://crates.io/crates/rapidfuzz),
[strsim](https://crates.io/crates/strsim),
[triple_accel](https://crates.io/crates/triple_accel),
[stringmetrics](https://crates.io/crates/stringmetrics),
[edit-distance](https://crates.io/crates/edit-distance),
[levenshtein](https://crates.io/crates/levenshtein), and
[textdistance](https://crates.io/crates/textdistance).
The benchmark uses [criterion](https://crates.io/crates/criterion) and the same string pairs as the [Python benchmark](index.md#benchmarks).
Each cell is the median time of one call to `distance(a, b)` on `&str` inputs (lower is better).
*Mixed* runs 1,000 different short ASCII pairs (4 to 64 characters, about 15% edits) per iteration and reports the time per pair,
so no crate profits from a branch predictor trained on a single pair.

| crate | ASCII | Latin-1 | CJK | Emoji | Realistic | Mixed (per pair) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`lev`** | **252 ns** | **470 ns** | **779 ns** | **909 ns** | **799 ns** | **87 ns** |
| `rapidfuzz` | 985 ns | 1,140 ns | 1,291 ns | 1,559 ns | 2,613 ns | 222 ns |
| `strsim` | 9,626 ns | 9,742 ns | 13.3 µs | 16.2 µs | 45.2 µs | 1,602 ns |
| `stringmetrics` | 10.5 µs | 14.0 µs | 12.2 µs | 19.1 µs | 59.5 µs | 1,648 ns |
| `levenshtein` | 11.2 µs | 11.1 µs | 12.6 µs | 14.8 µs | 55.7 µs | 1,709 ns |
| `textdistance` | 11.4 µs | 11.4 µs | 11.5 µs | 11.5 µs | 55.2 µs | 1,730 ns |
| `edit-distance` | 13.6 µs | 13.7 µs | 14.4 µs | 16.2 µs | 68.7 µs | 1,995 ns |
| `triple_accel` | 29.6 µs | n/a | n/a | n/a | 148.0 µs | 4,622 ns |

Runtime against string length, for ASCII pairs with every fifth character substituted:

![Rust length benchmark – light](assets/benchmark_rust_length_light.svg#only-light){ width="75%" }
![Rust length benchmark – dark](assets/benchmark_rust_length_dark.svg#only-dark){ width="75%" }

!!! note
    Measured on an Apple M2 Pro with rustc 1.95.0, `lev-rs` 0.6.0 and the crate versions in
    [`benches/compare/Cargo.toml`](https://github.com/KarelZe/lev/blob/main/benches/compare/Cargo.toml).
    Every crate is called directly, so the compiler can inline it as in user code, and must return the same
    distance as `lev` before it is timed. Every crate gets the same fat LTO build profile. Medians come from
    criterion's 100 samples (20 at 4096 characters) after 3 s of warm-up and 5 s of measurement;
    the 25th to 75th percentile range stays within 1% of the median for single pairs and 6% for *Mixed*.
    All crates count Unicode scalar values (`char`s), except `triple_accel`, which compares bytes and
    therefore only runs on pure-ASCII pairs. `triple_accel` vectorizes with AVX2 or SSE4.1 only;
    on AArch64 it falls back to scalar code, so expect it to be faster on x86-64.

### Benchmark Scripts

The Rust comparison lives in its own crate, so the competitors never become dependencies of `lev-rs`:

```bash
cd benches/compare
cargo bench -- --warm-up-time 3 --measurement-time 5  # as published; `cargo bench -- kind/` for one group
cd ../..
uv run scripts/benchmark_rust.py  # Markdown table, JSON and length plots
```
