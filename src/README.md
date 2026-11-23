# lev-rs

`lev-rs` is an extremely fast Rust crate for comparing strings. It computes the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and a similarity ratio based on the [longest common subsequence](https://en.wikipedia.org/wiki/Longest_common_subsequence). The crate has no dependencies and is imported as `lev`.

## Usage

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

Both functions take `&str` and count Unicode scalar values (`char`s), so an accented letter or an emoji is one character. `ratio` is `2 · LCS / (|s1| + |s2|)`, the indel similarity, and matches `rapidfuzz`'s normalized indel similarity.

Under the hood, `lev` strips common prefixes and suffixes, then runs the bit-parallel algorithms of Myers (1999) and Hyyrö (2003), with a diagonal band for long, similar strings. On x86-64 the hot path is compiled a second time for x86-64-v2 and picked at run time, so the default target needs no `target-cpu` flags.

See [docs.rs](https://docs.rs/lev-rs) for the API. The crate requires Rust 1.91 or newer.

## Benchmarks

`lev` is compared against common Levenshtein crates: [rapidfuzz](https://crates.io/crates/rapidfuzz), [strsim](https://crates.io/crates/strsim), [triple_accel](https://crates.io/crates/triple_accel), [stringmetrics](https://crates.io/crates/stringmetrics), [edit-distance](https://crates.io/crates/edit-distance), [levenshtein](https://crates.io/crates/levenshtein), and [textdistance](https://crates.io/crates/textdistance). Each cell is the median time of one `distance(a, b)` call, measured with [criterion](https://crates.io/crates/criterion) (lower is better). *Mixed* is the time per pair over 1,000 different short ASCII pairs.

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_rust_length_dark.svg">
  <img alt="Benchmark across string lengths" src="https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_rust_length_light.svg">
</picture>

> Measured on an Apple M2 Pro with rustc 1.95.0. `triple_accel` compares bytes, so it only runs on pure-ASCII pairs, and vectorizes on x86-64 only. See the [benchmark details](https://karelze.github.io/lev/rust/#benchmarks) for the method, or run [`benches/compare`](https://github.com/KarelZe/lev/tree/main/benches/compare) yourself.

## Related

The same code backs the [`lev-rs`](https://pypi.org/project/lev-rs/) Python package.

## License

`lev-rs` is licensed under the [MIT license](https://github.com/KarelZe/lev/blob/main/LICENSE).
