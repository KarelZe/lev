# lev

`lev` is an extremely fast Python library for the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and similarity ratio, written in Rust. 🦀

## Installation

```bash
uv add lev-rs
```

or if you prefer slow:

```bash
pip install lev-rs
```

## Usage

```python
import lev

lev.distance("kitten", "sitting")   # 3
lev.distance("résumé", "resume")    # 2
lev.distance("日本語", "日本")       # 1

lev.ratio("kitten", "sitting")      # 0.769...
lev.ratio("", "")                   # 1.0
```

For more details on the API see the [API Reference](https://karelze.github.io/lev/api/).

## Benchmarks

`lev` is benchmarked against the fastest Python Levenshtein libraries:
[rapidfuzz](https://github.com/rapidfuzz/RapidFuzz),
[editdistance](https://github.com/roy-ht/editdistance), and
[edlib](https://github.com/Martinsos/edlib). We excluded slower implementations like
[pylev](https://github.com/toastdriven/pylev/tree/main) and [python-Levenshtein](https://github.com/ztane/python-Levenshtein).

> Benchmarks were run on an Apple Mac Mini M2 Pro (macOS 26.2) using Python 3.13. Each string pair is exactly 100 characters long. Results represent the total wall time for 1,000 repetitions using Python's `timeit`. To reproduce, run [`uv run python scripts/benchmark.py`](https://github.com/KarelZe/lev/blob/main/scripts/benchmark.py).

### ASCII

`lev` is significantly faster than the other libraries on 100-character ASCII strings.

![ASCII benchmark – dark](https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_ascii_dark.svg)

### Other Encodings

`lev` maintains its lead across all four CPython string-encoding kinds.

#### Latin-1

![Latin-1 benchmark – dark](https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_latin_1_dark.svg)

#### CJK

![CJK benchmark – dark](https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_cjk_dark.svg)

#### Emoji

![Emoji benchmark – dark](https://raw.githubusercontent.com/KarelZe/lev/main/docs/assets/benchmark_emoji_dark.svg)

## Contact

To get in contact, please open an issue or contact me via `github@markusbilz.com`.
