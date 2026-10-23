# lev

## Introduction

`lev` is an extremely fast Python library for the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and similarity ratio, written in Rust. 🦀

## Installation

Install the latest version from [pypi](https://pypi.org/project/lev-rs/):

```bash
uv add lev-rs
```

or if you prefer slow:

```bash
pip install lev-rs
```

On x86-64, lev needs a CPU with SSE4.2 and POPCNT (x86-64-v2, any CPU from 2009 on).

## Usage

```python
import lev

lev.distance("kitten", "sitting")   # 3
lev.distance("résumé", "resume")    # 2
lev.distance("日本語", "日本")       # 1

lev.ratio("kitten", "sitting")      # 0.615...
lev.ratio("abc", "xyz")             # 0.0
lev.ratio("", "")                   # 1.0
```

For more details on the API see the [API Reference](api.md).

## Benchmarks

`lev` is benchmarked against the fastest Python Levenshtein libraries:
[rapidfuzz](https://github.com/rapidfuzz/RapidFuzz),
[editdistance](https://github.com/roy-ht/editdistance), and
[edlib](https://github.com/Martinsos/edlib). We excluded slower implementations like
[pylev](https://github.com/toastdriven/pylev/tree/main) and [python-Levenshtein](https://github.com/ztane/python-Levenshtein).

!!! example "Benchmarks"
    Benchmarks were run by the [`benchmark-plots`](https://github.com/KarelZe/lev/blob/main/.github/workflows/benchmark-plots.yml) workflow
    on a GitHub-hosted `ubuntu-24.04` runner (AMD EPYC 7763, x86-64, pinned to one core) using Python 3.13.
    The ASCII, Latin-1, CJK, and Emoji pairs are exactly 100 characters long;
    the realistic-text pair is natural-length prose.
    Results represent the total wall time for 1,000 repetitions using Python's `timeit`, best of 25 runs.
    To refresh the figures, run `gh workflow run benchmark-plots.yml`
    and commit the `benchmark-plots-linux-x86-64` artifact to `docs/assets/`.
    To reproduce locally, run [`uv run scripts/benchmark.py`](https://github.com/KarelZe/lev/blob/main/scripts/benchmark.py).
    The performance of each pull request is checked by the
    [A/B benchmark](https://github.com/KarelZe/lev/blob/main/.github/workflows/ab-benchmark.yml) workflow.

### ASCII

`lev` is significantly faster than the other libraries on 100-character ASCII strings.

![ASCII benchmark – light](assets/benchmark_ascii_light.svg#only-light){ width="75%" }
![ASCII benchmark – dark](assets/benchmark_ascii_dark.svg#only-dark){ width="75%" }


### Other Encodings

`lev` maintains its lead across all four CPython string-encoding kinds.

#### Latin-1


![Latin-1 benchmark – light](assets/benchmark_latin_1_light.svg#only-light){ width="75%" }
![Latin-1 benchmark – dark](assets/benchmark_latin_1_dark.svg#only-dark){ width="75%" }


#### CJK

![CJK benchmark – light](assets/benchmark_cjk_light.svg#only-light){ width="75%" }
![CJK benchmark – dark](assets/benchmark_cjk_dark.svg#only-dark){ width="75%" }

#### Emoji

![Emoji benchmark – light](assets/benchmark_emoji_light.svg#only-light){ width="75%" }
![Emoji benchmark – dark](assets/benchmark_emoji_dark.svg#only-dark){ width="75%" }

### Realistic Text

This pair is a short customer support message with a handful of natural typos.

![Realistic text benchmark – light](assets/benchmark_realistic_light.svg#only-light){ width="75%" }
![Realistic text benchmark – dark](assets/benchmark_realistic_dark.svg#only-dark){ width="75%" }
