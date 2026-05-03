# lev

## Introduction

`lev` is an extremely fast Python library for the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and similarity ratio. Written in Rust. 🦀

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

## Benchmarks

<!-- Add benchmark results and charts here -->
