# lev

High-performance Levenshtein distance and similarity ratio for Python, implemented in Rust.

## Introduction

<!-- Add a brief description of the library here -->

## Quickstart

```python
import lev

lev.distance("kitten", "sitting")   # 3
lev.distance("résumé", "resume")    # 2
lev.distance("日本語", "日本")       # 1

lev.ratio("kitten", "sitting")      # 0.769...
lev.ratio("", "")                   # 1.0
```

## Installation

```bash
uv add lev-rs
```

or

```bash
pip install lev-rs
```

## Benchmarks

<!-- Add benchmark results and charts here -->
