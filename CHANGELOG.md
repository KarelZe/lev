## [0.6.0](https://github.com/KarelZe/lev/compare/0.5.1...0.6.0) (2026-10-30)


### Features

* publish the crate to crates.io as lev-rs ✨ ([3fb0e4c](https://github.com/KarelZe/lev/commit/3fb0e4cde0327110f76c40c6b436c4b965d3ba8a))


### Performance

* affix early-exit, adaptive hash sizing, and banded loop peeling ⚡ ([#241](https://github.com/KarelZe/lev/issues/241)) ([8782bb7](https://github.com/KarelZe/lev/commit/8782bb74144bf51f56507fbe952a8285b3877c8e))
* preallocate transcoded code units in the Rust API ⚡ ([cea86df](https://github.com/KarelZe/lev/commit/cea86dfb65670128f194fbd5122463809d693ca2))

## [0.5.1](https://github.com/KarelZe/lev/compare/0.5.0...0.5.1) (2026-10-27)


### Performance

* 6-cycle single-word Myers step on Apple silicon ⚡ ([#221](https://github.com/KarelZe/lev/issues/221)) ([8769302](https://github.com/KarelZe/lev/commit/8769302b8be3a434fa398e028738ebf36e5166a3))
* expand TINY_M to 16 with 128-bit SIMD pattern matching ⚡ ([#229](https://github.com/KarelZe/lev/issues/229)) ([1ce4738](https://github.com/KarelZe/lev/commit/1ce4738c1c9705f4d154b00d535cd97cea309cd9))
* jump matching runs in mbleven with word-level common prefix ⚡ ([#222](https://github.com/KarelZe/lev/issues/222)) ([28a3324](https://github.com/KarelZe/lev/commit/28a3324bf17f8c28cca4b0fd2d7762fa5eb08377))
* populate multi-word hash table in a single probe pass ⚡ ([#220](https://github.com/KarelZe/lev/issues/220)) ([4215b4a](https://github.com/KarelZe/lev/commit/4215b4a632fd1d6805442c8ac4a02de2b7d3acea))
* transfer Myers step algebraic rewrite to x86-64 and Neoverse ⚡ ([#231](https://github.com/KarelZe/lev/issues/231)) ([4c18cd7](https://github.com/KarelZe/lev/commit/4c18cd7df4656db492a18a27b445c685b156db9b))
* widen small_ub chunk scan to 32 bytes on AVX2 in x86-64-v2 copy ⚡ ([#228](https://github.com/KarelZe/lev/issues/228)) ([8b5fd15](https://github.com/KarelZe/lev/commit/8b5fd15a1dbea03db07dfa9f767d0139636f6ccc))

## [0.5.0](https://github.com/KarelZe/lev/compare/0.4.1...0.5.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* `lev.ratio` returns different values. Substitutions now cost 2 instead of 1, e.g. `ratio("kitten", "sitting")` changes from 0.769 to 0.615 and `ratio("abc", "xyz")` from 0.5 to 0.0.

### Bug Fixes

* replace implementation of ratio with more conventional definition 🐞 ([#168](https://github.com/KarelZe/lev/issues/168)) ([a1c61d4](https://github.com/KarelZe/lev/commit/a1c61d486f432f8aa11273718b0356eb40aa6eac))
* restore long similar-pair speed of 0.4.1 🐛 ([#207](https://github.com/KarelZe/lev/issues/207)) ([2fec76f](https://github.com/KarelZe/lev/commit/2fec76fcd1ba09f8a6edbd57a0a5a6addef125ff))


### Performance

* Apple silicon kernels: pinned Myers step, funnel-shift form, strips ⚡ ([0b42a1c](https://github.com/KarelZe/lev/commit/0b42a1ceee2d2ba2b5ecf7767ddcdd025b5eaa4b))
* call distance and ratio through fast-call entry points ⚡ ([a91b7d6](https://github.com/KarelZe/lev/commit/a91b7d612caff830a52e3db221ef47aa825c32c0))
* compare 32 bytes per branch when stripping common affixes ⚡ ([ffb4e99](https://github.com/KarelZe/lev/commit/ffb4e9996f23844f6616a18c17e89030b9bbda59))
* compare mixed-kind strings 16 code units per branch ⚡ ([32708f4](https://github.com/KarelZe/lev/commit/32708f4c1ae0e9974c51fdb507f346fc45b3fb02))
* direct peq table for long mixed-kind pairs with a UCS-1 side ⚡ ([#196](https://github.com/KarelZe/lev/issues/196)) ([0ac9c5e](https://github.com/KarelZe/lev/commit/0ac9c5e4fbc0665b812b7c5319093415b8aeb501))
* LCS full matrix in strips on x86-64 ⚡ ([#209](https://github.com/KarelZe/lev/issues/209)) ([c568b0a](https://github.com/KarelZe/lev/commit/c568b0affa4f03c63f420499c6907b8eb89b2aa0))
* Neoverse-tuned AArch64 kernels ⚡ ([#208](https://github.com/KarelZe/lev/issues/208)) ([a32e48d](https://github.com/KarelZe/lev/commit/a32e48df81f9fb6d8aa6cd8f5113015330d4ad8c))
* x86-64 multi-word kernel with andn and popcnt ⚡ ([#202](https://github.com/KarelZe/lev/issues/202)) ([865ed7b](https://github.com/KarelZe/lev/commit/865ed7b32b56b6ab8cf03176a96d72514e7ebf44))

## [0.4.1](https://github.com/KarelZe/lev/compare/0.4.0...0.4.1) (2026-10-22)


### Bug Fixes

* publish CPython 3.14 and 3.15 wheels for macOS and Windows ([#194](https://github.com/KarelZe/lev/issues/194)) ([cd5a6c2](https://github.com/KarelZe/lev/commit/cd5a6c247ed7ce1882da6fb6f8207855adda1aa7))

## [0.4.0](https://github.com/KarelZe/lev/compare/0.3.0...0.4.0) (2026-10-14)


### Features

* support free-threaded Python builds ✨ ([#181](https://github.com/KarelZe/lev/issues/181)) ([8fa6d4c](https://github.com/KarelZe/lev/commit/8fa6d4cb768ea75c58fcb061d09367ffb0e523fe))

## [0.3.0](https://github.com/KarelZe/lev/compare/0.2.6...0.3.0) (2026-10-10)


### Features

* support Python 3.15 ✨ ([#178](https://github.com/KarelZe/lev/issues/178)) ([7915158](https://github.com/KarelZe/lev/commit/7915158d287a550d4ad7882107842e18d3ff4024))

## [0.2.6](https://github.com/KarelZe/lev/compare/0.2.5...0.2.6) (2026-10-03)


### Performance

* drop redundant masking in the Hyyrö loop ⚡ ([#175](https://github.com/KarelZe/lev/issues/175)) ([0cef08a](https://github.com/KarelZe/lev/commit/0cef08a23d6a7d134b6193fea18758362e3f3569))
* size the peq table of very long ascii patterns to 128 rows ⚡ ([#169](https://github.com/KarelZe/lev/issues/169)) ([2f3ab4e](https://github.com/KarelZe/lev/commit/2f3ab4eac6b58ba04bc79ea84cd3b65e9758666c))
* skip redundant peq table initialisation ⚡ ([#173](https://github.com/KarelZe/lev/issues/173)) ([80513ff](https://github.com/KarelZe/lev/commit/80513ff0845eed284bfbe2d92163d1aef7c21ac7))

## 0.2.5 (2026-09-03)

- build: move dev and docs to project dependencies🔧 (#160)

## 0.2.4 (2026-08-25)

### Perf

- improve performance on long cjk and emoji strings🚀 (#136)

## 0.2.3 (2026-07-09)

### Fix

- correct typo in project.description🐞 (#106)

### Perf

- improve performance on exceptionally long strings🚀 (#131)
- improve speed for >= python 3.14🚀 (#129)

## 0.2.2 (2026-05-05)

### Perf

- simplify and improve implementation of hot path🚀 (#100)

## 0.2.1 (2026-05-03)

### Fix

- align version of crate with python package version🐞 (#84)

## 0.2.0 (2026-05-03)

### Feat

- add official support for py 3.14✨ (#81)

## 0.1.0 (2026-05-03)

### Feat

- add initial rust implementation for levenshtein distance🦀
- add basic benchmarking + rust project
- add basic python setup 🐍

### Fix

- make arguments to distance and ratio positional only🐞

### Perf

- improve performance on very short strings🚀 (#69)
- improve performance for cjk and emoji strings🚀 (#65)
- improve performance on long strings🚀 (#61)
- improve performance of ascii and unicode strings🚀 (#54)
