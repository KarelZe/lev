## 0.2.5 (2026-09-03)

- build: move dev and docs to project dependencies🔧 (#160)

## [0.3.0](https://github.com/KarelZe/lev/compare/0.2.6...0.3.0) (2026-10-10)


### Features

* support Python 3.15 ✨ ([#178](https://github.com/KarelZe/lev/issues/178)) ([7915158](https://github.com/KarelZe/lev/commit/7915158d287a550d4ad7882107842e18d3ff4024))

## [0.2.6](https://github.com/KarelZe/lev/compare/0.2.5...0.2.6) (2026-10-03)


### Performance

* drop redundant masking in the Hyyrö loop ⚡ ([#175](https://github.com/KarelZe/lev/issues/175)) ([0cef08a](https://github.com/KarelZe/lev/commit/0cef08a23d6a7d134b6193fea18758362e3f3569))
* size the peq table of very long ascii patterns to 128 rows ⚡ ([#169](https://github.com/KarelZe/lev/issues/169)) ([2f3ab4e](https://github.com/KarelZe/lev/commit/2f3ab4eac6b58ba04bc79ea84cd3b65e9758666c))
* skip redundant peq table initialisation ⚡ ([#173](https://github.com/KarelZe/lev/issues/173)) ([80513ff](https://github.com/KarelZe/lev/commit/80513ff0845eed284bfbe2d92163d1aef7c21ac7))

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
