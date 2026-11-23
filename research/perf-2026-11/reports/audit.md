# Layer audit: which optimizations of lev hamper performance (Apple M2 Pro, CPython 3.13 / 3.14)

Scope: origin/main 5a6859c (lev 0.6.1), copied from `main-src`. One build per suspect with exactly that
layer disabled or re-gated, all built from copies under `scratchpad/audit/`, measured interleaved on this
machine. Nothing was built or changed in `main-src` or in the main checkout.

All paths below are under
`/private/tmp/claude-501/-Users-markusbilz-Documents-git-lev/33906bae-f780-4c7b-8ea7-23a2aa9e4dd7/scratchpad/audit/`.

## TL;DR

- Two layers lose on realistic, varied inputs and should be re-gated:
  1. **Adaptive single-word hash sizing (#241, 64 slots for m <= 32)**: on realistic CJK / Japanese
     batches with 17-40 character patterns, always using 128 slots is **21-29% faster** on dissimilar
     pairs and 10-12% faster on related pairs. The 64-slot table only wins on a single repeated pair
     (+6..13%), which is what test_bench and CodSpeed measure, and on contiguous alphabets (emoji +5..7%,
     Cyrillic neutral).
  2. **mbleven for ub = 3 with diff <= 1 (7 or 6 edit scripts) on patterns of 17-32 units**: skipping it
     (falling through to the bit-parallel kernel) is **24-42% faster** on that class and 35% faster on a
     realistic batch of 3-word names with 3 scattered substitutions. No regressions anywhere.
- Two layers have measurable failure costs worth a narrower gate or a better give-up rule:
  3. **1-word small band (#260) when the length difference s is 32-63**: the band cannot give up before
     column s, so dissimilar pairs with a large length difference run most of the band first.
     Gating it to s < 32 (rung to s < 96) gives -7% on a realistic dissimilar-sentence batch and -16% on
     dissimilar sentences differing by 30-60 characters, but costs +29..32% on insertion-heavy similar
     pairs with s >= 32. Medium confidence; a give-up check that works for j < s would get both.
  4. **LargeCtx straight-to-full path without precomputed row offsets**: for hash-indexed (UCS-2/4)
     patterns of 513-704 units, or n >> m, building the `bases` vector as the band path already does is
     **17-27% faster** (distance and ratio). For ASCII it is neutral to +3%, so gate it to hash tables.
- Near-miss costs of the LargeCtx bands (Levenshtein band +12..45%, certified LCS band +23..44% at
  m = 800-1200 with 10-20% edits, +14..22% on realistic long related text) are real but they are not
  fixed by a simple re-gate: two variants (relaxed second pass, LCS gate at 1/5 of the words) only trade
  wins for losses. Keep, and give these bands a pace-based give-up.
- Everything else measured pays its way on this machine: table-free NEON masks (#266), SWAR for
  m <= 8, the Apple `asm!` step, TINY_M = 16 for wide units, mbleven overall, the small_ub gate, the
  speculative bands overall, LargeCtx bands overall.
- **Methodology finding**: benchmarks that repeat one pair (test_bench, CodSpeed, `ab_benchmark.py`)
  systematically favor branchy designs, because the branch predictor learns that pair's branch
  sequence. Three layers flip sign between repeated pairs and varied batches (hash sizing, TINY_M for
  wide units, mbleven on short Latin-1). test_bench needs varied-pair batch cases.
- R1 (kitten/sitting, ~35 ns on a quiet P-core): about **50-55% is fixed call overhead** (about half
  CPython's own vectorcall + map/deque floor, half the PyO3 trampoline and argument extraction), 45-50%
  is lev's work. `ratio` pays 5-6 ns more than `distance` for the float return.

## Method

### Builds (one change each, patches in `patches/<name>.diff`)

| build | change vs base |
|---|---|
| base, base2 | origin/main 5a6859c, same wheel installed twice (A/A noise floor) |
| tinywide8 | wide (UCS-2/4) patterns of 9-16 units leave the tiny path (u8 keeps TINY_M = 16) |
| nombleven | `Lev::MBLEVEN = false` (no small_ub gate, no mbleven) |
| mb3gate | skip mbleven when ub == 3 and m <= 64 |
| mb3gate32 | skip mbleven when ub == 3, n - m <= 1 and m <= 32 |
| noband | `Lev::SMALL_BAND = false` (no #260 bands) |
| bands32 | 1-word band only when s = n - m < 32, 2-word rung only when s < 96 |
| notablefree | no NEON table-free match masks (#266) |
| hash128 | single-word hash always 128 slots (undo #241's 64-slot table for m <= 32) |
| noswar | u8 tiny patterns of 1-8 bytes use the NEON compare instead of SWAR |
| nolargeband | LargeCtx Levenshtein and LCS go straight to the full matrix |
| large2pass | after a failed first LargeCtx pass, allow the next pass up to 2/3 of the words |
| lcsgate5 | certified LCS band only when a pass covers <= 1/5 of the words |
| basesalways | LargeCtx builds the `bases` row-offset vector also on its straight-to-full path |
| noasmstep | Apple single-word step in plain Rust (Neoverse form) instead of the pinned `asm!` |
| r1probe | (probe only) PyO3 entry points return right after argument extraction |

Wheels: `/Users/markusbilz/Documents/git/lev/.venv/bin/maturin build --release` per copy (shared
`CARGO_TARGET_DIR=audit/target`), installed with `uv pip install --no-deps --target inst/<name>`
(`inst314/` for 3.14). Every harness child prints the loaded `.so` path and its sha256 prefix; all
builds have distinct hashes (base c5514ceeac3763d5, base2 identical), see the headers of each results
file.

### Harnesses (`harness/`)

- `make_variants.py`: creates each variant copy from `base/` with exact-match string edits and writes
  `patches/<name>.diff`.
- `cases.py`: three case sets. Set 1 (171 cases, 289 case-function combos): 19 realistic **varied-pair
  batches** (`R:`; 60-3000 distinct pairs per batch: dictionary words and typos, full names and typos,
  Latin-1 accented names vs ASCII, product titles dissimilar/related, addresses with abbreviations,
  English-like sentences 60-200 chars dissimilar / 10 / 25 / 40% word edits / 3 typos / 3 scattered
  substitutions, CJK 10-40 dissimilar/related, CJK names 2-4, mixed-kind quotes, long related text
  600-1200), adversarial single pairs (`A:`) per layer, and all 45 `tests/test_bench.py` cases (`T:`).
  Set 2 (102 cases): mbleven classes (ub, diff) at m = 24/48/64/100 for ASCII and CJK, band pairs with
  large s, insertion-heavy pairs, LargeCtx near-miss pairs at 800-2048, Zipf-distributed CJK, Japanese
  kana/kanji, Cyrillic, emoji and mixed batches. Set 3: LargeCtx straight-to-full pairs.
- `run.py` + `report.py`: one subprocess per build per round, rotating order, best of `--repeat`
  batches with call counts calibrated once on base; also hashes every result so all builds are checked
  to produce identical outputs.
- `ab2.py`: **in-process paired A/B**. All builds' extensions are loaded into one process (distinct
  `.so` files, distinct PyO3 statics); per case, batches alternate between builds in rotating order and
  the per-round ratio of best times is taken; each round is a fresh process. Report: median ratio over
  rounds and its IQR. Built after the cross-process runs became unusable (see Noise).
- `r1.py`: R1 decomposition (operator.is_ floor, r1probe, tiny inputs) on 3.13 and 3.14.
- `difftest.py`: 30,000 random pairs per build (ASCII, CJK, uniform CJK, Cyrillic, emoji, tiny alphabet;
  m 1-1300; spread substitutions, mutations, insertion bursts, dissimilar, mixed kinds) against
  rapidfuzz `Levenshtein.distance` and `Indel.normalized_similarity`.
- Also ran the repo's `main-src/scripts/ab_benchmark.py --a inst/base --b inst/<v>` for the two top
  candidates.

### Noise

- `uptime` load was 3-4.6 during the first full run (`results/full1.json`, 9 rounds x 11 repeats, 10
  builds): the A/A build stayed within 0.974-1.024 on all 289 combos (geomean 0.999). These numbers are
  reliable at the 3% level.
- From about 23:45 the machine got 2-3x slower in absolute terms (kitten 35 ns -> 100-120 ns; another
  agent's rustc, Firefox, Google Drive and mediaanalysisd at ~90% each, user idle) and the cross-process
  timings became bimodal: whole child processes ran ~8% or ~17% slower. `results/full2*.json` show A/A
  ratios of 0.75-1.09 and are **discarded**; one `ab_benchmark.py` run of mb3gate32 showed every case
  1.14-1.17x "slower" (a pure artifact; the rerun gave geomean 0.994 with 25-35% IQRs).
- The in-process harness cancels this: A/A within 0.99-1.01 (a handful at 1.04) in all `ab2_*` runs
  despite load 4-6. All set-2 and set-3 conclusions below come from `ab2.py`. Absolute ns in the `ab2`
  files are 2-3x the quiet values; only ratios are used.
- Before each child, the harnesses wait while any `rustc` is running (other agents' builds).

### Correctness of the recommended variants

- All builds produced identical results to base on every case of every run (`mismatch={}` /
  `mismatches []` in all results files).
- `cargo test --release`: 48 passed for mb3gate32, hash128, bands32 and base.
- `pytest tests/test_lev.py` against each installed wheel (lev imported from `inst/<v>`): 241 passed
  for mb3gate32, hash128, bands32.
- `difftest.py`: 0 mismatches in 30,000 pairs each for mb3gate32, hash128, bands32, basesalways.

## Inventory of layers and gates (main-src)

| # | layer | gate | where |
|---|---|---|---|
| 1 | PyO3 entry, `pass_module`, `&Bound<PyString>` extraction, identity `is` shortcut, `view()` (3.14+: header decode) | every call | python.rs:118-147, 184-238 |
| 2 | kind dispatch, swap to m <= n, m == 0 | every call | lib.rs:370-410 |
| 3 | affix strip: 32-byte blocks, 8-byte words, bytes (same kind); 16-unit blocks (mixed); early exit when the prefix covers the pattern | every call | lib.rs:422-602 |
| 4 | tiny path: u8 m <= 8 SWAR, u8 9-16 NEON/SSE2, wide 1-16 const-N compares; out of line | m <= 16 (TINY_M) | lib.rs:613-816 |
| 5 | small_ub gate (NEON 16-byte blocks, abort past 3) + mbleven (1/1/3/2/1/7/6/3/1 scripts by (ub, diff)) | Lev, 17 <= m <= 512, n - m <= 3, ub <= 3 | lib.rs:854-1141 |
| 6 | single word u8: NEON table-free masks (Apple) / AVX2 (x86 V2, m <= 32) / 128-256 row table | 17 <= m <= 64; table-free when n <= 8m | kernels/mod.rs:151-195 |
| 7 | single word wide/mixed: open-addressing hash, 64 slots if m <= 32 else 128 | 17 <= m <= 64 | lib.rs:1214-1289 |
| 8 | multiword direct table 128/256 rows, 257 rows for mixed with a UCS-1 side; hash with dense rows for wide | 65 <= m | lib.rs:1552-1638, 1972-2231 |
| 9 | speculative 1-word band (64 diagonals), 2-word rung (128) with give-up at cols 4, 8, 16, then every 16 | Lev, m <= 704, s < 64 (band), W >= 5 and s < 128 (rung) | lib.rs:1494-1550, portable.rs:96-309 |
| 10 | LargeCtx: optimistic band t0 = max(64, s + 32), then `next_band` extrapolated passes while `band_blocks(t) * 3 <= w`, else full matrix (AArch64 8-word strips; x86 portable loop); `bases` vector only on the band path | m > 512 | lib.rs:1847-1869 |
| 11 | certified LCS band t = 64, passes while `lcs_band_words * 3 <= w`, hopeless check every 32 columns | ratio, m > 512 | lib.rs:1884-1947 |
| 12 | Apple `asm!` single-word step; funnel-shift multiword; Neoverse forms elsewhere | cfg | kernels/aarch64/*.rs |
| 13 | x86: V2 copy chosen at import; per call `is_x86_feature_detected!` for avx2 (word_u8, small_ub) and bmi1 (multiword W <= 4, lcs_full) | runtime | kernels/mod.rs, x86_64/v2.rs |
| 14 | Rust API `Units` transcoding (2 passes + 1 Vec per non-ASCII string), V2 detection per call | Rust API only | lib.rs:123-220 |

Note on the briefing: main has **no Hamming-UB pass** in LargeCtx anymore. `run_body` does the
optimistic pass, then passes at `next_band(t, j, n) = max(t * n / j * 5 / 4, 2t)` while the band fits
in a third of the words, then the full matrix.

## Per-layer findings

Ratios are variant / base (above 1 means the variant is slower, i.e. the layer pays). "full1" = cross-
process run, 9 rounds; "ab2" = in-process paired run, 5 rounds; IQRs in the results files.

### 1. Single-word hash sizing 64/128 slots (#241): RE-GATE (use 128 for m 17-32, or 64 only for contiguous alphabets)

| case | hash128 / base | source |
|---|---:|---|
| R:cjk-dis (CJK 10-40, 86-char alphabet), dist / ratio | 0.792 / 0.771; 0.781 / 0.778 | full1; ab2 set1 |
| R:cjk17-32-dis, dist / ratio | 0.718 / 0.711 | ab2 set2 |
| R:cjkz17-32-dis (Zipf over 3000 CJK chars) | 0.762 / 0.787 | ab2 set2 |
| R:ja17-32-dis (kana + kanji) | 0.726 / 0.712 | ab2 set2 |
| R:cjk17-32-rel, cjkz17-32-rel, ja17-32-rel | 0.88-0.92 | ab2 set2 |
| R:cjkz10-40-dis | 0.820 / 0.830 | ab2 set2 |
| R:cyr17-32-dis (contiguous Cyrillic) | 1.016 / 1.006 (n.s.) | ab2 set2 |
| R:emoji17-32-dis (contiguous) | 1.050 / 1.073 | ab2 set2 |
| R:mixed-quote17-45 | 1.001 / 1.009 | ab2 set2 |
| single repeated pairs hash-cjk-20-x1, hash-cyr-20-x1, hash-cjk-32-x1 | 1.06-1.13 | ab2 set2 |
| T:short-ucs2-32 (test_bench), dist / ratio | 1.089 / 1.117 (ab2), 1.083 / 1.096 (ab_benchmark.py) | |

Mechanism: linear probing at 37-50% load (24-32 keys in 64 slots) needs about 1.7-1.8 probes per text
character that is not in the pattern (simulated: 1.69 probes for 64 slots vs 1.24 for 128 on uniform
CJK), and the probe loop's exit is a data-dependent branch. Repeating one pair lets the predictor learn
the sequence, varied text does not. Contiguous alphabets (Cyrillic, kana-only, emoji) hash collision-
free under Fibonacci hashing, so 64 slots are fine there. A cheap refinement (unmeasured): take 64 slots
only when the pattern's code-point span is < 64 (one min/max pass over <= 32 units), else 128. Expected
gain on realistic CJK/Japanese single-word pairs: -10..-29%; cost: +6..13% on repeated test_bench-style
pairs if the plain revert is used. Confidence: medium-high on M2 (x86 hash path had spills in #260; needs
the CI run below).

### 2. mbleven for ub = 3 on short patterns: RE-GATE (skip when ub == 3, n - m <= 1, m <= 32)

| case | mb3gate32 | mb3gate (m <= 64, any diff) | nombleven |
|---|---:|---:|---:|
| A:mb-ascii-24-ub3d0 / ub3d1 | 0.641 / 0.740 | 0.657 / 0.754 | 0.637 / 0.734 |
| A:mb-ascii-24-ub3d2 | 1.013 | 1.326 | 1.317 |
| A:mb-ascii-48-ub3d0 / ub3d1 | 1.005 / 1.000 | 0.762 / 1.042 | 0.741 / 1.022 |
| A:mb-ascii-64-ub3d1 / ub3d2 | 1.003 / 1.006 | 1.186 / 2.123 | 1.172 / 2.102 |
| A:mb-cjk-24-ub3d0 / ub3d1 | 0.582 / 0.764 | 0.576 / 0.776 | 0.560 / 0.747 |
| A:mb-cjk-48-ub3d1 / 64-ub3d1 | 0.978 / 0.982 | 1.128 / 1.295 | 1.097 / 1.263 |
| A:mb-*-ub2d0 (24-100) | 0.97-1.01 | 0.98-1.01 | 1.14-5.8 |
| R:names3-subs3 (3-word names, 3 spread subs) | 0.646 | 0.643 | 0.617 |
| R:titles-subs3 | 0.958 | 0.754 | 0.720 |
| R:cjk-subs3 (CJK 17-64) | 0.970 | 1.236 | 1.167 |
| R:sent-subs3 (60-200) | 0.979 (n.s.) | 1.001 | 2.089 |

(all ab2 set2b; mb3gate32 has no case outside 0.96-1.03 except the wins.)

Mechanism: with diff 0 or 1 mbleven runs all 7 or 6 scripts (best can only reach diff early when the
strings are equal), each a chain of `common_prefix` calls with block-loop setup and unpredictable exits;
at m <= 32 the NEON table-free kernel (about 2 ns per column) or the 64-slot hash kernel is cheaper.
Break-even is near m = 48 for diff 0 and below 48 for diff 1; diff 2 (3 scripts) and ub <= 2 always pay.
mbleven as a whole is a large win (nombleven: 1.4-19x slower on ub <= 3 pairs at m >= 48; R:sent-subs3
2.1-2.3x; R:mixed-quote 1.53x in full1), so keep it. The small_ub gate's own cost is small: 1.5-4.6%
when a scan runs to the end and fails (A:mbfail-*-s4, nombleven 0.954-0.985 in full1), within noise on
dissimilar pairs (A:mbgate-*-dis 0.987-1.010). One unexplained item: R:names-latin1 (accented vs plain
names) is 11-12% faster without mbleven (full1 0.880, ab2 0.891) but unchanged with mb3gate32; it comes
from ub <= 2 pairs and failing scans at m 17-30 on varied input, too small to chase further here.
Est. gain: -24..-42% on (ub = 3, diff <= 1, m 17-32) pairs, about -35% on short multi-word names with 3
scattered typos. Confidence: high on M2; x86 break-even unmeasured.

### 3. Speculative small bands (#260): KEEP, RE-GATE the 1-word band to s < 32 (medium confidence)

| case | noband | bands32 |
|---|---:|---:|
| R:sent-dis (60-200, unrelated) | 0.868 (full1), 0.874-0.878 (ab2) | 0.927-0.933 |
| R:sent-dis-eqlen (lengths within +-5) | 0.947 | 1.000 |
| R:sent-dis-difflen (30-60 longer) | 0.833 | 0.839 |
| A:band-{ascii,sent}-100-vs-140/160-dis | 0.75-0.79 | 0.75-0.79 |
| A:band-{ascii,sent}-300-vs-340/400-dis | 0.84-0.86 | 0.87-0.95 |
| R:sent-rel10 / rel25 / rel40 / typo3 | 1.049 / 1.039-1.044 / 0.930-0.933 / 1.146-1.148 | 1.000 / 1.009 / 0.990 / 1.001 |
| R:sent-ins (5-25 doubled spaces) | 1.241 | 0.999 |
| A:band-ascii-{100,200,300,500}-e10..e25 | 1.22-3.02 | (unchanged, s < 32) |
| A:band-ascii-{100,200,300,500}-e30/e40 | 0.80-0.97 | |
| A:band-ascii-100-ins40/50 (s = 40-50, d ~ s) | 1.29 | **1.29** |
| A:band-ascii-300-ins50 | 2.26 | **1.32** |

Mechanism: the give-up rule looks at the cell on diagonal s, which is just `j` for columns j <= s, so
the band cannot give up before column s; on a dissimilar pair with s = 40-60 it runs 50-70 of 100-160
columns and then the full kernel runs anyway. With s close to 64 the band can only certify
d <= s + 3, so it almost never pays there. The near-miss cost at n ~ m (e30/e40, +10..25%) is the known
structural trade-off. Recommendation: either gate the 1-word band to s < 32 (data above: -7% on
realistic dissimilar sentences, -16% when lengths differ by 30-60; +29..32% on insertion-heavy similar
pairs that affix stripping did not already reduce, which I expect to be rare), or better, add a lower
bound usable for j < s (for instance the cell on the band's lowest diagonal, or `free` vs the minimum
matches needed) so the band gives up early on dissimilar large-s pairs without losing the insertion
class. Confidence: medium.

### 4. LargeCtx straight-to-full without row offsets: RE-GATE (build `bases` for hash-indexed tables)

| case (ab2 set3) | basesalways dist / ratio |
|---|---:|
| A:full-cjk-520/600/700-dis | 0.827 / 0.775, 0.777 / 0.764, 0.797 / 0.763 |
| A:full-cjk-800-vs-2400 | 0.796 / 0.725 |
| A:full-ascii-520..700 (dis, e30), 800-vs-2400 | 0.975-1.002 / 0.999-1.039 |
| A:large-*-2048-dis (band path, code unchanged) | 1.033-1.043 (layout) |

Mechanism: on the band path, `run_body` resolves each text char's row once into `bases`; on the early
`return kernels::full(self)` (w = 9-11, or s large) every AArch64 strip resolves `base_of(j)` again, a
hash probe for UCS-2/4. nolargeband shows the same effect in reverse: removing the band path (and with
it `bases`) makes dissimilar long pairs 5-25% slower (A:large-cjk-2048-dis 1.25 / 1.47). Change:
build `bases` before the early return only for hash-indexed contexts (`hash_tail_body`,
`hyrro_multiword_hash_body`), keep direct tables as is. Est. gain: -17..-27% for UCS-2/4 pairs of
513-704 units or with n >> m. Confidence: medium (the +3-4% on unchanged band-path cases is a layout
shift and needs checking on the targeted patch).

### 5. LargeCtx Levenshtein band: KEEP, tune give-up (no re-gate found)

nolargeband (ab2 set2b) vs base: wins of the band at <= 5% edits 1.48-2.47x (800-1200), 1.32-1.99x at
2048; dissimilar 1.06-1.25x (thanks to `bases`). Near-miss costs: 800-e10 0.691, 800-e15 0.744,
800-e20 0.790, 1000-e10 0.768, 1000-e15/e20 0.875/0.886, 1200-e10..e20 0.89-0.93; realistic
R:long-rel15 0.809, R:long-rel30 0.898, R:long-rel5 1.370. The first pass (4 of 13-19 blocks per
column) gives up only when accumulated edits exceed t + s, i.e. late for d slightly above 64.
`large2pass` (allow a second pass up to 2/3 of the words) helped 1000-e10 (0.837) and 1200-e15 (0.809)
but lost 2048-e20 1.96x and R:long-rel30 1.43x: **rejected**. A pace rule like the small band's
(give up when edits outpace `(t - s) * j / n` plus slack) would cut the failure cost; untested.

### 6. Certified LCS band (ratio, m > 512): KEEP for now, fix give-up; flag as the largest regression

nolargeband vs base for ratio: near-miss costs 800-e20 0.726, 1000-e15 0.703, 1000-e20 0.768,
1200-e10 0.943, 1200-e20 0.845, R:long-rel15 0.875, R:long-rel30 0.817 (i.e. the band makes ratio
14-44% slower there); wins 800-e5..e15 1.19-1.23, 1000-e5/e10 1.42/1.38, 1200-e5 1.65, 2048 1.2-2.6,
R:long-rel5 1.152. `lcsgate5` (band only when a pass covers <= 1/5 of the words) removes both the wins
and the losses at 800-1200 and cuts the third pass at 2048 (e15 1.39x, e20 1.28x slower): **rejected**.
The band pass costs much more per column than its word fraction (generic `lcs_step` over a dynamic slice,
plus `lcs_band_hopeless` popcounts every 32 columns), so giving up at half the text costs ~40%. Fix
direction: const-width strip kernel for the band and an earlier pace-based give-up. Confidence: medium.

### 7. TINY_M = 16 for wide units: KEEP

tinywide8 (wide patterns 9-16 to small_ub/mbleven/hash): realistic batches lose: R:cjk-dis 1.108/1.131
(full1), 1.111/1.135 (ab2), R:cjkz10-40-dis 1.160/1.189, R:cjk17-32-rel 1.127/1.124, R:ja17-32-rel
1.133/1.143; distance on repeated pairs loses too (kana10-12 1.098, emoji-12 1.084, cjk-14-s3 2.12 via
mbleven). It wins only on repeated single ratio pairs (cjk-16-dis 0.850, cjk-12-vs-48 0.805): the
branch-free const-N compare beats the probing hash once inputs vary. This supersedes the July memory
note (H3) that TINY_M = 8 was optimal for wide units.

### 8. Table-free NEON masks (#266, n <= 8m): KEEP

notablefree (full1 / ab2 set1): A:tf-ascii-*-x1 1.18-1.49, x4 1.03-1.07, x8 1.01-1.03, x9 control
0.97-0.99; realistic R:titles-dis 1.136/1.169 and 1.132/1.165, R:addr-rel 1.089/1.121 and 1.098/1.119,
R:sent-typo3 1.051/1.069 and 1.057/1.075, T:short-latin1-32-dis 1.345/1.531. The 8x limit is right for
ratio (x8 ratio 0.994-1.027); distance still wins 1.4-2.9% at x8, so a higher limit for distance only is
possible but small.

### 9. SWAR for u8 tiny patterns (m <= 8): KEEP

noswar (NEON for all u8 tiny): A:kitten 1.074/1.182 (full1), 1.065/1.165 (ab2), A:ascii-3-x1 1.27/1.23,
T:short-ascii-1 1.36/1.31, T:short-latin1-6 1.18/1.26, R:words-typo 1.05/1.09, R:names-typo 1.04/1.08.
NEON wins only at m = 8 (ratio ascii-8-x1 0.914) and on Latin-1 names (0.935-0.943). The NEON path's
stack copy + vector load costs more setup than SWAR's byte packing.

### 10. Apple `asm!` single-word step: KEEP

noasmstep: the plain-Rust step is 9-16% slower on the 17-64 single-word paths (A:tf-ascii-* distance
1.10-1.16, hash-cjk-* 1.05-1.12, T:short-ucs2-32 1.079-1.081, R:titles-dis 1.088, R:addr-rel 1.038). It
is 2.5-3.5% faster on tiny m (A:ascii-6-x1 0.967 / 0.965, A:kitten 0.970 in ab2), invisible on realistic
word batches (0.992-0.999). Possible micro-tweak: plain step inside the tiny path only (<= 3.5% at R1,
repeated pairs only); not worth the layout risk.

### 11. 257-row mixed table, 8-word strips, funnel form, `#[inline(never)]` boundaries: KEEP (not re-measured)

Earlier A/B evidence on all runners (257 rows +37..44% faster, strips and funnel form from #191/#208)
stands; nothing in these runs pointed at them. Layout side effects of unrelated changes were 1-4% in
these runs (e.g. hash128 on R:names-latin1 ratio 0.947, basesalways +3..4% on unchanged band-path cases),
consistent with the known hazard.

### 12. PyO3 wrapper, extraction, catch_unwind (R1 fixed overhead): KEEP (product constraint)

Quiet P-core numbers (full1, test_bench cases, 3.13): identity pair `short-equal-5` 18.8 ns (distance) /
24.4 ns (ratio); empty vs 100 19.7 / 25.9; "a" vs "b" 23.7 / 28.3; kitten/sitting 35.5-35.9 / 34.7.
Decomposition (`r1.py`, 9 interleaved rounds, slow machine state, same core type for all builds):

| 3.13 (ns, median) | value | 3.14 |
|---|---:|---:|
| `operator.is_` via the same map/deque driver (CPython floor) | 26.4 | 27.8 |
| r1probe distance (PyO3 trampoline + extraction + int return) | 50.5-52.9 | 52.2-54.9 |
| distance same object (identity shortcut) | 56.3 | 58.2 |
| distance "" vs "" (view + dispatch) | 54.5 | 55.6 |
| distance "hello" vs equal copy (affix strip to empty) | 64.1 | 67.3 |
| distance "a" vs "b" | 78.0 | 81.6 |
| distance kitten/sitting | 118.6 | 119.9 |
| r1probe ratio (float return) | 64.4-69.1 | 70.0-77.1 |
| ratio kitten/sitting | 119.3 | 129.7 |

Shares of a kitten distance call: CPython call floor ~22%, PyO3 entry ~21-22%, view + dispatch ~3%, strip
+ SWAR tiny kernel + popcounts ~54% (quiet full1 numbers give 55% overhead / 45% work: 19.7 of 35.5 ns).
The PyO3 share (~8 ns quiet) is the accepted #219 cost; the float return of `ratio` costs 5-6 ns quiet
(10-15 ns slow state); 3.14 adds 1-4 ns to distance and 5-10 ns to ratio (slow state). Within lev's
control at R1: the "a" vs "b" row shows ~23 ns slow state (~7 ns quiet) to compute a 1x1 problem; a
`m == 1` shortcut (distance = n - [c in text], indel = n + 1 - 2 [c in text]) would skip strip-suffix,
the out-of-line tiny call and SWAR setup. After affix stripping, m = 1 is the common outcome of single-
substitution typos (R:words-typo); unmeasured, est. 2-5 ns on those calls.

### 13. x86-only layers (unmeasured here): per-call `is_x86_feature_detected!`, V2 copy, AVX2 forms

Reasoning: std caches detection in an atomic; each check is one load and a predictable branch, and none
sits on the R1 path (tiny path never calls them). Expect <1%. The V2 copy matters mostly for POPCNT
(final popcounts, `delta`, small_ub counting): without it `count_ones` is a ~12-op sequence, a few ns at
R1, so it likely pays on short strings. See the CI plan below.

### 14. Rust API `Units` transcoding: unmeasured

Non-ASCII `&str` inputs cost one `chars().max()` pass, one transcode pass and one heap allocation per
string, plus a V2 `detected()` (6 cached checks) per call on x86. For short non-ASCII strings the two
allocations likely dominate (tens of ns). A stack buffer for, say, <= 64 chars would remove them; needs a
criterion bench (Python path unaffected).

## The CI A/B run that would settle the x86 questions

Use `.github/workflows/ab-benchmark.yml` (alternating base/head on ubuntu x86 EPYC, windows x86 Zen,
ubuntu arm Neoverse N2, windows arm, macOS M1), one head per question, two dispatches each:

1. hash128 and the span-gated refinement: does the 128-slot win on varied CJK hold with x86's hash-path
   spills?
2. mb3gate32: x86's single-word step is slower than Apple's `asm!` step, so the mbleven / kernel
   break-even may sit at a larger m; also run m = 33-48 ub-3 cases.
3. bands32 and basesalways on x86 and Neoverse.
4. Per-call detection: head caches `avx2`/`bmi1` in the copy choice (or a `static` bool read once) vs
   base; falsified if no case moves > 1%.
5. V2 copy value: head with `v2::detected()` forced to false vs base; if every case is within 3%, the
   second copy (code size x2, layout coupling) is not earning its keep.
6. AVX2 `word_u8` (m 17-32, n <= 8m) and AVX2 `small_ub` (32-byte blocks) each removed vs base.

Precondition: `ab_benchmark.py` only times test_bench's repeated pairs, which hide exactly the effects
found here. Add the varied-pair batches from `harness/cases.py` (the `R:` cases, 1000-3000 distinct
pairs each) as test_bench cases, or a `--realistic` case set, before running these. On dedicated CI
runners the cross-process design is fine; on this laptop use `harness/ab2.py`.

## Raw results files

- `results/full1.json` / `.txt`: cross-process, 10 builds, 9 rounds x 11 repeats, 289 combos (clean).
- `results/ab2_set1.json` / `.txt`: in-process, 10 builds, 5 rounds x 11 reps (realistic, tf, tiny, T:).
- `results/ab2_set2.json` / `.txt`: in-process, 10 builds, set 2 (mbleven, bands, LargeCtx, hash).
- `results/ab2_set2b.json` / `.txt`: in-process, mb3gate, mb3gate32, nombleven, lcsgate5, nolargeband.
- `results/ab2_set3.json` / `.txt`: in-process, basesalways, nolargeband.
- `results/r1_313.json`, `results/r1_314.json`: R1 decomposition.
- `results/abbench_hash128.md`, `results/abbench_mb3gate32.md` (artifact run), `results/abbench_mb3gate32_r2.md`.
- `results/full2*.json`: discarded (noise, see above).

## Ranked table

| rank | layer | verdict | evidence | est. gain from change | confidence |
|---:|---|---|---|---|---|
| 1 | single-word hash 64 slots for m 17-32 (#241) | re-gate: 128 slots, or 64 only when the pattern's code-point span < 64 | realistic CJK/Japanese 17-40: dissimilar 0.71-0.82, related 0.88-0.92 (ab2, 5 rounds, IQR <= 2.4%); repeated pairs 1.06-1.13; emoji 1.05-1.07; Cyrillic n.s. | -10..-29% on realistic UCS-2 single-word pairs | medium-high (M2 only) |
| 2 | mbleven at ub = 3, diff <= 1, m <= 32 | re-gate: skip, fall through to the kernel | ascii-24 0.64/0.74, cjk-24 0.58/0.76, R:names3-subs3 0.65; no case worse than 1.03 (ab2 set2b) | -24..-42% on that class | high |
| 3 | LargeCtx straight-to-full without `bases` (wide) | re-gate: build `bases` for hash-indexed tables | cjk 520-700 dis 0.76-0.83, cjk 800-vs-2400 0.73-0.80; ascii 0.98-1.04 (ab2 set3) | -17..-27% UCS-2/4 m 513-704 or n >> m | medium |
| 4 | 1-word small band for s >= 32 (#260) | re-gate to s < 32 (rung s < 96), or add a j < s give-up | R:sent-dis 0.93, dissimilar difflen 0.84, 100-vs-140/160 0.75-0.79; insertion-heavy s >= 32 1.29-1.32 | -7..-25% on dissimilar pairs with s 32-63 | medium |
| 5 | certified LCS band, m 705-1216 | keep, fix give-up / band kernel | near-miss ratio +14..44% (nolargeband 0.70-0.94), realistic long related +14..22%; wins 1.15-1.65x at <= 10% edits; lcsgate5 trades | up to -30% on near-miss ratio if a pace rule works | medium |
| 6 | LargeCtx Levenshtein band near-miss | keep, add pace give-up | 800-1000 at 10-20% edits +12..45%; R:long-rel15 +24%; large2pass rejected (2048-e20 1.96x, long-rel30 1.43x) | up to -25% on near-miss | medium |
| 7 | R1: m == 1 after affix strip | add shortcut (new, small) | "a" vs "b" costs ~7 ns quiet over the empty case | -2..-5 ns on single-substitution typo pairs | low (unmeasured) |
| 8 | TINY_M = 16 for wide units | keep | tinywide8 +9..19% on realistic CJK batches | none | high |
| 9 | table-free NEON masks (#266) | keep | without: +9..17% realistic titles/addresses, +18..49% at n = m | none | high |
| 10 | SWAR for u8 m <= 8 | keep | NEON instead: +4..9% realistic words/names, +7..36% R1 shapes | none | high |
| 11 | Apple `asm!` step | keep | plain step +9..16% on 17-64 paths, -3% only on repeated tiny pairs | none | high |
| 12 | mbleven overall + small_ub gate | keep | without: 1.4-19x on ub <= 3 pairs, R:sent-subs3 2.1-2.3x; gate failure 1.5-4.6% | none | high |
| 13 | small bands overall (#260) | keep (with rank 4) | without: e10-e25 1.2-3.0x slower, R:sent-typo3 1.15x; with: R:sent-dis +13%, e30/e40 +10..25% | see rank 4 | high |
| 14 | PyO3 entry / extraction / float return | keep (constraint) | ~50-55% of a kitten call is fixed overhead; PyO3 ~half of it (~8 ns); ratio float +5-6 ns | none allowed | high |
| 15 | per-call `is_x86_feature_detected!`, V2 copy, AVX2 forms | x86 CI A/B (plan above) | not measurable on M2 | likely <1% (detection); V2 likely pays at R1 | low |
| 16 | 257-row mixed table, 8-word strips, funnel form, `#[inline(never)]` | keep | prior multi-runner A/B; no contrary signal here | none | medium |
| 17 | Rust API `Units` transcoding | stack buffer for short non-ASCII (Rust API only) | reasoning: 2 allocations + 2 passes per call | tens of ns on short non-ASCII Rust calls | low (unmeasured) |
