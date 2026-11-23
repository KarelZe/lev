# Performance opportunities for lev: research results

Date: 2026-11-09. Code base: `origin/main` @ 5a6859c (0.6.1, with #260 and #266).

Six research agents worked from one shared briefing:

- literature;
- competitor implementations;
- novel bound and pruning methods;
- novel compute reformulations;
- architecture and toolchain;
- an audit of the current layers.

Files next to this one:

- `reports/*.md`: the six full agent reports.
- `audit/patches/*.diff`: one diff per audit build, against 5a6859c.
- `audit/harness/*.py`: the audit's benchmark harness and its varied-pair cases.

The simulators, asm probes, competitor clones and raw audit timings stayed in a temporary session directory
and are not included. The reports still cite their original paths under `/private/tmp/...`.

**Caveat.** Only the audit section (A1-A7) is measured: lev wall clock on the M2 Pro. No other number below is a lev wall-clock measurement. The gains come from:

- emitted asm (rustc 1.95.0, wheel flags);
- cycle models;
- work counts from simulators run on realistic pairs (11k prose, name, address, product and CJK-like pairs, plus lev's benchmark pairs).

Every idea still has to pass the AGENTS.md loop. Where agents disagreed, the table uses the more conservative figure.

## Do first

0. **Re-gate two current layers** (S, measured on M2; see the audit section).
   - **A1:** the 64-slot single-word hash for m 17-32 loses 10-29% on realistic CJK pairs.
   - **A2:** mbleven at ub = 3 for m <= 32 loses 24-42% against the kernel on that class.
   - Both wins are measured, not modelled, and both changes are gate edits.
   - Before tuning more gates, switch the benchmarks to batches of distinct pairs. Repeated pairs train the branch predictor, and that is why #241 looked like a win.
1. **Tight Ukkonen band in `LargeCtx::banded_body`** (S, high confidence).
   - The band computes rows `[j-t, j+t]`, i.e. diagonals `[-t, t]` (`lib.rs:1769`, `1778`, `1787`). I verified this in the code.
   - Certifying `d <= t` only needs diagonals `[-(t-s)/2, s+(t-s)/2]` with `s = n-m`, which is what edlib (`edlib.cpp:755`) and lev's own small band use.
   - Blocks per column at `t0`: 3 → 2 when s = 0, and about 5 → 3 when s = 100. Three expressions change.
   - Fuzzed against the DP: 331 pairs, 602 passes, always exact.
   - Expected −20..−35% per band pass on long similar pairs. Keep the `band_blocks(t)*3 <= w` gate unchanged; lowering it was rejected before.
2. **Small-band step algebra** (S, high confidence in asm).
   - #260's `portable::small_band` step (`portable.rs:134-142`) compiles to an 8-cycle chain on x86, against 6 on AArch64. x86 cannot fold the negations into `orn`.
   - Rewriting it with `!ph = X & !vn` gives 6 cycles on x86 without BMI1.
   - The skew identity below gives 5 cycles on AArch64.
   - Expected −10..−25% of band time on 65-704-char similar pairs.
3. **Orientation swap for short-vs-long pairs** (S for u8 with n <= 512, M for all kinds).
   - `compute` always makes the shorter string the bit-vector pattern (`lib.rs:374`). A pair with m <= 64 < n therefore runs n latency-bound single-word steps, with 64-m bits idle.
   - With the roles swapped: m steps when both strings fit one word (m < n <= 64), or m columns of the existing ⌈n/64⌉-word kernel, whose words give ILP.
   - Model: 1.5-3x at n >= 2m, e.g. `ascii-24-vs-384` about 3x. No change at n ≈ m, where the gate keeps today's path.
   - Found independently by the literature agent (Fredriksson tiling, Hyyrö's "reverse the roles") and the competitor agent (editdistance, Sassy, and RapidFuzz above 64).
   - It targets the common "short query vs longer title" workload and makes perf-ideas idea 4 unnecessary.
4. **Diagonal transition for long similar pairs** (M, high confidence on direction).
   - Distance: Landau-Vishkin. Ratio: Myers O(ND).
   - Both use the word-wide LCE that `common_prefix` already has, under a work budget, and run before the peq table is built for m > 704.
   - Cost: about (d+1)² cells plus n/8 compare words. Today's band costs about 3 blocks per column whatever d is.
   - Model: `ascii-2048-similar` 48x, `ascii-8192-similar` 57x, `ucs2-2048-similar` 29x. Typo'd long prose 3-12x.
   - Bails cost under 0.5% on dissimilar and near-miss m > 704 pairs.
   - Ranked first by three agents independently.
5. **SIMD match masks for wide-unit tiny patterns** (S-M, medium-high confidence).
   - `tiny_n` (`lib.rs:762-779`) is scalar for UCS-2/UCS-4/mixed patterns of 9-16 units, at 83-101 instructions per column on both arches. LLVM does not vectorize it.
   - Replacement on SSE2: `pcmpeqw`/`pcmpeqd` + packs + `pmovmskb`. On NEON: `cmeq` + `shrn`.
   - This returns the loop to the 6-7-cycle chain: −10..−35% per call for CJK, emoji and mixed strings of 9-16 characters.

## Ranked list

Ranked by expected value per unit of effort. "Agents" names who proposed or confirmed the idea:

- L: literature
- C: competitors
- B: bounds
- X: compute
- A: arch / toolchain

| # | Idea | Regime / kind / arch | Est. gain | Effort | Conf. | Agents |
|---|---|---|---|---|---|---|
| 1 | Tight Ukkonen band in `banded_body` | R5 similar, all kinds, Lev | −20..−35% per band pass | S | high | L, B |
| 2 | Small-band step rewrite (`!ph = X & !vn`; skew identity on ARM) | R3/R4 similar, Lev; x86 8→6 cycles, ARM 6→5 | −10..−25% of band time | S | high (asm) | A, X |
| 3 | Orientation swap (longer string in the bit vectors) | R1/R2 n >= 2m, R3 n ≫ m; all kinds, both metrics | 1.5-3x (model) | S-M | medium-high | L, C |
| 4 | Diagonal transition first for m > 704 (LV distance, Myers O(ND) ratio) | R5 similar, all kinds | 10-50x (model); bail < 0.5% | M | high (direction) | L, C, B |
| 5 | SIMD masks for wide-unit tiny patterns (m 9-16) | R1 UCS-2/4/mixed, all arches | −10..−35% per call | S-M | medium-high | A |
| 6 | Skewed single-word Myers step (5-deep chain) | R1/R2 Lev, all arches, m <= 56 | −4..−14% per call | S-M | high (verified exact) | X |
| 7 | AVX2 table-free masks for 33-64 bytes | R2 u8, n <= 4-8m, x86 | −5..−18% | S | medium-high | A |
| 8 | Band-derived upper bound U, then one certain tight pass | R5, 5-40% divergence (falls to the full matrix today) | 1.5-3.4x (model) | M | medium-high | B, L |
| 9 | x86-64-v3 copy (`andn`, `shlx`/`shrx`, AVX2 without per-call checks) | R1-R3 x86, all kinds | −4..−12% | M | medium | A |
| 10 | Ratio in 65-704: one-word LCS small band and/or Myers O(ND) | R3/R4 similar, ratio | 2-4x where certified | M | medium | B, L |
| 11 | NEON table-free `word_u8` on Neoverse, then MIDR core detection | R2/R3 on Linux/Windows arm, Linux on Apple, Oryon | −5..−20% (V2/V3), −10..−25% (Apple cores under Linux) | S then M-L | low-medium | A |
| 12 | Gap-pruned full matrix using U from #8 (edlib first/last block) | R5 dissimilar, m >= 1000 | −25..−35% x86, −20..−25% AArch64 | M (x86) / L (strips) | medium | B |
| 13 | AVX-512VL `vpternlogq` skewed step in an xmm lane | R1/R2 Lev, Zen 4 and Intel AVX-512 (not Zen 5) | −15..−30% per call | M | medium | X, A |
| 14 | Online band match masks (no table build when the band succeeds) | R3/R4 band successes | −5..−15% | M | low-medium | C |
| 15 | Persistent scratch with touched-entry reset (= TLS scratch item) | R3 setup, UCS-2/4 | −2..−6% | S-M | low | C |
| 16 | LCS alphabet compaction (drop units absent from the other string) | ratio, UCS-2/4 dissimilar | 1.7-3.7x fewer word steps (CJK), ~0 on Latin | S-M | medium | B |
| 17 | Two-chain band via rank convergence | R3/R4 similar, n >= 250, Apple only | −8..−18% of band kernel beyond #2 | M-L | medium | X |
| 18 | A*PA2 staggered lanes with new angles (see below) | R5 dissimilar, x86 W >= 4 | 1.3-2x | L | low-medium | C, L, X |
| 19 | Diagonal transition in 65-704 for distance | R3/R4 similar | realistic pair 2.7x; near-miss +16..+21% | S after #4 | low-medium | B, L |
| 20 | Toolchain: pin `rust-toolchain.toml`; PGO; alignment flags | all | 0 (variance), −2..−5% (PGO), ~0 (alignment) | S / M-L / S | high / low-medium / low | A |

Regimes: R1 m <= 16, R2 17-64, R3 65-512, R4 513-704, R5 > 704; m = shorter length after affix strip.

## Notes per idea

### 1. Tight band (`lib.rs:1762-1831`)
- Replace `t` with `(t-s)/2` in `last`/`needed` and with `(t+s)/2` in `first`.
- The pessimistic boundaries stay valid, since the tight band is a subset of today's.
- Same certification argument as `band_limit`. The proof that the give-up test stays valid is in `reports/novel-bounds.md`.
- Follow-up: edlib's per-column threshold tightening and block pruning inside the band (`edlib.cpp:792-830`), a further +10..30% (L, D).
- Falsify if `long-*-similar` and by_length 1024-8192 with 1-10% edits gain under 10%.

### 2. Small-band step
- x86: `nph = X & !vn` with `X = sum | vp | eq`, `vp' = mh | (nph & !xvs)`, `vn' = xvs & !nph`, mh in andn form (probe `band_new`).
- ARM: `vp' = mh | (x & !(xvs | vn))`, `vn' = (vn | !x) & xvs`. Chain: and, add, orr, bic, orr.
- The same rewrite may apply to `small_band2` (`portable.rs:239-253`), but that is throughput-balanced; measure it, don't expect a win.
- The band funnel shift is `shrd %cl`, 5 uops on Zen 3/4. An immediate-shift form is worth a look.
- Falsify if under 3% on the x86 A/B for similar 128-704 pairs.

### 3. Orientation swap
- **Where to call:**
  - u8 with both <= 64: `kernels::word_u8::<K, SLOTS>(longer, shorter)`. The table-free forms apply, since text <= 8 × pattern.
  - 65-512: `hyrro_multiword_direct_body` / `hash_kernel_body` with `(longer, shorter)`.
  - Skip `small_bands` and the band passes: they assume n >= m (`lib.rs:1532`, `1764`, `1855`, `1932`).
- **Gate:** a per-arch cost formula, swapping when `m·c(W') + table(W') < n·c(1) + table(1)`. At m = 60, n = 96 the model says keep today's path (0.97x).
- **Ratio:** gains more, because a single-word LCS column is about 3 cycles of latency while a multiword LCS word costs 1-1.5 cycles.
- **Risk:** setup on the long side (zeroing up to 16 KiB for Latin-1 W = 8; hash inserts for UCS-2/4 cost n probes). Mitigate with masks built only for the short string's distinct units, via SIMD compares over the long string. This is C's variant, which also removes every per-character hash probe.
- **Falsify** if `ascii-24-vs-384` gains under 20% or any n ≈ m case moves beyond noise.

### 4. Diagonal transition (LV / O(ND))
- **Budget:** `E = sqrt(β·C_est/6)`, restricted to the diamond `e + |k-s| <= E`, with a pace bail and a cap on total LCE words (periodic text).
- **On a bail,** pass the certified lower bound `d > e` to the band driver.
- **Mixed kinds:** LCE in 16-unit blocks, or skip LV.
- **Extensions:**
  - Bidirectional DT (Myers middle snake / BiWFA) to about double the reachable d (L).
  - Bound-and-align: start the band at the bound from a bail (L, I).
- **Medium regime:** keep it as a separate experiment (#19). The bounds simulator reproduces #260's near-miss class there:
  - prose edit5/256 +16..+20%;
  - `ascii-100` 25%-edit +21%.
- **Falsify** if `ascii-2048-similar` gains under 5x or any long-dissimilar case regresses beyond noise.

### 5. Wide-unit tiny masks
- Gate on m >= 8; at m <= 6 the scalar form is cheap.
- Keep it inside `hyrro_64_tiny_body` (already `#[inline(never)]` per copy) to contain layout effects.
- Add tiny CJK and emoji cases to `test_bench.py` if they are missing.

### 6. Skewed single-word step
- **Mechanism:**
  - Slide the window one row per column, as `small_band` does, so the only shift is `xvs = (eq|vn) >> 1`, which sits off the vp chain.
  - Keep 8 padding rows above row 1 and shift both words back every 8 columns. The eq of column t in a block is `peq[c] << (PAD+1-t)`.
  - This is exact for any n with m <= 56; 57-64 keeps today's step.
- **Verified:** 4000 pairs in Python and 20000 in Rust against the DP. LLVM emits the 5-deep form without `asm!` on both arches.
- **Gain:** kernel −12..−15% per column. Per call −10..−14% on `ascii-24-vs-384`, −4..−9% on 32-64-char dissimilar pairs, ~0 on kitten.
- **Interaction:** it composes with #3 and #13. After #3 it mainly helps n ≈ m pairs.
- **Falsify** if `ascii-24-vs-384` gains under 8% on M2.

### 7-9. x86
- **Fact:** the single-word step is written for `andn` but never gets it. The V2 copy does not enable BMI1 (`v2.rs:39`), so the chain is 7 cycles instead of 6.
- **v3 copy:**
  - It delivers `andn` and `shrx`.
  - A third copy adds about 35-45% to the wheel (V2 is 44% of x86 instructions).
  - One A/B round can decide between a third copy and replacing V2 with V3 (size-neutral; v2-only CPUs fall back to v1).
- **AVX2 33-64:** two `vpcmpeqb ymm` + two `vpmovmskb`. Tune the 8× text gate per arch, and measure ratio separately: the 3-cycle LCS chain may turn throughput-bound, as on M2.

### 8 and 12. Long pairs past the band gate
- **#8:**
  - Run a one-word sliding band to completion, with no give-up. Every band cell bounds its true value from above, so the final score is a certified U >= d.
  - Then run one tight pass at t = U, which certifies by construction.
  - In simulation U equalled d on every prose and random pair (at most 1.013 d). Pairs with 5-40% divergence today fall to the full matrix.
  - Gate the U pass on `next_band`'s extrapolation; ungated, unrelated pairs pay +6..11%.
- **#12** uses the same U for edlib-style pruning of the full matrix. About half the blocks are prunable on unrelated prose at m >= 1000. There is no gain at W <= 8.

### 11. AArch64
- Linux on Apple cores runs the Neoverse step at 7 cycles against 6 for the Apple `asm!`, because a shifted-operand `orr` on the chain costs 2 cycles there.
- LLVM's Oryon model has the same latency pattern as Apple, so the Apple forms suit Oryon.
- Cheapest step first: test #266's NEON table-free form on the Neoverse A/B runners (no detection needed).
- Then MIDR detection (`/proc/cpuinfo` or sysfs on Linux, registry on Windows) as a const of a hot-path copy.

### 13. AVX-512 ternlog step
- 4-cycle chain (vpand → vpaddq → 2 ternlogs), 10 vector ops per column, eq loaded straight into xmm.
- Zen 5 has 2-cycle vector integer latency (`vpaddq`, `vpternlogq` from operands 2/3), so it would lose there; it needs a CPU-family gate.
- GitHub x64 runners mix Zen 3 (no AVX-512), Zen 4 and Zen 5. The A/B "Hardware" step should log the CPU model. That is a CI change, so it needs your approval.

### 17. Rank convergence inside the band
- Start a second chain mid-text from a V-shaped guess (deltas −1 above diagonal s, +1 below).
- It converged in 100% of certified band pairs. Median 30-45 columns (p90 40-85) for 64 diagonals; median 44-108 for 128.
- No reversed table and no combine: a 2-word equality test and one subtraction.
- Gains only on 6-ALU cores (Apple), and only for n >= ~250.

### 18. Staggered lanes: what is new versus perf-ideas idea 3
- Use A*PA2's carry-free block form in the lanes (`pa-bitpacking/src/myers.rs:60-88`). AVX2 has no unsigned 64-bit compare, so a carry-out costs about 4-5 extra ops per word step.
- Keep the last 4 full peq rows in registers and assemble the staggered vector with 3 immediate blends (1 load + 3 blends per column) instead of gathers.
- Keep at most two vectors in flight (A*PA2: three or more "almost 2x slower").
- Use `vpternlogq` in a v4 copy.
- Scalar staggering is only worth 5-10% per A*PA2's own data.
- Still the only lever for long dissimilar pairs on x86. Under SETH that regime can only improve by constant factors.

## Audit: current layers that may hurt (measured)

**Method.**
- 16 single-change builds of `origin/main`, measured on the M2 Pro. Ratios are variant / base; above 1 means the layer pays.
- First round in separate processes: 9 rounds, 289 cases, A/A within 0.974-1.024.
- Later rounds used an in-process harness that loads every build into one process and alternates batches. It held A/A at 0.99-1.01 even under load, so only its ratios are used.
- Correctness: the four re-gate candidates had 0 mismatches against RapidFuzz on 30k random pairs. Three of them also pass `cargo test` and `pytest` against the built wheel.
- Patches and harness: `audit/patches/`, `audit/harness/ab2.py`. The raw timings (32 MB) are not included.

**Key method finding.**
- `test_bench.py`, CodSpeed and `ab_benchmark.py` repeat one pair per case, so the branch predictor learns the pair.
- Three layers flip sign between repeated pairs and batches of distinct pairs: hash sizing, TINY_M for wide strings, and mbleven on short Latin-1 names.
- Benchmarks need varied-pair batches (`harness/cases.py`) before more gate tuning. Otherwise gates get tuned to a predictor artefact.

| # | Layer | Verdict | Evidence (ratio variant/base) | Gain from the change | Conf. |
|---|---|---|---|---|---|
| A1 | Single-word hash with 64 slots for m 17-32 (#241) | **Re-gate:** always 128 slots, or 64 only when the pattern's code points span < 64 | Realistic CJK/Japanese 17-40: dissimilar 0.71-0.82, related 0.88-0.92; emoji 0.93-0.95; Cyrillic unchanged. Repeated pairs show the opposite (1.06-1.13, the source of #241's +8.3%) | −10..−29% on realistic UCS-2 single-word pairs | medium-high (M2 only) |
| A2 | mbleven at ub = 3 with length diff <= 1 and m <= 32 | **Re-gate:** skip it there, fall through to the kernel | ascii-24 0.64/0.74, cjk-24 0.58/0.76, 3-word names with 3 typos 0.65; no case worse than 1.03. Widening the skip to m <= 64 loses up to 2.4x | −24..−42% on that class | high |
| A3 | LargeCtx full matrix without precomputed row offsets for hash-indexed (wide) tables | **Re-gate:** build the offsets (`with_bases`) for hash tables | CJK 520-700 dissimilar 0.76-0.83, CJK 800 vs 2400 0.73-0.80; ASCII 0.98-1.04 | −17..−27% for UCS-2/4 at m 513-704 or n ≫ m | medium |
| A4 | 1-word small band at length difference 32-63 (#260) | **Re-gate** to s < 32 (2-word rung s < 96), or add a give-up that fires before column s | Dissimilar sentences 0.93; lengths differing by 30-60 0.84; 100 vs 140/160 0.75-0.79. Insertion-heavy similar pairs 1.29-1.32 (they lose the band) | −7..−25% on dissimilar pairs with large length difference | medium |
| A5 | Certified LCS band for `ratio`, m 705-1216 | Keep; fix its give-up | Near-miss ratio +14..44%, realistic long related text +14..22%; wins 1.15-1.65x at <= 10% edits. The 1/5-of-words gate only trades wins for losses | up to −30% on near-miss ratio | medium |
| A6 | LargeCtx Levenshtein band on near-miss pairs | Keep; add a pace-based give-up | 800-1000 chars at 10-20% edits +12..45%; 15% related +24%. A relaxed second pass is worse (2048-e20 1.96x) | up to −25% on near-miss | medium |
| A7 | 1-character pattern after affix strip | New shortcut (small) | "a" vs "b" costs ~7 ns more than the empty case | −2..−5 ns on single-substitution typos | low (not measured) |

**Kept, because each pays:**

| Layer | Evidence (time without the layer) |
|---|---|
| TINY_M = 16 for wide strings | Moving 9-16 to the hash is 9-19% slower on realistic CJK batches |
| #266 NEON table-free masks | +9..17% on realistic titles and addresses, +18..49% at n = m |
| SWAR for u8 m <= 8 | NEON instead is +4..9% on words and names, +7..36% on R1 shapes |
| Apple `asm!` step | Plain Rust is 9-16% slower on 17-64-char paths |
| mbleven and its `small_ub` gate overall | 1.4-19x slower on ub <= 3 pairs; the gate costs 1.5-4.6% when its scan fails |
| #260 small bands overall | 1.2-3.0x slower at 10-25% edits (A4 is the exception) |
| 257-row mixed table, 8-word strips, funnel form, `#[inline(never)]` boundaries | Nothing in this audit contradicts the earlier multi-runner A/B |

**Also from the audit:**
- **R1 cost split** (kitten/sitting, ~35 ns quiet):
  - Fixed overhead is 50-55% of the call: the CPython call floor ~22%, PyO3 entry and argument extraction ~21% (~8 ns, the accepted #219 cost), view and dispatch ~3%.
  - `ratio` pays 5-6 ns more for the float return. 3.14 adds 1-4 ns to `distance` and 5-10 ns to `ratio`.
  - So at R1, kernel work can win back at most about half of the call.
- **x86 questions still open:** the per-call `is_x86_feature_detected!` checks, whether the V2 copy pays, and the AVX2 forms need the CI A/B. `reports/audit.md` lists six heads to run. Add the varied-pair batches first.
- **Briefing correction:** LargeCtx has no Hamming-UB pass any more. It runs the first pass, then extrapolated passes while the band fits a third of the words, then the full matrix.
- **Rust API only:** `Units` transcoding makes 2 passes and 1 allocation per non-ASCII string. A stack buffer for short strings would save tens of ns (not measured).

## Investigated, not worth it

| Idea | Why not |
|---|---|
| Rank convergence over the full matrix (text chunks as parallel chains) | Row 0 is pinned (D[0][j] = j); at n ≈ m it never converged in any text model, m 16-512, both metrics |
| Hirschberg completion of a failed band; band-viability lower bounds (bag, q-gram) | Reversed table + combine eat the gain; the bounds fire on 2 of 71 near-miss failures |
| O(1) exits for short pairs (disjoint alphabets, LB = Hamming UB) | They fire on 9-56% of unrelated short pairs but cost about what they save (net −10..+3 cycles) |
| A* seed heuristics, bag distance, DNA-specific A*PA2 parts (2-bit profile, pre-pruning, incremental doubling) | +4 points of pruning at most on prose; alphabet assumptions |
| Packing sub-problems into one word / NEON lanes | No independent sub-problems in one pair; NEON integer ops are 2 cycles on Apple |
| Carry-select, Kogge-Stone, ADX for multiword | Multiword is throughput-bound, not ripple-bound; logic ops clobber CF/OF, so two ADX chains cannot interleave |
| Levenshtein DFA / Four Russians | 1-2k states at m = 8, 25-73k at m = 12; a cached lookup is no faster than the skewed step |
| Anti-diagonal u8/u16 DP (triple_accel, StringZilla wavefront) | StringZilla's own notes: Myers is 5-25x faster |
| StringZilla's "4 interleaved NEON chains" | Interleaves different pairs; for one pair it falls back to serial Myers |
| APX | Unstable `apxf` (rust #139284); hardware late 2026 / 2027, none on Zen 6 |
| SVE/SVE2 (MATCH, BDEP), SME, macOS `target-cpu` | No stable SVE intrinsics; 128-bit VL; `aarch64-apple-darwin` already defaults to `apple-m1` |
| BOLT, `multiversion` crate, Neoverse V2 retune | Little or no gain; V2 has N2's latencies |
| BitPAl, GenASM, Shifted Hamming Distance | Same op count as Myers, or worse asymptotics |
| arXiv 2111.08692 and 2610.17406 | Lemire's "Unicode at Gigabytes per Second" and "Fast Forgiving Base64 Decoding with SIMD"; neither is about edit distance. lev's NEON mask build (~9 ops) is already cheaper than the paper's (~13) |

Already at the state of the art, no action:

- The Myers step: 11-15 ops against the literature's 17-20.
- The LCS step: 4 ops, the minimum per Hyyrö 2017. A brute-force search found no shorter chain.
- The small band: Hyyrö's diagonal tiling with Ukkonen's band.
- The AArch64 strips: Fredriksson-style tiling.
- The x86 multiword add: already a clean `add`/`adc` chain.
- No competitor has a leaner Myers or LCS step.

## Status of perf-ideas.md

- **1** (table-free masks) and **2** (medium band): done, in #266 and #260. Follow-ups: #7 (AVX2 33-64), #11 (NEON on Neoverse), #2 above (band step).
- **3** (staggered lanes): still open, refined in #18.
- **4** (two chains): dominated by the orientation swap (#3) for n >= 2m. For n ≈ m, the skewed step (#6) is the cheaper ILP win.
- **5a/5c:** #11. **5b:** #9 (plus #7, #13). **5d:** a win32 SSE2 step gives about 2x kernel for under 1% of installs, low value. **5e:** no kernels before wheels exist.

## Measurement notes

- **Repeated-pair benchmarks mislead.** `test_bench.py`, CodSpeed and `ab_benchmark.py` time one pair over and over, so the branch predictor learns it. Three gates changed sign between repeated pairs and varied batches (audit). Add batches of distinct, realistic pairs to `test_bench.py` (template: `audit/harness/cases.py`) before tuning further gates.
- **Single-machine A/B under load.** An in-process harness that loads both builds into one process and alternates batches held A/A at 0.99-1.01 at load 4-6. Separate processes drifted up to 17%. See `audit/harness/ab2.py`.
- **Runner mix:** GitHub x64 runners mix Zen 3/4/5, and the arm runners mix Neoverse N2 and V3. A/B ratios within one job stay valid, but ISA-specific code (AVX-512) only runs on some draws. Logging `lscpu` in the A/B workflow needs your approval (CI change).
- **Toolchain:** pin it with `rust-toolchain.toml` so codegen does not drift between A/B runs (S). The repeated 5-20% layout swings make this worth doing before the kernel experiments.
- **Ordering:** #1, #2, #5, #7 are isolated, small, and measurable with existing cases. #3 and #4 need new benchmark cases:
  - short-vs-long u8 and UCS-2;
  - 2048-char pairs with 300/600 edits;
  - 20%-edit prose.
