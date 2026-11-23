# Literature report: algorithmic and SIMD reformulations for lev

Agent: literature. Code references are to the read-only export `main-src` (origin/main @ 5a6859c).
No wall-clock benchmarks were run. Counts come from two small scripts in
`scratchpad/proto/literature/` (`dt_count.py`, `swap_model.py`, `lcs_search.py`); downloaded
sources are in `scratchpad/proto/literature/downloads/`.

## 0. Sources and how they were read

| Source | Access | Status |
|---|---|---|
| Groot Koerkamp 2024, A*PA2, LIPIcs 312:17 | LIPIcs open-access PDF, full text | read fully incl. appendix A.1 |
| Hyyrö 2003, bit-vector Levenshtein/Damerau (NJC 10(1)) | PSC 2002 conference version (Proc. PSC'02 pp. 44-54, same paper before the journal version), scanned copy, pages rendered and read | sections 2.3, 3, 4 read |
| Fredriksson 2003, row-wise tiling (SPIRE, LNCS 2857) | paywalled, no preprint found | secondary: Hyyrö & Navarro, "Bit-parallel witnesses", Algorithmica 2005 preprint, section 7 (pp. 24-26), which describes and extends it; also Sassy README / blog ("horizontal blocks") |
| Hyyrö, Fredriksson & Navarro, Increased bit-parallelism (WEA 2004 / JEA 2005) | WEA 2004 preprint PDF (Navarro's site) | sections 1, 4.1 read |
| Šošić & Šikić 2017, edlib | source `edlib.cpp` (master) read; paper known | `myersCalcEditDistanceNW` lines 730-870 read |
| Ukkonen 1985, Inf. Control 64 | ScienceDirect blocked (bot wall); no copy | secondary: Hyyrö PSC'02 §2.3 (pp. 48-49) and §4 (p. 52), A*PA2 §1.2/§2 |
| Hirschberg 1975, CACM 18(6) | not fetched (ACM) | content is standard (linear-space LCS via forward + reverse score rows); lev's own perf-ideas idea 4 and the rejected two-chain measurement already engage it |
| Hyyrö 2017, Mining bit-parallel LCS-length algorithms (SPIRE) | Tampere repository behind a bot wall | abstract via search: LLCS needs >= 4 bit-vector ops per step under his model, five new 4-op variants. Checked chain depth myself (`lcs_search.py`) |
| arXiv 2111.08692 | resolves | **not edit distance**: Lemire, "Unicode at Gigabytes per Second" (2021), read fully |
| arXiv 2610.17406 | resolves | **not edit distance**: Lemire, "Fast Forgiving Base64 Decoding with SIMD Instructions" (v1 12 Oct 2026, v2 14 Oct 2026), read fully (v2 PDF, 24 pp.) |
| RapidFuzz `Levenshtein_impl.hpp`, `LCSseq_impl.hpp` | source | orientation logic read (lines 907-954) |
| Sassy (Beeloo & Groot Koerkamp, Bioinformatics 2026; Sassy2 bioRxiv 2026) | README + blog post (via fetch) | tiling in the text direction, 4 text chunks per AVX2 vector |
| QuickEd (Doblas et al., Bioinformatics 2025, doi:10.1093/bioinformatics/btaf112) | abstract via search | bound-and-align |
| Myers 1999 (JACM 46(3)), Myers 1986 O(ND) (Algorithmica 1), Landau-Vishkin 1989, WFA (Marco-Sola et al. 2021, doi:10.1093/bioinformatics/btaa777), BiWFA (2023, doi:10.1093/bioinformatics/btad074), Allison-Dix 1986, Crochemore et al. 2001, Hyyrö 2004 (LCS), BitPAl (Loving et al. 2014), GenASM (Cali et al. MICRO 2020), SHD (Xin et al. 2015), Masek-Paterson 1980, Backurs-Indyk 2015 | from prior knowledge; op counts cross-checked against A*PA2 appendix A.1 and Hyyrö PSC'02 figures 3/6 where possible | not re-read here (web search quota ran out); claims kept to well-known content |

Both arXiv links resolve but neither is about edit distance. Section 6 lists the few transferable
SIMD ideas and why they mostly do not pay.

## 1. Where lev already matches the literature (no action)

- **Myers/Hyyrö step op count.** Myers' step is 17 ops, 20 with multi-word carry-out (A*PA2 p. 6
  and app. A.1, p. 20-21); BitPAl's edit-distance form is also 20 (A*PA2 p. 5, app. A.1). lev's
  single-word step is 15 unshifted ops on Apple (`kernels/aarch64/apple.rs:15-55`) and 11-12 with
  and-not on x86 (`kernels/x86_64/bmi1.rs:36-47`). Nothing in the literature is shorter.
- **LCS step.** lev's `u = s & pm; s = (s + u) | (s - u)` (`lib.rs:1340-1348`) is the 4-op
  Hyyrö 2004 form; Hyyrö 2017 shows 4 ops is the minimum under his model. My search
  (`lcs_search.py`: every `g(f1 ± f2, h)` with one-op `f1, f2, h`, checked on reachable states)
  found only depth-3 chains (`(S+(S&M)) | (S&~M)` and one equivalent). No shorter loop-carried
  chain exists in that family.
- **Hyyrö 2003 diagonal tiling** (PSC'02 §3.1, pp. 49-51: shift VP/VN up one bit per column,
  realign PM by shift-or of two words) is exactly `kernels/portable.rs:97-193` (`xv >> 1`,
  funnel-shifted peq words) and uses Ukkonen's tight diagonal range (`portable.rs:110-113`,
  `band_limit` at `kernels/mod.rs:135-137`).
- **Edlib's block formulation** (hin/hout between 64-row blocks) is `banded_step`, `lib.rs:1682-1691`.
- **Fredriksson-style strips.** The AArch64 full matrix runs strips of 8 words that sweep the
  text and pass an edge byte per column (`kernels/aarch64/mod.rs:61-73`), the same
  register-resident tiling idea in the pattern direction.
- **LCS band** is already Ukkonen-tight: diagonals `[-t, slack + t]` (`lib.rs:1884-1906`).

## 2. Ideas that should pay off

### Idea A. Diagonal transition with packed extend for similar pairs (distance and ratio)

**Source.** Ukkonen 1985 (the unit-cost O(s·min(m,n)) "furthest reaching" algorithm, via Hyyrö
PSC'02 p. 52), Myers 1986 O(ND) (expected O(N + D²); the indel/LCS form is exactly what
`ratio` needs), Landau-Vishkin 1989, WFA/WFA2-lib (vectorized wavefront "compute" step, 8-chars-
per-compare "extend"). Empirical support: Hyyrö's own tests (PSC'02 p. 52) found Ukkonen's
increasing-order method fastest at k = 10, m = 100, ahead of every bit-parallel variant; A*PA2
(p. 14, 16) reports BiWFA faster than Edlib and A*PA2 for short (<1 kbp) and low-divergence pairs.

**Mechanism.** Keep, per cost s, the furthest row reached on each diagonal k in [-s, s]; step s
takes the max of three predecessors (vectorizable across k), then "extends" each diagonal along
matching characters with 8-byte XOR + count-trailing-zeros compares (32-byte blocks on the long
run along the optimal path, reusing `common_prefix_bytes`, `lib.rs:510-542`). Work is
O(n/8 + d²) with no peq table. A pace rule (as in `band_gives_up`, `kernels/mod.rs:125-128`)
aborts after 2-4 wavefronts when the furthest anti-diagonal is too short to finish under a cost cap
`d_max ≈ c·sqrt(n)`, and the pipeline falls through to today's path.

**lev today.** No diagonal transition anywhere (grep for wavefront/furthest/snake: none).
mbleven (`lib.rs:1093-1141`) is a d <= 3 enumeration, gated by a Hamming bound (`small_ub`,
`lib.rs:871-1015`) that fails on any early indel (the realistic pair gets ub = 186 per
perf-ideas). Similar pairs with 4 <= d <= ~30 pay a peq table plus n column steps: the small band
(`small_bands`, `lib.rs:1521-1550`, one Myers step plus funnel shift per column, about 6-7 cycles on
M2) for m <= 704, or `LargeCtx` (`lib.rs:1854-1869`, about 3 blocks per column) beyond. `ratio` has
no band below m = 513 and no mbleven at all.

**Work counts** (`dt_count.py`, after affix stripping; DTstep = diagonal updates, DTword = 8-byte
extend compares; lev path is what runs today for d > 3):

| pair | m | n | d | DT steps | DT words | lev today | indel D | O(ND) steps | O(ND) words |
|---|---|---|---|---|---|---|---|---|---|
| bench realistic | 190 | 192 | 7 | 64 | 81 | small band 192 cols + 3-word table | 8 | 45 | 64 |
| bench ascii | 73 | 73 | 8 | 81 | 86 | small band 73 cols + table | 16 | 153 | 156 |
| bench cjk (UCS-2) | 76 | 76 | 4 | 25 | 40 | small band 76 cols + hash | 8 | 45 | 58 |
| bench emoji (UCS-4) | 91 | 91 | 10 | 121 | 154 | small band 91 cols + hash | 20 | 231 | 262 |
| sweep 512, 8 edits | 478 | 478 | 7 | 64 | 117 | small band 478 cols + 8-word table | 10 | 66 | 117 |
| sweep 1024, 8 edits | 647 | 649 | 8 | 81 | 155 | small band 649 cols + 11-word table | 10 | 66 | 138 |
| long-ascii-2048-similar | 1913 | 1914 | 4 | 25 | 259 | LargeCtx W=30, band ~1914 cols x 3 blocks | - | - | - |
| long-ascii-8192-similar | 7006 | 7007 | 7 | 64 | 933 | LargeCtx W=110 | - | - | - |
| medium-ascii-100 (25% edits) | 100 | 102 | 21 | 484 | 480 | small band fails, W=2 | 32 | 561 | 556 |
| medium-ascii-300 (25%) | 280 | 290 | 65 | 4356 | 4365 | W=5 | 88 | 4005 | 4015 |
| bench latin1 | 97 | 97 | 27 | 784 | 743 | small band 97 cols | 44 | 1035 | 1004 |

Cost model (assumption, Apple M2, to be checked): ~10 ops per diagonal step (2 amortized vector
ops for the max, ~8 scalar for one extend word), independent across diagonals, so ~2.5-3.5
cycles; ~1.5 cycles per extra extend word; ~d mispredicts on the path diagonal's loop exits.
Realistic pair: DT ≈ 64·3 + 17·1.5 + 8·12 ≈ 300-350 cycles versus small band ≈ 192·6.5 + table
(3 KiB zeroing plus 190 ORs) ≈ 1,400 cycles. Break-even is roughly d ≈ 1.3·sqrt(n) against the
small band (d ≈ 13 at n = 100, 18 at 190, 34 at 700) and much higher against `LargeCtx`, where
the long similar pairs above drop from ~10^4-10^5 cycles to ~10^3.

**Regime/arch/kind.** R3/R4 similar (d from 4 to ~1.3·sqrt(n)), R5 similar (large gains), all
kinds (extend compares code units in place; mixed kinds widen like `block_mismatches`,
`lib.rs:422-430`), all arches (scalar plus a little SIMD). For `ratio`, Myers' O(ND) on indel
distance covers similar pairs at every m >= 17, where lev has no shortcut today.

**Expected gain.** Realistic pair 2-4x kernel time (per call maybe 1.5-2.5x after call overhead);
long similar pairs 10-100x; ratio similar pairs at 100-700 chars 2-4x. Zero gain on the 25%-edit
`medium-*` benchmark cases (d² too large); those must not regress.

**Effort.** M (one generic function over `CodeUnit` pairs, two wavefront arrays of i32 on the
stack for d_max <= 64, a pace rule, tail-safe extend).

**Risk.** Failure cost on dissimilar pairs: aborting at wavefront 2-4 costs 9-25 diagonal steps
(~30-90 cycles). That is under 3% for m >= 128 but up to ~10% of a W = 2 kernel at m ≈ 100, so
gate by m (start at m >= 65, or above the single-word path) and by the pace check. Branch
mispredicts in extend loops. Layout: keep it `#[inline(never)]` like the tiny path. Same structural
"just past the cap" failure class as #260's bands, so cap d_max so the worst failure is bounded
by about one band pass.

**Test.** `dt_count.py` pairs plus `benchmark_by_length.py` at 128-2048 with 1-32 edits,
`benchmark.py --kind realistic`, the `long-*-similar` and all `medium-*` cases, distance and ratio,
M2 and the A/B workflow. Property test against the DP oracle and RapidFuzz with random edits.
Falsified if the realistic pair gains < 15% or any `medium-*`/dissimilar case regresses beyond
noise after tuning the gate.

**Conflicts.** None with the rejected list. It subsumes mbleven's role for d <= 3 but mbleven can
stay. Not the "indel mbleven" that was rejected (that enumerated edit scripts; DT is O(n/8 + D²)).

### Idea B. Orientation swap: the longer string in the bit vectors when it fills words better (row-wise tiling)

**Source.** Fredriksson 2003 row-wise tiling (described in Hyyrö & Navarro, witnesses paper §7,
p. 24: "the text is cut into consecutive chunks of w characters... each row of each chunk is
computed in O(1) time"; the motivation is exactly the w - m wasted bits, their Fig. 14); Hyyrö
PSC'02 p. 51 ("simply reversing the roles of the strings A and B"); Hyyrö-Fredriksson-Navarro
2005 abstract ("wastes bit-parallelism's power in the common case where m is much smaller than
w"); Sassy's "tiling in the text direction". RapidFuzz already puts the longer string in the
bit vectors once both exceed 64 chars (`Levenshtein_impl.hpp:935-952`), but not below.

**Mechanism.** Levenshtein and LCS are symmetric. Today `compute` always makes the shorter string
the bit-vector pattern (`lib.rs:374`), so a pair with m <= 64 < n runs n latency-bound single-word
steps with 64 - m bits idle. With the roles swapped it runs m steps of the existing
`ceil(n/64)`-word kernel, whose words give ILP. When both fit one word (m < n <= 64) the swap
simply iterates over the shorter string: m steps instead of n, same per-step cost.

**Work model** (`swap_model.py`; Apple-like costs: single word ~6 cycles/step latency-bound,
W-word column ≈ max(2.5W, 4 + 2W) cycles, table = 128·W words zeroed + one OR per char; the
brief's "full kernel costs 1.5x the band per column at 65-128" suggests W = 2 columns cost ~10
cycles, so the W >= 2 figures below are optimistic by up to ~25%):

| m | n | steps now | steps swapped | modelled speedup |
|---|---|---|---|---|
| 8 | 64 | 64 | 8 | ~8x (both one word) |
| 16 | 60 | 60 | 16 | ~3.7x (both one word) |
| 24 | 96 | 96 | 48 (W=2) | 1.9x |
| 24 | 384 | 384 | 144 (W=6) | 3.1x |
| 32 | 512 | 512 | 256 (W=8) | 2.7x |
| 48 | 128 | 128 | 96 (W=2) | 1.5x |
| 60 | 96 | 96 | 120 (W=2) | 0.97x (keep current) |
| 65 | 512 | 1024 word-steps | 520 | 2.3x |
| 100 | 512 | 1024 | 800 | 1.7x |

For `ratio` the gain is larger: the single-word LCS chain is ~3 cycles/column while a multiword
LCS word costs ~1-1.5 cycles, so roughly 150/m for large n (estimate). On x86 the single-word
chain is also ~6 cycles but a word of the BMI1 multiword costs ~4.25 cycles (17 ops on 4 ALUs), so
the large-n gain is roughly 85/m instead of 118/m.

**lev today.** Not implemented: `compute_u8` (`lib.rs:821-847`), `compute_sorted`
(`lib.rs:1149-1167`) and `compute_sorted_mixed` (`lib.rs:1174-1194`) all take the shorter string as
pattern; the tiny path (`lib.rs:626-677`) loops over all n text chars.

**How to build it.** For u8 pairs call `kernels::word_u8::<K, SLOTS>(longer, shorter)` when both
<= 64 (the NEON/AVX2 table-free forms apply because text <= 8·pattern), and
`hyrro_multiword_direct_body` / `hash_kernel_body` with `(longer, shorter)` when the longer is 65-512,
skipping `small_bands` (it assumes n >= m: `let s = n - m` at `lib.rs:1532`). For n > 512 the
`LargeCtx` full kernel works but its table grows with n (128 x W words, 64 KiB at n = 4096):
either accept it or use Fredriksson's per-chunk table (strips of 8 words, table for the 512-char
chunk only, reusing the AArch64 edge machinery of `kernels/aarch64/mod.rs:61-73`). Gate with a
per-arch cost formula, e.g. swap when `m·c(W') + table(W') < n·c(1) + table(1)`.

**Regime/arch/kind.** R1/R2 with n > m (short query versus longer text, the common fuzzy-search
shape), R3 when m mod 64 is small and n ≫ m; every kind; both metrics; all arches (largest on
Apple where the single word is latency-bound).

**Expected gain.** 1.5-3x on `short-ascii-24-vs-96` / `-vs-384`-style pairs (model; real tables
and hash builds will eat part), ~0 for n ≈ m (gate keeps today's path).

**Effort.** S for the n <= 512 u8 path (dispatch plus gate, existing kernels), M including UCS-2/4
hash orientation and n > 512.

**Risk.** Table setup for the long side (bzero up to 16 KiB at Latin-1 W = 8; hash inserts for
UCS-2/4 cost n probes instead of m); wrong gate thresholds per arch; code-layout shifts in the
dispatchers (the recurring hazard). Correctness risk is low (symmetry), but every kernel's
`n >= m` assumption must be audited (`banded_body` asserts it, `lib.rs:1764`).

**Test.** m in {8, 16, 24, 32, 48, 64}, n in {m+1, 2m, 4m, 16m, 4096}, ASCII/Latin-1/UCS-2/
mixed, distance and ratio; `short-ascii-24-vs-96`, `short-ascii-24-vs-384`, all `short-*`.
Falsified if `ascii-24-vs-384` gains < 20% or any n ≈ m case moves beyond noise.

**Conflicts.** Dominates perf-ideas idea 4 (two chains) for n >= 2m: idea 4's ceiling is ~2x on
Apple with a 60-200 cycle combine; the swap uses existing kernels and no combine. Idea 4 only
keeps a niche at m ≤ 64, n ≈ m (see idea H). Not on the rejected list.

### Idea C. Ukkonen-tight static band in `LargeCtx::banded_body`

**Source.** Ukkonen 1985 via Hyyrö PSC'02 §2.3 (p. 48): with ed <= t and m <= n it suffices to
fill diagonals `-floor((t - n + m)/2) .. floor((t + n - m)/2)`; edlib uses the same initial
range (`edlib.cpp:755`: `min(k, (k + queryLength - targetLength) / 2)`).

**Mechanism and lev today.** `banded_body` computes rows `[j - t, j + t]`, i.e. diagonals
`[-t, t]` (`lib.rs:1769`, `1778`, `1786-1788`), width 2t + 1. A path of cost <= t ending on
diagonal s = n - m only touches `[-(t - s)/2, (t + s)/2]`, width t + 1. Certification is
unchanged (same Ukkonen argument as `band_limit`). Replace t by `(t - s)/2` in `last`/`needed` and
by `(t + s)/2` in `first`; nothing else changes (pessimistic boundaries stay valid because the
tight band is a subset).

**Estimate.** Blocks per column at t0 = max(64, s + 32) (`lib.rs:1855`): s = 0, t = 64: always 3
blocks now, always 2 tight (-33%). s = 100, t = 132: ~5 now, ~3 tight (-40%). Per-column overhead
(index math, peeled last block, give-up test) stays, so the band pass gets ~20-35% faster.

**Regime.** R5 similar pairs that Idea A does not catch (d above ~1.3·sqrt(n)) and every failed
pass before the full matrix. All arches/kinds, Levenshtein only (the LCS band is already tight).

**Gain** 15-35% on band-certified long pairs. **Effort** S. **Risk** low: correctness via the
oracle tests at band edges (s = 0, s = t, odd t - s); keep the gate `band_blocks(t) * 3 <= w`
(`lib.rs:1856`) unchanged in the first experiment so the set of banded pairs does not move.
**Test.** `long-*-similar`, `benchmark_by_length.py` 1024-8192 with 1-10% edits, ucs2 versions.
Falsified if under 10% on those. **Conflicts.** None (the rejected item lowered the gate; this
does not).

### Idea D. Edlib's dynamic threshold and block pruning inside the band

**Source.** `edlib.cpp:792-794` (tighten k each column: `k = min(k, score + max(remaining
columns, remaining rows))`), `:799-830` (add the next block only if its first row can still be
<= k; drop bottom/top blocks whose score is >= k + 64 or that fall outside the gap-cost volume
`g(u) + c_gap(u, v_t) <= k`; Spouge's computational volume 4 in A*PA2 p. 6), Ukkonen's cutoff
(Hyyrö PSC'02 p. 49).

**lev today.** `banded_body` keeps t fixed for the whole pass and only tracks the score of the
last block (`lib.rs:1809`); blocks are never dropped by score. Open item "non-speculative
edlib-style block pruning".

**Estimate.** Needs one score add per block step (+1 op on ~20) and a few compares at the band
ends per column. When d ≪ t (common: t0 = s + 32 even for d = 5), live blocks fall from 2-3 to
1-2 after the first columns, and the tightened k shrinks the static range from Idea C further.
Expect a further 10-30% on band-certified pairs, more on passes that would otherwise fail late.

**Effort** M. **Risk** medium: extra ops per block on pairs where pruning never triggers (d ≈ t);
the `>= k + 64` rule and the "+1" fudge edlib needs (`edlib.cpp:817`) show the bookkeeping is
error-prone; test band edges heavily. **Test** as Idea C, plus pairs with d ≈ 0.9·t. Falsified if
the gain over Idea C is < 5%.

### Idea E. A*PA2 column blocks: lane-major sweeps inside the band

**Source.** A*PA2 §3.1 (p. 7: decide the row range once per block of B = 256 columns; inside a
block, compute one 64-row lane over all B columns before the next lane) and §4.3 (p. 15-16:
blocks give 2.5x over Edlib-style per-column ranges on long reads, SIMD another 2.7x).

**Mechanism for lev.** In `banded_body` each block step loads and stores `pv[k]`, `mv[k]` from
memory and every column recomputes `needed`/`first` and runs the give-up test (`lib.rs:1776-1824`).
Lane-major order keeps a lane's `pv/mv` in registers for 256 columns and passes only the 2-bit
horizontal delta per column between lanes through a small buffer, exactly the edge array of the
AArch64 strips. The give-up test can move to block boundaries.

**Estimate.** Removes 2 loads + 2 stores per block step and most per-column bookkeeping: maybe
1.2-1.5x on the band pass (A*PA2's 2.5x includes memory effects of storing whole columns that lev
never had). **Effort** M. **Risk** coarser band (rounding to 256-column blocks adds rows), more
code; combine after C/D. **Test** as Idea C. Falsified if < 10% over C+D.

### Idea F. A*PA2 anti-diagonally staggered SIMD lanes for the x86 full matrix

**Source.** A*PA2 §3.3 (p. 8): four 64-bit words per AVX2 vector, staggered so lane k works on
column j - k; two vectors (8 lanes, 512 rows) for ILP; SIMD gave 2.7x (p. 16). The Myers step
is 20 vector instructions per vector (app. A.1, fig. 10a).

**New angle versus the rejected AVX2 kernels** (`kernels/mod.rs:68-76`): those kept the words of
one column in lanes, so carries rippled through lane masks on the critical path. Staggering makes
lane k + 1 consume lane k's horizontal output from the previous step: one lane rotate (`vpermq`)
per state vector per step, no intra-step ripple. Using Myers' hin form (hin < 0 injected into bit 0
of eq, as in `banded_step`) avoids the addition carry between words.

**Estimate (x86, AVX2).** Per step of 4 lanes: ~20 ALU vector ops + 2 rotates + building eq for 4
different text chars (4 scalar loads + ~4 inserts from the `bases` array that `run_body` already
builds, `lib.rs:1859`; A*PA2 notes gathers are "not always efficient", and post-2023 Intel
microcode makes `vpgatherqq` slow) ≈ 30 uops for 4 word-steps, ~7.5 uops per word-step versus
~17-20 scalar uops (`portable::full`, `kernels/portable.rs:313-338`). On 3 vector ports that is
~2.5 cycles per word-step against ~4.5-5 scalar: 1.5-2x for R5 dissimilar on x86, less at W 5-8
(prologue/epilogue of 3-7 steps per strip, padding lanes). AVX-512 with `vpternlogq` (~14 ops per
8 lanes) would be ~3-5x but only on Zen 4/5 and Intel server/Ice-Rocket Lake clients.
**Not for AArch64:** NEON holds 2 lanes, so ~10 vector ops per word-step on 4 pipes (Apple) or 2
(N2) versus 15 scalar ops on 6 (Apple) or 4 (N2) ALUs: a tie on Apple and V2, a loss on N2. The
AArch64 strips stay.

**Effort** L. **Risk** high: eq construction cost, code size, AVX2 runtime dispatch, layout. This
is perf-ideas idea 3; the literature confirms the mechanism but the regime (long dissimilar
natural-language pairs) is narrow. **Test** `long-ascii-2048-dissimilar`, `ucs2-1024-dissimilar`,
`benchmark_by_length.py` 256-4096 dissimilar on the x86 A/B runners. Falsified if < 20% over
`portable::full` at W >= 16.

### Idea G. Bidirectional diagonal transition (middle snake)

**Source.** Myers 1986 §4 (linear-space "middle snake"), BiWFA (Marco-Sola et al. 2023).
**Mechanism.** Run forward and reverse wavefronts alternately and stop when they overlap on a
diagonal; each side only reaches cost ~d/2, so the d² term halves (2·(d/2)² ≈ d²/2). Extends
Idea A's profitable range from d ≈ 1.3·sqrt(n) to ≈ 1.9·sqrt(n). **Effort** M on top of A
(overlap test per diagonal, reverse extend). **Risk** more code for a second-order gain.
**Test** sweep pairs with d near the Idea A cap. Do after A, only if A's cap is often hit.

### Idea H. Two chains packed in one word for short pairs (low confidence)

**Source.** Packing from Hyyrö-Fredriksson-Navarro 2005 §4.1 (p. 6: AND VP with
`ZM = (0 1^{m-1})^r` before the addition and HP/HN before the shifts to stop interference) plus
Hirschberg 1975's forward/backward split (perf-ideas idea 4).
**Mechanism.** For m <= 31 and n ≈ m (where Idea B cannot help), put the forward pattern in bits
0..31 and the reversed pattern in bits 32..63, feed `eq_f(b[j]) | eq_r(b[n-1-j]) << 32`, and run
n/2 steps; add `| F` (first-bit mask) after the shifts for the +1 top boundary. Then combine the
two halves' final columns (prefix sums, min). **Estimate** on Apple: (6 + ~1.5) cycles × n/2 plus
a ~40-60 cycle combine versus 6n: m = n = 30 gives ~150 versus 180 cycles (1.2x), better only for
n ≈ 60. **Effort** S-M. **Risk** the combine eats the gain at R1 sizes; mask work doubles. Probably
below the 3% bar at R1; test only if Idea B leaves R2 n ≈ m as the next hotspot. Falsified if <
5% at m = n = 32 on M2.

### Idea I. Bound-and-align: set the band from a measured bound instead of t0 = s + 32 and doubling

**Source.** QuickEd (Doblas et al. 2025): compute a cheap upper bound first, then one banded pass
with that bound (O(n·ŝ)). A*PA2 "future work" 2 (p. 17) says the same.
**Mechanism for lev.** After Idea A gives up at wavefront s (cost already > s with pace data), or
after a failed small band (`Err((j, score))`), the extrapolated score is a far better t0 than
`max(64, s + 32)` (`lib.rs:1855`) and avoids failed passes (`next_band`, `lib.rs:1649-1651`).
**Gain** small to medium on long moderately similar pairs that today need 2+ passes. **Effort** S
once A exists. **Risk** an underestimate costs one more pass (same as today). Overlaps the open
item "a fallback that reuses a failed band's work".

## 3. Ideas that will not pay off for lev (and why)

- **A*PA2 bit-transposed profile** (§3.4, p. 8): eq = AND over b bit-planes of `C_j XOR Eq'`.
  b = 2 for DNA; for natural language b = 7 (ASCII), 16 (UCS-2) or ~6 after dense remapping, i.e.
  ~11-13 ops per lane versus one table load. The paper itself limits it to alphabet size 4 (p. 16).
- **Incremental doubling** (§3.8): A*PA2's ablation says it slows simple alignments (fig. 11
  caption, p. 22); lev's failed passes already stop early, so little is recomputed.
- **Pre-pruning and the gap-chaining seed heuristic** (§3.6): k-mer seeds of 12-15 on DNA with a
  hash of all k-mers; on short natural-language strings the setup dominates (A*PA2-simple is
  faster than -full on 1 kbp reads, p. 14).
- **Diagonal-transition traceback, sparse block-boundary memory** (§3.2, §3.5): lev computes no
  alignment.
- **BitPAl encoding**: same 20 instructions as Myers (A*PA2 app. A.1).
- **GenASM**: Bitap (Wu-Manber) costs O(k·n·m/w), worse than Myers for exact global distance; its
  windowing is a hardware-accelerator design for approximate long-read alignment.
- **Shifted Hamming Distance**: a filter (2e + 1 shifted XOR masks); as a gate for Idea A it costs
  more (e.g. 37 SIMD passes at e = 18) than DT's own 2-4-wavefront pace check.
- **Four Russians (Masek-Paterson 1980)**: the word-RAM bit-parallel algorithm is already the
  practical form; table-lookup blocks lose to it on modern caches.
- **SETH lower bound (Backurs-Indyk 2015; LCS: Abboud et al. 2015, Bringmann-Künnemann 2015)**:
  no strongly subquadratic exact algorithm, so R5 dissimilar only improves by constants (Idea F,
  ILP); all big wins must be output-sensitive (Ideas A, C, D, I) or from bit utilization (Idea B).
- **Shorter Myers or LCS chains**: covered in section 1; nothing below lev's current forms.
- **Multi-pattern packing / Sassy text chunks with overlap**: both rely on search (local
  occurrences), where chunks can overlap by m + k; a global DP cannot split the text without a
  boundary column (the reason perf-ideas idea 4 stops at two chains).
- **Hirschberg two-chain kernel at W = 2**: measured and rejected (briefing); the m <= 64, n ≫ m
  variant is dominated by Idea B.
- **Farrar striped SIMD, anti-diagonal byte-cell SIMD**: designed for affine/scored DP; for unit
  costs the bit-parallel step packs 64 cells per ~15 ops, which byte-cell vectors do not beat.

## 4. The two arXiv papers

**arXiv 2111.08692** (Lemire, Unicode at Gigabytes per Second): SIMD UTF-8/UTF-16 validation and
transcoding (simdutf), 12-byte shuffle-table routine inside 64-byte blocks, ASCII fast paths.
lev's Python path never transcodes (PEP 393 buffers are read in place); only the Rust API's
`Units::new` transcodes non-ASCII `&str` (`lib.rs:179-240`), which is not a priority interpreter
path. Not relevant to time per Python call.

**arXiv 2610.17406** (Lemire, Fast Forgiving Base64 Decoding with SIMD, October 2026), every idea
checked against lev:
1. 64-character blocks so the classification mask fits one GPR (§4.2 "Why 64", p. 6): lev's match
   masks already follow this shape.
2. NEON 64-byte to 64-bit mask via AND with a 0x01..0x80 pattern and four rounds of pairwise
   adds (§5.2, p. 9): ~13 vector ops versus lev's 4 `cmeq` + 3 `bit` + `shrn` + move
   (`kernels/aarch64/neon.rs:86-113`, ~9). lev's is cheaper; no change.
3. Single-set-bit fast path built arithmetically (§5.1, p. 9; ablation: ~30% on M4, §6.9 p. 20):
   the analogue is special-casing the dominant easy case, which lev does (affix strip, mbleven);
   Idea A generalizes it. No direct transfer.
4. Clean-block fast path versus buffered path (§4.2, §6.8): same pattern as lev's speculative
   bands; the paper's lesson that a fused pass beats strip-then-process supports running Idea A
   before building any peq table.
5. AVX-512 VBMI `vpermi2b` 128-entry in-register lookup and `vpcompressb` (§5.3, p. 10): one could
   hold a byte-sliced 1 KiB ASCII peq in 16 zmm registers, but producing per-char u64 masks needs an
   8x64 byte transpose (~24 uops per 64 chars) to save one 1 KiB zeroing. No gain.
6. SVE2/RVV (§5.4, p. 10-11): predicates have no movemask, and the deployed SVE2 cores (Neoverse
   V2/V3, N2) use 128-bit vectors like NEON. No case for an SVE path in lev.
7. ILP evidence: the NEON kernel retires 5.3-5.8 instructions/cycle on M4 (§6.8, p. 19), consistent
   with lev's off-chain mask work being nearly free on Apple; it does not change the latency-bound
   single-word chain, which Idea B attacks instead.
8. Runtime selection of the widest kernel (§5): lev already does this for x86 (v2 copy plus AVX2
   checks); an x86-64-v3/v4 copy (perf-ideas 5b) is where AVX-512 mask compares would live.

## 5. Ranked table

| Rank | Idea | Regime | Est. gain | Effort | Confidence |
|---|---|---|---|---|---|
| 1 | A. Diagonal transition + packed extend (distance; Myers O(ND) for ratio), pace-gated | R3/R4 similar (d ≲ 1.3·sqrt(n)), R5 similar; all kinds, all arches | realistic 2-4x kernel; long similar 10-100x; ratio similar 2-4x; 0 on 25%-edit pairs | M | medium-high |
| 2 | B. Orientation swap (row-wise tiling): longer string in the bit vectors | R1/R2 with n > m, R3 with small m mod 64 and n ≫ m; both metrics | 1.5-3x at n >= 2m (model), 0 at n ≈ m | S (u8, n <= 512), M (all) | medium-high |
| 3 | C. Ukkonen-tight band in `LargeCtx::banded_body` | R5 similar beyond Idea A's cap, failed passes | 15-35% of the band pass | S | high |
| 4 | D. Edlib dynamic k + block pruning in the band | R5 similar | +10-30% over C | M | medium |
| 5 | I. Bound-and-align: band t0 from DT/small-band failure data | R4/R5 moderately similar needing 2+ passes | 0-40% on those pairs | S after A | medium-low |
| 6 | E. A*PA2 lane-major column blocks in the band | R5 similar | 1.2-1.5x over C+D | M | medium-low |
| 7 | F. Staggered AVX2/AVX-512 lanes, x86 full matrix | R5 dissimilar, x86 only | 1.5-2x (AVX2), more on AVX-512 | L | medium-low |
| 8 | G. Bidirectional DT | extends A's range | ~2x on the d² term near A's cap | M | medium |
| 9 | H. Packed forward/backward chains in one word | R2 n ≈ m, m <= 31 | 0-20% | S-M | low |

## 6. Top three

1. **Diagonal transition (Idea A).** The literature's O(n + d²) algorithms (Ukkonen 1985, Myers
   1986, WFA) are absent from lev, whose only small-d shortcut (mbleven) is limited to d <= 3 and
   gated by a Hamming bound that any early indel defeats. Work counts on lev's own benchmark pairs
   show 64 diagonal steps and 81 word compares for the realistic pair (versus a 3-word table plus
   192 columns), and 25-64 steps for the 2048/8192-char similar pairs that today run a 30-110-word
   band. Ratio benefits even more because it has no small-d shortcut. The cost is a pace-gated
   failure on dissimilar pairs, which must be bounded by gating on m and aborting after 2-4
   wavefronts.
2. **Orientation swap (Idea B).** Fredriksson's row-wise tiling and Hyyrö's "reverse the roles"
   remark point at a simple fix: when the shorter string leaves most of a word idle, iterate over
   the shorter string and put the longer one in the bit vectors. It reuses the existing single-word,
   multiword and table-free kernels, cuts steps from n to m·ceil(n/64), and replaces a
   latency-bound chain with ILP-friendly multiword columns; the model gives 1.5-3x for short
   queries against texts at least twice as long, with no change at n ≈ m. It is cheaper and
   stronger than perf-ideas idea 4 in that regime.
3. **Tight Ukkonen band in LargeCtx (Idea C), then edlib pruning (Idea D).** `banded_body` covers
   diagonals [-t, t] where Ukkonen's rule (used by Hyyrö, edlib and lev's own small band) needs
   only [-(t - s)/2, (t + s)/2]; at t0 that is 2 instead of 3 blocks per column for s = 0 and 3
   instead of ~5 for s = 100. It is a few lines with the same certification proof, and edlib's
   per-column threshold tightening and block dropping can follow as a second experiment.
