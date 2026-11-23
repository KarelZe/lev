# Novel bounds: exact methods that do less DP work (lev)

Agent: novel-bounds. Code read: `main-src` (origin/main 5a6859c). No lev wall-clock runs were made.
All numbers below are work counts from simulations (cells, LCE words, block-steps, word-steps),
turned into Apple M2 cycle estimates with an explicit, stated cost model.

## TL;DR

1. **LargeCtx's Ukkonen band is twice as wide as its certificate needs.** `banded_body` computes rows
   `[j - t, j + t]` (src/lib.rs:1778, 1786-1788) to certify `d <= t`, but every path of cost `<= t` stays in
   rows `[j - s - (t-s)/2, j + (t-s)/2]` (Ukkonen 1985; edlib and RapidFuzz use this tight form). The tight
   shape cuts block-steps by 33% at t = 64 and 38-41% at t >= 128, on every long-similar pair. Checked
   exact on 331 fuzz pairs (602 passes): it is exact and always certifies when `d <= t`. Effort S.
2. **Diagonal transition (Landau-Vishkin for distance, Myers 1986 O(ND) for ratio) with word-wide LCE,
   on a budget, before the table is built.** For m > 704 it costs O(d^2 + n/8) instead of about 3 blocks
   per column. Model: 48x on `ascii-2048-similar`, 57x on `ascii-8192-similar`, 29x on
   `ucs2-2048-similar`, 3-11x on long prose with word typos. When it gives up, it wastes under 0.5%. In the
   medium regime it is a speculative trade-off. It is 2.7x on the realistic benchmark pair, but costs up
   to +10..20% on near-miss pairs.
3. **A non-speculative upper bound from a 64-diagonal band, then one tight pass at t = U.** On every
   simulated prose and random pair, the 64-diagonal band's final score equalled d (U/d = 1.000-1.013).
   Long pairs with 5-40% divergence currently fall back to the full matrix. With one certain pass at
   t = U they take 0.29-0.69x the model cycles: 1.5-3.4x faster.
4. **(b) Headroom on dissimilar text exists, but only for m >= ~1000 and mainly on x86.** With the gap
   heuristic and U from (3), 48-52% of the 64-row blocks per column are needed on unrelated English prose
   at m = 1024-2000. With 8-word strips (AArch64) it is 64-81%. Bag, seed (A*PA) and q-gram heuristics
   add at most 4 points at d/n ~ 0.8.
5. **Negative results.** These give no gain: lower bounds that decide whether a band is viable, the
   Hirschberg "completion" that reuses a failed band, O(1) exits for short pairs, and A*-style seed
   heuristics. Details and numbers below.

## Simulation setup (reproducible)

Directory: `/private/tmp/claude-501/-Users-markusbilz-Documents-git-lev/33906bae-f780-4c7b-8ea7-23a2aa9e4dd7/scratchpad/proto/bounds/`

- `make_pairs.py`: builds `pairs.tsv` (11,413 pairs).
  - English prose: real docstrings of 100 CPython stdlib modules, 423K chars.
  - Families per length (20 to 2000):
    - random char edits at 5%, 20% and 50% (`edit5`, `edit20`, `edit50`);
    - word-level typos at 3% and 10% of words: transposition, drop, double, neighbour key
      (`typo3`, `typo10`);
    - unrelated text of equal length and of 1.5x length (`unrel`, `unrel15`).
  - Product names and addresses (unrelated and typo'd), dictionary words and proper names (unrelated
    and typo'd).
  - CJK-like text: Zipf over 3000 code points, unrelated and with 5% edits.
- `make_named.py`: `named.tsv`, lev's own cases. It copies `_rand_str` and `_mutate` from
  `tests/test_bench.py`, so the strings are byte-identical, and adds the realistic, long-similar and
  long-dissimilar pairs from `scripts/benchmark*.py`. `all.tsv` = both files.
- `sim.rs`: std-only Rust, built with `rustc -O --edition 2021 sim.rs -o sim`. Run it as
  `./sim <tsv> <mode>`, where mode is one of `lv`, `tight`, `prune`, `band`, `lcs`, `short`, `bail`,
  `named`, `ubband` or `fuzz`. The knobs `LV_BETA`, `LV_FACTOR`, `LV_E0`, `LV_MINM` and `LV_E1DIV` tune the
  LV bail rule. Each mode runs in under 30 s.
- It emulates, at cell level:
  - lev's normalization and affix strip;
  - the `small_ub`/mbleven gate;
  - `small_band`/`small_band2` with lev's exact give-up schedule and `band_gives_up`;
  - `LargeCtx::run_body` with `banded_body`'s pessimistic block boundaries and give-up test, in both
    band shapes.
  It also runs Landau-Vishkin, Myers O(ND), and full forward and backward DP.
- Every exactness claim is asserted in code against the full DP or LV:
  - LV and ND against the DP on all pairs up to 2.5M cells;
  - both band shapes in `fuzz` on 331 random pairs: alphabets of 2-27 symbols, 0-900 edits including
    40-char block insertions, m 513-1600;
  - every certified pass in `tight` and `ubband`.
- Outputs are saved next to the code: `out_*.txt`.

**Cost model (Apple M2, cycles per unit).** I chose these and they are uncertain; the ranking rests mostly
on work ratios.

| Unit | Cycles |
|---|---|
| Single-word Myers column or one-word band column (briefing: about 6 cycles per column) | 6 |
| `small_band2` column | 9 |
| Multiword or full-matrix word-step | 3 |
| `banded_step` block-step (hin/hout chain plus overhead) | 5 |
| Direct-table zeroing | 30·W |
| LV/ND cell (3 neighbour reads, max, clamp, first 8-byte compare) | 6 |
| Each extra 8-byte LCE word | 1.5 |
| One mispredict per edit | 14 |

Where a conclusion depends on these constants I say so.

---

## Idea 1: tight Ukkonen band in `LargeCtx::banded_body`

- **Source.** Ukkonen 1985 (doi:10.1016/S0019-9958(85)80046-2). edlib (Šošić & Šikić 2017) and RapidFuzz
  `levenshtein_hyrroe2003_block` use the asymmetric band, with a first last-block of
  `min(max, (max + len_diff)/2)`. In lev it is untried: lev's own small band uses the tight form
  (`band_limit`, src/kernels/mod.rs:135), but LargeCtx does not.
- **Mechanism.** In column j, a cell on diagonal δ = j - i lies on a path of cost at least
  `|δ| + |s - δ|`, so with `d <= t` every optimal path stays in `δ ∈ [-(t-s)/2, s + (t-s)/2]`. Set:
  - `needed` = block of row `j + (t-s)/2`;
  - `first` = block of row `j - s - (t-s)/2`;
  - initial `last` = block of row `(t-s)/2`.
  Nothing else changes: pessimistic boundaries, score tracking and the give-up test stay as they are.
- **Correctness.** The give-up test `score - max(excess, 0) > t` (src/lib.rs:1820-1823) stays valid for
  any band that contains the optimal path. Proof: let r be the optimal path's last row in column j and
  x = b - r. Then `score <= D(r,j) + x <= d - |e - x| + x <= d + max(e, 0)`. The `fuzz` mode confirms it:
  all 602 passes with `d <= t` certified and were exact in both shapes.
- **Regime.** R5 similar (m > 704), all kinds, distance only (`lcs_banded` is already tight in its own
  parameters).
- **Gain (block-steps; `out_tight.txt`, `out_ubband.txt`).**
  - Ratio tight/current is 0.66-0.67 at t = 64, which is 2 blocks per column instead of 3. It is
    0.59-0.62 when passes reach t >= 128.
  - `ascii-2048-similar`: 5620 to 3741 block-steps. `ascii-8192-similar`: 20926 to ~13900.
    `long-similar` (benchmark_by_version, d = 180): 17138 to 10086.
  - Per-column overhead (base load, score update, test) is unchanged, so wall time should fall by about
    20-33% on these cases.
- **Effort.** S: 3 expressions in src/lib.rs:1769/1778/1787, the doc comment at 1735-1743, and
  `band_blocks`.
- **Risk.** Low.
  - Keep the `band_blocks(t) * 3 <= w` gate (src/lib.rs:1856, 1862) numerically unchanged in the first
    experiment. A tighter count would allow passes at W 9-11, which touches the rejected "lower the
    LargeCtx gate" result.
  - Code layout of `banded` is the usual hazard.
- **Test.** `benchmark_by_version.py` long-similar, plus CodSpeed `ascii-2048-similar`,
  `ascii-8192-similar` and `ucs2-2048-similar`. Falsified if the gain is under 10% on these, or if
  `ascii-640` or the long-dissimilar cases move beyond noise. Run `banded_long_strings_match_oracle`
  and the Python property tests.
- **Conflicts.** None.

## Idea 2: diagonal transition with word-wide LCE, on a budget, before the table is built

- **Source.**
  - Landau & Vishkin 1989, Ukkonen 1985, and Myers 1986 "An O(ND) difference algorithm" (the indel form).
  - WFA (Marco-Sola et al., Bioinformatics 2021) and Heng Li's
    [editdist-U85](https://github.com/lh3/editdist-U85) ("comparable to edlib"). These are standard in
    genomics.
  - A RapidFuzz read of `Levenshtein_impl.hpp` found no diagonal-transition path; it has mbleven, the
    small band and blockwise Ukkonen.
  - Untried in lev, where mbleven is the d <= 3 special case.
- **Mechanism.**
  - For e = 0, 1, ..., keep for each diagonal k in [-e, e] the furthest row reachable with e edits. Each
    cell is `max(fr[k]+1, fr[k-1], fr[k+1]+1)`, then a longest-common-extension.
  - The LCE is lev's `common_prefix` (32-byte blocks, then 8-byte XOR plus ctz; src/lib.rs:1069-1081,
    already used by mbleven).
  - Stop when diagonal s reaches row m: then e = d exactly, a non-speculative certificate.
  - Cost is about (d+1)^2 cells plus about n/8 LCE words. Ratio uses Myers 1986 (parity diagonals,
    about (D+1)(D+2)/2 cells).
  - Budget:
    - E = sqrt(β · C_est / 6), where C_est is the a-priori cost of lev's path. Restrict to the diamond
      `e + |k - s| <= E`.
    - Bail by pace: projected `e·n/progress > factor·E`.
    - Also cap total LCE words. Periodic text extends many diagonals; LCE work is otherwise not bounded
      by the cell budget.
  - On a bail, pass the certified lower bound `d > e` and the pace estimate on to the band driver.
- **Placement.**
  - m > 704: in `compute_u8` / `compute_sorted` before `hyrro_multiword_direct` / `_hash` is called. The
    LargeCtx arm builds a `vec![0; ROWS*w]` table (src/lib.rs:1619-1622), 32 KB at W = 32, which LV
    skips entirely.
  - Optionally 150 <= m <= 704, as an experiment.

**Results (`out_named.txt`, `out_lv.txt`, `out_bail*.txt`).**

| Case | m, d | lev path, model cyc | LV/ND work | LV model cyc | Model speedup |
|---|---|---|---|---|---|
| ascii-2048-similar | 1913, 4 | large-band 29,000 | 25 cells, 262 words | 599 | 48x |
| ascii-8192-similar | 7006, 7 | large-band 107,930 | 64 cells, 936 words | 1,886 | 57x |
| ucs2-2048-similar | 1609, 4 | large-band 24,570 | 25 cells, 426 words | 845 | 29x |
| prose typo3 / 2000 | ~1500, 13 | 23,500 | 206 cells | 2,004 | 11.7x (median 13.5x) |
| prose typo10 / 2000 | ~1800, 36 | 27,900 | 1,475 cells | ~8-13K | 2.1-3.6x |
| realistic (benchmark.py) | 190, 7 | band1 1,242 | 64 cells; ND: 45 | 461 | 2.7x |
| long-similar (by_version) | 2008, 180 | 86,650 | bails | +0.1% | 1.0x (Idea 1 / 3 apply) |
| ascii-2048-dissimilar | 2048, 1816 | 198,673 | bails at e = 2 | +0.05% | 1.0x |

- **Failure cost.** For m > 704 bails cost 0.0-0.5% on every dissimilar or near-miss pair (`edit5-2000`
  0.1%, `edit20-2000` 0.0%, `unrel-2000` 0.0%).
- **Medium regime (65-704).** This is a speculative trade-off with the same structure as #260. Best rule
  found: β = 0.5, factor 1.2 from e = 3, e = 1 bail if progress < n/32.
  - Wins: the realistic pair 2.7x; `typo3/256` 0.59x model cycles; `typo3/512` 0.14x; `typo10/256` 0.66x;
    `cjk edit5/128` 0.82x.
  - Losses on near-miss pairs, where d sits just above E: `prose edit5/256` (d ~ 12) +16..20%;
    `cjk edit5/300` +18%.
  - Losses on lev's 25%-edit CodSpeed cases: `ascii-100` +21%, `latin1-100` +10-23%, `ucs2-100` +9-19%,
    `ucs4-200` +5-11%, `quote-300-sim` +3-9%.
  - LV for m <= 64 does not pay (`ascii-48-d6` 0.70x). mbleven's gate misses d <= 3 pairs in only
    0-4.7% of short typo pairs (`out_short.txt`).
- **Effort.** M: one function per metric (~150 lines) plus the LCE and the budget.
- **Risk.**
  - Periodic or low-entropy strings make LCE long on many diagonals; cap LCE words.
  - Mixed-kind pairs need a per-unit LCE (16-unit blocks via `block_mismatches`) or should skip LV.
  - In the medium regime the failure cost is real. The 6-cycles-per-cell constant decides the
    break-even there, but not for m > 704, where ratios are 10-50x.
- **Test.**
  - Rust: oracle tests for LV/ND, plus periodic strings and s up to E.
  - Benchmarks: CodSpeed long-similar trio and `benchmark_by_version` medium/long.
  - Falsified for m > 704 if `ascii-2048-similar` gains under 5x or any long-dissimilar case regresses
    beyond noise.
  - In the medium regime, keep only if the realistic pair gains >= 30% and no `_MEDIUM`/`_MIXED`
    CodSpeed case regresses > 3%. I expect this second bar to fail with the rules above.
- **Conflicts.** "Indel mbleven for ratio" was rejected (script enumeration, 1.7-2.4x slower). Myers ND
  is a different algorithm, with cells proportional to D², not the number of scripts. Not otherwise
  related to the rejected list.

## Idea 3: non-speculative upper bound from a 64-diagonal band, then one certain tight pass

- **Source.** Novel as a source of U. The pruning or certification framework is Ukkonen and edlib.
  edlib and A*PA2 double k or f_max instead.
- **Mechanism.**
  - When `run_body`'s optimistic pass fails (src/lib.rs:1862-1866), run a one-word sliding band to
    completion with no pace give-up: `portable::small_band` without `band_gives_up`, or `small_band2` for
    s < 128. Every band cell bounds its true value from above, so its final score is a certified
    U >= d.
  - Then run one tight pass (Idea 1) at t = U. It certifies by construction, with no further speculation.
  - Run this only if the tight band's block count at t = U is below about 0.6W (cost parity with the full
    matrix). Otherwise go straight to the full matrix.
  - Gate the U pass itself on the optimistic pass's extrapolation (`next_band`). For unrelated text that
    predicts a band wider than the gate, so skip the 6n band cost.
- **Why it works.** On every simulated prose and random pair, U equals d (U/d = 1.000; at most 1.013 on
  unrelated prose). Balanced indels keep the optimal path within ±32 diagonals. Today pairs with
  d/n ≈ 0.05-0.4 get `next_band` extrapolations whose ±t band fails the `band_blocks(t)*3 <= w` gate, so
  they run the full W words per column.
- **Gain (`out_ubband.txt`, model cycles, proposal/current).**
  - `prose edit20-2000` (d ≈ 360): 0.29-0.41.
  - `rand-2048-e300` (d = 277): 0.34.
  - `edit20-1024`: 0.52.
  - `rand-2048-e600`: 0.51. `rand-4096-e600`: 0.55.
  - `edit50-2000`: 0.66-0.69. `edit50-1024`: 0.76-0.79.
  - Similar pairs: 0.62-0.68, from Idea 1 alone.
  - Ungated unrelated pairs pay +6% (W = 32) or +11% (W = 16) for the U pass. That is why the gate is
    needed; gated, the change is 1.00.
  - With edlib-style gap pruning at the same U, instead of the geometric band, `edit50-2000` would need
    0.22 of the blocks rather than 0.37 (`out_prune.txt`).
- **Regime.** R5 (m > 704), moderately divergent long pairs, all kinds. Distance; the ratio analogue is a
  64-diagonal LCS band for a lower bound on LCS, which bounds indel from above.
- **Effort.** M: a no-give-up mode for the small band on LargeCtx data (its stride-w table already fits
  the `small_band` contract), plus the driver logic.
- **Risk.**
  - Pairs with long block indels give a loose U; the cost gate falls back to the full matrix, and the
    worst case is the 6n band pass, 2/W of the full matrix.
  - Layout of `run_body`.
- **Test.** Add CodSpeed cases for 2048-char pairs with 300 and 600 random edits and for prose with
  20% edits. Falsified if they gain under 30% or `ascii-2048-dissimilar` regresses.
- **Conflicts.** Do not lower the LargeCtx gate below W = 12 (rejected, +78%). This idea stays above it.

## Idea 4 (question b): block pruning in the full-matrix regime for dissimilar pairs

**Question.** Is there exact headroom below m·n/64 on dissimilar natural text?

**Simulation (`./sim pairs.tsv prune`, `out_prune.txt`).** For each column, the contiguous range of
64-row blocks containing any cell with `D + h <= U` (exact forward D) is counted, as a fraction of W.

| Pair (prose, Levenshtein) | d/n | gap, U = d | gap, U = band64 | gap, U = Hamming | max(gap, bag, seed3) | coarse (edlib bottom score) | tight Ukkonen t = U | 8-word strips, gap | perfect h |
|---|---|---|---|---|---|---|---|---|---|
| unrel / 256 | 0.80 | 0.64 | 0.64 | 0.73 | 0.60 | 0.66 | 0.77 | 1.00 | 0.25 |
| unrel / 512 | 0.82 | 0.57 | 0.57 | 0.68 | 0.53 | 0.58 | 0.72 | 1.00 | 0.14 |
| unrel / 1024 | 0.79 | 0.51 | 0.52 | 0.64 | 0.49 | 0.52 | 0.68 | 0.81 | 0.07 |
| unrel / 2000 | 0.79 | 0.47 | 0.48 | 0.61 | 0.46 | 0.49 | 0.65 | 0.64 | 0.03 |
| unrel15 / 2000 | 0.73 | 0.51 | 0.51 | 0.76 | 0.50 | 0.52 | 0.68 | 0.66 | 0.04 |
| edit50 / 2000 | 0.38 | 0.22 | 0.22 | 0.67 | 0.21 | 0.22 | 0.37 | 0.41 | 0.03 |
| indel unrel / 2000 | 1.17 | 0.67 | 0.67 | n/a | 0.66 | 0.67 | 0.84 | 0.81 | 0.04 |

**Findings.**

- There is real headroom. Unrelated English prose has d/n ≈ 0.8, close to the "all mismatches" rate, so
  any off-diagonal detour costs about 1.2 per diagonal of offset. In the second half of the matrix the
  gap heuristic prunes everything far from the diagonal: about half the blocks at m >= 1024.
- The cheap, block-edge-only test (edlib's bottom-score bound) loses only 1-2 points against exact
  per-cell values. U from a 64-diagonal band is as good as the oracle U = d. The Hamming bound is much
  worse.
- A*-style admissible heuristics add at most 4 points at this divergence: bag distance (character
  histograms of suffixes) and the A*PA seed heuristic with k = 3 or 4. This matches A*PA's own limit:
  seeds stop helping once the error rate exceeds about 1/k. **Not worth implementing.**
- Turning work into time:
  - **x86:** the Levenshtein full matrix is already `portable::full`'s per-column loop over all words
    (src/kernels/portable.rs:313). An edlib-style first/last adjustment adds O(1) per column (popcounts
    of the first and last block plus the tracked boundary scores), not per word. Estimate at m >= 1500:
    `0.49 × full + band pass (6n ≈ 6% at W = 32) + about 10 cycles per column ≈ 0.62-0.68 × full`, so
    −25..−35%. At m ≈ 1000 about −10..−20%.
  - **AArch64:** the strips (8 words) quantize the region. The strip column-interval form needs 0.64 of
    the steps at m = 2000 and 0.81 at 1024. That is about −20..−25% at m >= 2000 and nothing at 1024
    once the U pass and per-strip bookkeeping are paid.
  - **W <= 8** (65-512): the fraction is 0.57-0.82 and the U pass costs 2/W. No gain. Do not do it.
- Ratio (LCS) has less headroom (0.67-0.78 for unrelated text), and the LCS step is cheaper. Low value.
- **Source.** edlib's Ukkonen cutoff (Šošić & Šikić 2017), with the non-speculative U from Idea 3.
- **Effort.** M (x86 portable loop), L (AArch64 strips).
- **Risk.** Per-column bookkeeping cost on x86 (4 integer ALUs). Layout of `kernels::full`, where
  `if cfg!` placement is load-bearing (src/kernels/mod.rs:299-306).
- **Test.** CodSpeed x86 instruction counts on `ascii-2048-dissimilar` plus a new unrelated-prose
  2048-char case. Falsified if under 15% on x86 at m = 2048.
- **Conflicts.** It is the "non-speculative edlib-style block pruning" open item, applied to the full
  matrix rather than W <= 8.

## Question (a): failure cost of the speculative small band

- **Measured (`./sim pairs.tsv band`, `out_band.txt`).** Prose with uniform random edits; band waste in
  model cycles as a share of the full kernel.

| m | Edits | Band waste |
|---|---|---|
| 150 (W = 3) | 40-220 | 13-25% (give-up at 20-37% of n) |
| 300 (W = 5) | any | 4-10% |
| 640 (W = 10) | any | 1.3-3.6% |

  Closed form for uniformly spread edits with d = L(1+ε) and limit L: the pace rule fires at about
  `j_f/n ≈ 2 / (ε(L - s))`. That is 0.32n at ε = 0.1 and 0.13n at ε = 0.25. Averaged over
  d ∈ [L, 2L], that gives about 0.14n wasted band columns.
- **(i) Decide viability with a cheap lower bound: negative.**
  - Bounds tried: bag distance plus the q-gram count lemma (q = 2, 3), as in a GateKeeper or
    SneakySnake-style pre-alignment filter.
  - On the band's failing pairs they exceed the limit in only 2 of 71 near-miss failures (d 64-100).
    They fire mostly (33 of 36) on m >= 300 pairs with d >= about 1.6 × limit, and the pace check already
    rejects those at 1-4% waste.
  - They cost O(m + n) with a count table, about as much as the band. Not recommended.
- **(ii) Reuse a failed band's columns.**
  - Forward restart is not certifiable. Band cells with `D_band <= min(2k_lo + 1 + δ, 2(s + k_hi) + 1 - δ)`
    are exact; I derived this and asserted it in code. But a near-miss pair's optimal path may leave the
    band before j_f. Nothing computed rules that out, so any forward restart certifies at most what the
    band itself does.
  - Hirschberg completion does certify. Run the full kernel backward over columns `(j_f, n]` and combine
    at `j_f`. Out-of-band rows need only their two boundary rows, because `|δ| + R` is non-decreasing
    away from the band. This certified 92-100% of failures.
  - It loses at m <= 300 once the reversed table (30W + 1.5m cycles) and the combine are charged:
    +22..28% versus the current +13..25% at m = 150. At m = 640 it gives −7..+2% versus +1.3..3.6%. Net
    about 5 points on a class that is already cheap. **Not worth it**, consistent with the rejected
    two-chain Hirschberg measurement.
- **(iii) Non-speculative pruning.** See Ideas 3 and 4. For W <= 8 it does not pay.
  The useful lever for the medium regime is Idea 2's LV, which removes the band entirely for small d. For
  near-miss pairs no exact method found here beats "band, then full".

## Question (c): short pairs (R1/R2), O(1) exits (`out_short.txt`)

| Family | m after strip | d = n | Disjoint alphabets | lb = ub certified (n − overlap = n − best-shift aligned matches) |
|---|---|---|---|---|
| Random dictionary words | 6.3 | 38% | 9.2% | 24.5% |
| Random proper names | 4.5 | 53% | 31.2% | 56.5% |
| Product names and addresses (unrelated) | 26-38 | 0% | 0% | 0% |
| Typo'd words | 1.5 | 46% | 21% | 21% |

- In the typo'd word row the exits only fire on 1-2-char residues left after affix stripping.
- **Assessment: negative.** The certificate needs the overlap (a greedy matching from the match masks)
  and the best-shift aligned matches. Both cost about 2-3 cycles per text char before the chain, against
  a tiny-path chain of about 6 cycles per char for n ≈ 7 (about 40 cycles of a 140-170-cycle call).
  Expected net is −10 to +3 cycles per call. The disjoint-alphabet exit alone saves at most about 40
  cycles on 9-31% of short dissimilar pairs.
- Computing the bounds inside the chain, on the free ALUs, saves nothing, since the chain has already
  run.
- One exception, a benchmark artifact: `benchmark_by_version`'s long-dissimilar pair uses disjoint
  alphabets (d = n exactly). An O(n) disjointness check, from a 256-bit pattern bitmap taken from peq
  construction, would make it about 100x faster but says nothing about real data. I recommend replacing
  that pair with unrelated prose (d/n ≈ 0.8) rather than optimizing for it.

## Question (d): ratio (LCS / indel)

- **Myers 1986 O(ND)** is Idea 2's ratio half.
  - The realistic pair takes 45 cells. Today it runs the W = 3 LCS kernel (576 word-steps plus a table),
    for an estimated 2-3x.
  - Long similar pairs take 21-36 cells against `lcs_banded`'s 4 words per column (src/lib.rs:1663,
    1884): `ascii-2048-similar` ND = 21 cells and 258 words versus about 7.7K word-steps.
- **LCS small band** (64 diagonals sliding, one word). The design follows the `lcs_banded` comment
  (src/lib.rs:1876-1883):
  - Shift the state right by one per column, enter a 1 bit at bit 63, and add the dropped row's
    `!s & 1` to an accumulator.
  - Carry-in is 0 because frozen rows above emit no carry.
  - It certifies when `m + n - 2l <= s + 2·k_lo`.
  - Simulated certification at 65 <= m <= 512 (`out_lcs.txt`): 100% of `edit5`, `typo3`, `typo10` and
    `cjk edit5`; 83-100% of `edit20` at m <= 256; 0% of `edit20` at 512 and of `edit50`/`unrel`.
  - It pays only at W >= 4: the LCS full step is about 1.5-2 cycles per word, so W = 2-3 is already
    about one band column. Its similar-pair niche overlaps with ND.
  - This is an existing open item; my contribution is the certification rule and the coverage data.
- **Alphabet compaction for LCS (exact, novel for lev).** Text chars absent from the pattern (and pattern
  chars absent from the text) contribute nothing to the LCS: with `pm = 0` the column is a no-op
  (`u = 0`, so `s` stays unchanged; src/lib.rs:1340-1348). So `LLCS(A, B) = LLCS(A|Σ_B, B|Σ_A)`, and
  ratio uses the original lengths.
  - Word-step ratio after compaction: 0.27-0.58 on CJK-like unrelated pairs (`out_lcs.txt`); 0.93-0.98
    on English prose (useless there).
  - Cheapest form: filter the `bases` vector that `lcs_run` already builds (src/lib.rs:1937). Drop rows
    equal to the zero row: `n_keys * w` for the hash path, the empty row for direct tables. That is about
    1 cycle per char against W ≥ 9 words per column, so roughly free on English and 1.7-3.7x on CJK
    dissimilar.
  - For W <= 8 (`hash_kernel_body`) it needs branchless compaction into a stack buffer and pays only
    when W >= 4 and more than ~25% of chars drop. Gate on a count taken while resolving rows.

## Other notes (e)

- **Carry-free step for no-match Levenshtein columns.** With `eq = 0` the Myers step has no addition:
  `vp' = !vn & (vp << 1)`, `vn' = vn & ((!vp << 1) | 1)`. So those columns have no inter-word carry chain
  and could be vectorized across words. 24-53% of columns qualify on CJK-like dissimilar pairs, about 2%
  on English. A per-column branch would mispredict at a 50/50 mix. Speculative, low priority, untested.
- **LV's cheap lower bound** (`d > e` at a bail) can seed `run_body`'s first `t`. That is minor.

---

## Ranked table

| # | Idea | Regime | Est. gain | Effort | Confidence |
|---|---|---|---|---|---|
| 1 | Diagonal transition (LV / Myers ND) with word LCE, budgeted, before the table (m > 704) | R5 similar, all kinds, distance and ratio | 10-57x on lev's long-similar cases (model), 3-11x on typo'd long prose; < 0.5% when it bails | M | High on direction, medium on magnitude (6-cycles-per-cell constant) |
| 2 | Tight Ukkonen band shape in `banded_body` | R5 similar (band passes) | −20..−33% wall (−33..−41% block-steps) | S | High (proved, fuzzed) |
| 3 | 64-diagonal band to completion → U, then one certain tight pass at t = U | R5 with 5-40% divergence (today: full matrix) | 1.5-3.4x (model 0.29-0.69) | M | Medium-high (U = d on all sims; needs real timings) |
| 4 | Gap-pruned full matrix with U from #3 (edlib-style first/last per column) | R5 dissimilar, m ≥ ~1000, x86 first | −25..−35% x86, −20..−25% AArch64 at m ≥ 2000; 0 at W ≤ 8 | M (x86) / L (strips) | Medium |
| 5 | Myers ND for ratio, medium regime | R3 similar, ratio | 2-3x on realistic and typo pairs; same speculative failure class as #6 | M (shared with #1) | Medium |
| 6 | LV first stage in the medium regime (150 ≤ m ≤ 704) | R3/R4 similar | 2.7x realistic pair, 1.5-7x typo pairs; +10..20% on near-miss and 25%-edit pairs | S once #1 exists | Low-medium (likely fails the no-regression bar) |
| 7 | Alphabet compaction for LCS | Ratio, UCS-2/4 dissimilar (CJK), m > 512 first | 1.7-3.7x fewer word-steps on CJK dissimilar; ~0 on Latin text | S-M | Medium (depends on real CJK alphabet overlap) |
| 8 | LCS small band (one-word sliding) | Ratio, 65-512 similar, W ≥ 4 | 2-4x on similar pairs where ND bails | M | Medium |
| 9 | Carry-free no-match column step | CJK dissimilar, Levenshtein | unknown, ≤ 30% | M | Low |
| ✗ | Band-viability lower bounds (bag, q-gram) | R3 near-miss | none: fires on 2/71 near-miss failures | | High (negative) |
| ✗ | Hirschberg completion of a failed band | R3 near-miss | negative at m ≤ 300, ~5 points at 640 | | High (negative) |
| ✗ | A*PA seed or bag heuristics for dissimilar text | R3/R5 dissimilar | ≤ 4 points of extra pruning | | High (negative) |
| ✗ | R1 O(1) exits (disjoint alphabets, lb = ub) | R1 | net −10..+3 cycles per call | | High (negative) |

## Top 3

1. **Diagonal transition first for long pairs (#1).**
   - lev's band costs about 3 blocks per column no matter how small d is. LV costs (d+1)^2 cells plus
     n/8 compare words, and it skips the 2-32 KB peq table altogether.
   - On the CodSpeed long-similar trio the model says 29-57x. It bails for under 0.5% on everything else
     at m > 704.
   - Myers 1986 is the same code shape for `ratio`.
   - Cap LCE words for periodic text, and keep the medium-regime variant (#6) as a separate experiment,
     because there it re-creates #260's failure class.
2. **Tight band shape (#2).**
   - A provably correct three-expression change: rows `[j - s - (t-s)/2, j + (t-s)/2]` instead of
     `[j - t, j + t]`.
   - It removes a third or more of the block-steps of every LargeCtx band pass. It also helps the pairs
     LV hands back, such as the by_version long-similar pair with d = 180.
   - Fuzzed: exact on 602 passes, and it always certifies when d <= t.
3. **Band-derived upper bound plus one certain pass (#3).**
   - A 64-diagonal band run to the end gave U = d on every simulated pair.
   - With that bound the band driver needs no more speculation for long pairs with 5-40% divergence,
     which today fall to the full matrix. One tight pass at t = U certifies, for 1.5-3.4x.
   - The same U is what makes #4's x86 full-matrix pruning (about half the blocks on unrelated prose at
     m >= 1000) possible, which answers question (b): there is headroom, but only for long pairs and
     mostly on x86.
