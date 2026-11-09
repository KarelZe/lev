# Competitors and implementations: survey for lev

Agent: competitors. Code read from `scratchpad/main-src` (origin/main 5a6859c). Competitor sources are
shallow clones in `scratchpad/proto/competitors/<repo>` (read only, nothing built or run). The only
thing executed was a 60-line pure-Python oracle check (`scratchpad/scripts/check_ideas.py`, 3000 random
pairs, passed) that confirms the two algorithmic claims below: Myers with the *longer* string in the
bit-vector gives the same distance, and Landau-Vishkin diagonal transition is exact. No wall-clock runs.

Cycle estimates use lev's own documented costs: single-word step ~6 cycles/column on Apple, latency
bound (`kernels/aarch64/apple.rs:9-12`); multi-word ~15-17 ops per word, throughput bound from W = 2
(`kernels/x86_64/bmi1.rs:49-51`), ~3.3-3.5 cycles/word on M2 at the ~4.6 IPC cap noted for the rejected
Hirschberg kernel; single-word LCS chain 3 ops.

## 1. What each implementation does

### rapidfuzz-cpp (b3ed64f, 2026-11-03) and rapidfuzz-rs, python-Levenshtein, RapidFuzz Python
python-Levenshtein is a thin wrapper over RapidFuzz (`Levenshtein/src/Levenshtein/__init__.py:23-31`);
rapidfuzz-rs is a port of the same C++ (`rapidfuzz-rs/src/distance/levenshtein.rs:435-1104`).
- Short (shorter <= 64): textbook Hyyrö 2003 with a per-column score update
  (`rapidfuzz/distance/Levenshtein_impl.hpp:259-322`), ~19 ops/column. PatternMatchVector =
  256-entry direct array + 128-slot CPython-style hash for chars >= 256, zero-filled per call
  (`details/PatternMatchVector.hpp:62-134`, 4 KiB).
- mbleven2018 only when the caller passes `score_cutoff < 4` (`Levenshtein_impl.hpp:172-239, 928`).
- Band: `levenshtein_hyrroe2003_small_band` (`:419-505`) when `2*max+1 <= 64`, and a variant without a
  pattern table that maintains the band's match masks online in a (last position, mask) map, lazily
  shifted (`:507-636`). Both only reachable with a cutoff or hint.
- Multiword: `levenshtein_hyrroe2003_block` with edlib-style first/last block band driven by `max`
  (`:641-850`), hint doubling from 31 (`:881-895`).
- LCS: `lcs_unroll<N>` up to 8 words, `lcs_blockwise` band from the cutoff
  (`LCSseq_impl.hpp:205-364`); LCS mbleven for max_misses < 5 (`:67-139`).
- SIMD only across pairs (`levenshtein_hyrroe2003_simd`, `lcs_simd`, `:324-416`, `LCSseq_impl.hpp:141-201`).
- Affix strip by `std::mismatch`-style loop (`details/common_impl.hpp:38-80`).
- Unicode: code points via CPython kinds; no normalization.

### edlib (0ddc23e, 2025-05)
- Myers blocks in the hin/hout form (`edlib/src/edlib.cpp:412-447`, ~22 ops), Ukkonen band with k
  starting at 64 and doubling (`:198-217`).
- NW kernel tightens k every column (`k = min(k, score + max(remaining))`, `:790-794`) and prunes first
  and last blocks by cell value (`score >= k + 64`) and by diagonal (`:796-831`), plus a "strong reduce"
  every 2048 columns (`:835-870`).
- Alphabet compaction of both strings into dense indices; Peq has sigma + 1 rows, heap allocated
  (`:358-384`).
- Python binding: ASCII via UTF-8 encode, otherwise a Python-level dict mapping that fails above 256
  distinct chars (`bindings/python/edlib.pyx:9-52`).

### editdistance (9e05bc6, 2024-02)
- Python `hash()` of every element into a malloc'd int64 array (`src/editdistance/bycython.pyx:14-26`),
  `std::map` pattern table (`_editdistance.cpp:101-112`), textbook multiword Myers up to 10 words
  (`:30-56`), O(mn) DP beyond.
- Orientation: puts the **longer** string into the bit-vectors as long as it fits 10 words and only then
  swaps (`_editdistance.cpp:124-153`). Probably not designed for speed, but it is the transposition of
  idea C1 below.

### polyleven (2638be8, 2026-10)
- mbleven for k < 4 with a cutoff; ASCII shorter <= 64: textbook Myers with a 256-entry table
  (`polyleven.c:269-304`); otherwise row strips of one word each with the per-column carries stored
  as bits (`:210-267`) and a 128-slot hash per 64-row block (`:140-202`). Code points via
  `PyUnicode_READ`.

### StringZilla (d0c8dea, 5.3.2 on main; v5.0.0 notes fetched)
- Single pair: delegated to the serial Myers on every ISA ("one pair gains nothing from NEON batching",
  `include/stringzillas/similarities/neon.hpp:79-89`). Serial Myers is textbook Hyyrö with 1/2/4/8-word
  unrolled tiers and a per-column score (`serial.hpp:2150-2211`), no affix strip, no band.
- Table setup: zeroes only the entries of chars that occur in either string, then clears the pattern's
  entries after use, so a reused scratch never needs a full memset (`serial.hpp:2165-2172, 2207-2208`).
- The "4 interleaved NEON chains" are four `uint64x2_t` groups = 8 **different pairs** in the
  cross-product engine; measured on M5 Pro 60-byte all-pairs: 1 group 117.6, 2 → 165.5, 4 → 195.5,
  8 → 181.2 GCUPS (`neon.hpp:66-75`), i.e. 1.66x. Eq words are gathered with scalar loads into a stack
  array (`neon.hpp:147-159`). Ice Lake: 8 pairs per zmm, Myers boolean ops fused into VPTERNLOGQ
  (`icelake.hpp:1303-1315`). No SVE edit-distance kernels (release notes).
- Anti-diagonal u8/u16/u32/u64 "walkers" are used only for non-unit costs or beyond the SIMD Myers
  tiers; the authors' own comment: Myers is "~5x DP on short unit-cost pairs, and still ~25x the
  anti-diagonal DP" above 512 (`serial.hpp:2636-2642`).
- Unicode: byte Levenshtein by default; the UTF-8 variant takes an ASCII fast path, else transcodes to
  UTF-32 runes and hashes them (`serial.hpp:2809-2854`). Code points, no normalization.
- Published numbers: 100-byte DNA, one core, all-pairs: rapidfuzz.process.cdist 4,970 / 18,370 MCUPS vs
  StringZilla 15,680 / 22,844 MCUPS (Xeon4 / M5 Pro), "a first impression" (`README.md:83-86`). Batch
  only; no single-pair figure.

### A*PA2 / pa-bitpacking (Groot Koerkamp, bf2e14e)
- Step: edlib hin/hout block form, "20 operations, excluding eq" (`pa-bitpacking/src/myers.rs:24-54`),
  SIMD twin `compute_block_simd` (`:60-88`). No add carry between blocks: the incoming -1 is ORed into
  eq bit 0, so a lane never needs a carry-out.
- Staggered lanes: lane k works on block k at column i - k; the horizontal deltas move one lane per
  iteration by a lane rotate plus one scalar insert (`src/simd.rs:66-75, 215-284`). Eq for DNA comes
  from a 2-plane bit-sliced profile read with unaligned window loads (`src/profile.rs:79-153`); for the
  general "scatter" profile each lane's eq is a scalar load (`src/simd/scatter_profile.rs:169-216`).
- Own notes: "row::<2> seems best", "row::<3> and row::<L> use too many registers probably; almost 2x
  slower", 1 block scalar 1.4 vs 8 blocks in 2 x 4 lanes 3.2 (`src/simd.rs:9-31`, `src/lib.rs:3-15`):
  ~3.5x per block-step for 8 lanes vs one scalar block. The checked-in `bench.json` (scalar only, x86)
  shows scalar "N columns at once" staggering (`cols_ru/lh/4`) only 5-10% faster than plain `col`, and
  one-word row strips (`row`) 1.5x slower than `col` for the scatter profile at 512 rows (parsed with
  `scratchpad/scripts/parse_bench.py`): the same direction as lev's rejection of x86 strips.
- Op counts vs lev: lev's single-word step is 15 ops with a 7-op chain on Apple
  (`kernels/aarch64/apple.rs:9-55`), 9 ops with shifted operands on Neoverse
  (`kernels/aarch64/neoverse.rs:49-59`), ~15 with `andn` on x86 (`kernels/x86_64/bmi1.rs:36-47`);
  lev's multiword is ~17 ops/word (`kernels/portable.rs:47-76`). A*PA2's 20-op block is not leaner;
  its advantage is purely the SIMD lanes.

### Sassy (750ab7b, 2026-10)
- Approximate search (semi-global), but its core trick is the transposed orientation: the **text** is
  bit-packed 64 chars per word and the **pattern** is iterated ("Its main novelty is tiling these in the
  text direction", `README.md:29`; `src/search.rs:1103-1130`). For ASCII, the pattern's distinct chars
  get a dense index (`src/profiles/ascii.rs:18-29`) and each 64-byte text block gets one 64-bit mask per
  distinct pattern char by two 32-byte compares + movemask (`:76-88`). No table indexed by the alphabet.
- Text chunks in SIMD lanes work for search because matches restart; for a global distance that needs a
  combine at every chunk boundary (the perf-ideas idea 4 problem), so only the orientation transfers.

### triple_accel (0f2119a) and rust-bio `alignment::distance::simd`
- Banded anti-diagonal DP with u8 cells (u16/u32 on overflow), band from an exponential search
  starting at k = 30 (`triple_accel/src/levenshtein.rs:714-827, 1445-1454`), AVX2/SSE4.1 only, bytes
  only. rust-bio's simd distance is this crate (`rust-bio/src/alignment/distance.rs:63-140`). README
  claims "up to 20-30x speedups over their scalar counterparts" (vs scalar DP, not vs Myers).

### block-aligner (4fcf630)
- Adaptive block DP with i8/i16 cells for affine protein alignment; not a Levenshtein engine.

### WFA2-lib (f285a8f) and editdistancek (crates.io, nkkarpov)
- Diagonal transition (Landau-Vishkin / WFA) for "edit" and "indel" metrics: next wavefront is a
  vectorizable max of three neighbours (`WFA2-lib/wavefront/wavefront_compute_edit.c:72-98`), then
  each diagonal slides along matches 8 bytes at a time with xor + ctz
  (`wavefront_extend_kernels.c:65-91`; AVX2/AVX-512 forms extend 8 diagonals with gathers,
  `wavefront_extend_kernels_avx.c:94-170`). Exact pruning of diagonals for unequal lengths
  (`wavefront_compute_edit.c:219-275`). O(n s) worst, O(n + s^2) expected.
- editdistancek is the same algorithm in 40 lines with a 32/16-byte SIMD LCP (`src/lib.rs:15-54,
  58-123`): "O(d*min(n,m)) and O(d^2 + n) in expectation" (`README.md:8-10`).

### strsim, jellyfish, edit-distance and similar crates
- Plain O(mn) Wagner-Fischer (`strsim-rs/src/lib.rs:229-265`, `jellyfish/src/levenshtein.rs`). Nothing
  to take.

## 2. Comparison table

Legend: Y yes, N no, "x" = across pairs only (not usable for one pair per call).

| impl | short <= 64 | multiword 65-512 | long > 512 similar | long dissimilar | band / cutoff | affix strip | small-d special | SIMD in one pair | table setup | Unicode |
|---|---|---|---|---|---|---|---|---|---|---|
| **lev** | tiny path <= 16 (SWAR/NEON/SSE2, no table); 17-64 table-free NEON / AVX2 <= 32, else zeroed table; hash for UCS-2/4 | const-W kernels, Apple funnel / Neoverse / BMI1 forms | 64- and 128-diagonal speculative bands to m <= 704, then Ukkonen block band (edlib form) | full matrix: AArch64 8-word strips, x86 portable loop | internal only | Y, 32-byte blocks | mbleven ub <= 3, m <= 512 | mask building, affix, Hamming gate | direct 128/256/257 rows, 64/128-slot hash, dense rows for wide multiword | code points, CPython kinds in place, mixed pairs unwidened |
| rapidfuzz-cpp | textbook Hyyrö + per-column score | `lcs_unroll`/hyrroe block | block band only with cutoff/hint | blockwise | with cutoff: small band (incl. online masks), edlib-style block band | Y (scalar) | mbleven only with cutoff < 4 | x (cdist) | 256 direct + 128 hash zeroed (4 KiB) | code points |
| edlib | Myers block form, k doubling | same | Ukkonen band, per-column k tightening, value pruning | same kernel | Y | N | N | N | alphabet compaction, heap Peq | bytes; Python remaps <= 256 symbols |
| editdistance | textbook multiword, **longer string as pattern** up to 640 | same | O(mn) DP > 640 | DP | N | N | N | N | `std::map` of Python hashes | `hash()` per element |
| polyleven | textbook, 256 table | one-word row strips, bit-packed carries | strips | strips | N | N | mbleven with cutoff | N | 128-slot hash per block | code points |
| StringZilla | serial textbook Myers (single pair) | 1/2/4/8-word unrolled | generic Myers | generic Myers | N | N | N | x (4 NEON groups, 8 zmm lanes, ternlog) | zero only touched entries | bytes or transcoded runes + hash |
| pa-bitpacking / A*PA2 | edlib block form, 20 ops | staggered SIMD lanes (4 or 8), 2.7x claimed | A* + bands (aligner) | staggered lanes | via A* | N | N | **Y** (staggered lanes) | bit-sliced DNA profile / per-block scatter | DNA/ASCII |
| Sassy | transposed: text in bit-vectors | text chunks in lanes (search) | n/a (search) | n/a | k cutoff | N | N | Y (text chunks, search only) | SIMD compare per distinct pattern char | ASCII/DNA |
| triple_accel | anti-diagonal u8 band | same | exp. search k from 30 | u16/u32 DP | Y | N | N | Y (anti-diagonal) | none | bytes only |
| WFA2 / editdistancek | diagonal transition | same | O(n + s^2) | O(ns) | k bound (editdistancek) | implicit in first slide | the whole algorithm | WFA2: AVX2 gathers across diagonals; LCP 16/32 bytes | none | bytes |
| strsim / jellyfish | O(mn) | O(mn) | O(mn) | O(mn) | N | N | N | N | none | chars |

## 3. Ideas lev lacks

### C1. Transposed orientation for short-vs-long pairs (m <= 64, n well above m)
- **Source**: editdistance `_editdistance.cpp:124-153` (longer string as pattern), Sassy
  `README.md:29`, `src/search.rs:1103-1130`, `src/profiles/ascii.rs:18-29, 76-88`.
- **Mechanism**: Levenshtein and LCS are symmetric, and Myers works for any text length
  (D(m, n) = n + sum of the last column's vertical deltas holds for n < m too; oracle-checked). Put the
  *long* string into the bit-vectors and iterate the *short* one. Work goes from n latency-bound
  single-word steps to m * ceil(n/64) throughput-bound word steps, a factor ~m/64 fewer word steps
  plus the latency-to-throughput change. When both fit one word, the step count drops from max(m, n) to
  min(m, n) with the existing single-word kernel.
- **State in lev**: `compute` always makes the shorter string the pattern (`lib.rs:373-374`);
  `compute_u8` sends m <= 16 to the tiny path and 17-64 to `word_u8` over all n text chars
  (`lib.rs:826-839`), above 8 text chars per pattern char through a zeroed table
  (`kernels/mod.rs:158, 174-195`); wide kinds probe the hash once per text char
  (`lib.rs:1217-1274`). Nothing ever transposes.
- **Estimate (Apple, kernel part)**: `ascii-24-vs-384` (tests/test_bench.py): now 384 x ~6 = ~2.3k
  cycles. Transposed with the existing direct table (W = 6): 6 KiB zeroing + 384 inserts (~500) +
  24 x 6 x 3.5 = ~500, ~1.0k total: ~2x per call. With a Sassy-style builder (masks only for the ~16
  distinct chars of the short string, one 64-byte compare + pack per char and word, ~10 NEON ops):
  ~200 + ~500, ~3x. `ascii-24-vs-96`: 576 vs ~300 (W = 2 is near latency bound), ~1.6-2x. Both <= 64,
  e.g. m = 8 vs n = 48: 48 x 6 = 288 vs 8 x 6 + NEON pattern setup ~30 = ~80 cycles, ~200 cycles
  (~60 ns) saved per call. `ratio` (3-op LCS chain): 384 x 3 = 1.15k vs 24 x 6 x ~1.2 + setup,
  1.3x with the table, ~2.5x with the compare builder. x86 (Zen, 4 ALUs, ~7-cycle chain) and Neoverse
  (~5-cycle chain, ~4.3 cycles/word): steady-state ratio is m * 4.5 / (64 * 7) and m * 4.3 / (64 * 5),
  i.e. 3-4x fewer cycles at m = 24, break-even near m = 64 to 100.
- **Wide kinds**: transposed, the short side has <= 64 distinct units, so the long side's masks come
  from `vceqq_u16`/`_mm_cmpeq_epi16` (or u32) compares against each distinct unit: no hash at all on
  the n side (today n probes, `lib.rs:1255-1272`).
- **Regime**: R1/R2 with n >= ~2m (short query vs longer string), both metrics, all kinds; also
  m <= 64 against n > 512 (transposed full matrix, `kernels::full`/`lcs_full` with W = n/64, no bands).
- **Supersedes** perf-ideas idea 4 (two chains, <= 2x and only on Apple): transposition removes work
  instead of overlapping it. Does not conflict with any rejected item (the "dense byte->row index" was
  for the non-transposed table).
- **Effort**: M (gate + guarded entry: `small_bands` and `LargeCtx::run_body`/`lcs_run` assume n >= m,
  `lib.rs:1532, 1855, 1932`; they must be skipped on the transposed path). Builder is an extra S-M.
- **Risk**: gate tuning per arch (W = 2 is latency-bound, so n/m must be >= ~2; table zeroing vs
  compare builder); one extra compare on the hot tiny path (sub-ns); layout sensitivity of
  `compute_u8`.
- **Test**: cases m in {4, 8, 16, 24, 48, 64} x n in {2m, 4m, 16m, 64m}, ASCII/Latin-1/UCS-2/UCS-4/mixed,
  distance and ratio; existing `ascii-24-vs-96/-384`; regression guard on n ~ m (kitten, ascii-32,
  ascii-48-d6). Falsified if the transposed path is not >= 1.3x at n >= 4m on M2 and the A/B x86 runner.
  Oracle: swap property (distance(a, b) via both orientations) plus RapidFuzz diff test.

### C2. Diagonal transition (Landau-Vishkin / WFA "edit" and "indel") with SIMD LCP, for long similar pairs
- **Source**: editdistancek `src/lib.rs:15-54` (bounded LV) and `:58-123` (32/16-byte LCP);
  WFA2-lib `wavefront_compute_edit.c:72-98` (edit next-wavefront), `wavefront_extend_kernels.c:65-91`
  (blockwise LCP), `wavefront_compute_edit.c:219-275` (exact prune); WFA "indel" metric for `ratio`.
- **Mechanism**: for e = 0, 1, ... keep the furthest row reached on each diagonal with cost e, take the
  max of the three predecessors, then slide along equal code units with 16/32-byte compares and ctz.
  Exact when it reaches (m, n) on diagonal n - m; O((d + 1)^2) diagonal steps plus O(n / 16) compares
  along the path. The indel variant (no mismatch move) gives D = m + n - 2 LCS for `ratio`.
- **State in lev**: mbleven is a brute-force special case for ub <= 3 and m <= 512
  (`lib.rs:854-878, 1093-1141`); beyond that every similar pair runs bit-parallel bands: 64/128
  diagonals per column for 65 <= m <= 704 (`lib.rs:1521-1550`), and for m > 704 an Ukkonen block band
  at t0 = max(64, n - m + 32) (`lib.rs:1854-1869`) that runs 3-4 blocks of `banded_step` with the
  state in memory every column (`lib.rs:1762-1831`), ~15-20 cycles/column. `ratio` has no small band at
  all (`lib.rs:1465`); its m > 512 band costs ~4 words/column (`lib.rs:1663-1665, 1884-1906`).
- **Estimate**: per diagonal step ~5 cycles (loads, max, one 16-byte compare that mostly exits at
  once), plus ~2.5 cycles per 16 matched bytes and a mispredict per long slide.
  `ascii-2048-similar` (4 edits): after affix strip m ~ 1.2k, band ~1.2k x 15-20 = 18-25k cycles;
  LV: 25 steps + 77 compares + mispredicts ~ 0.4k. `ascii-8192-similar` (8 edits): ~6.5k x 18 = ~120k vs
  ~2k. That is 10-40x per call including call overhead and the affix scan. R4 (513-704, small band at
  ~7 cycles/column): d <= 15 → 2-5x. `ratio` long similar: 10-30x for the same reason. R3 similar
  (realistic pair m = 190, d = 7): ~0.45k vs ~1.6k (band + table), 1.5-2x per call, but see risk.
  LV loses to the bands once (d + 1)^2 * 5 exceeds n * (band cycles per column): d > ~0.6 sqrt(n) in R3,
  d > ~1.5 sqrt(n) in R5.
- **Gating**: (1) start at m > 512 for both metrics, where an early abort costs < 1% of any
  alternative (dissimilar R5 pays >= 40 n cycles for the full matrix). Abort when the furthest
  anti-diagonal reach falls behind a pace (e.g. < 8 units per edit after e >= 8, ~80 steps, ~400
  cycles) or a work budget of ~n/4 compares is spent (bounds periodic strings like `(ab)^k`, whose
  slides run on every other diagonal). (2) R3/R4 later, triggered only after the existing small band
  passes its column-16 pace check (`kernels/mod.rs:104-128`), so dissimilar pairs, which give up at
  column 4, never pay for LV.
- **Effort**: M (one scalar kernel, two metrics, a blockwise LCP for same-kind bytes, which lev has in
  `common_prefix_bytes`, `lib.rs:510-542`, and a blockwise mixed-kind LCP; today's mixed `common_prefix`
  is scalar, `lib.rs:1074-1079`).
- **Risk**: failure cost on medium-d pairs (bounded by the budget, same structural trade-off as #260);
  code size and layout; the R3 gate. Algorithm correctness is easy (oracle check passed).
- **Test**: `benchmark_by_version.py` long-similar, `ascii/ucs2-2048-similar`, `ascii-8192-similar`,
  near-identical pairs 128-8192 with 1-64 edits incl. indels, periodic strings (`"ab" * k` shifted,
  `"x" * 400` affix cases), long-dissimilar (must stay within noise). Falsified if long-similar does not
  gain >= 3x or long-dissimilar loses > 1%.
- Not in conflict with any rejected item (it is not a band; the rejected Hirschberg two-chain kernel is
  unrelated).

### C3. Staggered SIMD lanes (perf-ideas idea 3): new angles from A*PA2, StringZilla, WFA2
- Already on the list; only the new points:
  - Use the hin/hout block form in the lanes (`pa-bitpacking/src/myers.rs:60-88`): no carry-out per
    lane. With lev's Hyyrö carry form, AVX2 must synthesize an unsigned 64-bit carry-out (no
    `vpcmpuq`), as StringZilla's NEON multiword does with two `vcltq_u64` (`neon.hpp:279-285`), adding
    ~4-5 ops per word step.
  - Eq for a general alphabet is a per-lane scalar gather (`scatter_profile.rs:198-201`,
    StringZilla `neon.hpp:149-159`); the DNA bit-sliced profile does not generalize (8 planes for bytes
    would cost ~15 ops per eq).
  - Two vectors in flight is the sweet spot; three or more ran out of registers ("almost 2x slower",
    `simd.rs:20-24`).
  - x86-64-v4: VPTERNLOGQ cuts the lane step from ~21 AVX2 ops to ~14 (StringZilla
    `icelake.hpp:1303-1315`).
  - Scalar staggering (several columns at once in GPRs) measured only 5-10% in A*PA2's own data; do not
    pursue it as a cheap substitute.
- Expected: 1.3-2x on x86 full matrix (W >= 4, long dissimilar), unproven for NEON (2 lanes per vector).
  Effort L, confidence low-medium. C1 and C2 shrink this regime's share of user time only for
  dissimilar long pairs, so this remains the x86 R5-dissimilar lever.

### C4. Online band masks (no table build for the band) for 65 <= m <= 704
- **Source**: rapidfuzz `levenshtein_hyrroe2003_small_band` without a pattern table
  (`Levenshtein_impl.hpp:507-636`): each pattern char keeps (last position, mask); per column the char
  entering the window is shifted in and the text char's mask is read with one shift.
- **State in lev**: the band reads funnel-shifted rows of the full W-word table, which is zeroed and
  filled first (`lib.rs:1586-1597`, hash variant `lib.rs:1999-2060`).
- **Estimate**: saves the table build on band-success pairs: at m = 300 ASCII (W = 5) ~5 KiB zeroing +
  300 inserts, ~0.3-0.45k of ~2.5k cycles, 10-15%; less at W = 2. Extra ~6-8 ops per column stay off
  the Myers chain. Failure still builds the table, so dissimilar pairs pay only the band's existing
  early give-up.
- **Effort** M, **risk**: a second band form to maintain; overlap with C2 (C2 takes small-d pairs, C4
  then only helps moderate d). Confidence low-medium. Test: realistic pair, near-identical 128-704 with
  5-60 edits, 20%-edit pairs (must not regress).

### C5. Touched-entry table reset with a persistent scratch
- **Source**: StringZilla `serial.hpp:2165-2172, 2207-2208`. Same as the "TLS scratch" items already on
  the list; StringZilla is prior art for clearing only what was dirtied. Gain 2-6% on R3 multiword
  (16 KiB zeroing at W = 8 Latin-1 vs a ~14k-cycle kernel), TLS access cost on macOS eats part of it.
  Low priority.

### C6. edlib per-column k tightening and value-based block pruning
- **Source**: `edlib.cpp:790-870`. Already open on the list. With t0 = 64 the tightening rarely bites
  before the band reaches the bottom row, and the value test `score >= k + 64` does not drop blocks
  while k ~ t. C2 covers the regime far better. Mark as not worth doing before C2.

### Will not pay off for lev
- **Anti-diagonal u8/u16 SIMD DP** (triple_accel, StringZilla walkers, block-aligner): one AVX2 vector
  per anti-diagonal is ~13+ cycles per text char at the narrowest band vs lev's ~6-7 per column; the
  StringZilla authors measure Myers 5-25x faster than their own anti-diagonal kernels
  (`serial.hpp:2636-2642`).
- **Interleaved chains across pairs** (StringZilla 4 NEON groups, RapidFuzz cdist SIMD): needs a batch
  API, which is out of scope.
- **Bit-sliced profiles** (A*PA2 `BitProfile`): only cheap for sigma <= 4.
- **Alphabet compaction** (edlib) and per-block hashes (polyleven): the rejected dense byte index again.
- **Hirschberg / BiWFA**: memory savings for alignments; lev returns only a score.
- **Text tiling across lanes** (Sassy): a global distance needs a combine per chunk boundary.
- **Step forms**: no competitor has a leaner Myers or LCS step than lev's (Section 1, A*PA2 bullet).

## 4. Ranked table

| rank | idea | regime | est. gain | effort | confidence |
|---|---|---|---|---|---|
| 1 | C2 diagonal transition (LV/WFA edit + indel), SIMD LCP, gated m > 512 first | R4/R5 similar, both metrics (R3 later behind the band's pace check) | R5 similar 10-40x per call; R4 2-5x; R3 similar 1.5-2x | M | medium (algorithm certain, gain estimated) |
| 2 | C1 transposed orientation (long string in the bit-vectors) + Sassy-style per-distinct-char mask builder | R1/R2 with n >= 2m, m <= 64 vs n > 512; all kinds; both metrics | 1.5-3x on short-vs-long pairs (`ascii-24-vs-96/-384`), ~60 ns saved at m = 8, n = 48 | M | medium-high |
| 3 | C3 staggered lanes with hin/hout form (on list; new angles) | R5 dissimilar, x86 W >= 4 | 1.3-2x on x86 full matrix | L | low-medium |
| 4 | C4 online band masks | R3/R4 band successes | 5-15% | M | low-medium |
| 5 | C5 touched-entry reset / persistent scratch (on list) | R3 multiword setup | 2-6% | S-M | low |
| 6 | C6 edlib k tightening / value pruning (on list) | R5 similar | < C2, likely < 3% after C2 | M | low (do not pursue) |
| - | anti-diagonal DP, cross-pair chains, bit-sliced profiles, compaction, Hirschberg, text tiling | - | none or negative | - | high that they do not pay |

## 5. Top 3

1. **Diagonal transition for similar long pairs (C2).** Every similar pair beyond mbleven's reach
   (ub <= 3, m <= 512) runs a bit-parallel band whose cost is linear in n with 3-4 blocks per column at
   m > 704; LV costs O((d + 1)^2 + n/16) and turns a ~40 µs 8192-char near-duplicate into ~1 µs. It works
   for `ratio` through the indel variant, where lev has no small band at all. Start at m > 512 with a
   pace/budget abort so dissimilar pairs pay < 1%; prove the 10x-class win on `benchmark_by_version.py`
   long-similar before extending it down to R3.
2. **Transpose short-vs-long pairs (C1).** lev always iterates the longer string through a latency-bound
   6-cycle single-word chain. Putting the long string into multiword bit-vectors makes the work
   m * ceil(n/64) throughput-bound word steps (and min(m, n) steps when both fit a word), with masks
   built only for the short string's distinct units, which also removes all hash probes for UCS-2/4.
   Expect 1.5-3x on the existing `ascii-24-vs-96/-384` cases; it makes perf-ideas idea 4 unnecessary.
3. **Staggered SIMD lanes on x86 (C3).** Still the only lever for long dissimilar pairs on x86. If it is
   attempted, use A*PA2's carry-free hin/hout block in the lanes, scalar per-lane eq gathers, at most two
   vectors in flight, and VPTERNLOG in a v4 copy; skip scalar staggering, which A*PA2's own data puts at
   5-10%.
