# Novel ways to compute the same DP faster (novel-compute)

Agent: novel-compute. Code read: `main-src` @ 5a6859c. No wall-clock benchmarks were run; all
numbers below are from simulations, emitted assembly and a port-limited dependency model.
Prototypes and raw outputs live in
`/private/tmp/claude-501/-Users-markusbilz-Documents-git-lev/33906bae-f780-4c7b-8ea7-23a2aa9e4dd7/scratchpad/proto/compute/`:

| File | What |
|---|---|
| `textmodels.py` | text models: English-like words (Zipf), names, CJK-like (3000 symbols, Zipf), DNA, lev's 27-char benchmark alphabet; mutate/embed pair builders |
| `conv_full.py`, `conv_full_short.txt`, `conv_full_long.txt` | rank convergence of the full (unbanded) Myers and LCS state, m 16..512, n in {m, 4m, 16m} |
| `conv_band.py`, `conv_band64.txt`, `conv_band128.txt` | rank convergence inside lev's speculative band (`small_band` / `small_band2` semantics), with offset fix-up checked against the band result |
| `skew_check.py` | DP-oracle check of the skewed single-word kernel (4000 pairs, any n) |
| `skew.rs`, `skew_arm.s`, `skew_x86.s`, `skew_test` | Rust prototype of the skewed kernel + current loop, test vs DP (20000 pairs, passes), asm for aarch64 and x86-64-v2 (+BMI1 variant) |
| `sched.py` | loop model: loop-carried latency bound + greedy OoO estimate with P integer ports |
| `dfa_states.py` | reachable Levenshtein-DFA state counts for short patterns |
| `levcopy/target/release/deps/lev-*.s` | lev itself compiled to asm (aarch64) to read the `small_band` / `small_band2` loops |

## Headline

1. **Rank convergence (lead 1) does not work on the full matrix and does work inside the band.**
   With row 0 pinned (`D[0][j] = j`), a chain started from a guessed column only converges once the
   optimal path to every row leaves row 0 inside the chunk. For n ~ m that never happens (0 of 40
   trials, every model, both guesses, also for LCS). Inside lev's 64-diagonal band the window no
   longer contains row 0, and a V-shaped guess converges in 100% of certified pairs after a median
   of 30-45 columns (p90 40-85). That makes a two-chain band on Apple possible, but the band loop is
   only ~2 ops/column short of throughput-bound, so the win is moderate (model: 10-30% of the band
   kernel for n >= 200, nothing on 4-ALU x86).
2. **The best idea found is not parallelism but a shorter chain (lead 6): the skewed single-word
   step.** Sliding the window one row per column (as `small_band` already does) moves the shift off
   the vp chain. Written as `vp' = mh | (x & !(xvs | vn))`, `vn' = (vn | !x) & xvs`, the
   loop-carried chain is 5 ops instead of 6 on Apple/Neoverse and 6 instead of 7 on x86-64-v2.
   Periodic re-shifting with 8 padding rows makes it exact for **any text length** with m <= 56. Verified
   against the DP (Python + Rust, 24000 pairs); LLVM keeps the 5-deep form on both targets without
   `asm!`. Expected -12..-15% per column for every m <= 56 Levenshtein call (tiny, word_u8, hash), and
   the same identity gives -17% per column in `small_band` with a two-line change.
3. **AVX-512 `vpternlogq` on the skewed step in an xmm register** gives a 4-op chain (and, add,
   ternlog, ternlog), i.e. 4 cycles/column vs 6-7 for today's x86 scalar step, for x86 CPUs with
   AVX-512VL (Zen 4/5, Ice Lake and later servers).
4. Packing (lead 2), carry-select/ADX (lead 3), and DFA/Four Russians (lead 5) do not pay; reasons
   and numbers below. For staggered lanes (lead 4) there is one new trick (blend consecutive rows instead of
   gathering), plus a speculative scalar+NEON hybrid for the Apple full matrix.

---

## Lead 1: rank convergence / speculative chunking of the text

**Source.** Maleki, Musuvathi, Mytkowicz, "Parallelizing dynamic programming through rank
convergence", PPoPP 2014 (doi:10.1145/2555243.2555264); Mytkowicz et al., "Data-parallel
finite-state machines", ASPLOS 2014; Viterbi path convergence (traceback merging). For edit
distance I found no bit-parallel use; Maleki et al. apply it to (banded) Needleman-Wunsch, LCS and
Viterbi with scalar cells, the "fix-up" re-run being the same as here.

**Mechanism.** Split the text at column h. Chain 1 runs columns 1..h from the true start; chain 2
starts at h - L from a guessed state and runs to n, interleaved with chain 1 in one loop. At the
end, compare chain 2's saved state at column h with chain 1's final state; if equal (VP and VN),
chain 2's final state is exact (plus an offset, see below). If not, chain 1 continues alone from h
and compares at chain-2 checkpoints (every 16 columns) until they agree.

### Full matrix (pinned row 0): it does not converge

With `D[0][j] = j` fixed, the delta vector is the whole state, and a guessed start converges at
column j only if, for every row i, some optimal path to (i, j) leaves row 0 at or after the chunk
start (proof sketch: the all-+1 guess is an upper bound `D[i][a] <= a + i`, monotone in min-plus,
equal exactly where a path through row 0 inside the chunk is optimal; traceback paths can only
coalesce through row 0 because top rows cannot trace back to a cell (k > i, a)). For n ~ m the
bottom rows' optimal paths start near column 0, so convergence is impossible. Simulation
(`conv_full_short.txt`, 40 trials each, split at n/2, columns to convergence out of the remaining
n/2):

| model | m | n = m (unrel / similar) | n = 4m (unrel / embedded copy) | n = 16m (unrel / embedded) |
|---|---|---|---|---|
| english | 16 | never / never (plus) | 3/40 / 7/40 | 10/40 (med 108 of 128) / 14/40 |
| english | 64 | never / never | never / 8/40 (med 100 of 128) | never / 6/40 |
| names | 32 | never / never | never / 15/40 | 1/40 / 7/40 |
| cjk | 32 | never / never | 1/40 / 8/40 | 4/40 / 19/40 |
| dna | 16..64 | never / never | 0-9/40 | **40/40, med 64 / 124 / 261 (about 4m)** |
| random27 | 24-ish (bench ascii-24-vs-384) | never | never (m=32) | never (m=32 unrel) |

The V-shaped guess (`D[i][a] = |a - i|`, a lower bound) only helps when the true state already is
that shape (DNA, n = 16m: exact at once because the pattern is a subsequence of the prefix). LCS
(all-ones guess) behaves the same. `conv_full_long.txt` (m = 128 and 512, 6 trials each) confirms
it: never at n = m for any model; at n = 4m and 16m, English, names and random27 converge in at
most 3/6 trials, CJK in up to 5/6 (late, median 600-2300 columns), and only DNA at n = 16m
converges in every trial (all-+1 guess after about 4m columns).
Conclusion: **rank convergence is dead for the single-word and multiword full DP**, including the
benchmark's `ascii-24-vs-96/384` (random27, unrelated: never converges). For n >> m the only exact
2-chain option remains the user's idea 4 (Hirschberg forward/backward).

### Inside the band: it converges

The band window slides one row per column, so after `top` columns it no longer holds row 0; the
window state is free to forget its start up to an additive constant (rank-1 in min-plus terms).
The offset is fixed with one subtraction of the `free` counters at the compare column:
`free_final = free_chain1(h) - free_chain2(h) + free_chain2(n)`; `conv_band.py` asserts the
fixed-up score equals the band's own result in every converged trial.

Columns to convergence (`conv_band64.txt`, 12 certified pairs per cell, split at n/2):

| model, B = 64 | edits 2% | 5% | 10% | 25% (m=256) |
|---|---|---|---|---|
| english m=256..704, V guess | med 10-39, p90 44-90 | 42-45 / 46-59 | 32-43 / 55-56 | 54 / 71 |
| names | 1-38 / 38-41 | 35-38 / 41-55 | 26-33 / 44-46 | 47 / 65 |
| cjk | 7-34 / 35-36 | 32-34 / 36-45 | 34-43 / 39-50 | 34 / 48 |
| dna | 31-49 / 53-59 | 26-60 / 65-77 | 58-70 / 73-84 | 62 / 84 |
| random27 | 2-36 / 35-41 | 38-40 / 46-50 | 34-41 / 44-55 | 40 / 48 |
| all-+1 guess (any model) | 64-125 | 64-130 | 67-141 | 68-129 |
| bound sandwich (all +1 vs all -1 coalescing) | never | never | never | never |

Every certified pair converged with the V guess (one 25%-edit English pair at m = 128 ran out of
columns). For the 128-diagonal band (`conv_band128.txt`) the V guess needs median 44-108,
p90 75-158 columns: convergence length scales with the band width, roughly B/2..B. The sandwich
(coupling from the past) never coalesces: the two bounds converge to parallel, not equal, vectors,
so it cannot certify without chain 1.

**Why it would avoid Hirschberg's combine cost.** The rejected two-chain Hirschberg kernel paid
a reversed peq table (or `rbit` per column) plus an O(m) prefix-sum min-combine. Rank convergence
reuses the same forward table, combines with a 2-word equality test and one subtraction, and its
"extra work" is the L warm-up columns of chain 2, which run in the shadow of chain 1. Its failure
mode is graceful: chain 1 is the original computation, so a non-converging pair costs about the
baseline plus the interleaving overhead, unlike the speculative bands' structural failure cost.

**Where it applies in lev.** Only `kernels::portable::small_band` (`portable.rs:97-193`), the
1-word band for 65 <= m <= 704 with n - m < 64. `small_band2` is already balanced (lev's asm: ~40
ALU ops/column = 6.7 cycles on 6 ALUs vs a 6-cycle chain), and the multiword kernels are
throughput-bound, so a second chain cannot fit there.

**Cost model** (`sched.py`, 6 ALU ports, from lev's emitted `small_band` middle arm: 24 ALU ops,
2 loads, 6-cycle chain): one chain 6.0 cycles/column; two chains sharing the shift counter (chain 2
starts at h - L = 0 mod 64 so `k`/`!k` are shared) 7.4 cycles per pair; three chains 10.8. Band
time with L = 64, `T_rc = 7.4 (n + L)/2 + ~30`:

| n | today (6n) | 5-deep single chain (5n, see lead 6) | two chains |
|---|---|---|---|
| 128 | 768 | 640 | 742 |
| 200 | 1200 | 1000 | 1009 |
| 300 | 1800 | 1500 | 1380 |
| 704 | 4224 | 3520 | 2879 |

On 4 ALUs (x86 Zen 2-4, Neoverse N2) two chains cost 11.0 cycles per pair: no gain.

- **Regime:** R3/R4 similar pairs that the 1-word band certifies, n >= ~250, Apple only (maybe
  Zen 5 / Oryon with 6 ALUs).
- **Expected gain:** 8-18% of the band kernel over the 5-deep single chain at n = 300-704; 0% below
  n ~ 200. As a share of the call maybe 5-12% on `medium` similar pairs of 300+ chars.
- **Effort:** M-L (two chains in the chunked funnel loop with different word indices `q`,
  checkpoints, fallback, offset fix-up; the give-up checks stay on chain 1 only).
- **Risk:** layout sensitivity (two more arms per chunk), more registers (two states, two `free`,
  two text pointers: fits AArch64's 31), and pairs whose band gives up after column h are found
  only at the end (still cheaper than today because the whole interleaved run takes ~h columns).
- **Test:** `benchmark_by_length.py` similar pairs at 256/512/704 with 1-20 edits, plus 20%-edit
  pairs; falsified if the two-chain band is not >= 5% faster than the 5-deep single chain at n =
  512 on M2, or regresses n <= 200.
- **Conflict with rejected items:** none (not Hirschberg; no combine).

---

## Lead 6 (new, top pick): the skewed single-word step, a 5-op chain for any n

**Source:** novel for the full DP as far as I can tell. The skew itself is lev's own `small_band`
("the vertical inputs move up a bit (`xv >> 1`) instead of the horizontal ones moving down") and
RapidFuzz's `levenshtein_hyrroe2003_small_band`; both use it only inside a band, and both keep the
6-deep form `vp = mh | !(xvs | ph)`.

**Mechanism.** In the standard layout every column must shift the horizontal deltas down a row
(`ph << 1`, `mh << 1`), and those shifts sit on the vp chain after the add: lev's Apple step
(`apple.rs:21-35`) is and -> add -> orr -> lsl -> bic -> orr = 6 deep. In the skewed layout the
window slides down instead, so the only shift is `xvs = (eq | vn) >> 1`, which depends on vn and eq
and not on the add. With `x = sum | vp | eq` and `ph = vn | !x`:

```
!(xvs | ph) = !xvs & !vn & x = x & !(xvs | vn)
vp' = mh | (x & !(xvs | vn))        // and, add, orr(x), bic, orr  = 5 deep
vn' = (vn | !x) & xvs               // orn, and                  = 5 deep
```

`J = xvs | vn` is ready at depth 3 from vn, as is x from vp, so both outputs are at depth 5.

**Making it exact for any n (not just a band).** Keep PAD rows above row 1 that are copies of row
0 (vertical delta 0, never match): they supply the `+1` horizontal input at the top, as the
band's out-of-pattern rows do. After t <= PAD skewed columns, row 1 sits at bit PAD - t; every
PAD columns shift both state words back left by PAD (zeros in = fresh padding rows). The eq of
column t of a block is `peq[c] << (PAD + 1 - t)`, an immediate shift when the block is unrolled.
Result: `n + popcnt(vp & rows) - popcnt(vn & rows)` with rows at bit `PAD - t`. Valid for
m <= 64 - PAD, any n. At least one padding row must remain in the window during every step
(k <= PAD; k = PAD + 1 fails, found by the checker).

**Verification.** `skew_check.py`: 4000 random pairs, alphabets 2..27, PAD in {1..48}, every
block length k <= PAD, n up to 192, both the 5-deep and the plain form: all equal to the DP.
`skew.rs` test: 20000 pairs vs a DP oracle, PAD = 8, n up to 200: passes.

**Codegen (emitted, `skew_arm.s`, `skew_x86.s`).** LLVM keeps the form without `asm!`:

- AArch64 per column: `ldr` peq, `lsl #imm`, then `and, orr, add, orr, orn, and, orr, lsr, orr, bic,
  orr, orn, and` = 13 step ops (the current Apple step is 15), chain and -> add -> orr -> bic -> orr
  = 5. Block end: `lsl #8` on both words (1 cycle per 8 columns on the chain).
- x86-64-v2 (no BMI1, which is what lev's v2 copy runs): current loop chain is 7
  (and, add, not, or, and, add(shl), or on the mh path; `skew_x86.s` `current_word`), skewed loop 6.
  With BMI1 (`andn`): current 6, skewed 5 (`skew_word_bmi`: and, add, or, andn, or).

**Latency/throughput model** (`sched.py`): current Apple step 6.00 cycles/column (17 ALU ops incl.
loop), skewed 5.00 (16 ops, plus 2 ops per PAD columns); with 4 ALUs still 5.00. Adding the
re-shift: 5.125 at PAD = 8, 5.03 at PAD = 32.

**Where it applies (all Levenshtein, both pattern kinds):**
- `hyrro_inner` (`lib.rs:1309-1317`) via `kernels::myers_step` for every single-word caller:
  tiny path (m <= 16, `lib.rs:626-816`; PAD = 32 there), `word_u8` table and NEON/AVX2 table-free
  forms (17-64), `hyrro_64_hash` (UCS-2/4/mixed). Limit m <= 56 with PAD = 8 (57-64 keep today's
  step, or PAD = 64 - m >= 4 at 5.25 cycles).
- `small_band` (`portable.rs:134-142`): same identity, two-line change, 6 -> 5 cycles in the model
  (the band already slides, no re-shift needed). `small_band2` is throughput-balanced (6.7 vs 6), so
  the +1 op per word is likely neutral there; measure but do not expect a win.
- Not LCS/`ratio`: the LCS step (`lib.rs:1340-1348`) has no shift; its chain is already
  and -> add -> orr = 3.
- Not multiword: throughput-bound, and skewing would add funnel shifts.

**Expected gain.** Kernel: -14.6% per column on Apple and Neoverse (6 -> 5.125), -12.5% on x86 v2
(7 -> 6.125), -14.6% with BMI1. Per call it scales with the kernel share:
`ascii-24-vs-384` ~10-14%, `ascii-24-vs-96`, `ascii-32-dis`, `latin1-32-dis`, `ascii-48-d6` ~4-9%,
kitten/sitting ~0-2% (7 columns). Medium similar pairs through `small_band`: ~10-15% of the call
where the band dominates (e.g. the realistic m = 190 pair: ~200 columns x 1 cycle = ~55 ns).

- **Effort:** S for `small_band`; S-M for the single-word kernel (new `word` body with an
  unrolled PAD block and a re-shift; the tiny SIMD mask paths just add `<< imm`; the Apple `asm!`
  step can be retired or rewritten as a 13-instruction `asm!` if LLVM later reorders it).
- **Risk:** low algorithmic risk (exact, verified); layout sensitivity as always (the tiny and word
  paths have load-bearing `#[inline(never)]`); code size (unroll x8). m = 57-64 needs the old step.
- **Test:** CodSpeed instruction counts should drop slightly (13 vs 15 step ops); wall clock on M2
  and the A/B runners for `ascii-24-vs-384`, `ascii-32-dis`, `latin1-64-dis`, `ucs2-32`, kitten,
  `benchmark_by_length.py` 17-64; for the band, `ascii-120/160-sim2`, similar 128-704.
  Falsified if `ascii-24-vs-384` (pure kernel) does not improve by >= 8% on M2, or if the 5-deep
  form shows up as 6 deep in `cargo asm` after inlining into lev.
- **Conflict:** none. Not the rejected `andn` single-word step (#201): this changes the dependency
  structure, not the instruction selection, and also helps without BMI1.

---

## Lead 5b: AVX-512 `vpternlogq` + skewed step in an xmm register (x86 only)

**Source:** novel as a latency device (ternlog is used for multi-pattern bit-parallel matching
throughput, not for shortening a single Myers chain that I know of).

**Mechanism.** Keep vp/vn in xmm registers, load eq straight into xmm (`vmovq xmm, [peq + c*8]`, no
GPR transfer), and compute the skewed step with 3-input ternary logic:
`sum = vpaddq(vpand(eq, vp), vp)`; `x = tern(sum, vp, eq)`; `mh = tern(vp, eq, sum)`;
`xvs = vpsrlq(vpor(eq, vn), 1)`; `J = vpor(xvs, vn)`; `vp' = tern(mh, x, J)`;
`vn' = tern(vn, x, xvs)`; plus `vpsllq` for the skew. Chain: vpand -> vpaddq -> ternlog -> ternlog
= **4 cycles**, 10 vector ops per column (2.5 cycles on 4 vector ports, 3.3 on Intel's 3).

- **Regime:** single-word Levenshtein on x86 with AVX-512VL (Zen 4/5 incl. the EPYC CI runners,
  Ice Lake / Sapphire Rapids servers). Not on AArch64 (NEON integer ops take 2 cycles on Apple;
  SVE2's EOR3/BCAX are 2-3 cycles on Neoverse and lev has no SVE).
- **Gain:** kernel 6-7 -> 4 cycles/column, 35-43% on the single-word loop; ~15-30% per call for
  n >= 32.
- **Effort:** M (needs an AVX-512 copy or a runtime-checked `#[target_feature]` word kernel like
  `avx2::word_u8`; the per-call `is_x86_feature_detected!` costs a load + branch).
- **Risk:** AVX-512 frequency licence is a non-issue for xmm on Ice Lake+ and Zen 4; availability is
  mostly server-side; must be checked on CodSpeed/A/B x86 (no local hardware).
- **Test:** cross-compiled asm to confirm the 4-op chain; A/B on the EPYC runner for short-*,
  `ascii-24-vs-384`. Falsified if < 10% on `ascii-24-vs-384` on Zen 4.
- **Conflict:** the rejected AVX2 kernels used vectors across words with lane-mask carries; here the
  whole step stays in one lane, nothing crosses domains inside the loop.

---

## Lead 2: packing several sub-problems into one word or vector

- **What could go in the lanes?** Only independent chains of the same pair. The simulation above
  shows text chunks of the full DP do not converge for n ~ m (and rarely for n >> m), so chunk
  lanes have nothing to verify against. What remains is forward + backward halves (Hirschberg, the
  user's idea 4) in two lanes.
- **NEON 2 x 64 or 8 x 16 lanes:** isolation is free (`add v.2d`, `shl v.8h` do not cross lanes),
  but Apple's integer NEON ops take 2 cycles, so the 5-6 op chain becomes 10-12 cycles per vector
  step: 2 columns per 10-12 cycles = no better than one scalar chain at 5. 8 x 16-bit lanes for
  m <= 16 would give 8 columns per ~12 cycles, but needs 8 independent sub-problems that do not exist.
  Gathers for 8 different text chars: `tbl`/`tbx` over a 128-byte u8 peq (m <= 8) is 2 ops per 16
  lanes, cheap, but moot without lanes to fill.
- **Scalar 2 x 32 for m <= 23 (forward/backward):** the skewed form has no `| 1` shift-in, so the
  only isolation needed is (a) no carry out of lane 0, forced by keeping a guard bit of vp at 0
  (fold the guard into `xvs` via `eq | guard << 1`, off-chain), and (b) masking the PAD re-shift
  (1 op per PAD columns). About 15 ops per two columns at a 5-cycle chain, 2x on the kernel, but
  Hirschberg's combine (prefix sums + min over m rows, ~30-60 cycles) eats it unless n >> m.
- **Verdict:** will not pay in lev's regimes; at most a niche (m <= 23, n >= 8m). Effort M, gain
  0-5% overall. Do not pursue before idea 4 itself is measured.

## Lead 3: carry-select / carry-lookahead / Kogge-Stone / ADX for the multiword ripple

- The multiword loop is **throughput-bound, not ripple-bound**: lev's Apple multiword is ~13 ALU ops
  per word (`apple.rs:81-92`); W = 4 is ~55 ops = 9 cycles per column on 6 ALUs, while the carry
  ripple adds only 1 cycle per word along a wavefront that the OoO window already overlaps across
  columns (word 0 of column j+1 starts as soon as word 0 of column j is done). `bmi1.rs:51` says the
  same for x86 ("throughput-bound from two words on"). Carry-select doubles the add and the
  dependent ops per word; lookahead adds generate/propagate ops: both cost throughput to save latency
  that is not on the critical resource.
- **ADX dual chains:** `adcx`/`adox` only help if no flag-writing op sits between the adds; every
  AND/OR/XOR/ANDN clears both CF and OF, and the Myers word has ~10 of them, so two flag chains cannot
  be interleaved with the logic. The real x86 problem in `portable::full` (flags moved through
  `setb`/`bt` each word, `bmi1.rs:105-110`) is instruction count, already tackled by strips.
- **Speculating the hp/hn shift-in** as well as the carry needs 4 variants per word (2 carries x
  2 shift-in states, really 3 for hp/hn): more than 2x the work.
- **Verdict:** will not pay. No experiment recommended.

## Lead 4: wavefront / anti-diagonal / staggered lanes

- A*PA2 lanes are the user's idea 3. **New detail for the gather problem:** with lane k working on
  word k of column j - k, lane k needs `peq[t[j-k]][k]`. Instead of a gather (slow on Intel since the
  Downfall microcode, ~9+ cycles on Zen), keep a rolling window of the last 4 full peq rows (each one
  contiguous 32-byte load) and assemble the staggered vector with 3 immediate blends
  (`vpblendd`): 1 load + 3 blends per column, all off the chain. On NEON with 2 lanes it is 1 load +
  1 `ins`/`trn` per column.
- **Speculative new angle for Apple's full matrix (R5 dissimilar):** the full-matrix strips
  (`aarch64/mod.rs:61-73`) are ALU-throughput-bound (8 words x ~13 ops = 104 ops = ~17 cycles per
  column) while the 4 NEON pipes idle. Running one strip on the scalar ALUs and the next strip on
  staggered NEON lanes concurrently (coupled through lev's existing per-column `edges` array, with
  the NEON strip lagging by >= 64 columns so the cross-domain moves are off the chain) adds NEON
  throughput instead of replacing scalar (which is what the rejected AVX2 kernels did). NEON per word:
  ~20 ops (add without carry-out needs `cmhi` + carry-in add), 2 words per vector, 4 pipes: ~2.5
  cycles/word vs scalar ~2.2: combined at best ~1.8x on the full matrix, realistically 1.2-1.4x
  after edges traffic and stagger prologue/epilogue. Effort L, risk high, Apple-only. Worth a
  feasibility prototype only after the cheaper items.
- Wozniak/Farrar/triple_accel anti-diagonal byte lanes compute scores, not bit vectors; for unit
  costs the bit-parallel step does 64 cells per ~13 ops, which byte-lane anti-diagonals (16 cells per
  ~6 ops at 2-cycle NEON latency) do not beat. Not recommended.

## Lead 5: automata, Four Russians, peq-building tricks

- **Levenshtein DFA of the pattern** (`dfa_states.py`): reachable column states for English-like
  patterns: m = 4: 47; m = 6: 246-320; m = 8: 1083-2044 (x 8-9 classes = up to 18k transitions);
  m = 10: 4k-12k; m = 12: 25k-73k. Each transition costs one Myers step to build, so building even
  lazily costs more than running Myers over lev's n (10-400). With cross-call caching the per-char
  cost is one dependent L1 load + index math (~5-6 cycles), no better than the 5-cycle skewed step.
  **Will not pay.**
- **Four Russians:** the bit-parallel step already processes 64 rows per ~13 ops; block tables for
  small alphabets lose to it at lev's sizes. **Will not pay.**
- **PMULL/GFNI/VPSHUFBITQMB peq builds:** peq construction is O(m) scattered ORs plus O(sigma W)
  zeroing, off the column chain; the zeroing side is the TLS-scratch idea already on the list.
  Bit-matrix transposes do not reduce either term. Not pursued.
- **vpternlogq fusion:** see lead 5b; it is a real win only together with the skewed form.

## Other notes (lead 6 extras)

- Choosing the pattern by alphabet size or table cost: for n ~ m the step count is n either way;
  for n >> m lev already makes the short side the pattern (one word). No gain found.
- Two-column algebraic steps: composing two Myers steps keeps two dependent adds on the chain; no
  form below 2 x 5 was found. The skewed form is the shortening that exists.

---

## Ranked table

| # | Idea | Regime / arch | Est. gain | Effort | Confidence |
|---|---|---|---|---|---|
| 1 | Skewed single-word step (5-op chain, PAD re-shift, any n, m <= 56) | R1-R2 Levenshtein, all kinds; Apple, Neoverse, x86 | kernel -12..-15%; per call 4-14% (most on n >> m) | S-M | high (verified exact; codegen confirmed) |
| 2 | Same identity in `small_band` (`vp = mh \| (x & !(xvs \| vn))`) | R3/R4 similar pairs certified by the 1-word band; all arches | band kernel -15..-17%; per call 5-15% | S | high (model; same verified identity) |
| 3 | AVX-512VL ternlog skewed step in xmm (4-op chain) | R1-R2 Levenshtein on Zen 4/5, Ice Lake+ | kernel -35..-43%; per call 15-30% | M | medium (no x86 hardware here; latencies from vendor tables) |
| 4 | Rank-convergence two-chain band (V guess, L = 64, verify + fallback) | R3/R4 similar, n >= 250, Apple (6 ALUs) | band kernel -8..-18% beyond #2; 0 on 4-ALU cores | M-L | medium (convergence measured; throughput modelled) |
| 5 | Scalar+NEON hybrid strips for the Apple full matrix | R5 dissimilar, Apple | 1.2-1.4x on full matrix (optimistic) | L | low |
| 6 | Rolling-row blend instead of gather for staggered lanes | input to idea 3 (A*PA2) | enabler, not a gain by itself | S (within idea 3) | medium |
| 7 | 2 x 32 scalar packing (forward/backward) | m <= 23, n >= 8m | 0-5% | M | low; not recommended |
| 8 | Rank convergence on the full matrix (text chunks) | none | 0 (never converges for n ~ m) | - | high that it fails |
| 9 | Carry-select / Kogge-Stone / ADX multiword | none | negative | - | high that it fails |
| 10 | Levenshtein DFA / Four Russians | none | negative | - | high that it fails |

## Top 3

1. **Skewed single-word step.** Sliding the window instead of shifting the horizontal deltas moves the
   shift off the vp chain, and the identity `!(xvs | ph) = x & !(xvs | vn)` leaves and, add, orr, bic,
   orr: 5 cycles per column instead of 6 (Apple, Neoverse) or 6 instead of 7 (x86 v2), with 13 step ops
   instead of 15. Padding rows plus a shift-back every 8 columns make it exact for any text length
   with m <= 56. It is verified against the DP on 24000 random pairs, and LLVM emits the 5-deep form on
   both targets. It is a drop-in for every single-word Levenshtein caller (tiny, table, table-free,
   hash). Falsified if `ascii-24-vs-384` gains under 8% on M2.
2. **The same identity in `small_band`.** Changing two lines in `portable.rs:141` takes the band
   loop from a 6-deep to a 5-deep chain (as the scheduling model shows). It speeds up every medium similar pair
   that the 1-word band certifies. It is the cheapest experiment on this list and should run first.
3. **AVX-512 ternlog skewed step** for x86 server CPUs: a 4-op chain (vpand, vpaddq, two
   ternlogs) with the step held in one xmm lane and eq loaded straight from memory. It is the largest
   per-column cut found, but it needs a runtime-checked AVX-512 form and the CI EPYC runner to measure.
   Rank convergence (lead 1) is a real but smaller follow-up: it is useless on the full matrix
   (pinned row 0, simulated), but the V-shaped guess converges in about 30-45 columns inside the
   64-diagonal band, enough for an Apple-only two-chain band once n >= 250.
