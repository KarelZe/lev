# Architecture-specific and toolchain opportunities for lev

Agent: arch-toolchain. Code read: `main-src` @ 5a6859c (lev 0.6.1). Line numbers refer to `main-src`.
No wall-clock benchmarks were run. Every cycle figure below comes from emitted assembly (rustc 1.95.0,
same `.cargo/config.toml` flags and release profile as the wheels) plus published latency tables.

Artifacts:
- Build copy: `scratchpad/proto/arch/src-copy` (x86_64-linux, aarch64-linux and aarch64-apple-darwin `.s` under `target/`).
- Probe crate for hypothetical forms: `scratchpad/proto/arch/probe/src/lib.rs` (+ `i686.rs`), asm in `target/`,
  APX build in `target-apx/`.
- Loop extractor: `scratchpad/proto/arch/tools/fn.py <file.s> <demangled-substring> [all]`.

## 1. Verified facts that change the picture

1. **The x86 single-word step runs without `andn` today, at 7 cycles per column.** `x86_64::myers_step`
   (`src/kernels/x86_64/mod.rs:16-18`) calls `bmi1::myers_step_andn`, but the V2 copy enables only
   `popcnt,sse3,ssse3,sse4.1,sse4.2,cmpxchg16b` (`src/kernels/x86_64/v2.rs:39`). The V2 loop in
   `v2::distance_views` is 27 instructions with 3 `not` and 4 `mov`. The `vp` chain is 7 cycles:
   and, add, not, or, and (mh), add (mh<<1), or. With BMI1 the same source gives 23 instructions and
   a 6-cycle chain (probe `probe_word_v3`). The single-word step also drives the tiny path and the
   UCS-2/4 hash loops, since they share `hyrro_inner` (`src/lib.rs:1309-1317`).
2. **The small band step is 8 cycles on x86 and 6 on AArch64.** `portable::small_band`
   (`src/kernels/portable.rs:134-142`) computes `vp' = mh | !(xvs | ph)` with `ph = vn | !X`, which x86
   lowers to or, not, or, or, not, or on the chain (30-instruction loop, 8-cycle chain). AArch64 folds
   the negations into `orn`, which gives 6 cycles. Rewriting the step with `!ph = X & !vn`, as #221/#231
   did for the single-word step, gives 6 cycles on x86 even without BMI1 (probe `probe_band_new_v2`).
   The band funnel shift compiles to `shrd %cl`, which is 5 uops on Zen 3/4 and 3-4 on Intel.
3. **Wide-unit tiny patterns are scalar on both architectures.** For UCS-2, UCS-4 and mixed patterns of
   9-16 code units, `tiny_n` (`src/lib.rs:762-779`) compiles to 16 `cmp` + 16 `sete`/`cset` + shifts and
   ors per text character. The loop is 83-101 instructions on x86 V2 and 83 on Apple. That is
   throughput-bound at about 15-20 cycles per column on a 4-ALU core and 11-14 on Apple, against a
   6-7-cycle chain. LLVM does not vectorize it, which answers the open question in perf-ideas idea 1,
   step 3.
4. **The multiword add is already a clean `add`/`adc` chain.** At W = 2-8 there are W-1 consecutive
   `adc`, with no `setb`/`bt`. That round trip exists only in the portable `full` loop over heap words,
   as the `bmi1.rs:104-110` docs say. At W = 6 the probe loop is 157 instructions with 46 stack
   references and 38 GPR moves. At W = 8 it is 219 instructions, 81 stack references and 58 moves. The
   spills are real, but the chain is not the problem.
5. **Linux AArch64 runs the Neoverse step at 7 cycles on Apple cores.** The Linux loop ends in
   `orr x10, x13, x10, lsl #1` on the `vp` chain. That shifted-operand op takes 1 cycle on Neoverse and
   2 on Apple (and on Oryon, per LLVM's `AArch64SchedOryon.td`), so the chain is 6 cycles on N2/V2 and
   7 on Apple. The Apple `asm!` loop is 6 cycles there.
6. **Toolchain facts (rustc 1.95.0).**
   - `hint::cold_path` and `hint::select_unpredictable` are stable.
   - `apxf` is unstable: `#[target_feature(enable = "apxf")]` is error E0658, tracked in #139284.
     `-C target-feature=+apxf` passes through with a warning.
   - There are no SVE intrinsics in `std::arch::aarch64` (`svcntb` is unresolved), but `sve2` detection
     and `#[target_feature]` compile.
   - `-Z` options (`min-function-alignment`, `tune-cpu`) are nightly-only.
   - Default CPUs from `--print target-spec-json`:
     - `aarch64-apple-darwin`: `cpu: "apple-m1"`, so setting it again gains nothing.
     - `aarch64-unknown-linux-gnu`: generic `+v8a,+outline-atomics`.
     - `aarch64-pc-windows-msvc`: generic `+v8a,+neon`.
     - `i686`: `pentium4` (SSE2).
7. **Binary size.** The V2 copy is 72.6k of the crate's 163.8k x86 instructions (44%). The 0.6.1
   x86_64 manylinux cp313 wheel is 389 KB and win_amd64 is 311 KB. A third copy adds about 35-45%.
8. **CI hardware is mixed.** Per RunsOn's survey, GitHub x64 runners are EPYC 7763 (Zen 3, no AVX-512),
   9V74 (Zen 4) and 9V45 (Zen 5). Arm runners are Neoverse N2 and V3. Within one A/B job the ratio is
   valid, but AVX-512 code is exercised only on the Zen 4/5 draws. The "Hardware" step prints
   `platform.processor()`, which is empty on Linux.
9. **Zen 5 has 2-cycle vector integer latency.** `VPADDQ xmm` is 2 cycles on Zen 5 and 1 on Zen 4 and
   Intel. `VPTERNLOGQ xmm` is 2 cycles from operands 2/3 on Zen 5 (uops.info). Any plan that moves the
   scalar chain into vector registers loses on Zen 5.

## 2. x86-64

### 2.1 SIMD match masks for wide-unit tiny patterns (UCS-2, UCS-4, mixed; m = 9-16) (both arches)
- **Source:** perf-ideas idea 1 step 3 asked whether LLVM vectorizes `tiny_n`. It does not (fact 3).
  The kernel form below is novel for lev.
- **Mechanism:** Hold the pattern widened to u32 lanes (or u16 lanes for UCS-2) in vector registers.
  Per text unit, broadcast it, compare, then narrow to a 16-bit mask:
  - SSE2 (baseline x86), u16: 2× `pcmpeqw` + `packsswb` + `pmovmskb`.
  - SSE2, u32: 4× `pcmpeqd` + 2× `packssdw` + `packsswb` + `pmovmskb`.
  - NEON: `vceqq_u16`/`u32` + `vshrn`/`uzp` narrowing + the existing `shrn` movemask, as in
    `neon.rs:108-109`.
  - Text units above the pattern's range compare unequal automatically, so mixed pairs work.
- **Where:** R1, the dominant regime for CJK words, names and emoji strings. Distance and ratio, every
  arch. It also feeds `hyrro_64_tiny_body` for the mixed kinds (`src/lib.rs:658-676`).
- **Why it beats today:** Today's column costs about 5 ALU ops per pattern unit (80 at m = 16). That is
  about 16-20 cycles on Zen 3/4 and 11-14 on Apple, against a 6-7-cycle chain. The SIMD form costs 4-8
  vector uops off the integer ports, so the loop falls back to the chain. The kernel gets 1.6-2.7x
  faster per column at m = 9-16. At m <= 6 the scalar form is already cheap, so gate on m >= 8.
- **Expected gain:** -10..-35% per call for UCS-2/4/mixed at m = 9-16 with n ~ m to 4m. Gains are
  smaller where call overhead dominates (very short n).
- **Effort:** S-M: one form per arch in `kernels/{x86_64,aarch64}`, plus a portable fallback. Tests
  against `tiny_n`.
- **Risk:** Low algorithmic risk. Layout risk on the tiny path, which is already `#[inline(never)]`
  per copy (`lib.rs:625`). Keep it inside `hyrro_64_tiny_body`.
- **Test:** `benchmark_by_length.py` UCS-2/UCS-4/mixed at 9, 12, 16 (n = m, 4m). Add tiny CJK/emoji
  cases to `test_bench.py` if missing. Falsified if the x86 and macOS A/B show < 3% at m = 12-16.
- **Conflicts:** None. This is not the rejected "direct tables for wide units"; there is no table.

### 2.2 Small band step algebra for x86 (novel, found in asm)
- **Mechanism:** In `portable::small_band`/`small_band2`, compute `nph = X & !vn` with
  `X = sum | vp | eq`. Then `vp' = mh | (nph & !xvs)` and `vn' = xvs & !nph`, and mh takes the andn
  form `vp & !(sum & !eq)` (probe `band_new`). The cell values are identical; only the operation
  order changes.
- **Where:** R3/R4 similar pairs (65-704) on x86, Levenshtein (#260's band). The same rewrite applies
  to `small_band2` (`portable.rs:239-253`).
- **Cycle estimate:**
  - Today on x86: the chain is 8 cycles, about 25 ALU uops per column (30 instructions).
  - Rewritten: the chain is 6 cycles (32 instructions in V2, 26 with BMI1). Zen 3/4 (4 ALUs) then sit
    at about 6.5-7 cycles, throughput-bound. Intel Golden Cove (5 ALUs) and Zen 5 (6 ALUs) reach the
    6-cycle chain.
  - AArch64 is already at 6, so keep the old form there or confirm that its codegen is unchanged.
- **Expected gain:** -10..-25% of band time on x86, which is most of a call for near-identical
  65-704-char pairs.
- **Effort:** S. An x86 form of the step, selected by `cfg(target_arch = "x86_64")` inside the
  `columns!` macro. Do not change the portable form for AArch64.
- **Risk:** Layout. `small_band` is `#[inline(never)]` and shared by both copies.
- **Test:** A/B `medium-*` similar cases on linux/windows x86. Falsified if < 3% on similar 128-704
  pairs, or if the dissimilar give-up cases regress.
- **Conflicts:** None.

### 2.3 AVX2 table-free masks for 33-64 bytes (two compares)
- **Source:** perf-ideas idea 1 plan; untested, since #266 gated AVX2 at m <= 32.
- **Mechanism:** Two `vpcmpeqb ymm` + two `vpmovmskb` + `shl`/`or`. That is 29 instructions per
  column with the BMI1 step (probe `probe_word_v3_tablefree64`), against a table load plus zeroing
  and filling 1-2 KiB per call.
- **Where:** R2, u8 patterns of 33-64 with n <= 8m, x86 with AVX2, in the existing V2 copy behind the
  same runtime check as `avx2::word_u8` (`mod.rs:180-189`).
- **Why:** The 6 extra uops run on vector ports, and the integer loop stays chain-bound (7 cycles,
  about 19 ALU uops). #266 measured -8..-22% on x86 for 17-32. 33-64 has the same setup savings and
  slightly more per-column work.
- **Expected gain:** -5..-18% at n <= 4m and 33-64 chars. Neutral at 8m (tune the 8x gate per arch).
- **Effort:** S. **Risk:** Low. Ratio's 3-cycle LCS chain may turn throughput-bound, as on M2; measure
  ratio separately.
- **Test:** `benchmark_by_length.py` 33-64 ASCII/Latin-1, n = m, 4m, 8m. Falsified if < 3% at n = m on
  the x86 A/B.

### 2.4 x86-64-v3 copy of the hot path (perf-ideas 5b)
- **Mechanism:** A third `Isa` impl (`V3`) with
  `avx,avx2,bmi1,bmi2,fma,lzcnt,movbe,f16c,xsave` on top of V2's features, registered in `_lev`
  (`python.rs:251-262`) when detected. It needs a copy of each out-of-line function of `v2.rs`, so
  `v2!` should become a macro that stamps both copies.
- **What it buys, measured in asm:**
  - Single-word step 7 -> 6 cycles, 27 -> 23 instructions. This covers the tiny path, `word_u8`, both
    hash loops and mixed pairs, so -14% of single-word kernel time.
  - Band step down to 26 instructions, which helps Zen 3/4, where 2.2 alone stays throughput-bound.
  - `shrx`/`shlx` could replace the 5-uop `shrd %cl` in the band funnel. LLVM keeps `shrd` even
    with BMI2, so hide `hi << 1` behind an opaque value as `neoverse.rs:17-33` does.
  - AVX2 without per-call checks: negligible, about 1 cycle per call.
  - LLVM widens the affix XOR blocks (`lib.rs:499-505`) to `ymm`: under 1 ns except on very long
    shared affixes.
- **What it does not buy:**
  - Multiword W <= 4 already has BMI1 (`bmi1::multiword`). At W >= 5, and-not measured within noise
    (`mod.rs:64-67`).
  - The LCS steps have no negation to fold.
- **Expected gain:** R1/R2 on x86 -4..-12% per call (all kinds, both metrics for the tiny/hash
  Levenshtein path, ratio unchanged). R3 similar -3..-8% on top of 2.2.
- **Effort:** M.
- **Risk:**
  - Binary +35-45% (fact 7).
  - Layout: #201 died of it; a fresh copy recompiles everything.
  - Option B: replace V2 by V3 and keep the baseline. This is size-neutral, but v2-only CPUs fall to
    v1 code: pre-2013 Intel, Goldmont/Tremont Celerons, pre-2021 Pentium/Celeron without AVX, and VMs
    that mask AVX. Their v1-vs-v2 loss is small (POPCNT at the end, SSE4.1 loads).
- **Test:**
  - A/B with B = V3 copy on linux/windows x86 (all draws have AVX2).
  - Run a second A/B with the copy registered but only the step changed, to separate andn from
    layout.
  - Falsified if the single-word cases (17-64, all kinds) gain < 3% geomean, or if any untouched
    regime moves > 5% across two runs.
- **Conflicts:** Supersedes #201's shared-hot-path variant. The new angle is a separate copy, which
  perf-ideas 5b already anticipates.

### 2.5 AVX-512VL single-word step in an xmm register with `vpternlogq` (novel)
- **Mechanism:** The Myers step in a vector register: and, add, two 3-input ternlogs (X, mh), shifts,
  then two ternlogs for `vp'` and `vn'`. The chain is 5 cycles: and, add, ternlog, shift, ternlog. The
  loop is 18 instructions, 12 vector ops (probe `probe_word_v4_vec`; LLVM kept the ternlog
  immediates). The eq mask loads straight into xmm (`vmovq`), and the final popcounts move out once.
- **Where:** R1/R2 Levenshtein on Zen 4, Ice Lake/Tiger Lake/Rocket Lake and Xeon (Skylake-SP and
  later).
- **Why:** 7 -> 5 cycles (-29%) against V2 and -17% against 2.4. It is latency-bound with headroom:
  12 ops over 5 cycles on 3-4 vector ports. Not for LCS: its chain is already 3 cycles.
- **Catch:** On Zen 5 the 2-cycle vector latency makes it about 10 cycles, a 40%+ regression. Gate by
  CPUID family (AMD family 0x1A excluded), not only by `avx512vl`. Future Intel cores need re-checking.
- **Expected gain:** -10..-20% per call on Zen 4/Intel AVX-512 parts. Reach is limited: most Intel
  client parts since Alder Lake lack AVX-512.
- **Effort:** M (a v4 copy) or S-M (a runtime-checked single-word form like `avx2::word_u8`).
- **Risk:**
  - Family gating is maintenance debt.
  - CI only exercises it when a 9V74 (Zen 4) runner is drawn. The A/B must log the CPU model.
- **Test:** A/B jobs whose Hardware step shows Zen 4. Falsified if < 5% on 17-64 cases there, or any
  regression on 9V45 (Zen 5) draws, where the gate should keep it off.
- **Conflicts:** Distinct from the rejected AVX2 multiword: there is no cross-lane carry.

### 2.6 AVX-512BW table-free masks (one `vpcmpeqb zmm -> k`, `kmovq`) for 17-64
- **Mechanism:** 3 uops per character for any m <= 64 (probe `probe_word_v4_tablefree`, 25
  instructions per column).
- **Verdict: marginal over 2.3.** AVX2 already covers 17-64 with 1-2 compares off the chain. Zen 4
  double-pumps zmm. It is only worth adding if a v4 path (2.5) exists anyway. Gain over 2.3: 0-3%.

### 2.7 APX (32 GPRs, NDD, NF)
- **Measured in asm** (crate-wide `+apxf`, probe `target-apx/`):
  - W = 4: 97 -> 65 instructions.
  - W = 6: 157 -> 96 instructions, 46 -> 0 stack references, 38 -> 0 moves.
  - W = 8: 219 -> 137 instructions, 81 -> 14 stack references.
  - Single-word v3: 23 -> 20 instructions.
  - LLVM uses `{nf}` forms, so logic ops can interleave with the `adc` chain.
  - Plausibly -20..-35% at W = 5-8 and in the full matrix.
- **Verdict: will not pay before 2027.**
  - Hardware: Nova Lake (client, end of 2026, possibly not on every SKU) and Diamond Rapids (2027).
    AMD Zen 6 reportedly has no APX.
  - rustc: `apxf` is unstable as `#[target_feature]`, so a per-copy APX build needs nightly.
  - No CI runner. GCC already disables memory-form NDD on Nova Lake for performance.
- **Revisit when:** #139284 stabilizes and a cloud instance type with APX exists.

### 2.8 ADX (`adcx`/`adox`)
- **Verdict: no use.** A column has exactly one big-integer addition (fact 4). ADX's two independent
  carry chains help only with two simultaneous multiword additions, such as two pairs, which the API
  excludes.

### 2.9 32-bit x86 (win32) through SSE2 64-bit lanes (perf-ideas 5d)
- **Measured:**
  - i686 scalar u64 step: 65 instructions per column, with `shld`, `add`/`adc` pairs and
    read-modify-write on `(%esp)` on the chain (8 GPRs). Roughly 12-15 cycles.
  - SSE2 form (`pand`/`pandn`/`paddq`/`psllq`, probe `probe_word_i686_sse2`): 25 instructions and a
    6-cycle chain.
  - So about 2x on single-word kernels for win32 users.
- **Reach:** Pillow's 2023 data puts win32 at 0.9-1.9% of downloads for 3.9-3.11, and Pillow then
  dropped win32 for 3.12+. lev ships win32 for 3.10-3.15t, so expect well under 1% of installs.
- **Effort:** M (single-word only; multiword carries need SSE4.2 `pcmpgtq`, which i686 lacks).
- **Verdict:** Correct and cheap per user, but low priority.

## 3. ARM

### 3.1 NEON table-free `word_u8` on non-Apple AArch64 (test first, no detection needed)
- **State:** It is gated to Apple (`mod.rs:190-193`) because the Neoverse cores were unmeasured.
- **Why it might pay:**
  - N2 has 2 vector pipes: 10 NEON ops per byte take about 5 cycles, against the 6-cycle step. That
    is borderline throughput-bound, but it saves 1-2 KiB of zeroing per call.
  - V2/V3 have 4 vector pipes (LLVM `AArch64SchedNeoverseV2.td`), so about 2.5 cycles per byte.
- **Expected gain:** -5..-20% at 17-64 u8 on V2/V3, -5..+5% on N2.
- **Effort:** S: change the cfg; the form is already compiled and tested there.
- **Test:** A/B linux-aarch64/windows-aarch64 (N2, sometimes V3). Falsified if N2 regresses at n = m.
  If it wins only on V-series, it waits for 3.2.

### 3.2 Runtime core detection on AArch64 (perf-ideas 5a/5c)
- **Mechanism:** At import, read MIDR_EL1:
  - Linux: `/sys/devices/system/cpu/cpu0/regs/identification/midr_el1`, or the "CPU implementer" line
    of `/proc/cpuinfo`, or `mrs` when `AT_HWCAP & HWCAP_CPUID`.
  - Windows: the registry value `CP 4000` under `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`,
    which LLVM's host detection uses.
  - Implementer 0x61 (Apple) or 0x51 with the Oryon part: select an Apple-forms copy (`Isa` gains
    `const APPLE_CORES`). Docker Desktop guests report implementer 0x61 with part 0, so match on the
    implementer only.
- **What Linux on Apple loses today:**
  - Single-word chain 7 vs 6 cycles (-14%; fact 5).
  - NEON table-free off: -17..-24% at n = m for 17-64, measured in #266 on M2.
  - W = 2-3 multiword chain-bound in the Neoverse form (`extr` is 1 cycle on Apple, the shifted
    `orr` 2).
  - About -10..-25% on R2/R3 for those users.
- **Oryon:** shifted-register logic is 2 cycles and `EXTR` 1, the Apple pattern
  (`AArch64SchedOryon.td`), so the Apple forms fit.
- **Reach:**
  - Linux arm64 containers on Macs (Docker Desktop, Lima, OrbStack): mostly development and test
    loops, not production.
  - Asahi: negligible.
  - Windows on Snapdragon: about 1-2% of Windows PCs.
  - Best guess: low single-digit percent of installs, rarely the latency-critical ones.
- **Effort:** M-L. A second AArch64 copy roughly doubles the hot code (linux aarch64 wheel 265 KB).
  Detection needs per-OS code.
- **Risk:** Detection bugs, plus the copy's layout effects on N2.
- **Test:**
  - First the 5a measurement locally: force the Neoverse forms on macOS. It is cheap and decides
    whether to build this.
  - Then verify in a Linux VM on the M2. CI can only check detection on N2/V3 (falls back) and the
    forms on the macOS runner.
- **Verdict:** Worth doing only if 5a shows >= 10%.

### 3.3 Neoverse V1/V2/V3 retune
- **Verdict: no new step form needed.** V2's latencies match N2's: shifted logic 1 cycle, `EXTR`
  3 cycles outside the ROR idiom. V2 differs in width: 6 integer ALUs and 4 vector pipes.
  - The Neoverse step and multiword forms remain right.
  - Only throughput-bound choices could move: the 8x gate of 3.1, and the AArch64 `STRIP = 8`
    (`aarch64/mod.rs:16`).
- **Platforms:** Graviton 3 = V1, Graviton 4 and Axion = V2, Cobalt 100 = N2, Cobalt 200 = V3. Arm
  runners already draw some V3 (fact 8), so log the CPU before trusting a run.

### 3.4 SVE/SVE2 (MATCH/NMATCH, BDEP/BEXT, predicates)
- **Verdict: will not pay.**
  - MATCH answers "does pattern byte i occur anywhere in this 16-byte segment", not "which pattern
    bytes equal text char c". It does not build a match mask.
  - The per-character mask is `cmpeq` into a predicate, but SVE cannot move a predicate to a GPR
    without a store and reload (or SVE2.1 `pmov`).
  - VL is 128 bits on N2/V2/V3, so it is no wider than NEON, whose `shrn` movemask is already about
    10 ops per byte. BEXT on 64-bit lanes still needs a cross-lane combine.
  - Rust has no stable SVE intrinsics, so this would be whole-loop `asm!`.
  - The only 256-bit VL core is V1 (Graviton 3).

### 3.5 Apple M3/M4/M5 and SME
- **Steps:** The Apple `asm!` step (`apple.rs:20-35`) uses only unshifted 1-cycle ops, so it is
  robust to later cores. No public M4 latency table shows shifted-op or `EXTR` changes. Geekerwan
  reports wider decode and execution changes, but no per-op data. Re-check against Apple's Silicon
  CPU Optimization Guide v4 tables if a core makes shifted ops 1-cycle, since a shorter-instruction
  form might then tie. Expected 0-5%.
- **SME:** Streaming mode has high entry and exit cost, long-latency ops on a shared unit, and no
  non-streaming SVE on M4. Rust support is unstable. A latency-bound scalar chain gains nothing.
  **No.**
- **target-cpu for macOS:** The default is already `apple-m1` (fact 6). Raising it excludes M1. **No.**

### 3.6 NEON tricks
- `shrn` movemask: already used (`neon.rs:108`). Reuse it for 2.1.
- `vcnt` popcount: only at the end; `cnt` + `addv` is what LLVM emits already.
- `vtbl`: already used for pattern loading.
- **Verdict:** Nothing further.

## 4. Other targets (perf-ideas 5e)
- **Verdict on all four: ship wheels first; no custom kernels until a measurement shows a problem.**
- **wasm32 (Pyodide):**
  - SIMD128 has `i8x16.eq` + `i8x16.bitmask`, so tiny/table-free masks port trivially, and i64
    ops/`popcnt` are native.
  - PEP 783 Emscripten wheels and maturin's `wasm32-unknown-emscripten` make shipping feasible.
  - Value: low (browser use).
- **riscv64:**
  - Zbb has `andn`/`orn`/`cpop`. The step would get 6 cycles like BMI1 and an AArch64-like form.
  - The baseline rv64gc lacks Zbb, and `is_riscv_feature_detected!` is unstable, so a copy needs
    hwprobe by hand.
  - Few users.
- **ppc64le:** `andc`/`orc` are baseline, so LLVM already gets the short chain.
- **s390x:**
  - Big-endian. The header decode correctly falls back to the C API (`python.rs:30-34`), and the
    affix scans use explicit `from_le_bytes`.
  - Needs a QEMU test job before shipping.

## 5. Rust, LLVM and toolchain

### 5.1 Pin the Rust toolchain (new `rust-toolchain.toml`)
- **Why:**
  - Release wheels use whatever stable the maturin containers, runner images or rustup install that
    day.
  - A/B uses `dtolnay/rust-toolchain@stable`. Both float, so an LLVM upgrade can move lev 5-20%
    between releases with no code change (the layout hazard).
- **Mechanism:** Pin a version, bump it deliberately in its own PR with an A/B run, and note the
  version in the changelog.
- **Gain and effort:** 0% speed, but reproducible results. Effort S.
- **Risk:** The MSRV is 1.91 (`Cargo.toml`). Keep a CI job on the MSRV. Needs approval: this changes
  the release configuration.

### 5.2 Code-alignment flags as a layout-variance cure
- **Flags (stable via `-C llvm-args`):**
  - `-align-loops=64`.
  - `-align-all-functions=6`.
  - `-align-all-nofallthru-blocks=5`: pads only blocks entered by a jump, so no executed NOPs.
  - x86 only: `-x86-align-branch-boundary=32 -x86-align-branch=fused+jcc+jmp`, which mitigates the
    Skylake-family JCC erratum on many user laptops (Skylake to Comet Lake) but not on CI's AMD
    runners.
- **Not available on stable:** `-Zmin-function-alignment` and `#[align]` are nightly.
- **What it fixes:** Some of lev's swings are alignment: the 17% `ratio` move after a size change in
  `mod.rs:46-48`. Others are register allocation or inlining (`mod.rs:44-46`, `hash_tail` docs),
  which alignment does not fix.
- **Expected:** About 0% mean, with lower run-to-run variance across unrelated changes. Size +2-5%.
- **Test:** Two A/B rounds.
  1. Flags vs none on main (expect neutral).
  2. Re-run a known layout-sensitive change, such as `#[cfg]` instead of `if cfg!` in `full`, with
     the flags on in both A and B. If the 17% swing shrinks below 5%, adopt.
- **Effort:** S. It is a profile/rustflags change, so a separate PR (AGENTS.md rule 7).

### 5.3 PGO for release wheels
- **Mechanism:**
  - Build with `-Cprofile-generate` through maturin's `RUSTFLAGS`, run `test_bench.py`'s cases or
    `benchmark.py` on the target arch, then `llvm-profdata merge` and rebuild with `-Cprofile-use`.
    pydantic-core ships PGO wheels this way.
  - Every arch needs a native profiling run. aarch64 Linux is cross-built today; use
    `ubuntu-24.04-arm`. universal2 needs both slices.
- **What gains:**
  - R1 dispatch and PyO3 glue (branch layout, cold splitting): plausibly -2..-5% at 40-50 ns calls.
  - The kernels hardly change, since they are straight-line loops.
- **Layout:** PGO does not cure layout sensitivity. Function order and alignment still depend on code
  size, and it adds training-set sensitivity: untrained regimes get pessimized.
- **Effort:** M-L (CI). **Confidence:** Low-medium.
- **Test:** A PGO B vs a plain A in ab-benchmark, which needs a workflow change, with approval.
- **BOLT:** Linux-only. lev's hot set is a few KB that already fits L1i. **Will not pay.**

### 5.4 Hints and bounds checks
- **`hint::cold_path`** (stable in 1.95; the MSRV is 1.91, so use a `#[cold]` call wrapper or bump
  the MSRV) on:
  - the band give-up returns (`portable.rs:186`, `302`);
  - heap fallbacks (`scratch`, `lib.rs:1966`);
  - hash probe misses in table builds.
  - This is a deliberate layout lever that could replace some `#[inline(never)]` placement.
    Gain 0-3%.
- **`select_unpredictable`:** No good site. The hot loops are already branch-free except the hash
  probes, which were measured.
- **`assert_unchecked` and bounds checks:** The x86 asm has 117 `panic_bounds_check` and 82 div-by-zero
  call sites, all outside the hot column loops, so nothing to gain there.
- **Divisions in the band check:** `band_gives_up` (`mod.rs:126`) does a `divq`/`divl` every 16
  columns: 26-88 cycles on Skylake-family, partly hidden. Rewriting the pace test as a
  multiplication (`(score - s - SLACK) * n > (limit - s) * j`, floor semantics checked in tests)
  removes it. 0-3% of band time on older Intel. Effort S.

### 5.5 `#[target_feature]` copies vs the `multiversion` crate; inline asm vs intrinsics
- **Keep the `Isa` copy trick.** `multiversion` dispatches per function through a cached pointer,
  which blocks the inlining that `mod.rs:49-52` documents as necessary.
- **New forms:** plain Rust or intrinsics where LLVM keeps the form (andn: yes; ternlog: yes).
  `asm!` only where LLVM rewrites the schedule, as in the Apple step and NEON `bit`.

### 5.6 `panic = "abort"`
- **Effect:**
  - Removes landing pads and PyO3's unwind catch.
  - The 2026-10-22 note measured `catch_unwind` at 5-6 ns per `ratio` call on 3.14, in an older
    fastcall path.
  - Gain 0-3 ns per call at R1, smaller binaries.
- **Risk:** Behavior change: a latent panic aborts Python instead of raising `PanicException`.
  `cargo test` still unwinds. Release-profile PR, user decision. **Low priority.**

## 6. Ranked table

| # | Idea | Regime / arch / kind | Est. gain | Effort | Confidence |
|---|---|---|---|---|---|
| 1 | SIMD masks for wide-unit tiny patterns (2.1) | R1, m 9-16, UCS-2/4/mixed; x86 + AArch64; distance and ratio | -10..-35% per call | S-M | Medium-high (asm: 83-101 scalar instructions per column) |
| 2 | Small band step algebra on x86 (2.2) | R3/R4 similar, x86, Levenshtein | -10..-25% of band time | S | High (asm 8 -> 6 cycles); medium on wall clock (Zen 3/4 throughput) |
| 3 | AVX2 two-compare table-free 33-64 (2.3) | R2 u8 33-64, n <= 4-8m, x86 AVX2 | -5..-18% | S | Medium-high (#266 precedent) |
| 4 | x86-64-v3 copy: andn, shrx/shlx (2.4) | R1/R2 all kinds (Lev), R3 band; x86 | -4..-12% per call | M | Medium (layout risk, +35-45% binary) |
| 5 | NEON table-free on Neoverse (3.1) | R2 u8 17-64, Linux/Windows arm | -5..-20% V2/V3; about 0 on N2 | S | Low-medium |
| 6 | Pin toolchain (5.1) | Reproducibility, all | 0% (variance) | S | High |
| 7 | Alignment flags experiment (5.2) | Variance, x86 + arm | About 0% mean, less swing | S | Low-medium |
| 8 | AVX-512VL ternlog single-word step (2.5) | R1/R2 Lev on Zen 4 / Intel AVX-512 (not Zen 5) | -10..-20% | M | Medium (asm 5 cycles; gating, CI coverage) |
| 9 | AArch64 core detection, Apple forms on Linux/Oryon (3.2) | R2/R3 for Linux-on-Mac, Snapdragon | -10..-25% for those users | M-L | Medium; small reach |
| 10 | Remove the band-check division, add cold_path (5.4) | R3 band, older Intel | 0-3% | S | Low |
| 11 | PGO (5.3) | R1 dispatch | -2..-5% | M-L | Low-medium |
| 12 | win32 SSE2 step (2.9) | All single-word, win32 only | About 2x kernel, under 1% of installs | M | Medium on asm, low value |
| 13 | panic = "abort" (5.6) | R1 | 0-3 ns | S | Low; behavior change |
| - | Will not pay | APX (2.7) until 2027 + stable `apxf`; ADX (2.8); BOLT; SVE/SVE2 MATCH/BDEP (3.4); SME, macOS target-cpu (3.5); Neoverse V2 step retune (3.3); AVX-512BW masks alone (2.6); `multiversion`; kernels for riscv/ppc/s390x/wasm before wheels exist (4) | 0 | - | - |

## 7. Top 3

1. **SIMD match masks for UCS-2/UCS-4/mixed patterns of 9-16 units (2.1).** The asm shows LLVM never
   vectorized `tiny_n`. Each text unit costs 16 scalar compare+set pairs (83-101 instructions per
   column) on both x86 and Apple. That makes the loop throughput-bound at 2-3x the Myers chain, in
   the regime where CJK and emoji strings live. Baseline SSE2 (`pcmpeqw`/`pcmpeqd` + packs +
   `pmovmskb`) and NEON (`cmeq` + `shrn`) need no new copy, so one small PR covers every platform.
2. **Small band step algebra for x86 (2.2).** #260's band runs at 8 cycles per column on x86, against
   6 on AArch64, because the x86 lowering of `mh | !(xvs | ph)` keeps three negations on the chain.
   The `!ph = X & !vn` rewrite reaches 6 cycles without any new ISA (probe `band_new_v2`), and 26
   instructions with BMI1. It is an S-sized change confined to an x86 form of the band step.
3. **AVX2 33-64 masks now, then an x86-64-v3 copy (2.3 + 2.4).** The two-compare AVX2 form extends
   #266's -8..-22% to 33-64 inside the existing V2 copy, at low risk. The v3 copy is the clean way to
   finally get `andn` (7 -> 6 cycles on every single-word kernel, tiny and hash loops included) and
   `shrx` in the band. Decide between a third copy (+35-45% binary) and replacing V2 by V3 (size
   neutral, v2-only CPUs fall back to v1) with one A/B round. Make the A/B workflow log the CPU model,
   since GitHub's x64 pool mixes Zen 3, 4 and 5.

## Sources
- uops.info: [VPADDQ xmm](https://uops.info/html-instr/VPADDQ_XMM_XMM_XMM.html), [VPTERNLOGQ xmm](https://uops.info/html-instr/VPTERNLOGQ_XMM_XMM_XMM_I8.html), [SHRD r64,r64,cl](https://uops.info/html-instr/SHRD_R64_R64_CL.html)
- Chips and Cheese, [Zen 5 at Hot Chips 2024](https://old.chipsandcheese.com/2024/09/15/discussing-amds-zen-5-at-hot-chips-2024/) (2-cycle vector integer adds)
- LLVM scheduling models: [AArch64SchedOryon.td](https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AArch64/AArch64SchedOryon.td), [AArch64SchedNeoverseV2.td](https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AArch64/AArch64SchedNeoverseV2.td)
- APX: [Phoronix, Nova Lake AVX10.2 and APX](https://phoronix.com/news/Nova-Lake-Does-AVX10.2-APX), [Tom's Hardware, Diamond Rapids 2027](https://www.tomshardware.com/pc-components/cpus/intel-xeon-7-diamond-rapids-cpus-officially-launching-in-2027-on-intel-18a-p-next-gen-p-core-xeon-features-pcie-6-0-50-percent-higher-core-counts-and-twice-the-memory-bandwidth), [Rust tracking issue #139284](https://github.com/rust-lang/rust/issues/139284), [Unstable Book: apx_target_feature](https://doc.rust-lang.org/unstable-book/language-features/apx-target-feature.html)
- GitHub runner CPUs: [RunsOn CPU benchmarks](https://runs-on.com/benchmarks/github-actions-cpu-performance/)
- Docker on Apple silicon reports implementer 0x61: [docker/roadmap#384](https://github.com/docker/roadmap/issues/384)
- win32 download share: [discuss.python.org, dropping 32-bit packages (Pillow 2023)](https://discuss.python.org/t/dropping-32-bit-packages/5476/20)
- Oryon integer width: [Chips and Cheese, Oryon](https://chipsandcheese.com/p/qualcomms-oryon-core-a-long-time-in-the-making)
- M4 changes (no per-op latency data): [Eclectic Light, M4 P cores](https://eclecticlight.co/2024/11/11/inside-m4-chips-p-cores/)
