//! Bit-parallel kernels with target-specific forms, and the dispatch that
//! picks one form per target.
//!
//! # Layout
//!
//! - [`portable`]: the reference form of every kernel below.  Plain Rust,
//!   correct on every target and the fallback wherever nothing else is
//!   selected.
//! - `<target_arch>/<feature>`: forms that need an ISA extension, named
//!   after the `target_feature` that defines the form (e.g.
//!   [`x86_64::bmi1`], whose forms also need POPCNT; later `x86_64::avx2`,
//!   `aarch64::neon`).  Selected by runtime feature detection, or by
//!   `cfg(target_feature)` where the target guarantees it.
//! - `<target_arch>/<uarch>`: forms that use only the baseline ISA but are
//!   scheduled for one core family, named after that family as LLVM's
//!   `target-cpu` names it ([`aarch64::apple`], [`aarch64::neoverse`]).
//!   Selected at compile time by `cfg`, with `target_vendor = "apple"` as a
//!   proxy: it picks by OS vendor, not by core.  Apple targets always run
//!   on Apple cores.  Every other AArch64 target gets the Neoverse forms,
//!   including Windows on Snapdragon Oryon, whose cores behave more like
//!   Apple's (untuned: there is no runner to measure on).
//! - `<target_arch>`: forms that every core of one architecture runs, such
//!   as the strips of [`aarch64::full`], which need AArch64's 31 registers.
//!
//! The split follows `memchr` (`arch/{generic, x86_64/avx2, aarch64/neon}`)
//! and RustCrypto's `sha2` (`soft`, `x86_sha`, `aarch64_sha2`), with module
//! names spelled like `std::arch` and `target_feature`.  ISA features and
//! core tuning get separate names because only the former changes what a
//! CPU can run, as with `-march` and `-mtune` in GCC and Clang.
//!
//! # Naming
//!
//! A kernel has the same name and signature in every module that has a form
//! of it (`portable::multiword`, `aarch64::apple::multiword`,
//! `x86_64::bmi1::multiword`), so the path states the target.  Code outside
//! this module calls only the dispatchers here ([`myers_step`],
//! [`multiword`], [`full`], [`lcs_full`]), which hold every selection rule.
//!
//! # Selection
//!
//! Every dispatcher selects with `#[cfg(...)]` on its statements, so a
//! build instantiates only the forms its target can select (with both
//! AArch64 forms of [`multiword`] instantiated in one function, LLVM
//! allocated the registers of the strips differently).  ISA features are
//! checked with `is_x86_feature_detected!` on every call, not through a
//! cached function pointer, which would keep [`multiword`] from being
//! inlined into the short-pattern path.  std caches the result, so each
//! call costs one load and a branch.
//!
//! # Measured and rejected
//!
//! - `x86_64::bmi1::multiword` from five words: the state spills out of
//!   x86-64's 16 registers and the spills, not the ALU work, set the pace.
//!   The and-not form saved under 2% of the instructions there and measured
//!   within noise to 4% slower.
//! - An AVX2 `multiword` with the words as lanes of two vectors and the
//!   carries rippled through lane masks: a column took 40-65% fewer
//!   instructions but became one latency chain through the moves of the
//!   masks between vector and scalar registers, 1.1 to 1.9 times slower at
//!   five to eight words.
//! - [`full`] on x86-64 in strips of four words of `x86_64::bmi1::multiword`:
//!   12-23% slower than the portable loop over all words.
//! - [`full`] on x86-64 in strips of eight words of the AVX2 `multiword`: up
//!   to 1.8 times slower than the portable loop.
//!
//! # Compilation and tests
//!
//! Every module is compiled on every target; only items that cannot be
//! (`asm!`, `#[target_feature]`, `std::arch` intrinsics) carry
//! `#[cfg(target_arch = ...)]`.  `tests` checks each plain-Rust form
//! against the portable one or the DP oracle on every target, not only on
//! the one that selects it.

mod aarch64;
mod portable;
#[cfg(test)]
mod tests;
mod x86_64;

use crate::LargeCtx;

/// Edge byte between stacked strips of a tall pattern (see
/// [`aarch64::full`]): bit 0 is the carry of the column addition,
/// bits 1 and 2 the `+1`/`-1` horizontal delta leaving the strip's top row.
/// Below row 0 the delta is `+1` and there is no carry.
const EDGE_ROW0: u8 = 0b010;

/// One column of the single-word Levenshtein recurrence (see
/// [`crate::hyrro_inner`]): returns the new `(vp, vn)`.  Selects
/// [`aarch64::apple::myers_step`] on Apple targets,
/// [`aarch64::neoverse::myers_step`] on other AArch64 targets (by OS
/// vendor, not by core; see *Layout*) and [`portable::myers_step`]
/// elsewhere.
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    return aarch64::apple::myers_step(eq, vp, vn);
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    return aarch64::neoverse::myers_step(eq, vp, vn);
    #[cfg(not(target_arch = "aarch64"))]
    portable::myers_step(eq, vp, vn)
}

/// Multi-word Levenshtein kernel over `m` pattern rows in `W` words.
/// Returns the sum of the vertical deltas of these rows in the last text
/// column, i.e. their contribution to the distance on top of `D(0, n) = n`
/// (see [`crate::hyrro_inner`]).  With `edges`, the kernel is one strip of
/// a taller pattern: it reads each column's incoming edge from there and
/// writes the outgoing one back.
///
/// Selects `x86_64::bmi1::multiword` for whole patterns (no `edges`) of up
/// to four words on x86-64 CPUs with BMI1 and POPCNT; from five words the
/// state spills and the and-not form no longer pays (see *Measured and
/// rejected*).  Otherwise selects like [`myers_step`]: the Apple form on
/// Apple targets, the Neoverse form on other AArch64 targets (by OS vendor,
/// not by core) and [`portable::multiword`] elsewhere.
#[inline(always)]
pub(crate) fn multiword<const W: usize, I: Iterator<Item = [u64; W]>>(
    m: usize,
    pm_iter: I,
    edges: Option<&mut [u8]>,
) -> isize {
    #[cfg(target_arch = "x86_64")]
    if W <= 4
        && edges.is_none()
        && std::is_x86_feature_detected!("bmi1")
        && std::is_x86_feature_detected!("popcnt")
    {
        // SAFETY: both target features were just detected.
        return unsafe { x86_64::bmi1::multiword::<W, I>(m, pm_iter) };
    }
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    return aarch64::apple::multiword::<W, I>(m, pm_iter, edges);
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    return aarch64::neoverse::multiword::<W, I>(m, pm_iter, edges);
    #[cfg(not(target_arch = "aarch64"))]
    portable::multiword::<W, I>(m, pm_iter, edges)
}

/// Full-matrix Levenshtein distance (fallback for `w > MAX_W`).  Selects
/// the strips of [`aarch64::full`] on every AArch64 core, whose 31
/// general-purpose registers hold a strip's state, and the loop over all
/// words of [`portable::full`] elsewhere (strips lose on x86-64, see
/// *Measured and rejected*).
pub(crate) fn full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    #[cfg(target_arch = "aarch64")]
    return aarch64::full(ctx);
    #[cfg(not(target_arch = "aarch64"))]
    portable::full(ctx)
}

/// Full-matrix LCS length (fallback for `w > MAX_W`).  Selects strips on
/// x86-64 CPUs with BMI1 and POPCNT (`x86_64::bmi1::lcs_full`) and on every
/// AArch64 core ([`aarch64::lcs_full`]), and the loop over all words of
/// [`portable::lcs_full`] elsewhere.
pub(crate) fn lcs_full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("bmi1") && std::is_x86_feature_detected!("popcnt") {
        // SAFETY: both target features were just detected.
        return unsafe { x86_64::bmi1::lcs_full(ctx) };
    }
    #[cfg(target_arch = "aarch64")]
    return aarch64::lcs_full(ctx);
    #[cfg(not(target_arch = "aarch64"))]
    portable::lcs_full(ctx)
}

/// Sum of the vertical deltas `vp - vn` over the `m` pattern rows held in
/// `W` words; rows above the pattern in the top word hold garbage and are
/// masked off.
#[inline(always)]
fn delta<const W: usize>(mut vp: [u64; W], mut vn: [u64; W], m: usize) -> isize {
    let rows = u64::MAX >> (64 * W - m);
    vp[W - 1] &= rows;
    vn[W - 1] &= rows;
    let count = |v: &[u64; W]| v.iter().map(|x| x.count_ones() as isize).sum::<isize>();
    count(&vp) - count(&vn)
}
