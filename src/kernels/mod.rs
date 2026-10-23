//! Bit-parallel kernels with target-specific forms, and the dispatch that
//! picks one form per target.
//!
//! # Layout
//!
//! - [`portable`]: the reference form of every kernel below.  Plain Rust,
//!   correct on every target and the fallback wherever nothing else is
//!   selected.
//! - `<target_arch>/<feature>`: forms that need an ISA extension, named
//!   after the `target_feature` they enable (e.g. [`x86_64::bmi1`], later
//!   `x86_64::avx2`, `aarch64::neon`).  Selected by runtime feature
//!   detection, or by `cfg(target_feature)` where the target guarantees it.
//! - `<target_arch>/<uarch>`: forms that use only the baseline ISA but are
//!   scheduled for one core family, named after that family as LLVM's
//!   `target-cpu` names it ([`aarch64::apple`], [`aarch64::neoverse`]).
//!   Selected at compile time by `cfg`.
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

/// Whether the strips of [`full`] and [`lcs_full`] are used: on every
/// AArch64 core, whose 31 general-purpose registers hold a strip's state.
/// [`myers_step`] and [`multiword`] have one form for Apple M-series cores
/// and one for the others (Neoverse), selected by `target_vendor` in `cfg`.
const AARCH64: bool = cfg!(target_arch = "aarch64");

/// Edge byte between stacked strips of a tall pattern (see
/// [`aarch64::full`]): bit 0 is the carry of the column addition,
/// bits 1 and 2 the `+1`/`-1` horizontal delta leaving the strip's top row.
/// Below row 0 the delta is `+1` and there is no carry.
const EDGE_ROW0: u8 = 0b010;

/// One column of the single-word Levenshtein recurrence (see
/// [`crate::hyrro_inner`]): returns the new `(vp, vn)`.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
pub(crate) use aarch64::apple::myers_step;
#[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
pub(crate) use aarch64::neoverse::myers_step;
#[cfg(not(target_arch = "aarch64"))]
pub(crate) use portable::myers_step;

/// Multi-word Levenshtein kernel over `m` pattern rows in `W` words.
/// Returns the sum of the vertical deltas of these rows in the last text
/// column, i.e. their contribution to the distance on top of `D(0, n) = n`
/// (see [`crate::hyrro_inner`]).  With `edges`, the kernel is one strip of
/// a taller pattern: it reads each column's incoming edge from there and
/// writes the outgoing one back.
///
/// On x86-64 CPUs with BMI1, whole patterns (no `edges`) of up to four words
/// run `x86_64::bmi1::multiword`.  From five words the state spills out of
/// x86-64's 16 registers and the spills, not the ALU work, set the pace: the
/// and-not form saved under 2% of the instructions there and measured within
/// noise to 4% slower.  Apple silicon runs [`aarch64::apple::multiword`],
/// other AArch64 cores [`aarch64::neoverse::multiword`], every other target
/// [`portable::multiword`].
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
    // `cfg` rather than `if APPLE_SILICON`: with both forms instantiated
    // here, LLVM allocates the registers of the strips differently.
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    return aarch64::apple::multiword::<W, I>(m, pm_iter, edges);
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    return aarch64::neoverse::multiword::<W, I>(m, pm_iter, edges);
    #[cfg(not(target_arch = "aarch64"))]
    portable::multiword::<W, I>(m, pm_iter, edges)
}

/// Full-matrix Levenshtein distance (fallback for `w > MAX_W`): strips on
/// AArch64, one loop over all words elsewhere.
pub(crate) fn full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    if AARCH64 {
        aarch64::full(ctx)
    } else {
        portable::full(ctx)
    }
}

/// Full-matrix LCS length (fallback for `w > MAX_W`), split like [`full`].
pub(crate) fn lcs_full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    if AARCH64 {
        aarch64::lcs_full(ctx)
    } else {
        portable::lcs_full(ctx)
    }
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
