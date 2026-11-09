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
//!   [`x86_64::bmi1`], whose forms also need POPCNT, `x86_64::avx2` and
//!   [`aarch64::neon`]).  Selected by runtime feature detection, or by
//!   `cfg` where the target guarantees it.
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
//! [`multiword`], [`word_u8`], [`full`], [`lcs_full`]), which hold every
//! selection rule.
//!
//! # Selection
//!
//! The dispatchers select with `#[cfg(...)]` on their statements, so a
//! build instantiates only the forms its target can select (with both
//! AArch64 forms of [`multiword`] instantiated in one function, LLVM
//! allocated the registers of the strips differently).  [`full`] keeps
//! `if cfg!(...)`: with `#[cfg]`, the x86-64 code of the distance kernels
//! that call it changed size, and the shifted layout cost `ratio` 17% on
//! 800-character UCS-4 pairs on Linux x86-64.  ISA features are
//! checked with `is_x86_feature_detected!` on every call, not through a
//! cached function pointer, which would keep [`multiword`] from being
//! inlined into the short-pattern path.  std caches the result, so each
//! call costs one load and a branch.
//!
//! The crate itself is compiled for each target's baseline ISA; on x86-64
//! the hot path has a second copy with the x86-64-v2 features (see
//! [`crate::Isa`] and [`x86_64::v2`]).  The dispatchers take the copy as
//! `X`: the forms of [`x86_64::bmi1`] need the x86-64-v2 features besides
//! BMI1, so only that copy selects them, and there the check for POPCNT
//! is `X::X86_64_V2`, a constant, as it was `cfg!(target_feature)` in the
//! crate-wide x86-64-v2 build before.
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
//! - `aarch64::neon::word_u8` for texts of any length: at 16 text
//!   characters per pattern character `ratio` is 1-5% slower than the table
//!   on Apple M2 (see [`TABLE_FREE_MAX_TEXT_PER_ROW`]).  With the pattern
//!   copied into a padded stack buffer and loaded with `vld4q_u8`, which
//!   costs a `memcpy` call and a store-to-load stall, `ratio` gained 5%
//!   less on texts as long as the pattern.
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
pub(crate) mod x86_64;

use crate::{CodeUnit, Isa, Kernel, LargeCtx};

/// Text columns before the first give-up check of [`small_band`]; the
/// checks then double up to [`SMALL_BAND_CHECK`].  Dissimilar pairs give
/// up at the first: at 65-128 chars the full kernel costs only 1.5 times
/// as much per column as the band.
const SMALL_BAND_FIRST: usize = 4;

/// Text columns between the later give-up checks of [`small_band`].
const SMALL_BAND_CHECK: usize = 16;

/// Edits by which [`small_band`] may run ahead of its pace before it gives
/// up.
const SMALL_BAND_SLACK: usize = 2;

/// Text columns per edit of the fastest pace [`small_band`] keeps up with,
/// whatever its certification limit.
const SMALL_BAND_PACE: usize = 4;

/// Whether a band of [`small_band`] or [`small_band2`] with certification
/// limit `limit` gives up at text column `j` of `n` with `score`, the cell
/// of diagonal `s = n - m` in column `j`, which bounds the band's result
/// from below: when the result can no longer be certified, or when the
/// edits beyond `s` run [`SMALL_BAND_SLACK`] ahead of the pace that ends
/// at `limit`, or of one edit per [`SMALL_BAND_PACE`] columns on texts
/// short enough for that to be slower.
#[inline(always)]
pub(crate) fn band_gives_up(score: usize, s: usize, limit: usize, j: usize, n: usize) -> bool {
    let pace = ((limit - s) * j / n).min(j / SMALL_BAND_PACE);
    score > limit || score > s + SMALL_BAND_SLACK + pace
}

/// Largest distance a band of `diagonals` (64 or 128) certifies when the
/// text is `s` chars longer than the pattern: a path of cost d stays within
/// `floor((d - s) / 2)` diagonals of `[0, s]` (Ukkonen), and the band reaches
/// `floor((diagonals - 1 - s) / 2)` beyond it on either side.
#[inline(always)]
pub(crate) fn band_limit(diagonals: usize, s: usize) -> usize {
    s + 2 * ((diagonals - 1 - s) / 2) + 1
}

/// Edge byte between stacked strips of a tall pattern (see
/// [`aarch64::full`]): bit 0 is the carry of the column addition,
/// bits 1 and 2 the `+1`/`-1` horizontal delta leaving the strip's top row.
/// Below row 0 the delta is `+1` and there is no carry.
const EDGE_ROW0: u8 = 0b010;

/// Longest text, per pattern character, that [`word_u8`] gives a vector
/// form.  The vector forms save the table's setup once per call but spend
/// more work per text character.  On Apple M2 the NEON form's LCS loop,
/// whose chain is short, turns throughput-bound: it loses 1-5% at 16 text
/// characters per pattern character and wins 2-15% at 4 to 8.  x86-64
/// shares the limit untuned.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_vendor = "apple")
    )),
    allow(dead_code)
)]
const TABLE_FREE_MAX_TEXT_PER_ROW: usize = 8;

/// Single-word kernel `K` for a byte pattern of 17 to 64 characters (the
/// tiny path takes shorter ones) against a byte text.  `SLOTS` is the
/// alphabet of the pair: 128 for pure ASCII, 256 for Latin-1.
///
/// For texts of up to [`TABLE_FREE_MAX_TEXT_PER_ROW`] characters per
/// pattern character, selects a form that computes each text character's
/// match mask by vector compares against the pattern held in registers:
/// `aarch64::neon::word_u8` on Apple targets (by OS vendor, as for
/// [`myers_step`]; other AArch64 cores are unmeasured, and Neoverse N1 and
/// N2 have half the vector pipes of Apple's cores), and
/// `x86_64::avx2::word_u8` for patterns of up to 32 characters in the
/// x86-64-v2 copy `K::Isa` on CPUs with AVX2.  Otherwise selects
/// [`portable::word_u8`], which looks the mask up in a peq table.
#[inline(always)]
pub(crate) fn word_u8<K: Kernel, const SLOTS: usize>(pattern: &[u8], text: &[u8]) -> usize {
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_vendor = "apple")
    ))]
    let short_text = text.len() <= TABLE_FREE_MAX_TEXT_PER_ROW * pattern.len();
    #[cfg(target_arch = "x86_64")]
    if short_text
        && pattern.len() <= 32
        && <K::Isa as Isa>::X86_64_V2
        && std::is_x86_feature_detected!("avx2")
    {
        // SAFETY: the copy `K::Isa` has the x86-64-v2 features, AVX2 was
        // just detected, and the pattern has 17 to 32 bytes.
        return unsafe { x86_64::avx2::word_u8::<K, SLOTS>(pattern, text) };
    }
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    if short_text {
        return aarch64::neon::word_u8::<K, SLOTS>(pattern, text);
    }
    portable::word_u8::<K, SLOTS>(pattern, text)
}

/// One column of the single-word Levenshtein recurrence (see
/// [`crate::hyrro_inner`]): returns the new `(vp, vn)`.  Selects
/// [`aarch64::apple::myers_step`] on Apple targets,
/// [`aarch64::neoverse::myers_step`] on other AArch64 targets (by OS
/// vendor, not by core; see *Layout*), [`x86_64::myers_step`] on x86-64,
/// and [`portable::myers_step`] elsewhere.
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    return aarch64::apple::myers_step(eq, vp, vn);
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    return aarch64::neoverse::myers_step(eq, vp, vn);
    #[cfg(target_arch = "x86_64")]
    return x86_64::myers_step(eq, vp, vn);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
/// to four words in the x86-64-v2 copy `X` on CPUs with BMI1; from five
/// words the state spills and the and-not form no longer pays (see
/// *Measured and rejected*).  Otherwise selects like [`myers_step`]: the Apple form on
/// Apple targets, the Neoverse form on other AArch64 targets (by OS vendor,
/// not by core) and [`portable::multiword`] elsewhere.
#[inline(always)]
pub(crate) fn multiword<X: Isa, const W: usize, I: Iterator<Item = [u64; W]>>(
    m: usize,
    pm_iter: I,
    edges: Option<&mut [u8]>,
) -> isize {
    #[cfg(target_arch = "x86_64")]
    if W <= 4 && edges.is_none() && X::X86_64_V2 && std::is_x86_feature_detected!("bmi1") {
        // SAFETY: the copy `X` has the x86-64-v2 features, and BMI1 was
        // just detected.
        return unsafe { x86_64::bmi1::multiword::<W, I>(m, pm_iter) };
    }
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    return aarch64::apple::multiword::<W, I>(m, pm_iter, edges);
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    return aarch64::neoverse::multiword::<W, I>(m, pm_iter, edges);
    #[cfg(not(target_arch = "aarch64"))]
    portable::multiword::<W, I>(m, pm_iter, edges)
}

/// Speculative Levenshtein distance of a pattern of `m` rows against `text`
/// (`m <= n < m + 64`), over a band of 64 diagonals around the ones from
/// `(0, 0)` to `(m, n)` in a single word, so each text column takes one
/// step instead of one per pattern word.  The peq rows have stride `w`;
/// text char `c` matches the row at `row(c) * w`.  Returns `Ok(distance)`
/// when the band certifies it (at most [`band_limit`]`(64, n - m)`, 63 or
/// 64 edits, see [`portable::small_band`]), else `Err((j, score))` with the
/// text column where it gave up (see [`band_gives_up`]) and the score
/// there, which it does early on pairs whose edits outpace that limit.
/// Selects [`portable::small_band`] on every target.
///
/// # Safety
///
/// `row(c) * w + w <= data.len()` for every `c` in `text`.
#[inline(always)]
pub(crate) unsafe fn small_band<T: CodeUnit, R: Fn(T) -> usize>(
    data: &[u64],
    w: usize,
    m: usize,
    text: &[T],
    row: R,
) -> Result<usize, (usize, usize)> {
    // SAFETY: forwarded caller contract.
    unsafe { portable::small_band(data, w, m, text, row) }
}

/// [`small_band`] over 128 diagonals in two words (`m <= n < m + 128`,
/// `w >= 2`): two steps per text column instead of one per pattern word,
/// certifying up to [`band_limit`]`(128, n - m)`, 127 or 128 edits.
/// Selects [`portable::small_band2`] on every target.
///
/// # Safety
///
/// `row(c) * w + w <= data.len()` for every `c` in `text`.
#[inline(always)]
pub(crate) unsafe fn small_band2<T: CodeUnit, R: Fn(T) -> usize>(
    data: &[u64],
    w: usize,
    m: usize,
    text: &[T],
    row: R,
) -> Result<usize, (usize, usize)> {
    // SAFETY: forwarded caller contract.
    unsafe { portable::small_band2(data, w, m, text, row) }
}

/// Full-matrix Levenshtein distance (fallback for `w > MAX_W`).  Selects
/// the strips of [`aarch64::full`] on every AArch64 core, whose 31
/// general-purpose registers hold a strip's state, and the loop over all
/// words of [`portable::full`] elsewhere (strips lose on x86-64, see
/// *Measured and rejected*).
pub(crate) fn full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    // `if cfg!` rather than `#[cfg]`: see *Selection*.
    if cfg!(target_arch = "aarch64") {
        aarch64::full(ctx)
    } else {
        portable::full(ctx)
    }
}

/// Full-matrix LCS length (fallback for `w > MAX_W`).  Selects strips in
/// the x86-64-v2 copy `X` on CPUs with BMI1 (`x86_64::bmi1::lcs_full`)
/// and on every AArch64 core ([`aarch64::lcs_full`]), and the loop over
/// all words of [`portable::lcs_full`] elsewhere.
pub(crate) fn lcs_full<X: Isa, F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    #[cfg(target_arch = "x86_64")]
    if X::X86_64_V2 && std::is_x86_feature_detected!("bmi1") {
        // SAFETY: the copy `X` has the x86-64-v2 features, and BMI1 was
        // just detected.
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
