//! Kernels for x86-64 CPUs with BMI1 (every x86-64 CPU since 2013, all of
//! them with the x86-64-v2 features), selected by runtime feature detection
//! in the x86-64-v2 copy of the hot path ([`super::v2`]).  The module is
//! named after BMI1, the feature that defines the and-not form of
//! `multiword` (`andn`).  The forms are compiled with the x86-64-v2
//! features too, as the copy that calls them is: POPCNT turns the final
//! scores into hardware popcounts, and SSE4.1 changes the code that loads
//! the rows.

use crate::kernels::delta;
#[cfg(target_arch = "x86_64")]
use crate::{lcs_multiword_kernel, LargeCtx};

/// Single-word Myers recurrence step taking advantage of BMI1 `andn` (`!a & b`)
/// and the PR #221 algebraic simplification.
///
/// # Safety
///
/// The CPU must support BMI1 target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1")]
#[allow(dead_code)]
#[inline]
pub(crate) unsafe fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    myers_step_andn(eq, vp, vn)
}

/// Plain-Rust form of [`myers_step`] taking advantage of and-not (`!a & b`),
/// which compiles to `andn` when BMI1 is enabled.
///
/// Uses the identities `!ph_s = (X << 1) & !(vn << 1)` and
/// `!(xv | ph_s) = (X << 1) & !(xv | (vn << 1))` with `X = sum | vp | eq`
/// (from PR #221), removing the `| 1` constant off the critical recurrence
/// path.
#[inline(always)]
pub(crate) fn myers_step_andn(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let sum = (eq & vp).wrapping_add(vp);
    let x = sum | (vp | eq);
    let mh = vp & !(sum & !eq);
    let xv = eq | vn;
    let vns = vn << 1;
    let xs = x << 1;
    let m = xv | vns;
    let s = !vns & xs;
    let a = !m & xs;
    ((mh << 1) | a, !s & xv)
}

/// [`multiword_andn`] compiled with BMI1, which runs each `!a & b` as one
/// `andn`.  The kernel is throughput-bound from two words on, so the fewer
/// instructions (15-18% per column at `W` = 2 and 4) translate into time.
///
/// # Safety
///
/// The CPU must support BMI1 and x86-64-v2 target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,popcnt,sse3,ssse3,sse4.1,sse4.2")]
pub(crate) unsafe fn multiword<const W: usize, I: Iterator<Item = [u64; W]>>(
    m: usize,
    pm_iter: I,
) -> isize {
    multiword_andn::<W, I>(m, pm_iter)
}

/// [`crate::kernels::portable::multiword`] (whole patterns only) with
/// `nhp = !hp` tracked instead of `hp`, so every negation pairs with an AND:
/// `hp = vn | !(d0 | vp)` becomes `nhp = !vn & (d0 | vp)`, and with
/// `nhp_s = !hp_s` (its shift-in is the negated `hp` bit, 0 below row 0)
/// `vp' = hn_s | (!d0 & nhp_s)` and `vn' = !nhp_s & d0`.  Exact for every
/// state; it only pays where `!a & b` is one instruction (`andn` with BMI1),
/// otherwise it costs a few more.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
pub(crate) fn multiword_andn<const W: usize, I: Iterator<Item = [u64; W]>>(
    m: usize,
    pm_iter: I,
) -> isize {
    let mut vp = [!0u64; W];
    let mut vn = [0u64; W];
    for pm_row in pm_iter {
        let (mut carry, mut prev_nhp, mut prev_hn) = (false, 0u64, 0u64);
        for k in 0..W {
            let (pm, v_p, v_n) = (pm_row[k], vp[k], vn[k]);
            let x = pm | v_n;
            let (sum, nc) = (x & v_p).carrying_add(v_p, carry);
            carry = nc;
            let d0 = (sum ^ v_p) | x;
            let nhp = !v_n & (d0 | v_p);
            let hn = v_p & d0;
            let (nhp_s, hn_s) = (nhp << 1 | prev_nhp, hn << 1 | prev_hn);
            vp[k] = hn_s | (!d0 & nhp_s);
            vn[k] = !nhp_s & d0;
            (prev_nhp, prev_hn) = (nhp >> 63, hn >> 63);
        }
    }
    delta(vp, vn, m)
}

/// Words per strip of [`lcs_full`]: a strip's state and the operands of its
/// carry chain fit x86-64's 16 registers.
#[cfg(target_arch = "x86_64")]
const LCS_STRIP: usize = 4;

/// [`crate::kernels::lcs_full`] strip by strip, like
/// [`crate::kernels::aarch64::lcs_full`] but four words wide.  The
/// portable loop over all words keeps the state in memory and, as every
/// logic op clobbers the flags, moves the carry between the flags and a
/// register on each word (`setb`, `bt`), a serial chain through the whole
/// column; a strip runs its words as one `adc` chain.  The LCS step has no
/// negation for BMI1 to fold, so the strips pay on their own; they share the
/// module and its detection with [`multiword`].
///
/// # Safety
///
/// The CPU must support BMI1 and x86-64-v2 target features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,popcnt,sse3,ssse3,sse4.1,sse4.2")]
pub(crate) unsafe fn lcs_full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    let mut carries = vec![0u8; ctx.n];
    let mut lcs = 0;
    for w0 in (0..ctx.w).step_by(LCS_STRIP) {
        let c = Some(&mut carries[..]);
        lcs += match ctx.w - w0 {
            1 => lcs_multiword_kernel::<1, _>(ctx.rows::<1>(w0), c),
            2 => lcs_multiword_kernel::<2, _>(ctx.rows::<2>(w0), c),
            3 => lcs_multiword_kernel::<3, _>(ctx.rows::<3>(w0), c),
            _ => lcs_multiword_kernel::<LCS_STRIP, _>(ctx.rows::<LCS_STRIP>(w0), c),
        };
    }
    lcs
}
