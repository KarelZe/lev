//! Kernels scheduled for Apple M-series cores.  They use only the baseline
//! AArch64 ISA; other AArch64 cores (Neoverse, Windows on ARM) were never
//! measured with them and take the portable forms.

use crate::kernels::{delta, EDGE_ROW0};
use crate::{lcs_multiword_kernel, LargeCtx};

/// [`crate::kernels::portable::myers_step`] with its schedule pinned.  On
/// Apple cores a logical op with a shifted operand has twice the latency of
/// a plain one, yet LLVM folds the shifts into the ops on the chain and
/// reassociates the `| 1` back onto it (10 cycles instead of 7 per column).
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    // SAFETY: register-only arithmetic.  `vp`/`vn` are overwritten in place
    // after their last read, which keeps loop-carried copies out of the loop.
    unsafe {
        std::arch::asm!(
            "and  {sum}, {eq}, {vp}",
            "orr  {t}, {eq}, {vp}",
            "orr  {xv}, {eq}, {vn}",
            "add  {sum}, {sum}, {vp}",
            "orr  {t}, {t}, {sum}",
            "orn  {ph}, {vn}, {t}",
            "orn  {t}, {eq}, {sum}",
            "and  {mh}, {t}, {vp}",
            "orr  {xv1}, {xv}, #1",
            "lsl  {phs}, {ph}, #1",
            "lsl  {vp}, {mh}, #1",
            "orr  {t}, {xv1}, {phs}",
            "orn  {vp}, {vp}, {t}",
            "orr  {phs}, {phs}, #1",
            "and  {vn}, {phs}, {xv}",
            eq = in(reg) eq,
            vp = inout(reg) vp,
            vn = inout(reg) vn,
            sum = out(reg) _,
            t = out(reg) _,
            xv = out(reg) _,
            xv1 = out(reg) _,
            phs = out(reg) _,
            ph = out(reg) _,
            mh = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// [`crate::kernels::multiword`] in the short-chain step of
/// [`crate::kernels::portable::myers_step`], with the bit shifted in at the
/// bottom of a word taken as the top bit of the word below, so every
/// shift-in is one funnel shift (`extr`).  On x86-64 this form keeps more
/// values live than 16 registers hold and gets SLP-vectorized from five
/// words on.
#[cfg_attr(
    not(all(target_arch = "aarch64", target_vendor = "apple")),
    allow(dead_code)
)]
#[inline(always)]
pub(crate) fn multiword<const W: usize, I: Iterator<Item = [u64; W]>>(
    m: usize,
    pm_iter: I,
    mut edges: Option<&mut [u8]>,
) -> isize {
    let mut vp = [!0u64; W];
    let mut vn = [0u64; W];
    for (j, pm_row) in pm_iter.enumerate() {
        let e = edges.as_ref().map_or(EDGE_ROW0, |e| e[j]);
        let mut carry = e & 1 != 0;
        let (top_hp, top_hn) = {
            // Edge bits 1 and 2 moved to bit 63.
            let (mut below_ph, mut below_mh) = ((e as u64) << 62, (e as u64) << 61);
            for k in 0..W {
                let (eq, v_p, v_n) = (pm_row[k], vp[k], vn[k]);
                let (sum, nc) = (eq & v_p).carrying_add(v_p, carry);
                carry = nc;
                let ph = v_n | !(sum | (v_p | eq));
                let mh = v_p & (eq | !sum);
                let xv = eq | v_n;
                let ph_s = ph << 1 | below_ph >> 63;
                vp[k] = (mh << 1 | below_mh >> 63) | !(xv | ph_s);
                vn[k] = ph_s & xv;
                (below_ph, below_mh) = (ph, mh);
            }
            (below_ph >> 63, below_mh >> 63)
        };
        if let Some(e) = edges.as_deref_mut() {
            e[j] = carry as u8 | (top_hp as u8) << 1 | (top_hn as u8) << 2;
        }
    }
    delta(vp, vn, m)
}

/// Words per strip of [`full`] and [`lcs_full`]: a strip's state stays in
/// registers while it sweeps the text.  Eight words (16 state registers for
/// Levenshtein plus temporaries) fit AArch64's 31 general-purpose
/// registers; on x86-64's 16 they would spill on every column.
const STRIP: usize = 8;

/// Run `$kernel` with the const `W` bound to the width `$w` in `1..=8`.
macro_rules! with_width {
    ($w:expr, $W:ident => $kernel:expr) => {
        match $w {
            1 => {
                const $W: usize = 1;
                $kernel
            }
            2 => {
                const $W: usize = 2;
                $kernel
            }
            3 => {
                const $W: usize = 3;
                $kernel
            }
            4 => {
                const $W: usize = 4;
                $kernel
            }
            5 => {
                const $W: usize = 5;
                $kernel
            }
            6 => {
                const $W: usize = 6;
                $kernel
            }
            7 => {
                const $W: usize = 7;
                $kernel
            }
            _ => {
                const $W: usize = 8;
                $kernel
            }
        }
    };
}

/// [`crate::kernels::full`] strip by strip: the pattern is cut into strips of
/// [`STRIP`] words, bottom first, and each strip sweeps the whole text,
/// handing its per-column edges to the strip above.
pub(crate) fn full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    let (m, w) = (ctx.m, ctx.w);
    let mut edges = vec![EDGE_ROW0; ctx.n];
    let mut delta = 0isize;
    for w0 in (0..w).step_by(STRIP) {
        let rows = (m - 64 * w0).min(64 * STRIP);
        let e = Some(&mut edges[..]);
        delta += with_width!((w - w0).min(STRIP), W => {
            crate::kernels::multiword::<W, _>(rows, ctx.rows::<W>(w0), e)
        });
    }
    (ctx.n as isize + delta) as usize
}

/// [`crate::kernels::lcs_full`] strip by strip.
pub(crate) fn lcs_full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    let mut carries = vec![0u8; ctx.n];
    let mut lcs = 0;
    for w0 in (0..ctx.w).step_by(STRIP) {
        let c = Some(&mut carries[..]);
        lcs += with_width!((ctx.w - w0).min(STRIP), W => {
            lcs_multiword_kernel::<W, _>(ctx.rows::<W>(w0), c)
        });
    }
    lcs
}
