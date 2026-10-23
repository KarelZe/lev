//! AArch64 kernels, one module per ISA feature or core family (see
//! [`crate::kernels`]), and the strips of the full matrices, which every
//! AArch64 core runs.

use crate::kernels::EDGE_ROW0;
use crate::{lcs_multiword_kernel, LargeCtx};

pub(super) mod apple;
pub(super) mod neoverse;

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
// Selected on AArch64 only; the tests run this form on every target.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
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
