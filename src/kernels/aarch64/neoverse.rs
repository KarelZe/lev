//! Kernels scheduled for Arm Neoverse cores (Graviton, Ampere, Azure Cobalt,
//! and the N2 behind GitHub's AArch64 runners).  They use only the baseline
//! AArch64 ISA and differ from [`super::apple`] in two latencies: on
//! Neoverse N1/N2/V1/V2 a logical op with a shifted operand takes one cycle
//! (two on Apple cores) and a two-register `extr` three (one on Apple).
//!
//! Both kernels are plain Rust, compiled and tested on every target.  Where
//! LLVM would undo the schedule, a value passes through [`opaque`], which
//! only hides it from the optimizer on AArch64 and is the identity
//! elsewhere.

use crate::kernels::{delta, EDGE_ROW0};

/// `x`, opaque to LLVM on AArch64: an empty `asm!` stops it from fusing
/// the ops on either side into a form that is slower on Neoverse.
#[inline(always)]
fn opaque(x: u64) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        let mut x = x;
        // SAFETY: an empty template; the register is passed through.
        unsafe {
            std::arch::asm!(
                "/* {0} */",
                inout(reg) x,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        x
    }
    #[cfg(not(target_arch = "aarch64"))]
    x
}

/// [`crate::kernels::portable::myers_step`] with every shift folded into a
/// logical op.  With `X = sum | vp | eq`, `!ph = X & !vn`, and the bit
/// shifted into `ph_s` is 1, so `!ph_s == !ph << 1` exactly.  Hence
/// `vn' = xv & !((X & !vn) << 1)` and `!(xv | ph_s) = C & (X << 1)`, where
/// `C = !xv & !(vn << 1)` does not depend on `vp` (its bit 0 is cleared by
/// `X << 1`).  Each output is then one shifted op past `X`: `and`, `add`,
/// `orr`, `and`, `orr` take 5 cycles from `vp` to `vp`, against 8 for the
/// portable step as LLVM schedules it, with the same 13 ops.  Without
/// [`opaque`], LLVM merges `C` into `X << 1 & !(xv | vn << 1)`, which
/// needs `X << 1` in a register: one op and one cycle more.
#[cfg_attr(
    not(all(target_arch = "aarch64", not(target_vendor = "apple"))),
    allow(dead_code)
)]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let sum = (eq & vp).wrapping_add(vp);
    let x = sum | vp | eq;
    let mh = vp & (eq | !sum);
    let xv = eq | vn;
    let c = opaque(!xv & !(vn << 1));
    let nph = x & !vn;
    ((mh << 1) | (c & (x << 1)), xv & !(nph << 1))
}

/// [`crate::kernels::multiword`] in the short-chain step of
/// [`crate::kernels::portable::myers_step`], with the bits shifted in at the
/// bottom of a word kept as 0/1 values.  LLVM turns `ph << 1 | below >> 63`
/// into a funnel shift (`extr`, 3 cycles on Neoverse, on the chain of every
/// word but the first); with the shifted-out bits made [`opaque`] it is a
/// 1-cycle `lsr` off the chain and one `orr` with a shifted operand on it.
/// `mh_s` is opaque as well, so that `vp'` is one `orn` rather than an
/// `mvn` and an `orr` with `mh << 1` folded in.  That takes the chain of a
/// word from `vp` to `vp` from 11-12 cycles (portable form) to 7, for about
/// the same number of ops.
#[cfg_attr(
    not(all(target_arch = "aarch64", not(target_vendor = "apple"))),
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
        let (mut below_ph, mut below_mh) = ((e >> 1 & 1) as u64, (e >> 2) as u64);
        for k in 0..W {
            let (eq, v_p, v_n) = (pm_row[k], vp[k], vn[k]);
            let (sum, nc) = (eq & v_p).carrying_add(v_p, carry);
            carry = nc;
            let ph = v_n | !(sum | (v_p | eq));
            let mh = v_p & (eq | !sum);
            let xv = eq | v_n;
            let ph_s = ph << 1 | below_ph;
            vp[k] = opaque(mh << 1 | below_mh) | !(xv | ph_s);
            vn[k] = ph_s & xv;
            (below_ph, below_mh) = (opaque(ph >> 63), opaque(mh >> 63));
        }
        if let Some(e) = edges.as_deref_mut() {
            e[j] = carry as u8 | (below_ph as u8) << 1 | (below_mh as u8) << 2;
        }
    }
    delta(vp, vn, m)
}
