//! Kernels scheduled for Apple M-series cores.  They use only the baseline
//! AArch64 ISA; other AArch64 cores (Neoverse, Windows on ARM) take
//! [`super::neoverse`].

use crate::kernels::{delta, EDGE_ROW0};

/// [`crate::kernels::portable::myers_step`] with a loop-carried chain of 5
/// single-cycle ops (`lsl`, `and`, `add`, `bic`, `eor` or `bic`), one
/// fewer than [`myers_step_chain6`].
///
/// The shift moves before the addition: with `E = eq << 1`, `P = vp << 1`
/// and `N = vn << 1`, `S = (E & P) + P` is `sum << 1`, ready after three
/// ops.  Each output is then a select on `S` between values that are ready
/// by then, two ops after `S` (`P & N == 0` from `vp & vn == 0`):
///
/// - `vp' = S ? Q1 : Q1 ^ D = Q1 ^ (D & !S)`, with
///   `Q1 = (E & P) | !(xv | N)` and `D = (xv ^ !P) & !(E | N)`;
/// - `vn' = xv & (N | !(S | P | E)) = f0 & !(S & !N)`, with
///   `f0 = xv & (N | !(P | E))`.
///
/// 18 unshifted ALU instructions, three more than [`myers_step_chain6`]: at
/// about six integer ops per cycle the column stays bound by the chain.
/// Each register is reused once its value is dead, so the block takes 7
/// registers (15 for [`myers_step_chain6`]).
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    // SAFETY: register-only arithmetic.  Every register is overwritten only
    // after the last read of the value it holds; `vp`/`vn` end up holding
    // the outputs, which keeps loop-carried copies out of the loop.
    unsafe {
        std::arch::asm!(
            "lsl  {e}, {eq}, #1",    // E = eq << 1
            "orr  {eq}, {eq}, {vn}", // xv = eq | vn
            "lsl  {p}, {vp}, #1",    // P = vp << 1
            "lsl  {n}, {vn}, #1",    // N = vn << 1
            "and  {t}, {e}, {p}",    // t = E & P
            "add  {vn}, {t}, {p}",   // S = t + P
            "eon  {vp}, {eq}, {p}",  // xe = xv ^ !P
            "orr  {p}, {p}, {e}",    // PE = P | E
            "orr  {e}, {e}, {n}",    // EN = E | N
            "bic  {vp}, {vp}, {e}",  // D = xe & !EN
            "orn  {p}, {n}, {p}",    // o = N | !PE
            "and  {p}, {eq}, {p}",   // f0 = xv & o
            "orr  {e}, {eq}, {n}",   // m = xv | N
            "orn  {t}, {t}, {e}",    // Q1 = t | !m
            "bic  {vp}, {vp}, {vn}", // y = D & !S
            "bic  {vn}, {vn}, {n}",  // sn = S & !N
            "eor  {vp}, {t}, {vp}",  // vp' = Q1 ^ y
            "bic  {vn}, {p}, {vn}",  // vn' = f0 & !sn
            eq = inout(reg) eq => _,
            vp = inout(reg) vp,
            vn = inout(reg) vn,
            e = out(reg) _,
            p = out(reg) _,
            n = out(reg) _,
            t = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// The previous Apple form of [`crate::kernels::portable::myers_step`], kept
/// to compare against [`myers_step`].
///
/// Uses the identity `ph_s = !((X << 1) & !(vn << 1))` and
/// `!(xv | ph_s) = (X << 1) & !(xv | (vn << 1))` to remove the `| 1` constant
/// off the critical path: 15 unshifted single-cycle ALU instructions with a
/// loop-carried chain of 6 (`and`, `add`, `orr`, `lsl`, `bic`, `orr`/`bic`).
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[cfg_attr(not(test), allow(dead_code))]
#[inline(always)]
pub(crate) fn myers_step_chain6(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    // SAFETY: register-only arithmetic.  `vp`/`vn` are overwritten in place
    // after their last read, which keeps loop-carried copies out of the loop.
    unsafe {
        std::arch::asm!(
            "and  {sum}, {eq}, {vp}",
            "orr  {u}, {eq}, {vp}",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {vns}, {vn}, #1",
            "add  {sum}, {sum}, {vp}",
            "orr  {m}, {xv}, {vns}",
            "orr  {x}, {u}, {sum}",
            "orn  {t}, {eq}, {sum}",
            "lsl  {xs}, {x}, #1",
            "and  {mh}, {t}, {vp}",
            "lsl  {mhs}, {mh}, #1",
            "bic  {s}, {xs}, {vns}",
            "bic  {a}, {xs}, {m}",
            "orr  {vp}, {mhs}, {a}",
            "bic  {vn}, {xv}, {s}",
            eq = in(reg) eq,
            vp = inout(reg) vp,
            vn = inout(reg) vn,
            sum = out(reg) _,
            u = out(reg) _,
            xv = out(reg) _,
            vns = out(reg) _,
            m = out(reg) _,
            x = out(reg) _,
            t = out(reg) _,
            xs = out(reg) _,
            mh = out(reg) _,
            mhs = out(reg) _,
            s = out(reg) _,
            a = out(reg) _,
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
