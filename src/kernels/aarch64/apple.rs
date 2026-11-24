//! Kernels scheduled for Apple M-series cores.  They use only the baseline
//! AArch64 ISA; other AArch64 cores (Neoverse, Windows on ARM) take
//! [`super::neoverse`].

use crate::kernels::{delta, EDGE_ROW0};

/// [`crate::kernels::portable::myers_step`] with a loop-carried chain of 5
/// single-cycle ops (`lsl`, `and`, `add`, `bic`, `eor`), one fewer than
/// [`myers_step_chain6`].
///
/// The shift moves before the addition: with `E = eq << 1`, `P = vp << 1`,
/// `N = vn << 1` and `xv = eq | vn`, `S = (E & P) + P` is `sum << 1`, and
/// each output is a two-op select on `S` (`P & N == 0` from `vp & vn == 0`):
///
/// - `vp' = A ^ (D & !S)` with `A = (E & P) | !(xv | N)` and
///   `D = (xv ^ !P) & !(E | N)`;
/// - `vn' = (xv & N) | (G & !S)` with `G = xv & !P & !(E | N)`;
///
/// and `E | N = xv << 1`.  18 instructions in 7 registers; `E` reads only
/// `eq`.  On an M2 Pro the column loop is bound less by the chain than by
/// the ops that become ready in the same cycle as the critical ones and by
/// program order, so the instructions are in order of readiness, with the
/// ops that have slack (`m`, `A`, `xn`) last in their cycle.  The latency
/// probe in `experiments/latprobe` scores this block (`chain5_ge`) at 5.9
/// cycles per column with `eq` from a table, [`myers_step_chain6`] at 6.55
/// (see `experiments/results.jsonl`).  Loops that compute the match masks
/// with integer ops take [`myers_step_chain6`] instead (see
/// [`crate::kernels::myers_step_few_ops`]).
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    // SAFETY: register-only arithmetic.  Every register is overwritten only
    // after the last read of the value it holds; `vp`/`vn` end up holding
    // the outputs, which keeps loop-carried copies out of the loop.
    unsafe {
        std::arch::asm!(
            "lsl  {a}, {eq}, #1",     // E = eq << 1
            "lsl  {vp}, {vp}, #1",    // P = vp << 1
            "orr  {eq}, {eq}, {vn}",  // xv = eq | vn
            "lsl  {vn}, {vn}, #1",    // N = vn << 1
            "and  {a}, {a}, {vp}",    // t = E & P
            "eon  {b}, {eq}, {vp}",   // xe = xv ^ !P
            "lsl  {c}, {eq}, #1",     // en = xv << 1 = E | N
            "bic  {d}, {eq}, {vp}",   // xp = xv & !P
            "add  {vp}, {a}, {vp}",   // S = t + P
            "bic  {b}, {b}, {c}",     // D = xe & !en
            "bic  {d}, {d}, {c}",     // G = xp & !en
            "orr  {c}, {eq}, {vn}",   // m = xv | N
            "bic  {b}, {b}, {vp}",    // y = D & !S
            "bic  {d}, {d}, {vp}",    // g = G & !S
            "orn  {a}, {a}, {c}",     // A = t | !m
            "and  {vn}, {eq}, {vn}",  // xn = xv & N
            "eor  {vp}, {a}, {b}",    // vp' = A ^ y
            "orr  {vn}, {vn}, {d}",   // vn' = xn | g
            eq = inout(reg) eq => _,
            vp = inout(reg) vp,
            vn = inout(reg) vn,
            a = out(reg) _,
            b = out(reg) _,
            c = out(reg) _,
            d = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// [`crate::kernels::portable::myers_step`] in the fewest ops, for loops
/// that compute each match mask with integer ops (see
/// [`crate::kernels::myers_step_few_ops`]).
///
/// Uses the identity `ph_s = !((X << 1) & !(vn << 1))` and
/// `!(xv | ph_s) = (X << 1) & !(xv | (vn << 1))` to remove the `| 1` constant
/// off the critical path: 15 unshifted single-cycle ALU instructions with a
/// loop-carried chain of 6 (`and`, `add`, `orr`, `lsl`, `bic`, `orr`/`bic`).
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
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
