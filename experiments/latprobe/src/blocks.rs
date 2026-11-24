//! The `myers_step` blocks the probe scores, as asm copies of lev's forms.
//!
//! To score a candidate: add a function with the signature
//! `fn(eq: u64, vp: u64, vn: u64) -> (u64, u64)` and list it in
//! `for_each_block!` at the bottom. `main` checks every listed block against
//! [`reference`] before timing it.

use std::arch::asm;

/// `portable::myers_step` (src/kernels/portable.rs), the reference.
#[inline(always)]
pub fn reference(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let sum = (eq & vp).wrapping_add(vp);
    let ph = vn | !(sum | (vp | eq));
    let mh = vp & (eq | !sum);
    let xv = eq | vn;
    let ph_s = (ph << 1) | 1;
    ((mh << 1) | !(xv | ph_s), ph_s & xv)
}

/// `aarch64::apple::myers_step`, the current Apple form: 15 ops, chain 6.
#[inline(always)]
pub fn chain6(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
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
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            sum = out(reg) _, u = out(reg) _, xv = out(reg) _, vns = out(reg) _,
            m = out(reg) _, x = out(reg) _, t = out(reg) _, xs = out(reg) _,
            mh = out(reg) _, mhs = out(reg) _, s = out(reg) _, a = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// The falsified chain-5 candidate (PR #288): 18 ops, chain 5.
#[inline(always)]
pub fn chain5(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {e}, {eq}, #1",
            "orr  {eq}, {eq}, {vn}",
            "lsl  {p}, {vp}, #1",
            "lsl  {n}, {vn}, #1",
            "and  {t}, {e}, {p}",
            "add  {vn}, {t}, {p}",
            "eon  {vp}, {eq}, {p}",
            "orr  {p}, {p}, {e}",
            "orr  {e}, {e}, {n}",
            "bic  {vp}, {vp}, {e}",
            "orn  {p}, {n}, {p}",
            "and  {p}, {eq}, {p}",
            "orr  {e}, {eq}, {n}",
            "orn  {t}, {t}, {e}",
            "bic  {vp}, {vp}, {vn}",
            "bic  {vn}, {vn}, {n}",
            "eor  {vp}, {t}, {vp}",
            "bic  {vn}, {p}, {vn}",
            eq = inout(reg) eq => _, vp = inout(reg) vp, vn = inout(reg) vn,
            e = out(reg) _, p = out(reg) _, n = out(reg) _, t = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// Calls `$m!(name, function)` for every block the probe scores.
macro_rules! for_each_block {
    ($m:ident) => {
        $m!("chain6 (current)", blocks::chain6);
        $m!("chain5 (falsified)", blocks::chain5);
    };
}
pub(crate) use for_each_block;
