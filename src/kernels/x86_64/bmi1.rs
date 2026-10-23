//! Kernels for x86-64 CPUs with BMI1 and POPCNT (every x86-64 CPU since
//! 2013), selected by runtime feature detection.

use crate::kernels::delta;

/// [`multiword_andn`] compiled with `bmi1` and `popcnt`, which run each
/// `!a & b` as one `andn`.  The kernel is throughput-bound from two words
/// on, so the fewer instructions (15-18% per column at `W` = 2 and 4)
/// translate into time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,popcnt")]
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
