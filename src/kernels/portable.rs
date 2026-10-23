//! Reference forms of the target-specific kernels: plain Rust, selected on
//! every target that has no form of its own.

use super::{delta, EDGE_ROW0};
use crate::{lcs_len, lcs_step, LargeCtx};

/// One column of the single-word Levenshtein recurrence: returns the new
/// `(vp, vn)`.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let sum = (eq & vp).wrapping_add(vp);
    let ph = vn | !(sum | (vp | eq));
    let mh = vp & (eq | !sum);
    let xv = eq | vn;
    let ph_s = (ph << 1) | 1;
    ((mh << 1) | !(xv | ph_s), ph_s & xv)
}

/// [`super::multiword`] in Hyyrö's form, shifting each word's top bits into
/// the next word as 0/1 values.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
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
        let (mut prev_hp, mut prev_hn) = ((e >> 1 & 1) as u64, (e >> 2) as u64);
        for k in 0..W {
            let (pm, v_p, v_n) = (pm_row[k], vp[k], vn[k]);
            let x = pm | v_n;
            let (sum, nc) = (x & v_p).carrying_add(v_p, carry);
            carry = nc;
            let d0 = (sum ^ v_p) | x;
            let hp = v_n | !(d0 | v_p);
            let hn = v_p & d0;
            let (hp_s, hn_s) = (hp << 1 | prev_hp, hn << 1 | prev_hn);
            vp[k] = hn_s | !(d0 | hp_s);
            vn[k] = hp_s & d0;
            (prev_hp, prev_hn) = (hp >> 63, hn >> 63);
        }
        if let Some(e) = edges.as_deref_mut() {
            e[j] = carry as u8 | (prev_hp as u8) << 1 | (prev_hn as u8) << 2;
        }
    }
    delta(vp, vn, m)
}

/// [`super::full`] as one loop over all words per column, with the state on
/// the heap.
// AArch64 selects the strips; the tests still run this form there.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    let (m, w) = (ctx.m, ctx.w);
    let (mut vp, mut vn) = (vec![!0u64; w], vec![0u64; w]);
    ctx.for_each_base(|base| {
        let (mut carry, mut prev_hp, mut prev_hn) = (false, 1u64, 0u64);
        for (k, (vp_k, vn_k)) in vp.iter_mut().zip(vn.iter_mut()).enumerate() {
            let (pm, v_p, v_n) = (ctx.peq(base, k), *vp_k, *vn_k);
            let x = pm | v_n;
            let (sum, nc) = (x & v_p).carrying_add(v_p, carry);
            carry = nc;
            let d0 = (sum ^ v_p) | x;
            let hp = v_n | !(d0 | v_p);
            let hn = v_p & d0;
            let (hp_s, hn_s) = (hp << 1 | prev_hp, hn << 1 | prev_hn);
            *vp_k = hn_s | !(d0 | hp_s);
            *vn_k = hp_s & d0;
            (prev_hp, prev_hn) = (hp >> 63, hn >> 63);
        }
    });
    // Distance = n + sum of the last column's vertical deltas (see
    // `hyrro_inner`); rows above the pattern hold garbage.
    vp[w - 1] &= u64::MAX >> (64 * w - m);
    vn[w - 1] &= u64::MAX >> (64 * w - m);
    let count = |v: &[u64]| v.iter().map(|x| x.count_ones() as usize).sum::<usize>();
    ctx.n + count(&vp) - count(&vn)
}

/// [`super::lcs_full`] as one loop over all words per column.
// AArch64 selects the strips; the tests still run this form there.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn lcs_full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    let mut s = vec![!0u64; ctx.w];
    ctx.for_each_base(|base| {
        lcs_step(&mut s, false, |k| ctx.peq(base, k));
    });
    lcs_len(&s)
}
