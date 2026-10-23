//! x86-64 kernels: forms that need only the baseline ISA here, and one
//! module per ISA feature or core family (see [`crate::kernels`]).

pub(super) mod bmi1;

/// [`crate::kernels::portable::myers_step`] for x86-64, which has no or-not:
/// it tracks `nph = !ph`, so `(ph << 1) | 1 == !(nph << 1)` and every NOT
/// moves onto a value off the `vp` chain.  That shortens the loop-carried
/// chain from 9-10 cycles per column to 7 on Skylake, Golden Cove and Zen 3
/// to 5, with the baseline ISA; with BMI1 each and-not is one `andn`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let sum = (eq & vp).wrapping_add(vp);
    let nph = !vn & (sum | (vp | eq));
    let mh = vp & !(sum & !eq);
    let xv = eq | vn;
    let s = nph << 1;
    ((mh << 1) | (!xv & s), !s & xv)
}
