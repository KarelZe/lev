//! x86-64 kernels: forms that need only the baseline ISA here, one module per
//! ISA feature ([`bmi1`]), and the x86-64-v2 copy of the hot path ([`v2`]).

pub(crate) mod bmi1;
pub(crate) mod v2;

/// [`crate::kernels::portable::myers_step`] for x86-64: uses the identities
/// `!ph_s = (X << 1) & !(vn << 1)` and `!(xv | ph_s) = (X << 1) & !(xv | (vn << 1))`
/// with `X = sum | vp | eq` to eliminate the `| 1` constant off the critical
/// recurrence path and express negations as `!a & b`.  On CPUs with BMI1,
/// each `!a & b` compiles directly to `andn`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    bmi1::myers_step_andn(eq, vp, vn)
}
