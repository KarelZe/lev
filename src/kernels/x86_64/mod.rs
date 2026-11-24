//! x86-64 kernels: forms that need only the baseline ISA here, one module per
//! ISA feature ([`avx2`], [`bmi1`]), and the x86-64-v2 copy of the hot path
//! ([`v2`]).

pub(crate) mod avx2;
pub(crate) mod bmi1;
pub(crate) mod v2;

/// [`crate::kernels::portable::myers_step`] for x86-64, with a loop-carried
/// chain of 5 ops and its negations expressed as `!a & b`
/// ([`bmi1::myers_step_andn`]).  On CPUs with BMI1, each `!a & b` compiles
/// directly to `andn`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    bmi1::myers_step_andn(eq, vp, vn)
}

/// [`myers_step`] in fewer ops with a chain of 6
/// ([`bmi1::myers_step_andn_chain6`]), for loops that compute the match
/// masks with integer ops (see [`crate::kernels::myers_step_few_ops`]).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
pub(crate) fn myers_step_chain6(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    bmi1::myers_step_andn_chain6(eq, vp, vn)
}
