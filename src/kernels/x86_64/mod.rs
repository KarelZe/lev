//! x86-64 kernels, one module per ISA feature or core family (see
//! [`crate::kernels`]), and the x86-64-v2 copy of the hot path ([`v2`]).

pub(super) mod bmi1;
pub(crate) mod v2;
