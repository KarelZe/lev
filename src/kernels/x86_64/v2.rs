//! The copy of the hot path for x86-64 CPUs with the x86-64-v2 features
//! (every x86-64 CPU since 2009), selected by runtime feature detection
//! when the module loads (see [`crate::Isa`]).
//!
//! The crate builds for `target-cpu=x86-64-v2` with these features turned
//! off again (`.cargo/config.toml`), so [`crate::Baseline`] runs on
//! x86-64-v1 and both copies keep the instruction scheduling of the
//! crate-wide x86-64-v2 build they replace (`target_feature` sets features
//! only).  The functions here turn the features back on; inlined into them,
//! the bodies of the out-of-line functions compile as in that build.
//! LAHF/SAHF, the last x86-64-v2 feature, has no stable name in rustc and
//! stays on in the whole crate; LLVM uses it only for x87 compares, which
//! lev has none of.
//!
//! rustc drops `#[inline(never)]` from `#[target_feature]` functions.  The
//! functions kept out of line therefore reach these through their
//! `#[inline(never)]` form in the crate, which has no target features, so
//! LLVM cannot inline into it and compiles it to one `jmp`.
//!
//! [`py`] exposes the copy's `#[pyfunction]` wrappers; PyO3's generated
//! argument-extraction code stays outside `#[target_feature]` and calls
//! [`distance`] and [`ratio`] here.

#[cfg(target_arch = "x86_64")]
use pyo3::{types::PyString, Borrowed};

#[cfg(target_arch = "x86_64")]
use crate::{CodeUnit, Isa, Kernel, LargeCtx};

/// The x86-64-v2 copy of the hot path.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct V2;

/// Put `#[target_feature]` with the features of [`V2`] on each item.
#[cfg(target_arch = "x86_64")]
macro_rules! v2 {
    ($($item:item)*) => {
        $(#[target_feature(enable = "popcnt,sse3,ssse3,sse4.1,sse4.2,cmpxchg16b")] $item)*
    };
}

/// Whether the CPU has the features of [`V2`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn detected() -> bool {
    std::is_x86_feature_detected!("popcnt")
        && std::is_x86_feature_detected!("sse3")
        && std::is_x86_feature_detected!("ssse3")
        && std::is_x86_feature_detected!("sse4.1")
        && std::is_x86_feature_detected!("sse4.2")
        && std::is_x86_feature_detected!("cmpxchg16b")
}

/// PyO3 wrappers for [`V2`], registered by `crate::_lev` when [`detected`] holds.
#[cfg(target_arch = "x86_64")]
pub(crate) mod py {
    use pyo3::prelude::*;
    use pyo3::types::PyString;

    crate::entry_points!(super::distance, super::ratio);
}

#[cfg(target_arch = "x86_64")]
impl Isa for V2 {
    const X86_64_V2: bool = true;
}

#[cfg(target_arch = "x86_64")]
v2! {
    /// [`crate::distance`] in this copy.
    pub(crate) unsafe fn distance(
        s1: Borrowed<'_, '_, PyString>,
        s2: Borrowed<'_, '_, PyString>,
    ) -> usize {
        crate::distance_in::<V2>(s1, s2)
    }

    /// [`crate::ratio`] in this copy.
    pub(crate) unsafe fn ratio(
        s1: Borrowed<'_, '_, PyString>,
        s2: Borrowed<'_, '_, PyString>,
    ) -> f64 {
        crate::ratio_in::<V2>(s1, s2)
    }

    /// [`crate::hyrro_64_tiny`] in this copy.
    pub(crate) unsafe fn hyrro_64_tiny<K: Kernel, T1: CodeUnit, T2: CodeUnit>(
        pattern: &[T1],
        text: &[T2],
    ) -> usize {
        crate::hyrro_64_tiny_body::<K, T1, T2>(pattern, text)
    }

    /// [`crate::mbleven`] in this copy: LLVM inlines its instance for this
    /// copy, which has no other caller.
    pub(crate) unsafe fn mbleven<X: Isa, T1: CodeUnit, T2: CodeUnit>(
        short: &[T1],
        long: &[T2],
        ub: usize,
    ) -> usize {
        crate::mbleven::<X, T1, T2>(short, long, ub)
    }

    /// [`crate::hyrro_multiword_direct`] in this copy.
    pub(crate) unsafe fn hyrro_multiword_direct<
        K: Kernel,
        const ROWS: usize,
        T1: CodeUnit,
        T2: CodeUnit,
    >(
        short: &[T1],
        long: &[T2],
    ) -> usize {
        crate::hyrro_multiword_direct_body::<K, ROWS, T1, T2>(short, long)
    }

    /// [`crate::hyrro_multiword_hash`] in this copy.
    pub(crate) unsafe fn hyrro_multiword_hash<K: Kernel, T1: CodeUnit, T2: CodeUnit>(
        short: &[T1],
        long: &[T2],
    ) -> usize {
        crate::hyrro_multiword_hash_body::<K, T1, T2>(short, long)
    }

    /// [`crate::LargeCtx::banded`] in this copy.
    pub(crate) unsafe fn banded<F: Fn(usize) -> usize>(
        ctx: &LargeCtx<'_, F>,
        t: usize,
        pv: &mut [u64],
        mv: &mut [u64],
    ) -> Result<usize, usize> {
        ctx.banded_body(t, pv, mv)
    }

    /// [`crate::small_ub`] in this copy: widens to 32 bytes on AVX2 when
    /// available, falling back to [`crate::small_ub_body`].
    pub(crate) unsafe fn small_ub<T: CodeUnit>(short: &[T], long: &[T]) -> Option<usize> {
        if std::is_x86_feature_detected!("avx2") {
            small_ub_avx2::<T>(short, long)
        } else {
            crate::small_ub_body::<T>(short, long)
        }
    }
}

/// AVX2 32-byte chunk scan for [`crate::small_ub`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn small_ub_avx2<T: CodeUnit>(short: &[T], long: &[T]) -> Option<usize> {
    use std::arch::x86_64::*;

    debug_assert!(short.len() <= long.len());
    let mut ub = long.len() - short.len();
    if ub > crate::MBLEVEN_MAX || short.len() > crate::MBLEVEN_MAX_LEN {
        return None;
    }
    let size = std::mem::size_of::<T>();
    let a = crate::as_bytes(short);
    let b = &crate::as_bytes(long)[..a.len()];
    let n = a.len();
    let mut i = 0;

    while i + 32 <= n {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        match size {
            1 => {
                let eq = _mm256_cmpeq_epi8(va, vb);
                let mask = _mm256_movemask_epi8(eq) as u32;
                let neq_mask = !mask;
                if neq_mask != 0 {
                    ub += neq_mask.count_ones() as usize;
                    if ub > crate::MBLEVEN_MAX {
                        return None;
                    }
                }
            }
            2 => {
                let eq = _mm256_cmpeq_epi16(va, vb);
                let mask = _mm256_movemask_epi8(eq) as u32;
                let neq_mask = !mask;
                if neq_mask != 0 {
                    ub += (neq_mask.count_ones() / 2) as usize;
                    if ub > crate::MBLEVEN_MAX {
                        return None;
                    }
                }
            }
            4 => {
                let eq = _mm256_cmpeq_epi32(va, vb);
                let mask = _mm256_movemask_epi8(eq) as u32;
                let neq_mask = !mask;
                if neq_mask != 0 {
                    ub += (neq_mask.count_ones() / 4) as usize;
                    if ub > crate::MBLEVEN_MAX {
                        return None;
                    }
                }
            }
            _ => unreachable!(),
        }
        i += 32;
    }

    while i + 16 <= n {
        let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        match size {
            1 => {
                let eq = _mm_cmpeq_epi8(va, vb);
                let mask = _mm_movemask_epi8(eq) as u32;
                let neq_mask = (!mask) & 0xFFFF;
                if neq_mask != 0 {
                    ub += neq_mask.count_ones() as usize;
                    if ub > crate::MBLEVEN_MAX {
                        return None;
                    }
                }
            }
            2 => {
                let eq = _mm_cmpeq_epi16(va, vb);
                let mask = _mm_movemask_epi8(eq) as u32;
                let neq_mask = (!mask) & 0xFFFF;
                if neq_mask != 0 {
                    ub += (neq_mask.count_ones() / 2) as usize;
                    if ub > crate::MBLEVEN_MAX {
                        return None;
                    }
                }
            }
            4 => {
                let eq = _mm_cmpeq_epi32(va, vb);
                let mask = _mm_movemask_epi8(eq) as u32;
                let neq_mask = (!mask) & 0xFFFF;
                if neq_mask != 0 {
                    ub += (neq_mask.count_ones() / 4) as usize;
                    if ub > crate::MBLEVEN_MAX {
                        return None;
                    }
                }
            }
            _ => unreachable!(),
        }
        i += 16;
    }

    let lane_bits = 8 * size;
    let lane_mask = if lane_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << lane_bits) - 1
    };

    while i + 8 <= n {
        let ca = a[i..i + 8].try_into().unwrap();
        let cb = b[i..i + 8].try_into().unwrap();
        let mut d = u64::from_le_bytes(ca) ^ u64::from_le_bytes(cb);
        while d != 0 {
            ub += 1;
            if ub > crate::MBLEVEN_MAX {
                return None;
            }
            let lane = d.trailing_zeros() as usize / lane_bits;
            d &= !(lane_mask << (lane * lane_bits));
        }
        i += 8;
    }
    for e in i / size..short.len() {
        if short[e].as_u64() != long[e].as_u64() {
            ub += 1;
            if ub > crate::MBLEVEN_MAX {
                return None;
            }
        }
    }
    Some(ub)
}
