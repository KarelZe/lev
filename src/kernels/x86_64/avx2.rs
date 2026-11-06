//! Kernels for x86-64 CPUs with AVX2 (Intel since Haswell, AMD since
//! Excavator), selected by runtime feature detection in the x86-64-v2 copy
//! of the hot path ([`super::v2`]), as the AVX2 block scan of its
//! `small_ub` is.  The forms are compiled with the x86-64-v2 features too,
//! so the code around their vector work compiles as in that copy.

#[cfg(target_arch = "x86_64")]
use crate::Kernel;

/// [`crate::kernels::word_u8`] without a peq table, for patterns of up to
/// 32 bytes: the pattern sits in one 32-byte register, and each text byte's
/// 32-bit match mask is one `vpcmpeqb` against the byte broadcast to all
/// lanes and one `vpmovmskb`, off the loop-carried chain.  Nothing is
/// zeroed or filled per call.
///
/// The pattern must have 17 to 32 bytes.  With `SLOTS == 128` both strings
/// are pure ASCII.
///
/// # Safety
///
/// The CPU must support AVX2 and the x86-64-v2 features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt,sse3,ssse3,sse4.1,sse4.2")]
pub(crate) unsafe fn word_u8<K: Kernel, const SLOTS: usize>(pattern: &[u8], text: &[u8]) -> usize {
    use std::arch::x86_64::*;
    /// `pshufb` indices that move bytes `d..16` of a register down to
    /// `0..16 - d` and zero the rest (index bit 7 set), from offset `d`.
    const SHIFT: [u8; 32] = {
        let mut a = [0x80u8; 32];
        let mut i = 0;
        while i < 16 {
            a[i] = i as u8;
            i += 1;
        }
        a
    };
    let m = pattern.len();
    debug_assert!((17..=32).contains(&m));
    // Pure-ASCII pairs pad with 0x80, which no text byte equals, so the
    // padding lanes never match; Latin-1 pairs mask them off.
    let valid = if SLOTS <= 128 {
        u64::MAX
    } else {
        u64::MAX >> (64 - m)
    };
    // SAFETY: AVX2 and SSSE3 are enabled.  The pattern loads read 16 bytes
    // at 0 and at `m - 16`, the `SHIFT` load at `32 - m <= 15`.
    let pat = unsafe {
        let lo = _mm_loadu_si128(pattern.as_ptr().cast());
        // Bytes 16..32: the last 16 bytes, moved down by `32 - m` lanes,
        // with zeros past the end of the pattern.
        let tail = _mm_loadu_si128(pattern.as_ptr().add(m - 16).cast());
        let idx = _mm_loadu_si128(SHIFT.as_ptr().add(32 - m).cast());
        let mut hi = _mm_shuffle_epi8(tail, idx);
        if SLOTS <= 128 {
            hi = _mm_or_si128(hi, _mm_and_si128(idx, _mm_set1_epi8(0x80u8 as i8)));
        }
        _mm256_set_m128i(hi, lo)
    };
    K::word(
        m,
        text.len(),
        text.iter().map(|&c| {
            let eq = _mm256_cmpeq_epi8(pat, _mm256_set1_epi8(c as i8));
            // Through `u32`: `vpmovmskb` sets the sign of its `i32` result
            // when byte 31 matches, which must not reach bits 32-63.
            _mm256_movemask_epi8(eq) as u32 as u64 & valid
        }),
    )
}
