//! Kernels that use the NEON vector unit, which every AArch64 core has.
//! Measured on Apple M2 only, so [`crate::kernels`] selects them on Apple
//! targets; the tests run them on every AArch64 target.

#[cfg(target_arch = "aarch64")]
use crate::Kernel;

/// [`crate::kernels::word_u8`] without a peq table: each text byte's
/// 64-bit match mask comes from vector compares against the pattern, held
/// in four registers, so nothing is zeroed or filled per call.
///
/// The pattern's 64 bytes (padded) are dealt out to four vectors, so lane
/// `j` of vector `k` holds byte `4j + k`.  Per text byte, four `cmeq` give
/// all-ones lanes where it matches.  Three bit inserts (`bit`) merge them
/// into one vector whose lane `j` holds the four flags of bytes `4j..4j + 4`
/// in both nibbles, and a narrowing shift (`shrn`) packs the sixteen lanes
/// into one word with bit `4j + k` set iff byte `4j + k` matches: ten vector
/// instructions per text byte, all off the loop-carried chain.
///
/// The inserts are inline assembly: LLVM knows that `cmeq` lanes are 0 or
/// all-ones and rewrites `vbslq_u8` or `vsriq_n_u8` into four `and` and
/// three `orr`, which made the LCS loop throughput-bound (8-22% slower than
/// the table at four text bytes per pattern byte on Apple M2).
///
/// Kept out of line, as [`crate::hyrro_64_tiny`]: inlined into the entry
/// points, it moved the hash loops of UCS-2 patterns and made `ratio` 9-12%
/// slower on them at up to 32 characters.
///
/// The pattern must have 17 to 64 bytes.  With `SLOTS == 128` both
/// strings are pure ASCII.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
#[inline(never)]
pub(crate) fn word_u8<K: Kernel, const SLOTS: usize>(pattern: &[u8], text: &[u8]) -> usize {
    use std::arch::aarch64::*;
    /// `IOTA[i] == i`: the lane indices of `tbx` from any offset up to 48.
    const IOTA: [u8; 64] = {
        let mut a = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = i as u8;
            i += 1;
        }
        a
    };
    let m = pattern.len();
    debug_assert!((17..=64).contains(&m));
    // Pure-ASCII pairs pad with a byte no text byte equals, so the padding
    // lanes never match; Latin-1 pairs mask them off.
    let (pad, valid) = if SLOTS <= 128 {
        (0xFF, u64::MAX)
    } else {
        (0x00, u64::MAX >> (64 - m))
    };
    // SAFETY: NEON is part of the AArch64 baseline.  Each pattern load
    // reads 16 bytes at `off <= m - 16`, each `IOTA` load at most at 47.
    let p = unsafe {
        let pad = vdupq_n_u8(pad);
        // Bytes `16k..16k + 16` of the pattern, padded past its end, from
        // the last 16 bytes when they run past it, moved into place by `tbx`.
        // Loading in registers, not through a padded buffer on the stack,
        // saves a `memcpy` call and a store-to-load stall.
        let chunk = |k: usize| {
            let off = (16 * k).min(m - 16);
            let bytes = vld1q_u8(pattern.as_ptr().add(off));
            let idx = vld1q_u8(IOTA.as_ptr().add(16 * k - off));
            vqtbx1q_u8(pad, bytes, idx)
        };
        let c0 = vld1q_u8(pattern.as_ptr());
        let (c1, c2, c3) = (chunk(1), chunk(2), chunk(3));
        // Deal the bytes out as `vld4q_u8` does: vector `k` gets bytes `4j + k`.
        let (e01, o01) = (vuzp1q_u8(c0, c1), vuzp2q_u8(c0, c1));
        let (e23, o23) = (vuzp1q_u8(c2, c3), vuzp2q_u8(c2, c3));
        (
            vuzp1q_u8(e01, e23),
            vuzp1q_u8(o01, o23),
            vuzp2q_u8(e01, e23),
            vuzp2q_u8(o01, o23),
        )
    };
    // Bit masks of the inserts: vector 1's flag into bits 1 and 5 of vector
    // 0's lanes, vector 3's into bits 3 and 7 of vector 2's, then bits 2, 3,
    // 6 and 7 of that into the first.
    // SAFETY: NEON is part of the AArch64 baseline.
    let (m1, m3, m23) = unsafe { (vdupq_n_u8(0x22), vdupq_n_u8(0x88), vdupq_n_u8(0xCC)) };
    K::word(
        m,
        text.len(),
        text.iter().map(|&c| {
            // SAFETY: register-only NEON instructions, part of the AArch64
            // baseline.
            let mask = unsafe {
                let c = vdupq_n_u8(c);
                let mut t = vceqq_u8(p.0, c);
                std::arch::asm!(
                    "bit {t:v}.16b, {e1:v}.16b, {m1:v}.16b",
                    "bit {u:v}.16b, {e3:v}.16b, {m3:v}.16b",
                    "bit {t:v}.16b, {u:v}.16b, {m23:v}.16b",
                    t = inout(vreg) t,
                    u = inout(vreg) vceqq_u8(p.2, c) => _,
                    e1 = in(vreg) vceqq_u8(p.1, c),
                    e3 = in(vreg) vceqq_u8(p.3, c),
                    m1 = in(vreg) m1,
                    m3 = in(vreg) m3,
                    m23 = in(vreg) m23,
                    options(pure, nomem, nostack, preserves_flags),
                );
                let packed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t));
                vget_lane_u64::<0>(vreinterpret_u64_u8(packed))
            };
            mask & valid
        }),
    )
}
