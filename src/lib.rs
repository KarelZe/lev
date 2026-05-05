//! High-performance Python bindings for the Levenshtein distance.
//!
//! This crate exposes two functions to Python via PyO3:
//!
//! * [`distance`] – the Levenshtein edit distance between two strings.
//! * [`ratio`]    – a normalized similarity score in `[0.0, 1.0]`.
//!
//! # Algorithm
//!
//! See <https://en.wikipedia.org/wiki/Levenshtein_distance> for the distance
//! definition.  Several optimizations are layered to minimise the constant factor:
//!
//! 1. **Identity short-circuit** – equal strings return immediately.
//! 2. **Common-affix stripping** – shared leading and trailing code units are
//!    removed before the main computation.
//! 3. **Zero-copy CPython buffer access** – Python stores strings in one of
//!    three compact internal encodings (UCS-1 / UCS-2 / UCS-4); see
//!    [PEP 393](https://peps.python.org/pep-0393/).  Kind, ascii flag, data
//!    pointer, and length are read from the object header into a [`UniView`]
//!    without any copy; subsequent dispatch operates on those locals only.
//!    - *UCS-1* (`u8`, ≤ U+00FF): peq is a flat `[u64; 128]` (pure ASCII) or
//!      `[u64; 256]` (Latin-1) stack array — O(1) direct-index lookup.
//!    - *UCS-2* (`u16`) and *UCS-4* (`u32`): peq is a 128-slot stack-allocated
//!      open-addressing hash table with Fibonacci hashing — O(1) amortized
//!      lookup at ≤ 50 % load.  Mixed-kind pairs upcast both sides to `u32`;
//!      no UTF-8 round-trip, lone surrogates preserved.
//! 4. **Hyyrö's bit-parallel algorithm** – O(⌈m/w⌉ · n) time with w = 64;
//!    see H. Hyyrö, "A Bit-Vector Algorithm for Computing Levenshtein and
//!    Damerau Edit Distances", *Nordic Journal of Computing*, 2003.
//!    - *Single-word* (`m ≤ 64`): one 64-bit word covers the whole pattern.
//!    - *Multi-word* (`m > 64`): ⌈m/64⌉ words with carry propagation.
//!      UCS-1 uses a flat `[u64; 256 × W]` stack peq (O(1) lookup, no heap).
//!      UCS-2/4 use a heap-allocated peq data array addressed through the
//!      same open-addressing hash index.

use std::os::raw::c_uint;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

// ---------------------------------------------------------------------------
// Sealed trait unifying UCS-2 (u16) and UCS-4 (u32) as peq-key types.
// ---------------------------------------------------------------------------

/// Implemented by every code-unit type that can be used in the hash-based peq
/// table.  `SENTINEL` is used to initialize the keys array; occupancy is
/// tracked separately via the values array (where 0 indicates an empty slot).
trait CodeUnit: Ord + Copy + Eq + Send + 'static {
    const SENTINEL: Self;
    fn as_u64(self) -> u64;
}
impl CodeUnit for u8 {
    const SENTINEL: Self = u8::MAX;
    #[inline(always)]
    fn as_u64(self) -> u64 {
        self as u64
    }
}
impl CodeUnit for u16 {
    const SENTINEL: Self = u16::MAX;
    #[inline(always)]
    fn as_u64(self) -> u64 {
        self as u64
    }
}
impl CodeUnit for u32 {
    const SENTINEL: Self = u32::MAX;
    #[inline(always)]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

/// Fibonacci hash slot: maps a 64-bit key into `[0, mask]` where `mask = 2^k - 1`.
///
/// Shifts by `64 - k` to extract the top k bits of the product.  The high bits
/// of a multiplicative hash have the best avalanche properties — they mix
/// contributions from all input bits via carry propagation — so this gives
/// better distribution than extracting middle or low bits.  The result is
/// already in `[0, mask]`, so no masking step is needed.
///
/// See <https://en.wikipedia.org/wiki/Hash_function#Fibonacci_hashing>.
#[inline(always)]
fn hslot(key: u64, shift: u32) -> usize {
    (key.wrapping_mul(0x9e3779b9_7f4a7c15_u64) >> shift) as usize
}

// ---------------------------------------------------------------------------
// CPython internal-buffer accessors
// ---------------------------------------------------------------------------

/// Snapshot of a Python string's internal layout.  All four fields are
/// derived from a single PyASCIIObject header read; subsequent dispatch
/// branches operate on locals only — no further FFI calls.
struct UniView {
    kind: c_uint,
    ascii: bool,
    data: *const u8,
    len: usize,
}

/// Read kind + ascii flag + data ptr + length from a Python string in one go.
/// Subsequent dispatch matches on `(view1.kind, view2.kind)` without
/// re-traversing pyo3's PyUnicode helpers.
#[inline(always)]
unsafe fn view(s: &Bound<'_, PyString>) -> UniView {
    let ptr = s.as_ptr();
    let kind = ffi::PyUnicode_KIND(ptr);
    UniView {
        kind,
        // PyUnicode_IS_ASCII and the PyASCIIObject state bitfield are opaque
        // on Python 3.14+; fall back to the 256-entry Latin-1 table (correct
        // for all u8 values; small perf cost for pure-ASCII strings on 3.14).
        #[cfg(not(Py_3_14))]
        ascii: ffi::PyUnicode_IS_ASCII(ptr) != 0,
        #[cfg(Py_3_14)]
        ascii: false,
        data: ffi::PyUnicode_DATA(ptr) as *const u8,
        len: ffi::PyUnicode_GET_LENGTH(ptr) as usize,
    }
}

#[inline(always)]
unsafe fn as_u8(v: &UniView) -> &[u8] {
    std::slice::from_raw_parts(v.data, v.len)
}
#[inline(always)]
unsafe fn as_u16(v: &UniView) -> &[u16] {
    std::slice::from_raw_parts(v.data as *const u16, v.len)
}
#[inline(always)]
unsafe fn as_u32(v: &UniView) -> &[u32] {
    std::slice::from_raw_parts(v.data as *const u32, v.len)
}

/// Levenshtein edit distance between two strings.
///
/// The distance is the minimum number of single-character insertions,
/// deletions, or substitutions required to transform `s1` into `s2`.
///
/// Lengths are measured in Unicode scalar values (`char`s), so
/// multi-byte characters count as a single edit regardless of their
/// UTF-8 encoded length.
///
/// Args:
///     s1 (str): First input string.
///     s2 (str): Second input string.
///
/// Returns:
///     Non-negative integer edit distance.
///
/// Examples:
///     >>> import lev
///     >>> lev.distance("kitten", "sitting")
///     3
///     >>> lev.distance("flaw", "lawn")
///     2
///     >>> lev.distance("résumé", "resume")
///     2
#[pyfunction]
#[pyo3(signature = (s1, s2, /))]
fn distance(
    _py: Python<'_>,
    s1: &Bound<'_, PyString>,
    s2: &Bound<'_, PyString>,
) -> PyResult<usize> {
    if s1.is(s2) {
        return Ok(0);
    }
    unsafe {
        let v1 = view(s1);
        let v2 = view(s2);
        Ok(compute(&v1, &v2))
    }
}

/// Calculate normalized Levenshtein similarity ratio in `[0.0, 1.0]`.
///
/// Defined as `1 - distance(s1, s2) / (len(s1) + len(s2))`, where
/// lengths are measured in Unicode scalar values.
///
/// Two empty strings return `1.0` by convention.
///
/// Args:
///     s1 (str): First input string.
///     s2 (str): Second input string.
///
/// Returns:
///     Similarity score between `0.0` (completely different) and `1.0` (identical).
///
/// Examples:
///     >>> import lev
///     >>> lev.ratio("kitten", "sitting")
///     0.7692307692307693
///     >>> lev.ratio("", "")
///     1.0
#[pyfunction]
#[pyo3(signature = (s1, s2, /))]
fn ratio(_py: Python<'_>, s1: &Bound<'_, PyString>, s2: &Bound<'_, PyString>) -> PyResult<f64> {
    if s1.is(s2) {
        return Ok(1.0);
    }
    unsafe {
        let v1 = view(s1);
        let v2 = view(s2);
        let total = v1.len + v2.len;
        if total == 0 {
            return Ok(1.0);
        }
        Ok(1.0 - compute(&v1, &v2) as f64 / total as f64)
    }
}

/// A Python module implemented in Rust for the Levenshtein distance.
#[pymodule]
fn lev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Top-level dispatch: read both views' kinds, branch once, run the right path.
// ---------------------------------------------------------------------------

/// Dispatch on `(kind, ascii)` tuples and return the edit distance directly.
#[inline(always)]
unsafe fn compute(v1: &UniView, v2: &UniView) -> usize {
    use ffi::{PyUnicode_1BYTE_KIND as K1, PyUnicode_2BYTE_KIND as K2, PyUnicode_4BYTE_KIND as K4};

    // 1. Normalize: v1 is the shorter pattern string (m), v2 is the text (n).
    let (v1, v2) = if v1.len <= v2.len { (v1, v2) } else { (v2, v1) };

    // 2. Dispatch based on kinds.
    match (v1.kind, v2.kind) {
        (K1, K1) => {
            let (b1, b2) = (as_u8(v1), as_u8(v2));
            if v1.ascii && v2.ascii {
                compute_u8::<true>(b1, b2)
            } else {
                compute_u8::<false>(b1, b2)
            }
        }
        (K2, K2) => compute_sorted(as_u16(v1), as_u16(v2)),
        (K4, K4) => compute_sorted(as_u32(v1), as_u32(v2)),
        // Mixed kinds: Iterate natively without any temporary buffer allocation.
        (K1, K2) => compute_sorted_mixed(as_u8(v1), as_u16(v2)),
        (K1, K4) => compute_sorted_mixed(as_u8(v1), as_u32(v2)),
        (K2, K4) => compute_sorted_mixed(as_u16(v1), as_u32(v2)),
        _ => {
            // Normalization ensures v1.len <= v2.len, but not v1.kind <= v2.kind.
            match (v1.kind, v2.kind) {
                (K2, K1) => compute_sorted_mixed(as_u16(v1), as_u8(v2)),
                (K4, K1) => compute_sorted_mixed(as_u32(v1), as_u8(v2)),
                (K4, K2) => compute_sorted_mixed(as_u32(v1), as_u16(v2)),
                _ => unreachable!(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Affix stripping
// ---------------------------------------------------------------------------

/// Strip common prefix and suffix from two slices of different types.
#[inline(always)]
fn strip_affix_mixed<'a, T1: CodeUnit, T2: CodeUnit>(
    a: &'a [T1],
    b: &'a [T2],
) -> (&'a [T1], &'a [T2]) {
    let prefix = a
        .iter()
        .zip(b.iter())
        .position(|(x, y)| x.as_u64() != y.as_u64())
        .unwrap_or_else(|| a.len().min(b.len()));

    let a = &a[prefix..];
    let b = &b[prefix..];

    let suffix = a
        .iter()
        .rev()
        .zip(b.iter().rev())
        .position(|(x, y)| x.as_u64() != y.as_u64())
        .unwrap_or_else(|| a.len().min(b.len()));

    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}

/// Strip common prefix and suffix from two slices.
#[inline(always)]
fn strip_affix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> (&'a [T], &'a [T]) {
    let prefix = a
        .iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| a.len().min(b.len()));

    let a = &a[prefix..];
    let b = &b[prefix..];

    let suffix = a
        .iter()
        .rev()
        .zip(b.iter().rev())
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| a.len().min(b.len()));

    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}

// ---------------------------------------------------------------------------
// UCS-1 pipeline
// ---------------------------------------------------------------------------

/// Strip affixes and run Hyyrö.  `ASCII = true` enables the 128-entry peq fast path.
#[inline(always)]
fn compute_u8<const ASCII: bool>(a: &[u8], b: &[u8]) -> usize {
    let (a, b) = strip_affix(a, b);
    if a.is_empty() {
        return b.len();
    }
    if a.len() <= 64 {
        if ASCII {
            hyrro_64_u8::<128>(a, b)
        } else {
            hyrro_64_u8::<256>(a, b)
        }
    } else {
        hyrro_multiword_bytes(a, b)
    }
}

// ---------------------------------------------------------------------------
// Generic pipelines
// ---------------------------------------------------------------------------

/// Strip affixes and run Hyyrö for same-kind strings (non-UCS-1).
#[inline(always)]
fn compute_sorted<T: CodeUnit>(a: &[T], b: &[T]) -> usize {
    let (a, b) = strip_affix(a, b);
    if a.is_empty() {
        return b.len();
    }
    if a.len() <= 64 {
        hyrro_64_sorted(a, b)
    } else {
        hyrro_multiword_sorted(a, b)
    }
}

/// Strip affixes and run Hyyrö for mixed-kind strings.
#[inline(always)]
fn compute_sorted_mixed<T1: CodeUnit, T2: CodeUnit>(a: &[T1], b: &[T2]) -> usize {
    let (a, b) = strip_affix_mixed(a, b);
    if a.is_empty() {
        return b.len();
    }
    if a.len() <= 64 {
        hyrro_64_mixed(a, b)
    } else {
        hyrro_multiword_mixed(a, b)
    }
}

// ---------------------------------------------------------------------------
// Hyyrö single-word variants
// ---------------------------------------------------------------------------

/// Hyyrö single-word variant for byte patterns (UCS-1).
#[inline(always)]
fn hyrro_64_u8<const SLOTS: usize>(pattern: &[u8], text: &[u8]) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let mut peq = [0u64; SLOTS];
    for (i, &c) in pattern.iter().enumerate() {
        // SAFETY: caller guarantees c < SLOTS (128 or 256).
        unsafe {
            *peq.get_unchecked_mut(c as usize) |= 1u64 << i;
        }
    }
    hyrro_inner(
        pattern.len(),
        text.iter()
            .map(|&c| unsafe { *peq.get_unchecked(c as usize) }),
    )
}

/// Hyyrö single-word variant with a stack-allocated hash table.
#[inline(always)]
fn hyrro_64_sorted<T: CodeUnit>(pattern: &[T], text: &[T]) -> usize {
    hyrro_64_generic(pattern, text.iter().map(|&c| c.as_u64()))
}

/// Hyyrö single-word variant for mixed-kind.
#[inline(always)]
fn hyrro_64_mixed<T1: CodeUnit, T2: CodeUnit>(pattern: &[T1], text: &[T2]) -> usize {
    hyrro_64_generic(pattern, text.iter().map(|&c| c.as_u64()))
}

/// Core single-word builder and runner for non-UCS-1 strings.
#[inline(always)]
fn hyrro_64_generic<K: CodeUnit, I: Iterator<Item = u64>>(pattern: &[K], text_iter: I) -> usize {
    let m = pattern.len();
    debug_assert!((1..=64).contains(&m));

    const SLOTS: usize = 128;
    const MASK: usize = SLOTS - 1;
    let mut keys = [K::SENTINEL; SLOTS];
    let mut vals = [0u64; SLOTS];

    let shift = 64 - MASK.count_ones();
    for (i, &c) in pattern.iter().enumerate() {
        let mut slot = hslot(c.as_u64(), shift);
        loop {
            if vals[slot] == 0 {
                keys[slot] = c;
                vals[slot] = 1u64 << i;
                break;
            }
            if keys[slot] == c {
                vals[slot] |= 1u64 << i;
                break;
            }
            slot = (slot + 1) & MASK;
        }
    }

    hyrro_inner(
        m,
        text_iter.map(|c| {
            let mut slot = hslot(c, shift);
            loop {
                let v = unsafe { *vals.get_unchecked(slot) };
                if v == 0 {
                    return 0;
                }
                if unsafe { keys.get_unchecked(slot).as_u64() } == c {
                    return v;
                }
                slot = (slot + 1) & MASK;
            }
        }),
    )
}

// ---------------------------------------------------------------------------
// Hyyrö's bit-parallel inner loop
// ---------------------------------------------------------------------------

/// Core Hyyrö loop. Specialized for m=64 to avoid masking.
#[inline(always)]
fn hyrro_inner<I: Iterator<Item = u64>>(m: usize, pm_iter: I) -> usize {
    if m == 64 {
        hyrro_inner_64(pm_iter)
    } else {
        hyrro_inner_masked(m, pm_iter)
    }
}

#[inline(always)]
fn hyrro_inner_64<I: Iterator<Item = u64>>(pm_iter: I) -> usize {
    let mut vp = !0u64;
    let mut vn = 0u64;
    let mut score = 64isize;

    for pm in pm_iter {
        let x = pm | vn;
        let (sum, _) = (x & vp).overflowing_add(vp);
        let d0 = (sum ^ vp) | x;
        let hp = vn | !(d0 | vp);
        let hn = vp & d0;

        score += (hp >> 63) as isize;
        score -= (hn >> 63) as isize;

        let hp_s = (hp << 1) | 1;
        let hn_s = hn << 1;
        vp = hn_s | !(d0 | hp_s);
        vn = hp_s & d0;
    }
    score as usize
}

#[inline(always)]
fn hyrro_inner_masked<I: Iterator<Item = u64>>(m: usize, pm_iter: I) -> usize {
    let mask = (1u64 << m) - 1;
    let mut vp = mask;
    let mut vn = 0u64;
    let mut score = m as isize;
    let msb = 1u64 << (m - 1);

    for pm in pm_iter {
        let x = pm | vn;
        let (sum, _) = (x & vp).overflowing_add(vp);
        let d0 = (sum ^ vp) | x;
        let hp = vn | !(d0 | vp);
        let hn = vp & d0;

        score += ((hp & msb) != 0) as isize;
        score -= ((hn & msb) != 0) as isize;

        let hp_s = (hp << 1) | 1;
        let hn_s = hn << 1;
        vp = (hn_s | !(d0 | hp_s)) & mask;
        vn = (hp_s & d0) & mask;
    }
    score as usize
}

// ---------------------------------------------------------------------------
// Multi-word Hyyrö (m > 64)
// ---------------------------------------------------------------------------

/// Multi-word Hyyrö for mixed kinds.
#[inline(always)]
fn hyrro_multiword_mixed<T1: CodeUnit, T2: CodeUnit>(pattern: &[T1], text: &[T2]) -> usize {
    hyrro_multiword_sorted_generic(pattern, text)
}

/// Const-generic inner kernel. Unrolled for speed.
#[inline(always)]
fn multiword_kernel<const W: usize, I: Iterator<Item = [u64; W]>>(m: usize, pm_iter: I) -> usize {
    let last_bits = m - (W - 1) * 64; // 1..=64
    let msb_mask = 1u64 << (last_bits - 1);

    let mut vp = [!0u64; W];
    let mut vn = [0u64; W];
    if last_bits < 64 {
        vp[W - 1] = (1u64 << last_bits) - 1;
    }
    let mut score = m as isize;

    for pm_row in pm_iter {
        let mut carry = false;
        let mut prev_hp = 1u64;
        let mut prev_hn = 0u64;
        for k in 0..W {
            let pm = pm_row[k];
            let x = pm | vn[k];
            let (sum, nc) = (x & vp[k]).carrying_add(vp[k], carry);
            carry = nc;
            let d0 = (sum ^ vp[k]) | x;
            let hp = vn[k] | !(d0 | vp[k]);
            let hn = vp[k] & d0;
            if k == W - 1 {
                score += ((hp & msb_mask) != 0) as isize;
                score -= ((hn & msb_mask) != 0) as isize;
            }
            let (hp_msb, hn_msb) = (hp >> 63, hn >> 63);
            vp[k] = (hn << 1 | prev_hn) | !(d0 | (hp << 1 | prev_hp));
            vn[k] = (hp << 1 | prev_hp) & d0;
            (prev_hp, prev_hn) = (hp_msb, hn_msb);
        }
    }
    score as usize
}

/// Multi-word Hyyrö for UCS-1 slices.
fn hyrro_multiword_bytes(short: &[u8], long: &[u8]) -> usize {
    debug_assert!(short.len() > 64);
    let m = short.len();
    let w = (m + 63) / 64;

    macro_rules! run {
        ($W:literal) => {{
            let mut peq = [0u64; 256 * $W];
            for (i, &c) in short.iter().enumerate() {
                unsafe {
                    *peq.get_unchecked_mut(c as usize * $W + i / 64) |= 1u64 << (i % 64);
                }
            }
            multiword_kernel::<$W, _>(
                m,
                long.iter().map(|&c| {
                    let base = c as usize * $W;
                    let mut row = [0u64; $W];
                    for k in 0..$W {
                        row[k] = unsafe { *peq.get_unchecked(base + k) };
                    }
                    row
                }),
            )
        }};
    }
    match w {
        2 => run!(2),
        3 => run!(3),
        4 => run!(4),
        5 => run!(5),
        6 => run!(6),
        7 => run!(7),
        8 => run!(8),
        _ => hyrro_multiword_fallback(short, long, w, m),
    }
}

/// Heap-allocated fallback for very long patterns.
fn hyrro_multiword_fallback(short: &[u8], long: &[u8], w: usize, m: usize) -> usize {
    let last_bits = m - (w - 1) * 64;
    let top_mask = 1u64 << (last_bits - 1);
    let mut peq = vec![0u64; 256 * w];
    for (i, &c) in short.iter().enumerate() {
        peq[c as usize * w + i / 64] |= 1u64 << (i % 64);
    }
    let mut vp = vec![!0u64; w];
    let mut vn = vec![0u64; w];
    if last_bits < 64 {
        vp[w - 1] = (1u64 << last_bits) - 1;
    }
    let mut score = m as isize;
    for &tj in long {
        let base = tj as usize * w;
        let mut carry = false;
        let mut prev_hp = 1u64;
        let mut prev_hn = 0u64;
        for k in 0..w {
            let pm = peq[base + k];
            let x = pm | vn[k];
            let (sum, nc) = (x & vp[k]).carrying_add(vp[k], carry);
            carry = nc;
            let d0 = (sum ^ vp[k]) | x;
            let hp = vn[k] | !(d0 | vp[k]);
            let hn = vp[k] & d0;
            if k == w - 1 {
                score += ((hp & top_mask) != 0) as isize - ((hn & top_mask) != 0) as isize;
            }
            let (hp_msb, hn_msb) = (hp >> 63, hn >> 63);
            vp[k] = (hn << 1 | prev_hn) | !(d0 | (hp << 1 | prev_hp));
            vn[k] = (hp << 1 | prev_hp) & d0;
            (prev_hp, prev_hn) = (hp_msb, hn_msb);
        }
    }
    score as usize
}

/// Multi-word Hyyrö for non-UCS-1 strings.
fn hyrro_multiword_sorted<T: CodeUnit>(short: &[T], long: &[T]) -> usize {
    hyrro_multiword_sorted_generic(short, long)
}

fn hyrro_multiword_sorted_generic<T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2]) -> usize {
    debug_assert!(short.len() > 64);
    let m = short.len();
    let w = (m + 63) / 64;

    let mut keys: Vec<u64> = short.iter().map(|&c| c.as_u64()).collect();
    keys.sort_unstable();
    keys.dedup();
    let n_keys = keys.len();
    let mut data = vec![0u64; (n_keys + 1) * w];
    for (i, &c) in short.iter().enumerate() {
        let key = c.as_u64();
        let ki = keys.partition_point(|&x| x < key);
        data[ki * w + i / 64] |= 1u64 << (i % 64);
    }

    let hash_size = (n_keys * 2 + 2).next_power_of_two();
    let hash_mask = hash_size - 1;
    let mut hash_idx: Vec<u32> = vec![u32::MAX; hash_size];
    let hshift = 64 - hash_mask.count_ones();
    for (ki, &key) in keys.iter().enumerate() {
        let mut slot = hslot(key, hshift);
        loop {
            if hash_idx[slot] == u32::MAX {
                hash_idx[slot] = ki as u32;
                break;
            }
            slot = (slot + 1) & hash_mask;
        }
    }

    macro_rules! run {
        ($W:literal) => {{
            multiword_kernel::<$W, _>(
                m,
                long.iter().map(|&c| {
                    let key = c.as_u64();
                    let mut slot = hslot(key, hshift);
                    let base = loop {
                        let ki = unsafe { *hash_idx.get_unchecked(slot) };
                        if ki == u32::MAX {
                            break n_keys * $W;
                        }
                        if unsafe { *keys.get_unchecked(ki as usize) } == key {
                            break ki as usize * $W;
                        }
                        slot = (slot + 1) & hash_mask;
                    };
                    let mut row = [0u64; $W];
                    for k in 0..$W {
                        row[k] = unsafe { *data.get_unchecked(base + k) };
                    }
                    row
                }),
            )
        }};
    }
    match w {
        2 => run!(2),
        3 => run!(3),
        4 => run!(4),
        5 => run!(5),
        6 => run!(6),
        7 => run!(7),
        8 => run!(8),
        _ => {
            let last_bits = m - (w - 1) * 64;
            let top_mask = 1u64 << (last_bits - 1);
            let zero_base = n_keys * w;
            let mut vp = vec![!0u64; w];
            let mut vn = vec![0u64; w];
            if last_bits < 64 {
                vp[w - 1] = (1u64 << last_bits) - 1;
            }
            let mut score = m as isize;
            for &tj in long {
                let key = tj.as_u64();
                let mut slot = hslot(key, hshift);
                let base = loop {
                    let ki = unsafe { *hash_idx.get_unchecked(slot) };
                    if ki == u32::MAX {
                        break zero_base;
                    }
                    if unsafe { *keys.get_unchecked(ki as usize) } == key {
                        break ki as usize * w;
                    }
                    slot = (slot + 1) & hash_mask;
                };
                let mut carry = false;
                let mut prev_hp = 1u64;
                let mut prev_hn = 0u64;
                for k in 0..w {
                    let pm = data[base + k];
                    let x = pm | vn[k];
                    let (sum, nc) = (x & vp[k]).carrying_add(vp[k], carry);
                    carry = nc;
                    let d0 = (sum ^ vp[k]) | x;
                    let hp = vn[k] | !(d0 | vp[k]);
                    let hn = vp[k] & d0;
                    if k == w - 1 {
                        score += ((hp & top_mask) != 0) as isize - ((hn & top_mask) != 0) as isize;
                    }
                    let (hp_msb, hn_msb) = (hp >> 63, hn >> 63);
                    vp[k] = (hn << 1 | prev_hn) | !(d0 | (hp << 1 | prev_hp));
                    vn[k] = (hp << 1 | prev_hp) & d0;
                    (prev_hp, prev_hn) = (hp_msb, hn_msb);
                }
            }
            score as usize
        }
    }
}

// ---------------------------------------------------------------------------
// Test helpers (bypass the PyO3 layer)
// ---------------------------------------------------------------------------

#[cfg(test)]
fn levenshtein(s1: &str, s2: &str) -> usize {
    // Normalize: s1 is shorter pattern, s2 is longer text.
    let (s1, s2) = if s1.chars().count() <= s2.chars().count() {
        (s1, s2)
    } else {
        (s2, s1)
    };
    if s1.is_ascii() && s2.is_ascii() {
        compute_u8::<true>(s1.as_bytes(), s2.as_bytes())
    } else {
        let a: Vec<u32> = s1.chars().map(|c| c as u32).collect();
        let b: Vec<u32> = s2.chars().map(|c| c as u32).collect();
        compute_sorted(&a, &b)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference O(m·n) DP — small, obviously correct, used as oracle.
    fn naive(a: &[char], b: &[char]) -> usize {
        let (m, n) = (a.len(), b.len());
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }
        for i in 1..=m {
            for j in 1..=n {
                let cost = (a[i - 1] != b[j - 1]) as usize;
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[m][n]
    }

    fn check(a: &str, b: &str, expected: usize) {
        assert_eq!(levenshtein(a, b), expected, "({a:?}, {b:?})");
        assert_eq!(levenshtein(b, a), expected, "({b:?}, {a:?}) symmetry");
    }

    #[test]
    fn empty_strings() {
        check("", "", 0);
        check("abc", "", 3);
        check("", "abc", 3);
    }

    #[test]
    fn equal_strings() {
        check("hello", "hello", 0);
        check("a", "a", 0);
        check("日本語", "日本語", 0);
    }

    #[test]
    fn classic_pairs() {
        check("kitten", "sitting", 3);
        check("saturday", "sunday", 3);
        check("flaw", "lawn", 2);
        check("gumbo", "gambol", 2);
        check("intention", "execution", 5);
    }

    #[test]
    fn single_edits() {
        check("a", "b", 1);
        check("a", "ab", 1);
        check("ab", "a", 1);
        check("abc", "abxc", 1);
        check("abc", "axc", 1);
    }

    #[test]
    fn fully_disjoint() {
        check("aaaa", "bbbb", 4);
        check("abcde", "fghij", 5);
    }

    #[test]
    fn unicode_inputs() {
        check("résumé", "resume", 2);
        check("café", "cafe", 1);
        check("日本語", "日本", 1);
        check("🦀🐍", "🐍🦀", 2);
        check("naïve", "naive", 1);
    }

    #[test]
    fn boundary_64() {
        let a64: String = "a".repeat(64);
        let a65: String = "a".repeat(65);
        check(&a64, &a64, 0);
        check(&a64, &a65, 1);
        let mut shifted = String::from("b");
        shifted.push_str(&"a".repeat(63));
        check(&a64, &shifted, 1);
    }

    #[test]
    fn long_inputs_multiword() {
        let a: String = "abc".repeat(40); // 120 chars
        let mut b = a.clone();
        b.insert(0, 'x');
        check(&a, &b, 1);
        let c = format!("{a}xyz");
        check(&a, &c, 3);
    }

    /// Directly exercise `hyrro_multiword_bytes` (m > 64) against the naive DP.
    #[test]
    fn multiword_bytes_matches_oracle() {
        let oracle = |a: &[u8], b: &[u8]| -> usize {
            let ac: Vec<char> = a.iter().map(|&c| c as char).collect();
            let bc: Vec<char> = b.iter().map(|&c| c as char).collect();
            naive(&ac, &bc)
        };
        // ASCII bytes (< 128): verify correctness.
        let check_ascii = |a: &[u8], b: &[u8]| {
            let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            assert!(s.len() > 64 && s.iter().all(|&c| c < 128) && l.iter().all(|&c| c < 128));
            assert_eq!(hyrro_multiword_bytes(s, l), oracle(s, l));
        };
        // Latin-1 bytes: any u8 value allowed.
        let check_latin1 = |a: &[u8], b: &[u8]| {
            let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            assert!(s.len() > 64);
            assert_eq!(hyrro_multiword_bytes(s, l), oracle(s, l));
        };

        // Benchmark-like: two long ASCII strings that diverge after a shared prefix.
        let s1 = b"Lets pretend Marshall Mathers never picked up a pen".repeat(8);
        let s2 = b"Lets pretend things woulda been no different".repeat(8);
        let (sh, lo) = if s1.len() <= s2.len() {
            (&s1[..], &s2[..])
        } else {
            (&s2[..], &s1[..])
        };
        assert_eq!(hyrro_multiword_bytes(sh, lo), oracle(sh, lo));

        // Fully disjoint long ASCII strings.
        check_ascii(&b"a".repeat(100), &b"b".repeat(100));

        // Insertions at different positions (cycling through lowercase ASCII).
        let base: Vec<u8> = (0u8..80).map(|i| b'a' + i % 26).collect();
        let mut ins_front = vec![b'z'];
        ins_front.extend_from_slice(&base);
        check_ascii(&base, &ins_front);

        let mut ins_mid = base[..40].to_vec();
        ins_mid.push(b'z');
        ins_mid.extend_from_slice(&base[40..]);
        check_ascii(&base, &ins_mid);

        // Boundary: exactly 65 chars.
        check_ascii(&b"x".repeat(65), &b"y".repeat(65));

        // Latin-1: bytes spanning full 0..=255 range, 2-word boundary (128 chars).
        let p128: Vec<u8> = (0u8..128).collect();
        let q128: Vec<u8> = (1u8..=128).collect(); // 128 is a Latin-1 byte
        check_latin1(&p128, &q128);

        // Latin-1 disjoint.
        check_latin1(&vec![200u8; 80], &vec![201u8; 80]);
    }

    /// Directly exercise `hyrro_multiword_sorted` (m > 64) against the naive DP.
    #[test]
    fn multiword_sorted_matches_oracle() {
        let check_u32 = |a: &[u32], b: &[u32]| {
            let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            assert!(
                s.len() > 64,
                "test case must be long enough to hit multiword"
            );
            let got = hyrro_multiword_sorted(s, l);
            let ac: Vec<char> = s
                .iter()
                .map(|&c| char::from_u32(c).unwrap_or('?'))
                .collect();
            let bc: Vec<char> = l
                .iter()
                .map(|&c| char::from_u32(c).unwrap_or('?'))
                .collect();
            let exp = naive(&ac, &bc);
            assert_eq!(got, exp, "u32 mismatch ({} vs {} chars)", s.len(), l.len());
        };

        // Emoji run: each emoji is one code point — build 80-char runs.
        let emoji_a: Vec<u32> = [0x1f980u32, 0x1f40d, 0x1f389, 0x1f38a, 0x1f388]
            .iter()
            .copied()
            .cycle()
            .take(80)
            .collect();
        let emoji_b: Vec<u32> = [0x1f40du32, 0x1f980, 0x1f389, 0x1f38a, 0x1f388]
            .iter()
            .copied()
            .cycle()
            .take(80)
            .collect();
        check_u32(&emoji_a, &emoji_b);

        // CJK run: 80 characters from the Japanese test string.
        let cjk_a: Vec<u32> = "日本語のテスト文字列"
            .chars()
            .map(|c| c as u32)
            .cycle()
            .take(80)
            .collect();
        let cjk_b: Vec<u32> = "日本語のテスツ文字列"
            .chars()
            .map(|c| c as u32)
            .cycle()
            .take(80)
            .collect();
        check_u32(&cjk_a, &cjk_b);

        // Exactly 2 words (128 elements).
        let a128: Vec<u32> = (0u32..128).collect();
        let b128: Vec<u32> = (1u32..=128).collect();
        check_u32(&a128, &b128);

        // Fully disjoint.
        let all_a: Vec<u32> = vec![1u32; 80];
        let all_b: Vec<u32> = vec![2u32; 80];
        check_u32(&all_a, &all_b);
    }

    #[test]
    fn affix_stripping_does_not_change_result() {
        check("xxx_kitten_yyy", "xxx_sitting_yyy", 3);
        check("prefix-foo", "prefix-bar", 3);
        check("foo-suffix", "bar-suffix", 3);
    }

    #[test]
    fn ucs2_pipeline_matches_oracle() {
        let cases: &[(&str, &str, usize)] = &[
            ("日本語", "日本", 1),
            ("日本語のテスト", "日本語のテスツ", 1),
            ("한국어", "한국", 1),
            ("中文", "中英文", 1),
        ];
        for &(a, b, expected) in cases {
            let au: Vec<u16> = a.encode_utf16().collect();
            let bu: Vec<u16> = b.encode_utf16().collect();
            assert_eq!(compute_sorted(&au, &bu), expected, "u16 ({a:?}, {b:?})");
            assert_eq!(levenshtein(a, b), expected, "u32 ({a:?}, {b:?})");
        }
    }

    #[test]
    fn ucs4_pipeline_matches_oracle() {
        let cases: &[(&str, &str, usize)] = &[
            ("🦀🐍", "🐍🦀", 2),
            ("🎉🎊🎈", "🎊🎈", 1),
            ("😀😁😂", "😀😂", 1),
        ];
        for &(a, b, expected) in cases {
            let au: Vec<u32> = a.chars().map(|c| c as u32).collect();
            let bu: Vec<u32> = b.chars().map(|c| c as u32).collect();
            assert_eq!(compute_sorted(&au, &bu), expected, "u32 ({a:?}, {b:?})");
            assert_eq!(levenshtein(a, b), expected, "u32 ({a:?}, {b:?})");
        }
    }

    #[test]
    fn oracle_random_cases() {
        let long_a = "a".repeat(70);
        let long_b = "a".repeat(65);
        let long_c = "a".repeat(80);
        let long_d = "b".repeat(80);
        let cases: &[(&str, &str)] = &[
            ("hello", "world"),
            ("kitten", "sitting"),
            ("intention", "execution"),
            ("abcdefghijklmnopqrstuvwxyz", "zyxwvutsrqponmlkjihgfedcba"),
            ("", ""),
            ("a", ""),
            ("", "a"),
            ("aaaa", "aaaa"),
            ("abababab", "babababa"),
            ("the quick brown fox", "the quik brwn fx"),
            ("résumé", "résume"),
            ("日本語のテスト", "日本語のテスツ"),
            (&long_a, &long_b),
            (&long_c, &long_d),
        ];
        for &(a, b) in cases {
            let ac: Vec<char> = a.chars().collect();
            let bc: Vec<char> = b.chars().collect();
            assert_eq!(levenshtein(a, b), naive(&ac, &bc), "({a:?}, {b:?})");
        }
    }

    #[test]
    fn bench_multiword_raw() {
        use std::time::Instant;
        // Simulate the benchmark strings after affix stripping.
        let s1_full = b"Lets pretend things woulda been no different".repeat(8);
        let s2_full = b"Lets pretend Marshall Mathers never picked up a pen".repeat(8);
        // strip 13-char common prefix "Lets pretend "
        let short = &s1_full[13..];
        let long = &s2_full[13..];
        assert!(short.len() <= long.len());
        let n = 10_000u32;
        let mut sink = 0usize;
        let t0 = Instant::now();
        for _ in 0..n {
            sink += hyrro_multiword_bytes(short, long);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        #[cfg(debug_assertions)]
        eprintln!("NOTE: debug build — release will be ~10-20x faster");
        eprintln!(
            "raw Rust hyrro_multiword_bytes: {:.3} μs/call  (sink={})",
            us, sink
        );
        assert!(sink > 0); // prevent DCE
    }

    #[test]
    fn ratio_basic() {
        let r = |a: &str, b: &str| -> f64 {
            let total = a.chars().count() + b.chars().count();
            if total == 0 {
                1.0
            } else {
                1.0 - levenshtein(a, b) as f64 / total as f64
            }
        };
        assert!((r("", "") - 1.0).abs() < 1e-12);
        assert!((r("abc", "abc") - 1.0).abs() < 1e-12);
        assert!((r("kitten", "sitting") - (1.0 - 3.0 / 13.0)).abs() < 1e-12);
        assert!((r("abc", "xyz") - 0.5).abs() < 1e-12);
    }
}
