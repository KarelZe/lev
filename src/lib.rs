//! High-performance Python bindings for the Levenshtein distance.
//!
//! This crate exposes two functions to Python via PyO3:
//!
//! * [`distance`] – the Levenshtein edit distance between two strings.
//! * [`ratio`]    – a normalized similarity score in `[0.0, 1.0]`.
//!
//! # Algorithm
//!
//! Several well-known optimizations are combined to keep the constant
//! factor small:
//!
//! 1. **Identity short-circuit** – equal strings return immediately.
//! 2. **Common-affix stripping** – matching leading and trailing code
//!    units are removed before the main computation.
//! 3. **Zero-copy CPython buffer access** – Python stores strings in one
//!    of three packed internal encodings (UCS-1 / UCS-2 / UCS-4).  Kind,
//!    ascii flag, data pointer, and length are read into a single
//!    [`UniView`] per string — subsequent dispatch uses locals only, no
//!    further FFI.  For UCS-1 (`u8`, ≤ U+00FF) the single-word peq table
//!    is a flat `[u64; 128]` stack array when both strings are ASCII-flagged
//!    (half-size memset), or `[u64; 256]` otherwise; the multi-word peq is
//!    always `[u64; 256 × W]` on the stack (no heap allocation, all u8
//!    values are valid indices).  For UCS-2 (`u16`) and UCS-4 (`u32`) — and
//!    for mixed-kind pairs — a stack-allocated sorted `(T, u64)` peq array
//!    is used (O(log m) lookup, zero heap allocation).  Mixed-kind pairs
//!    upcast both buffers to `u32` via `to_u32_buf`; no UTF-8 round-trip,
//!    no `to_str()`, lone surrogates handled correctly.
//! 4. **Hyyrö's bit-parallel algorithm** (Hyyrö, 2003) – runs in
//!    `O(⌈m / w⌉ · n)` time with `w = 64`.  Two variants are used:
//!    - *Single-word* (`m ≤ 64`): one 64-bit word covers the whole pattern.
//!      UCS-1 uses a flat `[u64; 128/256]` peq table (O(1) lookup); UCS-2/4
//!      use a sorted `(T, u64)` peq array (O(log m) lookup).
//!    - *Multi-word* (`m > 64`): `⌈m/64⌉` words with carry propagation.
//!      UCS-1 keeps the flat peq table (O(1) lookup per text byte); UCS-2/4
//!      use a sorted flat peq layout with one binary search per text unit.
//!      The GIL is released before the multi-word u32 path so other Python
//!      threads can make progress during long computations.

use std::os::raw::c_uint;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

// ---------------------------------------------------------------------------
// Sealed trait unifying UCS-2 (u16) and UCS-4 (u32) as peq-key types.
// ---------------------------------------------------------------------------

/// Implemented by every code-unit type that can be used in the sorted peq
/// array.  `SENTINEL` fills unused array slots and is never searched.
trait CodeUnit: Ord + Copy + Eq + Send + 'static {
    const SENTINEL: Self;
}
impl CodeUnit for u16 { const SENTINEL: Self = u16::MAX; }
impl CodeUnit for u32 { const SENTINEL: Self = u32::MAX; }

// ---------------------------------------------------------------------------
// Deferred computation result — avoids passing `py` into pure helper fns.
// ---------------------------------------------------------------------------

/// Either a finished distance or owned slices ready for Wagner-Fischer.
/// The `Wf` variant is produced when the shorter input exceeds 64 code units;
/// the caller can then release the GIL before dispatching `wagner_fischer`.
enum Prep<T> {
    Done(usize),
    Wf(Vec<T>, Vec<T>),
}

// ---------------------------------------------------------------------------
// CPython internal-buffer accessors
// ---------------------------------------------------------------------------

/// Snapshot of a Python string's internal layout.  All four fields are
/// derived from a single PyASCIIObject header read; subsequent dispatch
/// branches operate on locals only — no further FFI calls.
struct UniView {
    kind:  c_uint,
    ascii: bool,
    data:  *const u8,
    len:   usize,
}

/// Read kind + ascii flag + data ptr + length from a Python string in one go.
/// Subsequent dispatch matches on `(view1.kind, view2.kind)` without
/// re-traversing pyo3's PyUnicode helpers.
#[inline(always)]
unsafe fn view(s: &Bound<'_, PyString>) -> UniView {
    let ptr  = s.as_ptr();
    let kind = ffi::PyUnicode_KIND(ptr);
    UniView {
        kind,
        ascii: ffi::PyUnicode_IS_ASCII(ptr) != 0,
        data:  ffi::PyUnicode_DATA(ptr) as *const u8,
        len:   ffi::PyUnicode_GET_LENGTH(ptr) as usize,
    }
}

#[inline(always)] unsafe fn as_u8 (v: &UniView) -> &[u8]  { std::slice::from_raw_parts(v.data,                v.len) }
#[inline(always)] unsafe fn as_u16(v: &UniView) -> &[u16] { std::slice::from_raw_parts(v.data as *const u16, v.len) }
#[inline(always)] unsafe fn as_u32(v: &UniView) -> &[u32] { std::slice::from_raw_parts(v.data as *const u32, v.len) }

/// Materialise any Python string as `Vec<u32>` for mixed-kind pairs.
/// Lone surrogates in UCS-2 are preserved (no UTF-8 round-trip).
unsafe fn to_u32_buf(v: &UniView) -> Vec<u32> {
    match v.kind {
        ffi::PyUnicode_1BYTE_KIND => (0..v.len).map(|i| *v.data.add(i) as u32).collect(),
        ffi::PyUnicode_2BYTE_KIND => {
            let p = v.data as *const u16;
            (0..v.len).map(|i| *p.add(i) as u32).collect()
        }
        _ => std::slice::from_raw_parts(v.data as *const u32, v.len).to_vec(),
    }
}

/// Levenshtein edit distance between two strings.
///
/// The distance is the minimum number of single-character insertions,
/// deletions, or substitutions required to transform `s1` into `s2`.
/// Lengths are measured in Unicode scalar values (`char`s), so
/// multi-byte characters count as a single edit regardless of their
/// UTF-8 encoded length.
///
/// # Examples
///
/// ```python
/// >>> import lev
/// >>> lev.distance("kitten", "sitting")
/// 3
/// >>> lev.distance("flaw", "lawn")
/// 2
/// >>> lev.distance("résumé", "resume")
/// 2
/// ```
#[pyfunction]
#[pyo3(signature = (s1, s2, /))]
fn distance(py: Python<'_>, s1: &Bound<'_, PyString>, s2: &Bound<'_, PyString>) -> PyResult<usize> {
    // All paths access CPython's packed buffer directly — no UTF-8 encode,
    // no char decode, no failure modes.  GIL released for Wagner-Fischer.
    unsafe {
        let v1 = view(s1);
        let v2 = view(s2);
        Ok(apply(py, prep_dispatch(&v1, &v2)))
    }
}

/// Normalized Levenshtein similarity ratio in `[0.0, 1.0]`.
///
/// Defined as `1 - distance(s1, s2) / (len(s1) + len(s2))`, where the
/// lengths are measured in Unicode scalar values. Two empty strings
/// return `1.0` by convention.
///
/// # Examples
///
/// ```python
/// >>> import lev
/// >>> lev.ratio("kitten", "sitting")
/// 0.7692307692307693
/// >>> lev.ratio("", "")
/// 1.0
/// ```
#[pyfunction]
#[pyo3(signature = (s1, s2, /))]
fn ratio(py: Python<'_>, s1: &Bound<'_, PyString>, s2: &Bound<'_, PyString>) -> PyResult<f64> {
    unsafe {
        let v1 = view(s1);
        let v2 = view(s2);
        let total = v1.len + v2.len;
        if total == 0 { return Ok(1.0); }
        Ok(1.0 - apply(py, prep_dispatch(&v1, &v2)) as f64 / total as f64)
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

/// Dispatch on `(kind, ascii)` tuples and select the cheapest pipeline.
/// Returns either a finished distance or owned slices ready for Wagner-Fischer.
///
/// Same-kind UCS-1 has a dedicated 128-entry peq fast-path when both strings
/// are ASCII (saves 1 KB of memset vs. the generic 256-entry path).  UCS-2 and
/// UCS-4 use the sorted-peq generic pipeline.  Mixed-kind pairs upcast both
/// buffers to `u32`.
#[inline(always)]
unsafe fn prep_dispatch(v1: &UniView, v2: &UniView) -> Prep<u32> {
    use ffi::{PyUnicode_1BYTE_KIND as K1, PyUnicode_2BYTE_KIND as K2, PyUnicode_4BYTE_KIND as K4};
    match (v1.kind, v2.kind) {
        (K1, K1) => {
            let b1 = as_u8(v1);
            let b2 = as_u8(v2);
            // Pure ASCII fast-path: half-size (1 KB) peq table.  Both flags
            // must be set since pattern bytes index into peq during fill and
            // text bytes index into it during the inner loop; either flank
            // having a byte >= 128 would overflow the 128-entry array.
            if v1.ascii && v2.ascii {
                box_u8(prep_bytes::<true>(b1, b2))
            } else {
                box_u8(prep_bytes::<false>(b1, b2))
            }
        }
        (K2, K2) => box_fixed(prep_fixed(as_u16(v1), as_u16(v2))),
        (K4, K4) => prep_fixed(as_u32(v1), as_u32(v2)),
        // Mixed-kind: upcast both to u32 and run the generic pipeline.
        // Consistent with same-kind paths; lone surrogates preserved.
        _ => {
            let a = to_u32_buf(v1);
            let b = to_u32_buf(v2);
            prep_fixed(&a, &b)
        }
    }
}

/// Convert `Prep<u8>` to `Prep<u32>` so `prep_dispatch` returns a single type.
/// `Done` requires no copy.  `Wf(u8)` widens lazily after the GIL is released
/// (cheap: this branch is only hit when `short > 64`, where O(m·n) dominates).
#[inline]
fn box_u8(p: Prep<u8>) -> Prep<u32> {
    match p {
        Prep::Done(d)  => Prep::Done(d),
        Prep::Wf(s, l) => Prep::Wf(
            s.into_iter().map(|b| b as u32).collect(),
            l.into_iter().map(|b| b as u32).collect(),
        ),
    }
}

/// Convert `Prep<u16>` to `Prep<u32>` (same rationale as `box_u8`).
#[inline]
fn box_fixed(p: Prep<u16>) -> Prep<u32> {
    match p {
        Prep::Done(d)  => Prep::Done(d),
        Prep::Wf(s, l) => Prep::Wf(
            s.into_iter().map(|c| c as u32).collect(),
            l.into_iter().map(|c| c as u32).collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Affix stripping
// ---------------------------------------------------------------------------

#[inline(always)]
fn strip_affix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> (&'a [T], &'a [T]) {
    let prefix = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    let (a, b) = (&a[prefix..], &b[prefix..]);
    let suffix = a.iter().rev().zip(b.iter().rev()).take_while(|(x, y)| x == y).count();
    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}

// ---------------------------------------------------------------------------
// UCS-1 pipeline  (dedicated O(1) [u64; N] peq table; N = 128 for ASCII)
// ---------------------------------------------------------------------------

/// Strip affixes and run Hyyrö, or package slices for deferred WF.
/// `ASCII = true` enables the 128-entry peq fast path.
#[inline(always)]
fn prep_bytes<const ASCII: bool>(a: &[u8], b: &[u8]) -> Prep<u8> {
    if a == b { return Prep::Done(0); }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() { return Prep::Done(b.len()); }
    if b.is_empty() { return Prep::Done(a.len()); }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        Prep::Done(if ASCII { hyrro_64_ascii(short, long) } else { hyrro_64_bytes(short, long) })
    } else {
        // Multi-word Hyyrö: stack-allocated peq, no heap allocation.
        // The GIL is held; u8 peq lookup is O(1) and fast enough.
        Prep::Done(hyrro_multiword_bytes(short, long))
    }
}

/// Hyyrö single-word variant for ASCII-only patterns.  Uses a 128-entry peq
/// (1 KB) instead of the 256-entry table needed by Latin-1 — halves the
/// stack-init cost.  Caller must ensure all bytes of `pattern` and `text`
/// are < 128 (e.g. via `PyUnicode_IS_ASCII`).
#[inline(always)]
fn hyrro_64_ascii(pattern: &[u8], text: &[u8]) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    debug_assert!(pattern.iter().all(|&b| b < 128));
    debug_assert!(text.iter().all(|&b| b < 128));
    let mut peq = [0u64; 128];
    let mut bit = 1u64;
    for &c in pattern {
        // SAFETY: caller guarantees c < 128.
        unsafe { *peq.get_unchecked_mut(c as usize) |= bit; }
        bit <<= 1;
    }
    // SAFETY: caller guarantees text bytes < 128.
    hyrro_inner(pattern.len(), text.len(), |j| unsafe {
        *peq.get_unchecked(*text.get_unchecked(j) as usize)
    })
}

/// Hyyrö single-word variant for the general byte case (Latin-1 etc.).
/// `pattern.len()` in `1..=64`.
#[inline(always)]
fn hyrro_64_bytes(pattern: &[u8], text: &[u8]) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let mut peq = [0u64; 256];
    let mut bit = 1u64;
    for &c in pattern {
        peq[c as usize] |= bit;
        bit <<= 1;
    }
    // SAFETY: indices come from text bytes; peq is 256 entries.  We use
    // get_unchecked for both lookups to suppress bounds checks the compiler
    // can already prove redundant.
    hyrro_inner(pattern.len(), text.len(), |j| unsafe {
        *peq.get_unchecked(*text.get_unchecked(j) as usize)
    })
}

// ---------------------------------------------------------------------------
// Generic pipeline for UCS-2 (u16), UCS-4 (u32), and mixed-kind (u32)
// ---------------------------------------------------------------------------

/// Strip affixes and run Hyyrö, or package slices for deferred WF.
#[inline(always)]
fn prep_fixed<T: CodeUnit>(a: &[T], b: &[T]) -> Prep<T> {
    if a == b { return Prep::Done(0); }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() { return Prep::Done(b.len()); }
    if b.is_empty() { return Prep::Done(a.len()); }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        Prep::Done(hyrro_64_sorted(short, long))
    } else {
        Prep::Wf(short.to_vec(), long.to_vec())
    }
}

/// Hyyrö single-word variant with a stack-allocated sorted `(T, u64)` peq
/// array.  No heap allocation; O(log m) peq lookup per text character.
/// `pattern.len()` must be in `1..=64`.
#[inline(always)]
fn hyrro_64_sorted<T: CodeUnit>(pattern: &[T], text: &[T]) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));

    let mut peq: [(T, u64); 64] = [(T::SENTINEL, 0); 64];
    let mut peq_len = 0usize;

    let mut bit = 1u64;
    for &c in pattern {
        match peq[..peq_len].binary_search_by_key(&c, |p| p.0) {
            Ok(idx)  => peq[idx].1 |= bit,
            Err(idx) => {
                peq.copy_within(idx..peq_len, idx + 1);
                peq[idx] = (c, bit);
                peq_len += 1;
            }
        }
        bit <<= 1;
    }
    let peq = &peq[..peq_len];

    // SAFETY: index `j` is bounded by `text.len()`, the iteration count
    // passed to hyrro_inner.
    hyrro_inner(pattern.len(), text.len(), |j| {
        let c = unsafe { *text.get_unchecked(j) };
        match peq.binary_search_by_key(&c, |p| p.0) {
            Ok(idx) => peq[idx].1,
            Err(_)  => 0,
        }
    })
}

// ---------------------------------------------------------------------------
// GIL-aware dispatch
// ---------------------------------------------------------------------------

/// Finalise a [`Prep`] result.  `Done` returns immediately.  `Wf` releases
/// the GIL before running Wagner-Fischer so other threads can make progress
/// during long O(m·n) computations.
#[inline]
fn apply(py: Python<'_>, prep: Prep<u32>) -> usize {
    match prep {
        Prep::Done(d)  => d,
        Prep::Wf(s, l) => py.detach(move || hyrro_multiword(&s, &l)),
    }
}

// ---------------------------------------------------------------------------
// Hyyrö's bit-parallel inner loop (single 64-bit word)
// ---------------------------------------------------------------------------

/// Core Hyyrö loop shared by all pipelines.  `pm_of(j)` returns the
/// pattern-match bitmask for text element at index `j`.  An indexed closure
/// (rather than an iterator) keeps the loop trip-count statically known to
/// the optimizer, which is important on tight bit-parallel code where any
/// hidden Iterator::next call would balloon the schedule.
///
/// Reference: H. Hyyrö, *A bit-vector algorithm for computing
/// Levenshtein and Damerau edit distances*, Nordic Journal of Computing,
/// 2003.
#[inline(always)]
fn hyrro_inner<F: FnMut(usize) -> u64>(m: usize, n: usize, mut pm_of: F) -> usize {
    let mut vp    = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    let mut vn    = 0u64;
    // Top-bit mask of the score cell (hoisted: same value every iter).
    let top       = 1u64 << (m - 1);
    let mut score = m as isize;

    for j in 0..n {
        let pm = pm_of(j);
        let x  = pm | vn;
        let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x;
        let hp = vn | !(d0 | vp);
        let hn = vp & d0;

        // Branchless top-bit extract: AND with mask, then test-nonzero.
        // Compiles to `tst; cset` on AArch64 — two ops, no branches.
        score += (hp & top != 0) as isize - (hn & top != 0) as isize;

        let hp_s = (hp << 1) | 1;
        let hn_s = hn << 1;
        vp = hn_s | !(d0 | hp_s);
        vn = hp_s & d0;
    }

    score as usize
}

// ---------------------------------------------------------------------------
// Multi-word Hyyrö (m > 64): replaces the WF fallback for long inputs.
// ---------------------------------------------------------------------------

/// Add two u64 values with a carry-in bit; return (sum, carry-out).
#[inline(always)]
fn carrying_add(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (s1, c1) = a.overflowing_add(b);
    let (s2, c2) = s1.overflowing_add(carry as u64);
    (s2, c1 | c2)
}

/// Const-generic inner kernel shared by all multi-word paths.
///
/// With `W` known at compile time the compiler can:
///  * allocate `vp` and `vn` as `[u64; W]` on the stack (no heap access
///    in the hot loop);
///  * fully unroll `for k in 0..W`;
///  * constant-fold `k == W - 1` so the score update emits a single
///    branchless instruction only in the last unrolled iteration;
///  * keep the entire state in registers across outer iterations.
///
/// `pm_of(j)` is inlined by the monomorphic dispatch — it should not
/// contain any heap allocation.
#[inline(always)]
fn multiword_kernel<const W: usize, F: Fn(usize) -> [u64; W]>(
    m: usize,
    n: usize,
    pm_of: F,
) -> usize {
    let last_bits = m - (W - 1) * 64; // 1..=64
    let top_mask = 1u64 << (last_bits - 1);

    let mut vp = [!0u64; W];
    let mut vn = [0u64; W];
    if last_bits < 64 {
        vp[W - 1] = (1u64 << last_bits) - 1;
    }
    let mut score = m as isize;

    for j in 0..n {
        let pm_row = pm_of(j);
        let mut carry = false;
        let mut prev_hp = 1u64;
        let mut prev_hn = 0u64;
        for k in 0..W {
            let pm = pm_row[k];
            let x = pm | vn[k];
            let (sum, nc) = carrying_add(x & vp[k], vp[k], carry);
            carry = nc;
            let d0 = (sum ^ vp[k]) | x;
            let hp = vn[k] | !(d0 | vp[k]);
            let hn = vp[k] & d0;
            if k == W - 1 {
                score += (hp & top_mask != 0) as isize - (hn & top_mask != 0) as isize;
            }
            let (hp_msb, hn_msb) = (hp >> 63, hn >> 63);
            vp[k] = (hn << 1 | prev_hn) | !(d0 | (hp << 1 | prev_hp));
            vn[k] = (hp << 1 | prev_hp) & d0;
            (prev_hp, prev_hn) = (hp_msb, hn_msb);
        }
    }
    score as usize
}

/// Multi-word Hyyrö for UCS-1 slices.  Each `run!(W)` arm stack-allocates a
/// flat `[u64; 256 × W]` peq table (O(1) lookup, zero heap allocation) and
/// dispatches to `multiword_kernel<W>` for `W = 2..=8`.  With W known at
/// compile time the compiler stack-allocates `vp`/`vn` and fully unrolls the
/// inner loop.  ALPHA=256 covers all u8 values without a bounds check.
fn hyrro_multiword_bytes(short: &[u8], long: &[u8]) -> usize {
    debug_assert!(short.len() > 64);
    let m = short.len();
    let w = (m + 63) / 64;

    macro_rules! run {
        ($W:literal) => {{
            // Stack-allocated peq: no malloc/free per call.
            // [u64; 256 * W] = W KB for W=1..8 (max 16 KB at W=8).
            let mut peq = [0u64; 256 * $W];
            for (i, &c) in short.iter().enumerate() {
                // SAFETY: c as usize < 256 (u8), i/64 < W (i < m ≤ 64*W).
                unsafe { *peq.get_unchecked_mut(c as usize * $W + i / 64) |= 1u64 << (i % 64); }
            }
            multiword_kernel::<$W, _>(m, long.len(), |j| {
                // SAFETY: j < long.len(), c < 256, base+k < 256*W.
                let base = unsafe { *long.get_unchecked(j) } as usize * $W;
                let mut row = [0u64; $W];
                for k in 0..$W {
                    row[k] = unsafe { *peq.get_unchecked(base + k) };
                }
                row
            })
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
        // Heap-allocated fallback for patterns longer than 512 code units.
        _ => {
            let last_bits = m - (w - 1) * 64;
            let top_mask = 1u64 << (last_bits - 1);
            let mut peq = vec![0u64; 256 * w];
            for (i, &c) in short.iter().enumerate() {
                unsafe { *peq.get_unchecked_mut(c as usize * w + i / 64) |= 1u64 << (i % 64); }
            }
            let mut vp = vec![!0u64; w];
            let mut vn = vec![0u64; w];
            if last_bits < 64 { vp[w - 1] = (1u64 << last_bits) - 1; }
            let mut score = m as isize;
            for &tj in long {
                let base = tj as usize * w;
                let mut carry = false;
                let mut prev_hp = 1u64;
                let mut prev_hn = 0u64;
                for k in 0..w {
                    let pm = unsafe { *peq.get_unchecked(base + k) };
                    let x = pm | unsafe { *vn.get_unchecked(k) };
                    let vp_k = unsafe { *vp.get_unchecked(k) };
                    let (sum, nc) = carrying_add(x & vp_k, vp_k, carry);
                    carry = nc;
                    let d0 = (sum ^ vp_k) | x;
                    let vn_k = unsafe { *vn.get_unchecked(k) };
                    let hp = vn_k | !(d0 | vp_k);
                    let hn = vp_k & d0;
                    if k == w - 1 { score += (hp & top_mask != 0) as isize - (hn & top_mask != 0) as isize; }
                    let (hp_msb, hn_msb) = (hp >> 63, hn >> 63);
                    unsafe {
                        *vp.get_unchecked_mut(k) = (hn << 1 | prev_hn) | !(d0 | (hp << 1 | prev_hp));
                        *vn.get_unchecked_mut(k) = (hp << 1 | prev_hp) & d0;
                    }
                    (prev_hp, prev_hn) = (hp_msb, hn_msb);
                }
            }
            score as usize
        }
    }
}

/// Multi-word Hyyrö for u32 slices (UCS-4 / mixed-kind upcast paths).
/// Peq is a sorted `keys / data` layout; each text lookup is one
/// `binary_search` (O(log k)).  Same const-generic dispatch as the u8 path.
fn hyrro_multiword(short: &[u32], long: &[u32]) -> usize {
    debug_assert!(short.len() > 64);
    let m = short.len();
    let w = (m + 63) / 64;

    let mut keys: Vec<u32> = short.iter().copied().collect();
    keys.sort_unstable();
    keys.dedup();
    let n_keys = keys.len();
    // One extra zero row at the end: "not found" → index n_keys → zeros.
    let mut data = vec![0u64; (n_keys + 1) * w];
    for (i, &c) in short.iter().enumerate() {
        let ki = keys.partition_point(|&x| x < c);
        data[ki * w + i / 64] |= 1u64 << (i % 64);
    }

    macro_rules! run {
        ($W:literal) => {{
            let k_ref = &keys;
            let d_ref = &data;
            let nk = n_keys;
            multiword_kernel::<$W, _>(m, long.len(), |j| {
                let c = unsafe { *long.get_unchecked(j) };
                let base = match k_ref.binary_search(&c) {
                    Ok(ki) => ki * $W,
                    Err(_)  => nk * $W, // zero row
                };
                let mut row = [0u64; $W];
                for k in 0..$W {
                    row[k] = unsafe { *d_ref.get_unchecked(base + k) };
                }
                row
            })
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
            // Heap-allocated fallback for very long patterns (w > 8).
            let last_bits = m - (w - 1) * 64;
            let top_mask = 1u64 << (last_bits - 1);
            let zero_base = n_keys * w;
            let mut vp = vec![!0u64; w];
            let mut vn = vec![0u64; w];
            if last_bits < 64 { vp[w - 1] = (1u64 << last_bits) - 1; }
            let mut score = m as isize;
            for &tj in long {
                let base = match keys.binary_search(&tj) {
                    Ok(ki) => ki * w,
                    Err(_)  => zero_base,
                };
                let mut carry = false;
                let mut prev_hp = 1u64;
                let mut prev_hn = 0u64;
                for k in 0..w {
                    let pm = unsafe { *data.get_unchecked(base + k) };
                    let x = pm | unsafe { *vn.get_unchecked(k) };
                    let vp_k = unsafe { *vp.get_unchecked(k) };
                    let (sum, nc) = carrying_add(x & vp_k, vp_k, carry);
                    carry = nc;
                    let d0 = (sum ^ vp_k) | x;
                    let vn_k = unsafe { *vn.get_unchecked(k) };
                    let hp = vn_k | !(d0 | vp_k);
                    let hn = vp_k & d0;
                    if k == w - 1 { score += (hp & top_mask != 0) as isize - (hn & top_mask != 0) as isize; }
                    let (hp_msb, hn_msb) = (hp >> 63, hn >> 63);
                    unsafe {
                        *vp.get_unchecked_mut(k) = (hn << 1 | prev_hn) | !(d0 | (hp << 1 | prev_hp));
                        *vn.get_unchecked_mut(k) = (hp << 1 | prev_hp) & d0;
                    }
                    (prev_hp, prev_hn) = (hp_msb, hn_msb);
                }
            }
            score as usize
        }
    }
}

// ---------------------------------------------------------------------------
// Wagner-Fischer (kept as oracle for unit tests only)
// ---------------------------------------------------------------------------

fn wagner_fischer<T: Eq>(short: &[T], long: &[T]) -> usize {
    let m = short.len();
    let mut row: Vec<usize> = (0..=m).collect();
    for (j, lj) in long.iter().enumerate() {
        let mut diag = row[0];
        row[0] = j + 1;
        for (i, si) in short.iter().enumerate() {
            let cost  = (si != lj) as usize;
            let above = row[i + 1];
            let left  = row[i];
            let new   = (above + 1).min(left + 1).min(diag + cost);
            diag      = above;
            row[i + 1] = new;
        }
    }
    row[m]
}

// ---------------------------------------------------------------------------
// Test helpers (bypass the PyO3 layer; no GIL release)
// ---------------------------------------------------------------------------

#[cfg(test)]
fn finish<T: Eq>(prep: Prep<T>) -> usize {
    match prep {
        Prep::Done(d)  => d,
        Prep::Wf(s, l) => wagner_fischer(&s, &l),
    }
}

#[cfg(test)]
fn levenshtein(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        finish(prep_bytes::<true>(s1.as_bytes(), s2.as_bytes()))
    } else {
        // No Python interpreter in unit tests; cast chars to u32 directly.
        if s1 == s2 { return 0; }
        let a: Vec<u32> = s1.chars().map(|c| c as u32).collect();
        let b: Vec<u32> = s2.chars().map(|c| c as u32).collect();
        finish(prep_fixed(&a, &b))
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
        for i in 0..=m { dp[i][0] = i; }
        for j in 0..=n { dp[0][j] = j; }
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
        let (sh, lo) = if s1.len() <= s2.len() { (&s1[..], &s2[..]) } else { (&s2[..], &s1[..]) };
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

    /// Directly exercise `hyrro_multiword` (u32, m > 64) against the naive DP.
    #[test]
    fn multiword_u32_matches_oracle() {
        let check_u32 = |a: &[u32], b: &[u32]| {
            let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            assert!(s.len() > 64, "test case must be long enough to hit multiword");
            let got = hyrro_multiword(s, l);
            let ac: Vec<char> = s.iter().map(|&c| char::from_u32(c).unwrap_or('?')).collect();
            let bc: Vec<char> = l.iter().map(|&c| char::from_u32(c).unwrap_or('?')).collect();
            let exp = naive(&ac, &bc);
            assert_eq!(got, exp, "u32 mismatch ({} vs {} chars)", s.len(), l.len());
        };

        // Emoji run: each emoji is one code point — build 80-char runs.
        let emoji_a: Vec<u32> = [0x1f980u32, 0x1f40d, 0x1f389, 0x1f38a, 0x1f388]
            .iter().copied().cycle().take(80).collect();
        let emoji_b: Vec<u32> = [0x1f40du32, 0x1f980, 0x1f389, 0x1f38a, 0x1f388]
            .iter().copied().cycle().take(80).collect();
        check_u32(&emoji_a, &emoji_b);

        // CJK run: 80 characters from the Japanese test string.
        let cjk_a: Vec<u32> = "日本語のテスト文字列".chars().map(|c| c as u32).cycle().take(80).collect();
        let cjk_b: Vec<u32> = "日本語のテスツ文字列".chars().map(|c| c as u32).cycle().take(80).collect();
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
            assert_eq!(finish(prep_fixed(&au, &bu)), expected, "u16 ({a:?}, {b:?})");
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
            assert_eq!(finish(prep_fixed(&au, &bu)), expected, "u32 ({a:?}, {b:?})");
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
        let long  = &s2_full[13..];
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
        eprintln!("raw Rust hyrro_multiword_bytes: {:.3} μs/call  (sink={})", us, sink);
        assert!(sink > 0); // prevent DCE
    }

    #[test]
    fn ratio_basic() {
        let r = |a: &str, b: &str| -> f64 {
            let total = a.chars().count() + b.chars().count();
            if total == 0 { 1.0 } else { 1.0 - levenshtein(a, b) as f64 / total as f64 }
        };
        assert!((r("", "") - 1.0).abs() < 1e-12);
        assert!((r("abc", "abc") - 1.0).abs() < 1e-12);
        assert!((r("kitten", "sitting") - (1.0 - 3.0 / 13.0)).abs() < 1e-12);
        assert!((r("abc", "xyz") - 0.5).abs() < 1e-12);
    }
}
