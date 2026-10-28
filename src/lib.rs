//! High-performance Python bindings for the Levenshtein distance.
//!
//! This crate exposes two functions to Python via PyO3:
//!
//! * [`distance`] – the Levenshtein edit distance between two strings.
//! * [`ratio`]    – the indel similarity score in `[0.0, 1.0]`.
//!
//! # Algorithm
//!
//! See <https://en.wikipedia.org/wiki/Levenshtein_distance> for the distance
//! definition.  Several optimizations are layered to minimise the constant factor:
//!
//! 1. **Identity short-circuit** – the same object passed twice returns
//!    immediately.
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
//!      lookup at ≤ 50 % load.  Mixed-kind pairs compare code units in place;
//!      no UTF-8 round-trip, lone surrogates preserved.
//! 4. **Hyyrö's bit-parallel algorithm** – O(⌈m/w⌉ · n) time with w = 64;
//!    see H. Hyyrö, "A Bit-Vector Algorithm for Computing Levenshtein and
//!    Damerau Edit Distances", *Nordic Journal of Computing*, 2003.
//!    - *Tiny* (`m ≤ 8`): no peq table; the match word is computed per text
//!      char (SWAR for byte patterns, an unrolled scan otherwise).
//!    - *Single-word* (`m ≤ 64`): one 64-bit word covers the whole pattern.
//!    - *Multi-word* (`m > 64`): ⌈m/64⌉ words with carry propagation.
//!      UCS-1 uses a `[[u64; W]; 128 or 256]` stack peq (O(1) lookup, no
//!      heap).  UCS-2/4 use a peq data array addressed through the same kind
//!      of open-addressing hash index, on the stack up to `m ≤ 512`.
//!    - *Large* (`m > 512`): one loop over all words per column; on
//!      AArch64, strips of eight words sweep the text one after another
//!      with the const-width kernel instead, handing the per-column carry
//!      and boundary bits upwards, so each strip's state stays in registers.
//!      The LCS full matrix runs in strips of four words on x86-64 too.
//!    - Steps with target-specific forms live in [`kernels`], one module per
//!      target, with a dispatcher that picks the form for the build target
//!      and CPU.
//! 5. **Ukkonen banding** (`m > 512`) – similar strings are resolved by a
//!    narrow diagonal band over Myers' blocked recurrence (O(n · d/w) instead
//!    of O(n · m/w)).  Band widths double until one proves its result; each
//!    pass aborts early, which keeps the overhead on dissimilar strings to a
//!    few percent before falling back to the full matrix.
//!
//! [`ratio`] reuses steps 1–3 verbatim and swaps step 4 for the bit-parallel
//! LCS recurrence of Crochemore et al. (2001) / Hyyrö (2004), from which the
//! indel distance follows as `m + n - 2 · LLCS`; indel distance is invariant
//! under common-affix stripping for the same reason Levenshtein distance is.
//! Step 5 has an LCS counterpart that restricts the recurrence to a band of
//! diagonals and certifies its result (see `LargeCtx::lcs_banded`).  mbleven
//! stays Levenshtein-only: against the three-op LCS step it does not pay.
//!
//! The crate builds for each target's baseline ISA.  On x86-64 the whole
//! pipeline is compiled a second time with the x86-64-v2 features, and the
//! module picks one of the two copies when it loads (see [`Isa`]).

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::os::raw::c_uint;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

mod kernels;

// ---------------------------------------------------------------------------
// Code units of the three CPython string kinds (UCS-1 / UCS-2 / UCS-4).
// ---------------------------------------------------------------------------

/// A CPython code unit (`u8`, `u16` or `u32`).  Mixed-kind pairs compare
/// code units widened to `u64`, or to `u32` where blocks of them are
/// compared as vectors.
trait CodeUnit: Copy + Eq + Into<u64> + Into<u32> {
    #[inline(always)]
    fn as_u64(self) -> u64 {
        Into::<u64>::into(self)
    }

    #[inline(always)]
    fn as_u32(self) -> u32 {
        Into::<u32>::into(self)
    }
}
impl CodeUnit for u8 {}
impl CodeUnit for u16 {}
impl CodeUnit for u32 {}

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
    (key.wrapping_mul(0x9e37_79b9_7f4a_7c15_u64) >> shift) as usize
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
///
/// # Safety
///
/// `s` must reference a valid, fully-initialized CPython `PyString` (`PyUnicodeObject`).
#[cfg(not(all(
    Py_3_14,
    target_endian = "little",
    not(all(Py_GIL_DISABLED, target_env = "msvc"))
)))]
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

/// Python 3.14+ fast path: pyo3-ffi routes `PyUnicode_KIND` / `PyUnicode_DATA`
/// through exported C functions there and drops `PyUnicode_IS_ASCII` entirely
/// (four cross-library calls per distance/ratio call, and the pure-ASCII peq
/// path would be lost).  The `PyASCIIObject` header layout is unchanged in
/// CPython 3.14/3.15 (state bitfield: `interned:2, kind:3, compact:1, ascii:1`,
/// LSB-first on little-endian targets — the same assumption pyo3-ffi itself
/// makes up to 3.13), so decode the header directly.  kind/compact/ascii are
/// immutable after string creation, so reading them needs no synchronization
/// on free-threaded builds.  Debug builds cross-check against the C API.
///
/// Free-threaded builds store `interned` as a whole byte (it is accessed
/// atomically), which moves the flags up to bit 8.  MSVC does not pack a
/// bitfield into the byte before it, so its free-threaded `state` is wider
/// than pyo3-ffi's `u32`; that combination takes the C-API path above.
///
/// # Safety
///
/// `s` must reference a valid, fully-initialized CPython `PyString` (`PyUnicodeObject`).
#[cfg(all(
    Py_3_14,
    target_endian = "little",
    not(all(Py_GIL_DISABLED, target_env = "msvc"))
))]
#[inline(always)]
unsafe fn view(s: &Bound<'_, PyString>) -> UniView {
    const KIND_SHIFT: u32 = if cfg!(Py_GIL_DISABLED) { 8 } else { 2 };
    let ptr = s.as_ptr();
    let obj = ptr as *mut ffi::PyASCIIObject;
    let state = (*obj).state;
    let kind = ((state >> KIND_SHIFT) & 0b111) as c_uint;
    let compact = (state >> (KIND_SHIFT + 3)) & 1 != 0;
    let ascii = (state >> (KIND_SHIFT + 4)) & 1 != 0;
    let data = if compact {
        if ascii {
            obj.add(1) as *const u8
        } else {
            (ptr as *mut ffi::PyCompactUnicodeObject).add(1) as *const u8
        }
    } else {
        (*(ptr as *mut ffi::PyUnicodeObject)).data.any as *const u8
    };
    debug_assert_eq!(
        kind,
        ffi::PyUnicode_KIND(ptr),
        "state bitfield layout changed"
    );
    debug_assert_eq!(
        data,
        ffi::PyUnicode_DATA(ptr) as *const u8,
        "compact data offset changed"
    );
    UniView {
        kind,
        ascii,
        data,
        len: (*obj).length as usize,
    }
}

/// Reinterpret a [`UniView`]'s data as a `u8` slice.
///
/// # Safety
///
/// `v.data` must point to `v.len` valid, initialized `u8` code units.
#[inline(always)]
unsafe fn as_u8(v: &UniView) -> &[u8] {
    // SAFETY: caller guarantees v.data points to v.len valid u8 elements.
    unsafe { std::slice::from_raw_parts(v.data, v.len) }
}

/// Reinterpret a [`UniView`]'s data as a `u16` slice.
///
/// # Safety
///
/// `v.data` must be properly aligned for `u16` and point to `v.len` valid, initialized `u16` code units.
#[inline(always)]
unsafe fn as_u16(v: &UniView) -> &[u16] {
    // SAFETY: caller guarantees v.data is aligned and points to v.len valid u16 elements.
    unsafe { std::slice::from_raw_parts(v.data.cast::<u16>(), v.len) }
}

/// Reinterpret a [`UniView`]'s data as a `u32` slice.
///
/// # Safety
///
/// `v.data` must be properly aligned for `u32` and point to `v.len` valid, initialized `u32` code units.
#[inline(always)]
unsafe fn as_u32(v: &UniView) -> &[u32] {
    // SAFETY: caller guarantees v.data is aligned and points to v.len valid u32 elements.
    unsafe { std::slice::from_raw_parts(v.data.cast::<u32>(), v.len) }
}

/// [`distance`] in the copy `X` of the hot path.
///
/// # Safety
///
/// The CPU must support the instruction set `X`.
#[inline(always)]
unsafe fn distance_in<X: Isa>(
    s1: Borrowed<'_, '_, PyString>,
    s2: Borrowed<'_, '_, PyString>,
) -> usize {
    if s1.is(s2) {
        return 0;
    }
    compute::<Lev<X>>(&view(&s1), &view(&s2))
}

/// [`ratio`] in the copy `X` of the hot path.
///
/// # Safety
///
/// The CPU must support the instruction set `X`.
#[inline(always)]
unsafe fn ratio_in<X: Isa>(s1: Borrowed<'_, '_, PyString>, s2: Borrowed<'_, '_, PyString>) -> f64 {
    if s1.is(s2) {
        return 1.0;
    }
    let (v1, v2) = (view(&s1), view(&s2));
    let total = v1.len + v2.len;
    if total == 0 {
        return 1.0;
    }
    1.0 - compute::<Lcs<X>>(&v1, &v2) as f64 / total as f64
}

/// Define the PyO3 [`distance`] and [`ratio`] entry points for one copy of the
/// hot path, calling `$distance` and `$ratio` with `Borrowed` handles so the
/// object pointers cross any `#[target_feature]` boundary in registers.
macro_rules! entry_points {
    ($distance:path, $ratio:path) => {
        /// Levenshtein edit distance between two strings.
        ///
        /// The distance is the minimum number of single-character insertions,
        /// deletions, or substitutions required to transform `s1` into `s2`.
        ///
        /// Characters are counted as Unicode code points, so an accented letter or an
        /// emoji counts as one character.
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
        // `pass_module` binds the functions to the module instead of registering
        // them `METH_STATIC`, which PyO3 0.28 does otherwise at a cost of a few ns per
        // call on CPython 3.11+ (PyO3 issue #6426).
        #[pyfunction]
        #[pyo3(pass_module, signature = (s1, s2, /))]
        pub(crate) fn distance(
            _m: &Bound<'_, PyModule>,
            s1: &Bound<'_, PyString>,
            s2: &Bound<'_, PyString>,
        ) -> usize {
            // SAFETY: `_lev` registers this copy's functions only when the CPU
            // has its features (`Baseline` always does).
            unsafe { $distance(s1.as_borrowed(), s2.as_borrowed()) }
        }

        /// Calculate how similar two strings are, as a score from `0.0` to `1.0`.
        ///
        /// A score of `1.0` means the strings are identical, and `0.0` means they
        /// have no character in common.  In between, the score is the share of
        /// characters the two strings have in common, known as the indel similarity
        /// ratio:
        ///
        /// $$
        /// \mathrm{ratio}(s_1, s_2)
        ///   = \frac{2 \cdot \mathrm{LCS}(s_1, s_2)}{|s_1| + |s_2|}
        /// $$
        ///
        /// where $\mathrm{LCS}$ is the length of the longest common subsequence and
        /// $|s|$ is the length of $s$.
        ///
        /// Characters are counted as Unicode code points, so an accented letter or an
        /// emoji counts as one character.
        ///
        /// Args:
        ///     s1 (str): First input string.
        ///     s2 (str): Second input string.
        ///
        /// Returns:
        ///     Similarity score between `0.0` (nothing in common) and `1.0` (identical).
        ///
        /// Examples:
        ///     >>> import lev
        ///     >>> lev.ratio("kitten", "sitting")
        ///     0.6153846153846154
        ///     >>> lev.ratio("abc", "xyz")
        ///     0.0
        ///     >>> lev.ratio("", "")
        ///     1.0
        #[pyfunction]
        #[pyo3(pass_module, signature = (s1, s2, /))]
        pub(crate) fn ratio(
            _m: &Bound<'_, PyModule>,
            s1: &Bound<'_, PyString>,
            s2: &Bound<'_, PyString>,
        ) -> f64 {
            // SAFETY: see `distance`.
            unsafe { $ratio(s1.as_borrowed(), s2.as_borrowed()) }
        }
    };
}
#[cfg(target_arch = "x86_64")]
pub(crate) use entry_points;

entry_points!(distance_in::<Baseline>, ratio_in::<Baseline>);

/// The compiled extension backing the `lev` package.
///
/// Private: the public API is re-exported from `lev/__init__.py`, so the name
/// is underscore-prefixed to keep it out of the package namespace.
#[pymodule]
fn _lev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(target_arch = "x86_64")]
    if kernels::x86_64::v2::detected() {
        use kernels::x86_64::v2::py;
        m.add_function(wrap_pyfunction!(py::distance, m)?)?;
        m.add_function(wrap_pyfunction!(py::ratio, m)?)?;
        return Ok(());
    }
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Copies of the hot path, one per instruction set
// ---------------------------------------------------------------------------

/// One copy of the hot path, compiled for one instruction set, which
/// [`Kernel::Isa`] carries through the pipeline.
///
/// [`Baseline`] is the copy for the target's baseline ISA, which every CPU
/// of the target runs, and the only one off x86-64.  On x86-64 the crate
/// builds for the x86-64-v1 ISA, and [`kernels::x86_64::v2::V2`] adds the
/// x86-64-v2 features; [`_lev`] installs its entry points when the CPU has
/// them.  A crate-wide `-C target-cpu=x86-64-v2` would get the features
/// too, but lets LLVM use them anywhere, so the wheels would crash with
/// SIGILL on older CPUs.  Detecting them per kernel instead would leave the
/// code between the kernels (affix scans, peq tables, the final popcounts)
/// on x86-64-v1.
///
/// Code inlined into a function is compiled with that function's target
/// features, so a copy needs its own form of each function the pipeline
/// keeps out of line.  [`hyrro_64_tiny`], [`hyrro_multiword_direct`],
/// [`hyrro_multiword_hash`] and [`LargeCtx::banded`] call their copy's form
/// of their body; [`mbleven`] is called through [`mbleven_in`], and the
/// copies with target features inline [`LargeCtx::run`] by hand.  These
/// functions test [`Isa::X86_64_V2`], a constant, rather than call methods
/// of the copy: rustc inlines MIR only into calls it can resolve before
/// monomorphization, and calls through trait methods changed the AArch64
/// code of the baseline copy.  As it is, that code is the same per function
/// as before there were copies.
///
/// A pipeline of a copy other than [`Baseline`] is only reached from that
/// copy's entry points, so the CPU has its features.
trait Isa {
    /// Whether the copy has the x86-64-v2 features: its out-of-line
    /// functions are those of [`kernels::x86_64::v2`], and the forms of
    /// [`kernels::x86_64::bmi1`], which need them besides BMI1, may run.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    const X86_64_V2: bool;
}

/// The copy of the hot path for the target's baseline ISA (see [`Isa`]).
struct Baseline;

impl Isa for Baseline {
    const X86_64_V2: bool = false;
}

// ---------------------------------------------------------------------------
// Top-level dispatch: read both views' kinds, branch once, run the right path.
// ---------------------------------------------------------------------------

/// Dispatch on `(kind, ascii)` tuples and return the `K` edit distance directly.
///
/// # Safety
///
/// Both views must point to valid initialized code-unit buffers corresponding
/// to their respective `kind` flags.
#[inline(always)]
unsafe fn compute<K: Kernel>(v1: &UniView, v2: &UniView) -> usize {
    use ffi::{PyUnicode_1BYTE_KIND as K1, PyUnicode_2BYTE_KIND as K2, PyUnicode_4BYTE_KIND as K4};

    // 1. Normalize: v1 is the shorter pattern string (m), v2 is the text (n).
    let (v1, v2) = if v1.len <= v2.len { (v1, v2) } else { (v2, v1) };

    // 2. Dispatch based on kinds.
    match (v1.kind, v2.kind) {
        (K1, K1) => {
            // SAFETY: kind is 1-byte, so data is valid &[u8].
            let (b1, b2) = unsafe { (as_u8(v1), as_u8(v2)) };
            if v1.ascii && v2.ascii {
                compute_u8::<K, true>(b1, b2)
            } else {
                compute_u8::<K, false>(b1, b2)
            }
        }
        // SAFETY: kind is 2-byte, so data is valid &[u16].
        (K2, K2) => compute_sorted::<K, _>(unsafe { as_u16(v1) }, unsafe { as_u16(v2) }),
        // SAFETY: kind is 4-byte, so data is valid &[u32].
        (K4, K4) => compute_sorted::<K, _>(unsafe { as_u32(v1) }, unsafe { as_u32(v2) }),
        // Mixed kinds compare code units in place, without a widened copy.
        (K1, K2) => compute_sorted_mixed::<K, _, _>(unsafe { as_u8(v1) }, unsafe { as_u16(v2) }),
        (K1, K4) => compute_sorted_mixed::<K, _, _>(unsafe { as_u8(v1) }, unsafe { as_u32(v2) }),
        (K2, K1) => compute_sorted_mixed::<K, _, _>(unsafe { as_u16(v1) }, unsafe { as_u8(v2) }),
        (K2, K4) => compute_sorted_mixed::<K, _, _>(unsafe { as_u16(v1) }, unsafe { as_u32(v2) }),
        (K4, K1) => compute_sorted_mixed::<K, _, _>(unsafe { as_u32(v1) }, unsafe { as_u8(v2) }),
        (K4, K2) => compute_sorted_mixed::<K, _, _>(unsafe { as_u32(v1) }, unsafe { as_u16(v2) }),
        _ => unreachable!("unknown PyUnicode kind"),
    }
}

// ---------------------------------------------------------------------------
// Affix stripping
// ---------------------------------------------------------------------------

/// Code units compared per branch by the mixed-kind scans.
const MIXED_BLOCK: usize = 16;

/// Number of positions where two blocks of different code-unit types
/// differ.  No early exit, so LLVM widens and compares the block as vectors.
#[inline(always)]
fn block_mismatches<T1: CodeUnit, T2: CodeUnit>(
    x: &[T1; MIXED_BLOCK],
    y: &[T2; MIXED_BLOCK],
) -> usize {
    x.iter()
        .zip(y)
        .map(|(p, q)| (p.as_u32() != q.as_u32()) as usize)
        .sum()
}

/// Strip common prefix and suffix from two slices of different types: whole
/// blocks first, then code unit by code unit from the first differing block.
#[inline(always)]
fn strip_affix_mixed<'a, T1: CodeUnit, T2: CodeUnit>(
    a: &'a [T1],
    b: &'a [T2],
) -> (&'a [T1], &'a [T2]) {
    let n = a.len().min(b.len());
    let mut prefix = 0;
    while prefix + MIXED_BLOCK <= n {
        let Some(blk_a) = a[prefix..].first_chunk::<MIXED_BLOCK>() else {
            break;
        };
        let Some(blk_b) = b[prefix..].first_chunk::<MIXED_BLOCK>() else {
            break;
        };
        if block_mismatches(blk_a, blk_b) != 0 {
            break;
        }
        prefix += MIXED_BLOCK;
    }
    while prefix < n && a[prefix].as_u64() == b[prefix].as_u64() {
        prefix += 1;
    }

    let a = &a[prefix..];
    let b = &b[prefix..];

    let (n, la, lb) = (n - prefix, a.len(), b.len());
    let mut suffix = 0;
    while suffix + MIXED_BLOCK <= n {
        let Some(blk_a) = a[..la - suffix].last_chunk::<MIXED_BLOCK>() else {
            break;
        };
        let Some(blk_b) = b[..lb - suffix].last_chunk::<MIXED_BLOCK>() else {
            break;
        };
        if block_mismatches(blk_a, blk_b) != 0 {
            break;
        }
        suffix += MIXED_BLOCK;
    }
    while suffix < n && a[la - suffix - 1].as_u64() == b[lb - suffix - 1].as_u64() {
        suffix += 1;
    }

    (&a[..la - suffix], &b[..lb - suffix])
}

/// Reinterpret a code-unit slice as raw bytes (sound: u8/u16/u32 are plain data).
#[inline(always)]
pub(crate) fn as_bytes<T: CodeUnit>(s: &[T]) -> &[u8] {
    // SAFETY: `s` is a valid slice of `CodeUnit` primitives (`u8`, `u16`, or `u32`),
    // which are plain data with no uninitialized bytes or padding. The total byte
    // length `std::mem::size_of_val(s)` correctly represents the backing memory.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
}

/// Bytes compared per branch by the affix scans: the XORs of four words are
/// ORed into one test, which LLVM lowers to 16-byte vector compares.
const AFFIX_BLOCK: usize = 32;

/// Whether two `AFFIX_BLOCK`-byte blocks differ, without an early exit.
#[inline(always)]
fn blocks_differ(x: &[u8; AFFIX_BLOCK], y: &[u8; AFFIX_BLOCK]) -> bool {
    let (wx, _) = x.as_chunks::<8>();
    let (wy, _) = y.as_chunks::<8>();
    wx.iter().zip(wy).fold(0u64, |acc, (p, q)| {
        acc | (u64::from_ne_bytes(*p) ^ u64::from_ne_bytes(*q))
    }) != 0
}

/// Length in bytes of the common prefix of `a` and `b`: whole blocks first,
/// then 8 bytes at a time from the first differing block on.
#[inline(always)]
fn common_prefix_bytes(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + AFFIX_BLOCK <= n {
        let Some(blk_a) = a[i..].first_chunk::<AFFIX_BLOCK>() else {
            break;
        };
        let Some(blk_b) = b[i..].first_chunk::<AFFIX_BLOCK>() else {
            break;
        };
        if blocks_differ(blk_a, blk_b) {
            break;
        }
        i += AFFIX_BLOCK;
    }
    while i + 8 <= n {
        let Some(chunk_a) = a[i..].first_chunk::<8>() else {
            break;
        };
        let Some(chunk_b) = b[i..].first_chunk::<8>() else {
            break;
        };
        let diff = u64::from_le_bytes(*chunk_a) ^ u64::from_le_bytes(*chunk_b);
        if diff != 0 {
            return i + (diff.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Length in bytes of the common suffix of `a` and `b`, scanned like
/// [`common_prefix_bytes`] from the end.
#[inline(always)]
fn common_suffix_bytes(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let (la, lb) = (a.len(), b.len());
    let mut i = 0;
    while i + AFFIX_BLOCK <= n {
        let Some(blk_a) = a[..la - i].last_chunk::<AFFIX_BLOCK>() else {
            break;
        };
        let Some(blk_b) = b[..lb - i].last_chunk::<AFFIX_BLOCK>() else {
            break;
        };
        if blocks_differ(blk_a, blk_b) {
            break;
        }
        i += AFFIX_BLOCK;
    }
    while i + 8 <= n {
        let Some(chunk_a) = a[..la - i].last_chunk::<8>() else {
            break;
        };
        let Some(chunk_b) = b[..lb - i].last_chunk::<8>() else {
            break;
        };
        let diff = u64::from_le_bytes(*chunk_a) ^ u64::from_le_bytes(*chunk_b);
        if diff != 0 {
            return i + (diff.leading_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[la - i - 1] == b[lb - i - 1] {
        i += 1;
    }
    i
}

/// Strip common prefix and suffix from two same-kind slices.
///
/// Code-unit equality is byte equality for equal-width slices, so the scan
/// runs on the raw bytes 8 at a time; matched byte counts are floor-divided
/// by the element size, which discards any partial element at a difference
/// boundary.
#[inline(always)]
fn strip_affix<'a, T: CodeUnit>(a: &'a [T], b: &'a [T]) -> (&'a [T], &'a [T]) {
    let size = std::mem::size_of::<T>();
    let prefix = common_prefix_bytes(as_bytes(a), as_bytes(b)) / size;

    let a = &a[prefix..];
    let b = &b[prefix..];

    let suffix = common_suffix_bytes(as_bytes(a), as_bytes(b)) / size;

    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}

// ---------------------------------------------------------------------------
// UCS-1 pipeline
// ---------------------------------------------------------------------------

/// Patterns at or below this length skip peq-table construction entirely; the
/// pattern-match word is computed per text char by vector comparisons or a
/// fully unrolled linear scan.  Zero-initializing the peq tables (1-2 KiB)
/// dominates the runtime for tiny inputs, so trading one table load for ≤ `TINY_M`
/// SIMD or compare+or ops wins.
const TINY_M: usize = 16;

/// Tiny-pattern kernel: no peq table, `pm` computed per text char directly
/// from the pattern.
///
/// Single-byte patterns up to 16 characters use 128-bit SIMD comparisons
/// (NEON on AArch64, SSE2 on x86-64), avoiding any peq table setup. Wider
/// patterns dispatch on a const length so the scan unrolls into `m` compares.
///
/// Kept out of line: inlining these loops into the dispatcher perturbs
/// register allocation of the single-word paths sharing that function.
/// Each copy of the hot path compiles its own (see [`Isa`]).
#[inline(never)]
fn hyrro_64_tiny<K: Kernel, T1: CodeUnit, T2: CodeUnit>(pattern: &[T1], text: &[T2]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if K::Isa::X86_64_V2 {
        // SAFETY: the CPU runs `K::Isa` (see `Isa`).
        return unsafe { kernels::x86_64::v2::hyrro_64_tiny::<K, T1, T2>(pattern, text) };
    }
    hyrro_64_tiny_body::<K, T1, T2>(pattern, text)
}

/// The body of [`hyrro_64_tiny`].
#[inline(always)]
fn hyrro_64_tiny_body<K: Kernel, T1: CodeUnit, T2: CodeUnit>(pattern: &[T1], text: &[T2]) -> usize {
    debug_assert!((1..=TINY_M).contains(&pattern.len()));
    if std::mem::size_of::<T1>() == 1 {
        if pattern.len() <= 8 {
            return tiny_swar::<K, _, _>(pattern, text);
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: AArch64 baseline guarantees NEON support, and `pattern` code units
            // are single-byte (`size_of::<T1>() == 1`).
            return unsafe { tiny_u8_simd::<K, T1, T2>(pattern, text) };
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: x86_64 baseline guarantees SSE2 support, and `pattern` code units
            // are single-byte (`size_of::<T1>() == 1`).
            return unsafe { tiny_u8_simd::<K, T1, T2>(pattern, text) };
        }
    }
    match pattern.len() {
        1 => tiny_n::<K, 1, _, _>(pattern, text),
        2 => tiny_n::<K, 2, _, _>(pattern, text),
        3 => tiny_n::<K, 3, _, _>(pattern, text),
        4 => tiny_n::<K, 4, _, _>(pattern, text),
        5 => tiny_n::<K, 5, _, _>(pattern, text),
        6 => tiny_n::<K, 6, _, _>(pattern, text),
        7 => tiny_n::<K, 7, _, _>(pattern, text),
        8 => tiny_n::<K, 8, _, _>(pattern, text),
        9 => tiny_n::<K, 9, _, _>(pattern, text),
        10 => tiny_n::<K, 10, _, _>(pattern, text),
        11 => tiny_n::<K, 11, _, _>(pattern, text),
        12 => tiny_n::<K, 12, _, _>(pattern, text),
        13 => tiny_n::<K, 13, _, _>(pattern, text),
        14 => tiny_n::<K, 14, _, _>(pattern, text),
        15 => tiny_n::<K, 15, _, _>(pattern, text),
        16 => tiny_n::<K, 16, _, _>(pattern, text),
        _ => unreachable!("tiny path requires 1..=TINY_M pattern chars"),
    }
}

/// [`hyrro_64_tiny`] for single-byte patterns using 128-bit NEON vector comparisons.
///
/// # Safety
///
/// Requires `size_of::<T1>() == 1` and `pattern.len() <= TINY_M`.
/// Target must support AArch64 NEON vector instructions.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn tiny_u8_simd<K: Kernel, T1: CodeUnit, T2: CodeUnit>(
    pattern: &[T1],
    text: &[T2],
) -> usize {
    use std::arch::aarch64::*;
    let m = pattern.len();
    debug_assert!(std::mem::size_of::<T1>() == 1 && (1..=TINY_M).contains(&m));
    let mut pat_buf = [0u8; 16];
    pat_buf[..m].copy_from_slice(as_bytes(pattern));
    let pat_vec = vld1q_u8(pat_buf.as_ptr());
    let valid = if m == 64 { u64::MAX } else { (1u64 << m) - 1 };

    const IDX_BYTES: [u8; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
    const MBITS_U16: [u16; 8] = [
        0x0101, 0x0202, 0x0404, 0x0808, 0x1010, 0x2020, 0x4040, 0x8080,
    ];
    let idx = vld1q_u8(IDX_BYTES.as_ptr());
    let mbits = vld1q_u16(MBITS_U16.as_ptr());

    K::word(
        m,
        text.len(),
        text.iter().map(|&c| {
            let c_val = c.as_u64();
            if c_val > 0xff {
                return 0;
            }
            let c_vec = vdupq_n_u8(c_val as u8);
            let eq = vceqq_u8(pat_vec, c_vec);
            let perm = vqtbl1q_u8(eq, idx);
            let anded = vandq_u16(vreinterpretq_u16_u8(perm), mbits);
            let mask = vaddvq_u16(anded) as u64;
            mask & valid
        }),
    )
}

/// [`hyrro_64_tiny`] for single-byte patterns using 128-bit SSE2 vector comparisons.
///
/// # Safety
///
/// Requires `size_of::<T1>() == 1` and `pattern.len() <= TINY_M`.
/// Target must support x86-64 SSE2 vector instructions.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tiny_u8_simd<K: Kernel, T1: CodeUnit, T2: CodeUnit>(
    pattern: &[T1],
    text: &[T2],
) -> usize {
    use std::arch::x86_64::*;
    let m = pattern.len();
    debug_assert!(std::mem::size_of::<T1>() == 1 && (1..=TINY_M).contains(&m));
    let mut pat_buf = [0u8; 16];
    pat_buf[..m].copy_from_slice(as_bytes(pattern));
    let pat_vec = _mm_loadu_si128(pat_buf.as_ptr() as *const __m128i);
    let valid = if m == 64 { u64::MAX } else { (1u64 << m) - 1 };

    K::word(
        m,
        text.len(),
        text.iter().map(|&c| {
            let c_val = c.as_u64();
            if c_val > 0xff {
                return 0;
            }
            let c_vec = _mm_set1_epi8(c_val as i8);
            let eq = _mm_cmpeq_epi8(pat_vec, c_vec);
            let mask = _mm_movemask_epi8(eq) as u64;
            mask & valid
        }),
    )
}

/// [`hyrro_64_tiny`] for a multi-byte pattern of exactly `N` code units.
#[inline(always)]
fn tiny_n<K: Kernel, const N: usize, T1: CodeUnit, T2: CodeUnit>(
    pattern: &[T1],
    text: &[T2],
) -> usize {
    let pattern: [u64; N] = std::array::from_fn(|i| pattern[i].as_u64());
    K::word(
        N,
        text.len(),
        text.iter().map(|&c| {
            let c = c.as_u64();
            let mut pm = 0u64;
            for (i, &p) in pattern.iter().enumerate() {
                pm |= ((p == c) as u64) << i;
            }
            pm
        }),
    )
}

/// [`hyrro_64_tiny`] for single-byte patterns: the pattern is packed one byte
/// per lane into a u64 and matched against each text char with SWAR, ~10
/// scalar ops per char.
#[inline(always)]
fn tiny_swar<K: Kernel, T1: CodeUnit, T2: CodeUnit>(pattern: &[T1], text: &[T2]) -> usize {
    const LO7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    const HI: u64 = 0x8080_8080_8080_8080;
    const ONES: u64 = 0x0101_0101_0101_0101;
    // Multiplying the lane LSBs (bits 8i) by this moves bit 8i to 56 + i with
    // no overlapping partial products, i.e. no carries.
    const GATHER: u64 = 0x0102_0408_1020_4080;
    let m = pattern.len();
    debug_assert!(std::mem::size_of::<T1>() == 1 && (1..=8).contains(&m));

    let packed = pattern
        .iter()
        .rev()
        .fold(0u64, |acc, &p| (acc << 8) | p.as_u64());
    // Lanes >= m hold zero bytes and would match NUL; mask them off.
    let valid = u64::MAX >> (64 - m);
    K::word(
        m,
        text.len(),
        text.iter().map(|&c| {
            let c = c.as_u64();
            if c > 0xff {
                return 0;
            }
            // Exact zero-byte test: the high bit of each lane of `hit` is set
            // iff that lane of `x` is zero (no cross-lane borrows).
            let x = packed ^ c.wrapping_mul(ONES);
            let hit = !(((x & LO7) + LO7) | x) & HI;
            ((hit >> 7).wrapping_mul(GATHER) >> 56) & valid
        }),
    )
}

/// Strip affixes and run kernel `K`.  `ASCII = true` enables the 128-entry peq
/// fast path.
#[inline(always)]
fn compute_u8<K: Kernel, const ASCII: bool>(a: &[u8], b: &[u8]) -> usize {
    let (a, b) = strip_affix(a, b);
    if a.is_empty() {
        return b.len();
    }
    if a.len() <= TINY_M {
        return hyrro_64_tiny::<K, _, _>(a, b);
    }
    if K::MBLEVEN {
        if let Some(ub) = small_ub::<K::Isa, _>(a, b) {
            return mbleven_in::<K::Isa, _, _>(a, b, ub);
        }
    }
    if a.len() <= 64 {
        if ASCII {
            hyrro_64_u8::<K, 128>(a, b)
        } else {
            hyrro_64_u8::<K, 256>(a, b)
        }
    } else {
        if ASCII {
            hyrro_multiword_direct::<K, 128, _, _>(a, b)
        } else {
            hyrro_multiword_direct::<K, 256, _, _>(a, b)
        }
    }
}

// ---------------------------------------------------------------------------
// Small-distance fast path (mbleven)
// ---------------------------------------------------------------------------

/// Largest internally-derived upper bound handled by [`mbleven`].
pub(crate) const MBLEVEN_MAX: usize = 3;

/// Longest pattern eligible for the mbleven gate.  Bounds the gate's
/// hamming scan; beyond this the banded kernels already handle
/// near-identical strings well.
pub(crate) const MBLEVEN_MAX_LEN: usize = 512;

/// Gate for the mbleven fast path (same-kind slices): `Some(ub)` when
/// `length difference + aligned mismatches <= MBLEVEN_MAX`.
///
/// Adapted from SIMD string equality & mismatch scanning techniques (such as
/// simdjson / rapidfuzz vector classification): compares chunks in 32-byte
/// AVX2 blocks on x86-64 when available, or 16-byte SIMD vector blocks on targets
/// supporting NEON or SSE2, falling back to 8-byte scalar blocks. Only blocks
/// containing differences inspect the mismatch count, and aborts as soon as
/// `ub > MBLEVEN_MAX`.
#[inline(always)]
fn small_ub<X: Isa, T: CodeUnit>(short: &[T], long: &[T]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if X::X86_64_V2 {
        // SAFETY: the CPU runs `X` (see `Isa`).
        return unsafe { kernels::x86_64::v2::small_ub::<T>(short, long) };
    }
    small_ub_body::<T>(short, long)
}

/// The body of [`small_ub`] for baseline ISAs.
#[inline(always)]
pub(crate) fn small_ub_body<T: CodeUnit>(short: &[T], long: &[T]) -> Option<usize> {
    debug_assert!(short.len() <= long.len());
    let mut ub = long.len() - short.len();
    if ub > MBLEVEN_MAX || short.len() > MBLEVEN_MAX_LEN {
        return None;
    }
    let size = std::mem::size_of::<T>();
    let a = as_bytes(short);
    let b = &as_bytes(long)[..a.len()];
    let n = a.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    // SAFETY: `i + 16 <= n`, so reading 16 bytes starting at `a.as_ptr().add(i)` and
    // `b.as_ptr().add(i)` is within the bounds of `a` and `b`. NEON intrinsics are part
    // of the baseline AArch64 ISA.
    unsafe {
        use std::arch::aarch64::*;
        while i + 16 <= n {
            let va = vld1q_u8(a.as_ptr().add(i));
            let vb = vld1q_u8(b.as_ptr().add(i));
            let eq = vceqq_u8(va, vb);
            let neq = vmvnq_u8(eq);
            if vmaxvq_u8(neq) != 0 {
                let diff_lanes = match size {
                    1 => vaddlvq_u8(vshrq_n_u8(neq, 7)) as usize,
                    2 => {
                        let neq16 = vmvnq_u16(vceqq_u16(
                            vreinterpretq_u16_u8(va),
                            vreinterpretq_u16_u8(vb),
                        ));
                        vaddlvq_u16(vshrq_n_u16(neq16, 15)) as usize
                    }
                    4 => {
                        let neq32 = vmvnq_u32(vceqq_u32(
                            vreinterpretq_u32_u8(va),
                            vreinterpretq_u32_u8(vb),
                        ));
                        vaddlvq_u32(vshrq_n_u32(neq32, 31)) as usize
                    }
                    _ => unreachable!(),
                };
                ub += diff_lanes;
                if ub > MBLEVEN_MAX {
                    return None;
                }
            }
            i += 16;
        }
    }

    #[cfg(target_arch = "x86_64")]
    // SAFETY: `i + 16 <= n`, so unaligned 16-byte reads from `a.as_ptr().add(i)` and
    // `b.as_ptr().add(i)` are within bounds. SSE2 is part of the baseline x86-64 ISA.
    unsafe {
        use std::arch::x86_64::*;
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
                        if ub > MBLEVEN_MAX {
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
                        if ub > MBLEVEN_MAX {
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
                        if ub > MBLEVEN_MAX {
                            return None;
                        }
                    }
                }
                _ => unreachable!(),
            }
            i += 16;
        }
    }

    let lane_bits = 8 * size;
    let lane_mask = if lane_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << lane_bits) - 1
    };

    while i + 8 <= n {
        let Some(ca) = a[i..].first_chunk::<8>() else {
            break;
        };
        let Some(cb) = b[i..].first_chunk::<8>() else {
            break;
        };
        let mut d = u64::from_le_bytes(*ca) ^ u64::from_le_bytes(*cb);
        while d != 0 {
            ub += 1;
            if ub > MBLEVEN_MAX {
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
            if ub > MBLEVEN_MAX {
                return None;
            }
        }
    }
    Some(ub)
}

/// [`small_ub`] for mixed-kind slices (no byte-comparable representation):
/// mismatches are counted a block at a time, aborting between blocks.
#[inline(always)]
fn small_ub_mixed<T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2]) -> Option<usize> {
    debug_assert!(short.len() <= long.len());
    let mut ub = long.len() - short.len();
    if ub > MBLEVEN_MAX || short.len() > MBLEVEN_MAX_LEN {
        return None;
    }
    let blocks = short.len() / MIXED_BLOCK * MIXED_BLOCK;
    for i in (0..blocks).step_by(MIXED_BLOCK) {
        let Some(blk_s) = short[i..].first_chunk::<MIXED_BLOCK>() else {
            break;
        };
        let Some(blk_l) = long[i..].first_chunk::<MIXED_BLOCK>() else {
            break;
        };
        ub += block_mismatches(blk_s, blk_l);
        if ub > MBLEVEN_MAX {
            return None;
        }
    }
    for (x, y) in short[blocks..].iter().zip(long[blocks..].iter()) {
        if x.as_u64() != y.as_u64() {
            ub += 1;
            if ub > MBLEVEN_MAX {
                return None;
            }
        }
    }
    Some(ub)
}

/// [`mbleven`] in the copy `X` of the hot path: the function itself, or on
/// x86-64 its form in [`kernels::x86_64::v2`].  Unlike the other functions
/// that stay out of line, `mbleven` is left to LLVM to inline or not: an
/// `#[inline(never)]` wrapper that picks the copy changed its AArch64 code.
#[inline(always)]
fn mbleven_in<X: Isa, T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2], ub: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    if X::X86_64_V2 {
        // SAFETY: the CPU runs `X` (see `Isa`).
        return unsafe { kernels::x86_64::v2::mbleven::<X, T1, T2>(short, long, ub) };
    }
    mbleven::<X, T1, T2>(short, long, ub)
}

/// Common prefix of two code-unit slices.  Same-kind slices compare words via
/// [`common_prefix_bytes`]; mixed-kind slices compare code units linearly.
#[inline(always)]
fn common_prefix<T1: CodeUnit, T2: CodeUnit>(a: &[T1], b: &[T2]) -> usize {
    if std::mem::size_of::<T1>() == std::mem::size_of::<T2>() {
        let size = std::mem::size_of::<T1>();
        common_prefix_bytes(as_bytes(a), as_bytes(b)) / size
    } else {
        let n = a.len().min(b.len());
        let mut i = 0;
        while i < n && a[i].as_u64() == b[i].as_u64() {
            i += 1;
        }
        i
    }
}

/// Exact distance when an upper bound `ub` in `1..=3` is known (mbleven,
/// Hyyrö 2018 formulation as used by rapidfuzz): try every candidate edit-op
/// sequence of cost <= `ub` and take the cheapest that aligns the strings.
///
/// Each sequence packs ops two bits per op from the LSB: `0b01` advances only
/// `long` (deletion from the longer string), `0b10` advances only `short`
/// (insertion), `0b11` advances both (substitution).  Identical matching runs
/// are jumped via [`common_prefix`].  O(ub * n) worst case, no peq table, no
/// allocation.  Instantiated per copy `X` of the hot path and called through
/// [`mbleven_in`].
fn mbleven<X: Isa, T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2], ub: usize) -> usize {
    if ub == 0 {
        return 0;
    }
    debug_assert!((1..=MBLEVEN_MAX).contains(&ub));
    let diff = long.len() - short.len();
    debug_assert!(diff <= ub);

    // Rows indexed by (ub, diff): (1,0),(1,1),(2,0),(2,1),(2,2),(3,0)..(3,3).
    static SEQS: [&[u64]; 9] = [
        &[0x03],
        &[0x01],
        &[0x0F, 0x09, 0x06],
        &[0x0D, 0x07],
        &[0x05],
        &[0x3F, 0x27, 0x2D, 0x39, 0x36, 0x1E, 0x1B],
        &[0x3D, 0x37, 0x1F, 0x25, 0x19, 0x16],
        &[0x35, 0x1D, 0x17],
        &[0x15],
    ];
    let row = (ub - 1) * (ub + 2) / 2 + diff;

    let mut best = usize::MAX;
    for &seq in SEQS[row] {
        let mut ops = seq;
        let (mut sp, mut lp, mut cost) = (0usize, 0usize, 0usize);
        loop {
            let matched = common_prefix(&short[sp..], &long[lp..]);
            sp += matched;
            lp += matched;
            if sp == short.len() || lp == long.len() {
                break;
            }
            cost += 1;
            if ops == 0 {
                break;
            }
            lp += (ops & 1) as usize;
            sp += ((ops >> 1) & 1) as usize;
            ops >>= 2;
        }
        cost += (short.len() - sp) + (long.len() - lp);
        best = best.min(cost);
        if best == diff {
            return diff;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Generic pipelines
// ---------------------------------------------------------------------------

/// Strip affixes and run kernel `K` for same-kind strings (non-UCS-1).
#[inline(always)]
fn compute_sorted<K: Kernel, T: CodeUnit>(a: &[T], b: &[T]) -> usize {
    let (a, b) = strip_affix(a, b);
    if a.is_empty() {
        return b.len();
    }
    if a.len() <= TINY_M {
        return hyrro_64_tiny::<K, _, _>(a, b);
    }
    if K::MBLEVEN {
        if let Some(ub) = small_ub::<K::Isa, _>(a, b) {
            return mbleven_in::<K::Isa, _, _>(a, b, ub);
        }
    }
    if a.len() <= 64 {
        hyrro_64_hash::<K, _, _>(a, b)
    } else {
        hyrro_multiword_hash::<K, _, _>(a, b)
    }
}

/// Strip affixes and run kernel `K` for mixed-kind strings.  A pair with a
/// UCS-1 side can only match below U+0100, so multi-word patterns take the
/// direct-table kernel instead of hashing every text char; single-word ones
/// keep the hash, whose 1 KiB table is cheaper to set up than 257 rows.
#[inline(always)]
fn compute_sorted_mixed<K: Kernel, T1: CodeUnit, T2: CodeUnit>(a: &[T1], b: &[T2]) -> usize {
    let (a, b) = strip_affix_mixed(a, b);
    if a.is_empty() {
        return b.len();
    }
    if a.len() <= TINY_M {
        return hyrro_64_tiny::<K, _, _>(a, b);
    }
    if K::MBLEVEN {
        if let Some(ub) = small_ub_mixed(a, b) {
            return mbleven_in::<K::Isa, _, _>(a, b, ub);
        }
    }
    if a.len() <= 64 {
        hyrro_64_hash::<K, _, _>(a, b)
    } else if std::mem::size_of::<T1>() == 1 || std::mem::size_of::<T2>() == 1 {
        hyrro_multiword_direct::<K, 257, _, _>(a, b)
    } else {
        hyrro_multiword_hash::<K, _, _>(a, b)
    }
}

// ---------------------------------------------------------------------------
// Hyyrö single-word variants
// ---------------------------------------------------------------------------

/// Row of code unit `c` in the direct peq table of [`hyrro_multiword_direct`].  UCS-1 units
/// index it as they are (the caller guarantees they are below `ROWS`).
/// Wider units occur only on the wide side of a mixed-kind pair with a
/// UCS-1 side, where `ROWS` is 257: units above U+00FF can never match
/// there, so they share row 256, which the text side never fills.
#[inline(always)]
fn direct_row<T: CodeUnit, const ROWS: usize>(c: T) -> usize {
    if std::mem::size_of::<T>() == 1 {
        c.as_u64() as usize
    } else {
        c.as_u64().min(ROWS as u64 - 1) as usize
    }
}

/// Single-word variant for byte patterns (UCS-1).
#[inline(always)]
fn hyrro_64_u8<K: Kernel, const SLOTS: usize>(pattern: &[u8], text: &[u8]) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let mut peq = [0u64; SLOTS];
    for (i, &c) in pattern.iter().enumerate() {
        // SAFETY: caller guarantees c < SLOTS (128 or 256).
        unsafe {
            *peq.get_unchecked_mut(c as usize) |= 1u64 << i;
        }
    }
    K::word(
        pattern.len(),
        text.len(),
        text.iter().map(|&c| {
            // SAFETY: When SLOTS == 128, input is verified ASCII (c < 128);
            // when SLOTS == 256, any u8 is < 256.
            unsafe { *peq.get_unchecked(c as usize) }
        }),
    )
}

/// Single-word variant for UCS-2/4 and mixed kinds, with the peq table in a
/// stack-allocated open-addressing hash.
#[inline(always)]
fn hyrro_64_hash<K: Kernel, T1: CodeUnit, T2: CodeUnit>(pattern: &[T1], text: &[T2]) -> usize {
    let m = pattern.len();
    debug_assert!((1..=64).contains(&m));

    const SLOTS: usize = 128;
    const MASK: usize = SLOTS - 1;
    // A slot's key is written before its value becomes non-zero and read only
    // after, so `vals` alone marks occupancy and `keys` needs no fill.
    let mut keys = [const { MaybeUninit::<T1>::uninit() }; SLOTS];
    let mut vals = [0u64; SLOTS];

    let shift = 64 - MASK.count_ones();
    for (i, &c) in pattern.iter().enumerate() {
        let mut slot = hslot(c.as_u64(), shift);
        loop {
            if vals[slot] == 0 {
                keys[slot].write(c);
                vals[slot] = 1u64 << i;
                break;
            }
            // SAFETY: vals[slot] != 0, so keys[slot] was written.
            if unsafe { keys[slot].assume_init() } == c {
                vals[slot] |= 1u64 << i;
                break;
            }
            slot = (slot + 1) & MASK;
        }
    }

    K::word(
        m,
        text.len(),
        text.iter().map(|&c| {
            let c = c.as_u64();
            let mut slot = hslot(c, shift);
            loop {
                // SAFETY: `slot` is produced by `hslot` and masked with `MASK = SLOTS - 1`,
                // ensuring `slot < SLOTS` is invariant.
                let v = unsafe { *vals.get_unchecked(slot) };
                if v == 0 {
                    return 0;
                }
                // SAFETY: `v != 0`, meaning `keys[slot]` was initialized during table setup,
                // and `slot < SLOTS` as established above.
                if unsafe { keys.get_unchecked(slot).assume_init().as_u64() } == c {
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

/// Core bit-parallel loop for a pattern of `m <= 64` rows (Myers 1999 in
/// Hyyrö's global form: the top boundary shifts in `+1`).
///
/// Bits above row `m - 1` are never masked off: every step is bitwise, a left
/// shift, or an addition, so information only flows towards higher bits and
/// garbage above the pattern never reaches the pattern rows.
///
/// The step is arranged for a short loop-carried chain: `vp & vn == 0` keeps
/// `vn` out of the addition, and `(sum ^ vp) | vp == sum | vp` lets `ph` skip
/// Myers' `xh`.  No score is tracked per column: `vp`/`vn` hold the vertical
/// deltas of the current column and `D(0, j) = j`, so the distance after the
/// last of the `n` columns is `n + popcount(vp) - popcount(vn)` over the
/// pattern rows.
#[inline(always)]
fn hyrro_inner<I: Iterator<Item = u64>>(m: usize, n: usize, pm_iter: I) -> usize {
    debug_assert!((1..=64).contains(&m));
    let (mut vp, mut vn) = (!0u64, 0u64);
    for eq in pm_iter {
        (vp, vn) = kernels::myers_step(eq, vp, vn);
    }
    let rows = u64::MAX >> (64 - m);
    n + (vp & rows).count_ones() as usize - (vn & rows).count_ones() as usize
}

// ---------------------------------------------------------------------------
// Bit-parallel LCS (Crochemore et al. 2001 / Hyyrö 2004)
// ---------------------------------------------------------------------------
//
// The state `s` is one bit per pattern row: a *zero* at row `i` marks a column
// position where the LCS length increases, so `LLCS = popcount(!s)` over the
// pattern's bits.  Starting from all-ones (column 0 has no increments) the
// column step is
//
//     u = s & pm;  s = (s + u) | (s - u)
//
// where `s - u == s & !pm` because `u` is a submask of `s`, so the subtraction
// never borrows.  Bits above the pattern length are never set in `pm`, so the
// `| (s - u)` term always restores them to one; a carry from the addition can
// disturb them only transiently and they never contribute to the popcount.
// Over several words the addition is one big-integer addition, so the carry
// is threaded from the low word upwards; `s - u` stays word-local.

/// One column of the LCS recurrence over the words `s`, with `pm(k)` giving
/// the pattern-match bits of word `k`; returns the outgoing carry.
#[inline(always)]
fn lcs_step(s: &mut [u64], mut carry: bool, pm: impl Fn(usize) -> u64) -> bool {
    for (k, sk) in s.iter_mut().enumerate() {
        let u = *sk & pm(k);
        let (sum, nc) = sk.carrying_add(u, carry);
        carry = nc;
        *sk = sum | (*sk - u);
    }
    carry
}

/// LCS length encoded in the state words `s`.
#[inline(always)]
fn lcs_len(s: &[u64]) -> usize {
    s.iter().map(|&x| (!x).count_ones() as usize).sum()
}

/// Single-word LCS loop; returns the LCS length.
#[inline(always)]
fn lcs_inner_64<I: Iterator<Item = u64>>(pm_iter: I) -> usize {
    let mut s = [!0u64];
    for pm in pm_iter {
        lcs_step(&mut s, false, |_| pm);
    }
    lcs_len(&s)
}

/// Multi-word LCS over `W` words; returns their share of the LCS length.
/// With `carries`, the kernel is one strip of a taller pattern (see
/// [`kernels::lcs_full`]): it reads each column's incoming carry from there
/// and writes the outgoing one back.
#[inline(always)]
fn lcs_multiword_kernel<const W: usize, I: Iterator<Item = [u64; W]>>(
    pm_iter: I,
    mut carries: Option<&mut [u8]>,
) -> usize {
    let mut s = [!0u64; W];
    for (j, pm_row) in pm_iter.enumerate() {
        let carry_in = carries.as_ref().is_some_and(|c| c[j] != 0);
        let carry = lcs_step(&mut s, carry_in, |k| pm_row[k]);
        if let Some(c) = carries.as_deref_mut() {
            c[j] = carry as u8;
        }
    }
    lcs_len(&s)
}

// ---------------------------------------------------------------------------
// Kernel selection: Levenshtein vs. indel (LCS) distance
// ---------------------------------------------------------------------------

/// Chooses the bit-parallel recurrence run by the shared peq-building
/// pipelines.  Every method returns an *edit distance*, so the pipelines,
/// their affix stripping, and their empty-pattern shortcut are identical for
/// both metrics (indel distance is affix-invariant for the same reason
/// Levenshtein distance is).
trait Kernel {
    /// The copy of the hot path the pipeline runs in.
    type Isa: Isa;

    /// Whether the mbleven fast path applies; it is Levenshtein-specific.
    const MBLEVEN: bool;

    /// Single-word kernel (`m <= 64`).
    fn word<I: Iterator<Item = u64>>(m: usize, n: usize, pm_iter: I) -> usize;

    /// Multi-word kernel (`64 < m <= 64 * MAX_W`), unrolled over
    /// `W = ceil(m / 64)`.
    fn words<const W: usize, I: Iterator<Item = [u64; W]>>(m: usize, n: usize, pm_iter: I)
        -> usize;

    /// Heap-backed kernel for very long patterns (`w > MAX_W`).
    fn large<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize;
}

/// Levenshtein distance (substitution, insertion, and deletion each cost 1),
/// in the copy `X` of the hot path.
struct Lev<X = Baseline>(PhantomData<X>);

impl<X: Isa> Kernel for Lev<X> {
    type Isa = X;

    const MBLEVEN: bool = true;

    #[inline(always)]
    fn word<I: Iterator<Item = u64>>(m: usize, n: usize, pm_iter: I) -> usize {
        hyrro_inner(m, n, pm_iter)
    }

    #[inline(always)]
    fn words<const W: usize, I: Iterator<Item = [u64; W]>>(
        m: usize,
        n: usize,
        pm_iter: I,
    ) -> usize {
        (n as isize + kernels::multiword::<X, W, I>(m, pm_iter, None)) as usize
    }

    #[inline(always)]
    fn large<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
        // In a copy with target features, LLVM keeps `run` (which has none)
        // out of line, so its body is inlined by hand.
        #[cfg(target_arch = "x86_64")]
        if X::X86_64_V2 {
            return ctx.run_body::<X>();
        }
        ctx.run::<X>()
    }
}

/// Indel distance: insertions and deletions cost 1, substitution is not an
/// operation (it decomposes into one of each, cost 2).  Derived from the LCS
/// length as `m + n - 2 * LLCS`.  In the copy `X` of the hot path.
struct Lcs<X = Baseline>(PhantomData<X>);

impl<X: Isa> Kernel for Lcs<X> {
    type Isa = X;

    const MBLEVEN: bool = false;

    #[inline(always)]
    fn word<I: Iterator<Item = u64>>(m: usize, n: usize, pm_iter: I) -> usize {
        m + n - 2 * lcs_inner_64(pm_iter)
    }

    #[inline(always)]
    fn words<const W: usize, I: Iterator<Item = [u64; W]>>(
        m: usize,
        n: usize,
        pm_iter: I,
    ) -> usize {
        m + n - 2 * lcs_multiword_kernel::<W, I>(pm_iter, None)
    }

    #[inline(always)]
    fn large<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
        ctx.m + ctx.n - 2 * ctx.lcs_run::<X>()
    }
}

// ---------------------------------------------------------------------------
// Multi-word Hyyrö (m > 64)
// ---------------------------------------------------------------------------

/// Widest pattern, in 64-bit words, handled by the const-width kernels with
/// stack peq tables; longer patterns go through [`LargeCtx`].
const MAX_W: usize = 8;

/// Multi-word entry point with a direct peq table of `ROWS` rows: UCS-1
/// pairs (128 rows for pure ASCII, 256 for Latin-1) and mixed-kind pairs
/// with a UCS-1 side (257 rows, see [`direct_row`]).  Each copy of the hot
/// path compiles its own (see [`Isa`]).
#[inline(never)]
fn hyrro_multiword_direct<K: Kernel, const ROWS: usize, T1: CodeUnit, T2: CodeUnit>(
    short: &[T1],
    long: &[T2],
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if K::Isa::X86_64_V2 {
        // SAFETY: the CPU runs `K::Isa` (see `Isa`).
        return unsafe {
            kernels::x86_64::v2::hyrro_multiword_direct::<K, ROWS, T1, T2>(short, long)
        };
    }
    hyrro_multiword_direct_body::<K, ROWS, T1, T2>(short, long)
}

/// The body of [`hyrro_multiword_direct`].
#[inline(always)]
fn hyrro_multiword_direct_body<K: Kernel, const ROWS: usize, T1: CodeUnit, T2: CodeUnit>(
    short: &[T1],
    long: &[T2],
) -> usize {
    debug_assert!(short.len() > 64);
    let m = short.len();
    let w = m.div_ceil(64);

    macro_rules! run {
        ($W:literal) => {{
            // An array-of-arrays literal is zeroed row by row in a loop;
            // `zeroed` lowers to one `bzero` call.
            // SAFETY: all-zero bytes are a valid `[[u64; W]; ROWS]`.
            let mut peq: [[u64; $W]; ROWS] = unsafe { std::mem::zeroed() };
            for (i, &c) in short.iter().enumerate() {
                // SAFETY: direct_row < ROWS and i / 64 < $W.
                unsafe {
                    *peq.get_unchecked_mut(direct_row::<_, ROWS>(c))
                        .get_unchecked_mut(i / 64) |= 1u64 << (i % 64);
                }
            }
            K::words::<$W, _>(
                m,
                long.len(),
                // SAFETY: direct_row < ROWS.
                long.iter()
                    .map(|&c| unsafe { *peq.get_unchecked(direct_row::<_, ROWS>(c)) }),
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
            let mut peq = vec![0u64; ROWS * w];
            for (i, &c) in short.iter().enumerate() {
                peq[direct_row::<_, ROWS>(c) * w + i / 64] |= 1u64 << (i % 64);
            }
            K::large(&LargeCtx {
                m,
                n: long.len(),
                w,
                data: &peq,
                // SAFETY: the kernels only ask for j < n.
                base_of: |j: usize| direct_row::<_, ROWS>(unsafe { *long.get_unchecked(j) }) * w,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Banded multi-word kernels (w > MAX_W)
// ---------------------------------------------------------------------------

/// Band parameter for the next pass after a band of parameter `t` gave up at
/// text column `j` of `n`.  The band's bound on the distance grows about
/// linearly with `j`, so extrapolating it to `n` (with a margin) estimates
/// the distance; at least double `t` so that the passes grow geometrically.
#[inline(always)]
fn next_band(t: usize, j: usize, n: usize) -> usize {
    (t * n / j * 5 / 4).max(2 * t)
}

/// Number of 64-row blocks a Levenshtein band of half-width `t` can touch in
/// one column.
#[inline(always)]
fn band_blocks(t: usize) -> usize {
    2 * t / 64 + 2
}

/// Number of words an LCS band of parameter `t` (diagonals `[-t, slack + t]`)
/// can touch in one column.
#[inline(always)]
fn lcs_band_words(slack: usize, t: usize) -> usize {
    (slack + 2 * t) / 64 + 2
}

/// Shared context for the very-long-pattern kernels (w > MAX_W): `data` is
/// the flat peq table with stride `w`, and `base_of(j)` is the offset of
/// text char `j`'s row in it.
struct LargeCtx<'a, F> {
    m: usize,
    n: usize,
    w: usize,
    data: &'a [u64],
    base_of: F,
}

impl<F: Fn(usize) -> usize> LargeCtx<'_, F> {
    /// Copy of `self` that reads row offsets from `bases`, one per text char.
    /// Resolving rows once pays off when the band passes sweep the text
    /// several times; `bases` costs a word per text char, so only build it
    /// for those.
    fn with_bases<'b>(&'b self, bases: &'b [usize]) -> LargeCtx<'b, impl Fn(usize) -> usize + 'b> {
        debug_assert_eq!(bases.len(), self.n);
        LargeCtx {
            m: self.m,
            n: self.n,
            w: self.w,
            data: self.data,
            // SAFETY: the kernels only ask for j < n = bases.len().
            base_of: move |j: usize| unsafe { *bases.get_unchecked(j) },
        }
    }

    /// Calls `f` with the row offset of each text char in order.  The
    /// offsets are resolved a chunk at a time into a stack buffer, so the
    /// lookups (hash probes for UCS-2/4) stay out of the kernel's serial
    /// loop without a buffer the size of the text.
    #[inline(always)]
    fn for_each_base(&self, mut f: impl FnMut(usize)) {
        const CHUNK: usize = 512;
        let mut buf = [0usize; CHUNK];
        for start in (0..self.n).step_by(CHUNK) {
            let chunk = &mut buf[..(self.n - start).min(CHUNK)];
            for (j, b) in (start..).zip(chunk.iter_mut()) {
                *b = (self.base_of)(j);
            }
            chunk.iter().for_each(|&base| f(base));
        }
    }

    /// Peq word `k` of the row starting at `base`.
    #[inline(always)]
    fn peq(&self, base: usize, k: usize) -> u64 {
        debug_assert!(k < self.w && base + k < self.data.len());
        // SAFETY: every base starts a length-w row of `data`, and k < w.
        unsafe { *self.data.get_unchecked(base + k) }
    }

    /// Ukkonen band of half-width `t` over Myers' blocked recurrence: blocks
    /// exchange only the horizontal delta `hin`/`hout` (Edlib's formulation),
    /// so per column only blocks intersecting rows `[j - t, j + t]` run.
    ///
    /// Out-of-band boundaries are pessimistic (`hin = +1` below a dropped
    /// block; `+1`-per-row extension above entering blocks), so every computed
    /// cell is `>=` its true value, while cells on any optimal path are exact
    /// whenever the true distance is `<= t` (such paths never leave the band).
    /// Returns `Ok(distance)` when the result proves itself (`score <= t`),
    /// else `Err(j)` with the text column where the band gave up.
    ///
    /// Kept out of line: inlined into the large entry functions, the block
    /// loop reloads the `pv`, `mv` and peq pointers from the stack on x86-64
    /// and loses the branch around the score update on AArch64.  Each copy
    /// `X` of the hot path compiles its own (see [`Isa`]).
    #[inline(never)]
    fn banded<X: Isa>(&self, t: usize, pv: &mut [u64], mv: &mut [u64]) -> Result<usize, usize> {
        #[cfg(target_arch = "x86_64")]
        if X::X86_64_V2 {
            // SAFETY: the CPU runs `X` (see `Isa`).
            return unsafe { kernels::x86_64::v2::banded(self, t, pv, mv) };
        }
        self.banded_body(t, pv, mv)
    }

    /// The body of [`Self::banded`].
    #[inline(always)]
    fn banded_body(&self, t: usize, pv: &mut [u64], mv: &mut [u64]) -> Result<usize, usize> {
        let (m, n, w) = (self.m, self.n, self.w);
        debug_assert!(n >= m);
        debug_assert!(t >= n - m);
        let (pv, mv) = (&mut pv[..w], &mut mv[..w]);
        // Lowest row covered when `k` is the last live block.
        let bottom = |k: usize| (64 * (k + 1)).min(m);
        let mut last = (t / 64).min(w - 1);
        let mut first = 0usize;
        pv[..=last].fill(!0);
        mv[..=last].fill(0);
        let mut score = bottom(last) as isize;
        let mut top_bit = (bottom(last) - 1) & 63;

        for j in 1..=n {
            let base = (self.base_of)(j - 1);
            let needed = ((j + t - 1) / 64).min(w - 1);
            if needed > last {
                pv[last + 1..=needed].fill(!0);
                mv[last + 1..=needed].fill(0);
                score += (bottom(needed) - bottom(last)) as isize;
                last = needed;
                top_bit = (bottom(last) - 1) & 63;
            }
            if j > t + 1 {
                first = ((j - t - 1) / 64).min(last);
            }
            let mut hin: i32 = 1;
            debug_assert!(first <= last && last < w);
            for k in first..=last {
                let eq = self.peq(base, k);
                // SAFETY: k <= last < w = pv.len() = mv.len().
                let (pv_k, mv_k) = unsafe { (*pv.get_unchecked(k), *mv.get_unchecked(k)) };
                let xv = eq | mv_k;
                let eq_in = eq | ((hin < 0) as u64);
                let xh = ((eq_in & pv_k).wrapping_add(pv_k) ^ pv_k) | eq_in;
                let ph = mv_k | !(xh | pv_k);
                let mh = pv_k & xh;
                if k == last {
                    score += ((ph >> top_bit) & 1) as isize - ((mh >> top_bit) & 1) as isize;
                }
                let hout = ((ph >> 63) & 1) as i32 - ((mh >> 63) & 1) as i32;
                let ph_s = (ph << 1) | ((hin > 0) as u64);
                let mh_s = (mh << 1) | ((hin < 0) as u64);
                // SAFETY: as for the reads above.
                unsafe {
                    *pv.get_unchecked_mut(k) = mh_s | !(xv | ph_s);
                    *mv.get_unchecked_mut(k) = ph_s & xv;
                }
                hin = hout;
            }
            // `score` never underestimates D[bottom(last)][j], and finishing
            // from there takes at least `score - excess` edits (D never
            // decreases along diagonal steps); once that exceeds `t` this band
            // can no longer prove a result, so bail out and let the caller
            // widen or fall back.
            let excess = (n - j) as isize - (m - bottom(last)) as isize;
            if score - excess.max(0) > t as isize {
                return Err(j);
            }
        }
        debug_assert_eq!(last, w - 1);
        if score as usize <= t {
            Ok(score as usize)
        } else {
            Err(n)
        }
    }

    /// Peq rows restricted to words `w0..w0 + W`, one per text char.
    #[inline(always)]
    fn rows<const W: usize>(&self, w0: usize) -> impl Iterator<Item = [u64; W]> + '_ {
        debug_assert!(w0 + W <= self.w);
        (0..self.n).map(move |j| {
            let base = (self.base_of)(j);
            std::array::from_fn(|k| self.peq(base, w0 + k))
        })
    }

    /// Bands of growing width before the full matrix.  Each pass aborts as
    /// soon as it can no longer prove a result, so a miss on dissimilar
    /// strings costs a few percent of the full kernel.  The bands run in
    /// the copy `X` of the hot path.
    fn run<X: Isa>(&self) -> usize {
        self.run_body::<X>()
    }

    /// The body of [`Self::run`], which the copies other than [`Baseline`]
    /// inline (see [`Lev::large`]).
    #[inline(always)]
    fn run_body<X: Isa>(&self) -> usize {
        let mut t = (self.n - self.m + 32).max(64);
        if band_blocks(t) * 3 > self.w {
            return kernels::full(self);
        }
        let bases: Vec<usize> = (0..self.n).map(&self.base_of).collect();
        let ctx = self.with_bases(&bases);
        let (mut pv, mut mv) = (vec![0u64; self.w], vec![0u64; self.w]);
        while band_blocks(t) * 3 <= self.w {
            match ctx.banded::<X>(t, &mut pv, &mut mv) {
                Ok(d) => return d,
                Err(j) => t = next_band(t, j, self.n),
            }
        }
        kernels::full(&ctx)
    }

    /// LCS over the matches on diagonals `j - i` in `[-t, n - m + t]`, rows
    /// widened to whole words, computed by running only the words that
    /// intersect that band in each column.  Returns `Ok(lcs)` when the band
    /// certifies it, else `Err(j)` with the text column where it gave up.
    ///
    /// Skipping a word is the same as zeroing its pattern-match bits for that
    /// column: words below the band are still all ones, and all-ones words
    /// without matches only pass the carry through; words above it, without
    /// matches and with no carry from row 0, stay unchanged and emit no
    /// carry.  The band only moves down, so `l` is the exact LCS over a
    /// superset of the in-band matches, hence `l <= LLCS`.  An alignment using
    /// any match off the band costs more than `n - m + 2t` indels, so `l` is
    /// exact whenever `m + n - 2l <= n - m + 2t`.
    fn lcs_banded(&self, t: usize, s: &mut [u64]) -> Result<usize, usize> {
        let (m, n, w) = (self.m, self.n, self.w);
        debug_assert!(n >= m);
        let slack = n - m;
        let s = &mut s[..w];
        s.fill(!0);
        for j in 1..=n {
            let base = (self.base_of)(j - 1);
            // Rows of the band in column j: `[j - 1 - slack - t, j - 1 + t]`.
            let first = j.saturating_sub(slack + t + 1) / 64;
            let last = ((j + t - 1) / 64).min(w - 1);
            lcs_step(&mut s[first..=last], false, |k| self.peq(base, first + k));
            if j % 32 == 0 && self.lcs_band_hopeless(j, t, s) {
                return Err(j);
            }
        }
        let l = lcs_len(s);
        if m + n - 2 * l <= slack + 2 * t {
            Ok(l)
        } else {
            Err(n)
        }
    }

    /// Whether the band pass can no longer certify (see [`Self::lcs_banded`]).
    ///
    /// Every alignment crosses column j at some row r and then costs at least
    /// `D(r, j) + |(m - r) - (n - j)|`, where `D(r, j) = r + j - 2 * P(r)` and
    /// `P(r)` counts LCS increments in rows < r.  As P grows by at most one
    /// per row, the minimum over r sits at `r = j - slack`, where the bound
    /// is `2j - slack - 2 * P(j - slack)`; for `j < slack` it sits at row 0
    /// and is just `slack`, which never exceeds the limit.
    fn lcs_band_hopeless(&self, j: usize, t: usize, s: &[u64]) -> bool {
        let slack = self.n - self.m;
        let Some(c) = j.checked_sub(slack) else {
            return false;
        };
        let (full, part) = (c / 64, c % 64);
        let mut p = lcs_len(&s[..full]);
        if part != 0 {
            p += (!s[full] & ((1u64 << part) - 1)).count_ones() as usize;
        }
        // p <= c, so this cannot underflow.
        2 * j - slack - 2 * p > slack + 2 * t
    }

    /// LCS length via bands of growing width, like [`Self::run`].
    fn lcs_run<X: Isa>(&self) -> usize {
        let slack = self.n - self.m;
        let mut t = 64;
        if lcs_band_words(slack, t) * 3 > self.w {
            return kernels::lcs_full::<X, _>(self);
        }
        let bases: Vec<usize> = (0..self.n).map(&self.base_of).collect();
        let ctx = self.with_bases(&bases);
        let mut s = vec![0u64; self.w];
        while lcs_band_words(slack, t) * 3 <= self.w {
            match ctx.lcs_banded(t, &mut s) {
                Ok(l) => return l,
                Err(j) => t = next_band(t, j, self.n),
            }
        }
        kernels::lcs_full::<X, _>(&ctx)
    }
}

/// `len` copies of `value`, in `stack` when they fit and in `heap` otherwise;
/// only the used prefix of `stack` is initialized.
#[inline(always)]
fn scratch<'a, const N: usize>(
    stack: &'a mut [MaybeUninit<u64>; N],
    heap: &'a mut Vec<u64>,
    len: usize,
    value: u64,
) -> &'a mut [u64] {
    if len <= N {
        let buf = &mut stack[..len];
        for x in buf.iter_mut() {
            x.write(value);
        }
        // SAFETY: All `len` elements in `buf` were initialized above with `x.write(value)`.
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u64>(), len) }
    } else {
        *heap = vec![value; len];
        heap
    }
}

/// Multi-word entry point for UCS-2/4 and mixed kinds: the peq rows are
/// indexed through an open-addressing hash of the pattern alphabet.  Each
/// copy of the hot path compiles its own (see [`Isa`]).
#[inline(never)]
fn hyrro_multiword_hash<K: Kernel, T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if K::Isa::X86_64_V2 {
        // SAFETY: the CPU runs `K::Isa` (see `Isa`).
        return unsafe { kernels::x86_64::v2::hyrro_multiword_hash::<K, T1, T2>(short, long) };
    }
    hyrro_multiword_hash_body::<K, T1, T2>(short, long)
}

/// The body of [`hyrro_multiword_hash`].
#[inline(always)]
fn hyrro_multiword_hash_body<K: Kernel, T1: CodeUnit, T2: CodeUnit>(
    short: &[T1],
    long: &[T2],
) -> usize {
    debug_assert!(short.len() > 64);
    let m = short.len();
    let w = m.div_ceil(64);

    // Dense-index the pattern alphabet in first-seen order with a single
    // open-addressing pass; each live entry packs (dense index << 32) | key.
    // Code units fit in 32 bits and the dense index is bounded by the string
    // length, so no live entry can equal the EMPTY sentinel.
    const EMPTY: u64 = u64::MAX;
    const STACK_M: usize = 64 * MAX_W;
    let hash_size = (m * 2).next_power_of_two();
    let hash_mask = hash_size - 1;
    let hshift = 64 - hash_mask.count_ones();
    // Both tables fit on the stack for every const-width kernel; two heap
    // round trips would cost as much as the kernel on ~100 chars.
    let (mut hash_stack, mut hash_heap) = ([MaybeUninit::uninit(); 2 * STACK_M], Vec::new());
    let hash = scratch(&mut hash_stack, &mut hash_heap, hash_size, EMPTY);
    let mut ki_stack = [0u16; STACK_M];
    let mut ki_heap = Vec::new();
    let kis: &mut [u16] = if m <= STACK_M {
        &mut ki_stack[..m]
    } else {
        ki_heap.resize(m, 0);
        &mut ki_heap[..]
    };
    let mut n_keys = 0u64;
    for (i, &c) in short.iter().enumerate() {
        let key = c.as_u64();
        debug_assert!(
            key <= u64::from(u32::MAX),
            "CodeUnit value must fit in 32 bits"
        );
        let mut slot = hslot(key, hshift);
        loop {
            let entry = hash[slot];
            if entry == EMPTY {
                let ki = n_keys;
                hash[slot] = ki << 32 | key;
                kis[i] = ki as u16;
                n_keys += 1;
                break;
            }
            if entry & 0xFFFF_FFFF == key {
                kis[i] = (entry >> 32) as u16;
                break;
            }
            slot = (slot + 1) & hash_mask;
        }
    }
    let n_keys = n_keys as usize;
    let hash = &*hash;

    // Dense row of a code unit; `n_keys` for units absent from the pattern.
    let row_of = |key: u64| -> usize {
        let mut slot = hslot(key, hshift);
        loop {
            // SAFETY: `slot` is produced by `hslot` and masked with `hash_mask = hash_slots - 1`,
            // ensuring `slot < hash.len()` is invariant.
            let entry = unsafe { *hash.get_unchecked(slot) };
            if entry == EMPTY {
                return n_keys;
            }
            if entry & 0xFFFF_FFFF == key {
                return (entry >> 32) as usize;
            }
            slot = (slot + 1) & hash_mask;
        }
    };

    // Rows sized by the actual alphabet; the trailing row stays all-zero and
    // serves text chars that do not occur in the pattern.
    let (mut data_stack, mut data_heap) =
        ([MaybeUninit::uninit(); (STACK_M + 1) * MAX_W], Vec::new());
    let data = scratch(&mut data_stack, &mut data_heap, (n_keys + 1) * w, 0);
    for (i, &ki) in kis.iter().enumerate() {
        let ki = ki as usize;
        debug_assert!(ki < n_keys, "pattern key must already be inserted");
        data[ki * w + i / 64] |= 1u64 << (i % 64);
    }

    macro_rules! run {
        ($W:literal) => {{
            let (rows, _) = data.as_chunks::<$W>();
            K::words::<$W, _>(
                m,
                long.len(),
                // SAFETY: row_of returns <= n_keys < rows.len().
                long.iter()
                    .map(|&c| unsafe { *rows.get_unchecked(row_of(c.as_u64())) }),
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
            K::large(&LargeCtx {
                m,
                n: long.len(),
                w,
                data,
                // SAFETY: the kernels only ask for j < n.
                base_of: |j: usize| row_of(unsafe { long.get_unchecked(j) }.as_u64()) * w,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Test helpers (bypass the PyO3 layer)
// ---------------------------------------------------------------------------

#[cfg(test)]
fn metric<K: Kernel>(s1: &str, s2: &str) -> usize {
    // Normalize: s1 is shorter pattern, s2 is longer text.
    let (s1, s2) = if s1.chars().count() <= s2.chars().count() {
        (s1, s2)
    } else {
        (s2, s1)
    };
    if s1.is_ascii() && s2.is_ascii() {
        compute_u8::<K, true>(s1.as_bytes(), s2.as_bytes())
    } else {
        let a: Vec<u32> = s1.chars().map(|c| c as u32).collect();
        let b: Vec<u32> = s2.chars().map(|c| c as u32).collect();
        compute_sorted::<K, _>(&a, &b)
    }
}

#[cfg(test)]
fn levenshtein(s1: &str, s2: &str) -> usize {
    metric::<Lev>(s1, s2)
}

#[cfg(test)]
fn indel(s1: &str, s2: &str) -> usize {
    metric::<Lcs>(s1, s2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference O(m·n) DP — small, obviously correct, used as oracle.
    pub(crate) fn naive(a: &[char], b: &[char]) -> usize {
        let (m, n) = (a.len(), b.len());
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, val) in dp[0].iter_mut().enumerate() {
            *val = j;
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

    /// Exhaustive mbleven validation: every pair over a binary alphabet with
    /// lengths (9..=11, 9..=11) whose gate bound qualifies must match the
    /// oracle exactly.
    #[test]
    fn mbleven_matches_oracle_exhaustive() {
        let strings = |len: usize| -> Vec<Vec<u8>> {
            (0u32..1 << len)
                .map(|bits| (0..len).map(|i| b'a' + ((bits >> i) & 1) as u8).collect())
                .collect()
        };
        let mut hits = 0usize;
        for la in 9..=11usize {
            for lb in la..=11usize {
                for a in strings(la) {
                    for b in strings(lb) {
                        if let Some(ub) = small_ub::<Baseline, _>(&a, &b) {
                            if ub == 0 {
                                continue;
                            }
                            let ac: Vec<char> = a.iter().map(|&c| c as char).collect();
                            let bc: Vec<char> = b.iter().map(|&c| c as char).collect();
                            assert_eq!(
                                mbleven::<Baseline, _, _>(&a, &b, ub),
                                naive(&ac, &bc),
                                "({a:?}, {b:?}, ub={ub})"
                            );
                            hits += 1;
                        }
                    }
                }
            }
        }
        assert!(hits > 100_000, "gate should trigger often here: {hits}");
    }

    /// Integration: random small-edit pairs (which exercise the small_ub gate
    /// through compute) agree with the oracle across lengths and alphabets.
    #[test]
    fn small_edit_pairs_match_oracle() {
        let mut state = 0x243F_6A88_85A3_08D3u64; // xorshift, deterministic
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let alphabets: [&[char]; 3] = [
            &['a', 'b', 'c', 'd', 'e', 'f'],
            &['é', 'ü', 'ø', 'å'],
            &['日', '本', '語', 'あ', 'い'],
        ];
        for _ in 0..4000 {
            let ab = alphabets[(rng() % 3) as usize];
            let n = 9 + (rng() % 120) as usize;
            let a: Vec<char> = (0..n)
                .map(|_| ab[(rng() % ab.len() as u64) as usize])
                .collect();
            let mut b = a.clone();
            for _ in 0..(rng() % 4) {
                match rng() % 3 {
                    0 if !b.is_empty() => {
                        let i = (rng() % b.len() as u64) as usize;
                        b[i] = ab[(rng() % ab.len() as u64) as usize];
                    }
                    1 if !b.is_empty() => {
                        b.remove((rng() % b.len() as u64) as usize);
                    }
                    _ => {
                        let i = (rng() % (b.len() as u64 + 1)) as usize;
                        b.insert(i, ab[(rng() % ab.len() as u64) as usize]);
                    }
                }
            }
            let sa: String = a.iter().collect();
            let sb: String = b.iter().collect();
            assert_eq!(levenshtein(&sa, &sb), naive(&a, &b), "({sa:?}, {sb:?})");
        }
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

    /// Directly exercise `hyrro_multiword_direct` (m > 64) against the naive DP.
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
            assert_eq!(hyrro_multiword_direct::<Lev, 256, _, _>(s, l), oracle(s, l));
        };
        // Latin-1 bytes: any u8 value allowed.
        let check_latin1 = |a: &[u8], b: &[u8]| {
            let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            assert!(s.len() > 64);
            assert_eq!(hyrro_multiword_direct::<Lev, 256, _, _>(s, l), oracle(s, l));
        };

        // Benchmark-like: two long ASCII strings that diverge after a shared prefix.
        let s1 = b"Lets pretend Marshall Mathers never picked up a pen".repeat(8);
        let s2 = b"Lets pretend things woulda been no different".repeat(8);
        let (sh, lo) = if s1.len() <= s2.len() {
            (&s1[..], &s2[..])
        } else {
            (&s2[..], &s1[..])
        };
        assert_eq!(
            hyrro_multiword_direct::<Lev, 256, _, _>(sh, lo),
            oracle(sh, lo)
        );

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
        check_latin1(&[200u8; 80], &[201u8; 80]);
    }

    /// Mixed-kind pairs with a UCS-1 side take the direct-table kernel with
    /// wide units folded into row 256; check both orientations, including a
    /// wide pattern whose units above U+00FF land in that row.
    #[test]
    fn mixed_direct_matches_oracle() {
        let mut state = 0x0123_4567_89AB_CDEF_u64;
        let narrow = ['a', 'b', 'é', '\u{ff}', ' '];
        let wide = ['a', 'é', '\u{100}', '\u{2019}', '\u{ffff}'];
        let mut gen = |alpha: &[char], len: usize| -> Vec<char> {
            (0..len)
                .map(|_| alpha[lcg(&mut state) % alpha.len()])
                .collect()
        };
        for &(m, n) in &[(65usize, 65usize), (100, 130), (300, 300), (520, 700)] {
            // UCS-1 pattern, UCS-2 text.
            let (a, b) = (gen(&narrow, m), gen(&wide, n));
            let a8: Vec<u8> = a.iter().map(|&c| c as u8).collect();
            let b16: Vec<u16> = b.iter().map(|&c| c as u16).collect();
            assert_eq!(compute_sorted_mixed::<Lev, _, _>(&a8, &b16), naive(&a, &b));
            assert_eq!(
                compute_sorted_mixed::<Lcs, _, _>(&a8, &b16),
                naive_indel(&a, &b)
            );
            // UCS-2 pattern, UCS-1 text.
            let (a, b) = (gen(&wide, m), gen(&narrow, n));
            let a16: Vec<u16> = a.iter().map(|&c| c as u16).collect();
            let b8: Vec<u8> = b.iter().map(|&c| c as u8).collect();
            assert_eq!(compute_sorted_mixed::<Lev, _, _>(&a16, &b8), naive(&a, &b));
            assert_eq!(
                compute_sorted_mixed::<Lcs, _, _>(&a16, &b8),
                naive_indel(&a, &b)
            );
        }
    }

    /// Directly exercise `hyrro_multiword_hash` (m > 64) against the naive DP.
    #[test]
    fn multiword_sorted_matches_oracle() {
        let check_u32 = |a: &[u32], b: &[u32]| {
            let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            assert!(
                s.len() > 64,
                "test case must be long enough to hit multiword"
            );
            let got = hyrro_multiword_hash::<Lev, _, _>(s, l);
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
    fn strip_affix_unaligned_lengths() {
        // One string a prefix of the other, lengths crossing the 8-byte chunk boundary.
        for n in 1..=17 {
            let a = "a".repeat(n);
            let b = "a".repeat(n + 1);
            check(&a, &b, 1);
            let c = format!("{a}b");
            check(&c, &a, 1);
        }
        // Prefix and suffix scans meeting in the middle.
        check("abcdefgh_XY_abcdefgh", "abcdefgh_YX_abcdefgh", 2);
        check("abcdefghij", "abcdefghij", 0);
    }

    /// The block scans must find a difference at every offset inside, at
    /// and past a block boundary, from either end.
    #[test]
    fn affix_scan_finds_every_mismatch() {
        for n in [31usize, 32, 33, 63, 64, 65, 100] {
            let a: Vec<u8> = (0..n).map(|i| i as u8).collect();
            assert_eq!(common_prefix_bytes(&a, &a), n);
            assert_eq!(common_suffix_bytes(&a, &a), n);
            for pos in 0..n {
                let mut b = a.clone();
                b[pos] ^= 0x80;
                assert_eq!(common_prefix_bytes(&a, &b), pos, "n={n}");
                assert_eq!(common_suffix_bytes(&a, &b), n - 1 - pos, "n={n}");
            }
        }
    }

    #[test]
    fn strip_affix_discards_partial_elements() {
        // u16 units where raw bytes match past an element boundary: the
        // half-matched element must not be stripped.
        let a16: Vec<u16> = vec![0x0101, 0x0102, 0x0201];
        let b16: Vec<u16> = vec![0x0101, 0x0202, 0x0201];
        let (sa, sb) = strip_affix(&a16, &b16);
        assert_eq!(sa, &[0x0102]);
        assert_eq!(sb, &[0x0202]);
        assert_eq!(compute_sorted::<Lev, _>(&a16, &b16), 1);
    }

    #[test]
    fn tiny_patterns_match_oracle() {
        // Deterministic LCG; small alphabet forces shared affixes and hits the
        // tiny path (m ≤ TINY_M after stripping) as well as empty-after-strip.
        let mut state = 0x243F_6A88_85A3_08D3_u64;
        let mut rng = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for _ in 0..4000 {
            let la = rng() % 25;
            let lb = rng() % 25;
            let a: String = (0..la)
                .map(|_| (b'a' + (rng() % 4) as u8) as char)
                .collect();
            let b: String = (0..lb)
                .map(|_| (b'a' + (rng() % 4) as u8) as char)
                .collect();
            let ac: Vec<char> = a.chars().collect();
            let bc: Vec<char> = b.chars().collect();
            assert_eq!(levenshtein(&a, &b), naive(&ac, &bc), "({a:?}, {b:?})");
        }
    }

    #[test]
    fn tiny_patterns_across_all_kinds() {
        let mut state = 0x85A3_08D3_243F_6A88_u64;
        let mut rng = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        // Latin-1 (u8 non-ascii)
        for _ in 0..1000 {
            let la = 1 + (rng() % 16);
            let lb = 1 + (rng() % 20);
            let a: Vec<u8> = (0..la).map(|_| 0xc0 + (rng() % 10) as u8).collect();
            let b: Vec<u8> = (0..lb).map(|_| 0xc0 + (rng() % 10) as u8).collect();
            let ac: Vec<char> = a.iter().map(|&c| c as char).collect();
            let bc: Vec<char> = b.iter().map(|&c| c as char).collect();
            assert_eq!(compute_u8::<Lev, false>(&a, &b), naive(&ac, &bc));
        }
        // UCS-2 (u16)
        for _ in 0..1000 {
            let la = 1 + (rng() % 16);
            let lb = 1 + (rng() % 20);
            let a: Vec<u16> = (0..la).map(|_| 0x0500 + (rng() % 10) as u16).collect();
            let b: Vec<u16> = (0..lb).map(|_| 0x0500 + (rng() % 10) as u16).collect();
            let ac: Vec<char> = a
                .iter()
                .map(|&c| char::from_u32(c as u32).unwrap())
                .collect();
            let bc: Vec<char> = b
                .iter()
                .map(|&c| char::from_u32(c as u32).unwrap())
                .collect();
            assert_eq!(compute_sorted::<Lev, _>(&a, &b), naive(&ac, &bc));
        }
        // UCS-4 (u32)
        for _ in 0..1000 {
            let la = 1 + (rng() % 16);
            let lb = 1 + (rng() % 20);
            let a: Vec<u32> = (0..la).map(|_| 0x1f600 + (rng() % 10) as u32).collect();
            let b: Vec<u32> = (0..lb).map(|_| 0x1f600 + (rng() % 10) as u32).collect();
            let ac: Vec<char> = a.iter().map(|&c| char::from_u32(c).unwrap()).collect();
            let bc: Vec<char> = b.iter().map(|&c| char::from_u32(c).unwrap()).collect();
            assert_eq!(compute_sorted::<Lev, _>(&a, &b), naive(&ac, &bc));
        }
        // Mixed: u8 pattern, u16 text
        for _ in 0..1000 {
            let la = 1 + (rng() % 16);
            let lb = 1 + (rng() % 20);
            let a: Vec<u8> = (0..la).map(|_| b'a' + (rng() % 6) as u8).collect();
            let b: Vec<u16> = (0..lb)
                .map(|_| {
                    if rng() % 3 == 0 {
                        (b'a' + (rng() % 6) as u8) as u16
                    } else {
                        0x0300 + (rng() % 5) as u16
                    }
                })
                .collect();
            let ac: Vec<char> = a.iter().map(|&c| c as char).collect();
            let bc: Vec<char> = b
                .iter()
                .map(|&c| char::from_u32(c as u32).unwrap())
                .collect();
            assert_eq!(compute_sorted_mixed::<Lev, _, _>(&a, &b), naive(&ac, &bc));
        }
    }

    /// The mixed-kind block scans must find a difference at every offset
    /// inside, at and past a block boundary, from either end, and count
    /// mismatches like the element-wise definition.
    #[test]
    fn mixed_scans_find_every_mismatch() {
        for n in [15usize, 16, 17, 31, 32, 33, 50] {
            let a: Vec<u8> = (0..n).map(|i| b'a' + (i % 26) as u8).collect();
            let wide: Vec<u16> = a.iter().map(|&c| c as u16).collect();
            assert_eq!(strip_affix_mixed(&a, &wide).0.len(), 0);
            for pos in 0..n {
                let mut b = wide.clone();
                b[pos] |= 0x100;
                let (sa, sb) = strip_affix_mixed(&a, &b);
                assert_eq!((sa, sb), (&a[pos..=pos], &b[pos..=pos]), "n={n}");
                assert_eq!(small_ub_mixed(&a, &b), Some(1), "n={n} pos={pos}");
            }
            let all: Vec<u16> = wide.iter().map(|&c| c | 0x100).collect();
            assert_eq!(small_ub_mixed(&a, &all), None, "n={n}");
        }
    }

    #[test]
    fn tiny_patterns_mixed_kinds() {
        // Tiny path through the mixed-kind pipeline (u8 pattern, u16 text).
        let a: Vec<u8> = b"abc".to_vec();
        let b: Vec<u16> = "axc".encode_utf16().collect();
        assert_eq!(compute_sorted_mixed::<Lev, _, _>(&a, &b), 1);
        let c: Vec<u16> = "xyz".encode_utf16().collect();
        assert_eq!(compute_sorted_mixed::<Lev, _, _>(&a, &c), 3);
    }

    /// Deterministic LCG used by the banded-kernel tests.
    pub(crate) fn lcg(state: &mut u64) -> usize {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as usize
    }

    pub(crate) fn rand_string(state: &mut u64, len: usize, alpha: &[u8]) -> String {
        (0..len)
            .map(|_| alpha[lcg(state) % alpha.len()] as char)
            .collect()
    }

    pub(crate) fn mutate(state: &mut u64, s: &str, edits: usize, alpha: &[u8]) -> String {
        let mut out: Vec<char> = s.chars().collect();
        for _ in 0..edits {
            let i = lcg(state) % out.len();
            match lcg(state) % 3 {
                0 => out[i] = alpha[lcg(state) % alpha.len()] as char,
                1 => out.insert(i, alpha[lcg(state) % alpha.len()] as char),
                _ => {
                    out.remove(i);
                }
            }
        }
        out.into_iter().collect()
    }

    #[test]
    fn banded_long_strings_match_oracle() {
        // m > 512 (w > 8) exercises the banded kernel: small edit counts hit
        // the optimistic pass, larger ones the retry/fallback logic.
        let mut state = 0xDEAD_BEEF_CAFE_F00D_u64;
        let alpha = b"abcdefghij";
        for &(len, edits) in &[(520, 1), (700, 3), (900, 30), (1500, 60), (900, 120)] {
            let a = rand_string(&mut state, len, alpha);
            let b = mutate(&mut state, &a, edits, alpha);
            let ac: Vec<char> = a.chars().collect();
            let bc: Vec<char> = b.chars().collect();
            assert_eq!(
                levenshtein(&a, &b),
                naive(&ac, &bc),
                "len {len} edits {edits}"
            );
        }
        // Substitution-only edits keep the Hamming upper bound tight while
        // d > 64, forcing the second (guaranteed) banded pass.
        let a = rand_string(&mut state, 700, alpha);
        let mut bc: Vec<char> = a.chars().collect();
        for _ in 0..100 {
            let i = lcg(&mut state) % bc.len();
            bc[i] = alpha[lcg(&mut state) % alpha.len()] as char;
        }
        let b: String = bc.iter().collect();
        let ac: Vec<char> = a.chars().collect();
        assert_eq!(levenshtein(&a, &b), naive(&ac, &bc));
    }

    #[test]
    fn banded_shifted_and_dissimilar() {
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let alpha = b"abcdefghij";
        // A 1-char front shift defeats both affix stripping and the Hamming
        // bound, but the optimistic band still catches d = 2 directly.
        let a = rand_string(&mut state, 800, alpha);
        let b = format!("x{}", &a[..a.len() - 1]);
        let (ac, bc): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
        assert_eq!(levenshtein(&a, &b), naive(&ac, &bc));
        // Unrelated strings: early-abort then full-matrix path.
        let c = rand_string(&mut state, 750, alpha);
        let d = rand_string(&mut state, 640, alpha);
        let (cc, dc): (Vec<char>, Vec<char>) = (c.chars().collect(), d.chars().collect());
        assert_eq!(levenshtein(&c, &d), naive(&cc, &dc));
        // Large length difference: banding is skipped outright (t0 too wide).
        let e = rand_string(&mut state, 600, alpha);
        let f = mutate(&mut state, &e, 5, alpha) + &rand_string(&mut state, 700, alpha);
        let (ec, fc): (Vec<char>, Vec<char>) = (e.chars().collect(), f.chars().collect());
        assert_eq!(levenshtein(&e, &f), naive(&ec, &fc));
    }

    #[test]
    fn banded_non_ascii_long_strings() {
        // Banded kernel through the hash-based (non-UCS-1) peq pipeline.
        let mut state = 0x0F0F_0F0F_1111_2222_u64;
        let alpha: Vec<char> = "ぁあぃいぅうぇえおかがきぎく".chars().collect();
        let a: String = (0..640)
            .map(|_| alpha[lcg(&mut state) % alpha.len()])
            .collect();
        let mut bc: Vec<char> = a.chars().collect();
        for _ in 0..7 {
            let i = lcg(&mut state) % bc.len();
            match lcg(&mut state) % 3 {
                0 => bc[i] = alpha[lcg(&mut state) % alpha.len()],
                1 => bc.insert(i, alpha[lcg(&mut state) % alpha.len()]),
                _ => {
                    bc.remove(i);
                }
            }
        }
        let b: String = bc.iter().collect();
        let ac: Vec<char> = a.chars().collect();
        assert_eq!(levenshtein(&a, &b), naive(&ac, &bc));
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
            assert_eq!(
                compute_sorted::<Lev, _>(&au, &bu),
                expected,
                "u16 ({a:?}, {b:?})"
            );
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
            assert_eq!(
                compute_sorted::<Lev, _>(&au, &bu),
                expected,
                "u32 ({a:?}, {b:?})"
            );
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
            sink += hyrro_multiword_direct::<Lev, 256, _, _>(short, long);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        #[cfg(debug_assertions)]
        eprintln!("NOTE: debug build — release will be ~10-20x faster");
        eprintln!(
            "raw Rust hyrro_multiword_direct: {:.3} μs/call  (sink={})",
            us, sink
        );
        assert!(sink > 0); // prevent DCE
    }

    // -----------------------------------------------------------------------
    // Indel distance / ratio
    // -----------------------------------------------------------------------

    /// Reference O(m·n) indel DP (insert and delete only), used as oracle.
    pub(crate) fn naive_indel(a: &[char], b: &[char]) -> usize {
        let (m, n) = (a.len(), b.len());
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, val) in dp[0].iter_mut().enumerate() {
            *val = j;
        }
        for i in 1..=m {
            for j in 1..=n {
                dp[i][j] = if a[i - 1] == b[j - 1] {
                    dp[i - 1][j - 1]
                } else {
                    (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1)
                };
            }
        }
        dp[m][n]
    }

    fn check_indel(a: &str, b: &str, expected: usize) {
        assert_eq!(indel(a, b), expected, "({a:?}, {b:?})");
        assert_eq!(indel(b, a), expected, "({b:?}, {a:?}) symmetry");
    }

    #[test]
    fn indel_basic() {
        check_indel("", "", 0);
        check_indel("", "abc", 3);
        check_indel("abc", "abc", 0);
        check_indel("abc", "xyz", 6);
        check_indel("a", "b", 2);
        // kitten/sitting: LCS "ittn" (4), so 6 + 7 - 8.
        check_indel("kitten", "sitting", 5);
        check_indel("résumé", "resume", 4);
        // Affix stripping must not disturb the result.
        check_indel("abcXdef", "abcYdef", 2);
        check_indel("prefix_middle_suffix", "prefix_MIDDLE_suffix", 12);
    }

    #[test]
    fn ratio_matches_indel_convention() {
        let r = |a: &str, b: &str| -> f64 {
            let total = a.chars().count() + b.chars().count();
            if total == 0 {
                1.0
            } else {
                1.0 - indel(a, b) as f64 / total as f64
            }
        };
        assert!((r("", "") - 1.0).abs() < 1e-12);
        assert!((r("abc", "abc") - 1.0).abs() < 1e-12);
        // Values cross-checked against rapidfuzz.fuzz.ratio / Levenshtein.ratio.
        assert!((r("kitten", "sitting") - 0.615_384_615_384_615_4).abs() < 1e-12);
        assert!((r("résumé", "resume") - 2.0 / 3.0).abs() < 1e-12);
        // Sharing no character is exactly 0.0, unlike a max-length or
        // Levenshtein-over-sum normalization.
        assert!(r("abc", "xyz").abs() < 1e-12);
        assert!(r("a", "b").abs() < 1e-12);
    }

    /// The LCS kernels must agree with the DP across every length regime:
    /// tiny (<= 8), single-word (<= 64), multi-word (<= 512), and the
    /// heap-backed `lcs_full` path (> 512), over ASCII and astral alphabets.
    #[test]
    fn indel_matches_oracle_across_length_regimes() {
        let mut state = 0x5EED_1234_ABCD_0001_u64;
        for alpha in [&b"ab"[..], &b"abcdefghij"[..]] {
            for &len in &[1usize, 5, 8, 9, 33, 64, 65, 100, 200, 513, 700] {
                for &edits in &[1usize, 3, 12] {
                    let a = rand_string(&mut state, len, alpha);
                    // `mutate` indexes modulo the current length, so keep at
                    // least one character alive even if every edit deletes.
                    let b = mutate(&mut state, &a, edits.min(len - 1).max(1), alpha);
                    let (ac, bc): (Vec<char>, Vec<char>) =
                        (a.chars().collect(), b.chars().collect());
                    assert_eq!(indel(&a, &b), naive_indel(&ac, &bc), "({a:?}, {b:?})");

                    // Same content promoted to UCS-4 must agree.
                    let map = |s: &str| -> String {
                        s.chars()
                            .map(|c| char::from_u32(0x1_0000 + c as u32).unwrap())
                            .collect()
                    };
                    let (wa, wb) = (map(&a), map(&b));
                    assert_eq!(indel(&wa, &wb), naive_indel(&ac, &bc), "astral ({a:?})");
                }
            }
        }
    }

    /// Unrelated strings of every length regime: no common affix to strip and
    /// a disjoint alphabet, so the kernels run over their full width.
    #[test]
    fn indel_disjoint_alphabets() {
        for &len in &[1usize, 8, 64, 65, 300, 600] {
            let a: String = std::iter::repeat_n('a', len).collect();
            let b: String = std::iter::repeat_n('b', len).collect();
            check_indel(&a, &b, 2 * len);
        }
    }

    /// Pairs whose lengths differ several-fold, in every length regime of the
    /// shorter string: embedded unchanged (exactly `n - m` apart), unrelated,
    /// and mutated inside random padding; ASCII and astral alphabets.
    #[test]
    fn very_different_lengths_match_oracle() {
        let mut state = 0xD1FF_E4E7_0000_0001_u64;
        let astral = |s: &str| -> String {
            s.chars()
                .map(|c| char::from_u32(0x1_0000 + c as u32).unwrap())
                .collect()
        };
        for alpha in [&b"ab"[..], &b"abcdefghij"[..]] {
            for &(m, n) in &[
                (0usize, 40usize),
                (1, 200),
                (7, 300),
                (9, 150),
                (40, 900),
                (64, 700),
                (65, 1000),
                (200, 1300),
                (530, 1700),
            ] {
                let short = rand_string(&mut state, m, alpha);
                let pad = rand_string(&mut state, n - m, alpha);
                for cut in [0, pad.len() / 3, pad.len()] {
                    let long = format!("{}{short}{}", &pad[..cut], &pad[cut..]);
                    assert_eq!(levenshtein(&short, &long), n - m, "embedded m={m} n={n}");
                }
                let edited = if m > 0 {
                    mutate(&mut state, &short, m / 10 + 1, alpha)
                } else {
                    String::new()
                };
                let unrelated = rand_string(&mut state, n, alpha);
                let padded = format!("{}{edited}{}", &pad[..pad.len() / 2], &pad[pad.len() / 2..]);
                for long in [unrelated, padded] {
                    let (sc, lc): (Vec<char>, Vec<char>) =
                        (short.chars().collect(), long.chars().collect());
                    let d = naive(&sc, &lc);
                    assert_eq!(levenshtein(&short, &long), d, "m={m} n={n}");
                    assert_eq!(levenshtein(&long, &short), d, "m={m} n={n} swapped");
                    let (sa, la) = (astral(&short), astral(&long));
                    assert_eq!(levenshtein(&sa, &la), d, "astral m={m} n={n}");
                }
            }
        }
    }
}
