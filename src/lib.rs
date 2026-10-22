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
//!    - *Large* (`m > 512`): one loop over all words per column, with the
//!      state on the heap.
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

use std::mem::MaybeUninit;
use std::os::raw::c_uint;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyString};

// ---------------------------------------------------------------------------
// Code units of the three CPython string kinds (UCS-1 / UCS-2 / UCS-4).
// ---------------------------------------------------------------------------

/// A CPython code unit (`u8`, `u16` or `u32`).  Mixed-kind pairs compare
/// code units widened to `u64`.
trait CodeUnit: Copy + Eq + Into<u64> {
    #[inline(always)]
    fn as_u64(self) -> u64 {
        self.into()
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
#[cfg(not(all(
    Py_3_14,
    target_endian = "little",
    not(all(Py_GIL_DISABLED, target_env = "msvc"))
)))]
#[inline(always)]
unsafe fn view(ptr: *mut ffi::PyObject) -> UniView {
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
#[cfg(all(
    Py_3_14,
    target_endian = "little",
    not(all(Py_GIL_DISABLED, target_env = "msvc"))
))]
#[inline(always)]
unsafe fn view(ptr: *mut ffi::PyObject) -> UniView {
    const KIND_SHIFT: u32 = if cfg!(Py_GIL_DISABLED) { 8 } else { 2 };
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
// `pass_module` binds the functions to the module instead of registering
// them `METH_STATIC`, which PyO3 0.28 does otherwise at a cost of a few ns per
// call on CPython 3.11+ (PyO3 issue #6426).
#[pyfunction]
#[pyo3(pass_module, signature = (s1, s2, /))]
fn distance(_m: &Bound<'_, PyModule>, s1: &Bound<'_, PyString>, s2: &Bound<'_, PyString>) -> usize {
    // SAFETY: both are `str` objects.
    unsafe { distance_of(s1.as_ptr(), s2.as_ptr()) }
}

/// [`distance`] of two `str` objects.
#[inline(always)]
unsafe fn distance_of(s1: *mut ffi::PyObject, s2: *mut ffi::PyObject) -> usize {
    if s1 == s2 {
        return 0;
    }
    compute::<Lev>(&view(s1), &view(s2))
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
fn ratio(_m: &Bound<'_, PyModule>, s1: &Bound<'_, PyString>, s2: &Bound<'_, PyString>) -> f64 {
    // SAFETY: both are `str` objects.
    unsafe { ratio_of(s1.as_ptr(), s2.as_ptr()) }
}

/// [`ratio`] of two `str` objects.
#[inline(always)]
unsafe fn ratio_of(s1: *mut ffi::PyObject, s2: *mut ffi::PyObject) -> f64 {
    if s1 == s2 {
        return 1.0;
    }
    let (v1, v2) = (view(s1), view(s2));
    let total = v1.len + v2.len;
    if total == 0 {
        return 1.0;
    }
    1.0 - compute::<Lcs>(&v1, &v2) as f64 / total as f64
}

// ---------------------------------------------------------------------------
// Fast entry points
// ---------------------------------------------------------------------------
//
// Python calls `METH_FASTCALL` functions that handle the common call, two
// positional `str` arguments, themselves.  Every other call (wrong arity,
// keywords, other types) and any panic is forwarded to the PyO3 functions
// above, which raise their usual errors.  This skips PyO3's generic argument
// extraction and per-call attach bookkeeping, about 9 ns of a ~30 ns call on
// short strings.  Name, docstring and text signature are PyO3's.

/// The PyO3 functions the fast entry points fall back to.
static SLOW_DISTANCE: OnceLock<Py<PyCFunction>> = OnceLock::new();
static SLOW_RATIO: OnceLock<Py<PyCFunction>> = OnceLock::new();

/// The arguments of a vectorcall when they are exactly two positional `str`s.
#[inline(always)]
unsafe fn str_pair(
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
) -> Option<(*mut ffi::PyObject, *mut ffi::PyObject)> {
    if nargs != 2 || !kwnames.is_null() {
        return None;
    }
    let (s1, s2) = (*args, *args.add(1));
    (ffi::PyUnicode_Check(s1) != 0 && ffi::PyUnicode_Check(s2) != 0).then_some((s1, s2))
}

/// Hand a call the fast path does not take to the PyO3 function in `slow`.
#[cold]
#[inline(never)]
unsafe fn forward(
    slow: &OnceLock<Py<PyCFunction>>,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // Set before the fast function exists (see `add_fast`).
    let slow = slow.get().map_or(std::ptr::null_mut(), Py::as_ptr);
    ffi::PyObject_Vectorcall(slow, args, nargs as usize, kwnames)
}

unsafe extern "C" fn distance_fast(
    _m: *mut ffi::PyObject,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    if let Some((s1, s2)) = str_pair(args, nargs, kwnames) {
        if let Ok(d) = catch_unwind(AssertUnwindSafe(|| distance_of(s1, s2))) {
            return ffi::PyLong_FromSize_t(d);
        }
    }
    forward(&SLOW_DISTANCE, args, nargs, kwnames)
}

unsafe extern "C" fn ratio_fast(
    _m: *mut ffi::PyObject,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    if let Some((s1, s2)) = str_pair(args, nargs, kwnames) {
        if let Ok(r) = catch_unwind(AssertUnwindSafe(|| ratio_of(s1, s2))) {
            return ffi::PyFloat_FromDouble(r);
        }
    }
    forward(&SLOW_RATIO, args, nargs, kwnames)
}

/// Add `fast` to `m` under the name and docstring of the PyO3 function
/// `slow`, which it falls back to.
fn add_fast(
    m: &Bound<'_, PyModule>,
    slow: Bound<'_, PyCFunction>,
    slow_cell: &'static OnceLock<Py<PyCFunction>>,
    fast: ffi::PyCFunctionFastWithKeywords,
) -> PyResult<()> {
    // SAFETY: `slow` is a PyCFunction, whose method def is a PyO3 static, so
    // its name and docstring pointers live for the whole process.
    let slow_def = unsafe { &*(*slow.as_ptr().cast::<ffi::PyCFunctionObject>()).m_ml };
    let def = Box::leak(Box::new(ffi::PyMethodDef {
        ml_name: slow_def.ml_name,
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunctionFastWithKeywords: fast,
        },
        ml_flags: ffi::METH_FASTCALL | ffi::METH_KEYWORDS,
        ml_doc: slow_def.ml_doc,
    }));
    let name = slow.getattr("__name__")?;
    slow_cell.get_or_init(|| slow.unbind());
    // SAFETY: `def` is leaked, so it outlives the function; `m` and its name
    // are valid objects.
    let fast = unsafe {
        Bound::from_owned_ptr_or_err(
            m.py(),
            ffi::PyCFunction_NewEx(def, m.as_ptr(), m.name()?.as_ptr()),
        )?
    };
    m.add(name.cast_into::<PyString>()?, fast)
}

/// The compiled extension backing the `lev` package.
///
/// Private: the public API is re-exported from `lev/__init__.py`, so the name
/// is underscore-prefixed to keep it out of the package namespace.
#[pymodule]
fn _lev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_fast(
        m,
        wrap_pyfunction!(distance, m)?,
        &SLOW_DISTANCE,
        distance_fast,
    )?;
    add_fast(m, wrap_pyfunction!(ratio, m)?, &SLOW_RATIO, ratio_fast)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Top-level dispatch: read both views' kinds, branch once, run the right path.
// ---------------------------------------------------------------------------

/// Dispatch on `(kind, ascii)` tuples and return the `K` edit distance directly.
#[inline(always)]
unsafe fn compute<K: Kernel>(v1: &UniView, v2: &UniView) -> usize {
    use ffi::{PyUnicode_1BYTE_KIND as K1, PyUnicode_2BYTE_KIND as K2, PyUnicode_4BYTE_KIND as K4};

    // 1. Normalize: v1 is the shorter pattern string (m), v2 is the text (n).
    let (v1, v2) = if v1.len <= v2.len { (v1, v2) } else { (v2, v1) };

    // 2. Dispatch based on kinds.
    match (v1.kind, v2.kind) {
        (K1, K1) => {
            let (b1, b2) = (as_u8(v1), as_u8(v2));
            if v1.ascii && v2.ascii {
                compute_u8::<K, true>(b1, b2)
            } else {
                compute_u8::<K, false>(b1, b2)
            }
        }
        (K2, K2) => compute_sorted::<K, _>(as_u16(v1), as_u16(v2)),
        (K4, K4) => compute_sorted::<K, _>(as_u32(v1), as_u32(v2)),
        // Mixed kinds compare code units in place, without a widened copy.
        (K1, K2) => compute_sorted_mixed::<K, _, _>(as_u8(v1), as_u16(v2)),
        (K1, K4) => compute_sorted_mixed::<K, _, _>(as_u8(v1), as_u32(v2)),
        (K2, K1) => compute_sorted_mixed::<K, _, _>(as_u16(v1), as_u8(v2)),
        (K2, K4) => compute_sorted_mixed::<K, _, _>(as_u16(v1), as_u32(v2)),
        (K4, K1) => compute_sorted_mixed::<K, _, _>(as_u32(v1), as_u8(v2)),
        (K4, K2) => compute_sorted_mixed::<K, _, _>(as_u32(v1), as_u16(v2)),
        _ => unreachable!("unknown PyUnicode kind"),
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

/// Reinterpret a code-unit slice as raw bytes (sound: u8/u16/u32 are plain data).
#[inline(always)]
fn as_bytes<T: CodeUnit>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Length in bytes of the common prefix of `a` and `b`, compared 8 bytes at a time.
#[inline(always)]
fn common_prefix_bytes(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let diff = x ^ y;
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

/// Length in bytes of the common suffix of `a` and `b`, compared 8 bytes at a time.
#[inline(always)]
fn common_suffix_bytes(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let (la, lb) = (a.len(), b.len());
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[la - i - 8..la - i].try_into().unwrap());
        let y = u64::from_le_bytes(b[lb - i - 8..lb - i].try_into().unwrap());
        let diff = x ^ y;
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
/// pattern-match word is computed per text char by a fully unrolled linear
/// scan.  Zero-initializing the peq tables (1-2 KiB) dominates the runtime for
/// tiny inputs, so trading one table load for ≤ `TINY_M` compare+or ops wins.
const TINY_M: usize = 8;

/// Tiny-pattern kernel: no peq table, `pm` computed per text char directly
/// from the pattern.
///
/// A plain loop over a runtime-length pattern gets auto-vectorized per text
/// char (widening loads, horizontal reduce, scalar tail), which costs several
/// times the bit-parallel step it feeds.  Single-byte patterns therefore use a
/// SWAR match; wider ones dispatch on a const length so the scan unrolls into
/// `m` compares.
///
/// Kept out of line: inlining these loops into the dispatcher perturbs
/// register allocation of the single-word paths sharing that function.
#[inline(never)]
fn hyrro_64_tiny<K: Kernel, T1: CodeUnit, T2: CodeUnit>(pattern: &[T1], text: &[T2]) -> usize {
    debug_assert!((1..=TINY_M).contains(&pattern.len()));
    if std::mem::size_of::<T1>() == 1 {
        return tiny_swar::<K, _, _>(pattern, text);
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
        _ => unreachable!("tiny path requires 1..=TINY_M pattern chars"),
    }
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
        if let Some(ub) = small_ub(a, b) {
            return mbleven(a, b, ub);
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
            hyrro_multiword_bytes::<K, 128>(a, b)
        } else {
            hyrro_multiword_bytes::<K, 256>(a, b)
        }
    }
}

// ---------------------------------------------------------------------------
// Small-distance fast path (mbleven)
// ---------------------------------------------------------------------------

/// Largest internally-derived upper bound handled by [`mbleven`].
const MBLEVEN_MAX: usize = 3;

/// Longest pattern eligible for the mbleven gate.  Bounds the gate's
/// hamming scan; beyond this the banded kernels already handle
/// near-identical strings well.
const MBLEVEN_MAX_LEN: usize = 512;

/// Gate for the mbleven fast path (same-kind slices): `Some(ub)` when
/// `length difference + aligned mismatches <= MBLEVEN_MAX`.
///
/// Scans the raw bytes one u64 block at a time (element sizes divide 8, so a
/// block always holds whole elements — same trick as
/// [`common_prefix_bytes`]).  Clean blocks cost one XOR + branch; only blocks
/// containing a difference are inspected per lane, and at most four such
/// blocks are touched before aborting, so the worst case stays cheap on both
/// early-mismatch and late-mismatch inputs.
#[inline(always)]
fn small_ub<T: CodeUnit>(short: &[T], long: &[T]) -> Option<usize> {
    debug_assert!(short.len() <= long.len());
    let mut ub = long.len() - short.len();
    if ub > MBLEVEN_MAX || short.len() > MBLEVEN_MAX_LEN {
        return None;
    }
    let size = std::mem::size_of::<T>();
    let lane_bits = 8 * size;
    let lane_mask = if lane_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << lane_bits) - 1
    };
    let a = as_bytes(short);
    let b = &as_bytes(long)[..a.len()];

    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, _) = b.as_chunks::<8>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        let mut d = u64::from_le_bytes(*ca) ^ u64::from_le_bytes(*cb);
        while d != 0 {
            ub += 1;
            if ub > MBLEVEN_MAX {
                return None;
            }
            let lane = d.trailing_zeros() as usize / lane_bits;
            d &= !(lane_mask << (lane * lane_bits));
        }
    }
    for e in (a.len() - rem_a.len()) / size..short.len() {
        if short[e].as_u64() != long[e].as_u64() {
            ub += 1;
            if ub > MBLEVEN_MAX {
                return None;
            }
        }
    }
    Some(ub)
}

/// [`small_ub`] for mixed-kind slices: plain element-wise scan with early
/// abort (mixed pairs have no byte-comparable representation).
#[inline(always)]
fn small_ub_mixed<T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2]) -> Option<usize> {
    debug_assert!(short.len() <= long.len());
    let mut ub = long.len() - short.len();
    if ub > MBLEVEN_MAX || short.len() > MBLEVEN_MAX_LEN {
        return None;
    }
    for (x, y) in short.iter().zip(long.iter()) {
        if x.as_u64() != y.as_u64() {
            ub += 1;
            if ub > MBLEVEN_MAX {
                return None;
            }
        }
    }
    Some(ub)
}

/// Exact distance when an upper bound `ub` in `1..=3` is known (mbleven,
/// Hyyrö 2018 formulation as used by rapidfuzz): try every candidate edit-op
/// sequence of cost <= `ub` and take the cheapest that aligns the strings.
///
/// Each sequence packs ops two bits per op from the LSB: `0b01` advances only
/// `long` (deletion from the longer string), `0b10` advances only `short`
/// (insertion), `0b11` advances both (substitution).  O(ub * n) worst case,
/// no peq table, no allocation.
fn mbleven<T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2], ub: usize) -> usize {
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
        while sp < short.len() && lp < long.len() {
            // SAFETY: sp/lp are bounded by the loop condition.
            let (cs, cl) = unsafe {
                (
                    short.get_unchecked(sp).as_u64(),
                    long.get_unchecked(lp).as_u64(),
                )
            };
            if cs != cl {
                cost += 1;
                if ops == 0 {
                    break;
                }
                lp += (ops & 1) as usize;
                sp += ((ops >> 1) & 1) as usize;
                ops >>= 2;
            } else {
                sp += 1;
                lp += 1;
            }
        }
        cost += (short.len() - sp) + (long.len() - lp);
        best = best.min(cost);
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
        if let Some(ub) = small_ub(a, b) {
            return mbleven(a, b, ub);
        }
    }
    if a.len() <= 64 {
        hyrro_64_hash::<K, _, _>(a, b)
    } else {
        hyrro_multiword_hash::<K, _, _>(a, b)
    }
}

/// Strip affixes and run kernel `K` for mixed-kind strings.
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
            return mbleven(a, b, ub);
        }
    }
    if a.len() <= 64 {
        hyrro_64_hash::<K, _, _>(a, b)
    } else {
        hyrro_multiword_hash::<K, _, _>(a, b)
    }
}

// ---------------------------------------------------------------------------
// Hyyrö single-word variants
// ---------------------------------------------------------------------------

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
        text.iter()
            .map(|&c| unsafe { *peq.get_unchecked(c as usize) }),
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
                let v = unsafe { *vals.get_unchecked(slot) };
                if v == 0 {
                    return 0;
                }
                // SAFETY: v != 0, so keys[slot] was written.
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
        let sum = (eq & vp).wrapping_add(vp);
        let ph = vn | !(sum | (vp | eq));
        let mh = vp & (eq | !sum);
        let xv = eq | vn;
        let ph_s = (ph << 1) | 1;
        vp = (mh << 1) | !(xv | ph_s);
        vn = ph_s & xv;
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

/// Multi-word LCS over `W` words; returns the LCS length.
#[inline(always)]
fn lcs_multiword_kernel<const W: usize, I: Iterator<Item = [u64; W]>>(pm_iter: I) -> usize {
    let mut s = [!0u64; W];
    for pm_row in pm_iter {
        lcs_step(&mut s, false, |k| pm_row[k]);
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

/// Levenshtein distance (substitution, insertion, and deletion each cost 1).
struct Lev;

impl Kernel for Lev {
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
        (n as isize + multiword_kernel::<W, I>(m, pm_iter)) as usize
    }

    #[inline(always)]
    fn large<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
        ctx.run()
    }
}

/// Indel distance: insertions and deletions cost 1, substitution is not an
/// operation (it decomposes into one of each, cost 2).  Derived from the LCS
/// length as `m + n - 2 * LLCS`.
struct Lcs;

impl Kernel for Lcs {
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
        m + n - 2 * lcs_multiword_kernel::<W, I>(pm_iter)
    }

    #[inline(always)]
    fn large<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
        ctx.m + ctx.n - 2 * ctx.lcs_run()
    }
}

// ---------------------------------------------------------------------------
// Multi-word Hyyrö (m > 64)
// ---------------------------------------------------------------------------

/// Widest pattern, in 64-bit words, handled by the const-width kernels with
/// stack peq tables; longer patterns go through [`LargeCtx`].
const MAX_W: usize = 8;

/// Const-generic inner kernel over `m` pattern rows in `W` words, unrolled
/// for speed, in Hyyrö's form (bits shifted out of the top of one word are
/// shifted in at the bottom of the next).  Returns the sum of the vertical
/// deltas of these rows in the last text column, i.e. their contribution to
/// the distance on top of `D(0, n) = n` (see [`hyrro_inner`]).
#[inline(always)]
fn multiword_kernel<const W: usize, I: Iterator<Item = [u64; W]>>(m: usize, pm_iter: I) -> isize {
    let mut vp = [!0u64; W];
    let mut vn = [0u64; W];
    for pm_row in pm_iter {
        let (mut carry, mut prev_hp, mut prev_hn) = (false, 1u64, 0u64);
        for k in 0..W {
            let (pm, v_p, v_n) = (pm_row[k], vp[k], vn[k]);
            let x = pm | v_n;
            let (sum, nc) = (x & v_p).carrying_add(v_p, carry);
            carry = nc;
            let d0 = (sum ^ v_p) | x;
            let hp = v_n | !(d0 | v_p);
            let hn = v_p & d0;
            let (hp_s, hn_s) = (hp << 1 | prev_hp, hn << 1 | prev_hn);
            vp[k] = hn_s | !(d0 | hp_s);
            vn[k] = hp_s & d0;
            (prev_hp, prev_hn) = (hp >> 63, hn >> 63);
        }
    }
    // Rows above the pattern in the top word hold garbage; mask them off.
    let rows = u64::MAX >> (64 * W - m);
    vp[W - 1] &= rows;
    vn[W - 1] &= rows;
    let count = |v: &[u64; W]| v.iter().map(|x| x.count_ones() as isize).sum::<isize>();
    count(&vp) - count(&vn)
}

/// Multi-word entry point for UCS-1 slices; every byte must be below `SLOTS`
/// (128 for pure ASCII, 256 otherwise), which sizes the peq table.
fn hyrro_multiword_bytes<K: Kernel, const SLOTS: usize>(short: &[u8], long: &[u8]) -> usize {
    debug_assert!(short.len() > 64);
    let m = short.len();
    let w = m.div_ceil(64);

    macro_rules! run {
        ($W:literal) => {{
            // An array-of-arrays literal is zeroed row by row in a loop;
            // `zeroed` lowers to one `bzero` call.
            // SAFETY: all-zero bytes are a valid `[[u64; W]; SLOTS]`.
            let mut peq: [[u64; $W]; SLOTS] = unsafe { std::mem::zeroed() };
            for (i, &c) in short.iter().enumerate() {
                // SAFETY: c < SLOTS (caller contract) and i / 64 < $W.
                unsafe {
                    *peq.get_unchecked_mut(c as usize).get_unchecked_mut(i / 64) |=
                        1u64 << (i % 64);
                }
            }
            K::words::<$W, _>(
                m,
                long.len(),
                // SAFETY: c < SLOTS (caller contract).
                long.iter()
                    .map(|&c| unsafe { *peq.get_unchecked(c as usize) }),
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
            let mut peq = vec![0u64; SLOTS * w];
            for (i, &c) in short.iter().enumerate() {
                peq[c as usize * w + i / 64] |= 1u64 << (i % 64);
            }
            K::large(&LargeCtx {
                m,
                n: long.len(),
                w,
                data: &peq,
                // SAFETY: the kernels only ask for j < n.
                base_of: |j: usize| unsafe { *long.get_unchecked(j) } as usize * w,
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
    fn banded(&self, t: usize, pv: &mut [u64], mv: &mut [u64]) -> Result<usize, usize> {
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
            for k in first..=last {
                let eq = self.peq(base, k);
                let (pv_k, mv_k) = (pv[k], mv[k]);
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
                pv[k] = mh_s | !(xv | ph_s);
                mv[k] = ph_s & xv;
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

    /// Full-matrix exact kernel (fallback for w > MAX_W): one loop over all
    /// words per column, with the state on the heap.
    fn full(&self) -> usize {
        let (m, w) = (self.m, self.w);
        let (mut vp, mut vn) = (vec![!0u64; w], vec![0u64; w]);
        self.for_each_base(|base| {
            let (mut carry, mut prev_hp, mut prev_hn) = (false, 1u64, 0u64);
            for (k, (vp_k, vn_k)) in vp.iter_mut().zip(vn.iter_mut()).enumerate() {
                let (pm, v_p, v_n) = (self.peq(base, k), *vp_k, *vn_k);
                let x = pm | v_n;
                let (sum, nc) = (x & v_p).carrying_add(v_p, carry);
                carry = nc;
                let d0 = (sum ^ v_p) | x;
                let hp = v_n | !(d0 | v_p);
                let hn = v_p & d0;
                let (hp_s, hn_s) = (hp << 1 | prev_hp, hn << 1 | prev_hn);
                *vp_k = hn_s | !(d0 | hp_s);
                *vn_k = hp_s & d0;
                (prev_hp, prev_hn) = (hp >> 63, hn >> 63);
            }
        });
        // Distance = n + sum of the last column's vertical deltas (see
        // `hyrro_inner`); rows above the pattern hold garbage.
        vp[w - 1] &= u64::MAX >> (64 * w - m);
        vn[w - 1] &= u64::MAX >> (64 * w - m);
        let count = |v: &[u64]| v.iter().map(|x| x.count_ones() as usize).sum::<usize>();
        self.n + count(&vp) - count(&vn)
    }

    /// Bands of growing width before the full matrix.  Each pass aborts as
    /// soon as it can no longer prove a result, so a miss on dissimilar
    /// strings costs a few percent of the full kernel.
    fn run(&self) -> usize {
        let mut t = (self.n - self.m + 32).max(64);
        if band_blocks(t) * 3 > self.w {
            return self.full();
        }
        let bases: Vec<usize> = (0..self.n).map(&self.base_of).collect();
        let ctx = self.with_bases(&bases);
        let (mut pv, mut mv) = (vec![0u64; self.w], vec![0u64; self.w]);
        while band_blocks(t) * 3 <= self.w {
            match ctx.banded(t, &mut pv, &mut mv) {
                Ok(d) => return d,
                Err(j) => t = next_band(t, j, self.n),
            }
        }
        ctx.full()
    }

    /// Full-matrix LCS kernel (fallback for w > MAX_W), one loop over all
    /// words per column like [`Self::full`]; returns the LCS length.
    fn lcs_full(&self) -> usize {
        let mut s = vec![!0u64; self.w];
        self.for_each_base(|base| {
            lcs_step(&mut s, false, |k| self.peq(base, k));
        });
        lcs_len(&s)
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
    fn lcs_run(&self) -> usize {
        let slack = self.n - self.m;
        let mut t = 64;
        if lcs_band_words(slack, t) * 3 > self.w {
            return self.lcs_full();
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
        ctx.lcs_full()
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
        // SAFETY: all `len` elements were just initialized, and
        // MaybeUninit<u64> has the layout of u64.
        unsafe { &mut *(buf as *mut [MaybeUninit<u64>] as *mut [u64]) }
    } else {
        *heap = vec![value; len];
        heap
    }
}

/// Multi-word entry point for UCS-2/4 and mixed kinds: the peq rows are
/// indexed through an open-addressing hash of the pattern alphabet.
fn hyrro_multiword_hash<K: Kernel, T1: CodeUnit, T2: CodeUnit>(short: &[T1], long: &[T2]) -> usize {
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
    let mut n_keys = 0u64;
    for &c in short {
        let key = c.as_u64();
        debug_assert!(key <= u32::MAX as u64, "CodeUnit value must fit in 32 bits");
        let mut slot = hslot(key, hshift);
        loop {
            let entry = hash[slot];
            if entry == EMPTY {
                hash[slot] = n_keys << 32 | key;
                n_keys += 1;
                break;
            }
            if entry & 0xFFFF_FFFF == key {
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
    for (i, &c) in short.iter().enumerate() {
        let ki = row_of(c.as_u64());
        debug_assert!(ki < n_keys, "pattern key must already be inserted");
        data[ki * w + i / 64] |= 1u64 << (i % 64);
    }

    macro_rules! run {
        ($W:literal) => {{
            K::words::<$W, _>(
                m,
                long.len(),
                long.iter().map(|&c| {
                    let base = row_of(c.as_u64()) * $W;
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
    fn naive(a: &[char], b: &[char]) -> usize {
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
                        if let Some(ub) = small_ub(&a, &b) {
                            if ub == 0 {
                                continue;
                            }
                            let ac: Vec<char> = a.iter().map(|&c| c as char).collect();
                            let bc: Vec<char> = b.iter().map(|&c| c as char).collect();
                            assert_eq!(
                                mbleven(&a, &b, ub),
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
            assert_eq!(hyrro_multiword_bytes::<Lev, 256>(s, l), oracle(s, l));
        };
        // Latin-1 bytes: any u8 value allowed.
        let check_latin1 = |a: &[u8], b: &[u8]| {
            let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            assert!(s.len() > 64);
            assert_eq!(hyrro_multiword_bytes::<Lev, 256>(s, l), oracle(s, l));
        };

        // Benchmark-like: two long ASCII strings that diverge after a shared prefix.
        let s1 = b"Lets pretend Marshall Mathers never picked up a pen".repeat(8);
        let s2 = b"Lets pretend things woulda been no different".repeat(8);
        let (sh, lo) = if s1.len() <= s2.len() {
            (&s1[..], &s2[..])
        } else {
            (&s2[..], &s1[..])
        };
        assert_eq!(hyrro_multiword_bytes::<Lev, 256>(sh, lo), oracle(sh, lo));

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
        for _ in 0..2000 {
            let la = rng() % 13;
            let lb = rng() % 13;
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
    fn tiny_patterns_mixed_kinds() {
        // Tiny path through the mixed-kind pipeline (u8 pattern, u16 text).
        let a: Vec<u8> = b"abc".to_vec();
        let b: Vec<u16> = "axc".encode_utf16().collect();
        assert_eq!(compute_sorted_mixed::<Lev, _, _>(&a, &b), 1);
        let c: Vec<u16> = "xyz".encode_utf16().collect();
        assert_eq!(compute_sorted_mixed::<Lev, _, _>(&a, &c), 3);
    }

    /// Deterministic LCG used by the banded-kernel tests.
    fn lcg(state: &mut u64) -> usize {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as usize
    }

    fn rand_string(state: &mut u64, len: usize, alpha: &[u8]) -> String {
        (0..len)
            .map(|_| alpha[lcg(state) % alpha.len()] as char)
            .collect()
    }

    fn mutate(state: &mut u64, s: &str, edits: usize, alpha: &[u8]) -> String {
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
            sink += hyrro_multiword_bytes::<Lev, 256>(short, long);
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

    // -----------------------------------------------------------------------
    // Indel distance / ratio
    // -----------------------------------------------------------------------

    /// Reference O(m·n) indel DP (insert and delete only), used as oracle.
    fn naive_indel(a: &[char], b: &[char]) -> usize {
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

    /// The very-long-pattern kernels must agree with the oracle on their own:
    /// the full matrices, which the public path reaches only when banding
    /// does not pay, and the band drivers, on similar and dissimilar pairs.
    #[test]
    fn large_kernels_match_oracle() {
        let mut state = 0xA5C3_0F0F_1234_5678_u64;
        let alpha = &b"abcdefgh"[..];
        for &(m, n, edits) in &[
            (530usize, 530usize, 5usize),
            (700, 900, 0),
            (1030, 1100, 60),
            (600, 2000, 0),
            (2000, 2000, 40),
        ] {
            let a = rand_string(&mut state, m, alpha);
            let b = if edits > 0 {
                mutate(&mut state, &a, edits, alpha)
            } else {
                rand_string(&mut state, n, alpha)
            };
            let (s, l) = if a.len() <= b.len() {
                (&a, &b)
            } else {
                (&b, &a)
            };
            let (s, l) = (s.as_bytes(), l.as_bytes());
            let w = s.len().div_ceil(64);
            let mut data = vec![0u64; 256 * w];
            for (i, &c) in s.iter().enumerate() {
                data[c as usize * w + i / 64] |= 1 << (i % 64);
            }
            let ctx = LargeCtx {
                m: s.len(),
                n: l.len(),
                w,
                data: &data,
                base_of: |j: usize| l[j] as usize * w,
            };
            let (sc, lc): (Vec<char>, Vec<char>) = (
                s.iter().map(|&c| c as char).collect(),
                l.iter().map(|&c| c as char).collect(),
            );
            let (d, e) = (naive(&sc, &lc), naive_indel(&sc, &lc));
            let llcs = (s.len() + l.len() - e) / 2;
            let msg = format!("m={} n={}", s.len(), l.len());
            assert_eq!(ctx.full(), d, "full {msg}");
            assert_eq!(ctx.run(), d, "run {msg}");
            assert_eq!(ctx.lcs_full(), llcs, "lcs_full {msg}");
            assert_eq!(ctx.lcs_run(), llcs, "lcs_run {msg}");
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
