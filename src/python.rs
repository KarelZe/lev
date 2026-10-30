//! Python bindings: the `_lev` extension module behind the `lev` package.
//!
//! Python strings are read in place, in whichever of CPython's three compact
//! encodings (UCS-1 / UCS-2 / UCS-4, [PEP 393](https://peps.python.org/pep-0393/))
//! they are stored in, and handed to the same [`compute`] pipeline as the
//! Rust API.

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

#[cfg(target_arch = "x86_64")]
use crate::kernels;
use crate::{compute, Baseline, Isa, Lcs, Lev, UniView};

// CPython's string kinds are the code-unit widths `UniView` uses.
const _: () = assert!(
    ffi::PyUnicode_1BYTE_KIND == crate::KIND_1BYTE
        && ffi::PyUnicode_2BYTE_KIND == crate::KIND_2BYTE
        && ffi::PyUnicode_4BYTE_KIND == crate::KIND_4BYTE
);

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
    let kind = ((state >> KIND_SHIFT) & 0b111) as std::os::raw::c_uint;
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

/// [`distance`] in the copy `X` of the hot path.
///
/// # Safety
///
/// The CPU must support the instruction set `X`.
#[inline(always)]
pub(crate) unsafe fn distance_in<X: Isa>(
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
pub(crate) unsafe fn ratio_in<X: Isa>(
    s1: Borrowed<'_, '_, PyString>,
    s2: Borrowed<'_, '_, PyString>,
) -> f64 {
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
        ///     >>> lev.distance("日本語", "日本")
        ///     1
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
        ///     >>> lev.ratio("résumé", "resume")
        ///     0.6666666666666667
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
