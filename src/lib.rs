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
//!    of three packed internal encodings (UCS-1 / UCS-2 / UCS-4).  We
//!    read the raw buffer directly via the CPython C API, paying zero
//!    UTF-8 decode cost.  For UCS-1 (`u8`, ≤ U+00FF) the peq table is a
//!    flat `[u64; 256]` array (O(1) lookup).  For UCS-2 (`u16`) and
//!    UCS-4 (`u32`) a unified generic implementation uses a
//!    stack-allocated sorted `(T, u64)` peq array (O(log m) lookup, zero
//!    heap allocation).  Mixed-kind pairs (e.g. UCS-1 vs UCS-2) decode
//!    via a reusable thread-local `Vec<char>` buffer and share the same
//!    generic path via `char`.
//! 4. **Hyyrö's bit-parallel algorithm** (Hyyrö, 2003) – runs in
//!    `O(⌈m / w⌉ · n)` time with `w = 64`.  The single-word variant is
//!    used whenever the shorter input is ≤ 64 code units, which covers
//!    the overwhelming majority of real-world inputs.
//! 5. **Two-row Wagner-Fischer** – a cache-friendly `O(m · n)` fallback
//!    for inputs whose shorter half exceeds 64 code units after affix
//!    stripping.  The GIL is released before this path runs so that other
//!    Python threads can make progress during long computations.

use std::cell::RefCell;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

// ---------------------------------------------------------------------------
// Thread-local reusable char buffers for the mixed-kind Unicode fallback.
// ---------------------------------------------------------------------------

thread_local! {
    static CHAR_BUFS: RefCell<(Vec<char>, Vec<char>)> = const {
        RefCell::new((Vec::new(), Vec::new()))
    };
}

// ---------------------------------------------------------------------------
// Sealed trait unifying UCS-2 (u16), UCS-4 (u32), and char as peq-key types.
// ---------------------------------------------------------------------------

/// Implemented by every code-unit type that can be used in the sorted peq
/// array.  `SENTINEL` fills unused array slots and is never searched.
trait CodeUnit: Ord + Copy + Eq + Send + 'static {
    const SENTINEL: Self;
}
impl CodeUnit for u16  { const SENTINEL: Self = u16::MAX;  }
impl CodeUnit for u32  { const SENTINEL: Self = u32::MAX;  }
impl CodeUnit for char { const SENTINEL: Self = char::MAX; }

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

/// Returns Python's internal UCS-1 buffer (1 byte / code point, ≤ U+00FF).
#[inline]
unsafe fn ucs1_bytes<'a>(s: &Bound<'a, PyString>) -> Option<&'a [u8]> {
    let ptr = s.as_ptr();
    if ffi::PyUnicode_KIND(ptr) != ffi::PyUnicode_1BYTE_KIND {
        return None;
    }
    let len  = ffi::PyUnicode_GET_LENGTH(ptr) as usize;
    let data = ffi::PyUnicode_DATA(ptr) as *const u8;
    Some(std::slice::from_raw_parts(data, len))
}

/// Returns Python's internal UCS-2 buffer (2 bytes / code point, ≤ U+FFFF).
#[inline]
unsafe fn ucs2_shorts<'a>(s: &Bound<'a, PyString>) -> Option<&'a [u16]> {
    let ptr = s.as_ptr();
    if ffi::PyUnicode_KIND(ptr) != ffi::PyUnicode_2BYTE_KIND {
        return None;
    }
    let len  = ffi::PyUnicode_GET_LENGTH(ptr) as usize;
    let data = ffi::PyUnicode_DATA(ptr) as *const u16;
    Some(std::slice::from_raw_parts(data, len))
}

/// Returns Python's internal UCS-4 buffer (4 bytes / code point, > U+FFFF).
#[inline]
unsafe fn ucs4_words<'a>(s: &Bound<'a, PyString>) -> Option<&'a [u32]> {
    let ptr = s.as_ptr();
    if ffi::PyUnicode_KIND(ptr) != ffi::PyUnicode_4BYTE_KIND {
        return None;
    }
    let len  = ffi::PyUnicode_GET_LENGTH(ptr) as usize;
    let data = ffi::PyUnicode_DATA(ptr) as *const u32;
    Some(std::slice::from_raw_parts(data, len))
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
    // Access CPython's packed buffer directly — no UTF-8 encode, no char
    // decode.  Short inputs (≤ 64 code units) run Hyyrö while holding the
    // GIL; long inputs copy the data then release the GIL for Wagner-Fischer.
    unsafe {
        if let (Some(b1), Some(b2)) = (ucs1_bytes(s1), ucs1_bytes(s2)) {
            return Ok(apply(py, prep_bytes(b1, b2)));
        }
        if let (Some(h1), Some(h2)) = (ucs2_shorts(s1), ucs2_shorts(s2)) {
            return Ok(apply(py, prep_fixed(h1, h2)));
        }
        if let (Some(w1), Some(w2)) = (ucs4_words(s1), ucs4_words(s2)) {
            return Ok(apply(py, prep_fixed(w1, w2)));
        }
    }
    // Mixed-kind fallback (e.g. UCS-1 vs UCS-2).  Rare in practice.
    // to_str() can fail for strings containing lone surrogates.
    Ok(levenshtein_unicode_full(s1.to_str()?, s2.to_str()?))
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
        if let (Some(b1), Some(b2)) = (ucs1_bytes(s1), ucs1_bytes(s2)) {
            let total = b1.len() + b2.len();
            if total == 0 { return Ok(1.0); }
            return Ok(1.0 - apply(py, prep_bytes(b1, b2)) as f64 / total as f64);
        }
        if let (Some(h1), Some(h2)) = (ucs2_shorts(s1), ucs2_shorts(s2)) {
            let total = h1.len() + h2.len();
            if total == 0 { return Ok(1.0); }
            return Ok(1.0 - apply(py, prep_fixed(h1, h2)) as f64 / total as f64);
        }
        if let (Some(w1), Some(w2)) = (ucs4_words(s1), ucs4_words(s2)) {
            let total = w1.len() + w2.len();
            if total == 0 { return Ok(1.0); }
            return Ok(1.0 - apply(py, prep_fixed(w1, w2)) as f64 / total as f64);
        }
    }
    // to_str() can fail for strings containing lone surrogates.
    Ok(ratio_unicode_full(s1.to_str()?, s2.to_str()?))
}

/// A Python module implemented in Rust for the Levenshtein distance.
#[pymodule]
fn lev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Affix stripping
// ---------------------------------------------------------------------------

fn strip_affix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> (&'a [T], &'a [T]) {
    let prefix = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    let (a, b) = (&a[prefix..], &b[prefix..]);
    let suffix = a.iter().rev().zip(b.iter().rev()).take_while(|(x, y)| x == y).count();
    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}

// ---------------------------------------------------------------------------
// UCS-1 pipeline  (dedicated O(1) [u64; 256] peq table)
// ---------------------------------------------------------------------------

/// Strip affixes and run Hyyrö, or package slices for deferred WF.
fn prep_bytes(a: &[u8], b: &[u8]) -> Prep<u8> {
    if a == b { return Prep::Done(0); }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() { return Prep::Done(b.len()); }
    if b.is_empty() { return Prep::Done(a.len()); }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        Prep::Done(hyrro_64_bytes(short, long))
    } else {
        Prep::Wf(short.to_vec(), long.to_vec())
    }
}

/// Hyyrö single-word variant for byte patterns.  `pattern.len()` in `1..=64`.
fn hyrro_64_bytes(pattern: &[u8], text: &[u8]) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let m = pattern.len();
    let mut peq = [0u64; 256];
    for (i, &c) in pattern.iter().enumerate() {
        peq[c as usize] |= 1u64 << i;
    }
    hyrro_inner(m, text.iter().map(|&b| peq[b as usize]))
}

// ---------------------------------------------------------------------------
// Generic pipeline for UCS-2 (u16), UCS-4 (u32), and char
// ---------------------------------------------------------------------------

/// Strip affixes and run Hyyrö, or package slices for deferred WF.
fn prep_fixed<T: CodeUnit>(a: &[T], b: &[T]) -> Prep<T> {
    if a == b { return Prep::Done(0); }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() { return Prep::Done(b.len()); }
    if b.is_empty() { return Prep::Done(a.len()); }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        Prep::Done(hyrro_64_sorted(short, long.iter().copied()))
    } else {
        Prep::Wf(short.to_vec(), long.to_vec())
    }
}

/// Hyyrö single-word variant with a stack-allocated sorted `(T, u64)` peq
/// array.  No heap allocation; O(log m) peq lookup per text character.
/// `pattern.len()` must be in `1..=64`.
fn hyrro_64_sorted<T: CodeUnit>(pattern: &[T], text: impl Iterator<Item = T>) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let m = pattern.len();

    let mut peq: [(T, u64); 64] = [(T::SENTINEL, 0); 64];
    let mut peq_len = 0usize;

    for (i, &c) in pattern.iter().enumerate() {
        let bit = 1u64 << i;
        match peq[..peq_len].binary_search_by_key(&c, |p| p.0) {
            Ok(idx)  => peq[idx].1 |= bit,
            Err(idx) => {
                peq.copy_within(idx..peq_len, idx + 1);
                peq[idx] = (c, bit);
                peq_len += 1;
            }
        }
    }
    let peq = &peq[..peq_len];

    hyrro_inner(m, text.map(|c| match peq.binary_search_by_key(&c, |p| p.0) {
        Ok(idx) => peq[idx].1,
        Err(_)  => 0,
    }))
}

// ---------------------------------------------------------------------------
// GIL-aware dispatch
// ---------------------------------------------------------------------------

/// Finalise a [`Prep`] result.  `Done` returns immediately.  `Wf` releases
/// the GIL before running Wagner-Fischer so other threads can make progress
/// during long O(m·n) computations.
#[inline]
fn apply<T: Eq + Send + 'static>(py: Python<'_>, prep: Prep<T>) -> usize {
    match prep {
        Prep::Done(d)    => d,
        Prep::Wf(s, l)   => py.detach(move || wagner_fischer(&s, &l)),
    }
}

// ---------------------------------------------------------------------------
// Mixed-kind fallback (e.g. UCS-1 vs UCS-2)
// ---------------------------------------------------------------------------

fn levenshtein_unicode_full(s1: &str, s2: &str) -> usize {
    if s1 == s2 { return 0; }
    CHAR_BUFS.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (a, b) = &mut *guard;
        a.clear(); a.extend(s1.chars());
        b.clear(); b.extend(s2.chars());
        // prep_fixed clones into owned Vecs for Wf, so drop of `guard` is safe.
        let prep = prep_fixed(a.as_slice(), b.as_slice());
        drop(guard);
        match prep {
            Prep::Done(d)  => d,
            Prep::Wf(s, l) => wagner_fischer(&s, &l),
        }
    })
}

fn ratio_unicode_full(s1: &str, s2: &str) -> f64 {
    CHAR_BUFS.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (a, b) = &mut *guard;
        a.clear(); a.extend(s1.chars());
        b.clear(); b.extend(s2.chars());
        let total = a.len() + b.len();
        if total == 0 { return 1.0; }
        let prep = prep_fixed(a.as_slice(), b.as_slice());
        drop(guard);
        let dist = match prep {
            Prep::Done(d)  => d,
            Prep::Wf(s, l) => wagner_fischer(&s, &l),
        };
        1.0 - dist as f64 / total as f64
    })
}

// ---------------------------------------------------------------------------
// Hyyrö's bit-parallel inner loop (single 64-bit word)
// ---------------------------------------------------------------------------

/// Core Hyyrö loop shared by all pipelines.  `pm_iter` yields the
/// pattern-match bitmask for each successive text element.
///
/// Reference: H. Hyyrö, *A bit-vector algorithm for computing
/// Levenshtein and Damerau edit distances*, Nordic Journal of Computing,
/// 2003.
#[inline(always)]
fn hyrro_inner(m: usize, pm_iter: impl Iterator<Item = u64>) -> usize {
    let mut vp = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    let mut vn = 0u64;
    let last   = 1u64 << (m - 1);
    let mut score = m;

    for pm in pm_iter {
        let x    = pm | vn;
        let d0   = (((x & vp).wrapping_add(vp)) ^ vp) | x;
        let hp   = vn | !(d0 | vp);
        let hn   = vp & d0;

        if hp & last != 0 { score += 1; }
        if hn & last != 0 { score -= 1; }

        let hp_s = (hp << 1) | 1;
        let hn_s = hn << 1;
        vp = hn_s | !(d0 | hp_s);
        vn = hp_s & d0;
    }

    score
}

// ---------------------------------------------------------------------------
// Wagner-Fischer fallback (two-row, generic over any Eq type)
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
        finish(prep_bytes(s1.as_bytes(), s2.as_bytes()))
    } else {
        levenshtein_unicode_full(s1, s2)
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
    fn long_inputs_use_wagner_fischer() {
        let a: String = "abc".repeat(40); // 120 chars
        let mut b = a.clone();
        b.insert(0, 'x');
        check(&a, &b, 1);
        let c = format!("{a}xyz");
        check(&a, &c, 3);
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
            assert_eq!(levenshtein(a, b), expected, "char ({a:?}, {b:?})");
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
            assert_eq!(levenshtein(a, b), expected, "char ({a:?}, {b:?})");
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
