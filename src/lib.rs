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
//! 3. **UCS-1 fast path** – for strings whose code points are all
//!    ≤ U+00FF (ASCII and all European text), Python stores the string
//!    internally as a packed `u8` array.  We access that buffer directly
//!    via the CPython C API, paying zero UTF-8 decode cost and running
//!    the algorithm on plain `&[u8]` with an O(1) `[u64; 256]` peq table.
//! 4. **Hyyrö's bit-parallel algorithm** (Hyyrö, 2003) – runs in
//!    `O(⌈m / w⌉ · n)` time, where `w = 64`. The single-word variant is
//!    used whenever the shorter input fits in a 64-bit register, which
//!    covers the overwhelming majority of real-world inputs.
//! 5. **Two-row Wagner-Fischer** – a cache-friendly `O(m · n)` fallback
//!    for the rare case where the shorter input exceeds 64 code units.
//! 4. **UCS-2 fast path** – for strings whose code points are all ≤ U+FFFF
//!    (CJK and most non-Latin scripts), Python stores the string internally
//!    as a packed `u16` array.  We access that buffer directly, avoiding
//!    UTF-8 decode, using a stack-allocated sorted `(u16, u64)` peq array.
//! 5. **UCS-4 fast path** – for strings with code points above U+FFFF
//!    (emoji and rare scripts), Python stores the string internally as a
//!    packed `u32` array.  Same direct-access approach with a sorted
//!    `(u32, u64)` peq array.
//! 6. **Full Unicode fallback** – for mixed-kind pairs (e.g. a UCS-1
//!    string vs. a UCS-2 string), a reusable thread-local `Vec<char>` buffer
//!    is decoded once per call, affix-stripped in `char` space, then fed
//!    to Hyyrö using a stack-allocated sorted `(char, u64)` peq array.

use std::cell::RefCell;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;

// ---------------------------------------------------------------------------
// Thread-local reusable buffers for the full Unicode fallback.
// ---------------------------------------------------------------------------

thread_local! {
    static CHAR_BUFS: RefCell<(Vec<char>, Vec<char>)> = const {
        RefCell::new((Vec::new(), Vec::new()))
    };
}

// ---------------------------------------------------------------------------
// UCS-1 buffer access
// ---------------------------------------------------------------------------

/// Return Python's internal UCS-1 byte buffer if kind == 1 (all code points
/// ≤ U+00FF), otherwise `None`.  The slice borrows from the Python object and
/// is valid as long as the GIL is held and `s` is alive — both guaranteed
/// within a `#[pyfunction]` call.
#[inline]
unsafe fn ucs1_bytes<'a>(s: &Bound<'a, PyString>) -> Option<&'a [u8]> {
    let ptr = s.as_ptr();
    if ffi::PyUnicode_KIND(ptr) != ffi::PyUnicode_1BYTE_KIND {
        return None;
    }
    let len = ffi::PyUnicode_GET_LENGTH(ptr) as usize;
    let data = ffi::PyUnicode_DATA(ptr) as *const u8;
    Some(std::slice::from_raw_parts(data, len))
}

/// Return Python's internal UCS-2 buffer if kind == 2 (all code points
/// ≤ U+FFFF, e.g. CJK), otherwise `None`.
#[inline]
unsafe fn ucs2_shorts<'a>(s: &Bound<'a, PyString>) -> Option<&'a [u16]> {
    let ptr = s.as_ptr();
    if ffi::PyUnicode_KIND(ptr) != ffi::PyUnicode_2BYTE_KIND {
        return None;
    }
    let len = ffi::PyUnicode_GET_LENGTH(ptr) as usize;
    let data = ffi::PyUnicode_DATA(ptr) as *const u16;
    Some(std::slice::from_raw_parts(data, len))
}

/// Return Python's internal UCS-4 buffer if kind == 4 (code points above
/// U+FFFF, e.g. emoji), otherwise `None`.
#[inline]
unsafe fn ucs4_words<'a>(s: &Bound<'a, PyString>) -> Option<&'a [u32]> {
    let ptr = s.as_ptr();
    if ffi::PyUnicode_KIND(ptr) != ffi::PyUnicode_4BYTE_KIND {
        return None;
    }
    let len = ffi::PyUnicode_GET_LENGTH(ptr) as usize;
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
fn distance(_py: Python<'_>, s1: &Bound<'_, PyString>, s2: &Bound<'_, PyString>) -> usize {
    // All three branches access CPython's internal packed buffer directly —
    // no UTF-8 encode, no char decode.  GIL is held throughout.
    unsafe {
        if let (Some(b1), Some(b2)) = (ucs1_bytes(s1), ucs1_bytes(s2)) {
            return levenshtein_bytes(b1, b2);
        }
        if let (Some(h1), Some(h2)) = (ucs2_shorts(s1), ucs2_shorts(s2)) {
            return levenshtein_u16(h1, h2);
        }
        if let (Some(w1), Some(w2)) = (ucs4_words(s1), ucs4_words(s2)) {
            return levenshtein_u32(w1, w2);
        }
    }
    // Mixed-kind fallback (e.g. UCS-1 vs UCS-2).  Rare in practice.
    let s1 = s1.to_str().expect("valid Python string");
    let s2 = s2.to_str().expect("valid Python string");
    levenshtein_unicode_full(s1, s2)
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
fn ratio(_py: Python<'_>, s1: &Bound<'_, PyString>, s2: &Bound<'_, PyString>) -> f64 {
    unsafe {
        if let (Some(b1), Some(b2)) = (ucs1_bytes(s1), ucs1_bytes(s2)) {
            let total = b1.len() + b2.len();
            if total == 0 { return 1.0; }
            return 1.0 - levenshtein_bytes(b1, b2) as f64 / total as f64;
        }
        if let (Some(h1), Some(h2)) = (ucs2_shorts(s1), ucs2_shorts(s2)) {
            let total = h1.len() + h2.len();
            if total == 0 { return 1.0; }
            return 1.0 - levenshtein_u16(h1, h2) as f64 / total as f64;
        }
        if let (Some(w1), Some(w2)) = (ucs4_words(s1), ucs4_words(s2)) {
            let total = w1.len() + w2.len();
            if total == 0 { return 1.0; }
            return 1.0 - levenshtein_u32(w1, w2) as f64 / total as f64;
        }
    }
    let s1 = s1.to_str().expect("valid Python string");
    let s2 = s2.to_str().expect("valid Python string");
    ratio_unicode_full(s1, s2)
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

/// Strip the longest common prefix and suffix from two slices.
fn strip_affix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> (&'a [T], &'a [T]) {
    let prefix = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    let (a, b) = (&a[prefix..], &b[prefix..]);
    let suffix = a
        .iter()
        .rev()
        .zip(b.iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}

// ---------------------------------------------------------------------------
// ASCII / UCS-1 pipeline
// ---------------------------------------------------------------------------

fn levenshtein_bytes(a: &[u8], b: &[u8]) -> usize {
    if a == b {
        return 0;
    }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        hyrro_64_bytes(short, long)
    } else {
        wagner_fischer(short, long)
    }
}

/// Hyyrö's single-word bit-parallel Levenshtein for byte patterns.
///
/// `pattern` must satisfy `1 ≤ pattern.len() ≤ 64`.
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
// UCS-2 pipeline (CJK and other BMP scripts: code points ≤ U+FFFF)
// ---------------------------------------------------------------------------

fn levenshtein_u16(a: &[u16], b: &[u16]) -> usize {
    if a == b { return 0; }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() { return b.len(); }
    if b.is_empty() { return a.len(); }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        hyrro_64_u16(short, long.iter().copied())
    } else {
        wagner_fischer(short, long)
    }
}

/// Hyyrö's single-word bit-parallel Levenshtein for UCS-2 patterns.
///
/// Uses a stack-allocated sorted `(u16, u64)` peq array — no heap allocation,
/// O(log m) lookup per text character.  `pattern.len()` must be in `1..=64`.
fn hyrro_64_u16(pattern: &[u16], text: impl Iterator<Item = u16>) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let m = pattern.len();
    let mut peq: [(u16, u64); 64] = [(u16::MAX, 0); 64];
    let mut peq_len = 0usize;
    for (i, &c) in pattern.iter().enumerate() {
        let bit = 1u64 << i;
        match peq[..peq_len].binary_search_by_key(&c, |p| p.0) {
            Ok(idx) => peq[idx].1 |= bit,
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
        Err(_) => 0,
    }))
}

// ---------------------------------------------------------------------------
// UCS-4 pipeline (emoji and supplementary code points: > U+FFFF)
// ---------------------------------------------------------------------------

fn levenshtein_u32(a: &[u32], b: &[u32]) -> usize {
    if a == b { return 0; }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() { return b.len(); }
    if b.is_empty() { return a.len(); }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        hyrro_64_u32(short, long.iter().copied())
    } else {
        wagner_fischer(short, long)
    }
}

/// Hyyrö's single-word bit-parallel Levenshtein for UCS-4 patterns.
///
/// Uses a stack-allocated sorted `(u32, u64)` peq array.
/// `pattern.len()` must be in `1..=64`.
fn hyrro_64_u32(pattern: &[u32], text: impl Iterator<Item = u32>) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let m = pattern.len();
    let mut peq: [(u32, u64); 64] = [(u32::MAX, 0); 64];
    let mut peq_len = 0usize;
    for (i, &c) in pattern.iter().enumerate() {
        let bit = 1u64 << i;
        match peq[..peq_len].binary_search_by_key(&c, |p| p.0) {
            Ok(idx) => peq[idx].1 |= bit,
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
        Err(_) => 0,
    }))
}

// ---------------------------------------------------------------------------
// Full Unicode fallback (mixed-kind pairs: e.g. UCS-1 vs UCS-2)
// ---------------------------------------------------------------------------

/// Decode `s` and compute the Levenshtein distance, using a thread-local
/// `Vec<char>` buffer to avoid repeated heap allocation.
fn levenshtein_unicode_full(s1: &str, s2: &str) -> usize {
    if s1 == s2 {
        return 0;
    }
    CHAR_BUFS.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (a, b) = &mut *guard;
        a.clear();
        a.extend(s1.chars());
        b.clear();
        b.extend(s2.chars());
        levenshtein_char_slices(a, b)
    })
}

fn ratio_unicode_full(s1: &str, s2: &str) -> f64 {
    CHAR_BUFS.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (a, b) = &mut *guard;
        a.clear();
        a.extend(s1.chars());
        b.clear();
        b.extend(s2.chars());
        let total = a.len() + b.len();
        if total == 0 {
            return 1.0;
        }
        let dist = levenshtein_char_slices(a, b);
        1.0 - (dist as f64) / (total as f64)
    })
}

/// Core computation over already-decoded `&[char]` slices.
fn levenshtein_char_slices(a: &[char], b: &[char]) -> usize {
    if a == b {
        return 0;
    }
    let (a, b) = strip_affix(a, b);
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() <= 64 {
        hyrro_64_unicode(short, long.iter().copied())
    } else {
        wagner_fischer(short, long)
    }
}

/// Hyyrö's single-word bit-parallel Levenshtein for Unicode patterns.
///
/// The peq table is a **stack-allocated sorted `(char, u64)` array** —
/// no heap allocation, no hashing.  Binary search over ≤ 64 entries
/// (O(log 64) = 6 comparisons) fits entirely in cache, beating FxHashMap
/// for the small maps that arise from patterns ≤ 64 chars.
///
/// `pattern` must satisfy `1 ≤ pattern.len() ≤ 64`.
fn hyrro_64_unicode(pattern: &[char], text: impl Iterator<Item = char>) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let m = pattern.len();

    let mut peq: [(char, u64); 64] = [(char::MAX, 0); 64];
    let mut peq_len: usize = 0;

    for (i, &c) in pattern.iter().enumerate() {
        let bit = 1u64 << i;
        match peq[..peq_len].binary_search_by_key(&c, |p| p.0) {
            Ok(idx) => peq[idx].1 |= bit,
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
        Err(_) => 0,
    }))
}

// ---------------------------------------------------------------------------
// Hyyrö's bit-parallel inner loop (single 64-bit word)
// ---------------------------------------------------------------------------

/// Core Hyyrö loop. `pm_iter` yields the pattern-match bitmask for each
/// successive text element.
///
/// Reference: H. Hyyrö, *A bit-vector algorithm for computing
/// Levenshtein and Damerau edit distances*, Nordic Journal of Computing,
/// 2003.
#[inline(always)]
fn hyrro_inner(m: usize, pm_iter: impl Iterator<Item = u64>) -> usize {
    let mut vp: u64 = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    let mut vn: u64 = 0;
    let last_bit = 1u64 << (m - 1);
    let mut score = m;

    for pm in pm_iter {
        let x = pm | vn;
        let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x;
        let hp_pre = vn | !(d0 | vp);
        let hn_pre = vp & d0;

        if hp_pre & last_bit != 0 {
            score += 1;
        }
        if hn_pre & last_bit != 0 {
            score -= 1;
        }

        let hp = (hp_pre << 1) | 1;
        let hn = hn_pre << 1;

        vp = hn | !(d0 | hp);
        vn = hp & d0;
    }

    score
}

// ---------------------------------------------------------------------------
// Wagner-Fischer fallback (two-row, generic)
// ---------------------------------------------------------------------------

/// Two-row Wagner-Fischer dynamic programming.
///
/// `short` should be the shorter of the two inputs to minimize working
/// memory; runtime is `O(short.len() · long.len())` with a single
/// `Vec<usize>` allocation of length `short.len() + 1`.
fn wagner_fischer<T: Eq>(short: &[T], long: &[T]) -> usize {
    let m = short.len();
    let mut row: Vec<usize> = (0..=m).collect();
    for (j, lj) in long.iter().enumerate() {
        let mut diag = row[0];
        row[0] = j + 1;
        for (i, si) in short.iter().enumerate() {
            let cost = (si != lj) as usize;
            let above = row[i + 1];
            let left = row[i];
            let new_val = (above + 1).min(left + 1).min(diag + cost);
            diag = above;
            row[i + 1] = new_val;
        }
    }
    row[m]
}

// ---------------------------------------------------------------------------
// Internal helpers for testing (bypass PyO3 layer)
// ---------------------------------------------------------------------------

#[cfg(test)]
fn levenshtein(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        levenshtein_bytes(s1.as_bytes(), s2.as_bytes())
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
            assert_eq!(levenshtein_u16(&au, &bu), expected, "ucs2 ({a:?}, {b:?})");
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
            // encode_utf16 gives surrogate pairs; we need scalar values (u32)
            let au: Vec<u32> = a.chars().map(|c| c as u32).collect();
            let bu: Vec<u32> = b.chars().map(|c| c as u32).collect();
            assert_eq!(levenshtein_u32(&au, &bu), expected, "ucs4 ({a:?}, {b:?})");
            assert_eq!(levenshtein(a, b), expected, "char ({a:?}, {b:?})");
        }
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
