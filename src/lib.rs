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
//! 3. **ASCII fast path** – when both inputs are pure ASCII the
//!    algorithm operates directly on `&[u8]`, avoiding `char` decoding.
//!    Pattern bitmasks live in a `[u64; 256]` array indexed by byte
//!    value.
//! 4. **Hyyrö's bit-parallel algorithm** (Hyyrö, 2003) – runs in
//!    `O(⌈m / w⌉ · n)` time, where `w = 64`. The single-word variant is
//!    used whenever the shorter input fits in a 64-bit register, which
//!    covers the overwhelming majority of real-world inputs.
//! 5. **Two-row Wagner-Fischer** – a cache-friendly `O(m · n)` fallback
//!    for the rare case where the shorter input exceeds 64 code units.
//!
//! Computation runs with the Python GIL released so other Python
//! threads make progress while a long call is in flight.

use std::collections::HashMap;

use pyo3::prelude::*;

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
fn distance(py: Python<'_>, s1: &str, s2: &str) -> usize {
    py.detach(|| levenshtein(s1, s2))
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
fn ratio(py: Python<'_>, s1: &str, s2: &str) -> f64 {
    py.detach(|| {
        if s1.is_ascii() && s2.is_ascii() {
            let total = s1.len() + s2.len();
            if total == 0 {
                return 1.0;
            }
            let dist = levenshtein_bytes(s1.as_bytes(), s2.as_bytes());
            1.0 - (dist as f64) / (total as f64)
        } else {
            let a: Vec<char> = s1.chars().collect();
            let b: Vec<char> = s2.chars().collect();
            let total = a.len() + b.len();
            if total == 0 {
                return 1.0;
            }
            let dist = levenshtein_chars(&a, &b);
            1.0 - (dist as f64) / (total as f64)
        }
    })
}

/// A Python module implemented in Rust for the Levenshtein distance.
#[pymodule]
fn lev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Core dispatch
// ---------------------------------------------------------------------------

/// Top-level dispatch into the ASCII or Unicode pipeline.
fn levenshtein(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        levenshtein_bytes(s1.as_bytes(), s2.as_bytes())
    } else {
        let a: Vec<char> = s1.chars().collect();
        let b: Vec<char> = s2.chars().collect();
        levenshtein_chars(&a, &b)
    }
}

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
// ASCII pipeline
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

/// Hyyrö's single-word bit-parallel Levenshtein for ASCII patterns.
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
// Unicode pipeline
// ---------------------------------------------------------------------------

fn levenshtein_chars(a: &[char], b: &[char]) -> usize {
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
        hyrro_64_chars(short, long)
    } else {
        wagner_fischer(short, long)
    }
}

/// Hyyrö's single-word bit-parallel Levenshtein for Unicode patterns.
///
/// `pattern` must satisfy `1 ≤ pattern.len() ≤ 64`.
fn hyrro_64_chars(pattern: &[char], text: &[char]) -> usize {
    debug_assert!((1..=64).contains(&pattern.len()));
    let m = pattern.len();
    let mut peq: HashMap<char, u64> = HashMap::with_capacity(m);
    for (i, &c) in pattern.iter().enumerate() {
        *peq.entry(c).or_insert(0) |= 1u64 << i;
    }
    hyrro_inner(m, text.iter().map(|c| peq.get(c).copied().unwrap_or(0)))
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
    // Initial vertical-positive vector: ones in the m low bits.
    let mut vp: u64 = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    let mut vn: u64 = 0;
    let last_bit = 1u64 << (m - 1);
    let mut score = m;

    for pm in pm_iter {
        let x = pm | vn;
        let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x;
        let hp_pre = vn | !(d0 | vp);
        let hn_pre = vp & d0;

        // Update score from the horizontal delta at the bottom row.
        if hp_pre & last_bit != 0 {
            score += 1;
        }
        if hn_pre & last_bit != 0 {
            score -= 1;
        }

        // Shift in the implicit `+1` horizontal delta at the top row.
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
        // Pattern length exactly 64 hits the `m == 64` branch in `hyrro_inner`.
        let mut shifted = String::from("b");
        shifted.push_str(&"a".repeat(63));
        check(&a64, &shifted, 1);
    }

    #[test]
    fn long_inputs_use_wagner_fischer() {
        // > 64 chars on the shorter side forces the WF path.
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
            // Cross the 64-char boundary on at least one side.
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
        // Replicate the public formula directly.
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
