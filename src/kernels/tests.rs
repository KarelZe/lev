//! Every form of every kernel against the DP oracle or the portable form,
//! on every target, not only on the one that selects it.

use super::aarch64::{self, apple, neoverse};
use super::x86_64::{self, bmi1};
use super::{myers_step, portable, EDGE_ROW0};
use crate::tests::{lcg, mutate, naive, naive_indel, rand_string};
use crate::{compute_sorted, compute_sorted_mixed, compute_u8, Baseline, Isa, LargeCtx, Lcs, Lev};

/// The and-not form must agree with Hyyrö's form and the oracle on every
/// target, not only where it is selected: on real pairs for every width,
/// and with Hyyrö's form on random pattern-match rows (the and-not form is an exact
/// rewrite for any state).
#[test]
fn multiword_andn_matches_oracle() {
    fn both<const W: usize>(m: usize, rows: &[[u64; W]]) -> (isize, isize) {
        (
            portable::multiword::<W, _>(m, rows.iter().copied(), None),
            bmi1::multiword_andn::<W, _>(m, rows.iter().copied()),
        )
    }
    fn check<const W: usize>(state: &mut u64) {
        let alpha = b"abcdefgh";
        let m = 64 * (W - 1) + 1 + lcg(state) % 64;
        let a = rand_string(state, m, alpha);
        let extra = lcg(state) % 50;
        for b in [
            mutate(state, &a, 5, alpha),
            rand_string(state, m + extra, alpha),
        ] {
            let (s, l) = if a.len() <= b.len() {
                (a.as_bytes(), b.as_bytes())
            } else {
                (b.as_bytes(), a.as_bytes())
            };
            if s.len().div_ceil(64) != W {
                continue;
            }
            let mut peq = [[0u64; W]; 256];
            for (i, &c) in s.iter().enumerate() {
                peq[c as usize][i / 64] |= 1 << (i % 64);
            }
            let rows: Vec<[u64; W]> = l.iter().map(|&c| peq[c as usize]).collect();
            let (sc, lc): (Vec<char>, Vec<char>) = (
                s.iter().map(|&c| c as char).collect(),
                l.iter().map(|&c| c as char).collect(),
            );
            let expected = naive(&sc, &lc) as isize - l.len() as isize;
            assert_eq!(both::<W>(s.len(), &rows), (expected, expected), "W={W}");
        }
        let rows: Vec<[u64; W]> = (0..200)
            .map(|_| std::array::from_fn(|_| (lcg(state) as u64) << 33 ^ lcg(state) as u64))
            .collect();
        let (t, n) = both::<W>(64 * W - 3, &rows);
        assert_eq!(t, n, "random rows W={W}");
    }
    let mut state = 0xA11D_0F0F_1234_5678_u64;
    for _ in 0..20 {
        check::<2>(&mut state);
        check::<3>(&mut state);
        check::<4>(&mut state);
        check::<5>(&mut state);
        check::<6>(&mut state);
        check::<7>(&mut state);
        check::<8>(&mut state);
    }
}

/// The very-long-pattern kernels must agree with the oracle on their own:
/// both forms of the full matrices on every target (the public path
/// reaches them only when banding does not pay), and the band drivers,
/// on similar and dissimilar pairs.
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
        assert_eq!(aarch64::full(&ctx), d, "aarch64::full {msg}");
        assert_eq!(portable::full(&ctx), d, "portable::full {msg}");
        assert_eq!(ctx.run::<Baseline>(), d, "run {msg}");
        assert_eq!(aarch64::lcs_full(&ctx), llcs, "aarch64::lcs_full {msg}");
        assert_eq!(portable::lcs_full(&ctx), llcs, "portable::lcs_full {msg}");
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("bmi1") && std::is_x86_feature_detected!("popcnt") {
            // SAFETY: both target features were just detected.
            let l = unsafe { bmi1::lcs_full(&ctx) };
            assert_eq!(l, llcs, "bmi1::lcs_full {msg}");
        }
        assert_eq!(ctx.lcs_run::<Baseline>(), llcs, "lcs_run {msg}");
    }
}

/// Every multi-word step form must agree with the oracle on every target,
/// not only on the one that selects it, as stacked strips and (Neoverse,
/// whose whole-pattern instances no other test reaches off AArch64) as
/// whole patterns.
#[test]
fn multiword_forms_match_oracle() {
    let mut state = 0xA5C3_0F0F_1234_5678_u64;
    let alpha = &b"abcdefgh"[..];
    for &m in &[65usize, 128, 129, 200, 300, 350, 385, 449, 512, 700, 1030] {
        let a = rand_string(&mut state, m, alpha);
        for b in [
            mutate(&mut state, &a, 5, alpha),
            rand_string(&mut state, m + m / 3, alpha),
        ] {
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
            let expected = naive(&sc, &lc);
            for form in ["portable", "apple", "neoverse"] {
                // Strips of up to four words over the whole pattern.
                let mut edges = vec![EDGE_ROW0; l.len()];
                let mut delta = 0isize;
                for w0 in (0..w).step_by(4) {
                    let rows = (s.len() - 64 * w0).min(256);
                    let e = Some(&mut edges[..]);
                    delta += match ((w - w0).min(4), form) {
                        (1, "portable") => portable::multiword::<1, _>(rows, ctx.rows::<1>(w0), e),
                        (2, "portable") => portable::multiword::<2, _>(rows, ctx.rows::<2>(w0), e),
                        (3, "portable") => portable::multiword::<3, _>(rows, ctx.rows::<3>(w0), e),
                        (_, "portable") => portable::multiword::<4, _>(rows, ctx.rows::<4>(w0), e),
                        (1, "apple") => apple::multiword::<1, _>(rows, ctx.rows::<1>(w0), e),
                        (2, "apple") => apple::multiword::<2, _>(rows, ctx.rows::<2>(w0), e),
                        (3, "apple") => apple::multiword::<3, _>(rows, ctx.rows::<3>(w0), e),
                        (_, "apple") => apple::multiword::<4, _>(rows, ctx.rows::<4>(w0), e),
                        (1, _) => neoverse::multiword::<1, _>(rows, ctx.rows::<1>(w0), e),
                        (2, _) => neoverse::multiword::<2, _>(rows, ctx.rows::<2>(w0), e),
                        (3, _) => neoverse::multiword::<3, _>(rows, ctx.rows::<3>(w0), e),
                        (_, _) => neoverse::multiword::<4, _>(rows, ctx.rows::<4>(w0), e),
                    };
                }
                let d = (l.len() as isize + delta) as usize;
                assert_eq!(d, expected, "{form} m={} n={}", s.len(), l.len());
            }
            // Whole patterns, as the const-width kernels run them.
            let whole = match w {
                2 => neoverse::multiword::<2, _>(s.len(), ctx.rows::<2>(0), None),
                3 => neoverse::multiword::<3, _>(s.len(), ctx.rows::<3>(0), None),
                4 => neoverse::multiword::<4, _>(s.len(), ctx.rows::<4>(0), None),
                5 => neoverse::multiword::<5, _>(s.len(), ctx.rows::<5>(0), None),
                6 => neoverse::multiword::<6, _>(s.len(), ctx.rows::<6>(0), None),
                7 => neoverse::multiword::<7, _>(s.len(), ctx.rows::<7>(0), None),
                8 => neoverse::multiword::<8, _>(s.len(), ctx.rows::<8>(0), None),
                _ => continue,
            };
            let d = (l.len() as isize + whole) as usize;
            assert_eq!(d, expected, "neoverse whole m={} n={}", s.len(), l.len());
        }
    }
}

/// Every single-word step must match the portable reference on every
/// reachable state (`vp & vn == 0`): the selected one, the Neoverse form on
/// every target, and the hand-scheduled Apple form on Apple silicon.
#[test]
fn myers_step_matches_portable() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut states: Vec<(u64, u64, u64)> = (0..200_000)
        .map(|_| {
            let (eq, vp) = (next(), next());
            (eq, vp, next() & !vp)
        })
        .collect();
    // Words with the top or bottom bit alone set or alone clear.
    let edges = [0, 1, 1 << 63, !0, !1, !0 >> 1];
    for eq in edges {
        for vp in edges {
            for vn in edges {
                states.push((eq, vp, vn & !vp));
            }
        }
    }
    for (eq, vp, vn) in states {
        let expected = portable::myers_step(eq, vp, vn);
        assert_eq!(myers_step(eq, vp, vn), expected);
        assert_eq!(neoverse::myers_step(eq, vp, vn), expected, "neoverse");
        assert_eq!(bmi1::myers_step_andn(eq, vp, vn), expected, "bmi1 andn");
        assert_eq!(x86_64::myers_step(eq, vp, vn), expected, "x86_64");
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("bmi1") {
            assert_eq!(unsafe { bmi1::myers_step(eq, vp, vn) }, expected, "bmi1");
        }
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        assert_eq!(apple::myers_step(eq, vp, vn), expected, "apple");
    }
}

/// The pipeline of copy `X` of the hot path on every string kind: the
/// Levenshtein and indel distances of `a` and `b` (`a` no longer), as
/// UCS-1 (pure ASCII and Latin-1 tables), UCS-2, UCS-4 and mixed kinds where
/// the code points fit.
fn copy_metrics<X: Isa>(a: &[char], b: &[char]) -> Vec<(usize, usize)> {
    let max = a.iter().chain(b).map(|&c| c as u32).max().unwrap_or(0);
    let (u8a, u8b): (Vec<u8>, Vec<u8>) = (
        a.iter().map(|&c| c as u8).collect(),
        b.iter().map(|&c| c as u8).collect(),
    );
    let (u16a, u16b): (Vec<u16>, Vec<u16>) = (
        a.iter().map(|&c| c as u16).collect(),
        b.iter().map(|&c| c as u16).collect(),
    );
    let (u32a, u32b): (Vec<u32>, Vec<u32>) = (
        a.iter().map(|&c| c as u32).collect(),
        b.iter().map(|&c| c as u32).collect(),
    );
    let mut out = Vec::new();
    if max < 0x80 {
        out.push((
            compute_u8::<Lev<X>, true>(&u8a, &u8b),
            compute_u8::<Lcs<X>, true>(&u8a, &u8b),
        ));
    }
    if max < 0x100 {
        out.push((
            compute_u8::<Lev<X>, false>(&u8a, &u8b),
            compute_u8::<Lcs<X>, false>(&u8a, &u8b),
        ));
        out.push((
            compute_sorted_mixed::<Lev<X>, _, _>(&u8a, &u32b),
            compute_sorted_mixed::<Lcs<X>, _, _>(&u8a, &u32b),
        ));
        out.push((
            compute_sorted_mixed::<Lev<X>, _, _>(&u16a, &u8b),
            compute_sorted_mixed::<Lcs<X>, _, _>(&u16a, &u8b),
        ));
    }
    if max < 0x10000 {
        out.push((
            compute_sorted::<Lev<X>, _>(&u16a, &u16b),
            compute_sorted::<Lcs<X>, _>(&u16a, &u16b),
        ));
        out.push((
            compute_sorted_mixed::<Lev<X>, _, _>(&u16a, &u32b),
            compute_sorted_mixed::<Lcs<X>, _, _>(&u16a, &u32b),
        ));
    }
    out.push((
        compute_sorted::<Lev<X>, _>(&u32a, &u32b),
        compute_sorted::<Lcs<X>, _>(&u32a, &u32b),
    ));
    out
}

/// Both copies of the hot path must agree with the oracle on every path of
/// the pipeline (tiny patterns, mbleven, one word, two to eight words, bands
/// and full matrices beyond), every string kind and both metrics.  The tests
/// build for the x86-64-v1 ISA (`.cargo/config.toml`), so the `Baseline`
/// copy, which pre-x86-64-v2 CPUs run, is tested on every CPU; the
/// x86-64-v2 copy runs where the CPU has its features.
#[test]
fn copies_match_oracle() {
    let mut state = 0xC0B1_E5C0_1234_5678_u64;
    let alphabets: [&[char]; 3] = [
        &['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        &['a', 'é', 'ü', 'ø', 'å', 'ß'],
        &['日', '本', 'a', '語', '😀', 'あ'],
    ];
    for alpha in alphabets {
        for &(m, edits) in &[
            (5usize, 2usize),
            (12, 1),
            (12, 9),
            (40, 3),
            (63, 20),
            (100, 2),
            (200, 40),
            (300, 300),
            (500, 7),
            (700, 10),
            (700, 700),
        ] {
            let a: Vec<char> = (0..m)
                .map(|_| alpha[lcg(&mut state) % alpha.len()])
                .collect();
            let mut b = a.clone();
            for _ in 0..edits {
                let i = lcg(&mut state) % b.len();
                match lcg(&mut state) % 3 {
                    0 => b[i] = alpha[lcg(&mut state) % alpha.len()],
                    1 => b.insert(i, alpha[lcg(&mut state) % alpha.len()]),
                    _ => {
                        b.remove(i);
                    }
                }
            }
            let (s, l) = if a.len() <= b.len() {
                (&a, &b)
            } else {
                (&b, &a)
            };
            let expected = (naive(s, l), naive_indel(s, l));
            for got in copy_metrics::<Baseline>(s, l) {
                assert_eq!(got, expected, "Baseline m={} n={}", s.len(), l.len());
            }
            #[cfg(target_arch = "x86_64")]
            if super::x86_64::v2::detected() {
                for got in copy_metrics::<super::x86_64::v2::V2>(s, l) {
                    assert_eq!(got, expected, "V2 m={} n={}", s.len(), l.len());
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn small_ub_v2_matches_baseline() {
    if super::x86_64::v2::detected() {
        use super::x86_64::v2::V2;
        use crate::small_ub;

        for len in [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, 128, 256] {
            let a: Vec<u8> = (0..len).map(|i| b'a' + (i % 26) as u8).collect();
            let mut b = a.clone();
            assert_eq!(
                small_ub::<V2, _>(&a, &b),
                small_ub::<Baseline, _>(&a, &b),
                "identical len={len}"
            );

            if len > 0 {
                // 1 mismatch at start, middle, end
                for pos in [0, len / 2, len - 1] {
                    b[pos] ^= 1;
                    assert_eq!(
                        small_ub::<V2, _>(&a, &b),
                        small_ub::<Baseline, _>(&a, &b),
                        "1 diff len={len} pos={pos}"
                    );
                    b[pos] ^= 1;
                }
            }

            if len >= 3 {
                // 3 mismatches
                let (p0, p1, p2) = (0, len / 2, len - 1);
                b[p0] ^= 1;
                b[p1] ^= 1;
                b[p2] ^= 1;
                assert_eq!(
                    small_ub::<V2, _>(&a, &b),
                    small_ub::<Baseline, _>(&a, &b),
                    "3 diffs len={len}"
                );
                // 4 mismatches (exceeds MBLEVEN_MAX)
                if len >= 4 {
                    let p3 = len / 4;
                    b[p3] ^= 1;
                    assert_eq!(small_ub::<V2, _>(&a, &b), None, "4 diffs len={len}");
                    assert_eq!(small_ub::<V2, _>(&a, &b), small_ub::<Baseline, _>(&a, &b),);
                    b[p3] ^= 1;
                }
                b[p0] ^= 1;
                b[p1] ^= 1;
                b[p2] ^= 1;
            }

            // Also test u16 and u32
            let a16: Vec<u16> = a.iter().map(|&x| x as u16 + 0x300).collect();
            let mut b16 = a16.clone();
            if len > 0 {
                b16[len - 1] += 1;
                assert_eq!(
                    small_ub::<V2, _>(&a16, &b16),
                    small_ub::<Baseline, _>(&a16, &b16),
                    "u16 len={len}"
                );
            }

            let a32: Vec<u32> = a.iter().map(|&x| x as u32 + 0x10000).collect();
            let mut b32 = a32.clone();
            if len > 0 {
                b32[len - 1] += 1;
                assert_eq!(
                    small_ub::<V2, _>(&a32, &b32),
                    small_ub::<Baseline, _>(&a32, &b32),
                    "u32 len={len}"
                );
            }
        }
    }
}
