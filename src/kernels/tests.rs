//! Every form of every kernel against the DP oracle or the portable form,
//! on every target, not only on the one that selects it.

use super::aarch64::{self, apple, neoverse};
use super::x86_64::bmi1;
use super::{myers_step, portable, EDGE_ROW0};
use crate::tests::{lcg, mutate, naive, naive_indel, rand_string};
use crate::LargeCtx;

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
        assert_eq!(ctx.run(), d, "run {msg}");
        assert_eq!(aarch64::lcs_full(&ctx), llcs, "aarch64::lcs_full {msg}");
        assert_eq!(portable::lcs_full(&ctx), llcs, "portable::lcs_full {msg}");
        assert_eq!(ctx.lcs_run(), llcs, "lcs_run {msg}");
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
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        assert_eq!(apple::myers_step(eq, vp, vn), expected, "apple");
    }
}
