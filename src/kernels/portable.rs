//! Reference forms of the target-specific kernels: plain Rust, selected on
//! every target that has no form of its own.

use super::{band_gives_up, delta, EDGE_ROW0, SMALL_BAND_CHECK, SMALL_BAND_FIRST};
use crate::{lcs_len, lcs_step, CodeUnit, Kernel, LargeCtx};

/// [`super::word_u8`] with the match masks in a direct peq table of
/// `SLOTS` rows, zeroed and filled once per call.
#[inline(always)]
pub(crate) fn word_u8<K: Kernel, const SLOTS: usize>(pattern: &[u8], text: &[u8]) -> usize {
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

/// One column of the single-word Levenshtein recurrence: returns the new
/// `(vp, vn)`.
#[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
pub(crate) fn myers_step(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let sum = (eq & vp).wrapping_add(vp);
    let ph = vn | !(sum | (vp | eq));
    let mh = vp & (eq | !sum);
    let xv = eq | vn;
    let ph_s = (ph << 1) | 1;
    ((mh << 1) | !(xv | ph_s), ph_s & xv)
}

/// [`super::multiword`] in Hyyrö's form, shifting each word's top bits into
/// the next word as 0/1 values.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub(crate) fn multiword<const W: usize, I: Iterator<Item = [u64; W]>>(
    m: usize,
    pm_iter: I,
    mut edges: Option<&mut [u8]>,
) -> isize {
    let mut vp = [!0u64; W];
    let mut vn = [0u64; W];
    for (j, pm_row) in pm_iter.enumerate() {
        let e = edges.as_ref().map_or(EDGE_ROW0, |e| e[j]);
        let mut carry = e & 1 != 0;
        let (mut prev_hp, mut prev_hn) = ((e >> 1 & 1) as u64, (e >> 2) as u64);
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
        if let Some(e) = edges.as_deref_mut() {
            e[j] = carry as u8 | (prev_hp as u8) << 1 | (prev_hn as u8) << 2;
        }
    }
    delta(vp, vn, m)
}

/// The vertical deltas `(vp, vn)` that a column of [`small_band`] leaves,
/// from `x = sum | vp | eq` and `mh` of its step, its vertical inputs moved
/// up a bit (`xvs`), and the `vn` it started from.
///
/// With `ph = vn | !x` they are `mh | !(xvs | ph)` and `ph & xvs`.  `x` is
/// the last input to be ready (it waits for the column addition), so the
/// form here gives it one op before `vp` instead of three: `xvs | vn` is
/// ready while the addition runs.  On AArch64 the chain from `vp` to `vp`
/// is then and, add, orr, bic, orr (5 cycles, against 6 with `ph`).  The
/// x86-64 baseline the band compiles for has no and-not, so `!` costs an
/// op: there the chain falls from 8 cycles to 6, as `mh` and the new `vn`
/// each still wait for a `not` after the addition (5 with BMI1).
#[inline(always)]
pub(crate) fn band_deltas(x: u64, mh: u64, xvs: u64, vn: u64) -> (u64, u64) {
    (mh | (x & !(xvs | vn)), (vn | !x) & xvs)
}

/// [`super::small_band`]: the band's 64 diagonals in one word, whose
/// window slides one pattern row down per text column.
///
/// Bit `b` of column `j` (1-based) holds pattern row `j - top + b`, so the
/// state of column `j - 1`, already in that window, enters the step for
/// column `j`, and the step leaves column `j` one row lower: the vertical
/// inputs move up a bit (`xv >> 1`) instead of the horizontal ones moving
/// down (`ph << 1`).  Rows above row 1 (bits below `top` in the first
/// columns) start with vertical delta 0 and never match, so they stay
/// copies of row 0 (`D(0, j) = j`) and supply its `+1` horizontal delta.
/// Cells above the window count as unreachable (no carry into bit 0); the
/// row entering at the bottom assumes a mismatch (0 shifted in at bit 63).
/// Both only raise cells, so every computed cell bounds its true value
/// from above, and cells of paths inside the band are exact.
///
/// # Safety
///
/// `row(c) * w + w <= data.len()` for every `c` in `text`.
#[inline(never)]
pub(crate) unsafe fn small_band<T: CodeUnit, R: Fn(T) -> usize>(
    data: &[u64],
    w: usize,
    m: usize,
    text: &[T],
    row: R,
) -> Result<usize, (usize, usize)> {
    let n = text.len();
    debug_assert!(0 < m && m <= n && n - m < 64 && m <= 64 * w);
    let s = n - m;
    // Diagonals `j - i` in `[-k_lo, s + k_hi]`, the top one at bit 0.  A
    // path of cost d stays within `floor((d - s) / 2)` diagonals of
    // `[0, s]` (Ukkonen), so the band certifies every d up to `limit`.
    let k_lo = (63 - s) / 2;
    let k_hi = 63 - s - k_lo;
    let top = s + k_hi;
    let limit = s + 2 * k_lo + 1;
    // The top cell of a column grows by one except where its diagonal step
    // is free, and with no carry into bit 0 that is where `eq | vn` has
    // bit 0 set.  Diagonal `s`, which ends in (m, n), lies `k_hi` rows
    // lower: its cell is the top one plus the vertical deltas in `low`.
    let mut free = 0usize;
    let low = (1u64 << k_hi) - 1;
    let (mut vp, mut vn) = (!0u64 << top, 0u64);
    // Pattern index of bit 0 in the next column, plus 64.
    let mut pos = 64 - top;
    let (mut j, mut check) = (0, SMALL_BAND_FIRST);
    // The columns of `$chunk`, whose match masks are bits `$sh..` of the
    // word pair `$words` read from the row at `$base`.
    macro_rules! columns {
        ($chunk:expr, $sh:expr, |$base:ident| $words:expr) => {
            for (k, &c) in ($sh..).zip($chunk) {
                let $base = row(c) * w;
                let (lo, hi): (u64, u64) = $words;
                // Funnel shift by `k < 64`: `!k & 63 = 63 - k`, and
                // `hi << 1 << (63 - k)` is `hi << (64 - k)` without the
                // overflow at `k = 0`.
                let eq = lo.wrapping_shr(k as u32) | (hi << 1).wrapping_shl(!(k as u32));
                let sum = (eq & vp).wrapping_add(vp);
                let x = sum | (vp | eq);
                let mh = vp & (eq | !sum);
                let xv = eq | vn;
                let xvs = xv >> 1;
                free += (xv & 1) as usize;
                (vp, vn) = band_deltas(x, mh, xvs, vn);
            }
        };
    }
    loop {
        // Bit 0 of the next `len` columns lies in word `q - 1` of the peq
        // rows (`-1` and `w` read as zero), `sh` bits up.
        let (q, sh) = (pos / 64, pos % 64);
        debug_assert!(q <= w);
        let len = (64 - sh).min(check - j).min(n - j);
        // SAFETY: `j + len <= n`.
        let chunk = unsafe { text.get_unchecked(j..j + len) };
        // SAFETY (all three): the row at `base` has `w` words in `data`
        // (caller contract), and `0 < q < w` in the middle arm.
        if q == 0 {
            columns!(chunk, sh, |base| (0, unsafe { *data.get_unchecked(base) }));
        } else if q < w {
            columns!(chunk, sh, |base| unsafe {
                let p = data.as_ptr().add(base + q - 1);
                (*p, *p.add(1))
            });
        } else {
            columns!(chunk, sh, |base| (
                unsafe { *data.get_unchecked(base + w - 1) },
                0
            ));
        }
        j += len;
        pos += len;
        if j < check && j < n {
            continue;
        }
        // Checks double from `SMALL_BAND_FIRST` columns to every
        // `SMALL_BAND_CHECK`.
        check = if j < SMALL_BAND_CHECK {
            2 * j
        } else {
            j - j % SMALL_BAND_CHECK + SMALL_BAND_CHECK
        };
        // The cell of diagonal `s` in column j bounds the band's result
        // from below: every path crosses column j, a cell of it on
        // diagonal δ is at least `score - |δ - s|` (vertical deltas are
        // ±1), and finishing from there takes `|δ - s|` more edits.
        let score = j - free + (vp & low).count_ones() as usize - (vn & low).count_ones() as usize;
        if band_gives_up(score, s, limit, j, n) {
            return Err((j, score));
        }
        if j == n {
            return Ok(score);
        }
    }
}

/// [`super::small_band2`]: [`small_band`] over 128 diagonals in two words,
/// the low one on top.  The words share the column addition (its carry runs
/// from word 0 into word 1) and the move of the vertical inputs one bit up,
/// which pulls bit 0 of word 1 into bit 63 of word 0; every other step is
/// bitwise.  The match masks of a column are bits of three adjacent peq
/// words, funnel-shifted into two.
///
/// The step keeps the `ph` form that [`band_deltas`] replaces in
/// [`small_band`].  The carry into word 1 and the bit word 1 moves into
/// `xvs0` link the two words' chains, so that form would shorten the chain
/// only to about 5.5 cycles per column on Apple cores (from 6) and 7.5 on
/// x86-64 (from 8), for one more op per word in a loop whose 40-65
/// instructions per column already about fill the ALUs at that pace.  On
/// Neoverse it would lengthen the chain, putting the 3-cycle `extr` that
/// moves bit 0 of word 1 into `xvs0` on it.
///
/// # Safety
///
/// `row(c) * w + w <= data.len()` for every `c` in `text`.
#[inline(never)]
pub(crate) unsafe fn small_band2<T: CodeUnit, R: Fn(T) -> usize>(
    data: &[u64],
    w: usize,
    m: usize,
    text: &[T],
    row: R,
) -> Result<usize, (usize, usize)> {
    let n = text.len();
    debug_assert!(0 < m && m <= n && n - m < 128 && m <= 64 * w && w >= 2);
    let s = n - m;
    // Diagonals `j - i` in `[-k_lo, s + k_hi]`, the top one at bit 0 of
    // word 0; the limit follows as in `small_band`.
    let k_lo = (127 - s) / 2;
    let k_hi = 127 - s - k_lo;
    let top = s + k_hi;
    let limit = s + 2 * k_lo + 1;
    // `k_hi <= 64`, so the rows from the top to diagonal `s` lie in word 0.
    let mut free = 0usize;
    let low = u64::MAX.checked_shr(64 - k_hi as u32).unwrap_or(0);
    // `top >= 64`: column 0 is `+1` from bit `top` of the 128 on.
    let (mut vp0, mut vp1, mut vn0, mut vn1) = (0u64, !0u64 << (top - 64), 0u64, 0u64);
    // Pattern index of bit 0 in the next column, plus 128.
    let mut pos = 128 - top;
    let (mut j, mut check) = (0, SMALL_BAND_FIRST);
    // The columns of `$chunk`, whose match masks are bits `$sh..` of the
    // three words `$words` read from the row at `$base`.
    macro_rules! columns {
        ($chunk:expr, $sh:expr, |$base:ident| $words:expr) => {
            for (k, &c) in ($sh..).zip($chunk) {
                let $base = row(c) * w;
                let (a, b, c): (u64, u64, u64) = $words;
                // Funnel shifts by `k < 64`, as in `small_band`.
                let k = k as u32;
                let eq0 = a.wrapping_shr(k) | (b << 1).wrapping_shl(!k);
                let eq1 = b.wrapping_shr(k) | (c << 1).wrapping_shl(!k);
                let (sum0, carry) = (eq0 & vp0).overflowing_add(vp0);
                let (sum1, _) = (eq1 & vp1).carrying_add(vp1, carry);
                let ph0 = vn0 | !(sum0 | (vp0 | eq0));
                let ph1 = vn1 | !(sum1 | (vp1 | eq1));
                let mh0 = vp0 & (eq0 | !sum0);
                let mh1 = vp1 & (eq1 | !sum1);
                let (xv0, xv1) = (eq0 | vn0, eq1 | vn1);
                let (xvs0, xvs1) = (xv0 >> 1 | xv1 << 63, xv1 >> 1);
                free += (xv0 & 1) as usize;
                vp0 = mh0 | !(xvs0 | ph0);
                vn0 = ph0 & xvs0;
                vp1 = mh1 | !(xvs1 | ph1);
                vn1 = ph1 & xvs1;
            }
        };
    }
    loop {
        // Bit 0 of the next `len` columns lies in word `q - 2` of the peq
        // rows (words outside `0..w` read as zero), `sh` bits up.
        let (q, sh) = (pos / 64, pos % 64);
        debug_assert!(q <= w + 1);
        let len = (64 - sh).min(check - j).min(n - j);
        // SAFETY: `j + len <= n`.
        let chunk = unsafe { text.get_unchecked(j..j + len) };
        // SAFETY (all five): the row at `base` has `w >= 2` words in
        // `data` (caller contract), and `2 <= q < w` in the middle arm.
        match q {
            0 => columns!(chunk, sh, |base| (0, 0, unsafe {
                *data.get_unchecked(base)
            })),
            1 => columns!(chunk, sh, |base| unsafe {
                let p = data.as_ptr().add(base);
                (0, *p, *p.add(1))
            }),
            _ if q < w => columns!(chunk, sh, |base| unsafe {
                let p = data.as_ptr().add(base + q - 2);
                (*p, *p.add(1), *p.add(2))
            }),
            _ if q == w => columns!(chunk, sh, |base| unsafe {
                let p = data.as_ptr().add(base + w - 2);
                (*p, *p.add(1), 0)
            }),
            _ => columns!(chunk, sh, |base| (
                unsafe { *data.get_unchecked(base + w - 1) },
                0,
                0
            )),
        }
        j += len;
        pos += len;
        if j < check && j < n {
            continue;
        }
        check = if j < SMALL_BAND_CHECK {
            2 * j
        } else {
            j - j % SMALL_BAND_CHECK + SMALL_BAND_CHECK
        };
        // As in `small_band`.
        let score =
            j - free + (vp0 & low).count_ones() as usize - (vn0 & low).count_ones() as usize;
        if band_gives_up(score, s, limit, j, n) {
            return Err((j, score));
        }
        if j == n {
            return Ok(score);
        }
    }
}

/// [`super::full`] as one loop over all words per column, with the state on
/// the heap.
pub(crate) fn full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    let (m, w) = (ctx.m, ctx.w);
    let (mut vp, mut vn) = (vec![!0u64; w], vec![0u64; w]);
    ctx.for_each_base(|base| {
        let (mut carry, mut prev_hp, mut prev_hn) = (false, 1u64, 0u64);
        for (k, (vp_k, vn_k)) in vp.iter_mut().zip(vn.iter_mut()).enumerate() {
            let (pm, v_p, v_n) = (ctx.peq(base, k), *vp_k, *vn_k);
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
    ctx.n + count(&vp) - count(&vn)
}

/// [`super::lcs_full`] as one loop over all words per column.
// AArch64 selects the strips; the tests still run this form there.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn lcs_full<F: Fn(usize) -> usize>(ctx: &LargeCtx<'_, F>) -> usize {
    let mut s = vec![!0u64; ctx.w];
    ctx.for_each_base(|base| {
        lcs_step(&mut s, false, |k| ctx.peq(base, k));
    });
    lcs_len(&s)
}
