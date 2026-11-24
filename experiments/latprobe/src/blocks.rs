//! The `myers_step` blocks the probe scores, as asm copies of lev's forms.
//!
//! To score a candidate: add a function with the signature
//! `fn(eq: u64, vp: u64, vn: u64) -> (u64, u64)` and list it in
//! `for_each_block!` at the bottom. `main` checks every listed block against
//! [`reference`] before timing it.

use std::arch::asm;

/// `portable::myers_step` (src/kernels/portable.rs), the reference.
#[inline(always)]
pub fn reference(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let sum = (eq & vp).wrapping_add(vp);
    let ph = vn | !(sum | (vp | eq));
    let mh = vp & (eq | !sum);
    let xv = eq | vn;
    let ph_s = (ph << 1) | 1;
    ((mh << 1) | !(xv | ph_s), ph_s & xv)
}

/// `aarch64::apple::myers_step_chain6`, the Apple form for loops that compute
/// the match masks with integer ops (the main form before `chain5_ge`):
/// 15 ops, chain 6.
#[inline(always)]
pub fn chain6(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "and  {sum}, {eq}, {vp}",
            "orr  {u}, {eq}, {vp}",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {vns}, {vn}, #1",
            "add  {sum}, {sum}, {vp}",
            "orr  {m}, {xv}, {vns}",
            "orr  {x}, {u}, {sum}",
            "orn  {t}, {eq}, {sum}",
            "lsl  {xs}, {x}, #1",
            "and  {mh}, {t}, {vp}",
            "lsl  {mhs}, {mh}, #1",
            "bic  {s}, {xs}, {vns}",
            "bic  {a}, {xs}, {m}",
            "orr  {vp}, {mhs}, {a}",
            "bic  {vn}, {xv}, {s}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            sum = out(reg) _, u = out(reg) _, xv = out(reg) _, vns = out(reg) _,
            m = out(reg) _, x = out(reg) _, t = out(reg) _, xs = out(reg) _,
            mh = out(reg) _, mhs = out(reg) _, s = out(reg) _, a = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// The falsified chain-5 candidate (PR #288): 18 ops, chain 5.
#[inline(always)]
pub fn chain5(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {e}, {eq}, #1",
            "orr  {eq}, {eq}, {vn}",
            "lsl  {p}, {vp}, #1",
            "lsl  {n}, {vn}, #1",
            "and  {t}, {e}, {p}",
            "add  {vn}, {t}, {p}",
            "eon  {vp}, {eq}, {p}",
            "orr  {p}, {p}, {e}",
            "orr  {e}, {e}, {n}",
            "bic  {vp}, {vp}, {e}",
            "orn  {p}, {n}, {p}",
            "and  {p}, {eq}, {p}",
            "orr  {e}, {eq}, {n}",
            "orn  {t}, {t}, {e}",
            "bic  {vp}, {vp}, {vn}",
            "bic  {vn}, {vn}, {n}",
            "eor  {vp}, {t}, {vp}",
            "bic  {vn}, {p}, {vn}",
            eq = inout(reg) eq => _, vp = inout(reg) vp, vn = inout(reg) vn,
            e = out(reg) _, p = out(reg) _, n = out(reg) _, t = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

// Run 2 of the myers_step search (2026-11-24). "Level order": each op after
// every op that becomes ready in an earlier cycle. Table-eq scores of the
// final run are in experiments/results.jsonl.

/// chain6 in critical-first program order: 15 ops, chain 6. Idea: the
/// critical ops are older than the off-chain ops ready in the same cycle.
/// No gain: chain6 is already in level order.
#[inline(always)]
pub fn c6_crit(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "and  {s1}, {eq}, {vp}",
            "add  {sum}, {s1}, {vp}",
            "orr  {u}, {eq}, {vp}",
            "orr  {x}, {u}, {sum}",
            "orn  {t}, {eq}, {sum}",
            "lsl  {xs}, {x}, #1",
            "and  {mh}, {t}, {vp}",
            "lsl  {mhs}, {mh}, #1",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {vns}, {vn}, #1",
            "orr  {m}, {xv}, {vns}",
            "bic  {a}, {xs}, {m}",
            "orr  {vp}, {mhs}, {a}",
            "bic  {s}, {xs}, {vns}",
            "bic  {vn}, {xv}, {s}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            s1 = out(reg) _, sum = out(reg) _, u = out(reg) _, x = out(reg) _, t = out(reg) _, xs = out(reg) _,
            mh = out(reg) _, mhs = out(reg) _, xv = out(reg) _, vns = out(reg) _, m = out(reg) _, a = out(reg) _,
            s = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// chain5 in critical-first program order: 18 ops, chain 5. Worse than
/// chain5: zero-slack ops of the `vn'` path end up late in program order.
#[inline(always)]
pub fn c5_crit(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {e}, {eq}, #1",
            "lsl  {p}, {vp}, #1",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {n}, {vn}, #1",
            "and  {t}, {e}, {p}",
            "eon  {xe}, {xv}, {p}",
            "orr  {en}, {e}, {n}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "bic  {y}, {dd}, {sh}",
            "orr  {pe}, {p}, {e}",
            "orn  {o}, {n}, {pe}",
            "and  {f0}, {xv}, {o}",
            "bic  {sn}, {sh}, {n}",
            "orr  {m}, {xv}, {n}",
            "orn  {q1}, {t}, {m}",
            "eor  {vp}, {q1}, {y}",
            "bic  {vn}, {f0}, {sn}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            e = out(reg) _, p = out(reg) _, xv = out(reg) _, n = out(reg) _, t = out(reg) _, xe = out(reg) _,
            en = out(reg) _, sh = out(reg) _, dd = out(reg) _, y = out(reg) _, pe = out(reg) _, o = out(reg) _,
            f0 = out(reg) _, sn = out(reg) _, m = out(reg) _, q1 = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// chain5 in strict level order (each op after every op of the cycle
/// before it): 18 ops, chain 5. Idea: program order = order of readiness.
#[inline(always)]
pub fn c5_level(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {e}, {eq}, #1",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {p}, {vp}, #1",
            "lsl  {n}, {vn}, #1",
            "and  {t}, {e}, {p}",
            "eon  {xe}, {xv}, {p}",
            "orr  {pe}, {p}, {e}",
            "orr  {en}, {e}, {n}",
            "orr  {m}, {xv}, {n}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "orn  {o}, {n}, {pe}",
            "orn  {q1}, {t}, {m}",
            "and  {f0}, {xv}, {o}",
            "bic  {y}, {dd}, {sh}",
            "bic  {sn}, {sh}, {n}",
            "eor  {vp}, {q1}, {y}",
            "bic  {vn}, {f0}, {sn}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            e = out(reg) _, xv = out(reg) _, p = out(reg) _, n = out(reg) _, t = out(reg) _, xe = out(reg) _,
            pe = out(reg) _, en = out(reg) _, m = out(reg) _, sh = out(reg) _, dd = out(reg) _, o = out(reg) _,
            q1 = out(reg) _, f0 = out(reg) _, y = out(reg) _, sn = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// Chain 5, 19 ops (5 lsl), level order: `vp' = B ^ (D & S)` with
/// `B = P | (E & !N & !xv)`, `E & !N = (eq & !vn) << 1` shared with `f0`,
/// `E | N = xv << 1`, `t = (eq & vp) << 1`. Idea: lsl where chain5 has ALU
/// ops. Critical-first order of the same ops: 6.8 table eq.
#[inline(always)]
pub fn c5d(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {p}, {vp}, #1",
            "and  {t1}, {eq}, {vp}",
            "orr  {xv}, {eq}, {vn}",
            "bic  {q}, {eq}, {vn}",
            "lsl  {n}, {vn}, #1",
            "lsl  {t}, {t1}, #1",
            "eon  {xe}, {xv}, {p}",
            "lsl  {en}, {xv}, #1",
            "lsl  {enn}, {q}, #1",
            "bic  {xvp}, {xv}, {p}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "bic  {rr}, {enn}, {xv}",
            "bic  {f0}, {xvp}, {enn}",
            "and  {y}, {dd}, {sh}",
            "orr  {bb}, {p}, {rr}",
            "bic  {sn}, {sh}, {n}",
            "eor  {vp}, {bb}, {y}",
            "bic  {vn}, {f0}, {sn}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            p = out(reg) _, t1 = out(reg) _, xv = out(reg) _, q = out(reg) _, n = out(reg) _, t = out(reg) _,
            xe = out(reg) _, en = out(reg) _, enn = out(reg) _, xvp = out(reg) _, sh = out(reg) _, dd = out(reg) _,
            rr = out(reg) _, f0 = out(reg) _, y = out(reg) _, bb = out(reg) _, sn = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// Chain 5, 19 ops (5 lsl), level order: chain5 with `E | N = xv << 1`,
/// `t = (eq & vp) << 1`, `f0 = xv & !(((eq | vp) & !vn) << 1)`. Idea: at
/// most 3 ALU ops per cycle. Critical-first order: 6.9 table eq.
#[inline(always)]
pub fn c5f(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {p}, {vp}, #1",
            "and  {t1}, {eq}, {vp}",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {n}, {vn}, #1",
            "orr  {u}, {eq}, {vp}",
            "lsl  {t}, {t1}, #1",
            "eon  {xe}, {xv}, {p}",
            "lsl  {en}, {xv}, #1",
            "orr  {m}, {xv}, {n}",
            "bic  {w}, {u}, {vn}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "orn  {aa}, {t}, {m}",
            "lsl  {pw}, {w}, #1",
            "bic  {y}, {dd}, {sh}",
            "bic  {sn}, {sh}, {n}",
            "bic  {f0}, {xv}, {pw}",
            "eor  {vp}, {aa}, {y}",
            "bic  {vn}, {f0}, {sn}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            p = out(reg) _, t1 = out(reg) _, xv = out(reg) _, n = out(reg) _, u = out(reg) _, t = out(reg) _,
            xe = out(reg) _, en = out(reg) _, m = out(reg) _, w = out(reg) _, sh = out(reg) _, dd = out(reg) _,
            aa = out(reg) _, pw = out(reg) _, y = out(reg) _, sn = out(reg) _, f0 = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// c5f with the shift after the add (`S = sum << 1`), level order: 20 ops
/// (6 lsl), chain 5. Idea: the critical op of cycle 3 is an lsl.
/// Critical-first order: 6.9 table eq.
#[inline(always)]
pub fn c5g(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "and  {t1}, {eq}, {vp}",
            "lsl  {p}, {vp}, #1",
            "orr  {xv}, {eq}, {vn}",
            "orr  {u}, {eq}, {vp}",
            "lsl  {n}, {vn}, #1",
            "add  {sum}, {t1}, {vp}",
            "eon  {xe}, {xv}, {p}",
            "lsl  {en}, {xv}, #1",
            "bic  {w}, {u}, {vn}",
            "lsl  {t}, {t1}, #1",
            "orr  {m}, {xv}, {n}",
            "lsl  {sh}, {sum}, #1",
            "bic  {dd}, {xe}, {en}",
            "lsl  {pw}, {w}, #1",
            "orn  {aa}, {t}, {m}",
            "bic  {y}, {dd}, {sh}",
            "bic  {sn}, {sh}, {n}",
            "bic  {f0}, {xv}, {pw}",
            "eor  {vp}, {aa}, {y}",
            "bic  {vn}, {f0}, {sn}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            t1 = out(reg) _, p = out(reg) _, xv = out(reg) _, u = out(reg) _, n = out(reg) _, sum = out(reg) _,
            xe = out(reg) _, en = out(reg) _, w = out(reg) _, t = out(reg) _, m = out(reg) _, sh = out(reg) _,
            dd = out(reg) _, pw = out(reg) _, aa = out(reg) _, y = out(reg) _, sn = out(reg) _, f0 = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// c5_level with `E | N = xv << 1` (was `E | N`): 18 ops, chain 5. Idea:
/// lsl of a fresh value instead of an ALU op on a cold one; `N` gets slack.
#[inline(always)]
pub fn c5_level_en(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {e}, {eq}, #1",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {p}, {vp}, #1",
            "lsl  {n}, {vn}, #1",
            "and  {t}, {e}, {p}",
            "eon  {xe}, {xv}, {p}",
            "orr  {pe}, {p}, {e}",
            "lsl  {en}, {xv}, #1",
            "orr  {m}, {xv}, {n}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "orn  {o}, {n}, {pe}",
            "orn  {q1}, {t}, {m}",
            "and  {f0}, {xv}, {o}",
            "bic  {y}, {dd}, {sh}",
            "bic  {sn}, {sh}, {n}",
            "eor  {vp}, {q1}, {y}",
            "bic  {vn}, {f0}, {sn}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            e = out(reg) _, xv = out(reg) _, p = out(reg) _, n = out(reg) _, t = out(reg) _, xe = out(reg) _,
            pe = out(reg) _, en = out(reg) _, m = out(reg) _, sh = out(reg) _, dd = out(reg) _, o = out(reg) _,
            q1 = out(reg) _, f0 = out(reg) _, y = out(reg) _, sn = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// c5_level with `vn' = (xv & N) | (G & !S)`, `G = xv & !P & !(E | N)`:
/// 18 ops, chain 5. Idea: the `vn'` path ends in 2 ops after `S` like
/// `vp'`, and cycle 4 has 2 ops (chain5: 3).
#[inline(always)]
pub fn c5_level_g(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {e}, {eq}, #1",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {p}, {vp}, #1",
            "lsl  {n}, {vn}, #1",
            "and  {t}, {e}, {p}",
            "eon  {xe}, {xv}, {p}",
            "orr  {en}, {e}, {n}",
            "bic  {xp}, {xv}, {p}",
            "and  {xn}, {xv}, {n}",
            "orr  {m}, {xv}, {n}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "bic  {g}, {xp}, {en}",
            "orn  {q1}, {t}, {m}",
            "bic  {y}, {dd}, {sh}",
            "bic  {gs}, {g}, {sh}",
            "eor  {vp}, {q1}, {y}",
            "orr  {vn}, {xn}, {gs}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            e = out(reg) _, xv = out(reg) _, p = out(reg) _, n = out(reg) _, t = out(reg) _, xe = out(reg) _,
            en = out(reg) _, xp = out(reg) _, xn = out(reg) _, m = out(reg) _, sh = out(reg) _, dd = out(reg) _,
            g = out(reg) _, q1 = out(reg) _, y = out(reg) _, gs = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// chain5_ge with `t = (eq & vp) << 1` instead of `E & P`: 18 ops, no
/// loop-invariant op, chain 5. Worse: the op that reads only `eq` was
/// cheaper than the one that reads `vp`.
#[inline(always)]
pub fn c5_ge_noeq(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {p}, {vp}, #1",
            "and  {t1}, {eq}, {vp}",
            "orr  {xv}, {eq}, {vn}",
            "lsl  {n}, {vn}, #1",
            "lsl  {t}, {t1}, #1",
            "eon  {xe}, {xv}, {p}",
            "lsl  {en}, {xv}, #1",
            "bic  {xp}, {xv}, {p}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "bic  {g}, {xp}, {en}",
            "orr  {m}, {xv}, {n}",
            "bic  {y}, {dd}, {sh}",
            "bic  {gs}, {g}, {sh}",
            "orn  {q1}, {t}, {m}",
            "and  {xn}, {xv}, {n}",
            "eor  {vp}, {q1}, {y}",
            "orr  {vn}, {xn}, {gs}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            p = out(reg) _, t1 = out(reg) _, xv = out(reg) _, n = out(reg) _, t = out(reg) _, xe = out(reg) _,
            en = out(reg) _, xp = out(reg) _, sh = out(reg) _, dd = out(reg) _, g = out(reg) _, m = out(reg) _,
            y = out(reg) _, gs = out(reg) _, q1 = out(reg) _, xn = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// chain5_ge with `N` folded into `m = xv | (vn << 1)` and
/// `xn = xv & (vn << 1)` (shifted operands, 2 cycles): 17 ops, chain 5.
/// Worse: a shifted-operand op costs more issue bandwidth than lsl + op.
#[inline(always)]
pub fn c5_ge_shifted(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {e}, {eq}, #1",
            "lsl  {p}, {vp}, #1",
            "orr  {xv}, {eq}, {vn}",
            "and  {t}, {e}, {p}",
            "eon  {xe}, {xv}, {p}",
            "lsl  {en}, {xv}, #1",
            "bic  {xp}, {xv}, {p}",
            "add  {sh}, {t}, {p}",
            "bic  {dd}, {xe}, {en}",
            "bic  {g}, {xp}, {en}",
            "orr  {m}, {xv}, {vn}, lsl #1",
            "bic  {y}, {dd}, {sh}",
            "bic  {gs}, {g}, {sh}",
            "orn  {q1}, {t}, {m}",
            "and  {xn}, {xv}, {vn}, lsl #1",
            "eor  {vp}, {q1}, {y}",
            "orr  {vn}, {xn}, {gs}",
            eq = in(reg) eq, vp = inout(reg) vp, vn = inout(reg) vn,
            e = out(reg) _, p = out(reg) _, xv = out(reg) _, t = out(reg) _, xe = out(reg) _, en = out(reg) _,
            xp = out(reg) _, sh = out(reg) _, dd = out(reg) _, g = out(reg) _, m = out(reg) _, y = out(reg) _,
            gs = out(reg) _, q1 = out(reg) _, xn = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// Chain 5, 18 ops, 7 registers: c5_level_g with `E | N = xv << 1`, in
/// level order with the ops that have slack (`m`, `A`, `xn`) last in
/// their cycle. Idea: fewer ops ready with the critical ones per cycle
/// and program order = order of readiness. Same asm as
/// `aarch64::apple::myers_step`.
#[inline(always)]
pub fn chain5_ge(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (vp, vn);
    unsafe {
        asm!(
            "lsl  {a}, {eq}, #1",     // E = eq << 1
            "lsl  {vp}, {vp}, #1",    // P = vp << 1
            "orr  {eq}, {eq}, {vn}",  // xv = eq | vn
            "lsl  {vn}, {vn}, #1",    // N = vn << 1
            "and  {a}, {a}, {vp}",    // t = E & P
            "eon  {b}, {eq}, {vp}",   // xe = xv ^ !P
            "lsl  {c}, {eq}, #1",     // en = xv << 1 = E | N
            "bic  {d}, {eq}, {vp}",   // xp = xv & !P
            "add  {vp}, {a}, {vp}",   // S = t + P
            "bic  {b}, {b}, {c}",     // D = xe & !en
            "bic  {d}, {d}, {c}",     // G = xp & !en
            "orr  {c}, {eq}, {vn}",   // m = xv | N
            "bic  {b}, {b}, {vp}",    // y = D & !S
            "bic  {d}, {d}, {vp}",    // g = G & !S
            "orn  {a}, {a}, {c}",     // A = t | !m
            "and  {vn}, {eq}, {vn}",  // xn = xv & N
            "eor  {vp}, {a}, {b}",    // vp' = A ^ y
            "orr  {vn}, {vn}, {d}",   // vn' = xn | g
            eq = inout(reg) eq => _, vp = inout(reg) vp, vn = inout(reg) vn,
            a = out(reg) _, b = out(reg) _, c = out(reg) _, d = out(reg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (vp, vn)
}

/// Calls `$m!(name, function)` for every block the probe scores.
macro_rules! for_each_block {
    ($m:ident) => {
        $m!("chain6 (few-ops form)", blocks::chain6);
        $m!("chain5 (falsified)", blocks::chain5);
        $m!("c6_crit", blocks::c6_crit);
        $m!("c5_crit", blocks::c5_crit);
        $m!("c5_level", blocks::c5_level);
        $m!("c5d", blocks::c5d);
        $m!("c5f", blocks::c5f);
        $m!("c5g", blocks::c5g);
        $m!("c5_level_en", blocks::c5_level_en);
        $m!("c5_level_g", blocks::c5_level_g);
        $m!("c5_ge_noeq", blocks::c5_ge_noeq);
        $m!("c5_ge_shifted", blocks::c5_ge_shifted);
        $m!("chain5_ge (current)", blocks::chain5_ge);
    };
}
pub(crate) use for_each_block;
