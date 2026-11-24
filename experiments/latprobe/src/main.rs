//! Latency probe for the ops in lev's Apple `myers_step` on this core.
//!
//! Each single-op test runs a chain of 8 dependent copies of one instruction
//! per loop iteration; the loop counter runs in parallel, so ns per op is the
//! op's latency. `add` is taken as 1 cycle to convert ns to cycles.
//! The block tests run the old (chain 6) and new (chain 5) asm blocks with
//! `vp`/`vn` carried across iterations, once with a constant `eq` and once
//! with `eq` loaded from a 256-entry table indexed by a text byte, like the
//! real loop.

use std::arch::asm;
use std::hint::black_box;
use std::time::Instant;

const ITERS: u64 = 50_000_000;

macro_rules! chain8 {
    ($ins:literal) => {{
        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut x: u64 = black_box(0x1234_5678_9abc_def1);
            let y: u64 = black_box(0x0f0f_0f0f_0f0f_0f0f);
            let t = Instant::now();
            for _ in 0..ITERS {
                unsafe {
                    asm!($ins, $ins, $ins, $ins, $ins, $ins, $ins, $ins,
                         x = inout(reg) x, y = in(reg) y,
                         options(nomem, nostack));
                }
            }
            black_box(x);
            best = best.min(t.elapsed().as_secs_f64() * 1e9 / (ITERS * 8) as f64);
        }
        best
    }};
}

#[inline(always)]
fn step6(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
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

#[inline(always)]
fn step5(eq: u64, vp: u64, vn: u64) -> (u64, u64) {
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

fn block(step: fn(u64, u64, u64) -> (u64, u64), table: Option<(&[u64; 256], &[u8])>) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..5 {
        let (mut vp, mut vn) = (black_box(!0u64), black_box(0u64));
        let eq0 = black_box(0x8421_8421_8421_8421u64);
        let t = Instant::now();
        match table {
            None => {
                for _ in 0..ITERS {
                    (vp, vn) = step(eq0, vp, vn);
                }
            }
            Some((tab, text)) => {
                for i in 0..ITERS as usize {
                    let c = unsafe { *text.get_unchecked(i & (text.len() - 1)) };
                    let eq = unsafe { *tab.get_unchecked(c as usize) };
                    (vp, vn) = step(eq, vp, vn);
                }
            }
        }
        black_box((vp, vn));
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / ITERS as f64);
    }
    best
}

fn main() {
    let add = chain8!("add {x}, {x}, {y}");
    let ops = [
        ("add", add),
        ("and", chain8!("and {x}, {x}, {y}")),
        ("orr", chain8!("orr {x}, {x}, {y}")),
        ("eor", chain8!("eor {x}, {x}, {y}")),
        ("bic (chain via 1st)", chain8!("bic {x}, {x}, {y}")),
        ("bic (chain via inverted 2nd)", chain8!("bic {x}, {y}, {x}")),
        ("orn (chain via 1st)", chain8!("orn {x}, {x}, {y}")),
        ("orn (chain via inverted 2nd)", chain8!("orn {x}, {y}, {x}")),
        ("eon", chain8!("eon {x}, {x}, {y}")),
        ("lsl #1", chain8!("lsl {x}, {x}, #1 // {y}")),
        (
            "orr y, x lsl #1 (shifted chain)",
            chain8!("orr {x}, {y}, {x}, lsl #1"),
        ),
        (
            "orr x, y lsl #1 (plain chain)",
            chain8!("orr {x}, {x}, {y}, lsl #1"),
        ),
    ];
    println!("op latency, cycles (add = 1):");
    for (name, ns) in ops {
        println!("  {name:34} {:.2}  ({ns:.4} ns)", ns / add);
    }

    let mut tab = [0u64; 256];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for v in tab.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = s;
    }
    let text: Vec<u8> = (0..4096u32)
        .map(|i| b'a' + (i.wrapping_mul(2654435761) >> 27) as u8 % 26)
        .collect();
    println!("myers_step block, cycles per column (add = 1):");
    for (name, step) in [
        ("chain6 (current)", step6 as fn(u64, u64, u64) -> (u64, u64)),
        ("chain5 (candidate)", step5),
    ] {
        let c = block(step, None) / add;
        let l = block(step, Some((&tab, &text))) / add;
        println!("  {name:20} const eq {c:.2}   table eq {l:.2}");
    }
    println!("add latency {add:.4} ns => {:.2} GHz", 1.0 / add);
}
