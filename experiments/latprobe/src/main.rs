//! Latency probe for lev's Apple `myers_step` blocks (AArch64 only).
//!
//! `cargo run --release` scores every block in `blocks.rs` in cycles per
//! column, with `vp`/`vn` carried across iterations: once with a constant
//! `eq`, once with `eq` loaded from a 256-entry table indexed by a text byte,
//! like lev's ASCII loop. Each block is monomorphized into its own loop, so
//! nothing but the block, the loads and the loop counter runs per column.
//! `--ops` also prints the latency of single instructions.
//!
//! Cycles are ns divided by the latency of a chain of `add`s, taken as 1
//! cycle. Every number is the best of 7 runs; the spread (worst over best)
//! shows the noise.

mod blocks;

use std::arch::asm;
use std::hint::black_box;
use std::time::Instant;

const ITERS: u64 = 50_000_000;
const RUNS: usize = 7;

/// Best and worst ns per call of `f() / ITERS` over [`RUNS`] runs.
fn time(f: impl Fn() -> (u64, u64)) -> (f64, f64) {
    let (mut best, mut worst) = (f64::MAX, 0f64);
    for _ in 0..RUNS {
        let t = Instant::now();
        black_box(f());
        let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS as f64;
        best = best.min(ns);
        worst = worst.max(ns);
    }
    (best, worst)
}

macro_rules! chain8 {
    ($ins:literal) => {{
        time(|| {
            let mut x: u64 = black_box(0x1234_5678_9abc_def1);
            let y: u64 = black_box(0x0f0f_0f0f_0f0f_0f0f);
            for _ in 0..ITERS {
                unsafe {
                    asm!($ins, $ins, $ins, $ins, $ins, $ins, $ins, $ins,
                         x = inout(reg) x, y = in(reg) y, options(nomem, nostack));
                }
            }
            (x, 0)
        })
        .0 / 8.0
    }};
}

#[inline(never)]
fn run_const<F: Fn(u64, u64, u64) -> (u64, u64)>(step: F, eq: u64, iters: u64) -> (u64, u64) {
    let (mut vp, mut vn) = (!0u64, 0u64);
    for _ in 0..iters {
        (vp, vn) = step(eq, vp, vn);
    }
    (vp, vn)
}

#[inline(never)]
fn run_table<F: Fn(u64, u64, u64) -> (u64, u64)>(
    step: F,
    tab: &[u64; 256],
    text: &[u8; 4096],
    iters: u64,
) -> (u64, u64) {
    let (mut vp, mut vn) = (!0u64, 0u64);
    for i in 0..iters as usize {
        let eq = tab[text[i & 4095] as usize];
        (vp, vn) = step(eq, vp, vn);
    }
    (vp, vn)
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Panics unless `step` matches [`blocks::reference`] on random and edge
/// states with `vp & vn == 0`.
fn check(name: &str, step: impl Fn(u64, u64, u64) -> (u64, u64)) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let edges = [0, 1, 1 << 63, !0, !1, !0 >> 1];
    let mut states: Vec<(u64, u64, u64)> = (0..1_000_000)
        .map(|_| {
            let (eq, vp, vn) = (xorshift(&mut s), xorshift(&mut s), xorshift(&mut s));
            (eq, vp, vn & !vp)
        })
        .collect();
    for eq in edges {
        for vp in edges {
            for vn in edges {
                states.push((eq, vp, vn & !vp));
            }
        }
    }
    for (eq, vp, vn) in states {
        assert_eq!(
            step(eq, vp, vn),
            blocks::reference(eq, vp, vn),
            "{name}: eq={eq:#x} vp={vp:#x} vn={vn:#x}"
        );
    }
}

fn main() {
    let add = chain8!("add {x}, {x}, {y}");
    if std::env::args().any(|a| a == "--ops") {
        let ops = [
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
            println!("  {name:34} {:.2}", ns / add);
        }
    }

    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut tab = [0u64; 256];
    for v in tab.iter_mut() {
        *v = xorshift(&mut s);
    }
    let mut text = [0u8; 4096];
    for c in text.iter_mut() {
        *c = b'a' + (xorshift(&mut s) % 26) as u8;
    }

    println!(
        "cycles per column, best (worst) of {RUNS} runs, {:.2} GHz:",
        1.0 / add
    );
    println!("  {:22} {:>15} {:>15}", "block", "const eq", "table eq");
    macro_rules! score {
        ($name:expr, $f:path) => {{
            check($name, $f);
            let eq = black_box(0x8421_8421_8421_8421u64);
            let (cb, cw) = time(|| run_const($f, eq, black_box(ITERS)));
            let (tb, tw) = time(|| run_table($f, &tab, &text, black_box(ITERS)));
            println!(
                "  {:22} {:>7.2} ({:.2}) {:>7.2} ({:.2})",
                $name,
                cb / add,
                cw / add,
                tb / add,
                tw / add
            );
        }};
    }
    blocks::for_each_block!(score);
}
