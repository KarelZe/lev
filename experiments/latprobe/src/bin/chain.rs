//! Is the 6-op critical path of chain6 slower when 9 independent ops share
//! the loop? (a) the chain alone, (b) chain plus 9 ops off the chain that
//! read only loop-invariant values, (c) chain plus 9 ops that read the
//! chain's intermediate values (like the real block).
use std::arch::asm;
use std::hint::black_box;
use std::time::Instant;

const ITERS: u64 = 50_000_000;

fn run(f: &dyn Fn(u64) -> u64) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        black_box(f(black_box(7)));
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / ITERS as f64);
    }
    best
}

fn main() {
    let add = run(&|mut x| {
        for _ in 0..ITERS {
            unsafe {
                asm!("add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
                          "add {x}, {x}, #1", "add {x}, {x}, #1", x = inout(reg) x, options(nomem, nostack))
            }
        }
        x
    }) / 6.0;
    let (eq, k) = (black_box(0x8421_8421u64), black_box(0x0ff0_0ff0u64));
    let a = run(&|mut x| {
        for _ in 0..ITERS {
            unsafe {
                asm!("and {t}, {eq}, {x}", "add {t}, {t}, {x}", "orr {t}, {k}, {t}", "lsl {t}, {t}, #1",
                          "bic {t}, {t}, {k}", "orr {x}, {eq}, {t}",
                          x = inout(reg) x, t = out(reg) _, eq = in(reg) eq, k = in(reg) k, options(nomem, nostack))
            }
        }
        x
    });
    let b = run(&|mut x| {
        for _ in 0..ITERS {
            unsafe {
                asm!("and {t}, {eq}, {x}", "orr {f1}, {eq}, {k}", "orr {f2}, {k}, {eq}", "lsl {f3}, {k}, #1",
                          "add {t}, {t}, {x}", "orr {f4}, {f1}, {f3}", "orr {t}, {k}, {t}", "orn {f5}, {eq}, {f2}",
                          "lsl {t}, {t}, #1", "and {f6}, {f5}, {k}", "lsl {f7}, {f6}, #1", "bic {f8}, {f4}, {f3}",
                          "bic {t}, {t}, {k}", "orr {x}, {eq}, {t}", "bic {f9}, {f8}, {f7}",
                          x = inout(reg) x, t = out(reg) _, eq = in(reg) eq, k = in(reg) k,
                          f1 = out(reg) _, f2 = out(reg) _, f3 = out(reg) _, f4 = out(reg) _, f5 = out(reg) _,
                          f6 = out(reg) _, f7 = out(reg) _, f8 = out(reg) _, f9 = out(reg) _, options(nomem, nostack))
            }
        }
        x
    });
    let c = run(&|mut x| {
        for _ in 0..ITERS {
            unsafe {
                asm!("and {t}, {eq}, {x}", "orr {f1}, {eq}, {x}", "orr {f2}, {k}, {x}", "lsl {f3}, {x}, #1",
                          "add {t}, {t}, {x}", "orr {f4}, {f2}, {f3}", "orr {u}, {f1}, {t}", "orn {f5}, {eq}, {t}",
                          "lsl {u}, {u}, #1", "and {f6}, {f5}, {x}", "lsl {f7}, {f6}, #1", "bic {f8}, {u}, {f3}",
                          "bic {u}, {u}, {f4}", "orr {x}, {f7}, {u}", "bic {f9}, {f2}, {f8}",
                          x = inout(reg) x, t = out(reg) _, u = out(reg) _, eq = in(reg) eq, k = in(reg) k,
                          f1 = out(reg) _, f2 = out(reg) _, f3 = out(reg) _, f4 = out(reg) _, f5 = out(reg) _,
                          f6 = out(reg) _, f7 = out(reg) _, f8 = out(reg) _, f9 = out(reg) _, options(nomem, nostack))
            }
        }
        x
    });
    println!("cycles per iteration (add = 1, {:.2} GHz):", 1.0 / add);
    println!(
        "  (a) 6-op chain alone                       {:.2}",
        a / add
    );
    println!(
        "  (b) chain + 9 invariant ops                {:.2}",
        b / add
    );
    println!(
        "  (c) chain + 9 ops reading chain values     {:.2}",
        c / add
    );
}
