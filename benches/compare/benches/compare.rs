//! Compares `lev::distance` against other Levenshtein crates.
//!
//! Two criterion groups:
//!
//! * `kind/<kind>` – the five string pairs of `scripts/benchmark.py`: four
//!   100-character pairs with ASCII, Latin-1, CJK and emoji characters, plus
//!   a pair of typo'd English sentences.
//! * `length/<crate>/<n>` – pairs of `n` lowercase ASCII letters where every
//!   fifth one is substituted, like `scripts/benchmark_by_length.py`.
//! * `mixed/<crate>` – 1,000 different short ASCII pairs per iteration, so
//!   the branch predictor cannot learn one pair's path.
//!
//! All crates count `char`s.  A crate that only handles bytes runs only on
//! pairs where it returns the same distance as `lev`; the others are skipped
//! with a note on stderr.
//!
//! Run from `benches/compare`:
//!
//! ```text
//! cargo bench                       # everything
//! cargo bench -- kind/              # one group
//! cargo bench -- --save-baseline x  # criterion baselines work as usual
//! cargo bench -- --measurement-time 10 --warm-up-time 3  # longer runs
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};

/// Receives every crate under test from [`for_each_crate`].
///
/// Each crate arrives as its own closure or function item type, so the benchmark loop is
/// monomorphized per crate and the call can be inlined, as in a user's code.
/// An `fn` pointer table would add an indirect call to every measurement.
trait Visit {
    fn visit(&mut self, name: &str, f: impl Fn(&str, &str) -> usize + Copy);
}

fn for_each_crate(v: &mut impl Visit) {
    v.visit("lev", lev::distance);
    v.visit("rapidfuzz", |a, b| {
        rapidfuzz::distance::levenshtein::distance(a.chars(), b.chars())
    });
    v.visit("strsim", strsim::levenshtein);
    // Bytes, not chars: only agrees with the others on ASCII.
    v.visit("triple_accel", |a, b| {
        triple_accel::levenshtein::levenshtein(a.as_bytes(), b.as_bytes()) as usize
    });
    // `stringmetrics::levenshtein` compares bytes; its iterator form takes chars.
    v.visit("stringmetrics", |a, b| {
        stringmetrics::levenshtein_limit_iter(a.chars(), b.chars(), u32::MAX) as usize
    });
    v.visit("edit-distance", |a, b| edit_distance::edit_distance(a, b));
    v.visit("levenshtein", levenshtein::levenshtein);
    v.visit("textdistance", textdistance::str::levenshtein);
}

/// `s.repeat(n)` cut to `len` chars.
fn cycle(s: &str, len: usize) -> String {
    s.chars().cycle().take(len).collect()
}

/// The five pairs of `scripts/benchmark.py`, under the same names.
fn kinds() -> Vec<(&'static str, String, String)> {
    const LEN: usize = 100;
    vec![
        (
            "ascii",
            cycle("The quick brown fox jumps over the lazy dog. ", LEN),
            cycle("The quick brown cat jumps over the lazy dog! ", LEN),
        ),
        (
            "latin1",
            cycle("café résumé naïve façade jalapeño Zürich smörgåsbord ", LEN),
            cycle(
                "cafe  resume  naive facade jalapeno Zurich  smorgasbord ",
                LEN,
            ),
        ),
        (
            "cjk",
            cycle("日本語のテスト文字列を長くするために繰り返します。", LEN),
            cycle("日本語のテスト文字列を短くするために繰り返します。", LEN),
        ),
        (
            "emoji",
            cycle("😀🎉🚀✨🐍🦀📦🔥💡🌟", LEN),
            cycle("😀🎉🚀✨🐍🦀📦🔥💡⭐", LEN),
        ),
        (
            "realistic",
            "Customer support was fantastic today. The technician arrived on time, \
             diagnosed the issue within minutes, and had everything working again \
             before lunch. I would definitely recommend this service to a friend \
             or colleague."
                .to_owned(),
            "Customer suport was fantastic today. The technician arived on time, \
             diagnosed the issue within mintues, and had everything working again \
             before lunch. I would definately recommend this service to a freind \
             or colleague."
                .to_owned(),
        ),
    ]
}

/// `n` lowercase letters, and a copy with every fifth one upper-cased (20% edits).
fn length_pair(n: usize) -> (String, String) {
    let a = cycle("abcdefghijklmnopqrstuvwxyz", n);
    let b = a
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if i % 5 == 0 {
                c.to_ascii_uppercase()
            } else {
                c
            }
        })
        .collect();
    (a, b)
}

/// A small xorshift generator, so the inputs are the same on every run.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// 1,000 pairs of 4 to 64 lowercase letters and spaces, each with about 15%
/// random substitutions, insertions and deletions.
fn mixed_pairs() -> Vec<(String, String)> {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz ";
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let letter = |rng: &mut Rng| ALPHABET[rng.below(ALPHABET.len())] as char;
    (0..1000)
        .map(|_| {
            let len = 4 + rng.below(61);
            let a: String = (0..len).map(|_| letter(&mut rng)).collect();
            let mut b = String::with_capacity(len + len / 4);
            for c in a.chars() {
                match rng.below(20) {
                    0 => b.push(letter(&mut rng)),        // substitution
                    1 => {}                               // deletion
                    2 => b.extend([c, letter(&mut rng)]), // insertion
                    _ => b.push(c),
                }
            }
            (a, b)
        })
        .collect()
}

/// Whether `f` agrees with `lev` on every pair; prints a note if it does not.
fn agrees(
    name: &str,
    label: &str,
    f: impl Fn(&str, &str) -> usize,
    pairs: &[(&str, &str)],
) -> bool {
    let bad = pairs.iter().find(|(a, b)| f(a, b) != lev::distance(a, b));
    if let Some((a, b)) = bad {
        let (got, want) = (f(a, b), lev::distance(a, b));
        eprintln!("skipping {name} on {label}: distance {got}, lev says {want}");
    }
    bad.is_none()
}

/// One pair, timed as a single call per iteration.
struct Single<'a, 'c> {
    group: &'a mut BenchmarkGroup<'c, WallTime>,
    id: &'a dyn Fn(&str) -> BenchmarkId,
    label: &'a str,
    a: &'a str,
    b: &'a str,
}

impl Visit for Single<'_, '_> {
    fn visit(&mut self, name: &str, f: impl Fn(&str, &str) -> usize + Copy) {
        let (a, b) = (self.a, self.b);
        if agrees(name, self.label, f, &[(a, b)]) {
            self.group.bench_function((self.id)(name), |bench| {
                bench.iter(|| f(black_box(a), black_box(b)))
            });
        }
    }
}

fn bench_kinds(c: &mut Criterion) {
    for (kind, a, b) in kinds() {
        let mut group = c.benchmark_group(format!("kind/{kind}"));
        for_each_crate(&mut Single {
            group: &mut group,
            id: &|name| BenchmarkId::from_parameter(name),
            label: kind,
            a: &a,
            b: &b,
        });
        group.finish();
    }
}

fn bench_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("length");
    for n in [8, 16, 32, 64, 128, 256, 512, 1024, 4096] {
        let (a, b) = length_pair(n);
        // Characters of the longer string per second, for criterion's report.
        group.throughput(Throughput::Elements(n as u64));
        // The quadratic crates take milliseconds per call at 4096 chars.
        group.sample_size(if n > 1024 { 20 } else { 100 });
        for_each_crate(&mut Single {
            group: &mut group,
            id: &|name| BenchmarkId::new(name, n),
            label: &n.to_string(),
            a: &a,
            b: &b,
        });
    }
    group.finish();
}

/// All pairs per iteration, so no single pair's branches are learned.
struct Mixed<'a, 'c> {
    group: &'a mut BenchmarkGroup<'c, WallTime>,
    pairs: &'a [(&'a str, &'a str)],
}

impl Visit for Mixed<'_, '_> {
    fn visit(&mut self, name: &str, f: impl Fn(&str, &str) -> usize + Copy) {
        let pairs = self.pairs;
        if agrees(name, "mixed", f, pairs) {
            self.group.bench_function(name, |bench| {
                bench.iter(|| {
                    pairs
                        .iter()
                        .map(|&(a, b)| f(black_box(a), black_box(b)))
                        .sum::<usize>()
                })
            });
        }
    }
}

fn bench_mixed(c: &mut Criterion) {
    let owned = mixed_pairs();
    let pairs: Vec<(&str, &str)> = owned
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let mut group = c.benchmark_group("mixed");
    group.throughput(Throughput::Elements(pairs.len() as u64));
    for_each_crate(&mut Mixed {
        group: &mut group,
        pairs: &pairs,
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_kinds, bench_lengths, bench_mixed
}
criterion_main!(benches);
