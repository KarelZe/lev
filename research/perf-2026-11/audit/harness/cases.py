# ruff: file-ignore[undocumented-public-function, docstring-missing-returns, ambiguous-unicode-character-string, printf-string-formatting]
"""
Benchmark cases for the lev layer audit: realistic batches, adversarial pairs, test_bench cases.

`build()` returns a list of cases; each case is a dict with
  name: str, kind: "pair" | "batch", fns: tuple of "distance"/"ratio",
  pair: (s1, s2) or batch: (list1, list2), tags: set of layer tags it targets.
Everything is deterministic (seeded RNG, files from /usr/share/dict).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

DICT = Path("/usr/share/dict")
ASCII = "abcdefghijklmnopqrstuvwxyz "
LOWER = "abcdefghijklmnopqrstuvwxyz"
CJK = "".join(chr(0x4E00 + i) for i in range(0, 600, 7))  # 86 common-block ideographs
CJK_SMALL = "ぁあぃいぅうぇえぉお"
EMOJI = "".join(chr(0x1F600 + i) for i in range(10))


def _lines(name: str) -> list[str]:
    return (DICT / name).read_text().split()


WORDS = [w for w in _lines("web2") if w.islower() and 3 <= len(w) <= 14]
NAMES = _lines("propernames")
CONNECT = _lines("connectives")


def rand_str(alpha: str, n: int, rng: random.Random) -> str:
    return "".join(rng.choice(alpha) for _ in range(n))


def mutate(s: str, alpha: str, edits: int, rng: random.Random) -> str:
    b = list(s)
    for _ in range(edits):
        i = rng.randrange(len(b))
        op = rng.randrange(3)
        if op == 0:
            b[i] = rng.choice(alpha)
        elif op == 1:
            b.insert(i, rng.choice(alpha))
        elif len(b) > 1:
            del b[i]
    return "".join(b)


def typo(s: str, k: int, rng: random.Random) -> str:
    """K keyboard-style typos: substitution, insertion, deletion or transposition."""
    b = list(s)
    for _ in range(k):
        if len(b) < 2:
            break
        i = rng.randrange(len(b) - 1)
        op = rng.randrange(4)
        if op == 0:
            b[i] = rng.choice(LOWER)
        elif op == 1:
            b.insert(i, rng.choice(LOWER))
        elif op == 2:
            del b[i]
        else:
            b[i], b[i + 1] = b[i + 1], b[i]
    return "".join(b)


def subs_spread(s: str, k: int, alpha: str, rng: random.Random) -> str:
    """K substitutions spread evenly, the first at 0 and the last at len-1 (survive affix strip)."""
    b = list(s)
    m = len(b)
    pos = [round(i * (m - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    for p in pos:
        c = rng.choice(alpha)
        while c == b[p]:
            c = rng.choice(alpha)
        b[p] = c
    return "".join(b)


def sentence(rng: random.Random, lo: int, hi: int) -> str:
    target = rng.randint(lo, hi)
    out: list[str] = []
    while sum(len(w) + 1 for w in out) < target:
        out.append(rng.choice(CONNECT) if rng.random() < 0.45 else rng.choice(WORDS))
    s = " ".join(out)
    return s[0].upper() + s[1:target] + "."


def related_sentence(s: str, frac: float, rng: random.Random) -> str:
    """Word-level edit: replace/insert/delete about frac of the words."""
    ws = s.rstrip(".").split(" ")
    k = max(1, round(frac * len(ws)))
    for _ in range(k):
        i = rng.randrange(len(ws))
        op = rng.randrange(3)
        new = rng.choice(CONNECT) if rng.random() < 0.45 else rng.choice(WORDS)
        if op == 0:
            ws[i] = new
        elif op == 1:
            ws.insert(i, new)
        elif len(ws) > 1:
            del ws[i]
    return " ".join(ws) + "."


BRANDS = [
    "Apple",
    "Samsung",
    "Sony",
    "LG",
    "Dell",
    "HP",
    "Lenovo",
    "Asus",
    "Acer",
    "Microsoft",
    "Logitech",
    "Bose",
    "Canon",
    "Nikon",
    "Philips",
    "Panasonic",
    "Xiaomi",
    "Huawei",
    "Google",
    "Amazon",
]
TYPES = [
    "Laptop",
    "Smartphone",
    "Headphones",
    "Monitor",
    "Keyboard",
    "Mouse",
    "Camera",
    "Tablet",
    "Smartwatch",
    "Speaker",
    "TV",
    "Router",
    "Charger",
    "Printer",
]
ATTRS = [
    "Wireless", "Bluetooth", "Noise Cancelling", "4K", "UHD", "OLED", "Gaming", "Ultra", "Pro", "Max", "Mini",
    "Plus", "Lite", "Black", "White", "Silver", "Blue", "Red", "Titanium", "128GB", "256GB", "512GB", "1TB",
    "8GB RAM", "16GB RAM", "15.6 inch", "27 inch", "USB-C", "Fast Charging", "2024 Model", "Refurbished",
    "with Case", "Dual SIM",
]  # fmt: skip


def title(rng: random.Random) -> tuple[str, list[str]]:
    head = f"{rng.choice(BRANDS)} {rng.choice(LOWER).upper()}{rng.randint(1, 99)} {rng.choice(TYPES)}"
    return head, rng.sample(ATTRS, rng.randint(2, 5))


def related_title(head: str, attrs: list[str], rng: random.Random) -> str:
    a = list(attrs)
    for _ in range(rng.randint(1, 3)):
        op = rng.randrange(3)
        if op == 0 and a:
            a[rng.randrange(len(a))] = rng.choice(ATTRS)
        elif op == 1:
            a.insert(rng.randrange(len(a) + 1), rng.choice(ATTRS))
        elif len(a) > 1:
            del a[rng.randrange(len(a))]
    return f"{head} {' '.join(a)}"


SUFFIX = [("Street", "St."), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Lane", "Ln"), ("Drive", "Dr")]
STATES = ["AL", "AK", "AZ", "CA", "CO", "CT", "FL", "GA", "IL", "IN", "MA", "MI", "NY", "NJ", "OH", "PA", "TX", "WA"]


def address(rng: random.Random) -> tuple[str, str]:
    num = rng.randint(1, 9999)
    street = rng.choice(NAMES)
    full, abbr = rng.choice(SUFFIX)
    city = rng.choice(NAMES) + rng.choice(["ville", "ton", " City", "field", "burg"])
    st, z = rng.choice(STATES), rng.randint(10000, 99999)
    a = f"{num} {street} {full}, {city}, {st} {z}"
    # Variant: abbreviation, dropped comma, maybe unit, maybe zip typo.
    zb = str(z) if rng.random() < 0.7 else str(z)[:-1] + str((z % 10 + 1) % 10)
    unit = f" Apt {rng.randint(1, 20)}{rng.choice('ABCD')}" if rng.random() < 0.3 else ""
    b = f"{num} {street} {abbr}{unit} {city} {st} {zb}"
    return a, b


def latin1_name(rng: random.Random) -> tuple[str, str]:
    acc = {"e": "é", "u": "ü", "o": "ö", "a": "á", "n": "ñ", "c": "ç"}
    s = f"{rng.choice(NAMES)} {rng.choice(NAMES)}"
    t = "".join(acc[c] if c in acc and rng.random() < 0.35 else c for c in s)
    if t == s:
        t = s.replace("e", "é", 1) if "e" in s else s + "é"
    return t, s


def build() -> list[dict]:
    rng = random.Random(20261109)
    cases: list[dict] = []
    both = ("distance", "ratio")

    def batch(name: str, pairs: list[tuple[str, str]], tags: set[str], fns=both) -> None:
        cases.append({
            "name": name,
            "kind": "batch",
            "fns": fns,
            "batch": ([p[0] for p in pairs], [p[1] for p in pairs]),
            "tags": tags,
        })

    def pair(name: str, s1: str, s2: str, tags: set[str], fns=both) -> None:
        cases.append({"name": name, "kind": "pair", "fns": fns, "pair": (s1, s2), "tags": tags})

    # ---------------- Realistic batches (varied pairs, branch predictors see a distribution) ----------
    batch("R:words-dis", [(rng.choice(WORDS), rng.choice(WORDS)) for _ in range(3000)], {"real", "tiny"})
    batch(
        "R:words-typo",
        [(w, typo(w, rng.randint(1, 2), rng)) for w in (rng.choice(WORDS) for _ in range(3000))],
        {"real", "tiny"},
    )
    names = [f"{rng.choice(NAMES)} {rng.choice(NAMES)}" for _ in range(3000)]
    batch("R:names-dis", [(a, f"{rng.choice(NAMES)} {rng.choice(NAMES)}") for a in names], {"real", "tiny", "tf", "mb"})
    batch("R:names-typo", [(a, typo(a, rng.randint(1, 3), rng)) for a in names], {"real", "tiny", "mb"})
    batch("R:names-latin1", [latin1_name(rng) for _ in range(3000)], {"real", "tiny"})
    tl = [title(rng) for _ in range(2000)]
    batch(
        "R:titles-dis",
        [(f"{h} {' '.join(a)}", "%s %s" % (title(rng)[0], " ".join(title(rng)[1]))) for h, a in tl],
        {"real", "tf", "band"},
    )
    batch(
        "R:titles-rel", [(f"{h} {' '.join(a)}", related_title(h, a, rng)) for h, a in tl], {"real", "tf", "band", "mb"}
    )
    batch("R:addr-rel", [address(rng) for _ in range(2000)], {"real", "tf", "tiny"})
    sents = [sentence(rng, 60, 200) for _ in range(1500)]
    batch("R:sent-dis", [(s, sentence(rng, 60, 200)) for s in sents], {"real", "band", "mb"})
    batch("R:sent-rel10", [(s, related_sentence(s, 0.10, rng)) for s in sents], {"real", "band"})
    batch("R:sent-rel25", [(s, related_sentence(s, 0.25, rng)) for s in sents], {"real", "band"})
    batch("R:sent-rel40", [(s, related_sentence(s, 0.40, rng)) for s in sents], {"real", "band"})
    batch("R:sent-typo3", [(s, typo(s, 3, rng)) for s in sents], {"real", "mb", "band"})
    batch("R:sent-subs3", [(s, subs_spread(s, 3, LOWER, rng)) for s in sents], {"real", "mb"})
    cjk = [rand_str(CJK, rng.randint(10, 40), rng) for _ in range(2000)]
    batch("R:cjk-dis", [(s, rand_str(CJK, rng.randint(10, 40), rng)) for s in cjk], {"real", "tinywide", "hash"})
    batch("R:cjk-rel", [(s, mutate(s, CJK, max(1, len(s) // 5), rng)) for s in cjk], {"real", "tinywide", "hash", "mb"})
    cjkn = [rand_str(CJK, rng.randint(2, 4), rng) for _ in range(3000)]
    batch("R:cjk-names", [(s, rand_str(CJK, rng.randint(2, 4), rng)) for s in cjkn], {"real", "tinywide"})
    batch(
        "R:mixed-quote",
        [
            (s.replace(" ", "’", 1), s.replace(" ", "'", 1)[: len(s) - rng.randint(0, 3)])
            for s in (sentence(rng, 15, 60) for _ in range(2000))
        ],
        {"real", "mixed", "tinywide"},
    )
    batch(
        "R:long-rel",
        [(s, related_sentence(s, 0.15, rng)) for s in (sentence(rng, 600, 1200) for _ in range(60))],
        {"real", "large"},
    )

    # ---------------- Tiny path for wide units (TINY_M = 16 vs 8) ----------------
    for m in (9, 12, 16):
        pair(f"A:cjk-{m}-dis", rand_str(CJK, m, rng), rand_str(CJK, m, rng), {"tinywide"})
    pair("A:cjk-12-vs-48", rand_str(CJK, 12, rng), rand_str(CJK, 48, rng), {"tinywide"})
    s = rand_str(CJK, 14, rng)
    pair("A:cjk-14-s3", s, subs_spread(s, 3, CJK, rng), {"tinywide", "mb"})
    s = rand_str(CJK_SMALL, 12, rng)
    pair("A:kana10-12-dis", s, rand_str(CJK_SMALL, 13, rng), {"tinywide"})
    pair("A:emoji-12-dis", rand_str(EMOJI, 12, rng), rand_str(EMOJI, 12, rng), {"tinywide"})
    pair("A:mixed-u16pat-12", "Don’t stop b", "Dont stop believin", {"tinywide", "mixed"})

    # ---------------- mbleven + small_ub gate ----------------
    for m in (24, 40, 60, 200, 500):
        s = rand_str(ASCII, m, rng)
        pair(f"A:mb-ascii-{m}-s2", s, subs_spread(s, 2, LOWER, rng), {"mb"}, ("distance",))
        pair(f"A:mb-ascii-{m}-s3", s, subs_spread(s, 3, LOWER, rng), {"mb"}, ("distance",))
        pair(f"A:mbfail-ascii-{m}-s4", s, subs_spread(s, 4, LOWER, rng), {"mb"}, ("distance",))
        t = rng.choice(LOWER) + s[1:] + "!"
        pair(f"A:mb-ascii-{m}-s1i1", s, t, {"mb"}, ("distance",))
    for m in (24, 60):
        pair(f"A:mbgate-ascii-{m}-dis", rand_str(ASCII, m, rng), rand_str(ASCII, m, rng), {"mb"}, ("distance",))
    for m in (24, 200):
        s = rand_str(CJK, m, rng)
        pair(f"A:mb-cjk-{m}-s3", s, subs_spread(s, 3, CJK, rng), {"mb"}, ("distance",))

    # ---------------- Speculative small bands (m 65-704) ----------------
    for m in (100, 200, 300, 500):
        s = rand_str(ASCII, m, rng)
        for e in (10, 20, 25, 30, 40):
            pair(f"A:band-ascii-{m}-e{e}", s, mutate(s, ASCII, m * e // 100, rng), {"band"}, ("distance",))
        pair(f"A:band-ascii-{m}-dis", s, rand_str(ASCII, m, rng), {"band"}, ("distance",))
    for m, e in ((120, 15), (120, 30), (250, 15), (250, 30)):
        s = sentence(rng, m, m)
        pair(f"A:band-sent-{m}-w{e}", s, related_sentence(s, e / 100, rng), {"band"}, ("distance",))
    s = rand_str(CJK, 150, rng)
    pair("A:band-cjk-150-e25", s, mutate(s, CJK, 37, rng), {"band", "hash"}, ("distance",))

    # ---------------- Table-free masks (u8, m 17-64, n <= 8m) ----------------
    for m in (17, 24, 40, 64):
        for x in (1, 4, 8):
            pair(f"A:tf-ascii-{m}-x{x}", rand_str(ASCII, m, rng), rand_str(ASCII, m * x, rng), {"tf"})
    pair("A:tf-ascii-17-x9", rand_str(ASCII, 17, rng), rand_str(ASCII, 17 * 9, rng), {"tf"})
    pair(
        "A:tf-latin1-30-x1",
        "é" + rand_str("àáâãäåæçèé abcdef", 29, rng),
        rand_str("àáâãäåæçèé abcdef", 30, rng),
        {"tf"},
    )

    # ---------------- Single-word hash sizing (UCS-2 m 17-64) ----------------
    for m in (20, 32, 40):
        for x in (1, 4):
            pair(f"A:hash-cjk-{m}-x{x}", rand_str(CJK, m, rng), rand_str(CJK, m * x, rng), {"hash"})

    # ---------------- u8 tiny path (SWAR vs NEON) and R1 shapes ----------------
    pair("A:kitten", "kitten", "sitting", {"swar", "r1"})
    for m in (3, 6, 8):
        for x in (1, 4):
            pair(f"A:ascii-{m}-x{x}", rand_str(LOWER, m, rng), rand_str(LOWER, m * x, rng), {"swar"})
    pair("A:ascii-12-x1", rand_str(LOWER, 12, rng), rand_str(LOWER, 12, rng), {"swar"})

    # ---------------- LargeCtx bands (m > 704) ----------------
    for m in (800, 1200, 2048):
        s = rand_str(ASCII, m, rng)
        for e in (2, 10, 18, 25, 40):
            pair(f"A:large-ascii-{m}-e{e}", s, mutate(s, ASCII, m * e // 100, rng), {"large"})
        pair(f"A:large-ascii-{m}-dis", s, rand_str(ASCII, m, rng), {"large"})

    # ---------------- test_bench.py cases ----------------
    tb = Path(__file__).resolve().parent.parent / "base" / "tests"
    sys.path.insert(0, str(tb))
    sys.path.insert(1, str(tb.parent.parent / "inst" / "base"))
    import test_bench

    for c in test_bench._CASES:
        s1, s2, _n = c.values
        name = c.id.split("]")[0]
        pair(f"T:{name}", s1, s2, {"tb"})
    return cases


if __name__ == "__main__":
    cs = build()
    print(len(cs), "cases;", sum(len(c["fns"]) for c in cs), "case-fn combos")
    import collections

    print(collections.Counter(t for c in cs for t in c["tags"]))


CYRILLIC = "".join(chr(c) for c in range(0x430, 0x450)) + " "


def build2() -> list[dict]:
    """Second-round cases: re-gate checks for mbleven, small bands, LargeCtx passes, hash sizing."""
    rng = random.Random(777)
    cases: list[dict] = []
    both = ("distance", "ratio")

    def batch(name, pairs, tags, fns=both):
        cases.append({
            "name": name,
            "kind": "batch",
            "fns": fns,
            "batch": ([p[0] for p in pairs], [p[1] for p in pairs]),
            "tags": tags,
        })

    def pair(name, s1, s2, tags, fns=both):
        cases.append({"name": name, "kind": "pair", "fns": fns, "pair": (s1, s2), "tags": tags})

    d = ("distance",)
    # ---- mbleven classes (ub = 3 with diff 0/1/2) around the one-word boundary ----
    for alpha, tag in ((ASCII, "ascii"), (CJK, "cjk")):
        for m in (24, 48, 64, 100):
            s = rand_str(alpha, m, rng)
            pair(f"A:mb-{tag}-{m}-ub3d0", s, subs_spread(s, 3, alpha, rng), {"mb"}, d)
            t = subs_spread(s, 2, alpha, rng) + rng.choice(alpha)  # 2 aligned mismatches + 1 extra char
            pair(f"A:mb-{tag}-{m}-ub3d1", s, t, {"mb"}, d)
            t = rng.choice([c for c in alpha if c != s[0]]) + s[1:] + rand_str(alpha, 2, rng)
            pair(f"A:mb-{tag}-{m}-ub3d2", s, t, {"mb"}, d)
            pair(f"A:mb-{tag}-{m}-ub2d0", s, subs_spread(s, 2, alpha, rng), {"mb"}, d)
    names = [f"{rng.choice(NAMES)} {rng.choice(NAMES)} {rng.choice(NAMES)}" for _ in range(3000)]
    batch("R:names3-subs3", [(a, subs_spread(a, 3, LOWER, rng)) for a in names], {"real", "mb"}, d)
    batch("R:names-latin1", [latin1_name(rng) for _ in range(3000)], {"real", "mb"}, d)
    tl = [title(rng) for _ in range(2000)]
    batch(
        "R:titles-subs3",
        [(t, subs_spread(t, 3, LOWER, rng)) for t in (f"{h} {' '.join(a)}" for h, a in tl)],
        {"real", "mb"},
        d,
    )
    batch(
        "R:cjk-subs3",
        [(s, subs_spread(s, 3, CJK, rng)) for s in (rand_str(CJK, rng.randint(17, 64), rng) for _ in range(2000))],
        {"real", "mb"},
        d,
    )
    sents = [sentence(rng, 60, 200) for _ in range(1500)]
    batch("R:sent-subs3", [(s, subs_spread(s, 3, LOWER, rng)) for s in sents], {"real", "mb"}, d)

    # ---- small bands: dissimilar pairs with a large length difference, and wins with large s ----
    for m, n in ((100, 120), (100, 140), (100, 160), (300, 340), (300, 400)):
        pair(f"A:band-ascii-{m}-vs-{n}-dis", rand_str(ASCII, m, rng), rand_str(ASCII, n, rng), {"band"}, d)
        a = sentence(rng, m, m)
        pair(f"A:band-sent-{m}-vs-{n}-dis", a, sentence(rng, n, n), {"band"}, d)
    for m, k in ((100, 40), (100, 50), (300, 50), (300, 90)):
        s = rand_str(ASCII, m, rng)
        t = list(s)
        for _ in range(k):
            t.insert(rng.randrange(1, len(t)), rng.choice(LOWER))
        t = "".join(t)
        pair(f"A:band-ascii-{m}-ins{k}", s, t, {"band"}, d)
        pair(f"A:band-ascii-{m}-ins{k}-s5", s, subs_spread(t, 5, LOWER, rng), {"band"}, d)
    batch("R:sent-dis", [(s, sentence(rng, 60, 200)) for s in sents], {"real", "band"}, d)
    eq = []
    for s in sents:
        t = sentence(rng, max(60, len(s) - 5), len(s) + 5)
        eq.append((s, t))
    batch("R:sent-dis-eqlen", eq, {"real", "band"}, d)
    dl = []
    for s in sents:
        t = sentence(rng, len(s) + 30, len(s) + 60)
        dl.append((s, t))
    batch("R:sent-dis-difflen", dl, {"real", "band"}, d)
    batch("R:sent-rel25", [(s, related_sentence(s, 0.25, rng)) for s in sents], {"real", "band"}, d)
    batch("R:sent-rel10", [(s, related_sentence(s, 0.10, rng)) for s in sents], {"real", "band"}, d)
    batch("R:sent-ins", [(s, s.replace(" ", "  ", rng.randint(5, 25))) for s in sents], {"real", "band"}, d)

    # ---- LargeCtx passes ----
    for m in (800, 1000, 1200, 2048):
        s = rand_str(ASCII, m, rng)
        for e in (5, 10, 15, 20):
            pair(f"A:large-ascii-{m}-e{e}", s, mutate(s, ASCII, m * e // 100, rng), {"large"})
        pair(f"A:large-ascii-{m}-dis", s, rand_str(ASCII, m, rng), {"large"})
    s = rand_str(CJK, 1000, rng)
    pair("A:large-cjk-1000-e10", s, mutate(s, CJK, 100, rng), {"large"})
    batch(
        "R:long-rel15",
        [(s, related_sentence(s, 0.15, rng)) for s in (sentence(rng, 600, 1200) for _ in range(60))],
        {"real", "large"},
    )
    batch(
        "R:long-rel5",
        [(s, related_sentence(s, 0.05, rng)) for s in (sentence(rng, 600, 1200) for _ in range(60))],
        {"real", "large"},
    )
    batch(
        "R:long-rel30",
        [(s, related_sentence(s, 0.30, rng)) for s in (sentence(rng, 600, 1200) for _ in range(60))],
        {"real", "large"},
    )

    # ---- single-word hash sizing (UCS-2/4/mixed, m 17-64) ----
    cjk = [rand_str(CJK, rng.randint(10, 40), rng) for _ in range(2000)]
    batch("R:cjk-dis-b", [(s, rand_str(CJK, rng.randint(10, 40), rng)) for s in cjk], {"real", "hash"})
    c2 = [rand_str(CJK, rng.randint(17, 32), rng) for _ in range(2000)]
    batch("R:cjk17-32-dis", [(s, rand_str(CJK, rng.randint(17, 32), rng)) for s in c2], {"real", "hash"})
    batch("R:cjk17-32-rel", [(s, mutate(s, CJK, max(4, len(s) // 4), rng)) for s in c2], {"real", "hash"})
    c3 = [rand_str(CJK, rng.randint(33, 64), rng) for _ in range(2000)]
    batch("R:cjk33-64-dis", [(s, rand_str(CJK, rng.randint(33, 64), rng)) for s in c3], {"real", "hash"})
    cy = [rand_str(CYRILLIC, rng.randint(17, 32), rng) for _ in range(2000)]
    batch("R:cyr17-32-dis", [(s, rand_str(CYRILLIC, rng.randint(17, 32), rng)) for s in cy], {"real", "hash"})
    em = [rand_str(EMOJI + "ab", rng.randint(17, 32), rng) for _ in range(2000)]
    batch("R:emoji17-32-dis", [(s, rand_str(EMOJI + "ab", rng.randint(17, 32), rng)) for s in em], {"real", "hash"})
    mq = [sentence(rng, 20, 45) for _ in range(2000)]
    batch(
        "R:mixed-quote17-45", [(s.replace(" ", "’", 1), sentence(rng, 20, 45)) for s in mq], {"real", "hash", "mixed"}
    )
    zr = random.Random(4242)
    vocab = [chr(zr.randrange(0x4E00, 0xA000)) for _ in range(3000)]
    weights = [1 / (i + 1) for i in range(len(vocab))]

    def zipf_cjk(n):
        return "".join(rng.choices(vocab, weights, k=n))

    kana = [chr(c) for c in range(0x3041, 0x3097)]

    def ja(n):
        return "".join(rng.choice(kana) if rng.random() < 0.6 else rng.choices(vocab, weights)[0] for _ in range(n))

    z = [zipf_cjk(rng.randint(17, 32)) for _ in range(2000)]
    batch("R:cjkz17-32-dis", [(s, zipf_cjk(rng.randint(17, 32))) for s in z], {"real", "hash"})
    batch("R:cjkz17-32-rel", [(s, mutate(s, vocab[:300], max(4, len(s) // 4), rng)) for s in z], {"real", "hash"})
    z3 = [zipf_cjk(rng.randint(33, 64)) for _ in range(2000)]
    batch("R:cjkz33-64-dis", [(s, zipf_cjk(rng.randint(33, 64))) for s in z3], {"real", "hash"})
    j = [ja(rng.randint(17, 32)) for _ in range(2000)]
    batch("R:ja17-32-dis", [(s, ja(rng.randint(17, 32))) for s in j], {"real", "hash"})
    batch("R:ja17-32-rel", [(s, mutate(s, kana, max(4, len(s) // 4), rng)) for s in j], {"real", "hash"})
    batch(
        "R:cjkz10-40-dis",
        [(s, zipf_cjk(rng.randint(10, 40))) for s in (zipf_cjk(rng.randint(10, 40)) for _ in range(2000))],
        {"real", "hash", "tinywide"},
    )
    for m in (20, 32):
        pair(f"A:hash-cjk-{m}-x1", rand_str(CJK, m, rng), rand_str(CJK, m, rng), {"hash"})
        pair(f"A:hash-cyr-{m}-x1", rand_str(CYRILLIC, m, rng), rand_str(CYRILLIC, m, rng), {"hash"})
    return cases


def build3() -> list[dict]:
    """LargeCtx straight-to-full cases (w 9-11, or n >> m) for the bases-always check."""
    rng = random.Random(99)
    cases = []

    def pair(name, s1, s2):
        cases.append({"name": name, "kind": "pair", "fns": ("distance", "ratio"), "pair": (s1, s2), "tags": {"large"}})

    for m in (520, 600, 700):
        pair(f"A:full-ascii-{m}-dis", rand_str(ASCII, m, rng), rand_str(ASCII, m, rng))
        pair(f"A:full-cjk-{m}-dis", rand_str(CJK, m, rng), rand_str(CJK, m, rng))
        s = rand_str(ASCII, m, rng)
        pair(f"A:full-ascii-{m}-e30", s, mutate(s, ASCII, m * 30 // 100, rng))
    pair("A:full-ascii-800-vs-2400", rand_str(ASCII, 800, rng), rand_str(ASCII, 2400, rng))
    pair("A:full-cjk-800-vs-2400", rand_str(CJK, 800, rng), rand_str(CJK, 2400, rng))
    pair("A:full-mixed-600-dis", rand_str(ASCII, 600, rng), rand_str(ASCII, 599, rng) + "\u2019")
    for m in (800, 2048):
        pair(f"A:large-ascii-{m}-dis", rand_str(ASCII, m, rng), rand_str(ASCII, m, rng))
        pair(f"A:large-cjk-{m}-dis", rand_str(CJK, m, rng), rand_str(CJK, m, rng))
    return cases
