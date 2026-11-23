# ruff: file-ignore[lowercase-imported-as-non-lowercase, ambiguous-unicode-character-string]
"""Random diff test of a lev build vs rapidfuzz on the classes the re-gates touch."""

import random
import sys

sys.path.insert(0, sys.argv[1])
from rapidfuzz.distance import Indel, Levenshtein

import lev

sys.path.insert(0, sys.argv[2])
import cases as C

rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
alphas = [
    C.ASCII,
    C.CJK,
    "".join(chr(0x4E00 + rng.randrange(20000)) for _ in range(500)),
    C.CYRILLIC,
    C.EMOJI + "ab",
    "abc",
]
bad = 0
n = 0
for _ in range(30000):
    al = rng.choice(alphas)
    m = rng.choice([rng.randint(1, 40), rng.randint(17, 70), rng.randint(60, 330), rng.randint(500, 1300)])
    s = C.rand_str(al, m, rng)
    k = rng.random()
    if k < 0.3:
        t = C.subs_spread(s, rng.randint(1, 4), al, rng) if m > 1 else s
        if rng.random() < 0.5:
            t += C.rand_str(al, rng.randint(0, 3), rng)
    elif k < 0.6:
        t = C.mutate(s, al, max(1, int(m * rng.random() * 0.5)), rng)
    elif k < 0.8:
        t = list(s)
        for _ in range(rng.randint(1, 80)):
            t.insert(rng.randrange(len(t) + 1), rng.choice(al))
        t = "".join(t)
    else:
        t = C.rand_str(al, max(1, m + rng.randint(-40, 80)), rng)
    if rng.random() < 0.2:
        t = t.replace(t[0], "’", 1) if t else t
    n += 1
    if lev.distance(s, t) != Levenshtein.distance(s, t):
        bad += 1
        print("dist", repr(s[:30]), repr(t[:30]))
    r = lev.ratio(s, t)
    e = Indel.normalized_similarity(s, t)
    if abs(r - e) > 1e-12:
        bad += 1
        print("ratio", repr(s[:30]), r, e)
print(sys.argv[1].split("/")[-1], "pairs", n, "mismatches", bad)
