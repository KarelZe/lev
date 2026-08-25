"""
Deterministic, network-free training corpus for PGO.

Generates `(s1, s2)` string pairs with realistic typos, balanced evenly
across the four CPython string kinds and stratified over the length bands
`compute()` (src/lib.rs) branches on, so every dispatch arm gets real
profile signal.

The balance is deliberately *not* that of real fuzzy-matching traffic
(search-suggestion / autocomplete / contact-dedup / spell-check), which is
overwhelmingly short ASCII. An earlier version of this file followed that
traffic shape — 78% ASCII, 1.5% CJK, 1.0% emoji, median length 7 and p90 17
— and the resulting profile made UCS-2/UCS-4 strings of 48 characters and up
5-13% *slower* than an unprofiled build, because LLVM spent its inlining
budget on the UCS-1 arms and laid out the rest as cold. PGO consumes branch
and call counts rather than semantics, so a corpus should cover the code,
not mimic the traffic.

No external wordlists, dictionaries, or network fetches: everything is a
small hardcoded literal, matching the convention already used for the
`realistic` kind in scripts/benchmark.py. This keeps the corpus reproducible
across OSes and CI runners, and avoids the live-fetch flakiness Ruff's own
PGO corpus hit (see astral-sh/ruff#27570's "Retry PGO corpus Git operations").

Import `generate_pairs()` from this module; it is not meant to be run
directly.
"""

from __future__ import annotations

import random

# A fixed seed makes the corpus, and therefore the resulting .profdata,
# reproducible bit-for-bit across machines and CI runs.
SEED = 0x1E7_2026

# ---------------------------------------------------------------------------
# Word pools
# ---------------------------------------------------------------------------

# Common English words spanning the length range typical of search queries,
# contact names, and product titles -- the dominant real-world Levenshtein
# use case (fuzzy dedup / autocomplete / spell-check).
ASCII_WORDS = [
    "apple",
    "banana",
    "orange",
    "grape",
    "cherry",
    "lemon",
    "mango",
    "peach",
    "coffee",
    "tea",
    "water",
    "juice",
    "bread",
    "cheese",
    "butter",
    "sugar",
    "salt",
    "pepper",
    "garlic",
    "onion",
    "tomato",
    "potato",
    "carrot",
    "lettuce",
    "chicken",
    "beef",
    "salmon",
    "shrimp",
    "pasta",
    "rice",
    "noodle",
    "soup",
    "john",
    "jane",
    "michael",
    "sarah",
    "david",
    "laura",
    "robert",
    "emily",
    "james",
    "linda",
    "william",
    "susan",
    "richard",
    "karen",
    "joseph",
    "nancy",
    "smith",
    "johnson",
    "williams",
    "brown",
    "jones",
    "garcia",
    "miller",
    "davis",
    "street",
    "avenue",
    "boulevard",
    "road",
    "lane",
    "drive",
    "court",
    "place",
    "york",
    "london",
    "paris",
    "berlin",
    "tokyo",
    "sydney",
    "toronto",
    "chicago",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "computer",
    "keyboard",
    "monitor",
    "printer",
    "router",
    "server",
    "laptop",
    "python",
    "rust",
    "java",
    "golang",
    "kotlin",
    "swift",
    "ruby",
    "scala",
    "function",
    "variable",
    "database",
    "network",
    "storage",
    "memory",
    "cache",
    "account",
    "password",
    "username",
    "email",
    "profile",
    "settings",
    "session",
    "invoice",
    "receipt",
    "payment",
    "shipping",
    "delivery",
    "warehouse",
    "order",
    "customer",
    "support",
    "service",
    "feedback",
    "review",
    "rating",
    "comment",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "runs",
    "fast",
    "hello",
    "world",
    "welcome",
    "goodbye",
    "thanks",
    "please",
    "sorry",
    "help",
    "restaurant",
    "hotel",
    "airport",
    "station",
    "hospital",
    "library",
    "museum",
    "engineer",
    "designer",
    "manager",
    "analyst",
    "developer",
    "director",
    "intern",
    "algorithm",
    "structure",
    "iterator",
    "generic",
    "closure",
    "pointer",
    "buffer",
]

# Real Latin-1 (accented) words/names -- exercises `compute_u8::<false>`.
LATIN1_WORDS = [
    "café",
    "résumé",
    "naïve",
    "façade",
    "jalapeño",
    "Zürich",
    "smörgåsbord",
    "François",
    "Müller",
    "José",
    "André",
    "Björn",
    "Søren",
    "Ångström",
    "Köln",
    "München",
    "Genève",
    "Málaga",
    "Córdoba",
    "Düsseldorf",
    "señor",
    "señora",
    "niño",
    "año",
    "piñata",
    "jalapeños",
    "crème",
    "brûlée",
    "château",
    "élan",
    "über",
    "größe",
    "weiß",
    "straße",
    "año nuevo",
    "café con leche",
    "crème fraîche",
    "vis-à-vis",
]

# Real short Japanese words (hiragana/katakana) -- exercises `compute_sorted`
# on u16 (UCS-2) slices, the same encoding kind used for CJK ideographs.
CJK_WORDS = [
    "こんにちは",
    "ありがとう",
    "さようなら",
    "おはよう",
    "こんばんは",
    "カタカナ",
    "ひらがな",
    "にほんご",
    "ともだち",
    "がっこう",
    "でんわ",
    "テスト",
    "コンピュータ",
    "インターネット",
    "プログラム",
    "ソフトウェア",
    "きょう",
    "あした",
    "きのう",
    "せんせい",
]

# Short emoji sequences -- exercises `compute_sorted` on u32 (UCS-4) slices,
# a realistic case for usernames/handles/reactions.
EMOJI_WORDS = [
    "😀",
    "🎉",
    "🚀",
    "✨",
    "🐍",
    "🦀",
    "📦",
    "🔥",
    "💡",
    "🌟",
    "😀🎉",
    "🚀✨",
    "🐍🦀",
    "📦🔥",
    "💡🌟",
    "😀🚀🎉",
    "👍",
    "❤️",
    "😂",
    "🙏",
]

# ---------------------------------------------------------------------------
# Typo injection
# ---------------------------------------------------------------------------


def _inject_typos(rng: random.Random, s: str, alphabet: str, edit_ratio: float) -> str:
    """
    Return `s` mutated by a small number of realistic single-character edits.

    Applies substitution, insertion, deletion, and adjacent-transposition at
    random positions, drawing replacement/inserted characters from `alphabet`
    so the mutated string stays within the same CPython string kind as the
    input (e.g. CJK typos stay CJK rather than degrading to ASCII).

    Returns:
        The mutated string; identical to `s` if it is empty.

    """
    if not s:
        return s
    chars = list(s)
    n_edits = max(1, round(len(chars) * edit_ratio))
    for _ in range(n_edits):
        if not chars:
            break
        op = rng.choice(("sub", "sub", "ins", "del", "transpose"))
        pos = rng.randrange(len(chars))
        if op == "sub":
            chars[pos] = rng.choice(alphabet)
        elif op == "ins":
            chars.insert(pos, rng.choice(alphabet))
        elif op == "del" and len(chars) > 1:
            del chars[pos]
        elif op == "transpose" and len(chars) > 1:
            other = min(pos + 1, len(chars) - 1)
            chars[pos], chars[other] = chars[other], chars[pos]
    return "".join(chars)


# ---------------------------------------------------------------------------
# Corpus shape
# ---------------------------------------------------------------------------

# Source pool per CPython string kind, paired with the alphabet typos are
# drawn from so a mutated string stays in the kind it started in.
_POOLS: dict[str, tuple[list[str], str]] = {
    "ascii": (ASCII_WORDS, "abcdefghijklmnopqrstuvwxyz"),
    "latin1": (LATIN1_WORDS, "".join(LATIN1_WORDS)),
    "cjk": (CJK_WORDS, "".join(CJK_WORDS)),
    "emoji": (EMOJI_WORDS, "".join(EMOJI_WORDS)),
}

# Target lengths grouped into strata, with the share of the corpus each takes.
# Retains a realistic short-string bias while actually covering the bands the
# kernels branch on: mbleven, single-word Myers (<=64), multiword (>64), and
# the banded fallback (>512).
_STRATA: tuple[tuple[tuple[int, ...], float], ...] = (
    ((4, 8, 12, 16), 0.35),
    ((24, 32, 48, 56, 64, 72, 96, 128), 0.40),
    ((192, 256, 384, 512, 640), 0.25),
)

# Share of the corpus given to pairs whose sides have different kinds, so the
# `compute_sorted_mixed` arms get profile signal too.
_MIXED_SHARE = 0.08

# Share appended as `(s, s)` to weight the identity short-circuit.
_IDENTICAL_SHARE = 0.025


def _build(rng: random.Random, pool: list[str], target: int) -> str:
    """
    Concatenate words drawn from `pool` until `target` characters are reached.

    Returns:
        A string of exactly `target` characters.

    """
    out: list[str] = []
    total = 0
    while total < target:
        word = rng.choice(pool)
        out.append(word)
        total += len(word)
    return "".join(out)[:target]


def _identical_pairs(rng: random.Random, pool: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    """
    Duplicate `s1` as `s2` (same object) to weight the identity fast path.

    Returns:
        A list of `n` `(s, s)` pairs sampled from `pool`.

    """
    pairs = []
    for _ in range(n):
        s, _ = rng.choice(pool)
        pairs.append((s, s))
    return pairs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_pairs(seed: int = SEED, total: int = 10_000) -> list[tuple[str, str]]:
    """
    Return a kind-balanced, length-stratified corpus of `(s1, s2)` pairs.

    The four CPython string kinds get an even share rather than the
    ASCII-dominated mix that real fuzzy-matching traffic shows. PGO consumes
    branch and call counts, not semantics: under-representing a kind leaves
    its kernels cold, and profile-driven layout then makes them measurably
    *slower* than an unprofiled build. `seed` and `total` are exposed for
    testing; production callers should use the defaults so the resulting
    profile is reproducible.

    Returns:
        A list of `(s1, s2)` string pairs.

    """
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    per_kind = round(total * (1.0 - _MIXED_SHARE) / len(_POOLS))
    for pool, alphabet in _POOLS.values():
        for lengths, share in _STRATA:
            for _ in range(round(per_kind * share)):
                base = _build(rng, pool, rng.choice(lengths))
                pairs.append((base, _inject_typos(rng, base, alphabet, rng.uniform(0.05, 0.20))))

    kinds = list(_POOLS)
    strata_weights = [share for _lengths, share in _STRATA]
    for _ in range(round(total * _MIXED_SHARE)):
        a, b = rng.sample(kinds, 2)
        lengths, _share = rng.choices(_STRATA, weights=strata_weights)[0]
        target = rng.choice(lengths)
        pairs.append((_build(rng, _POOLS[a][0], target), _build(rng, _POOLS[b][0], target)))

    # Sampled from what's built so far, so it inherits the kind/length mix.
    pairs += _identical_pairs(rng, pairs, round(len(pairs) * _IDENTICAL_SHARE))

    rng.shuffle(pairs)
    return pairs
