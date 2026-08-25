"""
Deterministic, network-free training corpus for PGO.

Generates `(s1, s2)` string pairs shaped like real fuzzy-matching traffic —
search-suggestion / autocomplete / contact-dedup / spell-check workloads —
rather than synthetic worst-case inputs. The mix is heavily weighted toward
short ASCII words with realistic typos (the dominant real-world case), with
a proportionally small tail into Latin-1 names, CJK/emoji tokens, mixed-kind
pairs, multi-word sentences, and very long strings, so every dispatch arm in
`compute()` (src/lib.rs) gets *some* profile signal without distorting the
corpus toward rare cases.

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

# A handful of full sentences/paragraphs for the multi-word (65-512 char) and
# long-tail (>512 char) buckets -- realistic near-duplicate-detection inputs
# (product reviews, support tickets, error messages), not padded filler.
PROSE = [
    "Customer support was fantastic today. The technician arrived on time, "
    "diagnosed the issue within minutes, and had everything working again "
    "before lunch. I would definitely recommend this service to a friend "
    "or colleague.",
    "The package arrived two days late and the box was slightly damaged, "
    "but everything inside was intact. Customer service issued a partial "
    "refund without any hassle, which I really appreciated given how "
    "stressful the whole week had already been.",
    "Error: connection to the database timed out after 30 seconds. Please "
    "check that the server is reachable and that the configured credentials "
    "are still valid, then retry the operation. If the problem persists, "
    "contact your system administrator.",
    "We are writing to confirm that your subscription has been renewed for "
    "another twelve months. Your next billing date is scheduled for the "
    "first of next month, and you can review or cancel your plan at any "
    "time from the account settings page.",
    "The new office is located just two blocks from the train station, "
    "with plenty of parking nearby and a small café on the ground floor "
    "that serves breakfast starting at seven in the morning every weekday.",
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


def _make_pair(rng: random.Random, base: str, alphabet: str, edit_ratio: float) -> tuple[str, str]:
    return base, _inject_typos(rng, base, alphabet, edit_ratio)


# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------

_ASCII_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_LATIN1_ALPHABET = "".join(LATIN1_WORDS)
_CJK_ALPHABET = "".join(CJK_WORDS)
_EMOJI_ALPHABET = "".join(EMOJI_WORDS)


def _ascii_short_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pairs = []
    for _ in range(n):
        n_words = rng.choice((1, 1, 1, 2))
        base = " ".join(rng.choice(ASCII_WORDS) for _ in range(n_words))
        pairs.append(_make_pair(rng, base, _ASCII_ALPHABET, rng.uniform(0.10, 0.25)))
    return pairs


def _latin1_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pairs = []
    for _ in range(n):
        n_words = rng.choice((1, 1, 2))
        base = " ".join(rng.choice(LATIN1_WORDS) for _ in range(n_words))
        pairs.append(_make_pair(rng, base, _LATIN1_ALPHABET, rng.uniform(0.10, 0.25)))
    return pairs


def _short_phrase_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pairs = []
    for _ in range(n):
        base = " ".join(rng.choice(ASCII_WORDS) for _ in range(rng.randint(3, 6)))
        pairs.append(_make_pair(rng, base, _ASCII_ALPHABET, rng.uniform(0.08, 0.18)))
    return pairs


def _multiword_sentence_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pairs = []
    for _ in range(n):
        base = rng.choice(PROSE)
        pairs.append(_make_pair(rng, base, _ASCII_ALPHABET, rng.uniform(0.03, 0.10)))
    return pairs


def _cjk_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pairs = []
    for _ in range(n):
        base = "".join(rng.choice(CJK_WORDS) for _ in range(rng.randint(1, 3)))
        pairs.append(_make_pair(rng, base, _CJK_ALPHABET, rng.uniform(0.10, 0.25)))
    return pairs


def _emoji_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pairs = []
    for _ in range(n):
        base = "".join(rng.choice(EMOJI_WORDS) for _ in range(rng.randint(1, 2)))
        pairs.append(_make_pair(rng, base, _EMOJI_ALPHABET, rng.uniform(0.10, 0.25)))
    return pairs


def _mixed_kind_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pools = (ASCII_WORDS, LATIN1_WORDS, CJK_WORDS, EMOJI_WORDS)
    pairs = []
    for _ in range(n):
        a_pool, b_pool = rng.sample(pools, 2)
        pairs.append((rng.choice(a_pool), rng.choice(b_pool)))
    return pairs


def _long_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pairs = []
    for _ in range(n):
        base = " ".join(rng.choice(PROSE) for _ in range(rng.randint(2, 4)))
        pairs.append(_make_pair(rng, base, _ASCII_ALPHABET, rng.uniform(0.02, 0.08)))
    return pairs


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
    Return a realistic, reproducible corpus of `(s1, s2)` pairs.

    Proportions approximate real fuzzy-matching traffic: dominated by short
    ASCII words/phrases, with a small tail into Latin-1, CJK, emoji,
    mixed-kind, multi-word, and very-long-string territory. `seed` and
    `total` are exposed for testing; production callers should use the
    defaults so the resulting profile is reproducible.

    Returns:
        A list of `(s1, s2)` string pairs.

    """
    rng = random.Random(seed)

    counts = {
        "ascii": round(total * 0.78),
        "latin1": round(total * 0.10),
        "phrase": round(total * 0.05),
        "sentence": round(total * 0.04),
        "cjk": round(total * 0.015),
        "emoji": round(total * 0.01),
        "mixed": round(total * 0.005),
        "long": round(total * 0.003),
    }

    pairs: list[tuple[str, str]] = []
    pairs += _ascii_short_pairs(rng, counts["ascii"])
    pairs += _latin1_pairs(rng, counts["latin1"])
    pairs += _short_phrase_pairs(rng, counts["phrase"])
    pairs += _multiword_sentence_pairs(rng, counts["sentence"])
    pairs += _cjk_pairs(rng, counts["cjk"])
    pairs += _emoji_pairs(rng, counts["emoji"])
    pairs += _mixed_kind_pairs(rng, counts["mixed"])
    pairs += _long_pairs(rng, counts["long"])

    # Identity short-circuit: ~2.5% of the corpus, sampled from what's built
    # so far so it inherits the same kind/length distribution.
    pairs += _identical_pairs(rng, pairs, round(len(pairs) * 0.025))

    rng.shuffle(pairs)
    return pairs
