"""
Deterministic, network-free, rebalanced training corpus for PGO.

Generates `(s1, s2)` string pairs covering:
1. Lengths from 1 to 4096 across all internal dispatch regimes:
   - Tiny SWAR (1 <= m <= 8)
   - Single-word Myers (9 <= m <= 64)
   - Multiword Myers (65 <= m <= 512)
   - Large / banded Ukkonen and full matrix fallback (m > 512 up to 4096)
2. All four CPython string kinds:
   - UCS-1 ASCII
   - UCS-1 Latin-1
   - UCS-2 (CJK / Cyrillic)
   - UCS-4 (Emoji)
   - Mixed-kind pairs (UCS-1 x UCS-2, UCS-1 x UCS-4, UCS-2 x UCS-4)
3. Diverse structural regimes:
   - Identical strings (identity fast-path)
   - Near-identical strings (1 to 3 edits: mbleven and narrow Ukkonen band)
   - Common affixes (shared prefix + core + shared suffix)
   - Moderate typos (5% - 20% edits)
   - Completely dissimilar / disjoint strings (early-abort & full matrix fallback)
   - Disparate lengths (m << n non-square matrices)

Balanced to prevent LLVM from treating UCS-2 and UCS-4 hash table lookups as cold.
"""

from __future__ import annotations

import random

SEED = 0x1E7_2026

# ---------------------------------------------------------------------------
# Word & Alphabet Pools
# ---------------------------------------------------------------------------

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
    "computer",
    "keyboard",
    "monitor",
    "printer",
    "router",
    "server",
    "laptop",
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
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "runs",
    "fast",
    "algorithm",
    "structure",
    "iterator",
    "generic",
    "closure",
    "pointer",
    "buffer",
]
ASCII_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

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
    "Valparaíso",
    "București",
    "Malmö",
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
    "vis-à-vis",
    "voilà",
    "déjà",
]
LATIN1_ALPHABET = "abcdefghijklmnopqrstuvwxyzàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ"

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
    "東京",
    "京都",
    "大阪",
    "日本",
    "世界",
    "富士山",
    "桜",
    "漢字",
    "文字列",
    "検索",
    "距離",
    "計算",
]
CJK_ALPHABET = "".join(CJK_WORDS)

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
    "🐱",
    "🐶",
    "🦊",
    "🦁",
    "🐼",
    "🐨",
    "🐯",
    "🦄",
    "🌈",
    "⚡",
    "🍀",
    "🍕",
]
EMOJI_ALPHABET = "".join(EMOJI_WORDS)

_POOLS: dict[str, tuple[list[str], str]] = {
    "ascii": (ASCII_WORDS, ASCII_ALPHABET),
    "latin1": (LATIN1_WORDS, LATIN1_ALPHABET),
    "ucs2": (CJK_WORDS, CJK_ALPHABET),
    "ucs4": (EMOJI_WORDS, EMOJI_ALPHABET),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_str(rng: random.Random, pool: list[str], target_len: int) -> str:
    """
    Concatenate pool words until target_len is reached.

    Returns:
        Generated string of target_len.

    """
    if target_len <= 0:
        return ""
    out: list[str] = []
    total = 0
    while total < target_len:
        w = rng.choice(pool)
        out.append(w)
        total += len(w)
    return "".join(out)[:target_len]


def _mutate(rng: random.Random, s: str, alphabet: str, n_edits: int) -> str:
    """
    Apply exactly n_edits insertions, deletions, substitutions, or transpositions.

    Returns:
        Mutated string.

    """
    if not s or n_edits <= 0:
        return s
    chars = list(s)
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
# Public generator
# ---------------------------------------------------------------------------


def generate_pairs(seed: int = SEED, total: int = 12_000) -> list[tuple[str, str]]:
    """
    Generate a balanced, multi-regime training corpus spanning lengths 1..4096.

    Weighting gives UCS-2 and UCS-4 strong representation alongside ASCII and
    Latin-1 so LLVM does not deprioritize non-UCS-1 hash lookup code.

    Returns:
        List of (s1, s2) pairs.

    """
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    # Length strata covering all algorithm gates
    tiny_lengths = (1, 2, 3, 4, 6, 8)
    short_lengths = (12, 16, 24, 32, 48, 64)
    medium_lengths = (72, 96, 128, 192, 256, 384, 512)
    large_lengths = (640, 800, 1024, 1536, 2048, 4096)

    all_length_groups = [
        (tiny_lengths, 0.20),
        (short_lengths, 0.35),
        (medium_lengths, 0.25),
        (large_lengths, 0.20),
    ]

    # Equal allocation for kinds: 25% ascii, 20% latin1, 27.5% ucs2, 27.5% ucs4
    kind_shares = {
        "ascii": 0.22,
        "latin1": 0.18,
        "ucs2": 0.30,
        "ucs4": 0.30,
    }

    # 1. Standard intra-kind pairs with various edit ratios
    for kind, (pool, alphabet) in _POOLS.items():
        kind_total = round(total * 0.70 * kind_shares[kind])
        for lengths, group_share in all_length_groups:
            n_pairs = round(kind_total * group_share)
            for _ in range(n_pairs):
                target = rng.choice(lengths)
                s1 = _build_str(rng, pool, target)
                mode = rng.random()
                if mode < 0.25:
                    # Small edits: 1 to 3 edits (trains mbleven and Ukkonen narrow band)
                    s2 = _mutate(rng, s1, alphabet, rng.randint(1, min(3, max(1, len(s1)))))
                elif mode < 0.60:
                    # Moderate edits (5% - 15%)
                    edits = max(1, round(len(s1) * rng.uniform(0.05, 0.15)))
                    s2 = _mutate(rng, s1, alphabet, edits)
                elif mode < 0.85:
                    # Dissimilar / unrelated strings (trains full matrix and abort)
                    s2 = _build_str(rng, pool, target)
                else:
                    # Common affixes: shared prefix + suffix
                    prefix = _build_str(rng, pool, rng.randint(2, max(3, len(s1) // 3)))
                    suffix = _build_str(rng, pool, rng.randint(2, max(3, len(s1) // 3)))
                    mid1 = _build_str(rng, pool, rng.randint(1, 16))
                    mid2 = _build_str(rng, pool, rng.randint(1, 16))
                    s1 = prefix + mid1 + suffix
                    s2 = prefix + mid2 + suffix
                pairs.append((s1, s2))

    # 2. Mixed-kind pairs (~12% of total)
    n_mixed = round(total * 0.12)
    kinds = list(_POOLS.keys())
    for _ in range(n_mixed):
        k1, k2 = rng.sample(kinds, 2)
        group = rng.choice(all_length_groups)[0]
        l1 = rng.choice(group)
        # some equal length, some unequal
        l2 = l1 if rng.random() < 0.6 else rng.choice(group)
        s1 = _build_str(rng, _POOLS[k1][0], l1)
        s2 = _build_str(rng, _POOLS[k2][0], l2)
        pairs.append((s1, s2))

    # 3. Disparate lengths (m << n) (~10% of total)
    n_disparate = round(total * 0.10)
    for _ in range(n_disparate):
        kind = rng.choice(kinds)
        pool, alphabet = _POOLS[kind]
        m = rng.choice((2, 4, 8, 16, 32, 64))
        n = m * rng.randint(3, 20)
        s1 = _build_str(rng, pool, m)
        s2 = _build_str(rng, pool, n)
        # either random or substring
        if rng.random() < 0.5:
            start = rng.randrange(len(s2) - len(s1) + 1)
            s2 = s2[:start] + _mutate(rng, s1, alphabet, 1) + s2[start + len(s1) :]
        pairs.append((s1, s2))

    # 4. Identity fast-path (~5% of total)
    n_identical = round(total * 0.05)
    for _ in range(n_identical):
        s, _ = rng.choice(pairs)
        pairs.append((s, s))

    rng.shuffle(pairs)
    return pairs
