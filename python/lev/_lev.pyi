"""Type stubs for the private `_lev` extension module backing `lev`."""

def distance(s1: str, s2: str, /) -> int:
    """
    Levenshtein edit distance between two strings.

    The distance is the minimum number of single-character insertions,
    deletions, or substitutions required to transform `s1` into `s2`.

    Characters are counted as Unicode code points, not as visible characters:
    `"é"` written as one code point counts as 1, but `"e"` plus a combining
    accent, flags, and emoji with skin tones or joiners count as 2 or more.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.

    Returns:
        Non-negative integer edit distance.

    Examples:
        >>> import lev
        >>> lev.distance("kitten", "sitting")
        3
        >>> lev.distance("flaw", "lawn")
        2
        >>> lev.distance("résumé", "resume")
        2
        >>> lev.distance("日本語", "日本")
        1

    """
    ...

def ratio(s1: str, s2: str, /) -> float:
    r"""
    Calculate how similar two strings are, as a score from `0.0` to `1.0`.

    A score of `1.0` means the strings are identical, and `0.0` means they
    have no character in common.  In between, the score is the share of
    characters the two strings have in common, known as the indel similarity
    ratio:

    $$
    \mathrm{ratio}(s_1, s_2)
      = \frac{2 \cdot \mathrm{LCS}(s_1, s_2)}{|s_1| + |s_2|}
    $$

    where $\mathrm{LCS}$ is the length of the longest common subsequence and
    $|s|$ is the length of $s$.

    This matches `rapidfuzz.fuzz.ratio(s1, s2) / 100` and `Levenshtein.ratio`.

    Characters are counted as Unicode code points, not as visible characters:
    `"é"` written as one code point counts as 1, but `"e"` plus a combining
    accent, flags, and emoji with skin tones or joiners count as 2 or more.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.

    Returns:
        Similarity score between `0.0` (nothing in common) and `1.0` (identical).

    Examples:
        >>> import lev
        >>> lev.ratio("kitten", "sitting")
        0.6153846153846154
        >>> lev.ratio("abc", "xyz")
        0.0
        >>> lev.ratio("", "")
        1.0
        >>> lev.ratio("résumé", "resume")
        0.6666666666666667

    """
    ...
