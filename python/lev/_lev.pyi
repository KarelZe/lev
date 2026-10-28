"""Type stubs for the private `_lev` extension module backing `lev`."""

def distance(s1: str, s2: str, /) -> int:
    """
    Calculate the Levenshtein edit distance between two strings.

    The Levenshtein distance is the minimum number of single-character insertions,
    deletions, or substitutions required to transform `s1` into `s2`.

    Characters are counted as Unicode code points, so an accented letter or an
    emoji counts as a single character.

    Args:
        s1: First input string.
        s2: Second input string.

    Returns:
        The non-negative integer edit distance between `s1` and `s2`.

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
    Calculate the normalized similarity ratio between two strings in `[0.0, 1.0]`.

    A score of `1.0` indicates identical strings, while `0.0` indicates strings
    with no common characters. The score represents the proportion of characters
    shared between the two strings, defined as the indel similarity ratio:

    $$\\mathrm{ratio}(s_1, s_2) = \\frac{2 \\cdot \\mathrm{LCS}(s_1, s_2)}{|s_1| + |s_2|}$$

    where $\\mathrm{LCS}$ is the length of the longest common subsequence and
    $|s|$ is the length of string $s$.

    Characters are counted as Unicode code points, so accented letters and
    emojis count as single characters.

    Args:
        s1: First input string.
        s2: Second input string.

    Returns:
        Similarity score between `0.0` (completely disjoint) and `1.0` (identical).

    Examples:
        >>> import lev
        >>> lev.ratio("kitten", "sitting")
        0.6153846153846154
        >>> lev.ratio("abc", "xyz")
        0.0
        >>> lev.ratio("", "")
        1.0
        >>> lev.ratio("résumé", "resume")
        0.6666666666666666

    """
    ...
