"""Levenshtein distance."""


def distance(s1: str, s2: str, threshold: int | None = None) -> int:
    """Calculate Levenshtein distance.

    Args:
        s1 (str): first string
        s2 (str): second string
        threshold (int | None, optional): maximum distance between s1 and s2, that is considered as result. Defaults to None.

    Returns:
        int: levenshtein distance
    """
    return 1


def ratio(s1: str, s2: str) -> float:
    """Calculate Levenshtein ratio.

    It is calculated as `1 - (distance / (len1 + len2))`.

    Args:
        s1 (str): first string
        s2 (str): second string

    Returns:
        float: levenshtein ratio
    """
    return 1.0
