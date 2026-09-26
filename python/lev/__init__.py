"""An extremely fast Python library for the Levenshtein distance and similarity ratio."""

from ._lev import distance, ratio

__all__ = ["distance", "ratio"]


def __dir__() -> list[str]:
    """
    List the public API only (PEP 562), hiding the private `_lev` extension.

    Returns:
        list[str]: the names in `__all__`.

    """
    return list(__all__)
