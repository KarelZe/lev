"""Pytest configuration for the `lev` test suite."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

# Tag benchmarks with the running interpreter version so CodSpeed keeps the
# results from each matrix job (py3.13/3.14/3.15) distinct instead of merging
# identically-named benchmarks.
_PY_TAG = f"py{sys.version_info.major}.{sys.version_info.minor}"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Append the Python version to each benchmark's node id."""
    for item in items:
        if item.get_closest_marker("benchmark") is not None:
            item._nodeid = f"{item._nodeid}[{_PY_TAG}]"
