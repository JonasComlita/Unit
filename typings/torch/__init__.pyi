"""Minimal stub for torch to silence editor diagnostics when full install
is not available. This is intentionally tiny and only for dev-time editing.
"""

from typing import Any


class Tensor: ...


def tensor(*args: Any, **kwargs: Any) -> Tensor: ...
