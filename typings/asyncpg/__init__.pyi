"""Minimal type stub for asyncpg to satisfy editors and linters.

This file intentionally contains only lightweight declarations so the
language server stops reporting 'missing import' while development
environment or build issues (C extension builds) are addressed separately.
"""

from typing import Any, Awaitable, Optional, Protocol


class Pool(Protocol):
    async def acquire(self) -> Any: ...
    async def close(self) -> None: ...


def create_pool(*args: Any, **kwargs: Any) -> Awaitable[Pool]: ...
