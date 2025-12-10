"""Type definitions for similarity detection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FunctionInfo:
    """Information about an extracted function."""

    name: str
    file: str
    start_line: int
    end_line: int
    loc: int
    hash: str
    text: str
    embedding: list[float] | None = field(default=None, repr=False)
