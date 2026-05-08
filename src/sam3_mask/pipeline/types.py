from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RunSummary:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    empty: int = 0
