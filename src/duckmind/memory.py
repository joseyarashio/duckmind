"""Memory port: cognition reads grounded context; writes remain evidence-gated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from duckmind.provenance import Event, LearningUpdate


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    text: str
    source_event_ids: tuple[str, ...]
    reliability: float


class MemoryPort(Protocol):
    def append_event(self, event: Event) -> None: ...

    def recall(self, query: str, *, limit: int) -> tuple[MemoryRecord, ...]: ...

    def propose_learning(self, update: LearningUpdate) -> None: ...

    def commit_learning(self, update_id: str) -> None: ...


class MemoryUnavailable(RuntimeError):
    """Recall may fail without stopping System 1 or the body safety path."""
