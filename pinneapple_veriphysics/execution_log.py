"""A real, thread-safe execution log collector -- so a caller (the API,
the UI) can see WHAT the orchestrated pipeline is actually doing while
it runs (which step it's on, which tool/solver/architecture it just
picked), not just its final result. Every entry is written by the
pipeline itself, at the moment it makes a real decision -- never
reconstructed after the fact or guessed.

(Extracted from Veriphysics' orchestrator; Apache-2.0 like the rest of PINNeAPPle. Execution queue, API and billing stay in Veriphysics.)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["LogEntry", "LogCollector"]


@dataclass(frozen=True)
class LogEntry:
    timestamp_unix: float
    message: str


class LogCollector:
    """Pass an instance of this into `analyze()`/`formulate_and_recommend()`
    (via their optional `log=` parameter) to observe the real steps
    they take as they take them. Safe to read from another thread while
    the pipeline is still running (e.g. an API handler polling it)."""

    def __init__(self) -> None:
        self._entries: List[LogEntry] = []
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        with self._lock:
            self._entries.append(LogEntry(timestamp_unix=time.time(), message=message))

    def entries(self) -> List[LogEntry]:
        with self._lock:
            return list(self._entries)


def emit(log: Optional[LogCollector], message: str) -> None:
    """Convenience helper: log to *log* if one was actually given,
    otherwise a real no-op -- every call site in the pipeline can call
    this unconditionally without an `if log is not None` guard."""
    if log is not None:
        log.log(message)
