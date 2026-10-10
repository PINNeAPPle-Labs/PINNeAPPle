"""Tests for pinneapple_veriphysics.execution_log."""
from __future__ import annotations

import time

from pinneapple_veriphysics.execution_log import LogCollector, emit


def test_log_collector_records_entries_in_order():
    collector = LogCollector()
    collector.log("step one")
    collector.log("step two")
    entries = collector.entries()
    assert [e.message for e in entries] == ["step one", "step two"]
    assert entries[0].timestamp_unix <= entries[1].timestamp_unix


def test_log_collector_starts_empty():
    assert LogCollector().entries() == []


def test_emit_writes_to_a_real_collector():
    collector = LogCollector()
    emit(collector, "hello")
    assert [e.message for e in collector.entries()] == ["hello"]


def test_emit_is_a_real_noop_when_no_collector_given():
    emit(None, "should not raise")  # must not raise


def test_log_collector_thread_safe_under_concurrent_writes():
    import threading

    collector = LogCollector()

    def _writer(n):
        for i in range(20):
            collector.log(f"writer-{n}-{i}")

    threads = [threading.Thread(target=_writer, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(collector.entries()) == 100  # 5 writers x 20 entries, none lost/corrupted
