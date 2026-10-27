"""Sandbox tests: what the controlled executor actually stops, and what it does not.

These tests exercise the real subprocess path (no mocks): a timeout, a
network attempt, an out-of-directory file read and a process spawn. They
also pin the *documented limitation* -- the executor must refuse to run at
all unless execution was explicitly enabled -- so a future change cannot
quietly make it permissive.
"""
from __future__ import annotations

import os

import pytest

from pinneapple_llm.qa import ISOLATION_LIMITATIONS, IsolationLevel, SandboxExecutor, SandboxLimits, Status


@pytest.fixture()
def executor():
    return SandboxExecutor(allow_execution=True,
                           limits=SandboxLimits(timeout_s=20, memory_mb=None))


def test_execution_is_blocked_unless_explicitly_enabled():
    result = SandboxExecutor().run_python("print('should not run')")
    assert result.status is Status.BLOCKED
    assert result.isolation is IsolationLevel.BLOCKED
    assert "disabled" in result.reason
    assert result.stdout == ""


def test_clean_script_passes_and_records_produced_files(executor):
    result = executor.run_python(
        "open('out.txt', 'w').write('hello')\nprint('done')\n")
    assert result.status is Status.PASS, result.stderr
    assert result.exit_code == 0
    assert [p.name for p in result.produced] == ["out.txt"]
    assert result.produced[0].sha256


def test_timeout_kills_the_process(executor):
    result = executor.run_python("import time\nwhile True:\n    time.sleep(0.05)\n")
    assert result.status is Status.FAIL
    assert result.timed_out is True
    assert "timed out" in result.reason


def test_network_access_is_refused(executor):
    result = executor.run_python(
        "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)\n")
    assert result.status is Status.FAIL
    assert any(v.startswith("network") for v in result.violations)


def test_reading_outside_the_workdir_is_refused(executor):
    result = executor.run_python("open('/etc/hosts').read()\n")
    assert result.status is Status.FAIL
    assert any(v.startswith("file_read") for v in result.violations)


def test_process_spawn_is_refused(executor):
    result = executor.run_python("import subprocess\nsubprocess.run(['echo', 'hi'])\n")
    assert result.status is Status.FAIL
    assert any(v.startswith("process") for v in result.violations)


def test_exception_in_artifact_is_a_failure_not_a_pass(executor):
    result = executor.run_python("raise RuntimeError('boom')\n")
    assert result.status is Status.FAIL
    assert result.exit_code != 0
    assert "boom" in result.stderr


def test_limitations_are_reported_with_every_result(executor):
    result = executor.run_python("print(1)\n")
    assert list(result.limitations) == list(ISOLATION_LIMITATIONS)
    assert any("subprocess" in entry for entry in result.limitations)


def test_result_is_serialisable(executor):
    result = executor.run_python("print(1)\n")
    payload = result.to_dict()
    assert payload["status"] == "PASS"
    assert payload["isolation"] == "subprocess"
