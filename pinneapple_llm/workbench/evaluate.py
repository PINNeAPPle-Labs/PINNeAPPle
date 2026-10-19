"""A small, checkable benchmark of the workbench on your machine and model.

Each task builds a tiny project in a temporary folder, gives the lead a request, and checks the outcome mechanically
(the project's tests pass, a file has the expected content, the answer names the right place). It reports pass/fail,
wall time, model calls and tokens per task: numbers measured on your hardware, for comparing local models with each
other. It is not a substitute for the public agentic-coding benchmarks.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .backend import Backend
from .runtime import Session, SessionSettings
from .tools import ApprovalPolicy

__all__ = ["TASKS", "TaskResult", "run_eval"]


@dataclass
class TaskResult:
    name: str
    passed: bool
    seconds: float
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    detail: str = ""


def _pytest(ws: Path) -> tuple[bool, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ws,
                       capture_output=True, text=True, timeout=300)
    return p.returncode == 0, (p.stdout + p.stderr)[-400:]


def _setup_bug(ws: Path) -> None:
    (ws / "stats.py").write_text(
        "def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
        "def variance(xs):\n    m = mean(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs) - 1\n")
    (ws / "test_stats.py").write_text(
        "from stats import mean, variance\n\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n\n\n"
        "def test_variance():\n    assert abs(variance([1, 2, 3, 4]) - 1.6666666666666667) < 1e-12\n")


def _check_bug(ws: Path, answer: str) -> tuple[bool, str]:
    ok, out = _pytest(ws)
    tests_intact = "1.6666666666666667" in (ws / "test_stats.py").read_text()
    return ok and tests_intact, out if tests_intact else "the test was changed"


def _setup_feature(ws: Path) -> None:
    (ws / "units.py").write_text("def c_to_f(c):\n    return c * 9 / 5 + 32\n")
    (ws / "test_units.py").write_text("from units import c_to_f\n\n\ndef test_c_to_f():\n    assert c_to_f(100) == 212\n")


def _check_feature(ws: Path, answer: str) -> tuple[bool, str]:
    ok, out = _pytest(ws)
    if not ok:
        return False, out
    probe = subprocess.run([sys.executable, "-c", "from units import f_to_c; assert abs(f_to_c(212) - 100) < 1e-9; "
                            "assert abs(f_to_c(-40) + 40) < 1e-9"], cwd=ws, capture_output=True, text=True)
    has_test = "f_to_c" in (ws / "test_units.py").read_text()
    return probe.returncode == 0 and has_test, probe.stderr[-300:] or ("" if has_test else "no test for f_to_c")


def _setup_locate(ws: Path) -> None:
    (ws / "pkg").mkdir()
    (ws / "pkg" / "__init__.py").write_text("")
    (ws / "pkg" / "io.py").write_text("import json\n\n\ndef load(path):\n    with open(path) as f:\n"
                                      "        return json.load(f)\n")
    (ws / "pkg" / "retry.py").write_text("import time\n\n\ndef backoff(n, base=0.5):\n"
                                         "    return [base * 2 ** i for i in range(n)]\n")


def _check_locate(ws: Path, answer: str) -> tuple[bool, str]:
    return "retry.py" in answer and "backoff" in answer, answer[-300:]


TASKS: dict[str, tuple[Callable[[Path], None], str, Callable[[Path, str], tuple[bool, str]]]] = {
    "fix_failing_test": (_setup_bug, "The test suite fails. Find the bug in the code (not the test), fix it, and run "
                                     "the tests to confirm.", _check_bug),
    "add_feature_with_test": (_setup_feature, "Add f_to_c(f) to units.py, the inverse of c_to_f, with a test in "
                                              "test_units.py. Run the tests.", _check_feature),
    "locate_code": (_setup_locate, "Which file and function compute exponential back-off delays? Answer with the "
                                   "file path and the function name.", _check_locate),
}


def run_eval(backend: Backend, names: list[str] | None = None, on_event=None) -> list[TaskResult]:
    results = []
    for name in names or list(TASKS):
        setup, request, check = TASKS[name]
        with tempfile.TemporaryDirectory(prefix=f"wb-eval-{name}-") as d:
            ws = Path(d)
            setup(ws)
            sess = Session(backend, str(ws), policy=ApprovalPolicy("yolo"), settings=SessionSettings(max_steps=25),
                           on_event=on_event)
            t0 = time.time()
            try:
                answer = sess.send(request)
                passed, detail = check(ws, answer)
            except Exception as e:  # noqa: BLE001 - a crash is a failed task
                passed, detail = False, f"{type(e).__name__}: {e}"
            u = sess.usage
            results.append(TaskResult(name, passed, round(time.time() - t0, 1), u["model_calls"],
                                      u["prompt_tokens"], u["completion_tokens"], detail.strip()))
    return results


def format_results(results: list[TaskResult], model: str) -> str:
    rows = [f"Workbench eval, model {model}", "", f"{'task':24s} {'result':6s} {'time s':>7s} {'calls':>5s} "
            f"{'tokens in':>9s} {'out':>6s}"]
    for r in results:
        rows.append(f"{r.name:24s} {'PASS' if r.passed else 'FAIL':6s} {r.seconds:7.1f} {r.model_calls:5d} "
                    f"{r.prompt_tokens:9d} {r.completion_tokens:6d}")
    rows.append(f"\n{sum(r.passed for r in results)}/{len(results)} passed")
    return "\n".join(rows)


def as_dicts(results: list[TaskResult]) -> list[dict]:
    return [asdict(r) for r in results]
