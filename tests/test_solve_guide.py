"""The runnable example in docs/guides/solve_a_problem.md runs as written.

Same discipline as ``tests/test_agents_md.py``: a guide that promises four lines a reader can paste is only worth
the promise if the four lines still run. This executes the block and checks that the comparison it prints graded
both methods ``ok`` -- if a method stops being able to solve the Poisson case, the guide fails here rather than on
a reader's machine.
"""
from __future__ import annotations

import re
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guides" / "solve_a_problem.md"
BLOCKS = re.findall(r"```python\n(.*?)```", GUIDE.read_text(encoding="utf-8"), flags=re.S)


def test_the_guide_has_one_runnable_block():
    assert len(BLOCKS) == 1, f"expected one Python block in {GUIDE.name}, found {len(BLOCKS)}"


def test_the_guide_example_runs_and_grades_both_methods(capsys):
    namespace: dict = {"__name__": "solve_guide"}
    exec(compile(BLOCKS[0], "docs/guides/solve_a_problem.md", "exec"), namespace)

    printed = capsys.readouterr().out
    rows = [line.split() for line in printed.splitlines() if line.startswith(("fem", "pinn"))]
    assert len(rows) == 2, printed
    assert all("ok" in row for row in rows), printed
    assert namespace["table"] is not None
