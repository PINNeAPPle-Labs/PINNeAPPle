"""The electromagnetics verification suite (``pinneapple_simulation.electromagnetics.verification``): every case
compares a solver or reference implementation with an independent result (closed form, published value, another
library) and, where it refines, with the expected order. These are the numbers of the docs table."""
from __future__ import annotations

import pytest

from pinneapple_simulation.electromagnetics import verification as v


@pytest.mark.parametrize("name", list(v.CASES))
def test_verification_case(name):
    """Each case: the closed form or reference solution is matched within tolerance and at the expected order."""
    r = v.run([name], quick=True)[0]
    assert r.passed, r.row() + " " + r.detail


def test_table_renders():
    rows = v.to_markdown(v.run(["Coaxial mutual inductance", "Bragg mirror (characteristic matrices)"]))
    assert rows.count("| pass |") == 2 and rows.startswith("| Case |")
