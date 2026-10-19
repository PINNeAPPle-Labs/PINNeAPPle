"""FMI import compatibility check: a pythonfmu FMI 2.0 Co-Simulation FMU passes, a broken FMU is reported as a
failed row, and the table groups rows by exporting tool and FMI version."""
import textwrap

import pytest

pytest.importorskip("fmpy")
pytest.importorskip("pythonfmu")

from pinneapple_simulation.external_solvers.modelica.fmi_compat import (  # noqa: E402
    CompatRow,
    check_fmus,
    markdown_table,
    platform_tag,
)

SOURCE = '''
from pythonfmu import Fmi2Causality, Fmi2Slave, Real


class {name}(Fmi2Slave):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x = 1.0
        self.register_variable(Real("x", causality=Fmi2Causality.output))

    def do_step(self, t, dt):
        {body}
        return True
'''


def _build(tmp_path, name, body):
    from pythonfmu.builder import FmuBuilder

    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(SOURCE.format(name=name, body=body)))
    return FmuBuilder.build_FMU(script, dest=tmp_path)


def test_a_working_fmu_passes_and_a_broken_one_is_reported(tmp_path):
    good = _build(tmp_path, "Decay", "self.x = self.x * (1 - dt)")
    bad = _build(tmp_path, "Broken", "raise RuntimeError('boom')")
    rows = check_fmus([good, bad], exporter="pythonfmu")
    by = {r.fmu: r for r in rows}
    assert by["Decay"].passed and by["Decay"].interface == "CS" and by["Decay"].fmi_version == "2.0"
    assert not by["Broken"].passed and by["Broken"].detail
    assert all(r.platform == platform_tag() for r in rows)


def test_markdown_table_groups_by_exporter_and_version():
    rows = [CompatRow("A", "2.0", "CS", "Tool 1", "x86_64-linux", True),
            CompatRow("A", "2.0", "ME", "Tool 1", "x86_64-linux", True),
            CompatRow("B", "3.0", "CS", "Tool 1", "x86_64-linux", False, "error")]
    md = markdown_table(rows)
    assert "**Tool 1, FMI 2.0**" in md and "**Tool 1, FMI 3.0**" in md
    assert "| A | CS, ME |" in md and "| B | (failed: CS) |" in md
