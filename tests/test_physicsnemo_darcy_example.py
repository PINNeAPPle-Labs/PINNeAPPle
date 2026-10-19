"""The exported predictions of examples/vs_physicsnemo/07: the verdict the figure shows next to each prediction
(``check/sample_verdicts.json``, from ``darcy_checks.sample_checks``) is what ``pinneapple check`` returns for it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinneapple_check import run
from pinneapple_physics import darcy_checks as dc

try:
    import tomllib
except ModuleNotFoundError:                               # Python 3.10
    import tomli as tomllib

CHECK = Path(__file__).resolve().parents[1] / "examples/vs_physicsnemo/07_physicsnemo_trains_pinneapple_checks/check"
RULE = {"energy_bound": "PC301", "l2_bound": "PC301", "flux": "PC101", "bc": "PC201", "positivity": "PC402",
        "a_min": "PC401", "a_max": "PC401", "contrast": "PC401", "high_fraction": "PC401",
        "interface_density": "PC401"}


@pytest.fixture(scope="module")
def report():
    return run([CHECK / "predictions"], config=CHECK / "pinneapple-check.toml")


def _targets():
    return {t["name"]: t for t in tomllib.loads((CHECK / "pinneapple-check.toml").read_text())["result"]}


def test_every_exported_prediction_is_declared():
    files = {p.relative_to(CHECK).as_posix() for p in (CHECK / "predictions").rglob("*.npz")}
    assert files == {t["path"] for t in _targets().values()} and len(files) == 12


def test_figure_verdicts_equal_pinneapple_check(report):
    shown = json.loads((CHECK / "sample_verdicts.json").read_text())
    failed_rules: dict[str, set[str]] = {}
    for v in report.violations:
        failed_rules.setdefault(v.target, set()).add(v.rule)
    for name, t in _targets().items():
        with np.load(CHECK / t["path"]) as z:
            checks = dc.sample_checks(z["a"].astype(float), z["u"].astype(float), t["parameters"]["f"],
                                      tolerances={k: t["tolerances"][v] for k, v in dc.CHECK_TOLERANCE_KEYS.items()},
                                      envelope=dc.CoefficientEnvelope.from_check_config(t["envelope"]))
        verdict = dc.sample_verdict(checks)
        assert verdict == shown[name]["verdict"], name                              # library == figure
        assert verdict == ("FAIL" if name in failed_rules else "PASS"), name        # library == CI
        assert {RULE[c["key"]] for c in checks if not c["passed"]} == failed_rules.get(name, set()), name


def test_exported_samples_cover_both_sides_of_the_envelope(report):
    shown = json.loads((CHECK / "sample_verdicts.json").read_text())
    assert all(shown[f"in_distribution/{m}_{j}"]["verdict"] == "PASS" for m in ("fno", "meshgraphnet") for j in (0, 1))
    assert all(shown[f"contrast/{m}_{j}"]["verdict"] == "FAIL" for m in ("fno", "meshgraphnet") for j in (0, 1))
    for m in ("fno", "meshgraphnet"):                       # _0 outside the envelope of a, _1 inside
        assert "interface_density" in shown[f"length_scale/{m}_0"]["failed"]
        assert not any(k in RULE and RULE[k] == "PC401" for k in shown[f"length_scale/{m}_1"]["failed"])
    got: dict[str, list[str]] = {}
    for v in report.violations:
        got.setdefault(v.target, set()).add(v.rule)
    assert {k: sorted(v) for k, v in got.items()} == json.loads((CHECK / "expected.json").read_text())
