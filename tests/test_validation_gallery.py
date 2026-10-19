"""The validation gallery (#705): every example in ``examples/validation`` runs at smoke size, ends in a validation
report, a trust card and the files ``pinneapple check`` reads, and catches the failure it documents; together they
cover every model family and every physics of the issue. Plus the reaction-diffusion reference solver the gallery
added, against its closed forms."""
from __future__ import annotations

import importlib.util
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from pinneapple_analysis.validation.gallery import FAMILIES, PHYSICS
from pinneapple_analysis.validation.report_artifact import check_report

GALLERY = Path(__file__).resolve().parents[1] / "examples" / "validation"
EXAMPLES = sorted(p.stem for p in GALLERY.glob("[0-9][0-9]_*.py"))
NEEDS = {"11_physicsnemo_fno_magnetostatics": "physicsnemo", "02_fno_darcy_onnx": "onnxruntime"}


def load(name):
    spec = importlib.util.spec_from_file_location(f"gallery_{name}", GALLERY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_gallery_covers_every_family_and_physics():
    assert len(EXAMPLES) >= 12
    mods = [load(n) for n in EXAMPLES]
    tags = lambda v: {v} if isinstance(v, str) else set(v)            # noqa: E731
    assert set().union(*(tags(m.FAMILY) for m in mods)) >= set(FAMILIES)
    assert set().union(*(tags(m.PHYSICS) for m in mods)) >= set(PHYSICS)
    for m in mods:
        assert callable(m.run) and m.CATCHES and m.__doc__ and "Failure caught" in m.__doc__


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs_and_catches_its_failure(name, tmp_path):
    pytest.importorskip("torch")
    if name in NEEDS:
        pytest.importorskip(NEEDS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = load(name).run(smoke=True, out_dir=tmp_path)
    assert any(c == res.expected or c.startswith(res.expected + ".") for c in res.caught), res.summary()
    assert res.good.decision in ("pass", "warn", "fail") and res.bad.decision != "pass"
    assert check_report(res.good) == [] and check_report(res.bad) == []
    card = json.loads(Path(res.files["trust_card"]).read_text())
    assert card["decisao"] in ("APPROVED", "REVIEW", "REJECT")
    sarif = json.loads(Path(res.files["sarif"]).read_text())
    assert sarif["version"].startswith("2.1")
    good_fails = {c.id for c in res.good.checks if c.status == "fail"}
    assert not any(c == res.expected or c.startswith(res.expected + ".") for c in good_fails)


def test_pinneapple_check_on_the_gallery_config():
    pytest.importorskip("torch")
    from pinneapple_check import run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = run([GALLERY])
    assert rep.exit_code("error") == 0, [(o.rule, o.target, o.message) for o in rep.violations]
    rules = {(o.target, o.rule) for o in rep.outcomes}
    assert ("cantilever-first-frequency", "PC303") in rules and ("heatsink-zoo", "PC601") in rules


def test_allen_cahn_interface_and_fisher_front_speed():
    from pinneapple_simulation.numerical_solvers.reaction_diffusion import ReactionDiffusion1D
    D = 1e-3
    ac = ReactionDiffusion1D("allen_cahn", D, n=256, length=2.0, dt=0.02)
    x = ac.x
    w = math.sqrt(2 * D)
    u0 = np.tanh((x - 0.5) / w) * np.tanh((1.5 - x) / w) * -1           # two interfaces, far apart
    u = ac.step(u0, 100)                                                # the tanh profile is stationary
    assert np.abs(u - u0).max() < 1e-3
    fk = ReactionDiffusion1D("fisher", D, n=512, length=4.0, dt=0.05)
    u = np.where(fk.x < 0.3, 1.0, 0.0) + np.where(fk.x > 3.7, 1.0, 0.0)   # invades the unstable state u = 0
    pos = []
    for _ in range(3):
        u = fk.step(u, 100)                                              # 5 time units
        pos.append(fk.x[np.argmax(u[: fk.n // 2] < 0.5)])
    speed = (pos[-1] - pos[0]) / 10.0                                   # between t = 5 and t = 15
    # a front from a step approaches the minimal KPP speed 2 sqrt(D) slowly (Bramson):
    # x(t) = 2 sqrt(D) t - (3/2) sqrt(D) ln t, i.e. 2 sqrt(D) (1 - 3 / (4 t)); at t = 10 that is 7.5 % below
    assert abs(speed / (2 * math.sqrt(D) * (1 - 3 / 40)) - 1) < 0.06
