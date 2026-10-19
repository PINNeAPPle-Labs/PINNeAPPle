"""`pinneapple check` on Darcy results (physics = "darcy"): the reference solver passes, broken fields fail."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pinneapple_check import run
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

N = 48


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    d = tmp_path_factory.mktemp("darcy")
    a = ob.threshold_coefficient(ob.periodic_grf(1, (N, N), alpha=2.0, tau=3.0, seed=4))[0]
    u = ob.darcy_2d(a)
    c = (np.arange(N) + 0.5) / N
    wide = ob.threshold_coefficient(ob.periodic_grf(1, (N, N), alpha=2.0, tau=3.0, seed=4), high=20.0, low=1.0)[0]
    fields = {"solver": (a, u), "scaled": (a, 1.2 * u), "offset": (a, u + 0.1 * u.max()),
              "wrong_coefficient": (wide, u)}
    for k, (ak, uk) in fields.items():
        np.savez(d / f"{k}.npz", x=c, y=c, a=ak, u=uk)
    toml = "".join(f'[[result]]\nname = "{k}"\npath = "{k}.npz"\nphysics = "darcy"\nparameters = {{ f = 1.0 }}\n'
                   'bounds = { a = [3, 12] }\nunits = { a = "1", u = "1" }\n' for k in fields)
    (d / "pinneapple-check.toml").write_text(toml)
    return d


def _rules(report):
    got: dict[str, set[str]] = {}
    for v in report.violations:
        got.setdefault(v.target, set()).add(v.rule)
    return got


def test_darcy_results(cases):
    rep = run(config=cases / "pinneapple-check.toml")
    got = _rules(rep)
    assert "solver" not in got
    assert got["scaled"] == {"PC101", "PC301"}             # 20 % too much flux leaves the domain
    assert got["offset"] >= {"PC201", "PC301"}               # u = 0 violated on the boundary
    assert got["wrong_coefficient"] >= {"PC401", "PC301"}    # a outside its declared range; not the solution
    rules = {o.rule for o in rep.outcomes if o.target == "solver"}
    assert {"PC101", "PC201", "PC301", "PC402"} <= rules


def test_darcy_cli_exit_code_and_no_torch(cases):
    code = ("import sys; from pinneapple_check.cli import check_main; "
            "rc = check_main(['--config', sys.argv[1], 'solver.npz']); "
            "assert 'torch' not in sys.modules; sys.exit(rc)")
    ok = subprocess.run([sys.executable, "-c", code, str(cases / "pinneapple-check.toml")], cwd=cases)
    assert ok.returncode == 0
    bad = subprocess.run([sys.executable, "-m", "pinneapple_check", "--config", str(cases / "pinneapple-check.toml")],
                         cwd=cases, capture_output=True, text=True)
    assert bad.returncode == 1 and "PC301" in bad.stdout


def test_darcy_envelope_in_pinneapple_check(tmp_path):
    from pinneapple_physics import darcy_checks as dc
    train = ob.threshold_coefficient(ob.periodic_grf(200, (N, N), alpha=2.0, tau=3.0, seed=9))
    env = dc.CoefficientEnvelope.fit(train)
    short = ob.threshold_coefficient(ob.periodic_grf(1, (N, N), alpha=2.0, tau=30.0, seed=5))[0]
    inside = train[np.flatnonzero(env.contains(train))[0]]
    toml = ""
    for k, a in {"short": short, "inside": inside}.items():
        dc.save_darcy_npz(tmp_path / f"{k}.npz", a, ob.darcy_2d(a))
        toml += dc.check_config_entry(k, f"{k}.npz", envelope=env)
    (tmp_path / "pinneapple-check.toml").write_text(toml)
    rep = run(config=tmp_path / "pinneapple-check.toml")
    bad = {(v.target, v.subject) for v in rep.violations}
    assert ("short", "a-envelope:interface-density") in bad and not any(t == "inside" for t, _ in bad)
    assert {o.subject for o in rep.outcomes if o.target == "inside" and o.rule == "PC401"} >= {
        "a-envelope:contrast", "a-envelope:interface-density", "a-envelope:high-fraction"}


def test_darcy_envelope_config_errors(tmp_path):
    from pinneapple_check import ConfigError, load_config
    (tmp_path / "pinneapple-check.toml").write_text(
        '[[result]]\npath = "a.npz"\nphysics = "darcy"\nenvelope = { length = [1, 2] }\n')
    with pytest.raises(ConfigError, match="envelope"):
        load_config(tmp_path / "pinneapple-check.toml")
