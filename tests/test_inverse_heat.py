"""Tests for the Inverse Heat Lab app: the fin engine, input checks, the job API and the replication snippet."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "inverse_heat"))
sys.path.insert(0, APP)

from inverse_heat import engine  # noqa: E402


def test_fin_exact_limits():
    import numpy as np
    xi = np.linspace(0, 1, 11)
    th = engine.fin_exact(xi, 25.0, 16.0, 0.005, 0.05)
    assert th[0] == pytest.approx(1.0)
    assert np.all(np.diff(th) < 0)                      # cools along the fin
    assert engine.fin_q_exact(50.0, 16.0, 0.005, 0.05, 55.0) > engine.fin_q_exact(25.0, 16.0, 0.005, 0.05, 55.0)


def test_fin_learns_h_from_synthetic_readings():
    kw = dict(k=16.0, d_mm=5.0, length_mm=50.0, t_base=80.0, t_air=25.0, sensors_mm=[10, 20, 30, 40, 50])
    readings = engine.fin_synthetic_readings(**kw, h_true=25.0, noise=0.0)
    r = engine.run_fin(**kw, readings=readings, h_guess=100.0, steps=1500, h_true=25.0)
    assert r["h_least_squares"] == pytest.approx(25.0, rel=0.01)
    assert r["h_pinn"] == pytest.approx(25.0, rel=0.05)
    assert r["q_pinn_W"] == pytest.approx(r["q_true_W"], rel=0.05)


@pytest.mark.parametrize("bad", [
    dict(sensors_mm=[10], readings=[60.0]),                      # too few
    dict(sensors_mm=[10, 20, 80], readings=[60.0, 50.0, 45.0]),  # beyond the tip
    dict(readings=[90.0, 50.0, 45.0, 40.0, 38.0]),               # above the wall
    dict(readings=[60.0, 50.0, 45.0]),                           # count mismatch
])
def test_fin_rejects_bad_input(bad):
    kw = dict(k=16.0, d_mm=5.0, length_mm=50.0, t_base=80.0, t_air=25.0, sensors_mm=[10, 20, 30, 40, 50])
    kw.update(bad)
    with pytest.raises(engine.InputError):
        engine.run_fin(**kw, steps=0)


def test_api_job_flow_and_precomputed():
    from fastapi.testclient import TestClient
    from inverse_heat.api import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        meta = c.get("/api/meta").json()
        assert set(meta["snippets"]) == {"fin", "plate", "block"}
        assert c.post("/api/fin/run", json={"sensors_mm": [10], "readings": [60]}).status_code == 422
        jid = c.post("/api/fin/run", json={"readings": [65.1, 54.9, 47.7, 43.3, 42.9]}).json()["id"]
        for _ in range(300):
            j = c.get(f"/api/jobs/{jid}").json()
            if j["status"] != "running":
                break
            time.sleep(0.5)
        assert j["status"] == "done", j.get("error")
        assert 20 < j["result"]["h_pinn"] < 30 and j["history"]
        plate = c.get("/api/plate/precomputed").json()
        assert len(plate["field_pinn"]) == len(plate["y_mm"]) and len(plate["field_pinn"][0]) == len(plate["x_mm"])
        block = c.get("/api/block/precomputed").json()
        nz, ny, nx = block["shape"]
        assert len(block["field_pinn"]) == nz * ny * nx == len(block["field_reference"])
        assert c.get("/figures/2d/plate_result_card.png").status_code == 200
        assert c.get("/api/jobs/nope").status_code == 404


def _run_script(tmp_path, case, **overrides):
    import re
    from inverse_heat.snippets import SNIPPETS
    code = SNIPPETS[case]["code"]
    for k, v in overrides.items():                       # the same substitution the page does
        code, n = re.subn(rf"^{k} = [^#\n]*", f"{k} = {v!r}  ", code, flags=re.M)
        assert n == 1, k
    script = tmp_path / SNIPPETS[case]["file"]
    script.write_text(code)
    root = os.path.abspath(os.path.join(APP, "..", ".."))
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=900,
                         env={**os.environ, "PYTHONPATH": root}, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    return run.stdout


def _last_h(out):
    return float(out.strip().splitlines()[-1].split("h = ")[1].split()[0])


def test_fin_script_runs_as_published(tmp_path):
    """The 1D script shown in the app is complete: run it unchanged and check the h it prints."""
    assert _last_h(_run_script(tmp_path, "fin")) == pytest.approx(25.0, rel=0.05)


def test_fin_script_with_your_readings(tmp_path):
    """The page fills T_READ etc. with the user's data; demo readings of an aluminium fin with h = 120."""
    kw = dict(k=200.0, d_mm=10.0, length_mm=100.0, t_base=90.0, t_air=20.0, sensors_mm=[20, 40, 60, 80, 100])
    readings = engine.fin_synthetic_readings(**kw, h_true=120.0, noise=0.0)
    out = _run_script(tmp_path, "fin", K=200.0, D_MM=10.0, L_MM=100.0, T_WALL=90.0, T_AIR=20.0,
                      X_MM=[20, 40, 60, 80, 100], T_READ=readings)
    assert _last_h(out) == pytest.approx(120.0, rel=0.05)


@pytest.mark.parametrize("case,short", [("plate", dict(STEPS=60)), ("block", dict(STEPS=20, LBFGS_ITERS=3))])
def test_2d_3d_scripts_run(tmp_path, case, short):
    """The 2D and 3D scripts run end to end (shortened; the full runs are recorded in the example's results)."""
    out = _run_script(tmp_path, case, **short)
    assert "demo check" in out and _last_h(out) > 0
