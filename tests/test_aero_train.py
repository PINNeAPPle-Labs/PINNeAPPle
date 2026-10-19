"""Library side of the aero optimizer: surrogate training API, Pareto representatives, infeasibility tally, case zip."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_design.aero.case import COEF_SCRIPT, case_zip, forces  # noqa: E402
from pinneapple_design.aero.dataset import steady_values  # noqa: E402
from pinneapple_design.aero.geometry import BOUNDS, naca4, valid  # noqa: E402
from pinneapple_design.aero.mesh import GridSpec, grid_points  # noqa: E402
from pinneapple_design.aero.optimize import Engine, infeasibility_reasons  # noqa: E402
from pinneapple_design.aero.surrogate import FIELDS  # noqa: E402
from pinneapple_design.aero.train import (  # noqa: E402
    coefs,
    load_runs,
    metrics,
    predict_gnn,
    predict_mlp,
    split,
    train_gnn,
    train_mlp,
    train_surrogates,
)
from pinneapple_design.design_optimizer.pareto import representatives  # noqa: E402


# ---------------------------------------------------------------------------------------------- synthetic runs
def _write_run(d, rec, n_cells):
    rng = np.random.default_rng(abs(hash(rec["id"])) % 2**32)
    with open(os.path.join(d, "runs", rec["id"] + ".json"), "w") as f:
        json.dump(rec, f)
    np.savez(os.path.join(d, "runs", rec["id"] + ".npz"), p=rng.normal(0, 0.3, n_cells).astype(np.float32),
             Ux=rng.normal(1, 0.1, n_cells).astype(np.float32), Uy=rng.normal(0, 0.1, n_cells).astype(np.float32),
             nut=rng.uniform(0, 1e-4, n_cells).astype(np.float32))


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("aero_runs"))
    os.makedirs(os.path.join(d, "runs"))
    g = GridSpec()
    n_cells = g.ni * g.nj
    mid = BOUNDS.mean(1)
    rng = np.random.default_rng(0)
    k = 0
    while k < 6:                                                  # 6 LHS shapes x 2 angles, known coefficients
        s = mid + 0.15 * (BOUNDS[:, 1] - BOUNDS[:, 0]) * rng.uniform(-1, 1, 6)
        if not valid(s)[0]:
            continue
        for b, a in enumerate((0.0, 6.0)):
            _write_run(d, {"id": f"s{k:03d}_a{b}", "set": "lhs", "shape_id": f"s{k:03d}", "shape": s.tolist(),
                           "alpha": a, "reynolds": 4e6, "status": "converged", "Cl": 0.1 * a + 0.05 * k,
                           "Cd": 0.008 + 0.0005 * a + 0.0002 * k, "Cm": -0.05 - 0.002 * a}, n_cells)
        k += 1
    ref = naca4("2412")
    for a in (0.0, 4.0):
        _write_run(d, {"id": f"NACA2412_a{int(a):+03d}", "set": "reference", "shape_id": "NACA 2412",
                       "shape": ref.tolist(), "alpha": a, "reynolds": 4e6, "status": "converged",
                       "Cl": 0.11 * a + 0.22, "Cd": 0.0088 + 0.0003 * a, "Cm": -0.047 + 0.001 * a}, n_cells)
    # an oscillating run (averaged), a deep-stall run (dropped) and a failed one (dropped)
    hist = [[100 * i, 1.0 + 0.01 * i, 0.02 + 0.001 * i, -0.04] for i in range(8)]
    base = {"set": "lhs", "shape_id": "s000", "shape": json.load(open(os.path.join(d, "runs", "s000_a0.json")))["shape"],
            "alpha": 12.0, "reynolds": 4e6, "Cl": 9.0, "Cd": 9.0, "Cm": 9.0, "history": hist}
    _write_run(d, {**base, "id": "s000_a2", "status": "unsteady", "Cl_osc": 0.05}, n_cells)
    _write_run(d, {**base, "id": "s000_a3", "status": "unsteady", "Cl_osc": 0.5}, n_cells)
    _write_run(d, {**base, "id": "s000_a4", "status": "failed"}, n_cells)
    return d


def test_load_runs_steady_values(runs_dir):
    recs = load_runs([runs_dir])
    ids = {r["id"] for r in recs}
    assert len(recs) == 6 * 2 + 2 + 1 and "s000_a3" not in ids and "s000_a4" not in ids
    r = next(r for r in recs if r["id"] == "s000_a2")
    H = np.array([[100 * i, 1.0 + 0.01 * i, 0.02 + 0.001 * i, -0.04] for i in range(8)])[-5:]
    assert r["Cl"] == pytest.approx(H[:, 1].mean()) and r["Cd"] == pytest.approx(H[:, 2].mean())
    assert all(os.path.exists(r["npz"]) for r in recs)
    # the same rule as verification, which must not modify its input
    raw = [{"status": "unsteady", "Cl_osc": 0.05, "history": H.tolist(), "Cl": 9.0}]
    assert steady_values(raw)[0]["Cl"] == pytest.approx(H[:, 1].mean()) and raw[0]["Cl"] == 9.0


def test_split_holds_out_whole_shapes(runs_dir):
    recs = load_runs([runs_dir])
    tr, te, ref = split(recs, frac=0.3, seed=1)
    tr_s, te_s = {r["shape_id"] for r in tr}, {r["shape_id"] for r in te}
    assert te_s and not tr_s & te_s and len(te_s) == round(0.3 * 6)
    for s in te_s:                                                # every angle of a held-out shape is held out
        assert len([r for r in te if r["shape_id"] == s]) == len([r for r in recs if r["shape_id"] == s])
    assert {r["shape_id"] for r in ref} == {"NACA 2412"} and all(r["set"] != "reference" for r in tr + te)
    assert len(tr) + len(te) + len(ref) == len(recs)


def test_metrics_match_hand_computation():
    t = np.array([[0.1, 0.010, -0.05], [0.5, 0.012, -0.04], [0.9, 0.020, -0.06], [1.2, 0.030, -0.05]])
    p = np.array([[0.12, 0.011, -0.05], [0.45, 0.012, -0.03], [0.95, 0.018, -0.06], [1.1, 0.033, -0.07]])
    m = metrics(t, p)
    e = p[:, 0] - t[:, 0]
    assert m["Cl"]["mae"] == pytest.approx(np.mean(np.abs(e)))
    assert m["Cl"]["max"] == pytest.approx(0.1)
    assert m["Cl"]["r2"] == pytest.approx(1 - np.sum(e ** 2) / np.sum((t[:, 0] - t[:, 0].mean()) ** 2))
    rel = np.abs(p[:, 1] - t[:, 1]) / t[:, 1]                     # 10 %, 0 %, 10 %, 10 %
    assert m["Cd"]["mape"] == pytest.approx(7.5)
    assert m["Cd"]["p90"] == pytest.approx(100 * np.percentile(rel, 90))
    assert "mape" not in m["Cl"]
    assert np.allclose(coefs([{"Cl": 1, "Cd": 2, "Cm": 3}]), [[1, 2, 3]])


def test_short_training_runs_and_predicts(runs_dir):
    recs = load_runs([runs_dir])
    tr, te, ref = split(recs)
    mlp, norm = train_mlp(tr, k=3, width=16, epochs=2)
    mean, sd = predict_mlp(mlp, norm, te + ref)
    assert mean.shape == sd.shape == (len(te + ref), 3) and np.isfinite(mean).all() and (sd >= 0).all()
    assert (mean[:, 1] > 0).all()                                 # Cd from exp(log Cd)
    logs = []
    gnn, gnorm, hist, lay = train_gnn(tr[:2], te[:1], minutes=None, epochs=2, batch=2, hidden=8, mp=1,
                                      log=logs.append)             # two graphs: one step per epoch keeps it fast
    assert len(hist) == 1 and hist[0]["epoch"] == 2 and logs
    c, ferr = predict_gnn(gnn, gnorm, lay, ref[:1])
    assert c.shape == (1, 3) and np.isfinite(c).all() and set(ferr) == set(FIELDS)


class _Interrupted(Exception):
    pass


def test_gnn_training_resumes_from_a_checkpoint_to_the_same_weights(runs_dir, tmp_path, monkeypatch):
    import pinneapple_design.aero.train as aero_train
    tr, te, _ = split(load_runs([runs_dir]))
    kw = dict(minutes=None, epochs=4, batch=1, hidden=8, mp=1, log=lambda s: None)
    full, *_ = train_gnn(tr[:3], te[:1], **kw)
    d = str(tmp_path / "ckpt")
    real_replace = os.replace

    def replace_then_die(src, dst):                                # the run is killed right after its first checkpoint
        real_replace(src, dst)
        raise _Interrupted
    monkeypatch.setattr(aero_train.os, "replace", replace_then_die)
    with pytest.raises(_Interrupted):
        train_gnn(tr[:3], te[:1], checkpoint_dir=d, checkpoint_every=2, **kw)
    monkeypatch.setattr(aero_train.os, "replace", real_replace)
    assert torch.load(os.path.join(d, "gnn_last.pt"), weights_only=False)["epoch"] == 2
    logs = []
    resumed, *_ = train_gnn(tr[:3], te[:1], checkpoint_dir=d, checkpoint_every=2, **{**kw, "log": logs.append})
    assert any("resumed at epoch 2" in m for m in logs)
    for a, b in zip(full.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError, match="other training settings"):
        train_gnn(tr[:3], te[:1], checkpoint_dir=d, **{**kw, "epochs": 6})


def test_train_surrogates_bundle_loads_in_the_engine(runs_dir):
    recs = load_runs([runs_dir])
    bundle, res = train_surrogates(recs, gnn_minutes=0, log=lambda s: None, mlp_k=2, mlp_width=16, mlp_epochs=2)
    assert res["counts"]["test_shapes"] == 1 and "mlp_test" in res and "mlp_reference_points" in res
    assert bundle["gnn_state"] is None
    eng = Engine(bundle)
    co = eng.mlp_coefficients(naca4("2412")[None], np.array([0.0, 4.0]))
    assert co["cl"].shape == (1, 2)


# ---------------------------------------------------------------------------------------------- Pareto picks
def test_representatives_hand_computed_front():
    # speed (max) vs stall speed (min); scaled to [0, 1] (1 = best): speed [0, .4, .8, 1], stall [1, .75, .4, 0]
    # squared distance to (1, 1): [1, .4225, .40, 1] -> the third member is the compromise
    F = [[100, 50], [120, 55], [140, 62], [150, 70]]
    picks = representatives(F, ("max", "min"), ("fastest", "safest"), ids=[10, 11, 12, 13])
    assert picks == {"fastest": 13, "safest": 10, "balanced": 12}
    assert list(picks) == ["fastest", "safest", "balanced"]
    assert representatives(F, ("max", "min"), ("fastest", None), weights=(1, 0)) == {"fastest": 3, "balanced": 3}
    assert representatives(F, ("max", "min"), ("fastest", "safest"), balanced=None) == {"fastest": 3, "safest": 0}
    assert representatives([], ("max", "min"), ("a", "b")) == {}
    with pytest.raises(ValueError):
        representatives(F, ("up", "min"), ("a", "b"))


def test_infeasibility_reasons():
    D = [{"feasible": False, "violations": ["stall speed", "thickness"]},
         {"feasible": False, "violations": ["stall speed"]},
         {"feasible": False, "violations": []},
         {"feasible": True, "trust": "low"},
         {"feasible": True, "trust": "high"}]
    r = infeasibility_reasons(D)
    assert r == {"stall speed": 2, "thickness": 1, "no solution": 1, "low surrogate trust": 1}
    assert list(r) == ["stall speed", "thickness", "no solution", "low surrogate trust"]
    assert infeasibility_reasons([]) == {}


# ---------------------------------------------------------------------------------------------- case zip
def _foam_list(kind, a):
    rows = "\n".join(f"({' '.join(f'{v:.12g}' for v in r)})" if kind == "vector" else f"{r:.12g}" for r in a)
    return (f"FoamFile\n{{\n    version 2.0;\n}}\ndimensions [0 2 -2 0 0 0 0];\n"
            f"internalField nonuniform List<{kind}> {len(a)}\n(\n{rows}\n)\n;\n")


def _run_script(tmp_path, shape, alpha, p_wall, U_wall):
    data = case_zip(shape, alpha)
    z = zipfile.ZipFile(io.BytesIO(data))
    z.extractall(tmp_path)
    case = os.path.join(tmp_path, f"airfoil_a{alpha:g}")
    g = GridSpec()
    n = g.ni * g.nj
    p = np.zeros(n)
    p[:g.ni] = p_wall
    U = np.zeros((n, 3))
    U[:g.ni, :2] = U_wall
    os.makedirs(os.path.join(case, "100"))
    with open(os.path.join(case, "100", "p"), "w") as f:
        f.write(_foam_list("scalar", p))
    with open(os.path.join(case, "100", "U"), "w") as f:
        f.write(_foam_list("vector", U))
    out = subprocess.run([sys.executable, "-I", "coefficients.py"], cwd=case, capture_output=True, text=True,
                         timeout=120, check=True).stdout
    m = re.search(r"Cl = (\S+)\s+Cd = (\S+)\s+Cm\(c/4, nose-up \+\) = (\S+)", out)
    return z, tuple(float(v) for v in m.groups())


def test_case_zip_contents_and_coefficient_script(tmp_path):
    shape = naca4("2412")
    alpha = 2.0
    P2 = grid_points(shape, GridSpec())
    w = P2[0]
    mid = 0.5 * (w + np.roll(w, -1, 0))
    # p = y (wall face midpoints), no flow: the net force is -grad(p) * area = (0, -A) in body axes (divergence
    # theorem; the midpoint rule is exact for a linear p on the polygon), so Cl = -A cos(a) / q, Cd = -A sin(a) / q
    z, (cl, cd, cm) = _run_script(tmp_path / "a", shape, alpha, mid[:, 1], np.zeros((len(w), 2)))
    names = set(z.namelist())
    for f in ("constant/polyMesh/points", "constant/polyMesh/faces", "constant/polyMesh/owner",
              "constant/polyMesh/neighbour", "constant/polyMesh/boundary", "0/U", "0/p", "system/controlDict",
              "grid.json", "coefficients.py", "Allrun"):
        assert f"airfoil_a2/{f}" in names, f
    assert z.read("airfoil_a2/coefficients.py").decode() == COEF_SCRIPT
    info = json.loads(z.read("airfoil_a2/grid.json"))
    assert info["ni"] == GridSpec().ni and info["alpha"] == alpha and info["nu"] == pytest.approx(1 / 4e6)
    assert (z.getinfo("airfoil_a2/Allrun").external_attr >> 16) & 0o111
    area = 0.5 * abs(np.sum(w[:, 0] * np.roll(w[:, 1], -1) - np.roll(w[:, 0], -1) * w[:, 1]))   # shoelace
    a = np.radians(alpha)
    # NACA 4-digit thickness y_t = 5 t (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2 + 0.2843 x^3 - 0.1015 x^4) encloses
    # 2 * 5 t (0.2969 * 2/3 - 0.1260/2 - 0.3516/3 + 0.2843/4 - 0.1015/5) = 0.6851 t (camber does not change it)
    assert area == pytest.approx(10 * 0.12 * (0.2969 * 2 / 3 - 0.126 / 2 - 0.3516 / 3 + 0.2843 / 4 - 0.1015 / 5), rel=0.02)
    assert cl == pytest.approx(-area * np.cos(a) / 0.5, abs=2e-4)
    assert cd == pytest.approx(-area * np.sin(a) / 0.5, abs=2e-5)
    # with shear: the script agrees with the library's own wall integration
    rng = np.random.default_rng(1)
    p_w, U_w = rng.normal(0, 0.5, len(w)), rng.normal(0, 1e-4, (len(w), 2))
    _, (cl, cd, cm) = _run_script(tmp_path / "b", shape, alpha, p_w, U_w)
    ref = forces(P2, p_w, U_w, 1 / 4e6, alpha)
    assert (cl, cd, cm) == pytest.approx((ref["Cl"], ref["Cd"], ref["Cm"]), abs=2e-4)
