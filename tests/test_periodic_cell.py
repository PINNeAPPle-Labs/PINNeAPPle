"""Periodic unit cell: TCDA cell checks, the (frequency x scan) dataset, the FNO with held-out error and the
equal-budget comparison (fake solver), and an EMerge Floquet run (PINNEAPPLE_SLOW=1)."""
import os

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.antenna.runner import ResultTable
from pinneapple_simulation.electromagnetics.periodic import benchmark as bm
from pinneapple_simulation.electromagnetics.periodic.tcda import InvalidCell, TCDACell

FREQS = [3.0, 4.0, 5.0, 6.0]
SCANS = [0.0, 30.0, 45.0]
NAMES = list(bm.CELL_RANGES)


def _fake_map(x):
    """A smooth active-reflection map (dB) of the cell parameters, standing in for the solver."""
    h, w, g, ts, er = x
    f = np.array(FREQS)[:, None]
    th = np.deg2rad(np.array(SCANS))[None, :]
    return -3 - 10 * np.exp(-((f - 3 - 0.2 * h) ** 2)) * (1 - 0.3 * np.sin(th) ** 2) * (1 + 0.05 * w - 0.1 * g) \
        + 0.2 * ts - 0.1 * er


def test_cell_checks_grating_lobe_and_bounds():
    TCDACell().check(6.0, 45.0)                        # 22 mm < 50 / (1 + sin 45) = 29.3 mm
    with pytest.raises(InvalidCell):
        TCDACell(a=31.0).check(6.0, 45.0)
    with pytest.raises(InvalidCell):
        TCDACell(g=8.0).check()
    with pytest.raises(ValueError):
        TCDACell.from_dict({"nope": 1})


def test_to_arrays_keeps_only_complete_cells():
    rows = []
    for c, x in enumerate([(8, 2, 1, 2, 3), (10, 3, 1.5, 1, 2)]):
        Y = _fake_map(x)
        for i, f in enumerate(FREQS):
            for j, t in enumerate(SCANS):
                if c == 1 and i == 3 and j == 2:
                    continue                            # one point missing: the cell is left out
                rows.append({"tag": f"c{c}", **dict(zip(NAMES, x, strict=True)), "f_GHz": f, "theta_deg": t,
                             "S11_dB": Y[i, j]})
    X, Y, tags = bm.to_arrays(ResultTable(rows), NAMES, FREQS, SCANS)
    assert tags == ["c0"] and X.shape == (1, 5) and Y.shape == (1, 4, 3)
    assert np.allclose(Y[0], _fake_map((8, 2, 1, 2, 3)))
    assert bm.worst_case(Y)[0] == pytest.approx(_fake_map((8, 2, 1, 2, 3)).max())


def _sample(n, seed):
    from pinneapple_data.parameter_sampling import sample_parameters
    X = np.array([[r[k] for k in NAMES] for r in sample_parameters(bm.CELL_RANGES, n, method="lhs", seed=seed)])
    return X, np.array([_fake_map(x) for x in X])


def test_fno_learns_the_map_with_heldout_error():
    pytest.importorskip("torch")
    X, Y = _sample(48, 0)
    pred, info = bm.fit_fno(X[:40], Y[:40], wall_s=10.0, seed=0)
    h = bm.heldout(pred, X[40:], Y[40:])
    assert pred(X[:2]).shape == (2, 4, 3) and info["n_train"] + info["n_val"] == 40
    spread = float(np.std(Y))
    assert h["rmse_dB"] < 0.5 * spread                  # clearly better than predicting a constant


def test_equal_budget_counts_solver_calls(tmp_path):
    pytest.importorskip("torch")
    calls = []

    def solve(designs, label):
        calls.append((label, len(designs)))
        X = np.array([[d[k] for k in NAMES] for d in designs])
        return X, np.array([_fake_map(x) for x in X])

    out = bm.equal_budget(solve, bm.CELL_RANGES, 12, n_verify=3, seed=0, fno_wall_s=5.0, n_candidates=500,
                          log=lambda s: None)
    by = {}
    for lab, n in calls:
        by[lab] = by.get(lab, 0) + n
    assert by == {"doe": 9, "verify": 3, "bo": 12}
    assert out["fno_optimizer"]["calls"] == 12 and out["bayesian"]["calls"] == 12
    # every best value is a solver value
    allv = [bm.worst_case(_fake_map(np.array([d[k] for k in NAMES]))[None])[0] for d in
            [out["fno_optimizer"]["best_design"], out["bayesian"]["best_design"]]]
    assert out["fno_optimizer"]["best_dB"] == pytest.approx(allv[0])
    assert out["bayesian"]["best_dB"] == pytest.approx(allv[1])
    assert len(out["bayesian"]["trace"]) == 12 and np.all(np.diff(out["bayesian"]["trace"]) <= 0)


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="EMerge Floquet solve (set PINNEAPPLE_SLOW=1)")
def test_emerge_cell_scan_reuses_the_mesh(tmp_path):
    pytest.importorskip("emerge")
    from pinneapple_simulation.electromagnetics.periodic.tcda import simulate
    rows = simulate(TCDACell(), [4.0, 6.0], [(0.0, 0.0), (30.0, 0.0)])
    assert len(rows) == 4 and len({r["n_tets"] for r in rows}) == 1
    s = np.array([r["S11_dB"] for r in rows])
    assert np.all(s < 0) and np.all(s > -60)
    single = simulate(TCDACell(), [4.0, 6.0], [(30.0, 0.0)])
    assert np.allclose([r["S11_dB"] for r in single], s[2:], atol=1e-6)    # same as solving that angle alone
