"""Potential-based surrogates: input convexity, consistency by construction, training on mixed data, study plumbing."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_physics.potential import consistency_check, tree
from pinneapple_physics.potential.surrogate import (
    PICNN,
    DirectSurrogate,
    MixedData,
    PotentialSurrogate,
    Scaling,
    fit,
)
from pinneapple_physics.potential.thermo import helmholtz_T_rho

R_S = 188.92  # J/(kg K), CO2


def _vdw_specific():
    a, b, cv = 188.0, 9.7e-4, 650.0  # van der Waals CO2 constants per kg (a / M^2, b / M), illustrative c_v

    def fn(T, rho):
        return cv * T * (1 - torch.log(T / 300.0)) + R_S * T * torch.log(rho) - R_S * T * torch.log(1 - b * rho) - a * rho

    return helmholtz_T_rho(fn)


def _mixed(n=40, seed=0):
    rng = np.random.default_rng(seed)
    A = _vdw_specific()
    T, R, P, Y = [], [], [], []
    for p in ("p", "s", "u", "cv", "cp", "w"):
        t, r = rng.uniform(320, 380, n), rng.uniform(80, 400, n)
        q = tree(A, 2, at={"T": t, "rho": r}, symbolic=False)
        T.append(t)
        R.append(r)
        P += [p] * n
        Y.append(q[p])
    return MixedData(np.concatenate(T), np.concatenate(R), np.array(P, dtype=object), np.concatenate(Y))


def test_picnn_is_convex_in_y():
    torch.manual_seed(0)
    net = PICNN(1, 1, hidden=16, layers=3)
    x = torch.randn(64, 1, dtype=torch.float64)
    y = torch.linspace(-3, 3, 64, dtype=torch.float64)[:, None].requires_grad_(True)
    f = net(x, y)
    (g,) = torch.autograd.grad(f.sum(), y, create_graph=True)
    (h,) = torch.autograd.grad(g.sum(), y)
    assert (h >= -1e-12).all()


def test_potential_surrogate_is_consistent_and_convex_option_is_stable():
    data = _mixed()
    sc = Scaling.from_data(data)
    T, rho = np.linspace(320, 380, 9), np.linspace(80, 400, 9)
    for convex in (False, True):
        torch.manual_seed(1)
        m = PotentialSurrogate(sc, hidden=16, layers=2, convex=convex, init_scale=1.0)  # untrained, rough
        rep = consistency_check(lambda pt, m=m: m.predict(pt["T"], pt["rho"]), "helmholtz_T_rho",
                                at={"T": T, "rho": rho}, tol=1e-4)
        # where an (untrained) potential is unstable (dp/drho < 0, or c_v < 0, which input convexity in v does not
        # exclude), w^2 < 0 and w is floored: only that identity can fail
        assert all(r["ok"] for r in rep.rows if r["name"] != "speed of sound"), str(rep)
    q = m.predict(*np.meshgrid(np.linspace(300, 420, 13), np.linspace(50, 900, 13)))
    h = 1e-3
    TT, RR = np.meshgrid(np.linspace(300, 420, 13), np.linspace(50, 900, 13))
    dp = (m.predict(TT, RR * (1 + h))["p"] - m.predict(TT, RR * (1 - h))["p"]) / (2 * h * RR.reshape(-1))
    assert (dp > 0).all() and np.isfinite(q["cp"]).all()  # convex in v: mechanically stable everywhere
    torch.manual_seed(1)
    d = DirectSurrogate(sc, hidden=16, layers=2)
    rep = consistency_check(lambda pt: d.predict(pt["T"], pt["rho"]), "helmholtz_T_rho", at={"T": T, "rho": rho})
    assert not rep.ok  # separate outputs: nothing ties them together


def test_both_surrogates_learn_mixed_data():
    data = _mixed()
    sc = Scaling.from_data(data)
    for cls in (DirectSurrogate, PotentialSurrogate):
        torch.manual_seed(0)
        m = cls(sc, hidden=16, layers=2)
        h = fit(m, data, epochs=300, lr=3e-3)
        assert h["loss"][-1] < 0.2 * h["loss"][0], (cls.__name__, h["loss"][0], h["loss"][-1])
        assert h["parameters"] > 0 and h["steps"] == 300


def test_scaling_log_transform_is_continued_below_the_floor():
    sc = Scaling(300.0, 10.0, 400.0, 100.0, floor={"cp": 100.0})
    y = torch.tensor([-50.0, 50.0, 100.0, 1000.0], dtype=torch.float64, requires_grad=True)
    g = sc.g("cp", y)
    (grad,) = torch.autograd.grad(g.sum(), y)
    assert torch.isfinite(g).all() and (grad > 0).all()
    assert float(g[3]) == pytest.approx(np.log(1000.0))


def test_co2_study_runs_end_to_end_on_a_tiny_budget():
    pytest.importorskip("CoolProp")
    from pinneapple_physics.potential import co2_study as cs

    grids = cs.default_grids(5)
    res = cs.run_study(range(2), grids=grids, n_per_prop=10, epochs=5, hidden=8, layers=2)
    s = res["summary"]
    assert set(s["models"]) == set(cs.MODELS)
    iv = s["models"]["direct"]["interp"]["nrmse"]["p"]
    assert iv["lo"] <= iv["mean"] <= iv["hi"]
    assert "Maxwell relation (Helmholtz)" in s["models"]["potential"]["hotter"]["identities"]
    assert set(s["paired"]) == {"potential", "potential_icnn"}
    assert len(res["runs"][0]["models"]["potential"]["isotherms"]["310"]["cp"]) == cs.ISO_RHO.size
