"""Rotor design with FEA, a GP surrogate and NSGA-II (``machines.design`` on the design-study core of
``pinneapple_design.design_optimizer``): geometry validity, the FEA evaluator, and the hybrid-vs-FEA-only study."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machines import design as D

pytest.importorskip("sklearn")


def test_evaluator_marks_invalid_designs_without_a_solve():
    ev = D.rotor_evaluator(n_gap=192, n_positions=3)
    X = np.array([[0.0189, 0.0065, 0.01096, 0.042, 0.0015], [0.026, 0.0065, 0.01096, 0.042, 0.0015]])
    out = ev(X)
    assert out["_valid"].tolist() == [True, False] and ev.n_fea == 1
    assert out["magnet_mass"][0] == pytest.approx(2 * 8 * 0.0189 * 0.0065 * 0.08382 * 7500.0)
    assert 250.0 < out["torque"][0] < 400.0 and np.isnan(out["torque"][1])


def test_hybrid_study_verifies_every_front_design_with_fea():
    P = D.rotor_design_problem(D.rotor_evaluator(n_gap=192, n_positions=3))
    r = D.hybrid_vs_fea_only(P, n_initial=8, rounds=1, n_verify=3, ga_budget=12, surrogate_budget=200)
    assert r["hybrid"]["fea_calls"] <= 11 and r["fea_only"]["fea_calls"] <= 12
    assert r["hybrid"]["front"] and all(np.isfinite(d["torque"]) for d in r["hybrid"]["front"])  # FEA values
    assert 0.0 < r["hybrid"]["hypervolume"] <= 1.21 + 1e-12
    curve = r["fea_only"]["hypervolume_curve"]
    assert all(b >= a - 1e-12 for a, b in zip(curve, curve[1:], strict=False))  # never decreases
