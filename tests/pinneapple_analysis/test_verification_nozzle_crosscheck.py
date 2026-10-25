"""pinneapple_analysis.verification.nozzle_crosscheck: the ideal-nozzle half is the closed form; the PINN half runs a
short training on the preset's tagged geometry and the disagreement report is consistent with its numbers."""
from __future__ import annotations

import math

import pytest

from pinneapple_analysis.verification.nozzle_crosscheck import scaled_field_head, screen_nozzle
from pinneapple_physics.closed_form.rocket_nozzle import nozzle_performance

INPUTS = dict(throat_radius=0.05, exit_radius=0.12, p_chamber=4e6, T_chamber=2800.0, gamma=1.2, R_gas=345.0)


def test_analytic_only_is_the_closed_form():
    out = screen_nozzle(**INPUTS, run_pinn=False)
    ref = nozzle_performance(gamma=1.2, R_gas=345.0, T_chamber=2800.0, p_chamber=4e6, area_ratio_exit=(0.12 / 0.05) ** 2)
    assert out["analytic"] == ref and "pinn" not in out
    assert out["inputs"]["nozzle_length_m"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        screen_nozzle(**{**INPUTS, "exit_radius": 0.04}, run_pinn=False)


def test_scaled_field_head_rescales_per_field():
    torch = pytest.importorskip("torch")
    head = scaled_field_head(torch.nn.Identity(), [2.0, 10.0], [1.0, -5.0])
    assert head(torch.tensor([[1.0, 1.0]])).tolist() == [[3.0, 5.0]]


def test_pinn_half_runs_and_the_report_matches_its_numbers():
    out = screen_nozzle(**INPUTS, epochs=20, n_collocation=128, n_bc_per_face=50, hidden_dim=16, n_layers=2)
    pinn, d = out["pinn"], out["disagreement"]
    assert all(math.isfinite(pinn[k]) for k in ("mach_exit", "p_exit_pa", "T_exit_k", "final_training_loss"))
    assert d["mach_exit_relative"] == pytest.approx(abs(pinn["mach_exit"] - out["analytic"]["mach_exit"])
                                                    / out["analytic"]["mach_exit"])
    assert d["methods_agree"] == (d["mach_exit_relative"] <= 0.3 and d["p_exit_relative"] <= 0.3)
    assert (out["verdict"] == "agree") == (d["methods_agree"] and pinn["guardrail_trustworthy"])
    assert {c["name"] for c in pinn["guardrail_checks"]} >= {"pde_residual"}
