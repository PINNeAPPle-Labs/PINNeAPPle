"""pinneapple_simulation.terramechanics.mobility: the closed-form Mohr-Coulomb limit against the compiled R2 residual
of ``bekker_wong_surrogate_2d`` (the source app's verify_analytical_crosscheck.py, checks 1 and 2), the per-constraint
analysis on hand-built surrogates whose answers are known, and a short real training."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pinneapple_physics as pp  # noqa: E402
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem  # noqa: E402
from pinneapple_simulation.terramechanics.mobility import (  # noqa: E402
    analyze_wheel_surrogate,
    contact_entry_angle,
    mobility_analysis,
    mohr_coulomb_traction_limit,
)

C, PHI, R, B = 1400.0, 30.0, 0.125, 0.060


def test_entry_angle_round_trips_the_sinkage():
    th = contact_entry_angle(R, 0.03)
    assert R * (1 - math.cos(th)) == pytest.approx(0.03, abs=1e-12)
    with pytest.raises(ValueError):
        contact_entry_angle(R, R)


class _Const(torch.nn.Module):
    """Returns fixed (Fx, Fz, My) for every input, tied to the input graph so autograd terms exist."""

    def __init__(self, fx, fz, my=1e6):
        super().__init__()
        self.v = (float(fx), float(fz), float(my))
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = 0.0 * x[:, :1]
        return torch.cat([self.v[0] + z, self.v[1] + z, self.v[2] + z], dim=1)


def _compiled_pde_residual(slip, sinkage, fx, fz):
    spec = pp.get_preset("bekker_wong_surrogate_2d", c_Pa=C, phi_deg=PHI, R_m=R, b_m=B)
    loss_fn = compile_problem(spec)
    x = torch.tensor([[slip, sinkage]], dtype=torch.float32, requires_grad=True)
    nf = len(spec.fields)
    batch = {"x_col": x, "ctx": {}, "x_bc": torch.zeros((0, 2)), "y_bc": torch.zeros((0, nf)),
             "x_ic": torch.zeros((0, 2)), "y_ic": torch.zeros((0, nf)), "x_data": torch.zeros((0, 2)),
             "y_data": torch.zeros((0, nf))}
    out = loss_fn(_Const(fx, fz), None, batch)
    return float((out["pde"] if "pde" in out else out["total"]).item())


@pytest.mark.parametrize("slip,sinkage,fz", [(0.30, 0.030, 50.0), (0.10, 0.010, 20.0), (0.60, 0.050, 80.0)])
def test_closed_form_limit_agrees_with_the_compiled_r2_residual(slip, sinkage, fz):
    limit = mohr_coulomb_traction_limit(C, PHI, R, B, sinkage, fz)
    assert _compiled_pde_residual(slip, sinkage, limit - 1e-6, fz) < 1e-6
    # 10 N over the limit: relu(10)^2 averaged with the three satisfied constraints R3-R5 = 100 / 4
    assert _compiled_pde_residual(slip, sinkage, limit + 10.0, fz) == pytest.approx(25.0, abs=1e-3)


class _Analytic(torch.nn.Module):
    """Fz = k z, Fx = frac * Mohr-Coulomb limit * (1 - exp(-s / s0)), My = R Fx * torque_factor."""

    def __init__(self, k=2.0e3, frac=0.8, torque_factor=1.1):
        super().__init__()
        self.k, self.frac, self.tf = k, frac, torque_factor
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        s, z = x[:, :1], x[:, 1:2]
        fz = self.k * z
        theta = torch.arccos(torch.clamp(1 - z / R, -1.0, 1.0))
        limit = C * B * R * theta + fz * math.tan(math.radians(PHI))
        fx = self.frac * limit * (1 - torch.exp(-s / 0.15))
        return torch.cat([fx, fz, self.tf * R * fx], dim=1)


def _spec():
    return pp.get_preset("bekker_wong_surrogate_2d", c_Pa=C, phi_deg=PHI, R_m=R, b_m=B)


def test_analysis_of_a_consistent_surrogate():
    spec = _spec()
    out = analyze_wheel_surrogate(_Analytic(), spec, c_Pa=C, phi_deg=PHI, R_m=R, b_m=B, Fz_N=50.0, n_slip=11)
    assert out["constraints_consistent"] and out["sinkage_solve_reliable"]
    z_lo, z_hi = spec.domain_bounds["sinkage"]
    target = 50.0 / 2.0e3
    if z_lo <= target <= z_hi:
        assert all(abs(r["sinkage_m"] - target) < (z_hi - z_lo) / 399 + 1e-12 for r in out["per_slip"])
    assert out["max_drawbar_pull_at_slip"] == pytest.approx(out["per_slip"][-1]["slip"])


def test_analysis_flags_each_violated_constraint():
    spec = _spec()
    over = analyze_wheel_surrogate(_Analytic(frac=1.5, torque_factor=0.5), spec, c_Pa=C, phi_deg=PHI, R_m=R, b_m=B,
                                   Fz_N=50.0, n_slip=11)
    checks = over["constraint_checks"]
    assert not over["constraints_consistent"]
    assert checks["r2_mohr_coulomb"]["violations"] > 0 and checks["r4_torque_covers_thrust"]["violations"] > 0
    assert checks["r1_zero_pull_at_zero_slip"]["passed"]          # Fx(0) = 0 still holds
    with pytest.raises(ValueError):
        analyze_wheel_surrogate(_Analytic(), spec, c_Pa=C, phi_deg=PHI, R_m=R, b_m=B, Fz_N=-1.0)


def test_short_training_runs_end_to_end():
    out = mobility_analysis(epochs=30, n_collocation=128, hidden_dim=16, n_layers=2, n_slip=6)
    assert len(out["per_slip"]) == 6 and out["guardrail"]["checks"]
    assert out["verdict"].startswith(("physically consistent", "not fully consistent"))
    assert np.isfinite(out["training"]["final_training_loss"])
