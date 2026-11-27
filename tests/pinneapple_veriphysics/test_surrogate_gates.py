"""The model-side gates: does a surrogate keep the conservation, the boundary and the equation it claims?"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.schema import AxisSpec, BoundaryCondition, Field, Physics, PhysicsDataBundle
from pinneapple_veriphysics.ledger import EvidenceLedger
from pinneapple_veriphysics.surrogate_gates import (
    RESIDUALS,
    advection_residual,
    burgers_residual,
    relative_l2,
    run_surrogate_gates,
)

NT, NX = 120, 240


def wave_field(speed: float = 0.4, amplitude: float = 0.3) -> np.ndarray:
    x = np.linspace(0.0, 1.0, NX)
    t = np.linspace(0.0, 1.0, NT)
    return 1.0 + amplitude * np.sin(2 * np.pi * (x[None, :] - speed * t[:, None]))


def bundle_for(u: np.ndarray, *, bcs=()) -> PhysicsDataBundle:
    x = np.linspace(0.0, 1.0, u.shape[1])
    t = np.linspace(0.0, 1.0, u.shape[0])
    return PhysicsDataBundle.from_arrays(
        {"field:rho": u, "axis:t": t, "axis:x": x},
        fields=[Field.from_array("rho", u, axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        physics=Physics(governing="rho_t + c rho_x = 0", boundary_conditions=list(bcs)),
        source="synthetic wave", license="CC0-1.0", kind="synthetic")


# ── accuracy ────────────────────────────────────────────────────────────────────────────────────────────────
def test_accuracy_is_the_documented_metric():
    exact = wave_field()
    good = run_surrogate_gates(bundle_for(exact), reference=bundle_for(exact), field="rho", gates=("accuracy",))
    assert good.get("accuracy").value == 0.0 and good.get("accuracy").passed

    off = exact * 1.10
    bad = run_surrogate_gates(bundle_for(off), reference=bundle_for(exact), field="rho", gates=("accuracy",))
    assert bad.get("accuracy").passed is False and abs(bad.get("accuracy").value - 0.10) < 2e-2
    # the tolerance, not the metric, decides: 1 % is a pass at 5e-2 and a failure at 5e-3
    close = run_surrogate_gates(bundle_for(exact * 1.01), reference=bundle_for(exact), field="rho",
                                gates=("accuracy",), tolerance=5e-3)
    assert close.get("accuracy").passed is False


def test_a_zero_reference_is_an_error_not_a_perfect_score():
    with pytest.raises(ValueError, match="identically zero"):
        relative_l2(np.zeros(3), np.zeros(3))


def test_no_reference_is_not_run():
    report = run_surrogate_gates(bundle_for(wave_field()), field="rho", gates=("accuracy",))
    assert report.get("accuracy").status == "not run"


# ── conservation ────────────────────────────────────────────────────────────────────────────────────────────
def test_conservation_reports_the_model_and_the_data_side_by_side():
    exact = wave_field()
    leaked = exact * (1.0 - 0.05 * np.linspace(0, 1, NT))[:, None]          # mass decaying 5 %
    report = run_surrogate_gates(bundle_for(leaked), reference=bundle_for(exact), field="rho",
                                 conserved="rho", gates=("conservation",))
    gate = report.get("conservation")
    assert gate.passed is False and gate.value == pytest.approx(0.05, abs=2e-3)
    assert "the data drifts" in gate.detail                                  # the reference's own drift is shown


def test_a_sign_changing_field_has_no_drift_to_measure():
    u = 0.3 * np.sin(2 * np.pi * (np.linspace(0, 1, NX)[None, :] - 0.4 * np.linspace(0, 1, NT)[:, None]))
    report = run_surrogate_gates(bundle_for(u), field="rho", conserved="rho", gates=("conservation",))
    assert report.get("conservation").status == "not run"
    assert "zero to rounding" in report.get("conservation").detail


# ── boundary conditions ─────────────────────────────────────────────────────────────────────────────────────
def test_a_dirichlet_condition_that_holds_passes():
    u = wave_field()
    bcs = [BoundaryCondition(field="rho", region="x_low", kind="dirichlet", value=float(u[0, 0])),
           BoundaryCondition(field="rho", region="x_high", kind="dirichlet", value=float(u[0, -1]))]
    # a field pinned to its edge value at every step
    pinned = u.copy()
    pinned[:, 0] = u[0, 0]
    pinned[:, -1] = u[0, -1]
    report = run_surrogate_gates(bundle_for(pinned, bcs=bcs), field="rho", gates=("boundary",))
    assert report.get("boundary").passed is True
    assert "2 Dirichlet condition(s)" in report.get("boundary").detail


def test_a_violated_boundary_condition_fails():
    u = wave_field()
    bcs = [BoundaryCondition(field="rho", region="x_low", kind="dirichlet", value=float(u[0, 0]))]
    report = run_surrogate_gates(bundle_for(u, bcs=bcs), field="rho", gates=("boundary",))
    gate = report.get("boundary")
    assert gate.passed is False and gate.value > 0


def test_a_region_that_names_no_edge_is_reported_not_ignored():
    bcs = [BoundaryCondition(field="rho", region="inlet", kind="dirichlet", value=0.0)]
    report = run_surrogate_gates(bundle_for(wave_field(), bcs=bcs), field="rho", gates=("boundary",))
    gate = report.get("boundary")
    assert gate.passed is False and "does not name an edge" in gate.detail


def test_no_boundary_condition_is_not_run():
    report = run_surrogate_gates(bundle_for(wave_field()), field="rho", gates=("boundary",))
    assert report.get("boundary").status == "not run"


# ── residual ───────────────────────────────────────────────────────────────────────────────────────────────
def test_the_residual_separates_the_right_speed_from_the_wrong_one():
    exact = bundle_for(wave_field())
    right = run_surrogate_gates(exact, field="rho", gates=("residual",), equation="advection",
                                parameters={"c": 0.4})
    wrong = run_surrogate_gates(exact, field="rho", gates=("residual",), equation="advection",
                                parameters={"c": 0.5})
    assert right.get("residual").passed is True and right.get("residual").value < 1e-2
    assert wrong.get("residual").passed is False and wrong.get("residual").value > 1.0


def test_the_residual_needs_an_equation_it_can_discretise():
    report = run_surrogate_gates(bundle_for(wave_field()), field="rho", gates=("residual",), equation="")
    assert report.get("residual").status == "not run"
    assert "will not infer one from the prose" in report.get("residual").detail

    unknown = run_surrogate_gates(bundle_for(wave_field()), field="rho", gates=("residual",),
                                  equation="navier-stokes")
    assert unknown.get("residual").status == "not run"
    assert "pass residual=" in unknown.get("residual").detail


def test_a_custom_operator_is_accepted():
    def my_residual(field, t, x, **kw):
        return np.zeros_like(field)

    report = run_surrogate_gates(bundle_for(wave_field()), field="rho", gates=("residual",),
                                 residual=my_residual)
    assert report.get("residual").passed is True
    assert report.get("residual").value == 0.0


def test_burgers_operator_is_registered_and_discriminates():
    assert "burgers" in RESIDUALS and RESIDUALS["burgers"] is burgers_residual
    x = np.linspace(0, 1, 64)
    t = np.linspace(0, 1, 32)
    flat = np.ones((32, 64))
    assert np.allclose(burgers_residual(flat, t, x, nu=0.01), 0.0)
    advecting = 1.0 + 0.1 * np.sin(2 * np.pi * (x[None, :] - 0.5 * t[:, None]))
    assert np.max(np.abs(advection_residual(advecting, t, x, c=0.5))) < 0.05


# ── the report ──────────────────────────────────────────────────────────────────────────────────────────────
def test_the_report_says_which_gates_ran_and_which_did_not():
    report = run_surrogate_gates(bundle_for(wave_field()), reference=bundle_for(wave_field()), field="rho",
                                 model="fno", conserved="rho")
    text = report.summary()
    assert "fno" in text and "accuracy" in text and "not run" in text
    assert report.to_dict()["gates"]
    assert report.ok in (True, False)


def test_the_ok_flag_needs_at_least_one_gate_to_have_run():
    report = run_surrogate_gates(bundle_for(wave_field()), field="rho", gates=("boundary",))
    assert report.get("boundary").status == "not run"
    assert report.ok is False


def test_the_verdicts_seal_into_the_ledger():
    exact = bundle_for(wave_field())
    report = run_surrogate_gates(exact, reference=exact, field="rho", model="fno", gates=("accuracy",))
    ledger = EvidenceLedger()
    report.record(ledger, inputs_sha256=exact.digest())
    assert ledger.status("surrogate/accuracy") == "passed"
    assert ledger.latest("surrogate/accuracy").outputs["model"] == "fno"
