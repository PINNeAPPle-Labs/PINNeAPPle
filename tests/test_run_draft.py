"""A drafted model run as a job (#1246): the battery-cell and solid-heat drafts on a box by finite differences with
the manufactured-solution order measured, the balance of the drafted law, the assumptions re-stated, a closed-form
slab, a PINN on the draft's own loss checked against the FD solution, and refusals that name their reason."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_problemdesign.formulate import formulate
from pinneapple_problemdesign.run_draft import EXPECTED_ORDER, run_draft

# illustrative values (as in tests/test_problemdesign_formulate.py): they only make the drafts solvable
CELL = dict(rho=2500, cp=1000, k=1.0, I=5, R0=0.03, EaR=2000, Tref=298.15, dUdT=1e-4, Vc=1.65e-5, h=10,
            Tamb=298.15, T0=298.15)
STEEL = dict(rho=7800, cp=500, k=45, q=1e5, h=50, Tamb=300, T0=300)


def _status(run):
    return {c["code"]: c["status"] for c in run.checks}


@pytest.mark.parametrize("phenomenon,values,box,t_end", [
    ("battery_cell_thermal", CELL, [0.065, 0.03, 0.0085], 600.0),
    ("solid_heat", STEEL, [0.1, 0.05, 0.02], 3600.0),
])
def test_the_drafts_run_end_to_end_on_a_box_at_the_expected_order(phenomenon, values, box, t_end):
    d = formulate(phenomenon, dims=3, values=values)
    r = run_draft(d, geometry={"box": box}, budget={"t_end": t_end, "n_t": 100, "n": 11})
    assert r.status == "done" and r.method == "fd" and "finite differences" in r.reason
    st = _status(r)
    assert all(v != "fail" for v in st.values()), r.checks
    assert st["run.mms_order"] == st["run.balance"] == st["run.time_step"] == "pass"
    assert abs(r.mms["observed_order"] - EXPECTED_ORDER) < 0.1                       # measured and stated
    errs = [lv["max_error"] for lv in r.mms["levels"]]
    assert errs[0] > errs[1] > errs[2]
    assert r.balance["imbalance_rel"] < 1e-9                                           # flux faces: exact balance
    assert r.balance["source"] > 0 and r.balance["inflow"] < 0                         # heated, losing to ambient
    assert r.fields["T"].shape == (11, 11, 11) and r.metrics["max"] > values["Tamb"]
    ids = {a["id"] for a in r.assumptions}
    assert {"box_domain", "lumped"} <= ids
    lumped = next(a for a in r.assumptions if a["id"] == "lumped")
    assert lumped["applied"] is False and "Bi = h*Lc/k" in lumped["evaluated"] and "solution" in lumped
    assert "MMS order" in r.summary() and r.to_dict()["passed"] is True


def test_the_battery_heat_depends_on_temperature_and_newton_follows_it():
    d = formulate("battery_cell_thermal", dims=1, values=CELL)
    r = run_draft(d, geometry=[0.065], budget={"t_end": 600, "n_t": 100, "n": 41})
    assert r.metrics["newton_iterations_max"] >= 2                                     # exp(EaR*(1/T - 1/Tref))
    assert abs(r.mms["observed_order"] - 2) < 0.05 and _status(r)["run.balance"] == "pass"


def test_steady_slab_matches_the_closed_form():
    # -k T'' = q, -k T' = h (T - Tamb) on both faces: T = Tamb + q L/(2h) + q x (L - x)/(2k), exact for FD
    d = formulate("solid_heat", dims=1, steady=True, values=STEEL)
    r = run_draft(d, geometry=[0.1], budget={"n": 41})
    x, L = r.axes["x"], 0.1
    exact = 300 + 1e5 * L / (2 * 50) + 1e5 * x * (L - x) / (2 * 45)
    assert np.max(np.abs(r.fields["T"] - exact)) < 1e-8
    assert _status(r)["run.mms_order"] == "pass" and "run.time_step" not in _status(r)
    assert any(a["id"] == "steady" and a["applied"] for a in r.assumptions)


def test_a_dirichlet_face_from_the_caller():
    faces = {"T": [{"kind": "dirichlet", "on": "x_min", "equation": "T = Tamb"},
                   {"kind": "robin", "on": "x_max", "equation": "-k*T_x = h*(T - Tamb)"},
                   {"kind": "neumann", "on": "y_min", "equation": "T_y = 0"},
                   {"kind": "neumann", "on": "y_max", "equation": "T_y = 0"},
                   {"kind": "initial", "on": "t_min", "equation": "T = T0"}]}
    d = formulate("solid_heat", dims=2, values=STEEL, faces=faces)
    r = run_draft(d, geometry=[0.1, 0.05], budget={"t_end": 3600, "n_t": 100, "n": 21})
    assert r.status == "done" and _status(r)["run.mms_order"] == "pass"
    assert np.allclose(r.fields["T"][0, :], 300.0)                                     # the held face
    assert r.balance["tolerance"] == 1e-2 and r.balance["imbalance_rel"] < 1e-2        # one-sided flux there


def test_the_pinn_trains_on_the_drafts_loss_and_is_compared_with_fd():
    d = formulate("solid_heat", dims=1, values=STEEL)
    r = run_draft(d, geometry=[0.1], method="pinn",
                  budget={"t_end": 3600, "n_t": 50, "n": 21, "adam_steps": 30, "lbfgs_steps": 0,
                          "n_collocation": 128})
    assert r.status == "done" and r.method == "pinn"
    st = _status(r)
    assert {"pinn.residual", "run.reference_fd", "run.mms_order", "model.mms"} <= set(st)
    assert st["run.mms_order"] == "inconclusive"                                        # a PINN has no grid order
    assert "reference.fd_max_error_rel" in r.metrics and r.fields["T"].shape == (21,)


@pytest.mark.parametrize("phenomenon,kwargs,method,needle", [
    ("arrhenius", {}, "auto", "algebraic relation"),
    ("arrhenius", {}, "fd", "algebraic relation"),
    ("solid_heat", {"dims": 1}, "fd", "to be given"),
    ("room_air_species", {"dims": 2, "values": dict(Sc=0.0, T0=293, Tin=293, Twall=293, Uin=0.2, c0=16, cin=16,
                                                    qT=0.0)}, "fd", "coupled fields"),
    ("solid_heat", {"dims": 1, "values": STEEL}, "fem", "method='fd'"),
])
def test_a_draft_without_a_discretisation_is_refused_with_the_reason(phenomenon, kwargs, method, needle):
    r = run_draft(formulate(phenomenon, **kwargs), method=method)
    assert r.status == "refused" and needle in r.reason, r.reason
    assert r.passed is False and r.to_dict()["status"] == "refused"


def test_a_nonlinear_condition_is_refused_by_fd():
    faces = {"T": [{"kind": "robin", "on": "x_min", "equation": "k*T_x = h*(T - Tamb)"},
                   {"kind": "robin", "on": "x_max", "equation": "-k*T_x = h*(T**4 - Tamb**4)/Tamb**3"},
                   {"kind": "initial", "on": "t_min", "equation": "T = T0"}]}
    d = formulate("solid_heat", dims=1, values=STEEL, faces=faces)
    r = run_draft(d, geometry=[0.1], method="fd")
    assert r.status == "refused" and "nonlinear" in r.reason


def test_flow_node_bundled_flow_and_capability():
    import json
    from pathlib import Path

    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert REGISTRY.get("problemdesign.run_draft") is not None
    doc = json.loads((Path(__file__).resolve().parents[1] / "pinneapple_flow" / "flows" /
                      "run_drafted_model.json").read_text())
    assert [n["type"] for n in doc["nodes"]][0] == "problemdesign.run_draft"
    for text in ("rodar o modelo proposto pelo formulate", "run the drafted model as a job"):
        assert match(text)[0][1].id == "problemdesign.run_draft", text
