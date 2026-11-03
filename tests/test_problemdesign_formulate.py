"""A full physical model for a problem no preset covers (#1175): composed from laws with units and sources, checked
(dimensions, closure, conditions on every face, a manufactured solution through the compiled residual), editable
with every check re-run, and a physics loss that trains in physical units."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_problemdesign.formulate import Coefficient, formulate, known

# illustrative values of a cylindrical-cell-sized slab (not a datasheet): they only make the problem solvable
CELL = dict(rho=2500, cp=1000, k=1.0, I=5, R0=0.03, EaR=2000, Tref=298.15, dUdT=1e-4, Vc=1.65e-5, h=10,
            Tamb=298.15, T0=298.15)


def _status(draft):
    return {c["code"]: c["status"] for c in draft.checks()}


def test_the_battery_cell_keeps_conduction_bernardi_heat_and_convection_on_every_face():
    d = formulate("battery_cell_thermal")
    assert d.coords == ["x", "y", "z", "t"] and d.fields == {"T": "K"}
    assert "lap(T)" in d.pdes[0] and "exp(EaR*(1/T - 1/Tref))" in d.pdes[0] and "I*T*dUdT" in d.pdes[0]
    faces = {c["on"] for c in d.conditions if c["kind"] == "robin"}
    assert faces == {f"{x}_{s}" for x in "xyz" for s in ("min", "max")}
    st = _status(d)
    assert st == {"model.parses": "pass", "model.dimensions": "pass", "model.closed": "pass",
                  "model.conditions": "pass", "model.constants": "inconclusive", "model.mms": "pass"}
    todo = next(c for c in d.checks() if c["code"] == "model.constants")["detail"]
    assert "R0" in todo and "dUdT" in todo                          # never invented: to be given
    assert any("Bernardi" in s for s in d.sources)
    lumped = d.assumptions[0]
    assert lumped.id == "lumped" and not lumped.applied and lumped.limit == "Bi < 0.1"


def test_the_lumped_reduction_is_offered_with_its_biot_number_never_applied():
    d = formulate("battery_cell_thermal", values={"h": 10, "k": 1.0, "Lc": 0.009})
    a = d.assumptions[0]
    assert a.value == pytest.approx(0.09) and a.holds is True and not a.applied
    assert "lap(T)" in d.pdes[0]


def test_the_ventilated_room_couples_flow_buoyancy_energy_and_species():
    d = formulate("room_air_species", dims=2)
    assert set(d.fields) == {"u", "v", "p", "T", "c"} and len(d.pdes) == 5
    assert d.pdes[2].endswith("rho*g*beta*(T - T0)")              # buoyancy along the vertical axis of the 2-D room
    assert d.coefficients["D"].value == pytest.approx(1.77e-5) and "Table 17.1-1" in d.coefficients["D"].source
    assert d.coefficients["rho"].value == pytest.approx(1.1614)
    assert _status(d)["model.dimensions"] == "pass" and _status(d)["model.conditions"] == "pass"
    applied = {a.id for a in d.assumptions if a.applied}
    assert applied == {"planar", "full_face_openings"}             # every reduction stated
    assert {a.id for a in d.assumptions if not a.applied} == {"well_mixed"}


def test_an_edit_that_breaks_dimensions_names_the_terms():
    d = formulate("room_air_species", dims=2)
    bad = d.edit(pde=(4, "c_t + u*c_x + v*c_y = D*lap(c) + Sc*u"))
    fail = next(c for c in bad.checks() if c["code"] == "model.dimensions")
    assert fail["status"] == "fail" and "Sc*u" in fail["detail"]
    assert bad.history[-1]["pde"][0] == 4 and d.history == []      # the old draft is untouched


def test_a_missing_face_and_initial_condition_are_found():
    d = formulate("solid_heat", dims=1)
    gone = d.edit(condition=(1, {"kind": "robin", "on": "x_min", "equation": "k*T_x = h*(T - Tamb)"}))
    gone.conditions = [c for c in gone.conditions if c["kind"] != "initial"]
    detail = next(c for c in gone.checks() if c["code"] == "model.conditions")["detail"]
    assert "T on x_max" in detail and "T initial condition" in detail


def test_steady_drops_the_time_derivative_and_says_so():
    d = formulate("solid_heat", dims=2, steady=True)
    assert "t" not in d.coords and "T_t" not in d.pdes[0]
    assert {a.id for a in d.assumptions if a.applied} == {"planar", "steady"}
    assert _status(d)["model.conditions"] == "pass"


def test_constants_from_the_user_and_an_edited_constant():
    d = formulate("battery_cell_thermal", dims=1, values=CELL)
    assert _status(d)["model.constants"] == "pass"
    assert d.coefficients["R0"].source == "user"
    d2 = d.edit(coefficient=Coefficient("R0", "ohm", "internal resistance at Tref", 0.025, "HPPC test, cell #3"))
    assert d2.coefficients["R0"].value == 0.025 and d.coefficients["R0"].value == 0.03


def test_latex_and_the_algebraic_relation():
    d = formulate("solid_heat", dims=1)
    tex = d.latex()
    assert r"\frac{\partial}{\partial t} T" in tex["pdes"][0] and r"\left(x,t" not in tex["pdes"][0]
    assert tex["conditions"][0]["kind"] == "robin"
    r = formulate("arrhenius")
    assert r.latex()["relation"][0].startswith("k = A e^{")
    assert _status(r) == {"model.dimensions": "pass"}
    assert "Ea" in {c["name"] for c in r.constants()}


def test_unknown_phenomenon_lists_the_known_ones():
    with pytest.raises(KeyError, match="battery_cell_thermal"):
        formulate("plasma")
    assert "room_air_species" in known()


def test_the_physics_loss_trains_in_physical_units_and_matches_finite_differences():
    pytest.importorskip("torch")
    import torch
    d = formulate("battery_cell_thermal", dims=1, values=CELL)
    assert d.scales() == {"T": (298.15, 1.0)}
    code = d.loss_code({"x": (0.0, 0.018), "t": (0.0, 600.0)})
    assert "scales=SCALES" in code
    ns: dict = {}
    exec(code.replace("adam_steps=2000, lbfgs_steps=200", "adam_steps=1000, lbfgs_steps=100"), ns)  # noqa: S102
    # explicit finite differences of the same model (ghost-node Robin faces)
    v, L, n = CELL, 0.018, 61
    x = np.linspace(0, L, n)
    dx = x[1]
    dt = 0.2 * dx * dx * v["rho"] * v["cp"] / v["k"]
    T, t = np.full(n, v["T0"], float), 0.0
    while t < 600 - 1e-9:
        h = min(dt, 600 - t)
        q = (v["I"] ** 2 * v["R0"] * np.exp(v["EaR"] * (1 / T - 1 / v["Tref"])) - v["I"] * T * v["dUdT"]) / v["Vc"]
        g = np.concatenate([[T[1] - 2 * dx * v["h"] * (T[0] - v["Tamb"]) / v["k"]], T,
                            [T[-2] - 2 * dx * v["h"] * (T[-1] - v["Tamb"]) / v["k"]]])
        T = T + h * (v["k"] * (g[2:] - 2 * g[1:-1] + g[:-2]) / dx ** 2 + q) / (v["rho"] * v["cp"])
        t += h
    X = torch.tensor(np.stack([x, np.full(n, 600.0)], 1), dtype=torch.float32)
    pinn = ns["net"](X).detach().numpy().ravel()
    rise = T.max() - v["T0"]
    assert 5 < rise < 9                                            # the cell heats by several kelvin in 10 min
    assert np.abs(pinn - T).max() < 0.05 * rise


def test_the_relations_code_calibrates_its_unknowns_from_measurements():
    code = formulate("arrhenius").loss_code()
    assert "value=..., lower=..., upper=..." in code                # bounds are the user's to give, never invented
    T = np.linspace(300.0, 400.0, 12)
    rng = np.random.default_rng(0)
    y = 1e7 * np.exp(-5e4 / (8.314462618 * T)) * (1 + 0.02 * rng.standard_normal(12))
    code = (code.replace("Parameter('A', value=..., lower=..., upper=...", "Parameter('A', value=1e6, lower=1e3, upper=1e10")
            .replace("Parameter('Ea', value=..., lower=..., upper=...", "Parameter('Ea', value=4e4, lower=1e4, upper=1e5")
            .replace("{'T': ...}, values=..., sigma=...", "{'T': T}, values=y, sigma=0.02*y"))
    ns: dict = {"T": T, "y": y}
    exec(code, ns)  # noqa: S102
    est = {p["name"]: p for p in ns["fit"].to_dict()["parameters"]}
    assert est["Ea"]["calibrated"] == pytest.approx(5e4, rel=0.02)
    lo, hi = est["A"]["ci_jacobian"]
    assert lo < 1e7 < hi
