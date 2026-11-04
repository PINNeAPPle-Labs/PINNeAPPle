"""The standard chip-design workflow (#1206): the 1-D stack closed form, the uniform-grid IR drop, Elmore against a
trapezoidal transient, the leakage loop, the placement and strap design and the refined re-verification."""
from dataclasses import replace

import numpy as np
import pytest

from pinneapple_design.chip import (
    Block,
    Floorplan,
    GridSpec,
    WireSpec,
    black_mttf_years,
    chip_design_workflow,
    default_floorplan,
    default_stack,
    elmore_delay,
    solve_thermal,
    stack_closed_form,
    transient_step,
    uniform_grid_closed_form,
)
from pinneapple_design.chip.interconnect import rc_ladder, wire_rc
from pinneapple_design.chip.power_grid import POISSON_SQUARE_MAX, mesh_drop


def _status(r, code):
    return next(c["status"] for c in r["checks"] if c["code"] == code)


@pytest.fixture(scope="module")
def default_run():
    return chip_design_workflow()


def test_default_design_passes_every_check(default_run):
    r = default_run
    assert r["verdict"] == "pass", r["failed"]
    assert r["thermal"]["T_max_c"] < r["thermal_initial"]["T_max_c"]         # the placement lowers the peak
    assert r["power_grid"]["max_drop_fraction"] <= 0.05
    assert abs(r["thermal_refined"]["T_max_c"] - r["thermal"]["T_max_c"]) < 1.0
    assert max(v["elmore_s"] for v in r["net_delays"].values()) <= 200e-12


def test_one_dimensional_stack_matches_the_resistance_network():
    st = default_stack()
    fp = Floorplan(10, 10, [Block("u", 50.0)], (1, 1), [0])
    r = solve_thermal(fp, st, nx=5, leakage=False)
    cf = stack_closed_form(st, 1e-4, 50.0)
    assert np.allclose(r["T_junction"], cf["T_j_c"], rtol=0, atol=1e-10)
    # refining the cells moves the source plane to the die face: the offset shrinks toward the face network
    fine = stack_closed_form(st.refined(8), 1e-4, 50.0)
    assert abs(fine["R_ja_k_w"] - cf["R_ja_face_k_w"]) < abs(cf["R_ja_k_w"] - cf["R_ja_face_k_w"])


def test_mesh_ir_drop_equals_the_discrete_sine_solution_and_converges_to_the_continuum():
    errs = []
    for n in (15, 31, 63):
        fixed = np.zeros((n + 2, n + 2), bool)
        fixed[[0, -1], :] = fixed[:, [0, -1]] = True
        load = np.zeros((n + 2, n + 2))
        load[1:-1, 1:-1] = 1e-3
        d = mesh_drop(load, np.full((n + 1, n + 2), 2.0), np.full((n + 2, n + 1), 2.0), fixed=fixed)
        exact = uniform_grid_closed_form(n, 2.0, 1e-3)
        assert np.max(np.abs(d[1:-1, 1:-1] - exact)) < 1e-12
        errs.append(abs(exact.max() / (POISSON_SQUARE_MAX * 1e-3 / 2.0 * (n + 1) ** 2) - 1))
    assert errs[0] > errs[1] > errs[2] and errs[1] / errs[2] > 3                # second order


def test_elmore_is_the_first_moment_and_an_upper_bound_of_the_50_percent_delay():
    w = WireSpec()
    r, c = wire_rc(5e-3, w, 60.0)
    R, C = rc_ladder(r, c, r_driver=w.r_driver_ohm, c_load=w.c_load_f, n=40)
    tau = elmore_delay(R, C)
    tr = transient_step(R, C, t_end=12 * tau, dt=tau / 4000)
    assert abs(tr["first_moment_s"] - tau) / tau < 1e-4
    assert tr["t50_s"] < tau
    # a lumped RC is the single-pole case: 50 % at ln 2 tau
    tr1 = transient_step(np.array([1e3]), np.array([1e-12]), t_end=12e-9, dt=1e-13)
    assert abs(tr1["t50_s"] / 1e-9 - np.log(2)) < 1e-4


def test_black_equation_reference_point_and_scalings():
    s = GridSpec()
    assert abs(black_mttf_years(s.j_ref_a_m2, s.t_ref_c, s) - s.mttf_ref_years) < 1e-12
    assert abs(black_mttf_years(2 * s.j_ref_a_m2, s.t_ref_c, s) - s.mttf_ref_years / 4) < 1e-12   # n = 2
    assert black_mttf_years(s.j_ref_a_m2, 125.0, s) < s.mttf_ref_years


def test_leakage_runaway_is_detected():
    st = replace(default_stack(), r_sink_k_w=3.0)
    fp = Floorplan(10, 10, [Block("hot", 20.0, 20.0, leak_beta=0.05)], (1, 1), [0])
    r = solve_thermal(fp, st, nx=6, max_iter=60)
    assert r["runaway"] and not r["converged"]


def test_tight_ir_limit_needs_wider_straps():
    a = chip_design_workflow(ir_drop_max=0.05, optimize_placement=False)
    b = chip_design_workflow(ir_drop_max=0.03, optimize_placement=False)
    assert b["strap_width_um"] > a["strap_width_um"]
    assert _status(b, "power_grid.ir_drop") == "pass"


def test_infeasible_delay_budget_is_reported():
    r = chip_design_workflow(delay_budget_ps=50.0, optimize_placement=True)
    assert _status(r, "placement.feasible") == "fail" and _status(r, "interconnect.delay") == "fail"


def test_the_default_floorplan_round_trips_through_its_dict():
    from pinneapple_design.chip import floorplan_from_dict
    fp = default_floorplan()
    d = fp.to_dict()
    fp2 = floorplan_from_dict({**d, "blocks": [dict(b, leakage_w=b["leakage_w"]) for b in d["blocks"]]})
    assert [b.name for b in fp2.blocks] == [b.name for b in fp.blocks] and fp2.nets == fp.nets


def test_the_planner_proposes_the_workflow_in_portuguese_and_english():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match
    for text in ("queda IR na rede de alimentação e eletromigração do chip",
                 "chip floorplan hot spot, IR drop and Elmore delay"):
        assert match(text)[0][1].id == "chip.design"
