"""Meshless and mesh-based solvers of the same 2-D problems (pinneapple_simulation.meshless), checked against
manufactured solutions: observed orders, conditioning, the Kansa shape-parameter trade-off, heat, references."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.meshless import (
    Problem,
    discretize,
    domain,
    meshless_study,
    point_cloud_domain,
    solve,
)

U = "sin(pi*x)*cos(2*y) + x*y"


def _err(method, dom, h, pr, **opt):
    d = dom.nodes(h)
    disc = discretize(method, dom, d, **opt)
    out = solve(pr, disc)
    E = dom.sample_eval(1500)
    v, ue = out["evaluate"](E), pr.u(E)
    return np.linalg.norm(v - ue) / np.linalg.norm(ue), out


def test_boundary_nodes_lie_on_the_boundary_and_include_the_corners():
    d = domain("lshape")
    ns = d.nodes(0.05)
    B = ns.X[ns.boundary]
    assert np.abs(d.sdf(B)).max() < 1e-9
    for c in [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]:
        assert np.hypot(*(B - c).T).min() < 1e-12
    assert (d.sdf(ns.X[ns.interior]) < -0.4 * 0.05 + 1e-12).all()
    hole = domain("plate_hole").nodes(0.05)
    assert np.abs(np.hypot(*(hole.X[hole.boundary] - 0.5).T) - 0.2).min() < 1e-12


@pytest.mark.parametrize("method,low,high", [("fd", 1.8, 2.3), ("fem", 1.8, 2.3), ("rbf_fd", 3.5, 6.0)])
def test_observed_order_on_the_l_shape(method, low, high):
    pr = Problem("poisson", exact=U)
    e1, _ = _err(method, domain("lshape"), 0.05, pr)
    e2, _ = _err(method, domain("lshape"), 0.025, pr)
    assert low < np.log2(e1 / e2) < high


def test_kansa_is_accurate_but_ill_conditioned():
    pr = Problem("poisson", exact=U)
    e, out = _err("kansa", domain("lshape"), 0.05, pr)
    assert e < 1e-4 and out["cond"] > 1e12
    _, fem = _err("fem", domain("lshape"), 0.05, pr)
    assert fem["cond"] < 1e4


def test_helmholtz_on_the_disk_and_heat_on_the_plate_with_a_hole():
    hz = Problem("helmholtz", exact=U, k=3.0)
    assert _err("rbf_fd", domain("disk"), 0.1, hz)[0] < 5e-3
    assert _err("fem", domain("disk"), 0.1, hz)[0] < 0.1
    heat = Problem("heat", exact="exp(-t)*sin(pi*x)*sin(pi*y) + x", t_end=0.1, n_steps=50)
    assert _err("rbf_fd", domain("plate_hole"), 0.05, heat)[0] < 1e-4
    assert _err("fd", domain("plate_hole"), 0.05, heat)[0] < 1e-2


def test_study_reports_orders_wins_and_the_kansa_sweep():
    r = meshless_study(Problem("poisson", exact=U), domain("square"), h=[0.1, 0.05], methods=("kansa", "rbf_fd", "fem"))
    d = r.to_dict()
    assert {w["criterion"].split(" ")[0] for w in d["wins"]} >= {"Accuracy", "Cheapest", "Fastest", "Best", "Least"}
    assert d["convergence"]["fem"]["order"] == pytest.approx(2.0, abs=0.3)
    assert len(d["eps_sweep"]) == 5 and all(s["cond"] for s in d["eps_sweep"] if np.isfinite(s["error"]))
    assert len(d["fields"]["methods"]["rbf_fd"]["values"]) == len(d["fields"]["points"])


def test_without_an_exact_solution_the_reference_is_a_fine_fem_solution():
    # -lap u = 1 on the unit disk, u = 0: exact (1 - r^2) / 4, used here only to check the reference itself
    pr = Problem("poisson", source="1", boundary="0")
    r = meshless_study(pr, domain("disk"), h=[0.2, 0.1], methods=("fem",))
    assert r.reference["kind"] == "fine FEM reference" and 0 < r.reference["estimated_error"] < 0.01
    exact = Problem("poisson", exact="(1 - x**2 - y**2)/4")
    e, _ = _err("fem", domain("disk"), 0.1, exact)
    assert r.rows[-1]["error"] == pytest.approx(e, rel=0.3)


def test_point_cloud_has_no_finite_differences_and_holds_out_points():
    sq = domain("square").nodes(0.05)
    pc = point_cloud_domain(sq.X, sq.boundary)
    r = meshless_study(Problem("poisson", exact="sin(pi*x)*sin(pi*y)"), pc, methods=("rbf_fd", "fd", "fem"))
    assert [s["method"] for s in r.skipped] == ["fd"]
    assert all(r_["error"] < 0.05 for r_ in r.rows)
    assert not (np.hypot(*(pc.eval_points[:, None] - pc.cloud.X[None]).transpose(2, 0, 1)) < 1e-12).any()


def test_pinn_trains_within_its_budget_on_the_nodes():
    r = meshless_study(Problem("poisson", exact="sin(pi*x)*sin(pi*y)"), domain("square"), h=[0.1],
                       methods=("pinn",), options={"pinn": {"budget_s": 1.0}})
    t = r.rows[0]["train"]
    assert t["stopped_by"] == "wall_s" and t["steps"] > 5 and np.isfinite(r.rows[0]["error"])


def test_bad_expressions_are_refused():
    with pytest.raises(ValueError):
        Problem("poisson", exact="__import__('os')")
    with pytest.raises(ValueError, match="unknown symbols"):
        Problem("poisson", exact="sin(z)")
