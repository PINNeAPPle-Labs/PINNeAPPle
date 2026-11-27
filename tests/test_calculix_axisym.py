"""The axisymmetric element: the mesh, the deck it is written into, and Lamé's thick cylinder against it.

Issue #142: ``app/prepomax/deck.py`` wrote ``C3D8`` bricks only, so an axisymmetric problem — Roark's rigid cylindrical
die among them — had no element. The two ways of faking it both fail: a polar block of bricks degenerates on ``r = 0``
and the solver refuses the Jacobian, and a staircase footprint changes the loaded area, which is the quantity the
closed form is about. This is the element instead, and the case that proves it: a thick cylinder under internal
pressure has an exact solution for **all three** stress components and for the bore displacement, so nothing here is
tuned to pass.
"""
from __future__ import annotations

import math

import pytest

from pinneapple_simulation.external_solvers.calculix.contact import cases as C
from pinneapple_simulation.external_solvers.calculix.contact import deck as D
from pinneapple_simulation.external_solvers.calculix.contact import run as R
from pinneapple_simulation.external_solvers.calculix.contact import verify as V
from pinneapple_simulation.external_solvers.calculix.contact.mesh import QuadGrid

STEEL_E, STEEL_NU = 206000.0, 0.3


# ----------------------------------------------------------------------------------------------- the closed form first
def test_lame_is_the_textbook_cylinder():
    """``sigma_theta(a) = p (b^2 + a^2)/(b^2 - a^2)``, ``sigma_r(a) = -p``, ``sigma_z = 2 nu A``."""
    p, a, b = 100.0, 10.0, 20.0
    bore = V.thick_cylinder(p, a, b, STEEL_E, STEEL_NU, a)
    assert bore["sigma_r"] == pytest.approx(-p, rel=1e-12)
    assert bore["sigma_theta"] == pytest.approx(p * (b * b + a * a) / (b * b - a * a), rel=1e-12)
    A = p * a * a / (b * b - a * a)
    assert bore["sigma_z"] == pytest.approx(2 * STEEL_NU * A, rel=1e-12)
    assert V.thick_cylinder(p, a, b, STEEL_E, STEEL_NU, b)["sigma_r"] == pytest.approx(0.0, abs=1e-12)
    # the radial displacement of the bore is positive (outward), small, and scales with the pressure
    u = bore["u_r"]
    assert 0 < u < 0.05
    assert V.thick_cylinder(2 * p, a, b, STEEL_E, STEEL_NU, a)["u_r"] == pytest.approx(2 * u, rel=1e-12)
    assert V.thick_cylinder(p, a, b, STEEL_E, STEEL_NU, 0.5 * a) == {}, "a point inside the bore is not the body"


# ---------------------------------------------------------------------------------------------------- the mesh itself
def test_the_quad_mesh_shares_its_mid_side_nodes():
    grid = QuadGrid([10.0, 15.0, 20.0], [0.0, 1.0, 2.0])
    assert len(grid.cells) == 4
    assert all(len(cell) == 8 for cell in grid.cells)
    assert len({n for cell in grid.cells for n in cell}) == len(grid.nodes), \
        "every node of a conforming mesh is used, and none is invented twice"
    corners, mids = grid.nodes[grid.cells[0][0]], grid.nodes[grid.cells[0][4]]
    assert mids[0] == pytest.approx((corners[0] + grid.nodes[grid.cells[0][1]][0]) / 2)
    # two stacked cells share the three nodes of the edge between them, mid-side node included
    shared = set(grid.cells[0]) & set(grid.cells[1])
    assert len(shared) == 3, "a conforming quadratic edge is two corners and one mid-side node"


def test_the_axis_has_to_stay_off_zero_and_the_coordinates_have_to_increase():
    from pinneapple_simulation.external_solvers.calculix.contact.mesh import QuadGrid as Q

    with pytest.raises(ValueError, match="off r = 0"):
        Q([0.0, 10.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        Q([10.0, 20.0, 15.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="two coordinates"):
        Q([10.0], [0.0])


def test_the_ring_area_is_the_revolved_area_not_the_flat_one():
    """The number that makes an axisymmetric reaction readable: ``2 pi r h`` for a bore, ``pi (b^2 - a^2)`` for a face."""
    grid = QuadGrid([10.0, 15.0, 20.0], [0.0, 1.0])
    assert grid.ring_area("r", 10.0) == pytest.approx(2 * math.pi * 10.0 * 1.0)
    assert grid.ring_area("r", 20.0) == pytest.approx(2 * math.pi * 20.0 * 1.0)
    assert grid.ring_area("z", 1.0) == pytest.approx(math.pi * (20.0 ** 2 - 10.0 ** 2))


def test_the_faces_are_selected_by_the_plane_they_lie_in():
    grid = QuadGrid([10.0, 15.0, 20.0], [0.0, 1.0])
    bore = grid.faces("r", 10.0)
    assert len(bore) == 1 and bore[0][1] == 4, "the 4-1 edge of the only element is the bore"
    assert len(grid.faces("r", 20.0)) == 1 and grid.faces("r", 20.0)[0][1] == 2
    bottom = grid.faces("z", 0.0)
    assert len(bottom) == 2 and {f for _, f in bottom} == {1}, "two elements, one 1-2 edge each"


# -------------------------------------------------------------------------------------------- writing the 2-D deck
def test_geometry_2d_writes_two_coordinates_and_the_element_type():
    deck = D.Deck("teste")
    D.geometry_2d(deck, {1: (10.0, 0.0), 2: (10.5, 0.0), 3: (10.0, 1.0)},
                  [(1, [1, 2, 3, 4, 5, 6, 7, 8])], "CIL")
    text = deck.text()
    assert "*ELEMENT, TYPE=CAX8, ELSET=CIL" in text
    assert "1, 10, 0" in text, "a node line carries (r, z) and nothing else"
    assert "1, 1, 2, 3, 4, 5, 6, 7, 8" in text


def test_geometry_2d_refuses_what_ccx_would_read_as_something_else():
    deck = D.Deck("teste")
    with pytest.raises(ValueError, match="axisymmetric element"):
        D.geometry_2d(deck, {1: (10.0, 0.0)}, [], "CIL", type="C3D8")
    with pytest.raises(ValueError, match="writes \\(r, z\\)"):
        D.geometry_2d(deck, {1: (10.0, 0.0, 0.0)}, [], "CIL")
    with pytest.raises(ValueError, match="CAX8 needs 8"):
        D.geometry_2d(deck, {1: (10.0, 0.0)}, [(1, [1, 2, 3, 4])], "CIL")


# -------------------------------------------------------------------------------------------------- the case, solved
def test_the_case_is_in_the_library_and_declares_what_it_measures():
    case, params, built = C.build("cilindro_lame", {})
    assert case.test == "tests/test_calculix_axisym.py"
    assert params["raio_externo"] > params["raio_interno"] > 0
    assert built.analytic["ring_area"] == pytest.approx(2 * math.pi * params["raio_interno"] * params["altura"])
    assert "*ELEMENT, TYPE=CAX8" in built.deck
    assert "CAX8" in " ".join(built.notes)
    checks = case.checks(params, built, {"stress": {}, "displacement": {}})
    assert all(c.outcome == "inconclusive" for c in checks), "nothing read is nothing passed"


SOLVER = R.ccx_binary()
needs_solver = pytest.mark.skipif(SOLVER is None, reason="no ccx binary (set CCX_BIN) in this environment")


@needs_solver
def test_the_thick_cylinder_closes_all_three_stresses_and_the_bore(tmp_path):
    """Measured in CalculiX 2.23: the three stresses and u_r inside 0.5 %, and the solver's force convention pinned.

    The force convention is the part that had to be measured rather than assumed: the axial reaction the ``.dat``
    prints is **1/180** of the ring total ``sigma_z * pi (b^2 - a^2)``, the same factor for four runs that vary the
    radius, the height and the pressure. The case checks the ratio, so a solver that changes it is caught.
    """
    result = R.run_case("cilindro_lame", {}, tmp_path, binary=SOLVER, timeout=600)
    assert result["status"] == "ok", result.get("error")
    failed = [c.as_dict() for c in result["checks"] if c.outcome == "fail"]
    assert not failed, failed
    assert {c.outcome for c in result["checks"]} == {"pass"}
    by_name = {c.name: c for c in result["checks"]}
    assert by_name["lame.sigma_theta_furo"].value == pytest.approx(166.67, rel=0.01)
    assert by_name["lame.reacao_axial"].value == pytest.approx(1 / 180, rel=1 / 180)
