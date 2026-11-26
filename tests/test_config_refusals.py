"""A configuration object refuses bad input with ``ValueError``, not with an internal crash (#625).

The generic construction sweep in ``test_breadth_six_packages.py`` builds every public dataclass with placeholder
values and treats a class's own ``ValueError`` as an honest refusal; anything else is reported as a defect. Seven of
them used to crash inside ``__post_init__`` -- a tuple indexed as if it had two entries, a string read as a mapping,
a division by a zero slot -- so the error a user saw named an operation and never the field that was wrong. Each
case below is the smallest input that used to crash, and what the class says now instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.benchmark.validity import ValidityDomain
from pinneapple_analysis.dynamics.dimensional import DimensionlessGroup
from pinneapple_design.chip.floorplan import Floorplan
from pinneapple_simulation.electromagnetics.fdtd import YeeGrid
from pinneapple_simulation.electromagnetics.fem3d.mesh import TetMesh
from pinneapple_simulation.electromagnetics.machines.motor3d.thermal3d import AxialThermalModel
from pinneapple_simulation.immersed.geometry import SDFGeometry


def _refusal(build, *words: str) -> str:
    """The message of the ``ValueError`` ``build`` raises, checked to mention what the field needs."""
    with pytest.raises(ValueError) as caught:
        build()
    msg = str(caught.value)
    for w in words:
        assert w in msg, f"the refusal does not say {w!r}: {msg!r}"
    return msg


def test_validity_domain_refuses_bounds_that_are_not_a_mapping():
    """``bounds`` is documented as variable -> (low, high); a tuple of numbers is not that."""
    _refusal(lambda: ValidityDomain("box", (), source="test"), "bounds", "tuple")


def test_dimensionless_group_refuses_exponents_that_are_not_a_mapping():
    _refusal(lambda: DimensionlessGroup("test"), "exponents", "str")


def test_floorplan_refuses_slots_without_the_two_counts():
    """``slots`` is (nx, ny); empty used to be an ``IndexError`` on the first entry."""
    _refusal(lambda: Floorplan(1.0, 1.0, [], (), []), "slots")


def test_yee_grid_refuses_a_shape_without_three_axes():
    """Empty shape and spacing used to divide by zero inside the Courant limit."""
    _refusal(lambda: YeeGrid((), ()), "three entries")


def test_yee_grid_refuses_a_non_positive_spacing():
    _refusal(lambda: YeeGrid((4, 4, 4), (0.1, 0.0, 0.1)), "spacing")


def test_tet_mesh_refuses_nodes_that_are_not_a_coordinate_table():
    """One-dimensional nodes used to fail with ``too many indices`` from deep inside the jacobian."""
    _refusal(lambda: TetMesh(np.zeros(3), np.zeros(3, int), np.zeros(3, int)), "nodes", "(N, 3)")


def test_tet_mesh_refuses_a_region_without_one_tag_per_tetrahedron():
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    _refusal(lambda: TetMesh(nodes, np.array([[0, 1, 2, 3]]), np.zeros(2, int)), "one tag per tetrahedron")


def test_axial_thermal_model_refuses_a_missing_geometry():
    """``geometry.length`` used to be read straight off ``None``."""
    _refusal(lambda: AxialThermalModel(None), "geometry")


def test_sdf_geometry_refuses_a_box_without_a_lower_and_upper_corner():
    _refusal(lambda: SDFGeometry(lambda X: np.zeros(len(X)), ()), "box")


def test_sdf_geometry_refuses_corners_of_different_dimension():
    _refusal(lambda: SDFGeometry(lambda X: np.zeros(len(X)), ([0.0, 0.0], [1.0, 1.0, 1.0])), "box")


def test_the_refusals_do_not_reject_a_valid_object():
    """The other half of the guard: the validation added for the sweep must not turn a good input away."""
    assert ValidityDomain("box", {"x": (0.0, 1.0)}, source="test").variables == ["x"]
    assert DimensionlessGroup({"L": 1}).formula() == "L"
    assert Floorplan(1.0, 1.0, [], (2, 2), []).n_slots == 4
    assert YeeGrid((4, 4, 4), (0.1, 0.1, 0.1)).dt > 0
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert len(TetMesh(nodes, np.array([[0, 1, 2, 3]]), np.zeros(1, int)).volumes) == 1
    assert SDFGeometry(lambda X: np.zeros(len(X)), ([0.0] * 3, [1.0] * 3)).dim == 3
