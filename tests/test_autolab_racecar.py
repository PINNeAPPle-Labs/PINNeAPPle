"""pinneapple_autolab.domains.racecar without OpenFOAM: design checks, field points and regions, the mission."""
import numpy as np
import pytest

from pinneapple_autolab.domains.racecar import (
    BASELINE,
    MESHES,
    VARIABLES,
    RaceCarAero,
    flagship_mission,
    force_convergence,
)


@pytest.fixture(scope="module")
def sim():
    return RaceCarAero()


def test_baseline_passes_every_check(sim):
    assert sim.check(BASELINE) == []


def test_checks_name_the_problem(sim):
    low = sim.check({**BASELINE, "ride_height": 0.015})
    assert any("floor: 15.0 mm above the road" in s for s in low)
    clash = sim.check({**BASELINE, "front_wing_aoa": 9.0, "front_flap_aoa": 12.0})
    assert any("front wing: main plane and flap intersect" in s for s in clash)


def test_bounds_keep_wing_elements_apart(sim):
    corners = [{**BASELINE, "front_wing_aoa": 5.0, "front_flap_aoa": 12.0},
               {**BASELINE, "rear_wing_aoa": 7.5, "rear_flap_aoa": 16.0}]
    assert all(sim.check(p) == [] for p in corners)


def test_field_points_have_a_fixed_count_and_regions(sim):
    a = sim.field_points(BASELINE)
    b = sim.field_points({**BASELINE, "diffuser_angle": 18.0, "rear_flap_aoa": 16.0})
    assert a.shape == b.shape == (sim.n_points, 3) and np.all(a[:, 1] >= -1e-9)
    assert sim.regions["diffuser"].sum() > 0 and not np.any(sim.regions["diffuser"] & ~sim.regions["floor"])
    assert not np.allclose(a, b)


def test_mission_and_meshes():
    m = flagship_mission(24, 3)
    assert [v.name for v in m.variables] == [v.name for v in VARIABLES] and m.budget.max_runs == 24
    assert {c.name for c in m.constraints} == {"downforce_N", "balance_front"} and not m.resolved
    assert "campaign" in MESHES and MESHES["campaign"]["body_levels"]["front_wing"] == (5, 6)
    with pytest.raises(ValueError):
        RaceCarAero(resolution="ultra")


def test_force_convergence():
    h = [{"iteration": i, "drag_N": 100.0 + (i == 300) * 2, "downforce_N": 200.0} for i in (100, 200, 300)]
    c = force_convergence(h)
    assert c["drag_N_rel_change"] == pytest.approx(2 / 102) and c["over_iterations"] == 200
