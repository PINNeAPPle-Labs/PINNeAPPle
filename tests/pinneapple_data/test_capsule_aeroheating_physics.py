"""The physics checks, on walls small enough to work out by hand.

Every wall here is built as a `Wall` dataclass rather than read from a file, because what is under test is the
arithmetic: a square facing the flow has one normal and one value of C_p, and a ring of three nodes has a spread that
can be written down. The dataset's own numbers are checked against the real file in `test_capsule_aeroheating.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.datasets import capsule_aeroheating_physics as ph
from pinneapple_data.datasets.capsule_aeroheating import HIGH_FIDELITY, LOW_FIDELITY, Wall

GAMMA, GAS_CONSTANT = 1.4, 287.05
CP = GAMMA * GAS_CONSTANT / (GAMMA - 1.0)                  # 1004.675 J kg-1 K-1


def _square_wall(*, tilt: float = 0.0, fields=None, area=1.0, wall_temperature=300.0) -> Wall:
    """One square face, by default in the plane x = 0 (so its normal is along the flow) and of area 1 m2.

    ``tilt`` rotates it about the y axis in degrees: 0 faces the flow, 90 is parallel to it."""
    side = np.sqrt(area)
    corners = np.array([[0.0, 0.0, 0.0], [0.0, side, 0.0], [0.0, side, side], [0.0, 0.0, side]])
    if tilt:
        angle = np.deg2rad(tilt)
        rotate = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0],
                           [-np.sin(angle), 0.0, np.cos(angle)]])
        corners = corners @ rotate.T
    values = {"heat_flux": np.full(4, 1.0e5), "pressure": np.full(4, 1.0e3),
              "arc_length": np.zeros(4), "film_coefficient": np.full(4, 0.03),
              "recovery_enthalpy": np.full(4, 9.0e6)}
    values.update(fields or {})
    film, recovery = values["film_coefficient"], values["recovery_enthalpy"]
    values["heat_flux"] = film * (recovery - CP * wall_temperature)
    return Wall(row=0, member="square", fidelity=LOW_FIDELITY, points=corners,
                cells=np.array([[0, 1, 2], [0, 2, 3]]), cell_type="triangle",
                fields=values, units={name: "1" for name in values},
                parameters={"mach": 10.0, "diameter": 1.0}, parameter_units={"mach": "1", "diameter": "m"},
                scalars={"heat_load": float(values["heat_flux"][0] * area), "freestream_pressure": 100.0,
                         "freestream_density": 0.01, "freestream_velocity": 3000.0},
                scalar_units={"heat_load": "W"}, constants={"ratio_of_specific_heats": GAMMA,
                                                            "gas_constant": GAS_CONSTANT,
                                                            "wall_temperature": wall_temperature})


# ---------------------------------------------------------------------------
# the gas
# ---------------------------------------------------------------------------

def test_cp_is_the_dataset_s_own_gamma_and_R():
    assert ph.cp_of(_square_wall()) == pytest.approx(CP) == pytest.approx(1004.675)


def test_cp_refuses_a_gamma_at_or_below_one():
    wall = _square_wall()
    wall.constants["ratio_of_specific_heats"] = 1.0
    with pytest.raises(ValueError, match="gamma"):
        ph.cp_of(wall)


# ---------------------------------------------------------------------------
# the panel method, recomputed
# ---------------------------------------------------------------------------

def test_the_stagnation_coefficient_has_the_limit_of_a_modified_newtonian_method():
    """Not 2: the modified method takes the stagnation value from the shock, and the limit is 2K/gamma = 1.839."""
    assert ph.stagnation_pressure_coefficient(1.4, 1e4) == pytest.approx(1.839, abs=1e-3)
    assert ph.stagnation_pressure_coefficient(1.4, 100.0) > ph.stagnation_pressure_coefficient(1.4, 6.0)


def test_the_stagnation_coefficient_at_mach_11_is_the_one_the_dataset_uses():
    """The stagnation pressure the file reports is p_inf (1 + gamma M^2 C_p,max / 2) with this coefficient."""
    coefficient = ph.stagnation_pressure_coefficient(1.4, 11.19)
    assert coefficient == pytest.approx(1.833, abs=1e-3)


def test_the_stagnation_coefficient_refuses_a_subsonic_run():
    with pytest.raises(ValueError, match="not supersonic"):
        ph.stagnation_pressure_coefficient(1.4, 0.8)


def test_normals_are_unit_length_and_point_out_of_the_face():
    normals = ph.surface_normals(_square_wall())
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.allclose(np.abs(normals[:, 0]), 1.0)                  # the face is across the flow


def test_a_face_across_the_flow_sees_the_stagnation_pressure_and_a_face_along_it_sees_the_freestream():
    wall = _square_wall()
    across = ph.newtonian_pressure(wall)
    q_inf = 0.5 * wall.scalars["freestream_density"] * wall.scalars["freestream_velocity"] ** 2
    assert np.allclose(across["sin2_theta"], 1.0)
    assert np.allclose(across["pressure"], wall.scalars["freestream_pressure"] + q_inf * across["coefficient"])
    along = ph.newtonian_pressure(_square_wall(tilt=90.0))
    assert np.allclose(along["sin2_theta"], 0.0, atol=1e-12)
    assert np.allclose(along["pressure"], wall.scalars["freestream_pressure"])


def test_a_tilted_face_sees_sin_squared_of_its_tilt():
    tilted = ph.newtonian_pressure(_square_wall(tilt=45.0))
    assert np.allclose(tilted["sin2_theta"], 0.5, atol=1e-12)


def test_flipping_the_winding_does_not_change_the_pressure():
    wall = _square_wall()
    flipped = _square_wall()
    flipped.cells = wall.cells[:, ::-1].copy()
    assert np.allclose(ph.surface_normals(flipped)[:, 0], -ph.surface_normals(wall)[:, 0])
    assert np.allclose(ph.newtonian_pressure(flipped)["pressure"], ph.newtonian_pressure(wall)["pressure"])


def test_the_pressure_check_passes_on_a_panel_run_whose_field_is_that_pressure():
    wall = _square_wall()
    wall.fields["pressure"] = ph.newtonian_pressure(wall)["pressure"]
    assert ph.checks(wall)[1].passed is True


# ---------------------------------------------------------------------------
# the rings
# ---------------------------------------------------------------------------

def _ringed_wall(values_per_ring, nodes_per_ring=4) -> Wall:
    """Rings of nodes at increasing arc length, with the given values on each ring."""
    arcs, field = [], []
    for ring, value in enumerate(values_per_ring):
        arcs.extend([0.1 * ring] * nodes_per_ring)
        field.extend([value] * nodes_per_ring)
    arcs, field = np.array(arcs), np.array(field)
    wall = _square_wall()
    wall.fields["arc_length"] = arcs
    wall.fields["heat_flux"] = field
    return wall


def test_a_ring_that_agrees_with_itself_has_no_spread():
    spread = ph.axial_spread(_ringed_wall([1.0, 2.0, 3.0]))
    assert spread["spread"].size == 3 and np.allclose(spread["spread"], 0.0)


def test_a_ring_that_disagrees_reports_its_relative_spread():
    wall = _ringed_wall([1.0, 2.0])
    wall.fields["heat_flux"][4:] = [1.0, 2.0, 3.0, 4.0]              # the second ring spreads by std/mean = 0.447
    spread = ph.axial_spread(wall)
    assert spread["spread"][0] == pytest.approx(0.0)
    assert spread["spread"][1] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]) / 2.5)


def test_a_ring_of_fewer_than_three_nodes_is_left_out():
    wall = _ringed_wall([1.0, 2.0], nodes_per_ring=2)                 # two rings of two nodes
    assert ph.axial_spread(wall)["spread"].size == 0
    assert ph.checks(wall)[0].passed is None                         # nothing to measure is not a pass


def test_the_symmetry_check_fails_on_a_wall_that_is_not_axisymmetric():
    wall = _ringed_wall([1.0, 2.0])
    wall.fields["heat_flux"][4:] = [1.0, 2.0, 3.0, 4.0]
    result = ph.checks(wall)[0]
    assert result.check_id == "capsule.axial_symmetry" and result.passed is False


# ---------------------------------------------------------------------------
# the integral
# ---------------------------------------------------------------------------

def test_the_heat_load_is_the_field_times_the_area():
    wall = _square_wall(area=2.5)
    integral = ph.heat_load(wall)
    assert integral["area"] == pytest.approx(2.5)
    assert integral["integral"] == pytest.approx(float(wall.fields["heat_flux"][0]) * 2.5)


def test_the_heat_load_check_fails_when_the_scalar_and_the_field_disagree():
    wall = _square_wall()
    wall.scalars["heat_load"] = 2.0 * float(wall.fields["heat_flux"][0] * 1.0)
    result = ph.checks(wall)[2]
    assert result.check_id == "capsule.heat_load" and result.passed is False


def test_a_quad_is_integrated_by_fanning_it_into_triangles():
    wall = _square_wall(area=1.0)
    wall.cells = np.array([[0, 1, 2, 3]])
    wall.cell_type = "quad"
    assert ph.cell_areas(wall)[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the hotter wall
# ---------------------------------------------------------------------------

def test_the_hot_wall_formula_reduces_to_the_field_at_the_wall_temperature():
    wall = _square_wall()
    assert np.allclose(ph.wall_heat_flux(wall, 300.0), wall.fields["heat_flux"])


def test_the_heating_falls_to_zero_at_the_adiabatic_wall_temperature():
    wall = _square_wall()
    adiabatic = float(wall.fields["recovery_enthalpy"][0]) / CP
    assert np.allclose(ph.wall_heat_flux(wall, adiabatic), 0.0, atol=1e-6)


def test_the_radiative_temperature_is_below_the_adiabatic_one_and_above_the_wall():
    wall = _square_wall(fields={"recovery_enthalpy": np.full(4, 9.0e6), "film_coefficient": np.full(4, 0.03)})
    temperature = ph.radiative_equilibrium_temperature(wall, emissivity=0.85)
    adiabatic = 9.0e6 / CP
    assert np.all(temperature > 300.0) and np.all(temperature < adiabatic)


def test_the_radiative_temperature_solves_the_equation_it_claims_to():
    wall = _square_wall(fields={"recovery_enthalpy": np.full(4, 9.0e6), "film_coefficient": np.full(4, 0.03)})
    temperature = ph.radiative_equilibrium_temperature(wall, emissivity=0.85)
    residual = (ph.wall_heat_flux(wall, temperature)
                - 0.85 * ph.STEFAN_BOLTZMANN * temperature ** 4)
    assert np.allclose(residual, 0.0, atol=1e-6 * 1e5)


def test_a_blacker_wall_runs_cooler():
    wall = _square_wall(fields={"recovery_enthalpy": np.full(4, 9.0e6), "film_coefficient": np.full(4, 0.03)})
    grey = ph.radiative_equilibrium_temperature(wall, emissivity=0.5)
    black = ph.radiative_equilibrium_temperature(wall, emissivity=1.0)
    assert np.all(black < grey)


def test_an_emissivity_out_of_range_is_refused():
    wall = _square_wall()
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="emissivity"):
            ph.radiative_equilibrium_temperature(wall, emissivity=bad)


def test_the_closure_is_an_identity_and_is_reported_as_machine_zero():
    assert ph.closure_identity_residual(_square_wall()) < 1e-12


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def test_the_four_checks_and_what_each_one_answers_for():
    wall = _square_wall()
    wall.fields["pressure"] = ph.newtonian_pressure(wall)["pressure"]
    results = {r.check_id: r for r in ph.checks(wall)}
    assert set(results) == {"capsule.axial_symmetry", "capsule.newtonian_pressure", "capsule.heat_load",
                            "capsule.radiative_equilibrium"}
    assert results["capsule.newtonian_pressure"].passed is True       # the panel method's own field
    assert results["capsule.radiative_equilibrium"].passed is None     # reported, not checked
    wall.fidelity = HIGH_FIDELITY
    assert {r.check_id: r for r in ph.checks(wall)}["capsule.newtonian_pressure"].passed is None


def test_the_checks_are_written_into_a_bundle_s_verification():
    from pinneapple_data.schema import PhysicsDataBundle
    from pinneapple_data.schema.models import Verification

    wall = _square_wall()
    wall.fields["pressure"] = ph.newtonian_pressure(wall)["pressure"]
    bundle = PhysicsDataBundle.from_arrays({}, fields=[], axes=[], source="a square wall", license="CC-BY-4.0")
    assert bundle.manifest.verification.checks == []
    ph.record(bundle, ph.checks(wall))
    assert [c.check_id for c in bundle.manifest.verification.checks][:1] == ["capsule.axial_symmetry"]
    assert isinstance(bundle.manifest.verification, Verification)
    assert bundle.manifest.verification.passed is True
    assert "capsule.radiative_equilibrium" not in bundle.manifest.verification.required   # it ran; it does not judge


def test_a_summary_carries_every_number_the_checks_produce():
    wall = _square_wall()
    wall.fields["pressure"] = ph.newtonian_pressure(wall)["pressure"]
    summary = ph.summary(wall).as_dict()
    assert summary["member"] == "square" and summary["fidelity"] == LOW_FIDELITY
    assert summary["wall_area"] == pytest.approx(1.0)
    assert summary["pressure_median_error"] < 1e-12
    assert summary["closure_residual"] < 1e-12
    assert summary["radiative_temperature_max"] > 300.0
