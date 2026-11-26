"""The suspension corner: the closed loop closes, the tube is the tube, the finite elements agree with the beam, the
bushes scale as they must, and the loop finds a lighter feasible design every time.

The references here are the ones the models claim to reproduce: the linkage's rigid lengths (an exact invariant of the
closed loop), the polygonal area of the tube mesh (exact for the discretisation), beam theory (``M c / I`` and
``P L^3 / 3 E I``), Euler's buckling load, and the linear scaling of a compliance with the bush rate — which is what the
first-order model says must hold.
"""
from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest

from pinneapple_design.suspension import (
    BASELINE_DESIGN,
    Design,
    TubeSection,
    arm_mass_kg,
    calculix,
    compliance_modes,
    corner_load_cases,
    evaluate,
    fem_tube,
    front_corner,
    kinematic_sweep,
    mode_responses,
    pose,
    pose_from_angles,
    suspension_design,
    suspension_study,
    suspension_verify,
    tube_mesh,
    wheel_compliance,
)
from pinneapple_design.suspension.study import X_OUTPUTS
from pinneapple_design.suspension.verification import (
    arm_models,
    cantilever,
    kinematics_agreement,
    parallelogram,
    rigid_bushes,
    run_all,
    summary,
)

#: the cases that need the external solver: the ``gmsh`` Python module and ``ccx`` (native or through Docker)
HAS_EXTERNAL = importlib.util.find_spec("gmsh") is not None and calculix.available()


@pytest.fixture(scope="module")
def spec():
    return front_corner()


def test_the_loop_closes_over_the_whole_travel(spec):
    sweep = kinematic_sweep(spec, (-60.0, 60.0, 6.0))
    assert sweep.residual_m < 1e-9
    # the wheel really is at the travel asked for: the constraint the solver enforces
    s = spec
    for travel_mm in (-50.0, -20.0, 0.0, 25.0, 50.0):
        p = pose(s, travel_mm * 1e-3)
        assert p.wheel_centre[2] == pytest.approx(s.wheel_centre[2] + travel_mm * 1e-3, abs=1e-9)


def test_rigid_lengths_are_invariant(spec):
    for travel_mm in (-55.0, -30.0, 0.0, 30.0, 55.0):
        p = pose(spec, travel_mm * 1e-3)
        assert np.linalg.norm(p.upper_ball - p.lower_ball) == pytest.approx(spec.kingpin_length_m, abs=1e-9)
        assert np.linalg.norm(p.tierod_outer - spec.tierod_inner) == pytest.approx(spec.tierod_length_m, abs=1e-9)
        assert np.linalg.norm(p.lower_ball - spec.lca_front) == pytest.approx(
            np.linalg.norm(spec.lca_outer - spec.lca_front), abs=1e-9)
        assert np.linalg.norm(p.upper_ball - spec.uca_rear) == pytest.approx(
            np.linalg.norm(spec.uca_outer - spec.uca_rear), abs=1e-9)


def test_camber_is_gained_in_bump_and_lost_in_rebound(spec):
    bump = pose(spec, 0.040).camber_deg
    rest = pose(spec, 0.0).camber_deg
    rebound = pose(spec, -0.040).camber_deg
    assert rest == pytest.approx(-1.0, abs=0.05)          # the reference corner carries 1 deg of static camber
    assert bump < rest < rebound                          # negative = the top of the wheel leaning inboard
    assert -1.2 < kinematic_sweep(spec).summary()["camber_gain_deg_per_50mm"] < -0.2


def test_bump_steer_crosses_zero_when_the_tie_rod_moves_up_or_down(spec):
    """The tie rod's inner point is the bump-steer adjuster: raising it changes the sign of the steer in bump."""
    from dataclasses import replace

    values = []
    for dz in (-0.010, 0.0, 0.010):
        moved = replace(spec, tierod_inner=spec.tierod_inner + np.array([0.0, 0.0, dz]))
        values.append(kinematic_sweep(moved).summary()["bump_steer_deg_per_50mm"])
    assert values[0] > values[1] > values[2]
    assert values[0] > 0.0 > values[2]


def test_the_tube_mesh_is_the_tube(spec):
    section = TubeSection(0.032, 0.0035)
    n = 24
    mesh = tube_mesh(0.35, section, n_theta=n, n_radial=2, n_len=8)
    # the polygonal (inscribed, 24-chord) area x length: exact for this discretisation
    exact = 0.5 * n * math.sin(2.0 * math.pi / n) * (section.r_out ** 2 - section.r_in ** 2) * 0.35
    v = 0.0
    p = [mesh.nodes[mesh.elements[:, k]] for k in range(8)]
    for i, j, k, m in ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)):
        v += float(np.einsum("ij,ij->i", p[j] - p[i], np.cross(p[k] - p[i], p[m] - p[i])).sum() / 6.0)
    assert v == pytest.approx(exact, rel=1e-9)
    assert mesh.nodes[:, 0].max() == pytest.approx(0.35, abs=1e-12)      # the tip ring sits at the length


def test_finite_elements_reproduce_the_cantilever(spec):
    """A clamped tube under a tip load: ``sigma = P L c / I`` at the root and ``P L^3 / 3 E I`` at the tip."""
    section = TubeSection(0.032, 0.0035)
    L, P = 0.32, 1500.0
    out = fem_tube(spec, section, L, (0.0, 0.0, -P), n_theta=24, n_radial=2, n_len=20)
    assert out["ratio"] == pytest.approx(1.0, abs=0.05)
    assert out["tip_deflection_m"] == pytest.approx(out["tip_deflection_beam_m"], rel=0.05)


def test_euler_buckling_of_the_section(spec):
    from pinneapple_design.suspension import arm_check
    from pinneapple_design.suspension.loads import corner_load_cases
    section = TubeSection(0.032, 0.0035)
    p = pose(spec, 0.0)
    chk = arm_check(spec, p, corner_load_cases(spec)["bump 3 g"], section)
    leg = chk["leg_length_m"]
    euler = math.pi ** 2 * spec.elastic_modulus_pa * section.inertia_m4 / leg ** 2
    assert chk["euler_load_n"] == pytest.approx(euler, rel=1e-9)


def test_compliance_scales_with_the_bush_rate(spec):
    """The first-order model is linear in the bushes: twice the rate, half the steer — to the accuracy of the modes."""
    soft = wheel_compliance(spec.with_bushings(800e3))["toe_deg_per_kn"]
    stiff = wheel_compliance(spec.with_bushings(1600e3))["toe_deg_per_kn"]
    assert soft / stiff == pytest.approx(2.0, rel=0.02)


def test_rigid_bushes_give_no_compliance(spec):
    stiff = spec.with_bushings(1e9)
    assert abs(wheel_compliance(stiff)["toe_deg_per_kn"]) < 1e-3
    modes = mode_responses(spec)
    assert set(modes) == {m.name for m in compliance_modes(spec)}


def test_the_load_cases_are_ordered(spec):
    cases = corner_load_cases(spec)
    assert cases["bump 3 g"][2] == pytest.approx(3.0 * spec.corner_mass_kg * 9.80665, rel=1e-9)
    assert cases["cornering 1.1 g"][1] > 0.0
    assert cases["braking 1.2 g"][0] < 0.0                     # a braking force points rearward


def test_the_design_loop_returns_a_lighter_feasible_corner(spec):
    result = suspension_design(spec, steps=14)
    assert result.best["feasible"], result.best["violations"]
    assert result.best["mass_kg"] < result.baseline["mass_kg"]
    for name, excess in result.best["violations"].items():
        assert excess == 0.0, name
    # deterministic: the same start and the same number of rounds give the same design
    again = suspension_design(spec, steps=14)
    assert again.design.as_dict() == result.design.as_dict()


def test_evaluate_reports_every_constraint(spec):
    rec = evaluate(spec, Design())
    assert set(rec["measured"]) == {"static_margin", "buckling_factor", "camber_gain_deg_per_50mm",
                                    "bump_steer_deg_per_50mm", "toe_compliance_deg_per_kn", "wheel_rate_n_per_mm"}
    assert rec["mass_kg"] == pytest.approx(arm_mass_kg(rec["spec"]) + rec["bushing_mass_kg"])


def test_the_baseline_is_the_nominal_tube(spec):
    assert BASELINE_DESIGN.od_mm == pytest.approx(spec.tube_od_m * 1e3)
    assert BASELINE_DESIGN.wall_mm == pytest.approx(spec.tube_wall_m * 1e3)


def test_the_study_writes_its_files_and_passes_its_checks(tmp_path):
    out = suspension_study(out_dir=str(tmp_path), budget=14, mesh="coarse")
    assert all(c["status"] == "pass" for c in out["checks"]), [c for c in out["checks"] if c["status"] != "pass"]
    for name in ("scene.glb", "scene.json", "corner.glb", "wheels_in_bump.png", "compliance_vs_bush.png",
                 "design_search.png", "arm_stress.png", "suspension_design.json"):
        assert os.path.isfile(tmp_path / name), name
    assert out["verdict"].startswith("the design meets")


def test_the_study_returns_its_numbers_as_the_record_the_deck_reads(tmp_path):
    """``outputs`` is the flat record a presentation draws: every name is one the module declares with a unit, the
    curves it carries are the sweep's own arrays, and the verdict is the checks' — nothing here is typed in."""
    out = suspension_study(out_dir=str(tmp_path), budget=14, mesh="coarse")
    declared = {e["name"]: e["unit"] for e in X_OUTPUTS["suspension_study"]}
    assert set(out["outputs"]) == set(declared), sorted(set(out["outputs"]) ^ set(declared))
    scalars = {k: v for k, v in out["outputs"].items() if not isinstance(v, list)}
    assert all(isinstance(v, float) for v in scalars.values())
    curves = {k: v for k, v in out["outputs"].items() if isinstance(v, list)}
    assert set(curves) == {"position", "camber_deg", "toe_deg"}
    n = len(curves["position"])
    assert n >= 5 and all(len(v) == n for v in curves.values())
    assert curves["position"] == sorted(curves["position"])       # monotonic: it can be the charts' abscissa
    static = out["sweep_summary"]["static_camber_deg"]
    assert min(curves["camber_deg"]) <= static <= max(curves["camber_deg"])   # the curve holds the design height
    assert out["outputs"]["mass_kg"] == pytest.approx(out["metrics"]["mass_kg"])
    assert out["decision"]["verdict"] == ("OK" if all(c["status"] == "pass" for c in out["checks"]) else "NOT OK")
    assert out["decision"]["reasons"]
    again = suspension_verify(out_dir=str(tmp_path), budget=14, mesh="fine")
    assert set(again["outputs"]) == {e["name"] for e in X_OUTPUTS["suspension_verify"]}
    assert again["decision"]["verdict"] in ("OK", "NOT OK")


# -------------------------------------------------------------------------------------- cases with a known answer
def test_the_two_kinematics_formulations_agree(spec):
    """The closure unknowns and the joint angles are different parameterisations of one mechanism: what separates them
    is numerical error, so it must be at the level of the solvers' tolerances, not of the millimetre."""
    agree = kinematics_agreement(design=spec, travels_mm=(-50.0, -25.0, 0.0, 25.0, 50.0))
    assert agree["worst_camber_deg"] < 1e-6
    assert agree["worst_toe_deg"] < 1e-6
    assert agree["worst_wheel_centre_m"] < 1e-9
    for row in agree["rows"]:                       # each one measured, not only the worst
        assert row["d_camber_deg"] < 1e-6 and row["d_toe_deg"] < 1e-6
    # and the joint-angle solver is a mechanism in its own right: it lands on the travel and keeps the rigid lengths
    p = pose(spec, 0.030)
    q = pose_from_angles(spec, 0.030, seed=(p.theta, p.phi, p.psi))
    assert q.wheel_centre[2] == pytest.approx(spec.wheel_centre[2] + 0.030, abs=1e-9)
    assert np.linalg.norm(q.upper_ball - q.lower_ball) == pytest.approx(spec.kingpin_length_m, abs=1e-9)
    assert np.linalg.norm(q.tierod_outer - spec.tierod_inner) == pytest.approx(spec.tierod_length_m, abs=1e-9)


def test_a_parallelogram_gains_no_camber(spec):
    """Equal, parallel wishbones: the upright translates and the camber cannot change — a closed-form zero.

    The tie rod is the one part that can spoil it, so the case points it square to the travel (first-order effect gone)
    and reports what the second order leaves."""
    par = parallelogram()
    assert abs(par["tie_rod_perpendicular"]) < 1e-12
    assert abs(par["camber_gain_deg_per_50mm"]) < 1e-3
    as_built = abs(kinematic_sweep(spec).summary()["camber_gain_deg_per_50mm"])
    assert as_built > 1e3 * abs(par["camber_gain_deg_per_50mm"])     # four orders: the real corner against the zero
    assert abs(par["camber_change_full_bump_deg"]) < 0.05            # the tie rod's second-order coupling, measured


def test_the_bushes_stop_mattering_when_they_are_rigid(spec):
    """The first-order model is linear in the bush rate, so a thousand times the rate is a thousandth of the steer."""
    rb = rigid_bushes(design=spec)
    assert rb["measured_ratio"] == pytest.approx(rb["rate_ratio"], rel=0.01)
    assert rb["toe_deg_per_kn_stiff"] < 1e-3
    assert rb["toe_deg_per_kn_nominal"] > 0.1


def test_the_verification_report_measures_every_case_it_can(spec):
    """What runs without the external solver: three cases, each against a reference of its own, in its own unit."""
    report = run_all(external=False)
    rows = {r["case"]: r for r in summary(report)}
    assert {"two kinematics formulations", "parallelogram, first order", "parallelogram, full bump",
            "rigid bushes"} <= set(rows)
    assert all(r["reference"] == 0.0 for r in rows.values())
    assert report["cantilever"] == {"skipped": "skipped: external=False"}
    assert rows["cantilever"]["detail"] == "skipped: external=False"


@pytest.mark.skipif(not HAS_EXTERNAL, reason="CalculiX (ccx) or the gmsh module is not installed")
def test_the_tube_in_an_external_solver_matches_timoshenko(tmp_path):
    """gmsh builds the B-rep, ccx solves it: the pipeline is right when a case whose answer Timoshenko wrote down
    comes back within 2 % of it."""
    out = cantilever(workdir=str(tmp_path / "tube"))
    assert out["error_rel"] == pytest.approx(0.0, abs=0.02)
    assert out["bending_share"] > 0.9                  # bending dominates: the case is a cantilever, not a shear test
    assert out["mass_kg"] == pytest.approx(0.799, rel=0.02)     # the B-rep's volume, not the mesh's: it does not move


@pytest.mark.skipif(not HAS_EXTERNAL, reason="CalculiX (ccx) or the gmsh module is not installed")
def test_the_leg_solved_by_three_models_agrees(tmp_path):
    """One leg, one sizing case: beam theory, this library's hexahedra and CalculiX, within 5 % of each other."""
    out = arm_models(workdir=str(tmp_path / "leg"))
    assert abs(out["hex_to_beam"] - 1.0) < 0.05
    assert abs(out["calculix_to_beam"] - 1.0) < 0.05
    assert out["spread_rel"] < 0.05
    assert out["sigma_pa"]["beam"] > 0.0
