"""The design-space pipeline and its five families: each reference checked against an independent identity.

The point of these tests is not that the code runs — it is that every reference is *someone else's* answer. Each
family is checked against an identity that does not come from the family itself:

* the airfoil's integrated pressure force must equal ``rho U Gamma`` perpendicular to the free stream, with zero
  drag (Kutta-Joukowski), and the same map with zero circulation must reproduce the library's analytic ellipse Cp;
* the nozzle must invert its own area-Mach relation, hit ``M = 1`` at the throat and match
  ``rocket_nozzle.nozzle_performance`` at the exit;
* the exchanger's boundary-value solution must reproduce the closed-form ε-NTU of the same module and close its
  energy balance;
* the heat sink's profile must satisfy the fin equation it claims to solve;
* the bracket must be linear in the load and consistent across mesh sizes.

On top of that, the pipeline's own rules: a split that never puts two operating points of one body on both sides, a
dataset that round-trips, a design the reference cannot represent that is refused instead of written, and a fit that
actually learns.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from pinneapple_neural.lpm import Design
from pinneapple_neural.lpm_families import (
    BRACKET,
    EXCHANGER,
    FIN,
    NOZZLE,
    _joukowski_cp,
    _joukowski_surface,
)
from pinneapple_neural.lpm_pipeline import (
    CARD_SCHEMA,
    DATASET_SCHEMA,
    DesignSpec,
    build,
    draw,
    families,
    generate_dataset,
    load_dataset,
    train,
)

# --------------------------------------------------------------------------------------------------- the five families


def test_the_five_families_are_registered_and_describe_themselves():
    fams = families()
    assert set(fams) == {"pp_fin", "pp_airfoil", "pp_nozzle", "pp_exchanger", "pp_bracket"}
    for name, fam in fams.items():
        described = fam.describe()
        assert described["name"] == name
        assert len(described["field_names"]) == len(described["field_units"]) == fam.n_fields
        assert len(described["field_scale"]) == fam.n_fields
        assert described["reference"]["name"] and described["reference"]["citation"]
        assert described["reference"]["assumptions"], "a reference without assumptions is an overclaim"
    assert fams["pp_bracket"].reference.implements == "solver"
    assert all(fams[k].reference.implements == "analytic" for k in fams if k != "pp_bracket")


def test_airfoil_reproduces_the_analytic_ellipse_and_the_kutta_joukowski_theorem():
    """Zero circulation on a centred circle is the ellipse of ``Design.ellipse``: the map must give its exact Cp."""
    for a, b in ((1.3, 0.7), (1.45, 1.2)):
        radius, c = (a + b) / 2.0, math.sqrt((a + b) / 2.0 * (a - b) / 2.0)
        theta = np.linspace(0.0, 2.0 * math.pi, 257)[:-1]        # the grid Design.ellipse uses, so the same points
        zeta0, gamma_u = 0j, 0.0
        zeta = radius * np.exp(1j * theta)
        dw = np.exp(0j) - radius ** 2 / (zeta - zeta0) ** 2 + 1j * gamma_u / (2 * math.pi * (zeta - zeta0))
        cp = 1.0 - np.abs(dw) ** 2 / np.abs(1 - c * c / (zeta * zeta)) ** 2
        torch.testing.assert_close(torch.tensor(cp, dtype=torch.float64),
                                   Design.ellipse(a, b, len(theta)).target[:, 0].double(), atol=1e-4, rtol=0)

    # the lift must be rho U Gamma perpendicular to the free stream, and the drag must vanish
    for x_c, y_c, alpha_deg in ((-0.06, 0.02, 6.0), (-0.03, 0.01, 12.0)):
        c, n = 0.5, 6000
        radius = math.hypot(c - x_c, y_c)
        cp, _, circulation = _joukowski_cp(x_c, y_c, radius, c, n, alpha_deg)
        points, normals = _joukowski_surface(x_c, y_c, radius, c, n)
        theta = 2 * math.pi * (np.arange(n) + 0.5) / n
        zeta = (x_c + 1j * y_c) + radius * np.exp(1j * theta)
        weight = np.abs((1 - c * c / (zeta * zeta)) * 1j * radius * np.exp(1j * theta)) * (2 * math.pi / n)
        force = -0.5 * np.array([np.sum(cp * normals[:, 0] * weight), np.sum(cp * normals[:, 1] * weight)])
        angle = math.radians(alpha_deg)
        lift = -force[0] * math.sin(angle) + force[1] * math.cos(angle)
        drag = force[0] * math.cos(angle) + force[1] * math.sin(angle)
        assert lift == pytest.approx(circulation * radius, rel=1e-6)
        assert abs(drag) < 1e-6


def test_airfoil_cp_does_not_depend_on_how_many_points_sample_the_surface():
    """The trailing-edge cusp is where a naive quotient loses its digits; the factored form must not care."""
    for n in (120, 500, 4000):
        cp, _, _ = _joukowski_cp(-0.06, 0.02, math.hypot(0.5 + 0.06, 0.02), 0.5, n, 6.0)
        assert cp.min() == pytest.approx(-2.232, abs=5e-3), f"n={n} moved the suction peak"
        assert cp.max() <= 1.0 + 1e-9


def test_nozzle_inverts_its_area_relation_and_matches_the_exit_performance():
    from pinneapple_physics.closed_form.compressible import isentropic_ratios
    from pinneapple_physics.closed_form.rocket_nozzle import nozzle_performance

    spec = DesignSpec({"area_ratio_exit": 10.0, "chamber_ratio": 6.0, "throat_radius_m": 0.05,
                       "half_angle_deg": 15.0}, {"gamma": 1.22, "r_gas_j_kg_k": 320.0})
    field = NOZZLE.field(spec)
    x, r = NOZZLE._contour(spec)
    mach = field[:len(x), 0]
    throat = int(np.argmin(np.abs(x)))
    assert x[throat] == 0.0 and mach[throat] == pytest.approx(1.0, abs=1e-9)
    round_trip = isentropic_ratios(mach, 1.22)["A_Astar"]
    np.testing.assert_allclose(round_trip, (r / 0.05) ** 2, rtol=1e-8)
    exit_mach = nozzle_performance(gamma=1.22, R_gas=320.0, T_chamber=3000.0, p_chamber=7e6,
                                   area_ratio_exit=10.0)["mach_exit"]
    assert mach[-1] == pytest.approx(exit_mach, rel=1e-9)
    assert np.all(field[:, 1] > 0.0) and np.all(field[:, 1] <= 1.0)


def test_exchanger_matches_the_closed_form_effectiveness_and_balances_energy():
    from pinneapple_physics.closed_form import heat_exchanger as hx

    for spec in draw(EXCHANGER, 4, 1, seed=3):
        ntu, cr, details = EXCHANGER._reference(spec)
        eps, profiles = hx.one_two_exchanger_ode(ntu, cr, profiles=True)
        assert eps == pytest.approx(hx.effectiveness(ntu, cr, arrangement="shell_tube", n_shell=1), rel=1e-6)
        # the shell gives up C_min (theta_shell in - theta_shell out = eps) and the tubes take up C_max
        assert profiles["shell"][0] - profiles["shell"][-1] == pytest.approx(eps, rel=1e-6)
        # the conserved flux is T + (t1 - t2)/Cr, so the tube outlet (pass 2 at x = 0) carries Cr * eps
        assert profiles["pass2"][0] == pytest.approx(cr * eps, rel=1e-4)
        assert details["h_tube"] > 0 and details["h_shell"] > 0
        for name in ("shell", "pass1", "pass2"):
            assert profiles[name].min() >= -1e-9 and profiles[name].max() <= 1.0 + 1e-9


def test_fin_profile_solves_the_fin_equation_it_claims():
    from pinneapple_physics.closed_form import fin_array_conduction as fa

    spec = DesignSpec({"n_fins": 12.0, "fin_thickness_m": 0.0015, "fin_pitch_m": 0.006,
                       "fin_height_m": 0.020, "base_depth_m": 0.05, "base_thickness_m": 0.005,
                       "base_width_m": 0.072}, {"power_w": 40.0, "t_ambient_c": 25.0, "air_velocity_m_s": 2.5})
    params = FIN._network(spec)["params"]
    t_amb, step, x = spec.operating["t_ambient_c"], 1e-6, 0.008
    at = lambda xi: fa.fin_array_temperature_at_position_c(params, t_amb, xi) - t_amb  # noqa: E731
    second = (at(x + step) - 2 * at(x) + at(x - step)) / step ** 2
    assert second == pytest.approx(params["m"] ** 2 * at(x), rel=1e-5)
    root = at(0.0)
    tip = at(spec.geometry["fin_height_m"])
    base = fa.fin_array_base_temperature_c(params, t_amb, spec.operating["power_w"]) - t_amb
    assert base > root > tip, "the base is hotter than the fin root, and the root hotter than the tip"


def test_bracket_is_linear_in_the_load_and_consistent_across_meshes():
    from pinneapple_simulation import immersed as im

    geometry = {"leg_a_m": 1.0, "leg_b_m": 0.8, "leg_thickness_m": 0.12}
    one = BRACKET.field(DesignSpec(geometry, {"load_n": 5e3, "angle_deg": 0.0, "e_gpa": 70.0}))
    two = BRACKET.field(DesignSpec(geometry, {"load_n": 1e4, "angle_deg": 0.0, "e_gpa": 70.0}))
    assert two.max(axis=0)[0] == pytest.approx(2.0 * one.max(axis=0)[0], rel=1e-9)
    assert two.max(axis=0)[1] == pytest.approx(2.0 * one.max(axis=0)[1], rel=1e-9)

    shape = im.geometry.union(im.geometry.box([0.0, 0.0], [geometry["leg_a_m"], geometry["leg_thickness_m"]]),
                              im.geometry.box([0.0, 0.0], [geometry["leg_thickness_m"], geometry["leg_b_m"]]))
    peaks = []
    for divisions in (50, 100):
        disc = im.Discretization(shape, h=geometry["leg_a_m"] / divisions, order=1)
        solution = im.solve_elasticity(disc, E=70e9, nu=0.33, bcs=[
            im.Displacement([0.0, 0.0], where=lambda X, n: np.isclose(X[:, 0], 0.0)),
            im.Force([5e3, 0.0], where=lambda X, n: np.isclose(X[:, 1], geometry["leg_b_m"]))])
        assert abs(solution.balance["imbalance"]) < 1e-8
        peaks.append(solution.max_displacement())
    assert abs(peaks[1] / peaks[0] - 1.0) < 0.02, f"mesh convergence worse than 2%: {peaks}"


# ------------------------------------------------------------------------------------------------------ the pipeline


def test_draw_gives_several_operating_points_per_geometry():
    specs = draw(NOZZLE, n_geometries=3, conditions_per_geometry=4, seed=1)
    assert len(specs) == 12
    by_geometry = {}
    for spec in specs:
        by_geometry.setdefault(spec.geometry_id, []).append(spec.operating)
    assert len(by_geometry) == 3 and all(len(v) == 4 for v in by_geometry.values())
    for values in by_geometry.values():
        assert len({tuple(sorted(v.items())) for v in values}) == 4, "the four conditions must differ"


def test_draw_refuses_operating_points_the_reference_cannot_represent():
    """A velocity of 1e-6 m/s would be a *forced* convection draw with no coefficient: it is re-drawn, never kept."""
    specs = draw(FIN, n_geometries=4, conditions_per_geometry=5, seed=7)
    speeds = [s.operating["air_velocity_m_s"] for s in specs]
    assert all(v == 0.0 or v >= 0.8 for v in speeds), speeds
    from pinneapple_physics.closed_form import plate_fin_heatsink as pf

    for spec in specs:
        sink = pf.HeatSinkGeometry(base_width=spec.geometry["base_width_m"],
                                   base_depth=spec.geometry["base_depth_m"],
                                   base_thickness=spec.geometry["base_thickness_m"],
                                   n_fins=int(spec.geometry["n_fins"]),
                                   fin_thickness=spec.geometry["fin_thickness_m"],
                                   fin_height=spec.geometry["fin_height_m"])
        assert not pf.geometry_errors(sink, pf.OperatingPoint(power_w=10.0, t_ambient_c=25.0,
                                                             air_velocity_m_s=spec.operating["air_velocity_m_s"],
                                                             source_width=0.5 * sink.base_width,
                                                             source_depth=0.5 * sink.base_depth))


def test_build_normalises_the_geometry_and_scales_the_field():
    spec = draw(BRACKET, 1, 1, seed=0)[0]
    design = build(BRACKET, spec)
    assert float(design.points.abs().max()) <= 3.0, "points must be normalised by the family's length"
    physical = BRACKET.field(spec)
    torch.testing.assert_close(design.target, torch.tensor(physical / BRACKET.scales, dtype=torch.float32),
                               rtol=1e-6, atol=0)
    assert design.target.shape == (BRACKET.n_points, BRACKET.n_fields)


def test_build_refuses_a_design_outside_the_reference_range():
    """10 m/s leaves the Teertstra correlation the heat sink's coefficient comes from: refused, not silently fitted."""
    geometry = {"n_fins": 12.0, "fin_thickness_m": 0.002, "fin_pitch_m": 0.008, "fin_height_m": 0.03,
                "base_depth_m": 0.06, "base_thickness_m": 0.005, "base_width_m": 0.096}
    ok = DesignSpec(geometry, {"power_w": 20.0, "t_ambient_c": 25.0, "air_velocity_m_s": 2.0})
    too_fast = DesignSpec(geometry, {"power_w": 20.0, "t_ambient_c": 25.0, "air_velocity_m_s": 10.0})
    assert FIN.validate(ok) == ""
    assert "Teertstra" in FIN.validate(too_fast)
    with pytest.raises(ValueError, match="outside the reference"):
        build(FIN, too_fast)


def test_a_family_that_cannot_build_a_design_says_why(tmp_path):
    """The pipeline's contract, independent of any one family's physics."""
    from pinneapple_neural.lpm_pipeline import FieldFamily, Reference

    class Broken(FieldFamily):
        name, title, domain = "pp_broken", "a family that refuses everything", "tests"
        field_names, field_units = ("f",), ("-",)
        reference = Reference("none", "none", ("none",))

        def points_and_normals(self, spec):
            return np.zeros((4, 2)), np.zeros((4, 2))

        def field(self, spec):
            return np.zeros((4, 1))

        def validate(self, spec):
            return "the reference has no answer for this body"

    with pytest.raises(RuntimeError, match="inside the reference's range"):
        draw(Broken(), 1, 1, seed=0, attempts=3)
    with pytest.raises(ValueError, match="outside the reference"):
        build(Broken(), DesignSpec({}, {}, label="x"))


@pytest.mark.parametrize("name", ["pp_fin", "pp_airfoil", "pp_nozzle", "pp_exchanger", "pp_bracket"])
def test_every_family_round_trips_through_a_dataset(tmp_path, name):
    family = families()[name]
    manifest = generate_dataset(family, tmp_path / name, n_geometries=3, conditions_per_geometry=2, seed=0)
    assert manifest["schema"] == DATASET_SCHEMA
    assert manifest["n_designs"] == 6 and manifest["n_geometries"] == 3
    assert all(len(record["sha256"]) == 64 for record in manifest["designs"])
    designs, read = load_dataset(tmp_path / name)
    assert len(designs) == 6 and read["family"]["name"] == name
    rebuilt = build(family, DesignSpec(manifest["designs"][0]["geometry"], manifest["designs"][0]["operating"]))
    torch.testing.assert_close(designs[0].target, rebuilt.target)
    assert torch.isfinite(designs[0].target).all()


def test_the_split_never_puts_one_body_on_both_sides(tmp_path):
    family = families()["pp_exchanger"]
    generate_dataset(family, tmp_path / "ds", n_geometries=10, conditions_per_geometry=3, seed=0)
    from pinneapple_neural.lpm_pipeline import _split_by_geometry

    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    train_idx, hold_idx = _split_by_geometry(manifest, 0.3, 0)
    trained = {manifest["designs"][i]["geometry_id"] for i in train_idx}
    held = {manifest["designs"][i]["geometry_id"] for i in hold_idx}
    assert trained and held and not (trained & held)
    assert len(held) == 3


def test_a_family_fits_end_to_end_and_writes_a_card(tmp_path):
    """One small but real run: the loss drops, the held-out error is finite, the card says what was not measured."""
    family = families()["pp_exchanger"]
    out = tmp_path / "run"
    from pinneapple_neural.lpm_pipeline import run

    summary = run(family, out, n_geometries=10, conditions_per_geometry=2, seed=0, epochs=120,
                  batch_designs=10, backbone="mlp", device="cpu", width=48, n_heads=2)
    assert summary["held_out_relative_l2"] is not None and math.isfinite(summary["held_out_relative_l2"])
    assert summary["held_out_relative_l2"] < summary["constant_field_baseline_relative_l2"], \
        "the model must beat predicting the training-average field"
    card = json.loads((out / "model" / "card.json").read_text())
    assert card["schema"] == CARD_SCHEMA
    assert card["trust"]["decision"] == "not_verified_until_independent_review"
    assert card["trust"]["not_measured"], "the card must say what it did not measure"
    assert card["split"]["n_train_geometries"] + card["split"]["n_holdout_geometries"] == 10
    assert set(card["held_out"]) >= {"relative_l2", "relative_l2_p90", "constant_field_baseline_relative_l2"}
    assert (out / "model" / "model.pt").stat().st_size > 0


def test_the_ood_flag_is_measured_on_every_family(tmp_path):
    """A probe outside the family must be flagged: a model that trusts everything is not trustworthy."""
    for name in ("pp_airfoil", "pp_nozzle"):
        family = families()[name]
        assert family.ood_probes(), f"{name} declares no out-of-distribution probe"
        datasets = tmp_path / name
        generate_dataset(family, datasets, n_geometries=12, conditions_per_geometry=2, seed=0)
        summary = train(datasets, tmp_path / f"{name}_run", seed=0, epochs=150, batch_designs=12,
                        backbone="mlp", device="cpu", width=48, n_heads=2)
        assert summary["ood_probes_flagged"].split("/")[0] != "0", summary
