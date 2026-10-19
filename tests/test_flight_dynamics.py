"""6-DOF flight dynamics: attitude kinematics, conservation, the lattice in the loop, trim, derivatives, gear,
take-off against the closed form, and the rolling response to a spanwise-varying gust field."""
import math

import numpy as np
import pytest

from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.aircraft import Controls
from pinneapple_simulation.flight.control import StickFixed
from pinneapple_simulation.flight.rigid_body import (
    euler_to_quat,
    quat_to_dcm,
    quat_to_euler,
    state_vector,
)
from pinneapple_simulation.flight.simulate import energy_check, simulate
from pinneapple_simulation.flight.takeoff import raymer_takeoff, simulate_takeoff, takeoff_metrics
from pinneapple_simulation.flight.trim import (
    classify_modes,
    ground_equilibrium,
    modes,
    stability_derivatives,
    trim,
)
from pinneapple_simulation.flight.turbulence import FrozenField3D, FrozenLine, turbulence_spec


@pytest.fixture(scope="module")
def ga():
    return catalog.build("light_ga")


def test_euler_quaternion_round_trip_including_gimbal_lock():
    rng = np.random.default_rng(0)
    for _ in range(500):
        e = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
        C = quat_to_dcm(euler_to_quat(*e))
        np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(quat_to_dcm(euler_to_quat(*quat_to_euler(euler_to_quat(*e)))), C, atol=1e-9)
    for th in (math.pi / 2, -math.pi / 2):
        q = euler_to_quat(0.4, th, 0.9)
        e = quat_to_euler(q)
        assert all(np.isfinite(e)) and abs(e[1]) == pytest.approx(math.pi / 2)
        np.testing.assert_allclose(quat_to_dcm(euler_to_quat(*e)), quat_to_dcm(q), atol=1e-9)


def test_energy_and_angular_momentum_conserved(ga):
    x0 = state_vector((0, 0, -1000), (60, 2, 3), (0.2, 0.1, 0.3), (0.5, 1.2, -0.7))
    r = energy_check(ga, x0, t_end=10.0, dt=0.005)
    assert r["energy_rel_error"] < 1e-8 and r["momentum_rel_error"] < 1e-8 and r["quat_norm_error"] < 1e-12


def test_lattice_in_the_loop_matches_the_steady_vortex_lattice(ga):
    a = ga.aero
    sol = a.vlm
    for al in (0.0, math.radians(4)):
        v = np.array([50 * math.cos(al), 0, 50 * math.sin(al)])
        out = a.loads(v, np.zeros(3), 1.225)
        lift = sol.coefficients(al, 0.0)["CL"] * 0.5 * 1.225 * 2500 * ga.af.wing_area
        # Kutta-Joukowski with the inclined free stream vs the linearised lattice: equal to O(alpha^2)
        assert out["CL"] * 0.5 * 1.225 * 2500 * ga.af.wing_area == pytest.approx(lift, rel=0.01)
        assert abs(out["F"][1]) < 1e-6 * abs(out["F"][2]) and abs(out["M"][0]) < 1e-6 * abs(out["F"][2])


def test_trim_holds_stick_fixed(ga):
    tr = trim(ga, 55.0, 1500.0)
    assert tr.feasible and tr.residual < 1e-8
    h = simulate(ga, tr.x, StickFixed(tr.controls), 10.0, 0.01)
    assert np.abs(h.airspeed - 55.0).max() < 0.05
    assert np.abs(h.altitude - 1500.0).max() < 0.5
    assert np.abs(h.nz - 1.0).max() < 2e-3


@pytest.mark.parametrize("name", catalog.names())
def test_derivative_signs_and_stable_short_period(name):
    ac = catalog.build(name)
    V = {"light_ga": 55, "regional_turboprop": 120, "a320": 128, "motor_glider": 30}[name]
    tr = trim(ac, V, 1500.0)
    assert tr.feasible
    d = stability_derivatives(ac, tr)
    assert d["CL_alpha"] > 4 and d["Cm_alpha"] < 0 and d["Cm_q"] < 0 and d["Cl_p"] < 0
    assert d["Cn_beta"] > 0 and d["Cl_beta"] < 0 and d["Cn_r"] < 0 and d["Cm_de"] < 0 and d["CY_beta"] < 0
    assert 0.0 < d["static_margin"] < 0.5
    ev = classify_modes(modes(ac, tr)["eigenvalues"])
    sp = ev["oscillatory"][0]                                              # fastest pair: short period
    assert sp["zeta"] > 0.2


def test_ground_equilibrium_carries_the_weight(ga):
    x = ground_equilibrium(ga)
    ev = ga.evaluate(0.0, x, Controls(brake=1.0))
    assert sum(ev["gear_loads"]) == pytest.approx(ga.weight, rel=1e-6)
    assert abs(ev["dx"][11]) < 1e-8 and abs(ev["dx"][5]) < 1e-8


def test_takeoff_agrees_with_the_closed_form(ga):
    h = simulate_takeoff(ga, kind="light", dt=0.01)
    m = takeoff_metrics(ga, h)
    cf = raymer_takeoff(ga, t_rotation=1.0)
    assert m["lifted_off"] and m["V_LOF"] > m["V_R"] - 0.5
    assert 0.7 < m["distance_35ft_m"] / cf["total_35ft_m"] < 1.6
    assert m["climb_gradient_pct"] > 5 and m["stall_margin_ok"]


def test_spanwise_gusts_roll_the_aircraft_a_uniform_one_does_not(ga):
    tr = trim(ga, 55.0, 1500.0)
    s = turbulence_spec(1500.0, "severe")
    ctl = StickFixed(tr.controls)
    f3 = FrozenField3D(s, dx_min=1.0, seed=4)
    f1 = FrozenLine(s, length=20_000, dx=0.5, seed=4)
    f3.scale[:2] = 0.0                                                   # vertical gusts only: no sideslip
    f1.series[:2] = 0.0
    h3 = simulate(ga, tr.x, ctl, 6.0, 0.01, wind=f3)
    h1 = simulate(ga, tr.x, ctl, 6.0, 0.01, wind=f1)
    p3, p1 = np.abs(h3.x[:, 10]).max(), np.abs(h1.x[:, 10]).max()
    assert p3 > 0.02 and p1 < 1e-6 * p3


def test_flight_scene_and_figures(ga, tmp_path):
    from pinneapple_simulation.flight.figures import takeoff_figure
    from pinneapple_simulation.flight.visuals import body_vertices, flight_scene
    tr = trim(ga, 55.0, 300.0)
    h = simulate(ga, tr.x, StickFixed(tr.controls), 1.0, 0.02, record_gamma=True,
                 wind=FrozenLine(turbulence_spec(300.0, "light"), length=5000, dx=1.0))
    sc = flight_scene(ga, h, n_frames=10, wind=FrozenLine(turbulence_spec(300.0, "light"), length=5000, dx=1.0))
    d = sc.extras()["animation"]
    assert sc.animation.n_frames == 10 and "vlm_load" in {s.name for s in sc.surfaces}
    assert len(d["vectors"]) == 1 and len(d["hud"]) == 5
    # the CG maps to the body origin, the nose ends up forward (+x body), the right wing to -y (y left)
    np.testing.assert_allclose(body_vertices(ga, ga.cg), 0.0, atol=1e-12)
    assert body_vertices(ga, [0.0, 0, ga.cg[2]])[0] > 0 and body_vertices(ga, [ga.cg[0], 3.0, ga.cg[2]])[1] < 0
    # the track's first pose is the state's position in z-up axes (north, west, up)
    pose = sc.animation.pose("aircraft", 0)
    np.testing.assert_allclose(pose[:3, 3], [h.x[0, 0], -h.x[0, 1], -h.x[0, 2]], atol=1e-6)
    p = takeoff_figure({"light_ga": h}, {}, str(tmp_path / "f.png"))
    assert (tmp_path / "f.png").stat().st_size > 1000 and p.endswith("f.png")
