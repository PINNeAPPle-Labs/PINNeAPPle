"""pinneapple_systems.control: analysis, actuators and loops, tuning, all against closed forms."""
import math

import numpy as np
import pytest

from pinneapple_systems.control import (
    PID,
    Actuator,
    Cascade,
    Plant,
    StateFeedback,
    compare_architectures,
    controllability,
    dlqr,
    equilibrium,
    fit_fopdt,
    imc_pid,
    linearize,
    loop_eigenvalues,
    loop_margins,
    lqr,
    observability,
    place_pid_second_order,
    robustness,
    simc_pid,
    simulate_linear_step,
    simulate_loop,
    stability,
    step_metrics,
    transfer_function,
    tune_by_optimization,
    ultimate_gain,
    ziegler_nichols,
)


def msd(x, u, d, t):
    return np.array([x[1], -4.0 * x[0] - 0.8 * x[1] + u[0]])


# ---------------------------------------------------------------- analysis
def test_linearize_and_stability_match_the_mass_spring_damper():
    lin = linearize(msd, [0.0, 0.0], [0.0])
    assert np.allclose(lin["A"], [[0, 1], [-4, -0.8]], atol=1e-8)
    assert np.allclose(lin["B"], [[0], [1]], atol=1e-8)
    st = stability(lin["A"])
    assert st["classification"] == "stable (oscillatory)"
    wn, zeta = 2.0, 0.8 / 4.0
    assert np.allclose(sorted(st["s"].imag), [-wn * math.sqrt(1 - zeta ** 2), wn * math.sqrt(1 - zeta ** 2)])
    assert all(abs(m["damping_ratio"] - zeta) < 1e-9 for m in st["modes"])


def test_equilibrium_of_a_pendulum_with_torque():
    def pend(x, u, d, t):
        return np.array([x[1], -9.81 * math.sin(x[0]) + u[0]])
    eq = equilibrium(pend, [0.3, 0.0], [9.81 * math.sin(0.5)])
    assert eq["converged"] and abs(eq["x"][0] - 0.5) < 1e-9


def test_unstable_pole_doubling_time_matches_simulation():
    lam = 0.3
    st = stability([[lam]])
    assert st["classification"] == "unstable"
    assert abs(st["modes"][0]["doubling_time"] - math.log(2) / lam) < 1e-12
    r = simulate_linear_step([[lam]], [[0.0]], [[1.0]], t_end=1.0, n=11)
    assert r["t"][-1] == 1.0


def test_discrete_stability_maps_to_continuous_time():
    dt = 0.1
    st = stability([[math.exp(-0.5 * dt)]], dt=dt)
    assert abs(st["s"][0].real + 0.5) < 1e-12 and st["classification"] == "stable"


def test_controllability_and_observability_of_the_double_integrator():
    A = [[0, 1], [0, 0]]
    assert controllability(A, [[0], [1]])["controllable"]
    assert not controllability(A, [[1], [0]])["controllable"]
    assert observability(A, [[1, 0]])["observable"]
    assert not observability(A, [[0, 1]])["observable"]


def test_transfer_function_of_the_mass_spring_damper():
    tf = transfer_function([[0, 1], [-4, -0.8]], [[0], [1]], [[1, 0]])
    assert np.allclose(tf["den"], [1, 0.8, 4]) and abs(tf["dc_gain"][0] - 0.25) < 1e-12


def test_loop_margins_of_k_over_s_s1_s2():
    m = loop_margins(([1.0], [1, 3, 2, 0]))
    assert abs(m["gain_margin"] - 6.0) < 1e-3                 # analytic: 6 at w = sqrt(2)
    assert abs(m["phase_crossover"] - math.sqrt(2)) < 1e-3
    # gain crossover: |1/(jw(jw+1)(jw+2))| = 1
    wc = m["gain_crossover"]
    assert abs(wc * math.sqrt(wc ** 2 + 1) * math.sqrt(wc ** 2 + 4) - 1) < 1e-6
    pm = 180 - 90 - math.degrees(math.atan(wc)) - math.degrees(math.atan(wc / 2))
    assert abs(m["phase_margin"] - pm) < 0.05
    assert abs(m["delay_margin"] - math.radians(pm) / wc) < 1e-2
    assert m["ms"] > 1


def test_delay_reduces_the_phase_margin():
    m0 = loop_margins(([1.0], [1, 1, 0]))
    m1 = loop_margins(([1.0], [1, 1, 0]), delay=0.2)
    assert abs((m0["phase_margin"] - m1["phase_margin"]) - math.degrees(0.2 * m0["gain_crossover"])) < 0.1


def test_step_metrics_match_second_order_formulas():
    lin = linearize(msd, [0.0, 0.0], [0.0])
    r = simulate_linear_step(lin["A"], lin["B"], [[1, 0]], t_end=40, n=8001)
    sm = step_metrics(r["t"], r["y"][:, 0], reference=0.25)
    zeta = 0.2
    assert abs(sm["overshoot"] - 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))) < 0.05
    assert abs(sm["peak_time"] - math.pi / (2 * math.sqrt(1 - zeta ** 2))) < 0.01


# ---------------------------------------------------------------- actuators and loops
def test_actuator_rate_and_range_limits():
    a = Actuator(lo=0.0, hi=10.0, rate=2.0, initial=0.0)
    a.reset(0.5)
    vals = [a.step(100.0, 0.5) for _ in range(12)]
    assert np.allclose(np.diff([0.0] + vals)[:10], 1.0) and vals[-1] == 10.0


def test_integrating_actuator_integrates_a_clipped_rate():
    a = Actuator(lo=-100, hi=100, rate=3.0, integrating=True, initial=1.0)
    a.reset(0.1)
    for _ in range(10):
        a.step(50.0, 0.1)
    assert abs(a.applied - 4.0) < 1e-12


def test_actuator_delay_and_lag():
    a = Actuator(delay=0.3, initial=0.0)
    a.reset(0.1)
    out = [a.step(1.0, 0.1) for _ in range(5)]
    assert out == [0.0, 0.0, 0.0, 1.0, 1.0]
    b = Actuator(tau=1.0, initial=0.0)
    b.reset(0.01)
    for _ in range(100):
        b.step(1.0, 0.01)
    assert abs(b.applied - (1 - math.exp(-1.0))) < 1e-9


def first_order_plant():
    return Plant(rhs=lambda x, u, d, t: np.array([(-x[0] + 2.0 * u + d) / 5.0]), x0=[0.0], output=0)


def test_pid_tracks_a_step_and_back_calculation_reduces_windup():
    plant = first_order_plant()
    act = Actuator(lo=-1.0, hi=1.0)
    runs = {}
    for aw in ("none", "back_calculation", "conditional"):
        pid = PID(4.0, 2.0, 0.0, output_limits=(-1.0, 1.0), anti_windup=aw)
        runs[aw] = simulate_loop(plant, pid, act, 1.5, 60.0, 0.05, method="rk4")
    for r in runs.values():
        assert abs(r["y"][-1] - 1.5) < 1e-2
    assert runs["back_calculation"]["y_max"] < runs["none"]["y_max"] - 0.05
    assert runs["conditional"]["y_max"] < runs["none"]["y_max"] - 0.05


def test_derivative_on_measurement_has_no_setpoint_kick():
    pe = PID(1.0, 0.0, 5.0, derivative="error")
    pm = PID(1.0, 0.0, 5.0, derivative="measurement")
    for p in (pe, pm):
        p.update(0.0, 0.1, setpoint=0.0)
    assert abs(pe.update(0.0, 0.1, setpoint=1.0)) > 10 and abs(pm.update(0.0, 0.1, setpoint=1.0) - 1.0) < 1e-12


def test_pid_set_equilibrium_is_bumpless():
    p = PID(2.0, 0.5, 1.0, bias=3.0)
    p.set_equilibrium(1.0, 1.0, 7.0)
    assert abs(p.update(1.0, 0.1, rate=0.0, setpoint=1.0) - 7.0) < 1e-12


def double_integrator():
    return Plant(rhs=lambda x, u, d, t: np.array([u + d, x[0]]), x0=[0.0, 0.0], output=1, rate=lambda x: x[0])


def test_integrating_actuator_adds_a_pole_and_destabilises_the_same_pd():
    plant = double_integrator()
    pd = PID(1.0, 0.0, 1.4, derivative="measurement")
    servo = (pd, Actuator())
    rate = (PID(1.0, 0.0, 1.4, derivative="measurement"), Actuator(integrating=True))
    a = loop_eigenvalues(plant, *servo, x_eq=[0.0, 0.0], u_eq=0.0, reference=0.0, dt=0.01)
    b = loop_eigenvalues(plant, *rate, x_eq=[0.0, 0.0], u_eq=0.0, reference=0.0, dt=0.01)
    # continuous closed loops: s^2 + 1.4 s + 1 (stable) and s^3 + 1.4 s + 1 (missing s^2 term: unstable)
    assert a["stable"] and abs(a["max_real"] + 0.7) < 0.02
    assert not b["stable"]
    roots = np.roots([1, 0, 1.4, 1.0])
    assert abs(b["max_real"] - roots.real.max()) < 0.02


def test_compare_architectures_reports_both_views():
    plant = double_integrator()
    archs = {"servo": (PID(1.0, 0.0, 1.4), Actuator()),
             "rate": (PID(1.0, 0.0, 1.4), Actuator(integrating=True))}
    out = compare_architectures(plant, archs, reference=0.0, t_end=30.0, dt=0.01, disturbance=lambda t: 0.1,
                                x_eq=[0.0, 0.0], u_eq=0.0, stop=lambda x, t: "diverged" if abs(x[1]) > 50 else None)
    rows = {r["name"]: r for r in out["rows"]}
    assert rows["servo"]["stable_linear"] and rows["servo"]["stopped"] is None
    assert rows["rate"]["stable_linear"] is False and rows["rate"]["stopped"] == "diverged"


def test_cascade_and_state_feedback_stabilise_the_double_integrator():
    plant = double_integrator()
    outer = PID(1.0, 0.0, 0.0, derivative="measurement")
    inner = PID(2.0, 0.0, 0.0)
    cas = Cascade(outer, inner, inner_measurement=lambda x: x[0])
    r = simulate_loop(plant, cas, Actuator(), 1.0, 30.0, 0.01)
    assert abs(r["y"][-1] - 1.0) < 1e-2
    K = lqr([[0, 0], [1, 0]], [[1], [0]], np.eye(2), [[1.0]])["K"]          # state [v, x]: v' = u, x' = v
    sf = StateFeedback(K, x_ref=[0.0, 1.0])
    r2 = simulate_loop(plant, sf, Actuator(), 1.0, 30.0, 0.01)
    assert abs(r2["y"][-1] - 1.0) < 1e-2


# ---------------------------------------------------------------- tuning
def test_fit_fopdt_recovers_a_known_plant():
    t = np.linspace(0, 60, 601)
    y = 2.0 * 3.0 * (1 - np.exp(-np.clip(t - 4.0, 0, None) / 8.0))
    f = fit_fopdt(t, y, du=3.0)
    assert abs(f["K"] - 2.0) < 1e-3 and abs(f["tau"] - 8.0) < 1e-2 and abs(f["theta"] - 4.0) < 1e-2


def test_simc_imc_and_zn_rules():
    s = simc_pid(2.0, 10.0, 1.0)
    assert abs(s["kc"] - 10 / (2 * 2)) < 1e-12 and s["ti"] == 8.0           # min(tau, 4 (tau_c + theta)) = 8
    i = imc_pid(2.0, 10.0, 2.0, lam=2.0, kind="pi")
    assert abs(i["kc"] - 10 / (2 * 4)) < 1e-12 and i["ti"] == 10.0
    z = ziegler_nichols("pid", ku=6.0, tu=2 * math.pi / math.sqrt(2))
    assert abs(z["kc"] - 3.6) < 1e-12 and abs(z["td"] - z["ti"] / 4) < 1e-12
    u = ultimate_gain(([1.0], [1, 3, 2, 0]))
    assert abs(u["ku"] - 6.0) < 1e-3 and abs(u["tu"] - 2 * math.pi / math.sqrt(2)) < 1e-3


def test_pole_placement_places_the_poles_of_an_unstable_plant():
    g = place_pid_second_order(m=2.0, c=0.0, k=-3.0, b=0.5, wn=1.5, zeta=0.7, p=0.4)
    want = np.sort_complex(np.concatenate([np.roots([1, 2 * 0.7 * 1.5, 1.5 ** 2]), [-0.4]]))
    assert np.allclose(np.sort_complex(g["poles"]), want, atol=1e-9)


def test_lqr_double_integrator_closed_form_and_dlqr():
    r = lqr([[0, 1], [0, 0]], [[0], [1]], np.eye(2), [[1.0]])
    assert np.allclose(r["K"], [[1.0, math.sqrt(3)]], atol=1e-9)
    d = dlqr([[1, 0.1], [0, 1]], [[0.005], [0.1]], np.eye(2), [[1.0]])
    assert np.all(np.abs(d["poles"]) < 1)


def test_tune_by_optimization_improves_iae_under_an_overshoot_limit():
    plant = first_order_plant()

    def run(g):
        r = simulate_loop(plant, PID(g["kp"], g["ki"], 0.0), Actuator(), 1.0, 40.0, 0.05)
        m = step_metrics(r["t"], r["y"], reference=1.0, y0=0.0)
        return {"iae": m["iae"], "overshoot": m["overshoot"]}

    out = tune_by_optimization(run, {"kp": 0.3, "ki": 0.05}, {"kp": (0.01, 50), "ki": (0.001, 20)},
                               constraints={"overshoot": 5.0}, maxiter=40)
    assert out["improved"] and out["metrics"]["overshoot"] <= 5.0
    assert out["metrics"]["iae"] < out["initial_metrics"]["iae"]


def test_robustness_counts_stable_samples():
    def run(p):
        return {"stable": p["a"] < 0, "x": abs(p["a"])}
    rb = robustness(run, {"a": (-1.0, 1.0)}, n=40, metric_keys=("x",))
    assert 0.3 < rb["fraction_stable"] < 0.7 and rb["worst"]["x"]["value"] <= 1.0


def test_bad_options_are_refused():
    with pytest.raises(ValueError):
        PID(1.0, derivative="bad")
    with pytest.raises(ValueError):
        simulate_loop(first_order_plant(), PID(1.0), Actuator(), 1.0, 1.0, 0.1, method="leapfrog")
