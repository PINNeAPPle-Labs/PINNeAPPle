"""Inference tools of the bio line on small problems: SBINN, BINN, eikonal activation mapping, cell-migration
solver, Windkessel and the aorta caps."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_sbinn_recovers_a_decay_rate():
    from pinneapple_analysis.inverse_problems.sbinn import SBINN, SBINNConfig
    from pinneapple_physics.biology import ODESystem

    torch.set_num_threads(1)
    s = ODESystem.from_strings("ab", {"a": "-k*a", "b": "k*a - 0.5*b"}, {"a": 1.0, "b": 0.0}, {"k": 1.2})
    t = np.linspace(0, 5, 101)
    tr = s.simulate(t)
    sb = SBINN(s, ["k"], {"k": (0.5, 2.0)}, {"b": (t, tr["b"])}, aux={"a": [1.0, tr["a"][-1]]}, aux_times=[0, 5],
               config=SBINNConfig(hidden=(32, 32), n_features=0, epochs_data=1500, epochs=15000, lr=3e-3,
                                  record_every=500, dtype="float64"))
    res = sb.fit()
    assert res.parameters["k"] == pytest.approx(1.2, rel=0.03)      # a is never observed: inferred from b only


def test_cell_migration_solver_conserves_mass_without_growth():
    from pinneapple_physics.biology.cell_migration import solve_reaction_diffusion

    x = np.linspace(0, 2, 41)
    u0 = np.where(np.abs(x - 1) < 0.3, 1000.0, 100.0)
    t = np.linspace(0, 1, 5)
    U = solve_reaction_diffusion(lambda u: np.full_like(u, 0.05), lambda u: np.zeros_like(u), x, t, u0, n=401)
    m = np.trapezoid(U, x, axis=0)
    assert np.max(np.abs(m - m[0])) / m[0] < 2e-2                   # no-flux: mass constant (interp. error only)
    assert U[:, -1].max() < U[:, 0].max()                          # it spreads


def test_gls_error_definition():
    from pinneapple_physics.biology.cell_migration import gls_error

    um, ud = np.array([100.0, 1000.0]), np.array([110.0, 990.0])
    expected = np.mean([(10 / 100 ** 0.2) ** 2, (10 / 1000 ** 0.2) ** 2])
    assert gls_error(um, ud) == pytest.approx(expected)


def test_binn_runs_and_returns_terms():
    from pinneapple_analysis.inverse_problems.binn import BINNConfig, ReactionDiffusionBINN

    torch.set_num_threads(1)
    x = np.linspace(0.1, 1.9, 19)
    t = np.array([0, 0.5, 1.0, 1.5, 2.0])
    U = 300 + 1200 / (1 + np.exp(-(np.abs(x[:, None] - 1) - 0.5 + 0.2 * t[None, :]) * 8))
    b = ReactionDiffusionBINN(BINNConfig(n_iters=60, n_collocation=500, patience=10**6)).fit(x, t, U)
    D = b.predict_term("D", np.linspace(300, 1500, 5))
    T = b.predict_term("T", t)
    assert np.all((D >= 0) & (D <= 0.096)) and np.all((T > 0) & (T < 1))
    g, pred = b.forward_gls()
    assert pred.shape == U.shape and math.isfinite(g)


def test_eikonal_pinn_beats_nothing_on_a_planar_wave():
    from pinneapple_analysis.inverse_problems.activation_mapping import (
        EikonalPINN,
        EikonalPINNConfig,
        interpolate_baseline,
    )

    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, (40, 2))
    T = pts[:, 0] / 1.5                                            # planar wave, speed 1.5
    q = rng.uniform(1, 9, (200, 2))
    p = EikonalPINN(((0, 10), (0, 10)), EikonalPINNConfig(n_iters=1500, n_collocation=500, hidden=(32, 32)))
    Tp, Vp = p.fit(pts, T).predict(q)
    assert np.sqrt(np.mean((Tp - q[:, 0] / 1.5) ** 2)) < 0.3
    assert np.median(np.abs(Vp - 1.5)) < 0.3
    base = interpolate_baseline(pts, T, q)
    assert np.sqrt(np.mean((base - q[:, 0] / 1.5) ** 2)) < 1e-9     # linear data: interpolation is exact


def test_windkessel_matches_its_impedance():
    from pinneapple_simulation.hemodynamics import Windkessel, check_against_impedance

    wk = Windkessel(1e7, 1e-8, 1e8)
    for w in (2 * math.pi, 20.0):
        r = check_against_impedance(wk, w, 1e-4)
        assert r["amplitude_rel_error"] < 2e-3 and r["phase_error_rad"] < 3e-3


def test_split_rcr_mean_pressure_and_inflow_volume():
    from pinneapple_simulation.hemodynamics import MMHG, half_sine_inflow, split_rcr

    wk = split_rcr({"a": 0.02, "b": 0.01})
    q = {"a": 70e-6 * 8 / 9, "b": 70e-6 / 9}
    for k, w in wk.items():
        assert q[k] * (w.Rp + w.Rd) == pytest.approx(93 * MMHG)
    t = np.linspace(0, 1, 100001)
    assert np.trapezoid(half_sine_inflow(t), t) == pytest.approx(70e-6, rel=1e-6)


def test_explicit_windkessel_coupling_window():
    """The aorta bug: a 10 ms window was unstable, 1 ms is not. A lumped vessel L dQ/dt = p_in - P coupled to an RCR
    the way CoupledRun does it (P held over each window, then updated from the window's end flow)."""
    from pinneapple_simulation.hemodynamics import (
        Windkessel,
        explicit_coupling_gain,
        vessel_inertance,
    )

    L = vessel_inertance(0.05, 0.009)                               # a 5 cm right-subclavian-like branch

    def run(window, t_end=0.2):
        wk = Windkessel(2.71e8, 5.89e-10, 2.74e9)                    # the arch's rsa outlet from split_rcr
        Q, P, n = 0.0, 0.0, 200
        qs = []
        for _ in range(int(round(t_end / window))):
            for _ in range(n):                                       # the 3-D solver's own steps in the window
                Q += window / n * (12e3 - P) / L
            P = wk.step(Q, window)
            qs.append(Q)
        return np.array(qs), explicit_coupling_gain(wk, L, window)

    q10, g10 = run(0.01)
    q1, g1 = run(0.001)
    assert g10 > 2 and g1 < 1
    assert np.abs(q10[-1]) > 1e3 * np.abs(q1).max()                  # diverges, sign flipping
    assert np.all(np.sign(np.diff(q10[-6:])) != np.sign(np.diff(q10[-7:-1])))
    assert np.all(np.isfinite(q1)) and q1.max() < 12e3 / 2.71e8      # bounded by p_in / Rp
    assert np.count_nonzero(np.diff(np.sign(np.diff(q1)))) <= 1      # rises then relaxes: no oscillation


def test_aorta_caps_are_the_six_openings():
    pytest.importorskip("manifold3d")
    from pinneapple_design import geometry_library as gl
    from pinneapple_simulation.hemodynamics.aorta import _label_aorta_caps, find_caps

    g = gl.build("aortic_arch", lod="light")
    b = g.bodies[0]
    caps = find_caps(np.asarray(b.vertices), np.asarray(b.faces))
    assert len(caps) == 6
    named = _label_aorta_caps(caps, g.params)
    assert np.linalg.norm(named["inlet"]["centroid"]) < 1e-6
    for k, d in (("outlet_rsa", 0.009), ("outlet_rcca", 0.007), ("outlet_lcca", 0.008), ("outlet_lsa", 0.010)):
        assert named[k]["diameter"] == pytest.approx(d, rel=0.05)


def test_windkessel_inertance_and_coupling_gain_known_answers():
    """L = rho l / A for a rigid tube, and the explicit-coupling gain g = Rp dt / L."""
    from pinneapple_simulation.hemodynamics.windkessel import (
        Windkessel,
        explicit_coupling_gain,
        vessel_inertance,
    )
    assert vessel_inertance(0.1, 0.02) == pytest.approx(1060.0 * 0.1 / (3.141592653589793 * 0.02 ** 2 / 4))
    assert explicit_coupling_gain(Windkessel(Rp=1e8, C=1e-8, Rd=1e9), 1e6, 1e-3) == pytest.approx(0.1)
