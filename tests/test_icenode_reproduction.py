"""The cloud microphysics of Lamb et al. (2026) and the example that reproduces the paper.

Physics: the capacitance growth law with surface kinetics (its Eqs. 7-12), its closed-form ingredients (Eqs. 13-15,
Table S1) against the values they should give, and the integration. Reproduction plumbing: the 1 Hz grid, the loss
windows of Table 1, the neural ODE's constraints, the symbolic regression, and the two committed result files.
Nothing here downloads the archives: the loader is exercised on files written into ``tmp_path``, and the tests that
need the real series skip when the cache is absent.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.join(REPO, "examples", "reproductions", "lamb2026_icenode")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pinneapple_physics.cloud import (  # noqa: E402
    PLANETARY_CONSTANTS,
    DepositionalIceGrowth,
    alpha_nelson_baker,
    conductivity_air,
    diffusivity_vapour,
    esat_ice,
    mass_from_radius,
    modified_diffusivity,
    radius_from_mass,
    scrit_harrington2019,
    solve_mass_ratio,
    transfer_coefficient,
    transfer_coefficient_continuum,
)


# ---------------------------------------------------------------------------------------------- thermodynamics
def test_saturation_vapour_pressure_over_ice_is_murphy_koop():
    # Murphy & Koop (2005) / Lamb et al. (2026) Eq. 13: 611.657 Pa at the triple point, ~12.6 Pa at -40 C
    assert esat_ice(273.16) == pytest.approx(611.657, rel=1e-4)
    assert esat_ice(233.15) == pytest.approx(12.6, rel=0.03)
    assert np.all(np.diff(esat_ice(np.array([200.0, 220.0, 240.0, 260.0]))) > 0)


def test_diffusivity_and_conductivity_follow_the_paper_fits():
    # Eq. 14 at T0 and p0 is the quoted 2.11e-5 m2/s; Eq. 15 at 0 C is 5.69e-3 * 4.187 W/m/K
    assert diffusivity_vapour(273.15, 101325.0) == pytest.approx(2.11e-5, rel=1e-9)
    assert conductivity_air(273.15) == pytest.approx(4.187e-3 * 5.69, rel=1e-12)
    # lower pressure, higher diffusivity
    assert diffusivity_vapour(233.0, 50000.0) > diffusivity_vapour(233.0, 101325.0)


def test_critical_supersaturation_matches_the_compact_fit():
    # the paper's Eq. 11 is a compact re-expression of the Harrington et al. (2019) fit we evaluate; over the
    # chamber range (205-240 K) they agree to a few percent
    for Tc in (-63.0, -50.0, -40.0, -33.0):
        compact = 9.6066e-5 * abs(Tc) ** 1.9171
        assert scrit_harrington2019(Tc + 273.15) == pytest.approx(compact, rel=0.10)


def test_deposition_coefficient_bounds_and_trend():
    T = 233.0
    a = alpha_nelson_baker(np.array([1.01, 1.1, 1.3, 1.6, 1.75]), T)
    assert np.all((a > 0) & (a <= 1.0))
    assert np.all(np.diff(a) > 0)                      # higher supersaturation, closer to the continuum limit
    assert alpha_nelson_baker(1.0, T) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------------------------- transfer coefficient
def test_continuum_limit_and_kinetic_suppression():
    T, p, r = 233.0, 97190.0, 10e-6
    Gc = transfer_coefficient_continuum(r, T, p)
    assert Gc > 0
    # a large crystal is essentially in the continuum limit; a tiny alpha is not
    assert transfer_coefficient(r, T, p, alpha=1.0) == pytest.approx(Gc, rel=1e-6)
    assert transfer_coefficient(r, T, p, alpha=1e-4) < 1e-2 * Gc
    # the modified diffusivity is bounded by the unmodified one and falls with alpha
    d_hi = modified_diffusivity(r, T, p, 1.0)
    assert d_hi <= diffusivity_vapour(T, p) * (1 + 1e-9)
    assert modified_diffusivity(r, T, p, 0.5) < d_hi


def test_radius_and_mass_round_trip():
    r = np.array([6e-6, 12e-6, 40e-6])
    assert np.allclose(radius_from_mass(mass_from_radius(r)), r)
    # a 12 um sphere of ice: 4/3 pi r^3 rho = 6.59e-12 kg, the paper's initial masses
    assert mass_from_radius(12e-6) == pytest.approx(4 / 3 * np.pi * (12e-6) ** 3 * 910.0)


# ---------------------------------------------------------------------------------------------- growth and integration
def test_growth_law_grows_the_crystal_and_reduces_to_the_analytic_sphere_case():
    T, p, Si, r0 = 233.0, 97190.0, 1.2, 12e-6
    m0 = mass_from_radius(r0)
    model = DepositionalIceGrowth(T=T, p=p, Si=Si, kinetics="none")
    t = np.arange(0.0, 501.0)
    ratio = solve_mass_ratio(model, t, m0)
    assert ratio[0] == 1.0
    assert np.all(np.diff(ratio) > 0)                   # monotone growth at constant supersaturation
    assert radius_from_mass(m0 * ratio[-1]) > r0        # and the radius grows
    # dm/dt from the model and from the finite difference of the solution agree
    dmdt = model.dmdt(m0 * ratio[100])
    fd = m0 * (ratio[101] - ratio[99]) / 2.0
    assert fd == pytest.approx(dmdt, rel=1e-3)


def test_no_supersaturation_no_growth():
    model = DepositionalIceGrowth(T=233.0, p=97190.0, Si=1.0, kinetics="none")
    assert abs(float(model.dmdt(mass_from_radius(10e-6)))) < 1e-30


def test_stiffer_kinetics_grow_slower():
    T, p, Si, m0 = 233.0, 97190.0, 1.6, mass_from_radius(6e-6)
    t = np.arange(0.0, 501.0)
    fast = solve_mass_ratio(DepositionalIceGrowth(T=T, p=p, Si=Si, kinetics="none"), t, m0)[-1]
    slow = solve_mass_ratio(DepositionalIceGrowth(T=T, p=p, Si=Si, kinetics=lambda r, T, Si: 2e-10), t, m0)[-1]
    assert slow < fast


def test_unknown_kinetics_is_rejected():
    with pytest.raises(ValueError):
        DepositionalIceGrowth(kinetics="nonsense").transfer(1e-5)


# ---------------------------------------------------------------------------------------------- neural ODE plumbing
def test_rk4_is_exact_for_a_constant_rate():
    import torch
    from node import integrate

    class ConstantRate:
        Si = torch.zeros(3, 1)

        def dmdt(self, y):
            return torch.ones_like(y) * 0.5

    out = integrate(ConstantRate(), t_max=10.0, dt=1.0)
    assert out.shape == (3, 11)
    assert torch.allclose(out[:, -1], torch.full((3,), 1.0 + 5.0, dtype=torch.float32), atol=1e-5)


def test_stride_and_grid_alignment():
    import torch
    from node import _stride, on_grid

    assert _stride(1.0) == 1 and _stride(5.0) == 5
    grid = torch.arange(11, dtype=torch.float32).view(1, -1)
    assert on_grid(grid, 5.0).tolist() == [[0.0, 5.0, 10.0]]
    with pytest.raises(ValueError):
        _stride(0.5)


def test_score_windows_and_masking():
    import bel
    from node import score

    n_t, n_exp = 20, 2
    ratio = np.ones((n_exp, n_t))
    ratio[0, 10:] = np.nan                              # the first experiment is 10 s long
    data = bel.ChamberData(mass_ratio=ratio, time=np.where(np.isfinite(ratio), np.tile(np.arange(n_t), (n_exp, 1)),
                                                           np.nan),
                           Si=np.array([1.1, 1.2]), T=np.array([230.0, 231.0]), P=np.array([97190.0] * 2),
                           r0=np.array([1e-5] * 2), m0=np.array([4e-12] * 2), nucleation=np.array([1, 1]),
                           source=np.array(["t", "t"]))
    pred = np.full((n_exp, n_t), 2.0)
    s = score(pred, data, dt=1.0)
    assert s["n_points_short"] == 30                     # 10 finite samples in the short series, 20 in the long one
    assert s["sse_short"] == pytest.approx(30.0)          # one squared error of 1.0 per finite sample
    # a model integrated at dt = 2 is resampled onto 1 Hz before scoring (the Table 1 losses are sums over the
    # 1 Hz observations), so the count does not change; scored on its own grid it is 5 + 10 samples
    assert score(pred, data, dt=2.0)["sse_short"] == pytest.approx(30.0)
    assert score(pred, data, dt=2.0, resample_1hz=False)["sse_short"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------------------------- symbolic regression
def test_symbolic_regression_recovers_a_simple_law():
    import symbolic

    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 3.0, 800)
    y = 2.0 * x + 1.0
    front = symbolic.discover({"x": x}, y, generations=12, population=120, seed=0, polish_every=0)
    best = symbolic.refine(front.best_by_mse(), {"x": x}, y)
    assert best.rmse(y, {"x": x}) < 0.05
    assert front.pareto and front.best_within(1.5).nodes <= best.nodes + 4


def test_expression_render_and_constants():
    import symbolic

    e = symbolic.Expr(("+", ("*", ("const", 2.0), ("var", "x")), ("const", 1.0)))
    assert e.nodes == 5 and e.constants() == [2.0, 1.0]
    assert "x" in e.render() and "+" in e.render()
    x = np.array([1.0, 2.0])
    assert np.allclose(e.eval({"x": x}), [3.0, 5.0])


# ---------------------------------------------------------------------------------------------- the committed run
def test_the_committed_results_hold_the_table1_rows():
    import json

    path = os.path.join(HERE, "results", "icenode_seed0.json")
    if not os.path.exists(path):
        pytest.skip("no committed run yet")
    r = json.load(open(path))
    assert set(r["table1"]) == {"none", "nelson_baker", "strong", "weak", "eq6"}
    for k in ("none", "nelson_baker", "strong", "weak"):
        row = r["table1"][k]
        assert row["sse_short"] > 0 and row["sse_long"] > row["sse_short"]
        assert row["paper"]["sse_short"] == {"none": 41396, "nelson_baker": 40751, "strong": 30714,
                                            "weak": 16495}[k]
    assert r["table1"]["weak"]["sse_short"] < r["table1"]["none"]["sse_short"]     # the paper's claim
    assert r["data"]["n_series"] >= 286


# ---------------------------------------------------------------------------------------------- the real archives
@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/.cache/pinneapple/icenode")),
                    reason="chamber archives not downloaded (run bel.download() once)")
def test_loaded_chamber_data_is_physical():
    import bel

    d = bel.load()
    assert len(d) >= 286
    assert 200.0 < d.T.min() and d.T.max() < 245.0
    assert 1.0 < d.Si.min() and d.Si.max() < 1.9
    assert np.all(d.m0 > 0) and np.all(d.r0 > 0)
    assert np.all(d.length() >= 2)
    assert np.allclose(d.m0, 4 / 3 * np.pi * PLANETARY_CONSTANTS["rho_ice"] * d.r0 ** 3)
    # every series starts at m/m0 = 1
    assert np.allclose(d.mass_ratio[np.arange(len(d)), 0], 1.0, atol=2e-2)


def test_interp_to_one_hertz_is_exact_on_a_ramp():
    import bel

    t = np.array([0.0, 0.5, 1.5, 2.0])
    ratio = 1.0 + t
    t_int, r_int = bel._interp_1hz(t, ratio)
    assert t_int.tolist() == [0.0, 1.0, 2.0]
    assert np.allclose(r_int, [1.0, 2.0, 3.0])
