"""Dimensional analysis, Buckingham pi, scaling laws and curve collapse (pinneapple_analysis.dynamics.dimensional).

References: the textbook definitions of Re, Pe, Fo, Pr and Ra (checked against the existing, independent
pinneapple_analysis.verification.dimensional_analysis.compute_dimensionless_numbers where it covers them); the
pendulum period 2 pi sqrt(L/g) (small angles) and Stokes drag 3 pi mu U d, whose exponents dimensional analysis
must give exactly and whose constants a fit must recover; Newton cooling T - T_inf = dT exp(-t/tau) and the
semi-infinite conduction profile erfc(x / (2 sqrt(alpha t))), which collapse exactly under their dimensional
scalings.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.special import erfc

from pinneapple_analysis import dynamics as dyn
from pinneapple_analysis.dynamics.dimensional import (
    BuckinghamPiAnalyzer,
    CurveCollapseAnalyzer,
    Dimension,
    DimensionalMatrix,
    DimensionError,
    DimensionlessGroup,
    PhysicalQuantity,
    ScalingLaw,
    SimilarityMetrics,
    TrajectoryNormalizer,
    check_homogeneity,
    quantities,
)
from pinneapple_analysis.verification.dimensional_analysis import compute_dimensionless_numbers

Q = PhysicalQuantity


# =========================================================================== dimensions and units
def test_dimension_algebra_and_units():
    v = Dimension.of(L=1, T=-1)
    assert v * Dimension.of(T=1) == Dimension.of(L=1)
    assert (v ** 2) / v == v and (v ** Fraction(1, 2)).exponent("L") == Fraction(1, 2)
    assert Dimension().is_dimensionless and str(Dimension()) == "1"
    assert Dimension.from_unit("kg/(m·s)") == Dimension.of(M=1, L=-1, T=-1)
    assert Dimension.from_unit("W/m2/K") == Dimension.of(M=1, T=-3, Theta=-1)
    assert Dimension.from_unit("Pa s") == Dimension.from_unit("kg/(m·s)")
    mu = Q.from_unit("mu", 1.8e-2, "cP")             # centipoise -> Pa s
    assert mu.value == pytest.approx(1.8e-5) and mu.dimension == Dimension.of(M=1, L=-1, T=-1)
    with pytest.raises(DimensionError, match="offset"):
        Q.from_unit("T", 20.0, "°C")
    with pytest.raises(ValueError):
        Dimension.of(L=0.333333)                          # not a simple rational


def test_homogeneity_of_bernoulli_and_a_wrong_sum():
    p, rho, U, g, h = (Dimension.from_unit("Pa"), Dimension.from_unit("kg/m3"), Dimension.from_unit("m/s"),
                       Dimension.from_unit("m/s2"), Dimension.from_unit("m"))
    ok = check_homogeneity({"p": p, "rho U^2 / 2": rho * U ** 2, "rho g h": rho * g * h})
    assert ok.applicable and ok.ok
    bad = check_homogeneity({"p": p, "rho U": rho * U})
    assert not bad.applicable and "rho U" in bad.failures[0]


def test_dimensional_matrix_rank_and_integer_nullspace():
    qs = [Q.of("rho", M=1, L=-3), Q.of("U", L=1, T=-1), Q.of("L", L=1), Q.of("mu", M=1, L=-1, T=-1)]
    D = DimensionalMatrix(qs)
    assert D.bases == ["L", "M", "T"] and D.rank == 3
    (v,) = D.nullspace()
    assert all(x.denominator == 1 for x in v) and D.is_dimensionless(v)
    assert [int(x) for x in v] in ([1, 1, 1, -1], [-1, -1, -1, 1])
    with pytest.raises(ValueError, match="unique"):
        DimensionalMatrix([qs[0], qs[0]])


# =========================================================================== named groups
def _vals(**kw):
    return {k: Q.of(k, v) for k, v in kw.items()}


def test_reynolds_matches_the_existing_dimensionless_numbers():
    rho, U, L, mu = 998.0, 1.5, 0.05, 1.0e-3
    qs = [Q.from_unit("rho", rho, "kg/m3"), Q.from_unit("U", U, "m/s"), Q.from_unit("L", L, "m"),
          Q.from_unit("mu", mu, "Pa s")]
    r = BuckinghamPiAnalyzer(qs).analyze()
    assert r.ok and r.estimate["n_groups"] == 1 and "Reynolds" in r.estimate["recognized"]
    g = r.details["groups"][0]
    ref = compute_dimensionless_numbers(velocity=U, length=L, density=rho, dynamic_viscosity=mu).reynolds
    assert g.evaluate(quantities(qs)) == pytest.approx(ref, rel=1e-12)


def test_peclet_fourier_prandtl_recognized_and_valued():
    U, L, alpha, nu, t = 0.2, 0.1, 1.4e-7, 1.0e-6, 30.0
    pe = BuckinghamPiAnalyzer([Q.of("U", U, L=1, T=-1), Q.of("L", L, L=1), Q.of("alpha", alpha, L=2, T=-1)])
    g = pe.groups()[0]
    assert g.recognize() in ("Peclet", "1/Peclet")
    ref = compute_dimensionless_numbers(velocity=U, length=L, thermal_diffusivity=alpha).peclet_heat
    val = g.evaluate({"U": U, "L": L, "alpha": alpha})
    assert (val if g.recognize() == "Peclet" else 1 / val) == pytest.approx(ref, rel=1e-12)
    fo = BuckinghamPiAnalyzer([Q.of("alpha", L=2, T=-1), Q.of("t", T=1), Q.of("L", L=1)], target="t").groups()[0]
    assert fo.name == "Fourier" and fo.evaluate({"alpha": alpha, "t": t, "L": L}) == pytest.approx(alpha * t / L ** 2)
    pr = BuckinghamPiAnalyzer([Q.of("nu", L=2, T=-1), Q.of("alpha", L=2, T=-1)]).groups()[0]
    assert pr.recognize() in ("Prandtl", "1/Prandtl")
    ref_pr = compute_dimensionless_numbers(kinematic_viscosity=nu, thermal_diffusivity=alpha).prandtl
    assert pr.evaluate({"nu": nu, "alpha": alpha}) ** (1 if pr.recognize() == "Prandtl" else -1) == pytest.approx(ref_pr)


def test_rayleigh_is_a_product_of_the_groups_found():
    qs = [Q.of("g", L=1, T=-2), Q.of("beta", Theta=-1), Q.of("dT", Theta=1), Q.of("L", L=1),
          Q.of("nu", L=2, T=-1), Q.of("alpha", L=2, T=-1)]
    an = BuckinghamPiAnalyzer(qs)
    r = an.analyze()
    assert r.estimate["rank"] == 3 and r.estimate["n_groups"] == 3          # 6 quantities, L T Theta
    Ra = DimensionlessGroup({"g": 1, "beta": 1, "dT": 1, "L": 3, "nu": -1, "alpha": -1})
    assert Ra.recognize() == "Rayleigh"
    coeffs = an.express(Ra)
    assert coeffs is not None
    rebuilt = DimensionlessGroup({})
    for g in r.details["groups"]:
        rebuilt = rebuilt * g.power(coeffs[g.name])
    assert rebuilt.exponents == Ra.exponents
    assert an.express(DimensionlessGroup({"g": 1, "L": 1})) is None            # not dimensionless


def test_groups_in_a_force_length_time_system():
    # the same drag problem in F-L-T bases: F, rho = F T^2 L^-4, U, d, mu = F T L^-2 -> 2 groups
    qs = [Q.of("F", F=1), Q.of("rho", F=1, T=2, L=-4), Q.of("U", L=1, T=-1), Q.of("d", L=1),
          Q.of("mu", F=1, T=1, L=-2)]
    r = BuckinghamPiAnalyzer(qs, target="F").analyze()
    assert r.estimate["rank"] == 3 and r.estimate["n_groups"] == 2
    assert all(DimensionalMatrix(qs).is_dimensionless([g.exponents.get(q.name, 0) for q in qs])
               for g in r.details["groups"])


def test_symbolic_group():
    sympy = pytest.importorskip("sympy")
    g = DimensionlessGroup({"rho": 1, "U": 1, "L": 1, "mu": -1})
    expr = g.sympy()
    rho, U, L, mu = sympy.symbols("rho U L mu", positive=True)
    assert sympy.simplify(expr - rho * U * L / mu) == 0


# =========================================================================== scaling laws
def test_pendulum_exponents_exact_and_constant_fitted():
    qs = [Q.of("T", T=1), Q.of("L", L=1), Q.of("g", L=1, T=-2), Q.of("m", M=1)]
    an = BuckinghamPiAnalyzer(qs, target="T")
    r = an.analyze()
    assert r.diagnostic("isolated_dimension:M").status == "warn"            # mass cannot enter: irrelevant
    law = an.scaling_law()
    assert law.exact and law.exponents == {"L": 0.5, "g": -0.5}   # m drops out
    rng = np.random.default_rng(0)
    L = rng.uniform(0.2, 3.0, 40)
    g = rng.choice([9.81, 3.71, 1.62], 40)
    T = 2 * np.pi * np.sqrt(L / g) * np.exp(rng.normal(0, 0.01, 40))      # 1 % multiplicative noise
    fitted = law.fit_constant(T, {"L": L, "g": g})
    lo, hi = fitted.ci["log_C"]
    assert lo < math.log(2 * math.pi) < hi and fitted.constant == pytest.approx(2 * math.pi, rel=0.01)
    free = ScalingLaw.fit(T, {"L": L, "g": g})
    for k, b in (("L", 0.5), ("g", -0.5)):
        assert free.ci[k][0] < b < free.ci[k][1]
    assert free.r2 > 0.99 and not free.exact


def test_stokes_drag_law():
    qs = [Q.of("F", M=1, L=1, T=-2), Q.of("mu", M=1, L=-1, T=-1), Q.of("U", L=1, T=-1), Q.of("d", L=1)]
    an = BuckinghamPiAnalyzer(qs, target="F")
    law = an.scaling_law()
    assert law.exponents == {"mu": 1.0, "U": 1.0, "d": 1.0}
    mu, U, d = np.array([1e-3, 1e-2, 1.0]), np.array([1e-3, 2e-3, 5e-4]), np.array([1e-4, 3e-4, 1e-3])
    fitted = law.fit_constant(3 * np.pi * mu * U * d, {"mu": mu, "U": U, "d": d})
    assert fitted.constant == pytest.approx(3 * np.pi, rel=1e-12)
    assert fitted.predict({"mu": 1.0, "U": 1.0, "d": 1.0}) == pytest.approx(3 * np.pi)


def test_scaling_law_with_more_groups_is_refused():
    qs = [Q.of("F", M=1, L=1, T=-2), Q.of("rho", M=1, L=-3), Q.of("U", L=1, T=-1), Q.of("d", L=1),
          Q.of("mu", M=1, L=-1, T=-1)]
    with pytest.raises(DimensionError, match="f\\("):
        BuckinghamPiAnalyzer(qs, target="F").scaling_law()


# =========================================================================== insufficient / failure
def test_target_that_cannot_be_formed_is_reported():
    qs = [Q.of("q", M=1, T=-3), Q.of("k", M=1, L=1, T=-3, Theta=-1), Q.of("L", L=1)]   # heat flux without dT
    r = BuckinghamPiAnalyzer(qs, target="q").analyze()
    assert not r.applicable and any("insufficient quantities" in f for f in r.failures)
    assert r.diagnostic("isolated_dimension:Theta").value == "k"
    r2 = BuckinghamPiAnalyzer(qs + [Q.of("dT", Theta=1)], target="q").analyze()
    assert r2.ok and r2.estimate["n_groups"] == 1               # adding dT fixes it: q L / (k dT), a Nusselt number
    assert r2.details["groups"][0].exponents == {"q": 1, "L": 1, "k": -1, "dT": -1}


def test_no_group_and_bad_repeating_variables():
    r = BuckinghamPiAnalyzer([Q.of("L", L=1), Q.of("t", T=1)]).analyze()
    assert not r.applicable and "no dimensionless group" in r.failures[0]
    qs = [Q.of("rho", M=1, L=-3), Q.of("U", L=1, T=-1), Q.of("L", L=1), Q.of("mu", M=1, L=-1, T=-1)]
    with pytest.raises(DimensionError, match="independent"):
        BuckinghamPiAnalyzer(qs, target="mu", repeating=["rho", "U"]).analyze()
    with pytest.raises(ValueError, match="target"):
        BuckinghamPiAnalyzer(qs, target="mu", repeating=["rho", "U", "mu"]).analyze()
    with pytest.raises(KeyError):
        BuckinghamPiAnalyzer(qs, target="nu")
    with pytest.raises(DimensionError):
        DimensionalMatrix(qs, bases=["L", "T"])          # M is used but not a declared base


def test_characteristic_scales_unique_and_ambiguous():
    an = BuckinghamPiAnalyzer([Q.of("L", L=1), Q.of("alpha", L=2, T=-1)])
    c = an.characteristic_scales(Dimension.of(T=1))
    assert c["unique"] and c["scale"].exponents == {"L": 2, "alpha": -1}
    an2 = BuckinghamPiAnalyzer([Q.of("L", L=1), Q.of("alpha", L=2, T=-1), Q.of("U", L=1, T=-1)])
    c2 = an2.characteristic_scales(Dimension.of(T=1))
    assert not c2["unique"] and len(c2["ambiguity"]) == 1                # L/U vs L^2/alpha differ by Pe
    assert c2["ambiguity"][0].recognize() in ("Peclet", "1/Peclet")
    assert not an.characteristic_scales(Dimension.of(M=1))["exists"]


# =========================================================================== normalisation
def test_dimensional_normalizer_checks_dimensions_and_inverts():
    qs = quantities(Q.of("L", 0.01, L=1), Q.of("alpha", 1e-5, L=2, T=-1), Q.of("dT", 50.0, Theta=1))
    ts = DimensionlessGroup({"L": 2, "alpha": -1})
    n = TrajectoryNormalizer.dimensional(qs, time=ts, states={"T": DimensionlessGroup({"dT": 1})},
                                         state_dimensions={"T": Dimension.of(Theta=1)}, state_offsets={"T": 20.0})
    assert n.time_scale == pytest.approx(10.0)
    tr = dyn.from_array(np.linspace(0, 30, 7), 20 + 50 * np.exp(-np.linspace(0, 3, 7)), ("T",),
                        units={"t": "s", "T": "°C"})
    z = n.apply(tr)
    assert z.source["normalization"]["kind"] == "dimensional" and z.units == {"t": "1", "T": "1"}
    assert np.allclose(z.t, np.linspace(0, 3, 7)) and np.allclose(z.component("T"), np.exp(-z.t))
    back = n.invert(z)
    assert np.allclose(back.t, tr.t) and np.allclose(back.x, tr.x)
    with pytest.raises(DimensionError, match="not T"):
        TrajectoryNormalizer.dimensional(qs, time=DimensionlessGroup({"L": 1, "alpha": -1}))
    with pytest.raises(DimensionError, match="Theta"):
        TrajectoryNormalizer.dimensional(qs, states={"T": DimensionlessGroup({"L": 1})},
                                         state_dimensions={"T": Dimension.of(Theta=1)})


def test_empirical_normalizer_is_labelled_as_such():
    t = np.linspace(0, 5, 11)
    tr = dyn.from_array(t, 3 + 2 * np.sin(t), ("y",))
    for method in ("minmax", "zscore", "peak", "initial"):
        z = TrajectoryNormalizer.empirical(tr, method, time="span").apply(tr)
        info = z.source["normalization"]
        assert info["kind"] == "empirical" and "not a pi representation" in info["note"]
    mm = TrajectoryNormalizer.empirical(tr, "minmax").apply(tr)
    assert mm.component("y").min() == pytest.approx(0) and mm.component("y").max() == pytest.approx(1)
    with pytest.raises(ValueError, match="initial value is 0"):
        TrajectoryNormalizer.empirical(dyn.from_array(t, np.sin(t)), "initial")


# =========================================================================== collapse
def _cooling_family(taus, dTs, seed=0, n=80):
    rng = np.random.default_rng(seed)
    out = []
    for tau, dT in zip(taus, dTs, strict=True):
        t = np.sort(np.concatenate([[0.0], rng.uniform(0, 6 * tau, n)]))
        out.append(dyn.from_array(t, dT * np.exp(-t / tau), ("T",), kind="analytic"))
    return out


def test_cooling_curves_collapse_under_the_dimensional_scaling():
    taus = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    dTs = np.array([10.0, 30.0, 50.0, 20.0, 70.0])
    ca = CurveCollapseAnalyzer(_cooling_family(taus, dTs))
    r = ca.analyze(taus, dTs, label="t/tau, T/dT")
    assert r.ok and r.estimate["collapse"] < 1e-3 and r.estimate["raw"] > 1.0
    assert r.estimate["p_value"] <= 0.01                                    # permuted scales do much worse
    assert r.diagnostic("improvement_over_raw").value > 1e3


def test_semi_infinite_conduction_profiles_collapse_in_the_similarity_variable():
    alpha = 1e-5
    x = np.linspace(0, 0.05, 120)
    times = np.array([10.0, 40.0, 160.0, 640.0])
    profiles = [dyn.from_array(x, erfc(x / (2 * np.sqrt(alpha * t))), ("theta",), kind="analytic") for t in times]
    ca = CurveCollapseAnalyzer(profiles)
    r = ca.analyze(np.sqrt(alpha * times), np.ones(times.size), n_permutations=199)
    assert r.estimate["collapse"] < 1e-4 and r.estimate["p_value"] < 0.05


def test_fitted_exponents_and_their_stability():
    taus = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    ca = CurveCollapseAnalyzer(_cooling_family(taus, taus ** 0.5))         # dT = tau^(1/2)
    fit = ca.fit_exponents(taus, x0=(0.6, 0.2))
    assert fit["x_exponent"] == pytest.approx(1.0, abs=0.01) and fit["y_exponent"] == pytest.approx(0.5, abs=0.01)
    assert all(c > 0 for c in fit["curvature"])                            # a sharp minimum: stable
    sens = ca.sensitivity(taus, 1.0, 0.5)
    mid = sens["deltas"].index(0.0)
    assert sens["x_exponent"][mid] == min(sens["x_exponent"]) and sens["y_exponent"][mid] == min(sens["y_exponent"])
    r = ca.analyze(taus ** fit["x_exponent"], taus ** fit["y_exponent"], origin="fitted")
    assert r.diagnostic("scales_fitted_on_these_curves").status == "warn"


def test_empirical_normalisation_collapses_curves_of_different_laws():
    t = np.linspace(0, 1, 100)
    laws = [np.exp(-5 * t), 1 - t, 1 / (1 + np.exp(10 * (t - 0.5))), np.cos(np.pi * t / 2), (1 - t) ** 3]
    trs = [dyn.from_array(t, y) for y in laws]
    normed = [TrajectoryNormalizer.empirical(tr, "minmax", time="span").apply(tr) for tr in trs]
    r = CurveCollapseAnalyzer(normed).analyze(origin="empirical")
    assert r.diagnostic("collapse_by_construction").status == "warn" and r.estimate["origin"] == "empirical"
    assert r.diagnostic("permutation_control").status == "warn"            # no scales left to permute
    assert r.estimate["collapse"] < r.estimate["raw"] + 1e-12              # same curves: "collapsed" at 0.4


def test_degenerate_and_unsupported_collapse():
    t = np.linspace(0, 1, 50)
    flat = [dyn.from_array(t, np.full(50, c)) for c in (1.0, 2.0, 3.0)]
    r = CurveCollapseAnalyzer(flat).analyze([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert not r.applicable and "degenerate" in r.failures[0]
    apart = [dyn.from_array(t + k, t) for k in (0.0, 5.0, 10.0)]
    r2 = CurveCollapseAnalyzer(apart).analyze()
    assert not r2.applicable and "common support" in r2.failures[0]
    r3 = CurveCollapseAnalyzer(apart[:2]).analyze()
    assert any("at least 3" in f for f in r3.failures)


def test_noise_degrades_collapse_gracefully_and_reproducibly():
    taus = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    rng = np.random.default_rng(3)
    noisy = []
    for tr in _cooling_family(taus, np.ones(5)):
        noisy.append(dyn.from_array(tr.t, tr.component("T") + rng.normal(0, 0.02, tr.n), ("T",), kind="experimental"))
    ca = CurveCollapseAnalyzer(noisy)
    r = ca.analyze(taus, np.ones(5), seed=4)
    assert 1e-4 < r.estimate["collapse"] < 0.05 and r.estimate["p_value"] <= 0.02
    again = ca.analyze(taus, np.ones(5), seed=4)
    assert again.estimate == r.estimate and again.reproducibility.input_hash == r.reproducibility.input_hash


# =========================================================================== similarity
def test_similarity_between_a_model_and_a_prototype():
    g = BuckinghamPiAnalyzer([Q.of("rho", M=1, L=-3), Q.of("U", L=1, T=-1), Q.of("L", L=1),
                              Q.of("mu", M=1, L=-1, T=-1)]).groups()
    proto = {"rho": 1.2, "U": 30.0, "L": 2.0, "mu": 1.8e-5}
    model = {"rho": 1000.0, "U": 30.0 * 2.0 / 0.1 * (1.2 / 1000.0) * (1.0e-3 / 1.8e-5), "L": 0.1, "mu": 1.0e-3}
    r = SimilarityMetrics.groups(g, proto, model)
    assert r.estimate["similar"] and abs(next(iter(r.estimate["groups"].values()))["log_ratio"]) < 1e-12
    off = dict(model, U=model["U"] * 2)
    assert not SimilarityMetrics.groups(g, proto, off).estimate["similar"]
    t = np.linspace(0, 1, 30)
    d = SimilarityMetrics.curves(dyn.from_array(t, np.exp(-t)), dyn.from_array(t, np.exp(-t) * 1.01))
    assert d["relative_l2"] == pytest.approx(0.01, rel=1e-6)


def test_method_cards_and_contract():
    assert BuckinghamPiAnalyzer.card.role == "primitive" and CurveCollapseAnalyzer.card.role == "validation"
    r = BuckinghamPiAnalyzer([Q.of("U", L=1, T=-1), Q.of("L", L=1), Q.of("alpha", L=2, T=-1)]).analyze()
    names = [a.name for a in r.unchecked_assumptions]
    assert "complete list of quantities" in names
    s = r.to_study_result()
    assert s.metrics["estimate.n_groups"] == 1.0


def test_benchmarks_quick_all_pass():
    from pinneapple_analysis.dynamics.benchmarks import dimensional as bench

    cases = bench.run(quick=True)
    failed = [c.to_dict() for c in cases if not c.passed]
    assert not failed, failed
    assert {c.name for c in cases} >= {"reynolds", "pendulum_constant", "cooling_collapse", "empirical_false_collapse"}
