"""Dimensional-analysis benchmarks against known groups, known scaling laws and exact similarity solutions.

| Case | Reference | Source |
|---|---|---|
| ``reynolds`` / ``peclet`` / ``prandtl`` | the value from ``compute_dimensionless_numbers`` | the library's existing, independent implementation of the textbook formulas |
| ``fourier`` | ``alpha t / L^2`` | definition |
| ``rayleigh_in_span`` | Ra is a product of the groups found for ``g, beta, dT, L, nu, alpha`` | exact rational solve; 0 means it is |
| ``pendulum_constant`` | ``2 pi`` | small-angle period ``2 pi sqrt(L/g)``, data with 1 % log-normal noise |
| ``pendulum_free_exponent_L`` | ``1/2`` | free power-law fit, same data |
| ``stokes_constant`` | ``3 pi`` | Stokes drag ``3 pi mu U d`` |
| ``cooling_collapse`` | 0 | Newton cooling under ``t/tau``, ``T/dT``; measured against raw and permuted scales |
| ``conduction_collapse`` | 0 | ``erfc(x/(2 sqrt(alpha t)))`` under ``x/sqrt(alpha t)`` |
| ``fitted_exponent_x`` / ``_y`` | ``1`` and ``1/2`` | exponents recovered by minimising the collapse measure |
| ``empirical_false_collapse`` | the raw measure | five curves of five different laws, min-max and span normalised: shows what an empirical normalisation "collapses" |
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import erfc

from ...verification.dimensional_analysis import compute_dimensionless_numbers
from ..dimensional import (
    BuckinghamPiAnalyzer,
    CurveCollapseAnalyzer,
    DimensionlessGroup,
    PhysicalQuantity,
    ScalingLaw,
    TrajectoryNormalizer,
)
from ..trajectories import from_array
from ._case import BenchmarkCase

__all__ = ["run"]
Q = PhysicalQuantity


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run(quick: bool = False, seed: int = 0) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    rho, U, L, mu, alpha, nu, t = 998.0, 1.5, 0.05, 1.0e-3, 1.4e-7, 1.0e-6, 30.0
    ref = compute_dimensionless_numbers(velocity=U, length=L, density=rho, dynamic_viscosity=mu,
                                        thermal_diffusivity=alpha)
    ref_pr = compute_dimensionless_numbers(kinematic_viscosity=nu, thermal_diffusivity=alpha).prandtl

    def group_value(qs, values, name):
        g = BuckinghamPiAnalyzer(qs).groups()[0]
        v = g.evaluate(values)
        rec = g.recognize() or ""
        return 1 / v if rec.startswith("1/") else v, rec

    (re_v, _), c = _timed(lambda: group_value(
        [Q.of("rho", M=1, L=-3), Q.of("U", L=1, T=-1), Q.of("L", L=1), Q.of("mu", M=1, L=-1, T=-1)],
        {"rho": rho, "U": U, "L": L, "mu": mu}, "Reynolds"))
    cases.append(BenchmarkCase("reynolds", "Re", ref.reynolds, "compute_dimensionless_numbers", re_v,
                               tolerance=1e-9 * ref.reynolds, cost_s=c))
    (pe_v, _), c = _timed(lambda: group_value([Q.of("U", L=1, T=-1), Q.of("L", L=1), Q.of("alpha", L=2, T=-1)],
                                              {"U": U, "L": L, "alpha": alpha}, "Peclet"))
    cases.append(BenchmarkCase("peclet", "Pe", ref.peclet_heat, "compute_dimensionless_numbers", pe_v,
                               tolerance=1e-9 * ref.peclet_heat, cost_s=c))
    (pr_v, _), c = _timed(lambda: group_value([Q.of("nu", L=2, T=-1), Q.of("alpha", L=2, T=-1)],
                                              {"nu": nu, "alpha": alpha}, "Prandtl"))
    cases.append(BenchmarkCase("prandtl", "Pr", ref_pr, "compute_dimensionless_numbers", pr_v,
                               tolerance=1e-9 * ref_pr, cost_s=c))
    fo, c = _timed(lambda: BuckinghamPiAnalyzer([Q.of("alpha", L=2, T=-1), Q.of("t", T=1), Q.of("L", L=1)],
                                                target="t").groups()[0])
    cases.append(BenchmarkCase("fourier", "Fo", alpha * t / L ** 2, "alpha t / L^2",
                               fo.evaluate({"alpha": alpha, "t": t, "L": L}), tolerance=1e-15, cost_s=c,
                               limitations=f"group recognised as {fo.name}"))

    qs_ra = [Q.of("g", L=1, T=-2), Q.of("beta", Theta=-1), Q.of("dT", Theta=1), Q.of("L", L=1),
             Q.of("nu", L=2, T=-1), Q.of("alpha", L=2, T=-1)]
    coeffs, c = _timed(lambda: BuckinghamPiAnalyzer(qs_ra).express(
        DimensionlessGroup({"g": 1, "beta": 1, "dT": 1, "L": 3, "nu": -1, "alpha": -1})))
    cases.append(BenchmarkCase("rayleigh_in_span", "Ra not a product of the groups (0 = it is)", 0.0,
                               "exact rational solve", 0.0 if coeffs else 1.0, tolerance=0.0, cost_s=c,
                               convergence={"exponents": {k: str(v) for k, v in (coeffs or {}).items()}}))

    # ---- pendulum
    rng = np.random.default_rng(seed)
    n = 40
    Ls = rng.uniform(0.2, 3.0, n)
    gs = rng.choice([9.81, 3.71, 1.62], n)
    T = 2 * np.pi * np.sqrt(Ls / gs) * np.exp(rng.normal(0, 0.01, n))
    an = BuckinghamPiAnalyzer([Q.of("T", T=1), Q.of("L", L=1), Q.of("g", L=1, T=-2), Q.of("m", M=1)], target="T")
    law, c = _timed(lambda: an.scaling_law().fit_constant(T, {"L": Ls, "g": gs}))
    lo, hi = law.ci["log_C"]
    cases.append(BenchmarkCase("pendulum_constant", "C in T = C sqrt(L/g)", 2 * math.pi,
                               "small-angle pendulum", law.constant, mc_se=law.constant * (hi - lo) / 3.92,
                               cost_s=c, config={"n": n, "noise": "1 % log-normal", "seed": seed},
                               limitations="exponents exact from dimensions; the error bar is from the t interval "
                                           "on log C"))
    free, c = _timed(lambda: ScalingLaw.fit(T, {"L": Ls, "g": gs}))
    cases.append(BenchmarkCase("pendulum_free_exponent_L", "fitted exponent of L", 0.5, "dimensional analysis",
                               free.exponents["L"], mc_se=(free.ci["L"][1] - free.ci["L"][0]) / 3.92 * 1.0,
                               cost_s=c, config={"n": n}, convergence={"r2": free.r2,
                                                                       "exponent_g": free.exponents["g"]}))

    # ---- Stokes drag
    an_s = BuckinghamPiAnalyzer([Q.of("F", M=1, L=1, T=-2), Q.of("mu", M=1, L=-1, T=-1), Q.of("U", L=1, T=-1),
                                 Q.of("d", L=1)], target="F")
    mus, Us, ds = np.array([1e-3, 1e-2, 1.0]), np.array([1e-3, 2e-3, 5e-4]), np.array([1e-4, 3e-4, 1e-3])
    st, c = _timed(lambda: an_s.scaling_law().fit_constant(3 * np.pi * mus * Us * ds, {"mu": mus, "U": Us, "d": ds}))
    cases.append(BenchmarkCase("stokes_constant", "C in F = C mu U d", 3 * math.pi, "Stokes drag", st.constant,
                               tolerance=1e-12 * 3 * math.pi, cost_s=c,
                               limitations="noise-free data: checks the algebra, not the fit under noise"))

    # ---- collapses
    taus = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    dTs = np.array([10.0, 30.0, 50.0, 20.0, 70.0])
    rng = np.random.default_rng(seed + 1)
    fam = []
    for tau, dT in zip(taus, dTs, strict=True):
        tt = np.sort(np.concatenate([[0.0], rng.uniform(0, 6 * tau, 80)]))
        fam.append(from_array(tt, dT * np.exp(-tt / tau), ("T",), kind="analytic"))
    nperm = 199 if quick else 999
    r, c = _timed(lambda: CurveCollapseAnalyzer(fam).analyze(taus, dTs, n_permutations=nperm, seed=seed))
    cases.append(BenchmarkCase("cooling_collapse", "collapse measure under t/tau, T/dT", 0.0, "exact similarity",
                               r.estimate["collapse"], tolerance=1e-3, cost_s=c,
                               convergence={"raw": r.estimate["raw"], "p_value": r.estimate["p_value"],
                                            "permuted_median": r.uncertainty["permutation_null"]["median"]},
                               limitations="residual from linear interpolation of 81 irregular samples per curve"))
    alpha_c = 1e-5
    x = np.linspace(0, 0.05, 120)
    times = np.array([10.0, 40.0, 160.0, 640.0])
    prof = [from_array(x, erfc(x / (2 * np.sqrt(alpha_c * tt))), ("theta",), kind="analytic") for tt in times]
    r2, c = _timed(lambda: CurveCollapseAnalyzer(prof).analyze(np.sqrt(alpha_c * times), np.ones(4),
                                                               n_permutations=nperm, seed=seed))
    cases.append(BenchmarkCase("conduction_collapse", "collapse measure under x / sqrt(alpha t)", 0.0,
                               "similarity solution erfc(eta/2)", r2.estimate["collapse"], tolerance=1e-4, cost_s=c,
                               convergence={"raw": r2.estimate["raw"], "p_value": r2.estimate["p_value"]},
                               limitations="common support is set by the earliest profile (eta <= 0.05/sqrt(alpha t_max))"))
    fam2 = []
    for tau in taus:
        tt = np.sort(np.concatenate([[0.0], rng.uniform(0, 6 * tau, 80)]))
        fam2.append(from_array(tt, tau ** 0.5 * np.exp(-tt / tau), ("T",), kind="analytic"))
    fit, c = _timed(lambda: CurveCollapseAnalyzer(fam2).fit_exponents(taus, x0=(0.6, 0.2)))
    cases.append(BenchmarkCase("fitted_exponent_x", "x exponent", 1.0, "t / tau", fit["x_exponent"],
                               tolerance=0.01, cost_s=c, convergence={"curvature": fit["curvature"]}))
    cases.append(BenchmarkCase("fitted_exponent_y", "y exponent", 0.5, "dT = tau^(1/2)", fit["y_exponent"],
                               tolerance=0.01, cost_s=0.0))

    t = np.linspace(0, 1, 100)
    laws = [np.exp(-5 * t), 1 - t, 1 / (1 + np.exp(10 * (t - 0.5))), np.cos(np.pi * t / 2), (1 - t) ** 3]
    trs = [from_array(t, y) for y in laws]
    normed = [TrajectoryNormalizer.empirical(tr, "minmax", time="span").apply(tr) for tr in trs]
    r3, c = _timed(lambda: CurveCollapseAnalyzer(normed).analyze(origin="empirical", n_permutations=nperm))
    raw_laws = CurveCollapseAnalyzer(trs).measure()
    cases.append(BenchmarkCase("empirical_false_collapse", "collapse of 5 different laws after min-max + span",
                               raw_laws, "the same curves before normalisation", r3.estimate["collapse"],
                               tolerance=math.inf, cost_s=c,
                               convergence={"collapse_by_construction_flag":
                                            r3.diagnostic("collapse_by_construction").status},
                               limitations="no pass/fail: shows that a 'collapse' measure near 0.4 arises from five "
                                           "unrelated laws, which is why empirical normalisation is labelled"))
    return cases
