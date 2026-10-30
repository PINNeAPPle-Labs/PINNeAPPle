"""The compartmental models of Schiassi et al., "Physics-Informed Extreme Theory of Functional Connections Applied
to Data-Driven Parameters Discovery of Epidemiological Compartmental Models" (arXiv:2008.05554v1): equations,
true and known parameters, initial states, horizon and the run settings the paper reports (points, neurons,
activation, weight range). No result of the paper is in this module; the targets live in
``examples/reproductions/schiassi_epi2020/TARGETS.md``.

``synthetic_data`` builds the paper's data sets: the model integrated to high accuracy, plus ``delta * U(-1, 1)``
noise on every compartment at every point (eq. 28, read as U(-1, 1)).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinneapple_simulation.numerical_solvers.xtfc_general import Constraint, XTFCProblem

__all__ = ["EpiModel", "EPI_MODELS", "epi_model", "synthetic_data"]


@dataclass(frozen=True)
class EpiModel:
    name: str
    compartments: tuple[str, ...]
    rhs: Callable[[np.ndarray, dict], np.ndarray]                 # numpy, state (C,) -> derivative (C,)
    residual_terms: Callable[[dict, dict], list[torch.Tensor]]    # torch: (values, params) -> RHS per compartment
    true_params: dict[str, float]
    known: dict[str, float]
    unknowns: tuple[str, ...]
    y0: tuple[float, ...]
    horizon: float
    n_points: int
    n_neurons: int
    activation: str = "arctan"
    weight_range: tuple[float, float] = (-10.0, 10.0)
    source: str = ""

    def problem(self) -> XTFCProblem:
        """Each compartment with its initial value as an exact constraint (constant constraint function)."""
        fields = {c: [Constraint(0, 0.0)] for c in self.compartments}
        cfn = {c: (lambda X, v=v: torch.full((X.shape[0],), v, dtype=X.dtype)) for c, v in
               zip(self.compartments, self.y0, strict=True)}
        derivs = [(c, (a,)) for c in self.compartments for a in (0, 1)]
        return XTFCProblem([(0.0, self.horizon)], fields, cfn, lambda X, D: [], derivs)

    def residual(self, X, D, P):
        vals = {c: D[(c, (0,))] for c in self.compartments}
        rhs = self.residual_terms(vals, {**{k: v for k, v in self.known.items()}, **P})
        return [D[(c, (1,))] - r for c, r in zip(self.compartments, rhs, strict=True)]


def _sir(v, p):
    S, I, R = v["S"], v["I"], v["R"]  # noqa: E741 (the paper's notation)
    N = S + I + R
    return [p["mu"] * N - p["beta"] * S * I / N - p["mu"] * S,
            p["beta"] * S * I / N - p["gamma"] * I - p["mu"] * I,
            p["gamma"] * I - p["mu"] * R]


def _seir(v, p):
    S, E, I, R = v["S"], v["E"], v["I"], v["R"]  # noqa: E741 (the paper's notation)
    N = S + E + I + R
    return [p["mu"] * (N - S) - p["beta"] * S * I / N - p["nu"] * S,
            p["beta"] * S * I / N - (p["mu"] + p["sigma"]) * E,
            p["sigma"] * E - p["gamma"] * I - p["mu"] * I,
            p["gamma"] * I - p["mu"] * R + p["nu"] * S]


def _seirs(v, p):
    S, E, I, R = v["S"], v["E"], v["I"], v["R"]  # noqa: E741 (the paper's notation)
    N = S + E + I + R
    return [p["mu"] * N - p["beta"] * S * I / N + p["zeta"] * R - p["nu"] * S,
            p["beta"] * S * I / N - p["sigma"] * E - p["nu"] * E,
            p["sigma"] * E - p["gamma"] * I - p["mu"] * I,
            p["gamma"] * I - p["nu"] * R - p["zeta"] * R]


def _np_rhs(terms, names):
    def f(y, p):
        return np.array([float(t) for t in terms(dict(zip(names, y, strict=True)), p)])
    return f


_SRC = "Schiassi et al., arXiv:2008.05554v1"
EPI_MODELS: dict[str, EpiModel] = {
    "sir": EpiModel("sir", ("S", "I", "R"), _np_rhs(_sir, ("S", "I", "R")), _sir,
                    {"beta": 0.5, "gamma": 1 / 3}, {"mu": 0.1}, ("beta", "gamma"), (100.0, 5.0, 0.0), 15.0, 100, 50,
                    source=f"{_SRC} eq. 16, Sec. 4.1, Table 1 (n = 100, L = 50)"),
    "seir": EpiModel("seir", ("S", "E", "I", "R"), _np_rhs(_seir, ("S", "E", "I", "R")), _seir,
                     {"beta": 0.3, "gamma": 0.6, "sigma": 0.9}, {"mu": 0.5, "nu": 0.5}, ("beta", "gamma", "sigma"),
                     (70.0, 30.0, 10.0, 0.0), 15.0, 100, 80,
                     source=f"{_SRC} eq. 20, Sec. 4.2, Table 2 (n = 100, L = 80)"),
    # nu is not given for SEIRS; nu = mu keeps the population constant, as the text intends (TARGETS.md, item 1)
    "seirs": EpiModel("seirs", ("S", "E", "I", "R"), _np_rhs(_seirs, ("S", "E", "I", "R")), _seirs,
                      {"beta": 0.3, "gamma": 0.6, "sigma": 0.9, "zeta": 0.5}, {"mu": 0.5, "nu": 0.5},
                      ("beta", "gamma", "sigma", "zeta"), (70.0, 30.0, 10.0, 0.0), 15.0, 100, 80,
                      source=f"{_SRC} eq. 24, Sec. 4.3, Table 3 (n = 100, L = 80)"),
}


def epi_model(name: str) -> EpiModel:
    return EPI_MODELS[name]


def synthetic_data(model: EpiModel, delta: float, seed: int = 0, n: int | None = None):
    """``(t, clean, noisy)``: the model integrated (DOP853, rtol = atol = 1e-13) at ``n`` uniform points on
    [0, horizon], and ``clean + delta * U(-1, 1)`` drawn independently per point and compartment."""
    n = n or model.n_points
    t = np.linspace(0.0, model.horizon, n)
    p = {**model.known, **model.true_params}
    sol = solve_ivp(lambda _t, y: model.rhs(y, p), (0.0, model.horizon), model.y0, t_eval=t, method="DOP853",
                    rtol=1e-13, atol=1e-13)
    clean = sol.y.T
    noisy = clean + delta * np.random.default_rng(seed).uniform(-1.0, 1.0, size=clean.shape)
    return t, clean, noisy


def fit_shooting(model: EpiModel, t: np.ndarray, y: np.ndarray, p0: dict[str, float]) -> dict:
    """Baseline: the unknown parameters by nonlinear least squares on the integrated model (single shooting,
    ``scipy.optimize.least_squares`` around ``solve_ivp``; the initial state is known, as in the X-TFC fit).

    The rates are bounded to ``[0, 10]`` and the start ``p0`` is clipped into ``[1e-3, 10]``: a negative rate makes
    the model blow up and the integrator crawl. An integration that leaves ``1e6`` times the initial population is
    stopped and scored as a large misfit."""
    import time

    from scipy.optimize import least_squares

    names = list(model.unknowns)
    limit = 1e6 * sum(abs(v) for v in model.y0)

    def blowup(_t, s):
        return limit - np.abs(s).sum()
    blowup.terminal = True

    def resid(x):
        p = {**model.known, **dict(zip(names, x, strict=True))}
        sol = solve_ivp(lambda _t, s: model.rhs(s, p), (t[0], t[-1]), model.y0, t_eval=t, method="DOP853",
                        rtol=1e-10, atol=1e-10, events=blowup)
        if sol.status != 0 or sol.y.shape[1] != len(t):
            return np.full(y.size, 1e6)
        return (sol.y.T - y).ravel()

    t0 = time.perf_counter()
    x0 = np.clip([p0[k] for k in names], 1e-3, 10.0)
    out = least_squares(resid, x0, bounds=(0.0, 10.0), method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return {"params": dict(zip(names, map(float, out.x), strict=True)), "nfev": int(out.nfev),
            "seconds": time.perf_counter() - t0}


__all__.append("fit_shooting")
