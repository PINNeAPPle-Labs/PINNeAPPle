"""Two-layer energy-balance model of global-mean surface warming (Held et al. 2010, J. Climate 23:2418; Geoffroy et
al. 2013a, J. Climate 26:1841; efficacy of deep-ocean heat uptake: Geoffroy et al. 2013b, J. Climate 26:1859):

    C  dT/dt  = F - lambda T - eps gamma (T - T0)        (upper ocean + atmosphere + land)
    C0 dT0/dt = gamma (T - T0)                            (deep ocean)

Units: T in K, F in W m-2, time in years, heat capacities in W yr m-2 K-1, lambda and gamma in W m-2 K-1.

* :meth:`TwoLayerEBM.simulate` integrates exactly for forcing that is constant over each step (matrix exponential), so
  annual forcing series need no small step;
* :meth:`TwoLayerEBM.step_response` is the closed-form response to an abrupt forcing step: a fast mode (years) and
  a slow mode (centuries), as in Geoffroy et al. (2013a);
* :meth:`TwoLayerEBM.ecs` and :meth:`TwoLayerEBM.tcr` give the equilibrium climate sensitivity F2x/lambda and the
  transient response at year 70 of a 1 %/yr CO2 increase;
* :func:`fit_two_layer` estimates the parameters from a temperature series and its forcing by least squares, with
  standard errors from the Jacobian (an inverse problem: observed warming -> feedback and ocean heat uptake).
  Identifiability: an abrupt-forcing run constrains every parameter; a historical ramp constrains the transient
  response (TCR) well but trades lambda against the deep-ocean capacity, so lambda and ECS come back with large
  errors. The returned correlation matrix shows it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["TwoLayerEBM", "fit_two_layer", "F2X_DEFAULT"]

F2X_DEFAULT = 3.71          # W m-2, radiative forcing of doubled CO2 (Myhre et al. 1998 logarithmic fit, 5.35 ln 2)


@dataclass(frozen=True)
class TwoLayerEBM:
    lam: float              # climate feedback parameter
    gamma: float            # heat-exchange coefficient
    C: float                # upper-layer heat capacity
    C0: float               # deep-ocean heat capacity
    eps: float = 1.0        # efficacy of deep-ocean heat uptake

    def matrix(self) -> np.ndarray:
        g, e = self.gamma, self.eps
        return np.array([[-(self.lam + e * g) / self.C, e * g / self.C], [g / self.C0, -g / self.C0]])

    def simulate(self, forcing: Sequence[float], dt: float = 1.0, T_init: float = 0.0, T0_init: float = 0.0) -> dict[str, np.ndarray]:
        """Exact integration with forcing[k] held over [k dt, (k+1) dt]. Returns T and T0 at the step ends, and the
        top-of-atmosphere imbalance N = F - lambda T - (eps - 1) gamma (T - T0)."""
        from scipy.linalg import expm

        A = self.matrix()
        E = expm(A * dt)
        G = np.linalg.solve(A, (E - np.eye(2)) @ np.array([1 / self.C, 0.0]))
        x = np.array([T_init, T0_init], dtype=float)
        F = np.asarray(forcing, dtype=float)
        T, T0 = np.empty(len(F)), np.empty(len(F))
        for k, f in enumerate(F):
            x = E @ x + G * f
            T[k], T0[k] = x
        N = F - self.lam * T - (self.eps - 1) * self.gamma * (T - T0)
        return {"t": dt * np.arange(1, len(F) + 1), "T": T, "T0": T0, "N": N}

    def modes(self) -> dict[str, float]:
        """Time scales and amplitudes of the fast and slow modes of the step response, from the eigen-decomposition
        of the system matrix (valid for any efficacy): T(t) = T_eq [1 - a_f exp(-t/tau_f) - a_s exp(-t/tau_s)], with
        a_f + a_s = 1. For eps = 1 these are the time scales and amplitudes of Geoffroy et al. (2013a)."""
        A = self.matrix()
        ev, V = np.linalg.eig(A)
        ev, V = ev.real, V.real
        order = np.argsort(-1 / ev)                                  # fast (small tau) first
        ev, V = ev[order], V[:, order]
        x_eq = -np.linalg.solve(A, np.array([1 / self.C, 0.0]))      # equilibrium per unit forcing
        c = np.linalg.solve(V, x_eq)
        amps = V[0] * c / x_eq[0]
        return {"tau_fast": float(-1 / ev[0]), "tau_slow": float(-1 / ev[1]), "a_fast": float(amps[0]),
                "a_slow": float(amps[1]), "T_eq_per_Wm2": float(x_eq[0])}

    def step_response(self, t: np.ndarray, F: float = F2X_DEFAULT) -> np.ndarray:
        """Surface warming after an abrupt, constant forcing F applied at t = 0 (closed form, two exponentials)."""
        m = self.modes()
        t = np.asarray(t, dtype=float)
        return F * m["T_eq_per_Wm2"] * (1 - m["a_fast"] * np.exp(-t / m["tau_fast"]) - m["a_slow"] * np.exp(-t / m["tau_slow"]))

    def ecs(self, F2x: float = F2X_DEFAULT) -> float:
        return F2x / self.lam

    def tcr(self, F2x: float = F2X_DEFAULT, dt: float = 1.0) -> float:
        """Warming at year 70 of a 1 %/yr CO2 increase (forcing grows linearly to F2x at year 70)."""
        years = np.arange(0, 70, dt) + dt / 2
        return float(self.simulate(F2x * years / 70.0, dt)["T"][-1])

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "ecs": self.ecs(), "tcr": self.tcr()}


def fit_two_layer(T_obs: Sequence[float], forcing: Sequence[float], dt: float = 1.0, fit_eps: bool = False,
                  guess: TwoLayerEBM | None = None, sigma: float | None = None) -> dict[str, object]:
    """Least-squares fit of (lambda, gamma, C, C0[, eps]) to a warming series driven by ``forcing``.

    Parameters are fitted in log space (all positive). Returns the fitted model, standard errors (from J^T J and the
    residual variance, or ``sigma`` if the observation error is known), the residual RMS and the derived ECS and TCR
    with first-order (delta-method) errors."""
    from scipy.optimize import least_squares

    T_obs = np.asarray(T_obs, dtype=float)
    g = guess or TwoLayerEBM(lam=1.2, gamma=0.7, C=8.0, C0=100.0, eps=1.0)
    names = ["lam", "gamma", "C", "C0"] + (["eps"] if fit_eps else [])
    p0 = np.log([getattr(g, n) for n in names])

    def model(p):
        kw = dict(zip(names, np.exp(p), strict=True))
        if not fit_eps:
            kw["eps"] = g.eps
        return TwoLayerEBM(**kw)

    def resid(p):
        return model(p).simulate(forcing, dt)["T"] - T_obs

    sol = least_squares(resid, p0, method="trf", x_scale="jac")
    m = model(sol.x)
    r = sol.fun
    dof = max(len(r) - len(names), 1)
    s2 = sigma ** 2 if sigma is not None else float(r @ r) / dof
    try:
        cov_log = np.linalg.inv(sol.jac.T @ sol.jac) * s2
    except np.linalg.LinAlgError:
        cov_log = np.full((len(names), len(names)), np.nan)
    vals = np.exp(sol.x)
    cov = cov_log * np.outer(vals, vals)                            # log -> linear (delta method)
    se = dict(zip(names, np.sqrt(np.clip(np.diag(cov), 0, None)), strict=True))
    lam_se = se["lam"]
    sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(sd, sd)
    return {"model": m, "params": dict(zip(names, vals.tolist(), strict=True)), "stderr": se, "correlation": corr,
            "residual_rms": float(np.sqrt(np.mean(r * r))), "ecs": m.ecs(), "ecs_stderr": m.ecs() * lam_se / m.lam,
            "tcr": m.tcr(), "success": bool(sol.success), "covariance": cov}
