"""Conceptual ENSO models: the recharge oscillator and the delayed oscillator.

**Recharge oscillator** (Jin 1997, J. Atmos. Sci. 54:811; the linear stochastic form of Burgers et al. 2005, GRL
32:L13706 and Jin et al. 2020, in *El Nino Southern Oscillation in a Changing Climate*, AGU), with T the eastern
Pacific SST anomaly [K] and h the warm-water-volume / thermocline-depth anomaly [m], t in months::

    dT/dt = R(s) T + F1(s) h + sigma_T (1 + B T) xi_T
    dh/dt = -F2(s) T - eps(s) h + sigma_h xi_h

Each coefficient may carry an annual cycle c(s) = c0 + c1 cos(2 pi s / 12) + c2 sin(2 pi s / 12) of the calendar
month s (0 = January); the seasonal growth rate R(s) is the textbook origin of the spring predictability barrier
(Stein et al. 2014, J. Climate 27:4985). B != 0 is multiplicative (state-dependent) noise (Levine & Jin 2017, Clim.
Dyn. 48:3241 discuss its role in ENSO asymmetry). The constant-coefficient system has eigenvalues
``lambda = (R - eps) / 2 +- sqrt((R + eps)^2 / 4 - F1 F2)`` (:func:`ro_eigen`): growth rate (R - eps) / 2 and,
when F1 F2 > (R + eps)^2 / 4, period 2 pi / sqrt(F1 F2 - (R + eps)^2 / 4).

Parameters are estimated (:func:`fit_recharge`) by least squares of the centred monthly tendencies on the state
(times the annual harmonics), the method of the extended recharge oscillator of Zhao et al. (2024, Nature 630:891);
the noise amplitudes from the residuals, B by maximum likelihood of the T residuals.

**Delayed oscillator** (Suarez & Schopf 1988, J. Atmos. Sci. 45:3283; Battisti & Hirst 1989, J. Atmos. Sci.
46:1687)::

    dT/dt = a T - b T(t - tau) - c T^3

linearised roots ``lambda = a + W_k(-b tau exp(-a tau)) / tau`` (Lambert W, :func:`do_roots`).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ROParams", "ro_eigen", "ro_coefficients", "ro_tendency", "simulate_recharge", "fit_recharge",
           "do_rhs", "do_roots", "simulate_delayed", "oscillation_stats", "PARAM_NAMES"]

PARAM_NAMES = ("R", "F1", "F2", "eps")
_W = 2 * np.pi / 12.0


@dataclass
class ROParams:
    """Recharge-oscillator coefficients per month: ``coef[name] = (c0, c1, c2)`` (annual harmonics; c1 = c2 = 0 for a
    constant coefficient), noise amplitudes per sqrt(month), multiplicative noise B [1/K]."""
    coef: dict[str, tuple[float, float, float]]
    sigma_T: float = 0.0
    sigma_h: float = 0.0
    B: float = 0.0
    info: dict = field(default_factory=dict)

    @classmethod
    def constant(cls, R: float, F1: float, F2: float, eps: float, **kw) -> ROParams:
        return cls({"R": (R, 0.0, 0.0), "F1": (F1, 0.0, 0.0), "F2": (F2, 0.0, 0.0), "eps": (eps, 0.0, 0.0)}, **kw)

    def mean(self) -> dict[str, float]:
        return {k: float(v[0]) for k, v in self.coef.items()}

    def to_dict(self) -> dict:
        return {"coef": {k: list(map(float, v)) for k, v in self.coef.items()}, "sigma_T": self.sigma_T,
                "sigma_h": self.sigma_h, "B": self.B, **({"info": self.info} if self.info else {})}


def ro_eigen(R: float, F1: float, F2: float, eps: float) -> dict[str, float]:
    """Eigenvalues of the constant-coefficient recharge oscillator: growth rate [1/month], period [months] (inf
    when the eigenvalues are real) and the two eigenvalues."""
    tr, det = R - eps, -R * eps + F1 * F2
    disc = tr ** 2 / 4 - det
    if disc < 0:
        om = float(np.sqrt(-disc))
        return {"growth": tr / 2, "period": 2 * np.pi / om, "omega": om,
                "lambda": [complex(tr / 2, om), complex(tr / 2, -om)]}
    s = float(np.sqrt(disc))
    return {"growth": tr / 2 + s, "period": float("inf"), "omega": 0.0,
            "lambda": [complex(tr / 2 + s), complex(tr / 2 - s)]}


def ro_coefficients(p: ROParams, month: np.ndarray) -> dict[str, np.ndarray]:
    """Coefficients at (fractional) calendar month(s) ``month`` (0 = mid-January... any real number, period 12)."""
    m = np.asarray(month, float)
    c, s = np.cos(_W * m), np.sin(_W * m)
    return {k: v[0] + v[1] * c + v[2] * s for k, v in p.coef.items()}


def ro_tendency(p: ROParams, X: np.ndarray, month) -> np.ndarray:
    """Deterministic tendency (dT/dt, dh/dt) of states X (..., 2) at calendar month(s) ``month``."""
    k = ro_coefficients(p, month)
    T, h = X[..., 0], X[..., 1]
    return np.stack([k["R"] * T + k["F1"] * h, -k["F2"] * T - k["eps"] * h], axis=-1)


def simulate_recharge(p: ROParams, x0, n_months: int, *, month0: float = 0.0, sub: int = 10, n_members: int = 1,
                      stochastic: bool = True, seed: int = 0, method: str = "rk4",
                      month_offsets: np.ndarray | None = None) -> np.ndarray:
    """Integrate from x0 ((2,) or (n, 2) states) for ``n_months``; returns (n_members, n, n_months + 1, 2) samples at
    month ends. Deterministic part: RK4 (``method="rk4"``) or Euler; noise: Euler-Maruyama with ``sub`` steps per
    month (Ito; the multiplicative factor is clipped at zero). ``month_offsets`` (n,) gives each state its own start
    month (added to ``month0``)."""
    x0 = np.atleast_2d(np.asarray(x0, float))
    rng = np.random.default_rng(seed)
    dt = 1.0 / sub
    X = np.broadcast_to(x0, (n_members,) + x0.shape).copy()
    out = np.empty((n_members, x0.shape[0], n_months + 1, 2))
    out[:, :, 0] = X
    m = float(month0) + (np.zeros(x0.shape[0]) if month_offsets is None else np.asarray(month_offsets, float))
    with np.errstate(all="ignore"):
        for k in range(n_months):
            for _ in range(sub):
                if method == "rk4":
                    k1 = ro_tendency(p, X, m)
                    k2 = ro_tendency(p, X + 0.5 * dt * k1, m + 0.5 * dt)
                    k3 = ro_tendency(p, X + 0.5 * dt * k2, m + 0.5 * dt)
                    k4 = ro_tendency(p, X + dt * k3, m + dt)
                    X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                else:
                    X = X + dt * ro_tendency(p, X, m)
                if stochastic and (p.sigma_T > 0 or p.sigma_h > 0):
                    amp = p.sigma_T * np.maximum(1.0 + p.B * X[..., 0], 0.0)
                    X = X + np.sqrt(dt) * np.stack([amp * rng.standard_normal(X.shape[:-1]),
                                                    p.sigma_h * rng.standard_normal(X.shape[:-1])], axis=-1)
                m += dt
            out[:, :, k + 1] = X
    return out


def fit_recharge(T: np.ndarray, h: np.ndarray, month: np.ndarray | None = None, *, seasonal: Sequence[str] = (),
                 multiplicative: bool = False, lead: int = 1, sub: int = 4) -> ROParams:
    """Recharge oscillator fitted to monthly series T, h (pairs with a NaN are skipped).

    The drift is estimated by least squares of ``lead``-month predictions of the integrated model (RK4, ``sub``
    steps per month) from every observed state, each variable scaled by its standard deviation -- for a linear SDE
    the conditional mean is the deterministic flow, so this is unbiased where a regression of finite-difference
    tendencies on the state is not (a centred difference correlates with the noise). The start is the matrix
    logarithm of the one-month propagator C(1) C(0)^-1 (the linear inverse model estimate). Coefficients named in
    ``seasonal`` get annual harmonics of the calendar month (0-11). Noise: the standard deviation of the one-month
    residuals (the diffusion over one month, neglecting the drift's effect within the month); with
    ``multiplicative`` the factor (1 + B T) of the T noise by maximum likelihood.
    """
    from scipy.linalg import logm
    from scipy.optimize import least_squares
    T, h = np.asarray(T, float), np.asarray(h, float)
    month = np.zeros(len(T)) if month is None else np.asarray(month, float)
    X = np.stack([T, h], 1)
    i0 = np.arange(len(T) - lead)
    ok = np.isfinite(X[i0]).all(1) & np.isfinite(X[i0 + lead]).all(1)
    i0 = i0[ok]
    X0, X1, m0 = X[i0], X[i0 + lead], month[i0]
    x0, x1 = X[i0[i0 + 1 < len(T)]], X[i0[i0 + 1 < len(T)] + 1]
    G = np.linalg.lstsq(x0, x1, rcond=None)[0].T
    A = np.real(logm(G))
    names = list(PARAM_NAMES)
    start = {"R": A[0, 0], "F1": A[0, 1], "F2": -A[1, 0], "eps": -A[1, 1]}
    layout = [(n, 3 if n in seasonal else 1) for n in names]
    theta0 = np.concatenate([[start[n]] + [0.0] * (k - 1) for n, k in layout])
    sd = np.nanstd(X, axis=0)

    def unpack(th):
        coef, i = {}, 0
        for n, k in layout:
            v = th[i:i + k]
            coef[n] = (float(v[0]), float(v[1]) if k == 3 else 0.0, float(v[2]) if k == 3 else 0.0)
            i += k
        return ROParams(coef)

    def predict(p):
        return simulate_recharge(p, X0, lead, month0=0.0, sub=sub, stochastic=False, month_offsets=m0)[0, :, -1]

    def resid(th):
        r = ((predict(unpack(th)) - X1) / sd).ravel()
        return np.where(np.isfinite(r), r, 1e3)

    sol = least_squares(resid, theta0, x_scale=np.maximum(np.abs(theta0), 1e-3), max_nfev=300)
    p = unpack(sol.x)
    one = simulate_recharge(p, X0, 1, sub=sub, stochastic=False, month_offsets=m0)[0, :, -1]
    nxt = X[i0 + 1]
    res = nxt - one
    p.sigma_T, p.sigma_h = float(np.nanstd(res[:, 0])), float(np.nanstd(res[:, 1]))
    if multiplicative:
        from scipy.optimize import minimize_scalar
        r, Tc = res[:, 0], X0[:, 0]
        bound = 0.9 / max(np.abs(Tc).max(), 1e-6)

        def nll(B):
            a = np.maximum(1 + B * Tc, 1e-3)
            return 0.5 * len(r) * np.log(np.mean((r / a) ** 2)) + np.sum(np.log(a))

        p.B = float(minimize_scalar(nll, bounds=(-bound, bound), method="bounded").x)
        p.sigma_T = float(np.sqrt(np.mean((r / np.maximum(1 + p.B * Tc, 1e-3)) ** 2)))
    p.info = {"n": int(len(i0)), "seasonal": list(seasonal), "lead": lead,
              "lim_start": {k: float(v) for k, v in start.items()},
              "r2_T": float(1 - np.nanvar(res[:, 0]) / np.nanvar(nxt[:, 0] - X0[:, 0])),
              "r2_h": float(1 - np.nanvar(res[:, 1]) / np.nanvar(nxt[:, 1] - X0[:, 1])), "eigen": _eig_json(p)}
    return p


def _eig_json(p: ROParams) -> dict:
    e = ro_eigen(**p.mean())
    return {"growth": e["growth"], "period": e["period"] if np.isfinite(e["period"]) else None}


# ---------------------------------------------------------------------------
# Delayed oscillator
# ---------------------------------------------------------------------------

def do_rhs(x, xt, a: float = 1.0, b: float = 0.75, c: float = 1.0):
    """Battisti-Hirst / Suarez-Schopf delayed oscillator dT/dt = a T - b T(t - tau) - c T^3."""
    return a * x - b * xt - c * x ** 3


def do_roots(a: float, b: float, tau: float, branches: Sequence[int] = (0, -1, 1)) -> list[complex]:
    """Roots of the linear characteristic equation lambda = a - b exp(-lambda tau) on Lambert-W ``branches``,
    sorted by decreasing real part."""
    from scipy.special import lambertw
    z = -b * tau * np.exp(-a * tau)
    roots = [complex(a + lambertw(z, k) / tau) for k in branches]
    return sorted(roots, key=lambda r: -r.real)


def simulate_delayed(a: float, b: float, c: float, tau: float, history, n_steps: int, dt: float,
                     method: str = "rk4") -> np.ndarray:
    """Method of steps on a uniform grid (``tau`` a multiple of ``dt``): ``history`` holds the last tau / dt + 1
    values (oldest first); returns the n_steps new values. RK4 uses the mean of the delayed values at the half step
    (second order in the delayed argument) unless the grid is refined."""
    m = int(round(tau / dt))
    if abs(m * dt - tau) > 1e-9 * max(tau, 1):
        raise ValueError("tau must be a multiple of dt")
    hist = list(np.asarray(history, float)[-(m + 1):])
    out = np.empty(n_steps)
    for k in range(n_steps):
        x, xa, xb = hist[-1], hist[-1 - m], hist[-m] if m > 0 else hist[-1]
        if method == "rk4":
            xm = 0.5 * (xa + xb)
            k1 = do_rhs(x, xa, a, b, c)
            k2 = do_rhs(x + 0.5 * dt * k1, xm, a, b, c)
            k3 = do_rhs(x + 0.5 * dt * k2, xm, a, b, c)
            k4 = do_rhs(x + dt * k3, xb, a, b, c)
            new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            new = x + dt * do_rhs(x, xa, a, b, c)
        hist.append(new)
        out[k] = new
    return out


def oscillation_stats(x: np.ndarray, dt: float, *, skip: int = 0) -> dict[str, float]:
    """Period (mean spacing of upward zero crossings, linearly interpolated) and growth rate (slope of log |peak|
    over successive extrema) of a linear oscillation sampled every ``dt``."""
    x = np.asarray(x, float)[skip:]
    t = np.arange(len(x)) * dt
    up = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
    tc = t[up] + dt * (-x[up]) / (x[up + 1] - x[up])
    period = float(np.mean(np.diff(tc))) if len(tc) > 1 else float("nan")
    pk = np.nonzero((np.abs(x[1:-1]) >= np.abs(x[:-2])) & (np.abs(x[1:-1]) > np.abs(x[2:])))[0] + 1
    if len(pk) > 2:
        # parabolic refinement of each |x| peak
        a, b_, c_ = np.abs(x[pk - 1]), np.abs(x[pk]), np.abs(x[pk + 1])
        off = 0.5 * (a - c_) / (a - 2 * b_ + c_)
        tp = t[pk] + off * dt
        amp = b_ - 0.25 * (a - c_) * off
        growth = float(np.polyfit(tp, np.log(amp), 1)[0])
    else:
        growth = float("nan")
    return {"period": period, "growth": growth}
