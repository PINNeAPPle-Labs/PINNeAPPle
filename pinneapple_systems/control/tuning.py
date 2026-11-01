"""Controller tuning and robustness: FOPDT identification, SIMC / IMC / Ziegler-Nichols rules, PID pole placement
for second-order plants (stable or not), LQR, tuning by closed-loop optimisation, and Monte Carlo robustness.

Rules and their sources:

* **SIMC** (Skogestad 2003, *J. Process Control* 13, 291-309):
  ``Kc = tau / (K (tau_c + theta))``, ``Ti = min(tau, 4 (tau_c + theta))``, default ``tau_c = theta``.
* **IMC-PI / IMC-PID** for FOPDT (Rivera, Morari & Skogestad 1986): PI ``Kc = tau / (K (lambda + theta))``,
  ``Ti = tau``; PID ``Kc = (tau + theta/2) / (K (lambda + theta/2))``, ``Ti = tau + theta/2``,
  ``Td = tau theta / (2 tau + theta)``.
* **Ziegler-Nichols** (1942): the ultimate-gain table (``Kc = 0.6 Ku``, ``Ti = Tu/2``, ``Td = Tu/8`` for PID) and
  the reaction-curve table (``Kc = 1.2 tau / (K theta)``, ``Ti = 2 theta``, ``Td = theta/2``).
* **Pole placement** for ``m y'' + c y' + k y = b u``: a PID with derivative on a measured rate gives the closed
  loop ``m s^3 + (c + b Kd) s^2 + (k + b Kp) s + b Ki``. Matching ``(s^2 + 2 zeta wn s + wn^2)(s + p)`` fixes the
  three gains. This holds for ``k < 0`` (a statically unstable plant, such as a submarine in depth) as well.
* **LQR**: ``scipy.linalg.solve_continuous_are`` / ``solve_discrete_are``.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import linalg, optimize

from .analysis import loop_margins

__all__ = ["fit_fopdt", "simc_pid", "imc_pid", "ziegler_nichols", "ultimate_gain", "place_pid_second_order",
           "lqr", "dlqr", "tune_by_optimization", "robustness"]


def fit_fopdt(t: Sequence[float], y: Sequence[float], du: float, y0: float | None = None) -> dict[str, float]:
    """Fit ``K exp(-theta s) / (tau s + 1)`` to an open-loop step response by least squares.

    Parameters
    ----------
    t : array_like [s]
        Time from the step.
    y : array_like
        Output (any unit).
    du : float
        Size of the input step (input units).
    y0 : float, optional
        Output before the step (default ``y[0]``).

    Returns
    -------
    K : float
        Static gain (output units per input unit).
    tau : float [s]
        Time constant.
    theta : float [s]
        Dead time.
    rmse : float
        Root-mean-square fit error (output units).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y0 = float(y[0]) if y0 is None else float(y0)
    dy = y - y0

    def model(p):
        K, tau, theta = p
        s = np.clip(t - theta, 0.0, None)
        return K * du * (1.0 - np.exp(-s / max(tau, 1e-12)))

    K0 = dy[-1] / du if du else 1.0
    i63 = np.where(np.abs(dy) >= 0.632 * abs(dy[-1]))[0]
    tau0 = float(t[i63[0]] - t[0]) if i63.size else float(t[-1] - t[0]) / 3
    sol = optimize.least_squares(lambda p: model(p) - dy, [K0, max(tau0, 1e-3), 0.05 * tau0],
                                 bounds=([-np.inf, 1e-9, 0.0], [np.inf, np.inf, t[-1] - t[0]]))
    K, tau, theta = sol.x
    return {"K": float(K), "tau": float(tau), "theta": float(theta),
            "rmse": float(np.sqrt(np.mean((model(sol.x) - dy) ** 2)))}


def simc_pid(K: float, tau: float, theta: float, tau_c: float | None = None) -> dict[str, float]:
    """SIMC PI rule for a FOPDT plant (Skogestad 2003).

    Parameters
    ----------
    K : float
        Plant gain (output units per input unit).
    tau, theta : float [s]
        Time constant and dead time.
    tau_c : float [s], optional
        Desired closed-loop time constant (default ``theta``, the "tight" setting).

    Returns
    -------
    kc : float
        Proportional gain (input units per output unit).
    ti : float [s]
        Integral time.
    kp, ki, kd : float
        Parallel-form gains for :class:`~pinneapple_systems.control.loop.PID` (``ki = kc / ti``, ``kd = 0``).
    """
    tau_c = theta if tau_c is None else float(tau_c)
    kc = tau / (K * (tau_c + theta))
    ti = min(tau, 4.0 * (tau_c + theta))
    return {"kc": float(kc), "ti": float(ti), "kp": float(kc), "ki": float(kc / ti), "kd": 0.0}


def imc_pid(K: float, tau: float, theta: float, lam: float | None = None, kind: str = "pid") -> dict[str, float]:
    """IMC rule for a FOPDT plant (Rivera, Morari & Skogestad 1986).

    Parameters
    ----------
    K : float
        Plant gain (output units per input unit).
    tau, theta : float [s]
        Time constant and dead time.
    lam : float [s], optional
        Closed-loop filter time constant (default ``max(0.25 tau, 0.8 theta)``).
    kind : {'pi', 'pid'}
        Controller form.

    Returns
    -------
    kc : float
        Proportional gain (input units per output unit).
    ti : float [s]
        Integral time.
    td : float [s]
        Derivative time (0 for PI).
    kp, ki, kd : float
        Parallel-form gains.
    """
    lam = max(0.25 * tau, 0.8 * theta) if lam is None else float(lam)
    if kind == "pi":
        kc, ti, td = tau / (K * (lam + theta)), tau, 0.0
    elif kind == "pid":
        kc = (tau + theta / 2) / (K * (lam + theta / 2))
        ti = tau + theta / 2
        td = tau * theta / (2 * tau + theta)
    else:
        raise ValueError("kind must be 'pi' or 'pid'")
    return {"kc": float(kc), "ti": float(ti), "td": float(td), "kp": float(kc), "ki": float(kc / ti),
            "kd": float(kc * td)}


def ultimate_gain(L_unit, delay: float = 0.0, **kw) -> dict[str, float]:
    """Ultimate gain and period from the open loop at unit controller gain: ``Ku = gain margin``,
    ``Tu = 2 pi / w180``.

    Returns
    -------
    ku : float
        Ultimate gain (input units per output unit).
    tu : float [s]
        Ultimate period.
    """
    m = loop_margins(L_unit, delay=delay, **kw)
    return {"ku": m["gain_margin"], "tu": float(2 * math.pi / m["phase_crossover"])
            if np.isfinite(m["phase_crossover"]) else float("nan")}


def ziegler_nichols(kind: str = "pid", ku: float | None = None, tu: float | None = None, K: float | None = None,
                    tau: float | None = None, theta: float | None = None) -> dict[str, float]:
    """Ziegler-Nichols tuning from the ultimate gain (``ku``, ``tu``) or the reaction curve (``K``, ``tau``,
    ``theta``).

    Returns
    -------
    kc : float
        Proportional gain (input units per output unit).
    ti : float [s]
        Integral time (``inf`` for P).
    td : float [s]
        Derivative time.
    kp, ki, kd : float
        Parallel-form gains.
    """
    if ku is not None and tu is not None:
        table = {"p": (0.5, math.inf, 0.0), "pi": (0.45, tu / 1.2, 0.0), "pid": (0.6, tu / 2, tu / 8)}
        f, ti, td = table[kind]
        kc = f * ku
    elif None not in (K, tau, theta):
        a = K * theta / tau
        table = {"p": (1.0 / a, math.inf, 0.0), "pi": (0.9 / a, theta / 0.3, 0.0), "pid": (1.2 / a, 2 * theta,
                                                                                         theta / 2)}
        kc, ti, td = table[kind]
    else:
        raise ValueError("give ku and tu, or K, tau and theta")
    ki = 0.0 if not np.isfinite(ti) else kc / ti
    return {"kc": float(kc), "ti": float(ti), "td": float(td), "kp": float(kc), "ki": float(ki), "kd": float(kc * td)}


def place_pid_second_order(m: float, c: float, k: float, b: float, wn: float, zeta: float = 0.7,
                           p: float | None = None) -> dict[str, Any]:
    """PID gains that place the closed-loop poles of ``m y'' + c y' + k y = b u``.

    With derivative on a measured rate, the closed loop is ``m s^3 + (c + b Kd) s^2 + (k + b Kp) s + b Ki``.
    It is matched to ``m (s^2 + 2 zeta wn s + wn^2)(s + p)``. A negative ``k`` (statically unstable plant) is
    allowed.

    Parameters
    ----------
    m : float
        Inertia (output units: kg for a position in m, or any consistent set).
    c : float
        Damping of the linearised plant (same set; ``m`` per second).
    k : float
        Stiffness of the linearised plant (``m`` per second squared); negative when unstable.
    b : float
        Input gain (``m`` times output units per second squared per input unit).
    wn : float [rad/s]
        Natural frequency of the dominant pair.
    zeta : float [-]
        Damping ratio of the dominant pair.
    p : float [1/s], optional
        Real integral pole (default ``wn / 4``: slow integral action that removes the offset).

    Returns
    -------
    kp : float
        Proportional gain (input units per output unit).
    ki : float
        Integral gain (input units per output unit-second).
    kd : float
        Derivative gain (input units per output unit per second).
    poles : ndarray [1/s]
        Closed-loop poles obtained with these gains (check against the request).
    """
    p = wn / 4.0 if p is None else float(p)
    a2 = m * (2 * zeta * wn + p)
    a1 = m * (wn ** 2 + 2 * zeta * wn * p)
    a0 = m * wn ** 2 * p
    kd = (a2 - c) / b
    kp = (a1 - k) / b
    ki = a0 / b
    poles = np.roots([m, c + b * kd, k + b * kp, b * ki])
    return {"kp": float(kp), "ki": float(ki), "kd": float(kd), "poles": poles}


def lqr(A, B, Q, R) -> dict[str, Any]:
    """Continuous-time LQR: ``u = -K x`` minimising ``int (x'Qx + u'Ru) dt``.

    Returns
    -------
    K : ndarray
        Gain (input units per state unit).
    P : ndarray
        Solution of the algebraic Riccati equation.
    poles : ndarray [1/s]
        Eigenvalues of ``A - B K``.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    Q, R = np.atleast_2d(np.asarray(Q, float)), np.atleast_2d(np.asarray(R, float))
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return {"K": K, "P": P, "poles": np.linalg.eigvals(A - B @ K)}


def dlqr(Ad, Bd, Q, R) -> dict[str, Any]:
    """Discrete-time LQR: ``u_k = -K x_k`` minimising ``sum (x'Qx + u'Ru)``.

    Returns
    -------
    K : ndarray
        Gain (input units per state unit).
    P : ndarray
        Solution of the discrete algebraic Riccati equation.
    poles : ndarray [-]
        Eigenvalues of ``Ad - Bd K`` (inside the unit circle when stabilising).
    """
    Ad = np.atleast_2d(np.asarray(Ad, float))
    Bd = np.asarray(Bd, float).reshape(Ad.shape[0], -1)
    Q, R = np.atleast_2d(np.asarray(Q, float)), np.atleast_2d(np.asarray(R, float))
    P = linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return {"K": K, "P": P, "poles": np.linalg.eigvals(Ad - Bd @ K)}


def tune_by_optimization(simulate: Callable[[Mapping[str, float]], Mapping[str, float]],
                         initial: Mapping[str, float], bounds: Mapping[str, tuple[float, float]],
                         objective: str = "iae", constraints: Mapping[str, float] | None = None,
                         maxiter: int = 60, log_scale: bool = True) -> dict[str, Any]:
    """Tune gains by minimising a closed-loop metric under constraints, with Nelder-Mead on a penalty.

    Parameters
    ----------
    simulate : callable
        ``gains -> metrics``: runs the closed loop with these gains and returns a dict containing ``objective`` and
        every key in ``constraints``. Unstable or stopped runs should return ``inf`` or a large value.
    initial : mapping
        Starting gains.
    bounds : mapping
        ``name -> (lo, hi)`` per gain.
    objective : str
        Metric to minimise (e.g. 'iae', 'itae', 'energy').
    constraints : mapping, optional
        ``metric -> upper bound`` (e.g. ``{'overshoot': 10, 'ms': 2.0}``).
    maxiter : int
        Iterations of Nelder-Mead.
    log_scale : bool
        Search in the logarithm of the gains (all bounds must then be positive).

    Returns
    -------
    gains : dict
        Best feasible gains found (the initial gains when none improves).
    metrics : dict
        Metrics of the best gains.
    initial_metrics : dict
        Metrics of the initial gains.
    improved : bool
        The objective decreased with all constraints met.
    evaluations : int
        Closed-loop runs.
    """
    names = list(initial)
    lo = np.array([bounds[n][0] for n in names], float)
    hi = np.array([bounds[n][1] for n in names], float)
    cons = dict(constraints or {})
    if log_scale and np.any(lo <= 0):
        raise ValueError("log_scale needs positive bounds")
    fwd = (lambda v: np.log(v)) if log_scale else (lambda v: v)
    inv = (lambda z: np.exp(z)) if log_scale else (lambda z: z)
    count = {"n": 0}
    best = {"J": math.inf, "g": dict(initial), "m": None}

    def evaluate(g):
        count["n"] += 1
        m = dict(simulate(g))
        J = float(m.get(objective, math.inf))
        feasible = np.isfinite(J) and all(float(m.get(k, math.inf)) <= v for k, v in cons.items())
        return J, feasible, m

    J0, f0, m0 = evaluate(dict(initial))
    if f0:
        best.update(J=J0, m=m0)

    def cost(z):
        v = np.clip(inv(z), lo, hi)
        g = dict(zip(names, map(float, v), strict=True))
        J, feas, m = evaluate(g)
        if feas and J < best["J"]:
            best.update(J=J, g=g, m=m)
        pen = sum(max(0.0, float(m.get(k, math.inf)) - b) / max(abs(b), 1e-12) for k, b in cons.items())
        return (J if np.isfinite(J) else 1e30) * (1.0 + 10.0 * pen) + (1e30 if not np.isfinite(J) else 0.0)

    z0 = fwd(np.clip(np.array([initial[n] for n in names], float), lo, hi))
    optimize.minimize(cost, z0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6})
    return {"gains": best["g"], "metrics": best["m"] if best["m"] is not None else m0, "initial_metrics": m0,
            "improved": bool(best["m"] is not None and best["J"] < J0 - 1e-12 * max(1.0, abs(J0))),
            "evaluations": count["n"]}


def robustness(run: Callable[[Mapping[str, float]], Mapping[str, Any]], ranges: Mapping[str, tuple[float, float]],
               n: int = 64, seed: int = 0, stable_key: str = "stable",
               metric_keys: Sequence[str] = ()) -> dict[str, Any]:
    """Monte Carlo over uncertain parameters: how often the loop stays stable and how far it strays.

    Parameters
    ----------
    run : callable
        ``params -> result``: builds and runs the loop for one parameter set; the result holds ``stable_key``
        (bool) and the metrics in ``metric_keys``.
    ranges : mapping
        ``name -> (lo, hi)``, sampled uniformly (Latin hypercube).
    n : int
        Samples.
    seed : int
        Random seed.
    metric_keys : sequence of str
        Metrics to summarise.

    Returns
    -------
    fraction_stable : float [-]
        Share of samples with ``stable_key`` true.
    samples : list of dict
        Parameters and results per sample.
    worst : dict
        Per metric: the largest value and the parameters that gave it.
    quantiles : dict
        Per metric: 5 %, 50 % and 95 % quantiles (metric units).
    """
    from scipy.stats import qmc
    names = list(ranges)
    U = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    lo = np.array([ranges[k][0] for k in names])
    hi = np.array([ranges[k][1] for k in names])
    samples = []
    for row in lo + U * (hi - lo):
        p = dict(zip(names, map(float, row), strict=True))
        r = dict(run(copy.deepcopy(p)))
        samples.append({"params": p, **r})
    stable = np.array([bool(s.get(stable_key, False)) for s in samples])
    worst, quant = {}, {}
    for k in metric_keys:
        v = np.array([float(s.get(k, np.nan)) for s in samples])
        if np.all(np.isnan(v)):
            continue
        i = int(np.nanargmax(v))
        worst[k] = {"value": float(v[i]), "params": samples[i]["params"]}
        quant[k] = {q: float(np.nanquantile(v, q)) for q in (0.05, 0.5, 0.95)}
    return {"fraction_stable": float(stable.mean()) if stable.size else float("nan"), "samples": samples,
            "worst": worst, "quantiles": quant}
