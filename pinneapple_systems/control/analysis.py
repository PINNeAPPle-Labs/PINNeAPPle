"""Linear analysis of any dynamic model ``dx/dt = f(x, u, d, t)``: trim, linearisation, stability, controllability,
loop margins and step-response metrics.

The model is a plain Python callable. Nothing here knows about a particular plant::

    from pinneapple_systems.control import analysis as A

    f = lambda x, u, d, t: np.array([x[1], -2.0 * x[0] - 0.4 * x[1] + u[0]])
    lin = A.linearize(f, x=[0.0, 0.0], u=[0.0])
    A.stability(lin["A"])["classification"]          # 'stable (oscillatory)'

Finite differences are central, with a step scaled to each variable. :func:`linearize` reports the difference
between two step sizes as an error estimate, so a Jacobian that depends on the step is visible.

Margins (:func:`loop_margins`) are measured on the open loop ``L(jw)``, given as a transfer function or as any
callable ``w -> L(jw)``. The crossings are found on a logarithmic grid and refined by root finding:

* gain margin at the phase crossover;
* phase margin at the gain crossover;
* delay margin = phase margin / gain-crossover frequency;
* peak sensitivity ``Ms = max |1 / (1 + L)|``.

References: Astrom & Murray, *Feedback Systems* (2nd ed., 2021), ch. 5-6 and 10; Skogestad & Postlethwaite,
*Multivariable Feedback Control* (2005), ch. 2.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import optimize, signal

__all__ = ["equilibrium", "linearize", "stability", "controllability", "observability", "transfer_function",
           "loop_margins", "root_locus", "step_metrics", "simulate_linear_step"]

Model = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


def _vec(v, n: int | None = None) -> np.ndarray:
    a = np.atleast_1d(np.asarray([] if v is None else v, dtype=float)).ravel()
    if n is not None and a.size == 0:
        a = np.zeros(n)
    return a


def _call(f: Model, x, u, d, t) -> np.ndarray:
    return np.atleast_1d(np.asarray(f(x, u, d, t), dtype=float)).ravel()


def equilibrium(f: Model, x0: Sequence[float], u0: Sequence[float] | None = None, d0: Sequence[float] | None = None,
                t: float = 0.0, free: Sequence[int] | None = None, tol: float = 1e-10) -> dict[str, Any]:
    """Trim point of ``dx/dt = f(x, u, d, t)``: the state where the derivatives vanish, by root finding.

    Parameters
    ----------
    f : callable
        ``f(x, u, d, t) -> dx/dt``.
    x0 : array_like
        Initial guess of the state; the components not listed in ``free`` stay fixed.
    u0, d0 : array_like, optional
        Input and disturbance held during the search.
    t : float [s]
        Time passed to ``f``.
    free : sequence of int, optional
        Indices of the state adjusted by the search (default: all). The remaining derivatives must vanish with the
        rest fixed, which ``residual_norm`` reports.

    Returns
    -------
    x : ndarray
        Trim state (units of the caller's state).
    residual : ndarray
        ``f`` at the trim state (state units per second).
    residual_norm : float
        Euclidean norm of the residual (state units per second).
    converged : bool
        Root finder converged and ``residual_norm <= tol``.
    message : str
        Root finder message.
    """
    x0 = _vec(x0)
    u0 = _vec(u0)
    d0 = _vec(d0)
    idx = np.arange(x0.size) if free is None else np.asarray(list(free), dtype=int)

    def F(z):
        x = x0.copy()
        x[idx] = z
        r = _call(f, x, u0, d0, t)
        return r[idx] if free is not None else r

    sol = optimize.root(F, x0[idx], method="hybr", tol=tol)
    x = x0.copy()
    x[idx] = sol.x
    res = _call(f, x, u0, d0, t)
    rn = float(np.linalg.norm(res))
    return {"x": x, "residual": res, "residual_norm": rn, "converged": bool(sol.success and rn <= max(tol, 1e-8)),
            "message": str(sol.message)}


def _jac(fun: Callable[[np.ndarray], np.ndarray], z: np.ndarray, rel: float) -> np.ndarray:
    z = np.asarray(z, float)
    f0 = fun(z)
    J = np.zeros((f0.size, z.size))
    for j in range(z.size):
        h = rel * max(1.0, abs(z[j]))
        zp, zm = z.copy(), z.copy()
        zp[j] += h
        zm[j] -= h
        J[:, j] = (fun(zp) - fun(zm)) / (2 * h)
    return J


def linearize(f: Model, x: Sequence[float], u: Sequence[float] | None = None, d: Sequence[float] | None = None,
              t: float = 0.0, rel_step: float = 1e-5) -> dict[str, Any]:
    """Jacobians of ``f`` at ``(x, u, d)``: ``dx/dt ~ A dx + B du + E dd`` by central differences.

    Parameters
    ----------
    f : callable
        ``f(x, u, d, t) -> dx/dt``.
    x, u, d : array_like
        Operating point (the caller's units).
    t : float [s]
        Time passed to ``f``.
    rel_step : float [-]
        Finite-difference step relative to ``max(1, |value|)``.

    Returns
    -------
    A : ndarray
        State Jacobian (1/s).
    B : ndarray
        Input Jacobian (state units per input unit per second).
    E : ndarray
        Disturbance Jacobian (state units per disturbance unit per second).
    f0 : ndarray
        ``f`` at the point (state units per second); not zero away from equilibrium.
    jacobian_error : float [-]
        Largest relative change of A, B and E when the step is multiplied by 10: a Jacobian that depends on the
        step (non-smooth model, too small a step) shows here.
    """
    x = _vec(x)
    u = _vec(u)
    d = _vec(d)

    def blocks(h):
        A = _jac(lambda z: _call(f, z, u, d, t), x, h)
        B = _jac(lambda z: _call(f, x, z, d, t), u, h) if u.size else np.zeros((x.size, 0))
        E = _jac(lambda z: _call(f, x, u, z, t), d, h) if d.size else np.zeros((x.size, 0))
        return A, B, E

    A, B, E = blocks(rel_step)
    A2, B2, E2 = blocks(rel_step * 10)
    err = 0.0
    for M, M2 in ((A, A2), (B, B2), (E, E2)):
        if M.size:
            err = max(err, float(np.max(np.abs(M - M2)) / max(np.max(np.abs(M)), 1e-300)))
    return {"A": A, "B": B, "E": E, "f0": _call(f, x, u, d, t), "jacobian_error": err}


def stability(A: Sequence[Sequence[float]], dt: float | None = None, tol: float = 1e-9) -> dict[str, Any]:
    """Eigen-analysis of a linear(ised) system.

    Parameters
    ----------
    A : ndarray
        Continuous-time state matrix (1/s), or a discrete-time transition matrix when ``dt`` is given.
    dt : float [s], optional
        Sample time of a discrete-time ``A``. Its eigenvalues ``z`` are mapped to ``s = log(z) / dt``.
    tol : float [1/s]
        Real parts within ``tol`` of zero count as marginal.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues of ``A`` as given (1/s for continuous time, dimensionless for discrete time).
    s : ndarray
        Continuous-time equivalents (1/s).
    classification : str
        One of 'stable', 'stable (oscillatory)', 'marginal', 'unstable', 'unstable (oscillatory)'.
    max_real : float [1/s]
        Largest real part of ``s``.
    modes : list of dict
        Per eigenvalue:
        ``s`` (1/s), ``growth_rate`` [1/s], ``time_constant`` [s], ``doubling_time`` [s] (unstable modes),
        ``half_life`` [s] (stable modes), ``frequency`` [rad/s], ``damping_ratio`` [-] and the eigenvector.
    unstable_directions : list of ndarray
        Real parts of the eigenvectors of the unstable modes (state units).
    """
    A = np.atleast_2d(np.asarray(A, float))
    lam, V = np.linalg.eig(A)
    if dt is None:
        s = lam.astype(complex)
    else:
        z = lam.astype(complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(z) > 0, np.log(np.where(np.abs(z) > 0, z, 1.0)) / dt, -np.inf + 0j)
    modes = []
    for k, sk in enumerate(s):
        sig, wd = float(sk.real), float(abs(sk.imag))
        wn = float(abs(sk))
        mode = {"s": complex(sk), "growth_rate": sig, "frequency": wd,
                "damping_ratio": float(-sig / wn) if wn > 0 else float("nan"),
                "time_constant": float(1.0 / abs(sig)) if abs(sig) > tol else float("inf"),
                "eigenvector": V[:, k]}
        if sig > tol:
            mode["doubling_time"] = float(np.log(2.0) / sig)
        elif sig < -tol:
            mode["half_life"] = float(np.log(2.0) / -sig)
        modes.append(mode)
    max_real = float(np.max(s.real)) if s.size else float("-inf")
    osc = bool(np.any(np.abs(s.imag) > tol))
    if max_real > tol:
        osc_u = bool(np.any((s.real > tol) & (np.abs(s.imag) > tol)))
        cls = "unstable (oscillatory)" if osc_u else "unstable"
    elif max_real >= -tol:
        cls = "marginal"
    else:
        cls = "stable (oscillatory)" if osc else "stable"
    unstable = [np.real(m["eigenvector"]) for m in modes if m["growth_rate"] > tol]
    return {"eigenvalues": lam, "s": s, "classification": cls, "max_real": max_real, "modes": modes,
            "unstable_directions": unstable}


def controllability(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Kalman controllability matrix ``[B, AB, ..., A^(n-1) B]``.

    Returns
    -------
    rank : int
        Rank of the controllability matrix (full rank = ``n``: every state can be steered).
    n : int
        Number of states.
    controllable : bool
        ``rank == n``.
    condition_number : float [-]
        Ratio of the largest to the smallest singular value: large means some directions need large inputs.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    blocks = [B]
    for _ in range(A.shape[0] - 1):
        blocks.append(A @ blocks[-1])
    C = np.hstack(blocks)
    sv = np.linalg.svd(C, compute_uv=False)
    rank = int(np.linalg.matrix_rank(C))
    return {"rank": rank, "n": A.shape[0], "controllable": rank == A.shape[0],
            "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")}


def observability(A: Sequence[Sequence[float]], C: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Kalman observability matrix ``[C; CA; ...; C A^(n-1)]``.

    Returns
    -------
    rank : int
        Rank of the observability matrix.
    n : int
        Number of states.
    observable : bool
        ``rank == n``: every state can be reconstructed from the outputs.
    condition_number : float [-]
        Ratio of the largest to the smallest singular value.
    """
    A = np.atleast_2d(np.asarray(A, float))
    C = np.asarray(C, float).reshape(-1, A.shape[0])
    r = controllability(A.T, C.T)
    return {"rank": r["rank"], "n": r["n"], "observable": r["controllable"], "condition_number": r["condition_number"]}


def transfer_function(A, B, C, D=None, input_index: int = 0) -> dict[str, Any]:
    """Transfer function ``Y(s)/U(s)`` of one input of a state-space model (``scipy.signal.ss2tf``).

    Returns
    -------
    num : ndarray
        Numerator coefficients, highest power first, one row per output.
    den : ndarray
        Denominator coefficients, highest power first.
    poles : ndarray
        Roots of ``den`` (1/s).
    zeros : list of ndarray
        Roots of each output's numerator (1/s).
    dc_gain : ndarray
        ``G(0)`` per output (output units per input unit); ``inf`` with a pole at the origin.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    C = np.asarray(C, float).reshape(-1, A.shape[0])
    D = np.zeros((C.shape[0], B.shape[1])) if D is None else np.asarray(D, float).reshape(C.shape[0], B.shape[1])
    num, den = signal.ss2tf(A, B, C, D, input=input_index)
    num = np.atleast_2d(num)
    zeros = [np.roots(np.trim_zeros(r, "f")) if np.any(r) else np.array([]) for r in num]
    with np.errstate(divide="ignore", invalid="ignore"):
        dc = np.array([r[-1] / den[-1] if den[-1] != 0 else np.inf for r in num])
    return {"num": num, "den": np.asarray(den), "poles": np.roots(den), "zeros": zeros, "dc_gain": dc}


def _open_loop(L) -> Callable[[np.ndarray], np.ndarray]:
    if callable(L):
        return lambda w: np.asarray(L(np.asarray(w, float)), dtype=complex)
    num, den = L
    num, den = np.atleast_1d(np.asarray(num, float)), np.atleast_1d(np.asarray(den, float))
    return lambda w: np.polyval(num, 1j * np.asarray(w, float)) / np.polyval(den, 1j * np.asarray(w, float))


def loop_margins(L, delay: float = 0.0, w_min: float = 1e-4, w_max: float = 1e4, n: int = 4000) -> dict[str, Any]:
    """Classical margins of an open loop ``L(jw)`` under unit negative feedback.

    Parameters
    ----------
    L : tuple (num, den) or callable
        Open loop as polynomial coefficients in ``s`` (highest power first), or a callable ``w -> L(jw)``.
    delay : float [s]
        Pure transport delay added to the loop (``exp(-j w delay)``).
    w_min, w_max : float [rad/s]
        Frequency range searched.
    n : int
        Grid points before refinement.

    Returns
    -------
    gain_margin : float [-]
        ``1 / |L|`` at the phase crossover (-180 deg); ``inf`` when the phase never crosses it.
    gain_margin_db : float [dB]
        The gain margin in decibels.
    phase_crossover : float [rad/s]
        Frequency of the -180 deg crossing (nan when none).
    phase_margin : float [deg]
        ``180 + angle(L)`` at the gain crossover (``|L| = 1``); nan when ``|L|`` never crosses 1.
    gain_crossover : float [rad/s]
        Frequency where ``|L| = 1`` (nan when none).
    delay_margin : float [s]
        Extra delay that removes the phase margin: ``phase_margin [rad] / gain_crossover``.
    ms : float [-]
        Peak sensitivity ``max |1/(1+L)|`` on the grid (robustness: Ms < 2 is the usual target).
    ms_frequency : float [rad/s]
        Frequency of the sensitivity peak.
    """
    Lf = _open_loop(L)

    def Ld(w):
        return Lf(w) * np.exp(-1j * np.asarray(w, float) * delay)

    w = np.logspace(np.log10(w_min), np.log10(w_max), n)
    Lw = Ld(w)
    mag = np.abs(Lw)
    ph = np.unwrap(np.angle(Lw))
    # gain crossover: last downward crossing of |L| = 1
    wc, pm = float("nan"), float("nan")
    lm = np.log(np.maximum(mag, 1e-300))
    idx = np.where((lm[:-1] > 0) & (lm[1:] <= 0))[0]
    if idx.size:
        i = idx[-1]
        wc = float(optimize.brentq(lambda x: np.log(abs(Ld(np.array([x]))[0])), w[i], w[i + 1]))
        phc = np.interp(np.log(wc), np.log(w), ph)
        pm = float(np.degrees(phc) + 180.0)
        pm = (pm + 180.0) % 360.0 - 180.0
    # phase crossover: first crossing of -180 deg (mod 360)
    wpc, gm = float("nan"), float("inf")
    shifted = (ph + np.pi) / (2 * np.pi)
    k = np.floor(shifted)
    jdx = np.where(k[1:] != k[:-1])[0]
    for j in jdx:
        target = (max(k[j], k[j + 1])) * 2 * np.pi - np.pi

        def g(x, target=target):
            return np.interp(np.log(x), np.log(w), ph) - target
        try:
            x = float(optimize.brentq(g, w[j], w[j + 1]))
        except ValueError:
            continue
        m = float(abs(Ld(np.array([x]))[0]))
        if m > 0:
            wpc, gm = x, 1.0 / m
            break
    S = 1.0 / np.abs(1.0 + Lw)
    iS = int(np.argmax(S))
    dm = float(np.radians(pm) / wc) if np.isfinite(pm) and np.isfinite(wc) and wc > 0 else float("nan")
    return {"gain_margin": float(gm), "gain_margin_db": float(20 * np.log10(gm)) if np.isfinite(gm) else float("inf"),
            "phase_crossover": wpc, "phase_margin": pm, "gain_crossover": wc, "delay_margin": dm,
            "ms": float(S[iS]), "ms_frequency": float(w[iS])}


def step_metrics(t: Sequence[float], y: Sequence[float], reference: float | None = None, y0: float | None = None,
                 settle_band: float = 0.02) -> dict[str, float]:
    """Time-domain metrics of a step response.

    Parameters
    ----------
    t : array_like [s]
        Time samples (increasing).
    y : array_like
        Response (any unit).
    reference : float, optional
        Target value (default: the last sample).
    y0 : float, optional
        Starting value (default: the first sample).
    settle_band : float [-]
        Settling band as a fraction of the step size.

    Returns
    -------
    rise_time : float [s]
        10 % to 90 % of the step.
    overshoot : float [%]
        Peak beyond the reference, as a percentage of the step (0 when none).
    peak : float
        Extreme value in the direction of the step (units of ``y``).
    peak_time : float [s]
        Time of the peak.
    settling_time : float [s]
        Last time outside the settling band (``nan`` when it never settles in the window).
    steady_state_error : float
        ``reference - y[-1]`` (units of ``y``).
    iae : float
        Integral of ``|reference - y|`` (units of ``y`` times seconds).
    ise : float
        Integral of ``(reference - y)^2`` (units of ``y`` squared times seconds).
    itae : float
        Integral of ``t |reference - y|`` (units of ``y`` times seconds squared).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y0 = float(y[0]) if y0 is None else float(y0)
    r = float(y[-1]) if reference is None else float(reference)
    step = r - y0
    e = r - y
    tt = t - t[0]
    out = {"iae": float(np.trapezoid(np.abs(e), t)), "ise": float(np.trapezoid(e ** 2, t)),
           "itae": float(np.trapezoid(tt * np.abs(e), t)), "steady_state_error": float(e[-1])}
    if step == 0:
        out.update(rise_time=float("nan"), overshoot=0.0, peak=float(y[np.argmax(np.abs(y - y0))]),
                   peak_time=float(t[np.argmax(np.abs(y - y0))]), settling_time=float("nan"))
        return out
    p = (y - y0) / step

    def first_cross(level):
        i = np.where(p >= level)[0]
        if not i.size:
            return float("nan")
        i = int(i[0])
        if i == 0:
            return float(t[0])
        return float(np.interp(level, [p[i - 1], p[i]], [t[i - 1], t[i]]))

    out["rise_time"] = first_cross(0.9) - first_cross(0.1)
    ip = int(np.argmax(p))
    out["peak"] = float(y[ip])
    out["peak_time"] = float(t[ip])
    out["overshoot"] = float(max(0.0, (p[ip] - 1.0) * 100.0))
    outside = np.where(np.abs(p - 1.0) > settle_band)[0]
    if not outside.size:
        out["settling_time"] = float(t[0])
    elif outside[-1] == len(t) - 1:
        out["settling_time"] = float("nan")
    else:
        out["settling_time"] = float(t[outside[-1] + 1])
    return out


def simulate_linear_step(A, B, C, D=None, t_end: float = 10.0, n: int = 2001, amplitude: float = 1.0
                         ) -> dict[str, np.ndarray]:
    """Step response of a linear state-space model from rest (``scipy.signal.lsim``).

    Returns
    -------
    t : ndarray [s]
        Time samples.
    y : ndarray
        Output samples, one column per output (output units).
    x : ndarray
        State samples (state units).
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    C = np.asarray(C, float).reshape(-1, A.shape[0])
    D = np.zeros((C.shape[0], B.shape[1])) if D is None else np.asarray(D, float).reshape(C.shape[0], B.shape[1])
    t = np.linspace(0.0, t_end, n)
    u = amplitude * np.ones((n, B.shape[1]))
    _, y, x = signal.lsim((A, B, C, D), u, t)
    return {"t": t, "y": np.atleast_2d(np.asarray(y).T).T, "x": np.asarray(x)}


def _track(poles: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reorder ``poles`` so that branch ``i`` continues from ``reference[i]``: the pair closest in the plane wins.

    ``np.roots`` returns its roots in an arbitrary order, so a sweep drawn straight from it zigzags between branches
    that never cross. Greedy nearest matching is enough here because the gain step is what keeps the branches apart;
    when two poles meet, the pair that swaps is a genuine crossing and either order is right.
    """
    out = np.empty_like(poles)
    taken: set[int] = set()
    for i, ref in enumerate(reference):
        free = [j for j in range(len(poles)) if j not in taken]
        j = min(free, key=lambda k: abs(poles[k] - ref)) if free else i
        taken.add(j)
        out[i] = poles[j]
    return out


def root_locus(num, den, gains, *, discrete: bool = False, refine: int = 60) -> dict[str, Any]:
    """Where the closed-loop poles go as the gain sweeps, and the gain at which the loop crosses over.

    The closed loop is unit negative feedback around ``K * num(s) / den(s)``, so its poles are the roots of
    ``den(s) + K * num(s)``. For each gain in ``gains`` this returns those roots, the branches ordered so each one
    continues from the previous gain, and -- when the sweep passes through it -- the gain at which a pole reaches
    the stability boundary (the imaginary axis in continuous time, the unit circle in the z plane), found by
    bisection on the sign change and reported with the pole it lands on.

    Parameters
    ----------
    num, den : sequence
        Open-loop numerator and denominator, highest power first, in consistent units (``s`` or ``z``).
    gains : sequence
        The gains to sweep, non-negative and in the order they should be swept (they are sorted).
    discrete : bool
        ``True`` for a discrete-time plant: the boundary is ``|z| = 1`` instead of ``Re(s) = 0``.
    refine : int
        Bisection steps used to pin the crossing gain between two samples of the sweep.

    Returns
    -------
    gains : ndarray
        The sweep, sorted.
    poles : ndarray, complex, shape (len(gains), degree)
        Closed-loop poles, one row per gain, branch-ordered along the sweep.
    open_loop_poles : ndarray
        Roots of ``den`` (the gain-zero poles what every branch departs from).
    crossing_gain : float or None
        The gain where a pole reaches the boundary, ``None`` when the sweep never does.
    crossing_pole : complex or None
        The pole on the boundary at that gain.
    boundary : str
        ``"imaginary axis"`` or ``"unit circle"``, so a reader knows what was crossed.
    notes : list of str
        What limited the answer, in the degenerate cases: a numerator of zero (feedback has no effect), an improper
        loop (more zeros than poles, the closed loop has more poles than the plant), a sweep that starts on the
        boundary, or one that never crosses.

    Examples
    --------
    A first-order plant ``1 / (s + a)`` has a closed-loop pole at exactly ``-a - K``::

        locus = root_locus([1.0], [1.0, a], np.linspace(0, 10, 51))
        locus["poles"][:, 0] == -a - locus["gains"]

    Discrete time, ``1 / (z - 0.5)`` with unity feedback: ``z = 0.5 - K`` leaves the unit circle at ``K = 1.5``.

    The reference for the continuous case is Astrom & Murray, *Feedback Systems* (2nd ed., 2021), ch. 10; the
    branch-departure geometry (``n_poles - n_zeros`` branches leaving towards infinity) is not drawn here -- the
    returned poles are the finite roots at each gain, which is what a plot and a stability verdict need.
    """
    den = np.atleast_1d(np.asarray(den, float)).ravel()
    num = np.atleast_1d(np.asarray(num, float)).ravel()
    gains = np.atleast_1d(np.asarray(gains, float)).ravel()
    if den.size == 0 or not np.any(den):
        raise ValueError("the denominator must be a non-zero polynomial: it is the open loop without the gain")
    if gains.size == 0:
        raise ValueError("a root locus needs at least one gain to sweep")
    if np.any(gains < 0):
        raise ValueError("a root locus sweeps a non-negative gain: a negative gain is positive feedback, not this")
    gains = np.sort(gains)

    notes: list[str] = []
    num_zero = not np.any(num)
    if num_zero:
        notes.append("the numerator is zero: the gain does not reach the loop, so the poles are the open-loop ones")

    def poles_at(k: float) -> np.ndarray:
        if num_zero:
            return np.roots(den)
        c = np.polyadd(den, k * num) if num.size else den
        # a trailing zero is a root at the origin (an integrator), not something to trim away
        return np.roots(c) if np.any(c) else np.array([], dtype=complex)

    rows: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for k in gains:
        p = np.asarray(poles_at(float(k)), dtype=complex)
        if previous is not None and p.size == previous.size:
            p = _track(p, previous)
        rows.append(p)
        previous = p
    if not rows:
        poles = np.empty((0, 0), complex)
    else:
        # An improper loop has fewer closed-loop poles at K = 0 than at K > 0: the extra ones are at infinity, so
        # the width of the array is the widest gain and the missing entries are NaN -- a branch that does not exist
        # yet, which a plot must not draw as zero.
        width = max(r.size for r in rows)
        poles = np.full((len(rows), width), np.nan + 0j, dtype=complex)
        for j, p in enumerate(rows):
            poles[j, :p.size] = p
        if any(r.size != width for r in rows):
            notes.append(f"the number of closed-loop poles changes along the sweep ({min(r.size for r in rows)} to "
                         f"{width}): the missing branches are NaN, meaning a pole at infinity")

    def boundary_value(p: np.ndarray) -> float:
        """Signed distance to the boundary: > 0 outside (unstable), < 0 inside."""
        return float(np.max(np.abs(p) - 1.0)) if discrete else float(np.max(p.real))

    f = np.array([boundary_value(p) for p in poles])
    crossing_gain: float | None = None
    crossing_pole: complex | None = None
    # A crossing is a gain at or inside the boundary followed by one outside it. A loop that merely *starts* on the
    # boundary (an open-loop pole at the origin, as any plant with an integrator has at K = 0) is not a crossing:
    # the sweep leaves the boundary going inwards, and the gain that matters is the one it comes back out at.
    i = np.flatnonzero((f[:-1] <= 0.0) & (f[1:] > 0.0))
    if i.size:
        a = gains[i[0]]
        b = gains[i[0] + 1]
        for _ in range(max(1, int(refine))):
            mid = 0.5 * (a + b)
            fm = boundary_value(np.asarray(poles_at(float(mid)), dtype=complex))
            if fm <= 0.0:
                a = mid
            else:
                b = mid
        crossing_gain = float(0.5 * (a + b))
        p = np.asarray(poles_at(crossing_gain), dtype=complex)
        crossing_pole = complex(p[np.argmax(np.abs(p) - 1.0) if discrete else np.argmax(p.real)])
    elif f.size and abs(f[0]) <= 1e-12:
        notes.append("the sweep starts on the stability boundary: the crossing gain is the first gain")
    else:
        notes.append("the sweep never crosses the stability boundary at these gains")
    if not num_zero and num.size > den.size:
        notes.append(f"improper loop ({num.size - 1} zeros for {den.size - 1} poles): the closed loop has "
                     f"{num.size - 1} poles and they are all returned")
    return {"gains": gains, "poles": poles, "open_loop_poles": np.roots(den), "crossing_gain": crossing_gain,
            "crossing_pole": crossing_pole, "boundary": "unit circle" if discrete else "imaginary axis",
            "notes": notes}
