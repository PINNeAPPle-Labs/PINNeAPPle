"""The standard equation-discovery workflow (#1192): sparse regression over a library of candidate terms, the model
chosen on data never fitted, its terms with bootstrap intervals and inclusion probabilities, and the equations as
text the library parses (and as LaTeX).

* **ODEs** (:func:`discover_ode`, SINDy: Brunton, Proctor & Kutz, *PNAS* 113 (2016) 3932): states sampled in time;
  derivatives by a Savitzky-Golay smoothing derivative (noise is estimated and stated); a polynomial library of the
  states up to ``degree`` plus the caller's terms.
* **PDEs** (:func:`discover_pde`, PDE-FIND: Rudy, Brunton, Proctor & Kutz, *Sci. Adv.* 3 (2017) e1602614): a field
  on a grid; ``u_t`` regressed on products of ``u`` powers and its spatial derivatives.

Selection: sequentially thresholded least squares (STLSQ) over a path of thresholds; each model is scored by AICc
on the held-out part (the last ``holdout`` of the time window), so a term survives only when it predicts data it
never saw. The selected model is refitted ``n_bootstrap`` times on resampled rows (ensemble SINDy, Fasel et al.
*Proc. R. Soc. A* 478 (2022)): a term's inclusion probability and its coefficient interval. For ODEs the chosen
model is integrated from the held-out window's first state and compared with the data (a model that fits
derivatives but diverges is rejected).
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _savgol_derivative(y: np.ndarray, dt: float, window: int, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import savgol_filter
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(y, w, order, axis=0), savgol_filter(y, w, order, deriv=1, delta=dt, axis=0)


def _poly_library(Y: np.ndarray, names: Sequence[str], degree: int) -> tuple[np.ndarray, list[str]]:
    cols, labels = [np.ones(len(Y))], ["1"]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(Y.shape[1]), d):
            cols.append(np.prod(Y[:, combo], axis=1))
            labels.append("*".join(names[i] for i in combo))
    return np.column_stack(cols), labels


def _stlsq(Theta: np.ndarray, dy: np.ndarray, thr: float, ridge: float = 1e-10, iters: int = 20) -> np.ndarray:
    scale = np.linalg.norm(Theta, axis=0) + 1e-300
    A = Theta / scale
    xi = np.linalg.lstsq(A, dy, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi * 1.0) < thr * np.max(np.abs(xi)) if np.any(xi) else np.ones_like(xi, bool)
        xi[small] = 0
        big = ~small
        if not big.any():
            break
        Ab = A[:, big]
        xi[big] = np.linalg.solve(Ab.T @ Ab + ridge * np.eye(big.sum()), Ab.T @ dy)
    return xi / scale


def _prune(Theta: np.ndarray, dy: np.ndarray, xi: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """Drop terms whose contribution (RMS of coefficient x column) is below ``frac`` of the target's RMS, refit."""
    target = float(np.sqrt(np.mean(dy ** 2))) or 1.0
    keep = np.array([abs(c) * float(np.sqrt(np.mean(Theta[:, i] ** 2))) >= frac * target if c != 0 else False
                     for i, c in enumerate(xi)])
    out = np.zeros_like(xi)
    if keep.any():
        out[keep] = np.linalg.lstsq(Theta[:, keep], dy, rcond=None)[0]
    return out


def _aicc(rss: float, n: int, k: int) -> float:
    if k == 0:
        k = 1
    aic = n * math.log(max(rss / n, 1e-300)) + 2 * k
    return aic + (2 * k * (k + 1)) / max(n - k - 1, 1)


def _select(Theta_fit, dy_fit, Theta_val, dy_val, thresholds):
    best = None
    path = []
    for thr in thresholds:
        xi = _stlsq(Theta_fit, dy_fit, thr)
        k = int(np.count_nonzero(xi))
        rss = float(np.sum((Theta_val @ xi - dy_val) ** 2))
        score = _aicc(rss, len(dy_val), k)
        path.append({"threshold": float(thr), "terms": k, "held_out_rss": rss, "aicc": score})
        if best is None or score < best[0] - 1e-9 or (abs(score - best[0]) < 2 and k < best[2]):
            best = (score, xi, k, thr)
    return _prune(Theta_fit, dy_fit, best[1]), best[3], path


def _bootstrap(Theta, dy, thr, n, seed):
    rng = np.random.default_rng(seed)
    XI = []
    for _ in range(n):
        idx = rng.integers(0, len(dy), len(dy))
        XI.append(_stlsq(Theta[idx], dy[idx], thr))
    XI = np.array(XI)
    return (XI != 0).mean(0), np.percentile(XI, 2.5, axis=0), np.percentile(XI, 97.5, axis=0)


def _equation(lhs: str, xi: np.ndarray, labels: Sequence[str], digits: int = 4) -> str:
    terms = [f"{c:.{digits}g}*{lab}" if lab != "1" else f"{c:.{digits}g}" for c, lab in zip(xi, labels, strict=True)
             if c != 0]
    return f"{lhs} = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0")


def _latex(eq: str) -> str:
    import sympy as sp
    lhs, rhs = eq.split("=", 1)
    try:
        return sp.latex(sp.Eq(sp.Symbol(lhs.strip()), sp.sympify(rhs)))
    except (sp.SympifyError, TypeError):
        return eq


def _blocks(n: int, holdout: float, n_blocks: int = 10) -> np.ndarray:
    """Held-out rows: whole blocks spread over the series (block 2, 7, ... of ``n_blocks``), so the selection sees
    every regime of the signal (a decaying signal's last part is mostly noise)."""
    k = max(1, round(holdout * n_blocks))
    pick = np.linspace(1, n_blocks - 2, k).round().astype(int)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    mask = np.zeros(n, bool)
    for b in pick:
        mask[edges[b]:edges[b + 1]] = True
    return mask


def _fit_states(t, Y, names, degree, w, cut, thresholds, n_bootstrap, seed, holdout=0.2):
    dt = float(t[1] - t[0])
    Ys, dY = _savgol_derivative(Y, dt, w)
    Theta, labels = _poly_library(Ys, names, degree)
    XI = np.zeros((len(labels), len(names)))
    per = {}
    val = _blocks(cut, holdout)                       # selection blocks inside the fitting window
    fit = ~val
    for j, nm in enumerate(names):
        xi, thr, path = _select(Theta[:cut][fit], dY[:cut, j][fit], Theta[:cut][val], dY[:cut, j][val],
                                thresholds)
        XI[:, j] = xi
        per[nm] = {"threshold": thr, "path": path, "xi": xi}
    if n_bootstrap:
        for j, nm in enumerate(names):
            p_in, lo, hi = _bootstrap(Theta[:cut], dY[:cut, j], per[nm]["threshold"], n_bootstrap, seed + j)
            xi = per[nm].pop("xi")
            per[nm]["terms"] = {labels[i]: {"coefficient": float(xi[i]), "inclusion": float(p_in[i]),
                                            "interval95": [float(lo[i]), float(hi[i])]} for i in np.flatnonzero(xi)}
    return Ys, XI, labels, per


def _rollout(t, Y, Ys, XI, names, degree, start, stop):
    from scipy.integrate import solve_ivp

    def rhs(_t, y):
        th, _ = _poly_library(y[None, :], names, degree)
        return (th @ XI)[0]
    sol = solve_ivp(rhs, (t[start], t[stop - 1]), Ys[start], t_eval=t[start:stop], method="LSODA", rtol=1e-8,
                    atol=1e-10)
    if not sol.success or sol.y.shape[1] != stop - start:
        return float("inf")
    scale = float(np.sqrt(np.mean((Y - Y.mean(0)) ** 2)))            # the whole signal's size, not the window's
    return float(np.sqrt(np.mean((sol.y.T - Y[start:stop]) ** 2)) / max(scale, 1e-300))


def discover_ode(t: np.ndarray, Y: np.ndarray, names: Sequence[str], *, degree: int = 2, holdout: float = 0.2,
                 thresholds: Sequence[float] | None = None, window: int | None = None, n_bootstrap: int = 100,
                 rollout_fraction: float = 0.25, rollout_tol: float = 0.1, seed: int = 0) -> dict[str, Any]:
    """Discover ``dY/dt = f(Y)`` from states ``Y`` (n, d) sampled at uniform times ``t``.

    ``window``: the Savitzky-Golay window; by default each of 7, 11, 15, 21, 31, 41 points is tried and the one
    whose model best reproduces the held-out window wins. The rollout runs over the first ``rollout_fraction`` of
    the held-out window (chaotic systems diverge beyond their predictability time whatever the coefficients) and
    its error is relative to the whole signal's RMS."""
    t = np.asarray(t, float)
    Y = np.asarray(Y, float).reshape(len(t), -1)
    names = list(names)
    dt = float(t[1] - t[0])
    if np.max(np.abs(np.diff(t) - dt)) > 1e-9 * max(abs(dt), 1.0):
        raise ValueError("uniform time sampling is required")
    cut = int((1 - holdout) * len(t))
    stop = cut + max(10, int(rollout_fraction * (len(t) - cut)))
    thresholds = thresholds if thresholds is not None else np.geomspace(1e-4, 0.5, 30)
    windows = [window] if window else [w for w in (7, 11, 15, 21, 31, 41) if w < len(t) // 5]
    trials = []
    for w in windows:
        Ys, XI, labels, _ = _fit_states(t, Y, names, degree, w, cut, thresholds, 0, seed)
        trials.append((_rollout(t, Y, Ys, XI, names, degree, cut, stop), w))
    err, w = min(trials)
    Ys, XI, labels, per = _fit_states(t, Y, names, degree, w, cut, thresholds, n_bootstrap, seed)
    err = _rollout(t, Y, Ys, XI, names, degree, cut, stop)
    noise = float(np.std(Y - Ys) / max(np.std(Y), 1e-300))
    eqs = [_equation(f"{nm}_t", XI[:, j], labels) for j, nm in enumerate(names)]
    checks = [{"code": "discover.rollout", "status": "pass" if err <= rollout_tol else "fail",
               "detail": f"the discovered system integrated from the held-out window's first state over "
                         f"{t[stop - 1] - t[cut]:.3g} time units: RMS error {err:.2%} of the signal (allowed "
                         f"{rollout_tol:.0%})", "value": err},
              {"code": "discover.stable_terms", "status": "pass" if all(
                  v["inclusion"] >= 0.8 for p in per.values() for v in p.get("terms", {}).values()) else "fail",
               "detail": "every selected term is kept in at least 80 % of the bootstrap refits", "value": None}]
    return {"equations": eqs, "latex": [_latex(e) for e in eqs], "per_state": per, "library": labels,
            "noise_rel": noise, "window": w, "windows_tried": [{"window": ww, "rollout": e} for e, ww in trials],
            "checks": checks, "rollout_rel_error": err}


def _factors(label: str) -> list[str]:
    """``"u**2*u_x"`` -> ``["u**2", "u_x"]``: split on the product ``*`` only, never inside a power ``**``."""
    import re
    return [] if label == "1" else [f for f in re.split(r"(?<!\*)\*(?!\*)", label) if f]


def _pde_rollout(xi, labels, u, x, t, start, max_deriv) -> float:
    """Solve ``u_t = sum xi_k term_k`` by the method of lines on the data's grid from ``u[:, start]``, the two
    boundary values taken from the data, and return the RMS error over the remaining times relative to the field."""
    from scipy.integrate import solve_ivp
    dx = float(x[1] - x[0])
    active = [(c, lab) for c, lab in zip(xi, labels, strict=True) if c != 0]
    if not active:
        return float("inf"), float("nan")
    tt = t[start:]

    def terms(v):
        d = [v]
        for _ in range(max_deriv):
            d.append(np.gradient(d[-1], dx, edge_order=2))
        env = {"u": v, **{"u_" + "x" * k: d[k] for k in range(1, max_deriv + 1)}}
        out = np.zeros_like(v)
        for c, lab in active:
            val = np.ones_like(v)
            for f in _factors(lab):
                base, _, pw = f.partition("**")
                val = val * env[base] ** (int(pw) if pw else 1)
            out = out + c * val
        return out

    def rhs(_t, v):
        dv = terms(v)
        dv[0] = dv[-1] = 0.0                           # boundary values held: set below from the data
        return dv
    left = np.interp(tt, t, u[0])
    right = np.interp(tt, t, u[-1])
    v0 = u[:, start].copy()
    try:
        sol = solve_ivp(rhs, (tt[0], tt[-1]), v0, t_eval=tt, method="BDF", rtol=1e-6, atol=1e-9)
    except (ValueError, FloatingPointError):
        return float("inf"), float("nan")
    if not sol.success or sol.y.shape[1] != len(tt):
        return float("inf"), float("nan")
    sim = sol.y.copy()
    sim[0], sim[-1] = left, right
    scale = float(np.sqrt(np.mean((u - u.mean()) ** 2)))
    err = float(np.sqrt(np.mean((sim - u[:, start:]) ** 2)) / max(scale, 1e-300))
    persist = float(np.sqrt(np.mean((u[:, start:] - u[:, start:start + 1]) ** 2)) / max(scale, 1e-300))
    return err, persist


def _bump(m: int, h: float, p: int, k: int) -> np.ndarray:
    """The ``k``-th derivative of the test function ``(1 - xi^2)^p`` on ``2m + 1`` grid points of half-width ``h``
    (``xi`` in [-1, 1]), in physical units. It and its first ``p - 1`` derivatives vanish at both ends, so
    integration by parts against it leaves no boundary terms."""
    from numpy.polynomial import Polynomial
    xi = np.arange(-m, m + 1) / m
    poly = Polynomial([1.0, 0.0, -1.0]) ** p
    return (poly.deriv(k)(xi) / h ** k) if k else poly(xi)


def _label_parts(label: str) -> tuple[int, int]:
    """``"u**2*u_xx"`` -> ``(2, 2)``: the power of ``u`` and the order of the x-derivative factor."""
    pw, k = 0, 0
    for f in _factors(label):
        base, _, e = f.partition("**")
        if base == "u":
            pw += int(e) if e else 1
        else:
            k = len(base) - 2                                  # "u_xx" -> 2
    return pw, k


def _weak_system(u: np.ndarray, us: np.ndarray, x: np.ndarray, t: np.ndarray, labels: Sequence[str],
                 max_deriv: int, support: tuple[float, float], stride: tuple[int, int] = (4, 2)):
    """The weak form of ``u_t = sum_k xi_k term_k`` (weak SINDy: Messenger & Bortz, *J. Comput. Phys.* 443 (2021)
    110525): each row is the equation integrated against one test function ``phi(x, t) = (1 - xi^2)^p (1 - tau^2)^p``
    on a window of ``support`` (fractions of the x and t extents), the windows sliding over the grid.

    The derivatives move onto ``phi`` by integration by parts, so the regression never uses a derivative of the noisy
    data for ``u_t`` (``-int phi_t u``), for ``u``-powers, for pure derivatives ``u_x...`` (``(-1)^k int phi^(k) u``)
    or for ``u^p u_x`` (``-int phi_x u^(p+1)/(p+1)``). The other terms ``u^p u_(kx)`` (p >= 1, k >= 2) are
    integrated by parts once, so they need derivatives of the (smoothed) field ``us`` up to order ``k - 1`` instead
    of ``k``. Rows ordered by time, then x; returns ``(Theta, b, (n_x, n_t))``."""
    from scipy.signal import correlate
    dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
    mx = max(max_deriv + 2, int(round(support[0] * (len(x) - 1) / 2)))
    mt = max(3, int(round(support[1] * (len(t) - 1) / 2)))
    if 2 * mx + 1 >= len(x) or 2 * mt + 1 >= len(t):
        raise ValueError(f"the weak form needs more than {2 * mx + 1} x points and {2 * mt + 1} times for test "
                         f"functions of support {support}; the field is {u.shape}")
    p = max_deriv + 2
    bx = [_bump(mx, mx * dx, p, k) for k in range(max_deriv + 1)]
    bt = [_bump(mt, mt * dt, p, k) for k in range(2)]

    def W(f, a, b):
        return correlate(f, np.outer(bx[a], bt[b]) * dx * dt, mode="valid")[::stride[0], ::stride[1]]
    d = [us]
    for _ in range(max_deriv):
        d.append(np.gradient(d[-1], dx, axis=0, edge_order=2))
    cols = []
    for lab in labels:
        pw, k = _label_parts(lab)
        if k == 0:
            col = W(u ** pw, 0, 0)
        elif pw == 0:
            col = (-1) ** k * W(u, k, 0)
        elif k == 1:
            col = -W(u ** (pw + 1) / (pw + 1), 1, 0)
        else:                                              # u^p u_(kx) = (u^p u_((k-1)x))_x - p u^(p-1) u_x u_((k-1)x)
            col = -W(u ** pw * d[k - 1], 1, 0) - pw * W(u ** (pw - 1) * d[1] * d[k - 1], 0, 0)
        cols.append(col)
    b = -W(u, 0, 1)
    return np.stack([c.T.ravel() for c in cols], 1), b.T.ravel(), b.shape


def discover_pde(u: np.ndarray, x: np.ndarray, t: np.ndarray, *, max_power: int = 2, max_deriv: int = 3,
                 holdout: float = 0.2, thresholds: Sequence[float] | None = None, n_bootstrap: int = 50,
                 trim: int = 5, smooth: int | None = None, form: str = "strong",
                 weak_support: tuple[float, float] = (0.24, 0.3), seed: int = 0) -> dict[str, Any]:
    """Discover ``u_t = f(u, u_x, u_xx, ...)`` from a field ``u`` (nx, nt) on a uniform grid (PDE-FIND).

    ``smooth``: a Savitzky-Golay window (cubic) applied along x and t before differentiating; by default it is used
    when the field's high-frequency content says it is noisy (finite differences of noise grow as 1/dx^k).

    ``form``: ``"strong"`` regresses ``u_t`` on the library evaluated pointwise (PDE-FIND); ``"weak"`` regresses
    the equation integrated against test functions (:func:`_weak_system`, of ``weak_support`` fractions of the x
    and t extents), which removes the bias that noisy derivatives put in the strong form. A strong-form result is
    also checked against the weak form (``discover.weak_form``): a term that only the strong form selects, or that
    it drops, is named, because noise in the derivatives (errors in the variables of the regression) can make one
    term stand in for part of another, e.g. ``-u`` for part of ``u_xx`` on a smooth field."""
    if form not in ("strong", "weak"):
        raise ValueError("form is 'strong' (PDE-FIND) or 'weak' (weak SINDy)")
    from scipy.signal import savgol_filter
    u = np.asarray(u, float)
    u_raw = u
    # noise estimate: what a short cubic smoothing removes, relative to the field
    rough = float(np.std(u - savgol_filter(u, 7, 3, axis=0)) / max(np.std(u), 1e-300))
    if smooth is None and rough > 1e-4:
        smooth = 25
    if smooth:
        u = savgol_filter(savgol_filter(u, smooth, 3, axis=0), smooth, 3, axis=1)
        trim = max(trim, smooth)
    dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
    ut = np.gradient(u, dt, axis=1, edge_order=2)
    ders = [u]
    for _ in range(max_deriv):
        ders.append(np.gradient(ders[-1], dx, axis=0, edge_order=2))
    dnames = ["u"] + ["u_" + "x" * k for k in range(1, max_deriv + 1)]
    sl = (slice(trim, -trim), slice(trim, -trim))
    cols, labels = [], []
    for pw in range(0, max_power + 1):
        base = u[sl] ** pw
        bname = "" if pw == 0 else ("u" if pw == 1 else f"u**{pw}")
        for k in range(0, max_deriv + 1):
            if k == 0 and pw == 0:
                cols.append(np.ones_like(base))
                labels.append("1")
                continue
            if k == 0:
                cols.append(base)
                labels.append(bname)
            else:
                cols.append(base * ders[k][sl])
                labels.append(f"{bname}*{dnames[k]}" if bname else dnames[k])
    # drop the duplicate u*u (u**2 with k = 0 already) by label
    seen, C, L = set(), [], []
    for c, lab in zip(cols, labels, strict=True):
        if lab not in seen:
            seen.add(lab)
            C.append(c)
            L.append(lab)
    thresholds = thresholds if thresholds is not None else np.geomspace(1e-4, 0.5, 30)
    Theta_w, y_w, shape_w = _weak_system(u_raw, u, x, t, L, max_deriv, tuple(weak_support))
    cut_w = int((1 - holdout) * shape_w[1]) * shape_w[0]
    xi_w, thr_w, path_w = _select(Theta_w[:cut_w], y_w[:cut_w], Theta_w[cut_w:], y_w[cut_w:], thresholds)
    if form == "weak":
        Theta, y, cut = Theta_w, y_w, cut_w
        xi, thr, path = xi_w, thr_w, path_w
    else:
        Theta = np.stack([c.T.ravel() for c in C], 1)            # rows ordered by time, then x
        y = ut[sl].T.ravel()
        nt_in = u[sl].shape[1]
        cut = int((1 - holdout) * nt_in) * u[sl].shape[0]
        xi, thr, path = _select(Theta[:cut], y[:cut], Theta[cut:], y[cut:], thresholds)
    p_in, lo, hi = _bootstrap(Theta[:cut], y[:cut], thr, n_bootstrap, seed)
    eq = _equation("u_t", xi, L)
    rel = float(np.linalg.norm(Theta[cut:] @ xi - y[cut:]) / max(np.linalg.norm(y[cut:]), 1e-300))
    roll, persist = _pde_rollout(xi, L, u_raw, x, t, int((1 - holdout) * len(t)), max_deriv)
    what = "u_t" if form == "strong" else "the weak-form equation (test-function integrals)"
    checks = [{"code": "discover.held_out_fit", "status": "pass" if rel <= 0.1 else "fail",
               "detail": f"{what} predicted on the held-out times to {rel:.2%} relative error", "value": rel},
              {"code": "discover.rollout", "status": "pass" if roll <= min(0.05, 0.25 * persist) else "fail",
               "detail": f"the discovered PDE solved (method of lines, the data's boundary values) over the held-out "
                         f"times from the field at their start: RMS error {roll:.2%} of the field, against "
                         f"{persist:.2%} for a field that does not change (allowed: 5 % and a quarter of that)",
               "value": roll},
              {"code": "discover.stable_terms", "status": "pass" if all(p_in[i] >= 0.8 for i in np.flatnonzero(xi))
               else "fail", "detail": "every selected term is kept in at least 80 % of the bootstrap refits",
               "value": None}]
    weak = {"equation": _equation("u_t", xi_w, L), "support": list(weak_support), "threshold": thr_w,
            "terms": {L[i]: float(xi_w[i]) for i in np.flatnonzero(xi_w)}}
    if form == "strong":
        extra = [L[i] for i in np.flatnonzero(xi) if xi_w[i] == 0]
        missing = [L[i] for i in np.flatnonzero(xi_w) if xi[i] == 0]
        why = "; ".join(([f"only the strong form selects {extra}"] if extra else []) +
                        ([f"the strong form drops {missing}"] if missing else []))
        checks.append({"code": "discover.weak_form", "status": "fail" if why else "pass",
                       "detail": (f"the weak form (test functions; no derivative of the noisy data in the target) "
                                  f"selects the same terms: {weak['equation']}") if not why else
                                 (f"{why}: the weak form (test functions; no derivative of the noisy data in the "
                                  f"target) gives {weak['equation']}. Terms selected by one form only are not "
                                  f"supported by the data; noisy derivatives let one term stand in for another. "
                                  f"Use form='weak' for noisy data"),
                       "value": {"extra": extra, "missing": missing}})
    return {"equation": eq, "latex": _latex(eq), "smoothing": smooth, "roughness": rough, "form": form,
            "weak_form": weak,
            "terms": {L[i]: {"coefficient": float(xi[i]),
                                                                   "inclusion": float(p_in[i]),
                                                                   "interval95": [float(lo[i]), float(hi[i])]}
                                                           for i in np.flatnonzero(xi)},
            "library": L, "threshold": thr, "path": path, "checks": checks}
