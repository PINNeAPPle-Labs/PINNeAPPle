"""VeriPhysics grade of a model trained elsewhere (#1187): a field model ``u(x, t) -> fields`` (a PINN or neural
operator from any framework, through ONNX / TorchScript / a callable, or its predictions on points) graded against
the problem it claims to solve.

The problem is an equation system in the library's text grammar (:mod:`pinneapple_physics.symbolic_pde.text`, the
same one :mod:`pinneapple_problemdesign.formulate` writes) on a box, and/or reference data. Every check treats the
model as a black box: derivatives are central finite differences of its outputs (step ``fd_step`` of each axis'
span, second derivatives with the square-root step, nested for mixed ones), so it works for models whose framework
the library cannot differentiate.

Checks, each pass / fail / not applicable with its reason:

* ``grade.pde_residual``: each equation's residual at ``n_points`` fresh random points: its RMS over the RMS of the
  equation's largest term (a dimensionless relative residual) against ``residual_tol``; the finite-difference
  steps follow the model's output precision (``eps**(1/3)`` and ``eps**(1/4)`` of each span);
* ``grade.conditions``: each boundary / initial condition on its face, residual relative to the field's scale;
* ``grade.reference``: relative L2 error against reference data when given;
* ``grade.applicability``: the grading box inside the model's declared validity box;
* ``grade.robustness``: the output change for an input perturbation of ``eps`` of the span, against the change of
  the reference (or of the model itself at a finer perturbation: a smooth model scales linearly).

The decision is APPROVED (every applicable check passes), REVIEW (one inconclusive) or REJECT (one fails).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _fd(model, X: np.ndarray, j: int, h: float, order: int, k: int | None = None) -> np.ndarray:
    def shift(Z, i, d):
        Z = Z.copy()
        Z[:, i] += d
        return model(Z)
    if order == 1:
        return (shift(X, j, h) - shift(X, j, -h)) / (2 * h)
    if order == 2 and k is None:
        return (shift(X, j, h) - 2 * model(X) + shift(X, j, -h)) / (h * h)
    raise ValueError("unsupported derivative")


def _derivative(model, X, coords, wrt: str, spans, step, step2):
    """Derivative of every output along ``wrt`` (letters, e.g. 'xx', 'xt'), by central differences."""
    from collections import Counter
    c = Counter(wrt)
    if len(c) == 1:
        (ch, n), = c.items()
        j = coords.index(ch)
        h = step * spans[j] if n == 1 else step2 * spans[j]
        if n > 2:
            raise ValueError(f"derivatives of order {n} are not graded")
        return _fd(model, X, j, h, n)
    # mixed: d/da (d/db u)
    a, b = sorted(c)
    if c[a] != 1 or c[b] != 1:
        raise ValueError(f"mixed derivative {wrt} is not graded")
    ja, jb = coords.index(a), coords.index(b)
    ha, hb = step2 * spans[ja], step2 * spans[jb]

    def du_b(Z):
        return _fd(model, Z, jb, hb, 1)
    return _fd(du_b, X, ja, ha, 1)


def _env(model, X, coords, fields, keys, spans, step, step2):
    U = model(X)
    env = {c: X[:, i] for i, c in enumerate(coords)}
    env.update({f: U[:, i] for i, f in enumerate(fields)})
    cache: dict[str, np.ndarray] = {}
    for key, (f, wrt) in keys.items():
        if wrt not in cache:
            cache[wrt] = _derivative(model, X, coords, wrt, spans, step, step2)
        env[key] = cache[wrt][:, fields.index(f)]
    return env


def _terms(expr):
    import sympy as sp
    e = sp.expand(expr)
    return list(sp.Add.make_args(e))


def grade_model(model: Callable[[np.ndarray], np.ndarray], system, bounds: Mapping[str, tuple[float, float]], *,
                reference: tuple[np.ndarray, np.ndarray] | None = None,
                validity: Mapping[str, tuple[float, float]] | None = None, n_points: int = 2000,
                residual_tol: float = 0.05, condition_tol: float = 0.02, reference_tol: float = 0.05,
                fd_step: float | None = None, fd_step2: float | None = None, eps: float = 1e-3, seed: int = 0
                ) -> dict[str, Any]:
    """Grade ``model(X) -> (N, n_fields)`` (``X`` columns in ``system.coords`` order) on the box ``bounds``.

    ``system``: a parsed :class:`~pinneapple_physics.symbolic_pde.text.EquationSystem` with its coefficients
    given; ``reference``: ``(X_ref, U_ref)``; ``validity``: the model's declared validity box."""
    import sympy as sp

    from pinneapple_physics.symbolic_pde.pinn import _plain
    from pinneapple_physics.symbolic_pde.text import compile_expr, derivative_keys
    rng = np.random.default_rng(seed)
    coords, fields = list(system.coords), list(system.fields)
    lo = np.array([float(bounds[c][0]) for c in coords])
    hi = np.array([float(bounds[c][1]) for c in coords])
    spans = hi - lo

    probe = np.asarray(model(lo + np.random.default_rng(seed + 7).random((32, len(coords))) * spans))
    # steps that balance truncation and rounding of the model's own output precision (float32 networks round at
    # about 1e-7: a 1e-3 second-difference step would amplify that to 0.1). The precision is read from the values,
    # not the dtype: a float32 network cast to float64 still holds only float32 values.
    p64 = np.asarray(probe, np.float64)
    is32 = probe.dtype == np.float32 or bool(np.all(p64 == p64.astype(np.float32).astype(np.float64)))
    prec = float(np.finfo(np.float32).eps) if is32 else float(np.finfo(np.float64).eps)
    if fd_step is None:
        fd_step = prec ** (1 / 3)

    def f(Z):
        return np.asarray(model(np.asarray(Z, float)), float).reshape(len(Z), len(fields))
    rep = {sp.Symbol(k, real=True): sp.Float(v) for k, v in (system.coefficients or {}).items()}
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        ok = None if ok is None else bool(ok)
        checks.append({"code": code, "status": "pass" if ok is True else ("fail" if ok is False else "inconclusive"),
                       "detail": detail, "value": value})
    # interior residuals (keep away from the faces by a few FD steps)
    step2 = fd_step2 if fd_step2 is not None else prec ** 0.25
    margin = 3 * max(fd_step, step2) * spans
    X = lo + margin + rng.random((n_points, len(coords))) * (spans - 2 * margin)
    res_rows = []
    for e in system.pdes:
        r = e.residual.xreplace(rep).doit()
        keys = derivative_keys(r)
        env = _env(f, X, coords, fields, keys, spans, fd_step, step2)
        val = np.asarray(compile_expr(_plain(r, system))(env), float) * np.ones(len(X))
        mags = [np.abs(np.asarray(compile_expr(_plain(t, system))(env), float) * np.ones(len(X))) for t in _terms(r)]
        scale = np.max(mags, axis=0)
        # relative to the equation's typical largest term (a pointwise ratio blows up where every term vanishes)
        rel = float(np.sqrt(np.mean(val ** 2)) / max(float(np.sqrt(np.mean(scale ** 2))), 1e-300))
        res_rows.append({"equation": e.text, "rel_rms": rel})
    worst = max((r["rel_rms"] for r in res_rows), default=None)
    chk("grade.pde_residual", None if worst is None else worst <= residual_tol,
        "; ".join(f"{r['equation']}: relative RMS residual {r['rel_rms']:.2e}" for r in res_rows)
        + f" at {n_points} fresh points (allowed {residual_tol:g})", res_rows)
    # conditions
    U_scale = float(np.max(np.abs(f(X)))) or 1.0
    cond_rows = []
    for c in system.conditions:
        if c.kind == "periodic":
            continue
        n = max(200, n_points // 10)
        Z = lo + rng.random((n, len(coords))) * spans
        if c.kind == "initial":
            Z[:, coords.index(system.ctx.time)] = lo[coords.index(system.ctx.time)]
        elif c.on in ("boundary",):
            ax = rng.integers(0, len(coords) - (1 if system.ctx.time in coords else 0), n)
            side = rng.integers(0, 2, n)
            for i in range(n):
                Z[i, ax[i]] = lo[ax[i]] if side[i] == 0 else hi[ax[i]]
        else:
            axis, side = c.on.split("_")
            j = coords.index(axis)
            Z[:, j] = lo[j] if side == "min" else hi[j]
        r = (c.lhs - c.rhs).xreplace(rep).doit()
        keys = derivative_keys(r)
        # one-sided faces: shift inward by the FD stencil so central differences stay inside the box
        Zs = Z.copy()
        for j in range(len(coords)):
            Zs[:, j] = np.clip(Zs[:, j], lo[j] + margin[j], hi[j] - margin[j]) if keys else Zs[:, j]
        env = _env(f, Zs if keys else Z, coords, fields, keys, spans, fd_step, step2)
        val = np.asarray(compile_expr(_plain(r, system))(env), float) * np.ones(len(Z))
        rel = float(np.sqrt(np.mean(val ** 2)) / U_scale) if not keys else float(
            np.sqrt(np.mean(val ** 2)) / max(U_scale / float(np.min(spans)), 1e-300))
        cond_rows.append({"condition": f"{c.kind} {c.on}: {c.text}", "rel_rms": rel})
    worst_c = max((r["rel_rms"] for r in cond_rows), default=None)
    chk("grade.conditions", None if worst_c is None else worst_c <= condition_tol,
        "; ".join(f"{r['condition']}: {r['rel_rms']:.2e}" for r in cond_rows) + f" (allowed {condition_tol:g})"
        if cond_rows else "no condition in the problem", cond_rows)
    # reference data
    if reference is not None:
        Xr, Ur = (np.asarray(a, float) for a in reference)
        Ur = Ur.reshape(len(Xr), -1)
        err = float(np.linalg.norm(f(Xr) - Ur) / max(np.linalg.norm(Ur), 1e-300))
        chk("grade.reference", err <= reference_tol, f"relative L2 error {err:.2e} on {len(Xr)} reference points "
                                                    f"(allowed {reference_tol:g})", err)
    else:
        chk("grade.reference", None, "no reference data given")
    # applicability
    if validity:
        out = [f"{c} [{bounds[c][0]:g}, {bounds[c][1]:g}] beyond the declared [{validity[c][0]:g}, {validity[c][1]:g}]"
               for c in coords if c in validity and (bounds[c][0] < validity[c][0] - 1e-12 or
                                                     bounds[c][1] > validity[c][1] + 1e-12)]
        chk("grade.applicability", not out, "the graded box is inside the model's declared validity"
            if not out else "; ".join(out))
    else:
        chk("grade.applicability", None, "the model declares no validity box")
    # robustness: a smooth model's response to a small perturbation scales with it
    d1 = np.abs(f(X + eps * spans * rng.standard_normal(X.shape)) - f(X)).mean()
    d2 = np.abs(f(X + 0.1 * eps * spans * rng.standard_normal(X.shape)) - f(X)).mean()
    ratio = float(d1 / max(d2, 1e-300))
    chk("grade.robustness", 3.0 <= ratio <= 30.0, f"output change for a {eps:g}-span perturbation is {ratio:.1f}x the "
                                                 "change for a 10x smaller one (about 10 for a smooth model)", ratio)
    applicable = [c for c in checks if c["status"] != "inconclusive"]
    if any(c["status"] == "fail" for c in checks):
        decision = "REJECT"
    elif len(applicable) < len(checks):
        decision = "REVIEW"
    else:
        decision = "APPROVED"
    return {"decision": decision, "checks": checks, "box": {c: list(bounds[c]) for c in coords},
            "n_points": n_points, "method": f"central finite differences, steps {fd_step:g} / {step2:g} of each span"}


def grade_predictions(points: np.ndarray, values: np.ndarray, system, bounds, *, neighbors: int = 60,
                      **kw) -> dict[str, Any]:
    """Grade a model known only by its predictions on scattered ``points`` (a results file): they are interpolated
    by a local thin-plate-spline RBF (``scipy.interpolate.RBFInterpolator`` on the ``neighbors`` nearest points,
    smooth enough for second derivatives) on coordinates scaled to the unit box, then graded like a model. A grade
    of predictions is a grade of the predictions plus the interpolant: sparse or noisy points raise the residual."""
    from scipy.interpolate import RBFInterpolator
    P = np.asarray(points, float)
    V = np.asarray(values, float).reshape(len(P), -1)
    lo = np.array([bounds[c][0] for c in system.coords], float)
    sp_ = np.array([bounds[c][1] - bounds[c][0] for c in system.coords], float)
    rbf = RBFInterpolator((P - lo) / sp_, V, kernel="thin_plate_spline", neighbors=min(neighbors, len(P)))

    def model(X):
        return rbf((np.asarray(X, float) - lo) / sp_)
    spacing = (len(P) ** (-1 / P.shape[1]))              # mean point spacing in the unit box
    kw.setdefault("fd_step", spacing / 2)                # steps at the data's own scale: the interpolant is
    kw.setdefault("fd_step2", spacing)                   # not differentiated below its resolution
    kw.setdefault("n_points", 600)
    out = grade_model(model, system, bounds, **kw)
    out["method"] += f"; predictions on {len(P)} points through a local thin-plate RBF ({neighbors} neighbours)"
    return out


def as_callable(predictor, coords: Sequence[str]) -> Callable[[np.ndarray], np.ndarray]:
    """A :class:`pinneapple_analysis.surrogate.model_source.Predictor` as ``model(X) -> (N, n_outputs)``."""
    def model(X):
        out = predictor.predict({c: X[:, i] for i, c in enumerate(coords)})
        return np.column_stack([np.asarray(out[o], float) for o in predictor.outputs])
    return model
