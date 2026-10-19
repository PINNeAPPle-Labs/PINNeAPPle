"""Effect curves, local derivatives and the "which parameters can be frozen" decision.

- :func:`effect_curves`: partial dependence (the main effect: output averaged over the other inputs while one input
  sweeps its range; Friedman 2001) and individual conditional expectation curves (ICE; Goldstein et al. 2015). ICE
  curves that are not parallel reveal interactions.
- :func:`local_derivatives`: the Jacobian at a nominal point, by automatic differentiation when a torch function is
  given, otherwise by central finite differences; also scaled by each input's range (output change across the range,
  linearised) and as elasticities.
- :func:`freeze_recommendation` + :func:`verify_freeze`: an input can be frozen at its nominal value when the upper
  confidence bound of its total Sobol index is below a threshold for every output; the recommendation is then checked
  directly by re-running the model with those inputs frozen and comparing output variance and pointwise change.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pinneapple_analysis.uncertainty.distributions import as_space, to_values, unit_sample


def _y2(Y, n) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    if Y.shape[0] != n:
        raise ValueError(f"model returned {Y.shape[0]} rows for {n} inputs")
    return Y


def effect_curves(fn, space, n_grid: int = 21, n_background: int = 128, n_ice: int = 20, seed: int = 0,
                  outputs: Sequence[str] | None = None) -> dict:
    """Partial dependence and ICE curves of every input. Cost ``d * n_grid * n_background`` evaluations.

    The grid of each input runs over its distribution's quantiles 1 %-99 % (its range for a uniform). Returns
    ``{"names", "outputs", "curves": [{"name", "grid", "pdp" (m, n_grid), "ice" (m, n_ice, n_grid),
    "pdp_range" (m,), "ice_spread" (m,)}], "n_evals"}``; ``ice_spread`` is the mean standard deviation of the
    centred ICE curves (0 when the input acts additively).
    """
    sp = as_space(space)
    d = len(sp)
    bg = to_values(sp, unit_sample(d, n_background, "lhs", seed))
    q = np.linspace(0.0, 1.0, n_grid)
    curves = []
    n_evals = 0
    m = None
    for i, p in enumerate(sp):
        u = q if p.bounded else 0.01 + 0.98 * q
        grid = p.ppf(u)
        X = np.repeat(bg[None], n_grid, axis=0)                          # (n_grid, n_bg, d)
        X[:, :, i] = grid[:, None]
        Y = _y2(fn(X.reshape(-1, d)), n_grid * n_background)
        n_evals += Y.shape[0]
        m = Y.shape[1]
        Y = Y.reshape(n_grid, n_background, m)                            # grid, sample, output
        pdp = Y.mean(axis=1).T                                            # (m, n_grid)
        centred = Y - Y.mean(axis=0, keepdims=True)
        ice_spread = centred.std(axis=1).mean(axis=0)                     # (m,)
        ice = np.moveaxis(Y[:, :n_ice], 0, -1).transpose(1, 0, 2)         # (m, n_ice, n_grid)
        curves.append({"name": p.name, "unit": p.unit, "grid": grid, "pdp": pdp, "ice": ice,
                       "pdp_range": pdp.max(axis=1) - pdp.min(axis=1), "ice_spread": ice_spread})
    return {"names": [p.name for p in sp], "outputs": list(outputs or [f"y{j}" for j in range(m or 1)]),
            "curves": curves, "n_evals": n_evals, "n_background": n_background}


def local_derivatives(fn, x0, space=None, torch_fn: Callable | None = None, rel_step: float = 1e-4,
                      outputs: Sequence[str] | None = None) -> dict:
    """Jacobian ``dy/dx`` (m, d) at ``x0``.

    ``torch_fn`` (a function of a ``(n, d)`` float32 tensor returning ``(n, m)`` or a tuple whose first element is
    that) gives exact derivatives by automatic differentiation; otherwise central differences with step
    ``rel_step`` times the input's range (or ``|x0|``). With ``space`` the result also holds ``scaled`` (Jacobian
    times the range width: linearised output change across each range) and ``elasticity`` (``x/y dy/dx``).
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    d = x0.size
    y0 = _y2(fn(x0[None]), 1)[0]
    sp = as_space(space) if space is not None else None
    width = np.array([p.bounds[1] - p.bounds[0] for p in sp]) if sp else np.maximum(np.abs(x0), 1.0)
    if torch_fn is not None:
        import torch

        def f(x):
            out = torch_fn(x[None])
            out = out[0] if isinstance(out, (tuple, list)) else out
            return out.reshape(-1)

        J = torch.autograd.functional.jacobian(f, torch.as_tensor(x0, dtype=torch.float32)).double().numpy()
        J = J.reshape(-1, d)[: y0.size]
        method = "automatic differentiation (torch.autograd)"
    else:
        h = rel_step * np.where(width > 0, width, 1.0)
        X = np.repeat(x0[None], 2 * d, axis=0)
        for i in range(d):
            X[2 * i, i] += h[i]
            X[2 * i + 1, i] -= h[i]
        Y = _y2(fn(X), 2 * d)
        J = ((Y[0::2] - Y[1::2]) / (2 * h[:, None])).T
        method = f"central finite differences (step {rel_step:g} x range)"
    out = {"method": method, "x0": x0, "y0": y0, "jacobian": J,
           "names": [p.name for p in sp] if sp else [f"x{i}" for i in range(d)],
           "outputs": list(outputs or [f"y{j}" for j in range(y0.size)])}
    out["scaled"] = J * width[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["elasticity"] = np.where(np.abs(y0)[:, None] > 0, J * x0[None, :] / y0[:, None], np.nan)
    return out


def freeze_recommendation(indices, threshold: float = 0.01, morris_result=None) -> list[dict]:
    """Per input: ``freeze`` when the upper bound of the total-index interval is below ``threshold`` for every output;
    ``inconclusive`` when the estimate is below but the interval is not (more samples needed); ``keep`` otherwise.
    Evidence lists the total index and interval per output (and Morris ``mu_star`` when given)."""
    rows = []
    for i, name in enumerate(indices.names):
        st, hi = indices.ST[:, i], indices.ST_ci[1, :, i]
        if np.all(hi < threshold):
            decision = "freeze"
        elif np.all(np.nan_to_num(st, nan=1.0) < threshold):
            decision = "inconclusive"
        else:
            decision = "keep"
        ev = [{"output": o, "ST": float(st[j]), "ST_ci": [float(indices.ST_ci[0, j, i]), float(hi[j])],
               "S1": float(indices.S1[j, i])} for j, o in enumerate(indices.outputs)]
        row = {"name": name, "decision": decision, "max_ST": float(np.nanmax(st)), "max_ST_upper": float(np.nanmax(hi)),
               "evidence": ev}
        if morris_result is not None and name in morris_result.names:
            k = morris_result.names.index(name)
            row["morris_mu_star"] = morris_result.mu_star[:, k].tolist()
        rows.append(row)
    return rows


def verify_freeze(fn, space, frozen: Sequence[str], n: int = 1024, seed: int = 0, at: dict | None = None) -> dict:
    """Run the model on ``n`` samples with all inputs varying, and on the same samples with ``frozen`` inputs held at
    their nominal value (the median, or ``at[name]``). Returns per output the variance ratio (frozen / full) and
    the largest and 95th-percentile pointwise change relative to the output's standard deviation."""
    sp = as_space(space)
    names = [p.name for p in sp]
    unknown = [f for f in frozen if f not in names]
    if unknown:
        raise ValueError(f"unknown inputs to freeze: {unknown}")
    X = to_values(sp, unit_sample(len(sp), n, "lhs", seed))
    Xf = X.copy()
    for f in frozen:
        i = names.index(f)
        Xf[:, i] = (at or {}).get(f, sp[i].center)
    Y, Yf = _y2(fn(X), n), _y2(fn(Xf), n)
    sd = Y.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    diff = np.abs(Yf - Y) / sd
    return {"frozen": list(frozen), "n": int(n), "n_evals": 2 * int(n),
            "variance_ratio": (Yf.var(axis=0) / np.where(Y.var(axis=0) > 0, Y.var(axis=0), 1.0)).tolist(),
            "max_change_sd": diff.max(axis=0).tolist(), "p95_change_sd": np.quantile(diff, 0.95, axis=0).tolist(),
            "nominal": {f: float(Xf[0, names.index(f)]) for f in frozen}}


__all__ = ["effect_curves", "local_derivatives", "freeze_recommendation", "verify_freeze"]
