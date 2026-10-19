"""One sensitivity study: which parameters really control each response, and which can be frozen.

:func:`sensitivity_study` runs, on one model and one parameter space: Morris screening, Sobol first/total (and
optionally second-order) indices with bootstrap intervals, main-effect and ICE curves, local derivatives at the
nominal point (autodiff when the model has a torch function), the freeze recommendation and its direct check.
:func:`study_cost` gives the number of model evaluations before anything runs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from pinneapple_analysis.sensitivity.effects import (
    effect_curves,
    freeze_recommendation,
    local_derivatives,
    verify_freeze,
)
from pinneapple_analysis.sensitivity.morris import morris
from pinneapple_analysis.sensitivity.sobol import sobol_indices
from pinneapple_analysis.uncertainty.distributions import as_space


def study_cost(d: int, n_sobol: int = 1024, second_order: bool = False, r_morris: int = 20, n_grid: int = 15,
               n_background: int = 64, n_verify: int = 512, morris_on: bool = True, effects_on: bool = True) -> dict:
    """Model evaluations of each part of :func:`sensitivity_study` (``n_sobol`` rounded up to a power of two)."""
    N = 2 ** int(np.ceil(np.log2(max(int(n_sobol), 2))))
    parts = {"sobol": N * (2 * d + 2 if second_order else d + 2),
             "morris": r_morris * (d + 1) if morris_on else 0,
             "effects": d * n_grid * n_background if effects_on else 0,
             "derivatives": 2 * d + 1,
             "verify_freeze": 2 * n_verify}
    return {"parts": parts, "total": int(sum(parts.values())), "n_base": N}


def sensitivity_study(model, space, outputs: Sequence[str] | None = None, *, n_sobol: int = 1024,
                      second_order: bool = False, r_morris: int = 20, n_grid: int = 15, n_background: int = 64,
                      threshold: float = 0.01, n_boot: int = 200, level: float = 0.95, n_verify: int = 512,
                      nominal: Mapping[str, float] | None = None, fixed: Mapping[str, float] | None = None,
                      morris_on: bool = True, effects_on: bool = True, exact: Mapping | None = None,
                      seed: int = 0) -> dict:
    """Run the study.

    ``model`` is a :class:`pinneapple_analysis.study_models.StudyModel`; ``space`` lists the varied parameters
    (distributions); model inputs not in ``space`` are held at ``fixed[name]``. ``outputs`` selects responses
    (default all). ``exact`` (``{"S1": [...], "ST": [...]}`` for the first output) adds the error of the estimate
    against known indices (benchmark functions).
    """
    sp = as_space(space)
    names = [p.name for p in sp]
    unknown = [n for n in names if n not in model.input_names]
    if unknown:
        raise ValueError(f"parameters not in the model: {unknown}")
    fixed = dict(fixed or {})
    others = [n for n in model.input_names if n not in names]
    missing = [n for n in others if n not in fixed]
    if missing:
        raise ValueError(f"give a fixed value for the inputs that are not varied: {', '.join(missing)}")
    outs = list(outputs or model.output_names)
    cols = [model.output_index(o) for o in outs]
    idx = [model.input_names.index(n) for n in names]
    base = np.array([float(fixed.get(n, 0.0)) for n in model.input_names])

    def full(Xp):
        X = np.repeat(base[None], Xp.shape[0], axis=0)
        X[:, idx] = Xp
        return X

    def fn(Xp):
        return model.predict(full(np.atleast_2d(Xp)))[:, cols]

    n_evals = {}
    out = {"model": model.info(), "parameters": [p.to_dict() for p in sp], "outputs": outs,
           "units": {o: model.units.get(o, "") for o in outs}, "fixed": {n: fixed[n] for n in others},
           "threshold": threshold}
    s = sobol_indices(fn, sp, n=n_sobol, second_order=second_order, n_boot=n_boot, level=level, seed=seed,
                      outputs=outs)
    n_evals["sobol"] = s.n_evals
    out["sobol"] = s.to_dict()
    out["sobol"]["tables"] = {o: s.table(k) for k, o in enumerate(outs)}
    out["sobol"]["ranking"] = {o: s.ranking(k) for k, o in enumerate(outs)}
    if exact is not None:
        e1 = np.asarray(exact["S1"], dtype=float)
        eT = np.asarray(exact["ST"], dtype=float)
        out["exact"] = {"S1": e1.tolist(), "ST": eT.tolist(),
                        "max_abs_error_S1": float(np.max(np.abs(s.S1[0] - e1))),
                        "max_abs_error_ST": float(np.max(np.abs(s.ST[0] - eT))),
                        "S1_in_ci": [bool(s.S1_ci[0, 0, i] <= e1[i] <= s.S1_ci[1, 0, i]) for i in range(len(e1))],
                        "ST_in_ci": [bool(s.ST_ci[0, 0, i] <= eT[i] <= s.ST_ci[1, 0, i]) for i in range(len(eT))]}
    m = None
    if morris_on:
        m = morris(fn, sp, r=r_morris, seed=seed, level=level, outputs=outs)
        n_evals["morris"] = m.n_evals
        out["morris"] = m.to_dict()
    if effects_on:
        ec = effect_curves(fn, sp, n_grid=n_grid, n_background=n_background, seed=seed, outputs=outs)
        n_evals["effects"] = ec["n_evals"]
        out["effects"] = {"n_background": n_background, "n_grid": n_grid,
                          "curves": [{"name": c["name"], "unit": c["unit"], "grid": c["grid"].tolist(),
                                      "pdp": c["pdp"].tolist(), "ice": c["ice"].tolist(),
                                      "pdp_range": c["pdp_range"].tolist(), "ice_spread": c["ice_spread"].tolist()}
                                     for c in ec["curves"]]}
    x0 = np.array([float((nominal or {}).get(p.name, p.center)) for p in sp])
    torch_fn = None
    if model.torch_fn is not None:
        import torch
        base_t = torch.as_tensor(base, dtype=torch.float32)
        idx_t = torch.as_tensor(idx)

        def torch_fn(xp):
            X = base_t.repeat(xp.shape[0], 1)
            X = X.index_copy(1, idx_t, xp)
            return model.torch_fn(X)[:, cols]

    ld = local_derivatives(fn, x0, sp, torch_fn=torch_fn, outputs=outs)
    n_evals["derivatives"] = 1 if torch_fn is not None else 2 * len(sp) + 1
    out["derivatives"] = {"method": ld["method"], "x0": dict(zip(names, x0.tolist(), strict=True)),
                          "y0": dict(zip(outs, ld["y0"].tolist(), strict=True)),
                          "jacobian": ld["jacobian"].tolist(), "scaled": ld["scaled"].tolist(),
                          "elasticity": np.nan_to_num(ld["elasticity"], nan=0.0).tolist()}
    rec = freeze_recommendation(s, threshold, m)
    out["freeze"] = {"threshold": threshold, "rows": rec}
    frozen = [r["name"] for r in rec if r["decision"] == "freeze"]
    if frozen:
        at = {n: float(x0[names.index(n)]) for n in frozen}
        check = verify_freeze(fn, sp, frozen, n=n_verify, seed=seed + 7, at=at)
        n_evals["verify_freeze"] = check["n_evals"]
        check["outputs"] = outs
        out["freeze"]["check"] = check
    out["n_evals"] = {**n_evals, "total": int(sum(n_evals.values()))}
    return out


__all__ = ["sensitivity_study", "study_cost"]
