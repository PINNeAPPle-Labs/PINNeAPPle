"""Robustness studies that feed the applicability map with *measured*
evidence for the three items that used to be ``NOT_RUN``:

* :func:`extrapolation_study` -- generalization: the governing-equation
  residual, and (with an ensemble) the disagreement between independently
  trained models, on shells *outside* the training domain at growing
  margins. The largest margin where both stay acceptable is the measured
  extrapolation limit of the surrogate.
* :func:`ensemble_study` -- UQ: K extra seeds of the same architecture; the
  spread across members is an epistemic-uncertainty estimate. It is
  *uncalibrated* unless reference data is supplied
  (:func:`calibration_against_reference`).
* :func:`weight_sensitivity_study` -- loss weights: retrain with the PDE
  term up- and down-weighted and measure how much the prediction moves.

Everything here runs real training/evaluation and returns plain dicts of
the numbers it measured; nothing is estimated heuristically except the two
documented tolerances below. A residual that is small outside the domain
shows the surrogate still satisfies the PDE there -- it does NOT prove it
matches the true solution (boundary conditions do not constrain it), which
is why the ensemble spread is reported next to it and the map says so.

(Extracted from Veriphysics' orchestrator; Apache-2.0 like the rest of PINNeAPPle. Execution queue, API and billing stay in Veriphysics.)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# Documented heuristics (reviewer aids, not guarantees): ensemble/variant
# disagreement is "acceptable" when the RMS spread is below this fraction
# of the field's own RMS variation over the interior.
SPREAD_REL_TOL = 0.10
DEFAULT_MARGINS = (0.10, 0.25, 0.50)
DEFAULT_WEIGHT_FACTORS = (0.25, 4.0)


def _predict(model, x) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        y = model(torch.as_tensor(x, dtype=torch.float32))
        if hasattr(y, "y"):
            y = y.y
    return y.detach().cpu().numpy()


def _sample_box(bounds: Dict[str, Any], coords: Sequence[str], n: int) -> np.ndarray:
    return np.stack([np.random.uniform(*bounds[c], size=n) for c in coords], axis=1).astype(np.float32)


def _sample_shell(bounds: Dict[str, Any], coords: Sequence[str], margin: float, n: int) -> np.ndarray:
    """*n* uniform points in the box grown by *margin* (fraction of each
    side's width) that lie OUTSIDE the original box."""
    grown = {c: (bounds[c][0] - margin * (bounds[c][1] - bounds[c][0]),
                 bounds[c][1] + margin * (bounds[c][1] - bounds[c][0])) for c in coords}
    kept: List[np.ndarray] = []
    total = 0
    while total < n:
        cand = _sample_box(grown, coords, n * 4)
        outside = np.zeros(len(cand), dtype=bool)
        for i, c in enumerate(coords):
            outside |= (cand[:, i] < bounds[c][0]) | (cand[:, i] > bounds[c][1])
        cand = cand[outside]
        kept.append(cand)
        total += len(cand)
    return np.concatenate(kept)[:n]


def _pde_residual(spec, model, x_np: np.ndarray) -> float:
    """Mean-squared PDE residual at *x_np* -- same computation as
    ``PhysicsGuardrail._check_residual`` but at caller-chosen points."""
    import torch
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    loss_fn = compile_problem(spec)
    n_fields, n_coords = len(spec.fields), len(spec.coords)
    x = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_(True)
    empty_x, empty_y = torch.zeros((0, n_coords)), torch.zeros((0, n_fields))
    batch = {"x_col": x, "ctx": {}, "x_bc": empty_x, "y_bc": empty_y,
             "x_ic": empty_x, "y_ic": empty_y, "x_data": empty_x, "y_data": empty_y}
    y_hat = model(x)
    if hasattr(y_hat, "y"):
        y_hat = y_hat.y
    out = loss_fn(model, y_hat, batch)
    return float(out["pde"].item()) if isinstance(out, dict) and "pde" in out else float(out["total"].item())


def _field_scale(*arrays: np.ndarray) -> float:
    """RMS magnitude of the predicted field (not its variation around the
    ensemble mean, which can cancel across members and inflate ratios)."""
    return float(np.sqrt(np.mean(np.concatenate([a.ravel() for a in arrays]) ** 2))) or 1e-12


def _rel_spread(preds: np.ndarray, scale: float) -> float:
    """RMS across-member std divided by the field's RMS magnitude."""
    return float(np.sqrt(np.mean(np.var(preds, axis=0)))) / scale


def _train(spec, build_model: Callable[[], Any], *, epochs: int, n_collocation: int, seed: int, weights=None):
    import pinneapple_physics as pp

    model = build_model()
    pp.solve_pde(spec, model, epochs=epochs, n_collocation=n_collocation, seed=seed, weights=weights)
    return model


def ensemble_study(
    spec, build_model: Callable[[], Any], best_model, *, n_members: int, epochs: int, n_collocation: int,
    n_eval: int = 2048, margins: Sequence[float] = DEFAULT_MARGINS,
) -> Dict[str, Any]:
    """Train *n_members* extra seeds and measure member disagreement in the
    interior and on each outside shell. Returns the models too (key
    ``"members"``) so callers can reuse them; strip before serializing."""
    coords, bounds = list(spec.coords), spec.domain_bounds
    members = [best_model] + [
        _train(spec, build_model, epochs=epochs, n_collocation=n_collocation, seed=1000 + k)
        for k in range(n_members)
    ]
    x_in = _sample_box(bounds, coords, n_eval)
    preds_in = np.stack([_predict(m, x_in) for m in members])
    scale = _field_scale(preds_in)
    out: Dict[str, Any] = {
        "n_members": len(members),
        "interior_rel_spread": _rel_spread(preds_in, scale),
        "shell_rel_spread": {},
        "members": members,
    }
    for m in margins:
        x_sh = _sample_shell(bounds, coords, m, n_eval)
        out["shell_rel_spread"][m] = _rel_spread(np.stack([_predict(mm, x_sh) for mm in members]), scale)
    return out


def extrapolation_study(
    spec, model, *, residual_threshold: float, ensemble: Optional[Dict[str, Any]] = None,
    margins: Sequence[float] = DEFAULT_MARGINS, n_eval: int = 2048,
) -> Dict[str, Any]:
    """Residual (and ensemble spread, if given) on shells outside the
    training box. ``supported_margin`` is the largest margin such that it
    and every smaller margin keep residual <= threshold and spread <=
    ``SPREAD_REL_TOL``; ``0.0`` means even the smallest shell fails."""
    coords, bounds = list(spec.coords), spec.domain_bounds
    shells, supported = [], 0.0
    still_ok = True
    for m in margins:
        res = _pde_residual(spec, model, _sample_shell(bounds, coords, m, n_eval))
        spread = ensemble["shell_rel_spread"][m] if ensemble else None
        ok = res <= residual_threshold and (spread is None or spread <= SPREAD_REL_TOL)
        shells.append({"margin": m, "residual": res, "rel_spread": spread, "ok": ok})
        still_ok = still_ok and ok
        if still_ok:
            supported = m
    return {"shells": shells, "supported_margin": supported, "residual_threshold": residual_threshold,
            "spread_tolerance": SPREAD_REL_TOL, "used_ensemble": ensemble is not None}


def weight_sensitivity_study(
    spec, build_model: Callable[[], Any], best_model, *, epochs: int, n_collocation: int,
    factors: Sequence[float] = DEFAULT_WEIGHT_FACTORS, n_eval: int = 2048,
) -> Dict[str, Any]:
    """Retrain with ``w_pde`` scaled by each factor (other weights at their
    defaults) and measure the RMS change of the prediction vs *best_model*
    relative to the field's RMS magnitude. Note: the seed differs from the
    original run's, so a part of the difference is seed noise -- compare
    with the ensemble's ``interior_rel_spread``."""
    from pinneapple_physics.pinn_solver.compiler.loss import LossWeights

    base_w = LossWeights()
    coords = list(spec.coords)
    x_in = _sample_box(spec.domain_bounds, coords, n_eval)
    ref = _predict(best_model, x_in)
    scale = _field_scale(ref)
    variants = []
    for f in factors:
        w = LossWeights(w_pde=base_w.w_pde * f, w_bc=base_w.w_bc, w_ic=base_w.w_ic, w_data=base_w.w_data)
        m = _train(spec, build_model, epochs=epochs, n_collocation=n_collocation, seed=2000, weights=w)
        diff = float(np.sqrt(np.mean((_predict(m, x_in) - ref) ** 2))) / scale
        variants.append({"w_pde_factor": f, "rel_prediction_change": diff,
                         "residual": _pde_residual(spec, m, x_in)})
    return {"variants": variants, "max_rel_change": max(v["rel_prediction_change"] for v in variants),
            "tolerance": SPREAD_REL_TOL}


def calibration_against_reference(members: Sequence[Any], reference_x, reference_y) -> Dict[str, Any]:
    """Real calibration of the ensemble against supplied reference data via
    ``CalibrationMetrics`` -- the only way to turn the spread into a
    *calibrated* uncertainty. Returns ece/coverage/sharpness."""
    import torch
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics

    x = np.asarray(reference_x, dtype="float32")
    preds = np.stack([_predict(m, x) for m in members])
    mean, std = preds.mean(axis=0), preds.std(axis=0)
    y_true = np.asarray(reference_y, dtype="float32")
    t = lambda a: torch.as_tensor(np.asarray(a, dtype="float32"))  # noqa: E731
    return {
        "ece": CalibrationMetrics.expected_calibration_error(t(mean), t(y_true), t(std)),
        "coverage": CalibrationMetrics.coverage_at_level(t(mean), t(y_true), t(std), alpha=0.1),
        "target_coverage": 0.9,
        "sharpness": CalibrationMetrics.sharpness(t(std)),
    }


def condition_coverage(spec, n: int = 50_000) -> Dict[str, Any]:
    """How many points each auto-sampled (``callable``/``all``) boundary,
    initial or data condition selects under the SAME sampler the installed
    ``pinneapple_physics.solve_pde`` uses, so this reports what training
    will actually enforce.

    Current PINNeAPPle samples the faces of the domain box and raises when
    a condition selects nothing (``condition_sampling.sample_condition_points``).
    Older versions only drew uniform interior points and silently dropped a
    condition like ``np.isclose(X[:, 1], 0.0)`` (the ``burgers_1d`` preset)
    -- in that case this falls back to uniform sampling and reports it as
    unenforced, which is exactly what that version's training did.
    ``tag`` conditions are reported as ``None`` (need real geometry).
    """
    coords, bounds = list(spec.coords), spec.domain_bounds
    try:
        from pinneapple_physics.pde_environment.condition_sampling import sample_condition_points
        sampler = "box-face sampler (current solve_pde)"
    except ImportError:
        sample_condition_points = None
        sampler = "uniform interior sampler (older solve_pde silently drops empty conditions)"
    x = _sample_box(bounds, coords, n)
    conds: Dict[str, Any] = {}
    for c in spec.conditions:
        if c.selector_type not in ("all", "callable"):
            conds[c.name] = None
        elif sample_condition_points is not None:
            conds[c.name] = int(sample_condition_points(
                lambda X, _c=c: _c.mask(X, {}), bounds, coords, max(64, n // 4), np.random.default_rng(0)).shape[0])
        else:
            conds[c.name] = int(np.asarray(c.mask(x, {}), dtype=bool).sum())
    return {"n_samples": n, "sampler": sampler, "selected_points": conds,
            "unenforced": [k for k, v in conds.items() if v == 0]}


DEFAULT_PARAM_SHIFTS = (0.10, 0.30)


def parameter_shift_study(
    make_spec: Callable[..., Any], base_kwargs: Dict[str, Any], build_model: Callable[[], Any], base_model, *,
    residual_threshold: float, epochs: int, n_collocation: int,
    shifts: Sequence[float] = DEFAULT_PARAM_SHIFTS, max_params: int = 2, n_eval: int = 2048,
) -> Dict[str, Any]:
    """Parameter generalization: how far can a physical parameter move
    before the surrogate trained at the base value stops being usable?

    For each numeric parameter in *base_kwargs* (at most *max_params*) and
    each relative *shift*, the problem is rebuilt with
    ``make_spec(**{param: value * (1 + shift)})`` and a FRESH model is
    trained on it. That reference model is only a proxy for the truth, so it
    is trusted only if its own PDE residual passes *residual_threshold*
    (``reference_trusted``). The measured quantity is the RMS difference
    between the base surrogate and the reference model on the same interior
    points, relative to the reference field's RMS magnitude. A shift is ``ok``
    when the reference is trusted and that difference is within
    ``SPREAD_REL_TOL``; ``supported_shift`` is the largest shift for which it
    and every smaller shift are ok. Costs ``len(shifts) * n_params``
    trainings. Geometry changes are NOT covered (they need real geometry).
    """
    results: List[Dict[str, Any]] = []
    numeric = [(k, v) for k, v in base_kwargs.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0][:max_params]
    for name, value in numeric:
        supported, still_ok, rows = 0.0, True, []
        for sh in shifts:
            spec = make_spec(**{**base_kwargs, name: value * (1 + sh)})
            ref = _train(spec, build_model, epochs=epochs, n_collocation=n_collocation, seed=3000)
            x = _sample_box(spec.domain_bounds, list(spec.coords), n_eval)
            ref_pred = _predict(ref, x)
            diff = float(np.sqrt(np.mean((_predict(base_model, x) - ref_pred) ** 2))) / _field_scale(ref_pred)
            resid = _pde_residual(spec, ref, x)
            trusted = resid <= residual_threshold
            ok = trusted and diff <= SPREAD_REL_TOL
            rows.append({"shift": sh, "value": value * (1 + sh), "rel_prediction_error": diff,
                         "reference_residual": resid, "reference_trusted": trusted, "ok": ok})
            still_ok = still_ok and ok
            if still_ok:
                supported = sh
        results.append({"parameter": name, "base_value": value, "supported_shift": supported, "shifts": rows})
    return {"parameters": results, "tolerance": SPREAD_REL_TOL, "residual_threshold": residual_threshold}
