"""Steps of a study that take a *model* rather than numbers, in a form a pipeline can run and chain.

A sensitivity, uncertainty or identification study needs the model it studies, a function, while a pipeline step
can only pass JSON from one sandboxed process to the next. A **model reference** is that function as data::

    {"tool": "pinneapple_physics.closed_form.module:function",   # a public PINNeAPPle tool (resolve_model rules)
     "fixed": {"k": 15.0},                                         # inputs held at a value
     "inputs": ["q", "L"],                                         # inputs the study varies, in order
     "outputs": ["T_max"],                                         # outputs it reads (a mapping key, an attribute,
                                                                   # or the single value the tool returns)
     "call": "kwargs"}                                             # or "array": tool(X) with X rows in input order

:func:`model_from_ref` rebuilds it inside each step's own process, only from inside the library
(:func:`pinneapple_autolab.study.verify.resolve_model`). The steps return plain dicts with their numbers and their
checks (``{"code", "status": "pass" | "fail" | "inconclusive", "value", "detail"}``), so the pipeline shows what
was measured and whether it holds:

* :func:`sensitivity_step` - Sobol indices (``pinneapple_analysis.sensitivity.study.sensitivity_study``);
* :func:`uncertainty_step` - Monte Carlo propagation of input distributions, with its convergence check;
* :func:`identify_step` - parameters from measurements (``pinneapple_analysis.inverse_problems.calibrate``), with
  intervals and the fit on held-out rows; also the "fit" step of a data analysis;
* :func:`analyze_step` - predictability of a time series before any model
  (``pinneapple_systems.time_series.predictability.predictability_report``);
* :func:`forecast_step` - an autoregressive forecast scored per horizon against persistence and the mean on the
  last part of the series, which it never sees while fitting.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"


def _check(code: str, ok: bool | None, value: Any, detail: str) -> dict:
    return {"code": code, "status": INCONCLUSIVE if ok is None else (PASS if ok else FAIL), "value": value,
            "detail": detail}


def _scalar(result: Any, name: str) -> float:
    if isinstance(result, Mapping):
        result = result[name]
    elif not isinstance(result, (int, float, np.floating, np.integer, np.ndarray)) and hasattr(result, name):
        result = getattr(result, name)
    return float(np.asarray(result, float).ravel()[0])


def model_from_ref(ref: Mapping[str, Any]) -> Callable[[Mapping[str, float]], dict[str, float]]:
    """``f(point) -> {output: value}`` from a model reference (see the module text)."""
    from pinneapple_autolab.study.verify import resolve_model
    fn = resolve_model(str(ref["tool"]))
    fixed = dict(ref.get("fixed") or {})
    inputs, outputs = list(ref.get("inputs") or []), list(ref.get("outputs") or [])
    if not outputs:
        raise ValueError("a model reference names the outputs it reads")
    call = ref.get("call", "kwargs")
    if call not in ("kwargs", "array"):
        raise ValueError("call is 'kwargs' or 'array'")

    def f(point: Mapping[str, float]) -> dict[str, float]:
        missing = [n for n in inputs if n not in point]
        if missing:
            raise ValueError(f"missing inputs {missing}")
        if call == "array":
            out = fn(np.array([[float(point[n]) for n in inputs]]), **fixed)
            arr = np.asarray(out, float).reshape(1, -1)
            return {o: float(arr[0, i]) for i, o in enumerate(outputs)}
        out = fn(**fixed, **{n: float(point[n]) for n in inputs})
        return {o: _scalar(out, o) if (isinstance(out, Mapping) or len(outputs) > 1 or hasattr(out, o))
                else float(np.asarray(out, float).ravel()[0]) for o in outputs}
    f.inputs, f.outputs = inputs, outputs              # type: ignore[attr-defined]
    return f


def _batch(f, names: Sequence[str], output: str):
    def predict(X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.array([[f(dict(zip(names, row, strict=True)))[output]] for row in X])
    return predict


# ------------------------------------------------------------------------------------------------ sensitivity
def sensitivity_step(model: Mapping[str, Any], ranges: Mapping[str, Sequence[float]], output: str, *,
                     n: int = 1024, seed: int = 0, tol: float = 0.1) -> dict:
    """First-order and total Sobol indices of ``output`` over uniform ``ranges`` ``{input: [low, high]}``.

    Checks: ``sensitivity.indices_sum`` (sum of first-order indices <= 1, with 0.05 of sampling allowance);
    ``sensitivity.converged`` (the widest bootstrap 95 % interval of an index has a half-width <= ``tol``)."""
    from pinneapple_analysis.sensitivity.study import sensitivity_study
    from pinneapple_analysis.study_models import StudyModel
    f = model_from_ref(model)
    names = [n_ for n_ in f.inputs if n_ in ranges]
    if not names:
        raise ValueError("no varied input has a range")
    sm = StudyModel.from_function(_batch(f, names, output), names, [output], name=str(model["tool"]))
    space = [{"name": k, "dist": "uniform", "low": float(ranges[k][0]), "high": float(ranges[k][1])} for k in names]
    s = sensitivity_study(sm, space, [output], n_sobol=n, morris_on=False, effects_on=False, n_boot=100,
                          n_verify=0, seed=seed)["sobol"]
    s1 = {k: float(v) for k, v in zip(names, s["S1"][0], strict=True)}
    st = {k: float(v) for k, v in zip(names, s["ST"][0], strict=True)}
    half = max(0.5 * (hi - lo) for key in ("S1_ci", "ST_ci")
               for lo, hi in zip(s[key][0][0], s[key][1][0], strict=True))
    tot = sum(s1.values())
    return {"output": output, "S1": s1, "ST": st, "ranking": sorted(st, key=st.get, reverse=True), "n": n,
            "n_evals": int(s["n_evals"]),
            "checks": [_check("sensitivity.indices_sum", tot <= 1.05, tot, f"sum of first-order indices {tot:.3f}"),
                       _check("sensitivity.converged", half <= tol, half,
                              f"widest 95 % bootstrap half-width of an index: {half:.3f} (<= {tol})")]}


# ------------------------------------------------------------------------------------------------ uncertainty
def _sample(spec: Mapping[str, Any], rng: np.random.Generator, n: int) -> np.ndarray:
    d = str(spec.get("dist", "uniform")).lower()
    if d == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]), n)
    if d == "normal":
        return rng.normal(float(spec["mean"]), float(spec["std"]), n)
    if d == "lognormal":
        return rng.lognormal(float(spec["mu"]), float(spec["sigma"]), n)
    raise ValueError(f"unknown distribution {d!r} (uniform, normal, lognormal)")


def uncertainty_step(model: Mapping[str, Any], distributions: Mapping[str, Mapping[str, Any]], output: str, *,
                     n: int = 1000, seed: int = 0, quantiles: Sequence[float] = (0.05, 0.5, 0.95),
                     limit: float | None = None) -> dict:
    """Monte Carlo propagation of input ``distributions`` (``{input: {"dist": "normal", "mean", "std"} | {"dist":
    "uniform", "low", "high"} | {"dist": "lognormal", "mu", "sigma"}}``) to ``output``: mean, standard deviation,
    quantiles and, with ``limit``, the probability of exceeding it (with its binomial standard error).

    Check ``uq.converged``: mean and standard deviation from the two halves of the sample agree within three of
    their standard errors."""
    f = model_from_ref(model)
    rng = np.random.default_rng(seed)
    names = [k for k in f.inputs if k in distributions]
    X = {k: _sample(distributions[k], rng, n) for k in names}
    y = np.array([f({k: X[k][i] for k in names})[output] for i in range(n)])
    finite = np.isfinite(y)
    y = y[finite]
    a, b = y[: len(y) // 2], y[len(y) // 2:]
    se_mean = float(np.std(y, ddof=1) / math.sqrt(len(y) / 2)) * math.sqrt(2)
    se_std = float(np.std(y, ddof=1) / math.sqrt(2 * (len(y) / 2 - 1))) * math.sqrt(2)
    ok = abs(a.mean() - b.mean()) <= 3 * se_mean and abs(a.std(ddof=1) - b.std(ddof=1)) <= 3 * se_std
    out = {"output": output, "n": int(len(y)), "non_finite": int((~finite).sum()), "mean": float(y.mean()),
           "std": float(y.std(ddof=1)), "quantiles": {str(q): float(np.quantile(y, q)) for q in quantiles},
           "checks": [_check("uq.converged", bool(ok), {"mean_halves": [float(a.mean()), float(b.mean())]},
                             "mean and standard deviation of the two half-samples agree within 3 standard errors"
                             if ok else "the two half-samples disagree: increase n")]}
    if limit is not None:
        p = float((y > limit).mean())
        out["p_exceed"] = {"limit": float(limit), "p": p, "se": math.sqrt(max(p * (1 - p), 1e-300) / len(y))}
    return out


# ------------------------------------------------------------------------------------------------ identification
def identify_step(model: Mapping[str, Any], parameters: Mapping[str, Mapping[str, float]],
                  measurements: Mapping[str, Any], output: str, *, holdout_fraction: float = 0.2,
                  bootstrap: int = 50, seed: int = 0) -> dict:
    """Identify ``parameters`` (``{name: {"initial", "lower", "upper", "unit"}}``, inputs of the model) from
    ``measurements`` (``{"inputs": {column: [...]}, "values": [...], "sigma": float | [...]}``): the other inputs of
    the model come from the measurement columns, the rest from ``model["fixed"]``.

    Checks: ``identify.fits_data`` (reduced chi-square of the fit <= 2, i.e. residuals of the size of the stated
    noise); ``identify.identifiable`` (every parameter identifiable and away from its bounds); ``verify.held_out`` (the
    rows never used in the fit: mean (residual / sigma)^2 <= 4)."""
    from pinneapple_analysis.inverse_problems.calibration import calibrate
    from pinneapple_analysis.inverse_problems.forward import CallableModel, Observations, Parameter
    from pinneapple_autolab.study.verify import resolve_model
    fn = resolve_model(str(model["tool"]))
    fixed = dict(model.get("fixed") or {})
    cols = {k: np.asarray(v, float) for k, v in measurements["inputs"].items()}
    values = np.asarray(measurements["values"], float)
    sigma = np.broadcast_to(np.asarray(measurements.get("sigma", 1.0), float), values.shape).copy()

    def predict(theta, inputs):
        n = len(next(iter(inputs.values()))) if inputs else 1
        out = []
        for i in range(n):
            kw = {**fixed, **{k: float(v[i]) for k, v in inputs.items()}, **{k: float(v) for k, v in theta.items()}}
            r = fn(**kw)
            out.append(float(r) if isinstance(r, (int, float, np.floating)) else _scalar(r, output))
        return np.array(out)
    params = [Parameter(name=k, value=float(p.get("initial", 0.5 * (p["lower"] + p["upper"]))),
                        lower=float(p["lower"]), upper=float(p["upper"]), unit=str(p.get("unit", "")))
              for k, p in parameters.items()]
    cm = CallableModel(predict, inputs=list(cols), outputs=[output], parameters=params, name=str(model["tool"]))
    obs = Observations(inputs=cols, values=values, sigma=sigma)
    d = calibrate(cm, obs, params, holdout_fraction=holdout_fraction, bootstrap=bootstrap, seed=seed).to_dict()
    rows = {p["name"]: p for p in d["parameters"]}
    est = {k: float(r["calibrated"]) for k, r in rows.items()}
    ints = {k: [float(v) for v in r["ci_jacobian"]] if r.get("ci_jacobian") else None for k, r in rows.items()}
    chi2 = d["fit"].get("reduced_chi2")
    identifiable = all(r.get("identifiability") == "identifiable" and not r.get("at_bound") for r in rows.values())
    hold = (d.get("metrics") or {}).get("holdout", {}).get("after")
    hz = hold.get("chi2_per_point") if hold else None
    return {"output": output, "estimates": est, "intervals": ints, "reduced_chi2": chi2, "holdout": hold,
            "parameters": d["parameters"], "warnings": d.get("warnings", []),
            "checks": [_check("identify.fits_data", None if chi2 is None else float(chi2) <= 2.0, chi2,
                              "reduced chi-square of the fit: residuals against the stated noise (<= 2)"),
                       _check("identify.identifiable", identifiable,
                              {k: r.get("identifiability") for k, r in rows.items()},
                              "every parameter identifiable and away from its bounds"),
                       _check("verify.held_out", None if hz is None else float(hz) <= 4.0, hz,
                              "rows never used in the fit: mean (residual / sigma)^2 (<= 4, i.e. within 2 sigma RMS)")]}


# ------------------------------------------------------------------------------------------------ time series
def analyze_step(series: Sequence[float], *, dt: float = 1.0, seed: int = 0) -> dict:
    """Predictability of a series before any model (AR fit, entropy, surrogate tests, stationarity), with
    ``analyze.stationarity`` stating whether the tests ran on levels or on differences."""
    from pinneapple_systems.time_series.predictability import predictability_report
    r = predictability_report(np.asarray(series, float), dt=dt, seed=seed, lyapunov=False, n_surrogates=39)
    on = r.get("tested_on")
    return {"report": r, "checks": [_check("analyze.stationarity", True, on,
                                           f"tests ran on the {on} (differences when the AR(1) coefficient > 0.99)")]}


def _ar_forecast(hist: np.ndarray, intercept: float, coef: np.ndarray, h: int) -> np.ndarray:
    buf = list(hist[-len(coef):]) if len(coef) else []
    out = []
    for _ in range(h):
        y = intercept + sum(c * buf[-k - 1] for k, c in enumerate(coef))
        out.append(y)
        buf.append(y)
    return np.array(out)


def forecast_step(series: Sequence[float], horizon: int, *, test_fraction: float = 0.25,
                  max_order: int = 20) -> dict:
    """An AR(p) forecast (order by AIC, fitted on the first part of the series only) scored on rolling origins over
    the last ``test_fraction`` of the series, per horizon, against persistence and the training mean.

    Check ``verify.beats_baseline``: the model's mean absolute error is below both baselines at horizon 1 and
    averaged over the horizons; a series that no baseline beats is reported as such, never as a skilful model."""
    from pinneapple_systems.time_series.predictability import fit_ar
    x = np.asarray(series, float).ravel()
    n_train = int(len(x) * (1 - test_fraction))
    if n_train <= max_order + 10 or len(x) - n_train <= horizon:
        raise ValueError("the series is too short for this horizon and test fraction")
    ar = fit_ar(x[:n_train], max_order=max_order)
    mean = float(x[:n_train].mean())
    err = {m: np.zeros(horizon) for m in ("model", "persistence", "mean")}
    count = 0
    for t in range(n_train, len(x) - horizon):
        truth = x[t:t + horizon]
        err["model"] += np.abs(_ar_forecast(x[:t], ar.intercept, ar.coef, horizon) - truth)
        err["persistence"] += np.abs(x[t - 1] - truth)
        err["mean"] += np.abs(mean - truth)
        count += 1
    mae = {m: (e / count).tolist() for m, e in err.items()}
    m1, base1 = mae["model"][0], min(mae["persistence"][0], mae["mean"][0])
    mavg, baseavg = float(np.mean(mae["model"])), min(float(np.mean(mae["persistence"])), float(np.mean(mae["mean"])))
    ok = m1 < base1 and mavg < baseavg
    nxt = _ar_forecast(x, ar.intercept, ar.coef, horizon).tolist()
    return {"order": ar.order, "mae_by_horizon": mae, "origins": count, "forecast": nxt,
            "checks": [_check("forecast.horizon", True, mae["model"], "mean absolute error per horizon"),
                       _check("verify.beats_baseline", bool(ok), {"model_h1": m1, "best_baseline_h1": base1,
                                                                   "model_avg": mavg, "best_baseline_avg": baseavg},
                              "the model beats persistence and the mean out of sample" if ok else
                              "no skill over the naive baselines out of sample")]}


__all__ = ["model_from_ref", "sensitivity_step", "uncertainty_step", "identify_step", "analyze_step", "forecast_step"]
