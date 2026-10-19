"""The inverse-problems core in the shared data contracts (``pinneapple_data.contracts``).

Identifiability reports, calibrations and method comparisons become ``StudyResult`` files (study types
``"identifiability"``, ``"calibration"``, ``"inverse"``), parameters travel as a ``ParameterSpace``, and a
calibrated model with held-out metrics becomes a ``ModelRef``. So the apps chain: a parameter space from the
Campaign Planner is the input of the Identifiability Lab, whose identifiable parameters go to the Calibration Lab,
whose calibrated values and held-out errors go on to uncertainty or optimisation studies.

Checks are recorded only when they ran, each with its value (``CheckRecord`` refuses PASS/FAIL without one).
"""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .forward import Parameter

__all__ = ["parameters_from_space", "to_parameter_space", "identifiability_study", "calibration_study",
           "inverse_study"]

_GROUPS = {"geometry", "material", "boundary", "operating", "numerics", "other"}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("pinneapple")
    except Exception:  # noqa: BLE001 - running from a checkout without metadata
        return ""


def parameters_from_space(space, nominal: Mapping[str, float] | None = None) -> list[Parameter]:
    """``Parameter`` list from a ``ParameterSpace``: continuous and integer parameters are free (value: ``nominal``,
    else ``default``, else the middle of the range, geometric for log ones); constants are fixed. Categorical
    parameters are not estimable here and raise."""
    out = []
    for p in space.parameters:
        if p.kind == "constant":
            v = float(p.value)
            out.append(Parameter(p.name, v, unit=p.unit, fixed=True, log=False, description=p.description))
            continue
        if p.kind == "categorical":
            raise ValueError(f"parameter {p.name!r} is categorical: fix it (one value per study) to estimate the rest")
        lo, hi = float(p.low), float(p.high)
        v = (nominal or {}).get(p.name, p.default)
        if v is None:
            v = math.sqrt(lo * hi) if p.log else (lo + hi) / 2
        out.append(Parameter(p.name, float(v), lo, hi, p.unit, log=bool(p.log) or None,
                             description=p.description))
    return out


def to_parameter_space(params: Sequence[Parameter], name: str, description: str = "",
                       groups: Mapping[str, str] | None = None):
    """A ``ParameterSpace``: free parameters continuous inside their bounds, fixed ones constant."""
    from pinneapple_data.contracts import Parameter as CParameter
    from pinneapple_data.contracts import ParameterSpace

    ps = []
    for p in params:
        g = (groups or {}).get(p.name, "other")
        g = g if g in _GROUPS else "other"
        if p.fixed or not (math.isfinite(p.lower) and math.isfinite(p.upper)) or p.lower >= p.upper:
            ps.append(CParameter(name=p.name, kind="constant", value=p.value, unit=p.unit, group=g,
                                 description=p.description))
        else:
            ps.append(CParameter(name=p.name, low=p.lower, high=p.upper, unit=p.unit, log=bool(p.log), group=g,
                                 default=p.value, description=p.description))
    return ParameterSpace(name=name, description=description, parameters=ps)


def _provenance(app: str, study_id: str, source: str, notes: str = ""):
    from pinneapple_data.contracts import Provenance

    return Provenance(app=app, study_id=study_id, source=source, library_version=_version(), notes=notes)


def _finite(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def identifiability_study(report: Mapping[str, Any], params: Sequence[Parameter], *, app: str = "identifiability",
                          title: str = "", model: str = "", study_id: str | None = None):
    """``StudyResult`` (``study_type="identifiability"``) from ``IdentifiabilityReport.to_dict()``: one row per
    parameter (status, Cramér-Rao bound, profile interval), the rank and conditioning as metrics, one check per
    method that ran."""
    from pinneapple_data.contracts import CheckRecord, StudyResult, Table

    sid = study_id or uuid.uuid4().hex[:12]
    rows = report["parameters"]
    table = Table(columns={
        "parameter": [r["name"] for r in rows], "unit": [r["unit"] for r in rows],
        "status": [r["status"] for r in rows], "value": [_finite(r["value"]) for r in rows],
        "crlb_std": [_finite(r.get("crlb_std")) for r in rows],
        "crlb_relative": [_finite(r.get("crlb_relative")) for r in rows],
        "profile_lo": [_finite((r.get("profile_ci95") or [None, None])[0]) for r in rows],
        "profile_hi": [_finite((r.get("profile_ci95") or [None, None])[1]) for r in rows],
        "sensitivity": [_finite(r["sensitivity"]) for r in rows]})
    p = report["n_parameters"]
    checks = [CheckRecord(name="fisher_rank", status="PASS" if report["rank"] == p else "FAIL", value=report["rank"],
                          threshold=p, evidence=f"study {sid}: eigenvalues of the Fisher information",
                          detail="local structural identifiability at the analysis point")]
    for r in rows:
        checks.append(CheckRecord(name=f"identifiable:{r['name']}",
                                  status="PASS" if r["status"] == "identifiable" else "FAIL",
                                  value=r["status"], evidence=f"study {sid}: {r['reason']}"))
    if report.get("profiles"):
        for name, pr in report["profiles"].items():
            closed = pr["ci95"][0] is not None and pr["ci95"][1] is not None
            checks.append(CheckRecord(name=f"profile_likelihood:{name}", status="PASS" if closed else "FAIL",
                                      value=pr["verdict"], threshold="delta chi2 = 3.84 on both sides",
                                      evidence=f"study {sid}: {len(pr['grid'])} profile points"))
    else:
        checks.append(CheckRecord(name="profile_likelihood", status="NOT_RUN", detail="skipped for this model"))
    post = report.get("posterior")
    checks.append(CheckRecord(name="posterior_mcmc", status="N/A" if post is None else "PASS",
                              value=None if post is None else post["n_samples"],
                              evidence="" if post is None else f"study {sid}: adaptive Metropolis, acceptance "
                                                               f"{post['acceptance']}",
                              detail="posterior correlations" if post else "not run (expensive model or no bounds)"))
    metrics = {"rank": report["rank"], "n_parameters": p, "n_observations": report["n_observations"],
               "condition_number": _finite(report.get("condition_number")),
               "log_det_fisher": _finite(report.get("d_criterion")),
               "n_identifiable": sum(r["status"] == "identifiable" for r in rows)}
    limits = ["Fisher rank is a local numerical test at the analysis point, not a proof of global identifiability.",
              "Profile likelihood and posterior depend on the parameter bounds given."]
    return StudyResult(id=sid, study_type="identifiability", app=app, title=title or f"Identifiability: {model}",
                       parameter_space=to_parameter_space(params, name=model or "parameters"), table=table,
                       metrics=metrics, checks=checks, limits=limits,
                       provenance=_provenance(app, sid, f"model {model}", report.get("summary", "")),
                       metadata={"summary": report.get("summary"), "ambiguities": report.get("ambiguities"),
                                 "suggestions": report.get("suggestions"), "data": report.get("data")})


def calibration_study(result: Mapping[str, Any], params: Sequence[Parameter], *, app: str = "calibration",
                      title: str = "", model: str = "", model_spec: Mapping[str, Any] | None = None,
                      study_id: str | None = None):
    """``StudyResult`` (``study_type="calibration"``) from ``CalibrationResult.to_dict()``, with a ``ModelRef`` of
    the calibrated model (``uri="inline:"``, the calibrated values in ``metadata``) when held-out metrics exist."""
    from pinneapple_data.contracts import CheckRecord, ModelRef, StudyResult, Table

    sid = study_id or uuid.uuid4().hex[:12]
    rows = result["parameters"]

    def ci(r, k, i):
        v = r.get(k)
        return _finite(v[i]) if v else None

    table = Table(columns={
        "parameter": [r["name"] for r in rows], "unit": [r["unit"] for r in rows],
        "initial": [_finite(r["initial"]) for r in rows], "calibrated": [_finite(r["calibrated"]) for r in rows],
        "ci_jacobian_lo": [ci(r, "ci_jacobian", 0) for r in rows],
        "ci_jacobian_hi": [ci(r, "ci_jacobian", 1) for r in rows],
        "ci_bootstrap_lo": [ci(r, "ci_bootstrap", 0) for r in rows],
        "ci_bootstrap_hi": [ci(r, "ci_bootstrap", 1) for r in rows],
        "identifiability": [r["identifiability"] for r in rows]})
    m = result["metrics"]
    fit = result["fit"]
    metrics: dict[str, Any] = {"fit_rmse_before": _finite(m["fit"]["before"]["rmse"]),
                               "fit_rmse_after": _finite(m["fit"]["after"]["rmse"]),
                               "reduced_chi2": _finite(fit["reduced_chi2"]), "n_fit": m["fit"]["after"]["n"]}
    checks = []
    rc = fit["reduced_chi2"]
    checks.append(CheckRecord(name="reduced_chi2", status="PASS" if 0.25 <= rc <= 4 else "WARN", value=_finite(rc),
                              threshold="0.25 to 4", evidence=f"study {sid}: weighted residuals of the fit"))
    hold = m.get("holdout")
    models = []
    if hold:
        h = hold["after"]
        metrics.update(holdout_rmse_before=_finite(hold["before"]["rmse"]), holdout_rmse_after=_finite(h["rmse"]),
                       n_holdout=h["n"])
        checks.append(CheckRecord(name="holdout_error", status="PASS" if h["chi2_per_point"] <= 4 else "WARN",
                                  value=_finite(h["chi2_per_point"]), threshold="mean (error/sigma)^2 <= 4",
                                  evidence=f"study {sid}: {h['n']} held-out measurements, never fitted"))
        calibrated = {r["name"]: r["calibrated"] for r in rows}
        models.append(ModelRef(name=f"{model or 'model'}-calibrated", model_type="calibrated_physics_model",
                               uri="inline:", framework="pinneapple_analysis.inverse_problems",
                               metrics={"holdout_rmse": float(h["rmse"]), "holdout_mae": float(h["mae"])},
                               metrics_split="holdout", provenance=_provenance(app, sid, "calibration"),
                               metadata={"calibrated": calibrated, "model_spec": dict(model_spec or {})}))
    else:
        checks.append(CheckRecord(name="holdout_error", status="NOT_RUN", detail="no held-out measurements"))
    for r in rows:
        ok = r["ci_jacobian"] is not None
        checks.append(CheckRecord(name=f"estimable:{r['name']}", status="PASS" if ok else "FAIL",
                                  value=r["identifiability"],
                                  evidence=f"study {sid}: identifiability of the fit design at the calibrated values",
                                  detail=r.get("not_estimable", "")))
    limits = ["Intervals assume the model form is right; a reduced chi-square far above 1 says it is not.",
              "Non-identifiable parameters have no interval: their calibrated value is one of many that fit."]
    space = to_parameter_space([Parameter.from_dict({**p.to_dict(), "value": next(
        (r["calibrated"] for r in rows if r["name"] == p.name), p.value)}) for p in params], name=model or "model")
    return StudyResult(id=sid, study_type="calibration", app=app, title=title or f"Calibration: {model}",
                       parameter_space=space, table=table, metrics=metrics, checks=checks, models=models,
                       limits=limits, provenance=_provenance(app, sid, f"model {model}"),
                       metadata={"warnings": result.get("warnings", [])})


def inverse_study(comparison: Mapping[str, Any], params: Sequence[Parameter], *, app: str = "inverse",
                  title: str = "", study_id: str | None = None):
    """``StudyResult`` (``study_type="inverse"``) from ``compare_methods(...)``: one row per method and parameter,
    validation RMSE per method, truth-recovery checks when the truth is known."""
    from pinneapple_data.contracts import CheckRecord, StudyResult, Table

    sid = study_id or uuid.uuid4().hex[:12]
    cols: dict[str, list] = {k: [] for k in ("method", "parameter", "unit", "estimate", "lo95", "hi95", "truth",
                                             "identifiability")}
    checks = []
    metrics: dict[str, Any] = {"best_method": comparison.get("best")}
    for s in comparison["solutions"]:
        if s["status"] != "done":
            checks.append(CheckRecord(name=f"method:{s['method']}", status="N/A", detail=s.get("reason", "")))
            continue
        for r in s["parameters"]:
            iv = r.get("interval95") or [None, None]
            cols["method"].append(s["method"])
            cols["parameter"].append(r["name"])
            cols["unit"].append(r["unit"])
            cols["estimate"].append(_finite(r["estimate"]))
            cols["lo95"].append(_finite(iv[0]))
            cols["hi95"].append(_finite(iv[1]))
            cols["truth"].append(_finite(r.get("truth")))
            cols["identifiability"].append(r.get("identifiability"))
        if s.get("validation"):
            metrics[f"validation_rmse:{s['method']}"] = _finite(s["validation"]["rmse"])
        metrics[f"fit_rmse:{s['method']}"] = _finite(s["fit"]["rmse"])
        if s.get("truth_recovered") is not None:
            checks.append(CheckRecord(name=f"truth_recovery:{s['method']}",
                                      status="PASS" if s["truth_recovered"] else "FAIL",
                                      value=str(s["truth_recovered"]), threshold=s.get("truth_check", ""),
                                      evidence=f"study {sid}: synthetic data with a known answer"))
    name = comparison.get("problem", {}).get("name", "inverse problem")
    limits = ["Truth recovery is checked only on synthetic data with a known answer.",
              "A single PINN run gives no uncertainty."]
    return StudyResult(id=sid, study_type="inverse", app=app, title=title or name,
                       parameter_space=to_parameter_space(params, name=name), table=Table(columns=cols),
                       metrics=metrics, checks=checks, limits=limits, provenance=_provenance(app, sid, name),
                       metadata={"ranked_by": comparison.get("ranked_by")})
