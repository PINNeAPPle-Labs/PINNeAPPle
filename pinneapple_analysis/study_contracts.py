"""The shared data contracts (:mod:`pinneapple_data.contracts`) for uncertainty and sensitivity studies.

- :func:`space_from_contract` / :func:`space_to_contract`: a ``ParameterSpace`` is bounds per parameter; a study needs
  a distribution per parameter. A continuous or integer parameter becomes uniform on its bounds unless
  ``metadata["distributions"][name]`` gives another distribution (``{"dist": "normal", "mean": .., "std": ..}``)
  (integer parameters are sampled as continuous: the model must accept non-integer values);
  a constant parameter is held at its value; categorical and log-scaled parameters are refused (no ordering /
  no log-uniform sampling here) rather than silently treated as uniform.
- :func:`model_from_ref` / :func:`model_to_ref`: ``ModelRef`` <-> :class:`~pinneapple_analysis.study_models.StudyModel`
  (``zoo:<name>`` URIs, or ONNX bytes supplied with the reference).
- :func:`uncertainty_result` / :func:`sensitivity_result`: a ``StudyResult`` from the dicts of
  :func:`pinneapple_analysis.uncertainty.study.uncertainty_study` and
  :func:`pinneapple_analysis.sensitivity.study.sensitivity_study`, with the checks that actually ran.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pinneapple_analysis.study_models import StudyModel
from pinneapple_analysis.uncertainty.distributions import as_parameter

DEFAULT_GROUP = "other"


def _contracts():
    from pinneapple_data import contracts
    return contracts


def _as(cls, obj):
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, Mapping):
        return cls.model_validate(dict(obj))
    raise TypeError(f"expected a {cls.__name__} or a dict, got {type(obj).__name__}")


def is_parameter_space(obj: Any) -> bool:
    return (type(obj).__name__ == "ParameterSpace") or (isinstance(obj, Mapping) and obj.get("kind") == "ParameterSpace")


def space_from_contract(space) -> tuple[list[dict], dict[str, Any]]:
    """``(parameters, fixed)`` for a study from a ``ParameterSpace`` (object or dict)."""
    c = _contracts()
    ps = _as(c.ParameterSpace, space)
    dists = dict((ps.metadata or {}).get("distributions") or {})
    params, fixed = [], {}
    for p in ps.parameters:
        if p.kind == "constant":
            fixed[p.name] = p.value if p.value is not None else p.default
            continue
        if p.kind == "categorical":
            raise ValueError(f"{p.name}: categorical parameters are not supported in these studies "
                             "(fix it with kind 'constant', or run one study per category)")
        if p.name in dists:
            d = {"name": p.name, **dict(dists[p.name])}
        else:
            if p.log:
                raise ValueError(f"{p.name}: log-scaled parameter without a distribution; give one in "
                                 "metadata['distributions'] (e.g. lognormal)")
            if p.low is None or p.high is None:
                raise ValueError(f"{p.name}: needs low and high bounds or a distribution in metadata")
            d = {"name": p.name, "dist": "uniform", "low": float(p.low), "high": float(p.high)}
        if p.unit:
            d.setdefault("unit", p.unit)
        params.append(as_parameter(d).to_dict())
    return params, fixed


def space_to_contract(params: Sequence, name: str = "study", fixed: Mapping[str, Any] | None = None,
                      units: Mapping[str, str] | None = None):
    """A ``ParameterSpace`` with each parameter's support (0.1-99.9 % range for an unbounded distribution) as bounds
    and the distributions in ``metadata['distributions']``, so :func:`space_from_contract` reads it back exactly."""
    c = _contracts()
    out, dists = [], {}
    for spec in params:
        p = as_parameter(spec)
        lo, hi = p.bounds
        unit = p.unit or (units or {}).get(p.name, "")
        if p.dist == "constant":
            out.append(c.Parameter(name=p.name, kind="constant", value=p.params["value"], unit=unit))
            continue
        out.append(c.Parameter(name=p.name, low=float(lo), high=float(hi), unit=unit, group=DEFAULT_GROUP))
        dists[p.name] = {k: v for k, v in p.to_dict().items() if k not in ("name", "unit")}
    for k, v in (fixed or {}).items():
        out.append(c.Parameter(name=k, kind="constant", value=v, unit=(units or {}).get(k, "")))
    return c.ParameterSpace(name=name, parameters=out, metadata={"distributions": dists})


def model_from_ref(ref, onnx: bytes | None = None, outputs: Sequence[str] | None = None) -> StudyModel:
    """``zoo:<name>`` -> the zoo model; an ONNX reference (framework ``onnx`` or a ``.onnx`` URI) -> the graph in
    ``onnx`` bytes, named by the reference's inputs and outputs."""
    c = _contracts()
    r = _as(c.ModelRef, ref)
    outs = list(outputs or [v.name for v in r.outputs] or [])
    if r.uri.startswith("zoo:"):
        return StudyModel.from_zoo(r.uri[4:], outputs=outs or None)
    if r.framework.lower() == "onnx" or r.uri.lower().endswith(".onnx") or r.uri.startswith("onnx:"):
        if onnx is None:
            raise ValueError(f"the ModelRef {r.name!r} points to an ONNX graph: upload the .onnx file with it")
        return StudyModel.from_onnx(onnx, [v.name for v in r.inputs] or None, outs or None, name=r.name)
    raise ValueError(f"unsupported ModelRef uri {r.uri!r}: use zoo:<name> or an ONNX graph")


def model_to_ref(model: StudyModel):
    """A ``ModelRef`` for a study model: inputs with their validity range, outputs, held-out card metrics."""
    c = _contracts()
    uri = f"zoo:{model.name}" if model.kind == "zoo" else (f"onnx:{model.name}" if model.kind == "onnx"
                                                          else f"python:{model.name}")
    metrics = {}
    for k, v in (model.metadata.get("validation_metrics") or {}).items():
        if str(k).startswith("test/") and isinstance(v, (int, float)) and math.isfinite(v):
            metrics[str(k)] = float(v)
    inputs = [c.Variable(name=n, unit=model.units.get(n, ""), low=model.bounds.get(n, (None, None))[0],
                         high=model.bounds.get(n, (None, None))[1]) for n in model.input_names]
    outputs = [c.Variable(name=n, unit=model.units.get(n, "")) for n in model.output_names]
    return c.ModelRef(name=model.name, version=str(model.metadata.get("version") or "0.1.0"), uri=uri,
                      framework={"zoo": "torch", "onnx": "onnx"}.get(model.kind, "python"),
                      inputs=inputs, outputs=outputs, metrics=metrics, metrics_split="test",
                      dataset=c.DatasetRef(name=model.reference_dataset) if model.reference_dataset else None)


def _provenance(app: str, study_id: str, source: str = ""):
    c = _contracts()
    try:
        from pinneapple import __version__ as v
    except Exception:  # noqa: BLE001 - version is informative only
        v = ""
    return c.Provenance(app=app, study_id=study_id, source=source, library_version=str(v))


def uncertainty_result(out: Mapping, model: StudyModel, study_id: str, app: str = "uncertainty",
                       title: str = ""):
    """``StudyResult`` (``study_type='uncertainty'``) from an ``uncertainty_study`` dict."""
    c = _contracts()
    cf = out["conformal"]
    checks = [c.CheckRecord(name="held-out coverage of the prediction interval",
                            status="PASS" if cf["meets_target"] else "FAIL", value=cf["coverage"],
                            threshold=cf["target_coverage"],
                            evidence=f"{cf['n_test']} held-out cases, {int(100 * cf['ci_level'])} % Wilson interval "
                                     f"{cf['coverage_ci'][0]:.3f}-{cf['coverage_ci'][1]:.3f}")]
    low = [g for g in cf.get("groups", []) if g["coverage_ci"][1] < cf["target_coverage"]]
    if cf.get("groups"):
        checks.append(c.CheckRecord(name="coverage in every predicted-value group", status="WARN" if low else "PASS",
                                    value=min(g["coverage"] for g in cf["groups"]),
                                    threshold=cf["target_coverage"],
                                    evidence="under-covered: " + ", ".join(g["group"] for g in low) if low
                                    else f"{len(cf['groups'])} equal-count groups"))
    metrics: dict[str, Any] = {"coverage": cf["coverage"], "coverage_ci_low": cf["coverage_ci"][0],
                               "coverage_ci_high": cf["coverage_ci"][1], "target_coverage": cf["target_coverage"],
                               "interval_q": cf["q"], "mean_interval_width": cf["mean_width"], "n_cal": cf["n_cal"],
                               "n_test": cf["n_test"], "scale": out.get("scale")}
    cases = out.get("cases")
    if cases:
        s = cases["summary"]
        metrics.update(cases=s["n"], cases_to_high_fidelity=s["high_fidelity"])
        hf = cases.get("high_fidelity") or {}
        kept = (hf.get("by_route") or {}).get("surrogate")
        if kept:
            checks.append(c.CheckRecord(name="high-fidelity value inside the interval for the cases kept on the "
                                             "surrogate", status="PASS" if kept["in_interval"] == 1.0 else "WARN",
                                        value=kept["in_interval"], threshold=1.0,
                                        evidence=f"{kept['n']} cases solved by the high-fidelity solver"))
    prop = out.get("propagation")
    if prop:
        metrics.update(propagated_mean=prop["mean"], propagated_std=prop["std"],
                       p_within_limits=prop.get("p_within"),
                       p_within_limits_with_model_error=prop.get("p_within_with_model_error"))
        if prop.get("pce"):
            q2 = prop["pce"]["loo_q2"]
            checks.append(c.CheckRecord(name="polynomial chaos leave-one-out Q2", status="PASS" if q2 >= 0.9 else "WARN",
                                        value=q2, threshold=0.9, evidence=f"degree {prop['pce']['degree']}, "
                                                                          f"{prop['pce']['n_evals']} evaluations"))
    ps = None
    if prop:
        ps = space_to_contract(prop["uncertain"], name="uncertain inputs", fixed=prop.get("nominal"),
                               units=model.units)
    return c.StudyResult(id=study_id, study_type="uncertainty", app=app, status="done",
                         title=title or f"Uncertainty of {out['output']} ({model.name})", parameter_space=ps,
                         models=[model_to_ref(model)], metrics=metrics, checks=checks,
                         limits=["conformal coverage is marginal: it holds on average for cases like the "
                                 "calibration data, not case by case",
                                 "the applicability gate looks at inputs only (range and data density)",
                                 "propagation assumes independent inputs"],
                         provenance=_provenance(app, study_id, out.get("reference_note", "")))


def sensitivity_result(out: Mapping, model: StudyModel | None, study_id: str, app: str = "sensitivity",
                       title: str = ""):
    """``StudyResult`` (``study_type='sensitivity'``) from a ``sensitivity_study`` dict (or the runs-elsewhere dict
    with ``sobol``, ``freeze`` and ``parameters``)."""
    c = _contracts()
    s = out["sobol"]
    metrics: dict[str, Any] = {"n_evals": (out.get("n_evals") or {}).get("total"), "n_base": s["n_base"],
                               "parameters": len(s["names"])}
    for k, o in enumerate(s["outputs"]):
        top = s["ranking"][o][0] if "ranking" in s else s["names"][0]
        metrics[f"top_parameter[{o}]"] = top
        metrics[f"top_total_index[{o}]"] = float(s["ST"][k][s["names"].index(top)])
    frozen = [r["name"] for r in out["freeze"]["rows"] if r["decision"] == "freeze"]
    metrics["frozen"] = ", ".join(frozen)
    checks = []
    if out.get("exact"):
        ex = out["exact"]
        ok = all(ex["S1_in_ci"]) and all(ex["ST_in_ci"])
        checks.append(c.CheckRecord(name="estimates against the exact indices", status="PASS" if ok else "WARN",
                                    value=max(ex["max_abs_error_S1"], ex["max_abs_error_ST"]),
                                    evidence="exact value inside the bootstrap interval for "
                                             f"{sum(ex['S1_in_ci']) + sum(ex['ST_in_ci'])} of "
                                             f"{len(ex['S1_in_ci']) + len(ex['ST_in_ci'])} indices"))
    ck = out["freeze"].get("check")
    if ck:
        worst = max(ck["max_change_sd"])
        checks.append(c.CheckRecord(name="re-run with the frozen parameters fixed", status="PASS" if worst < 0.25
                                    else "WARN", value=worst, threshold=0.25,
                                    evidence=f"{ck['n']} samples; largest change in output standard deviations; "
                                             f"variance ratio {', '.join(f'{v:.4f}' for v in ck['variance_ratio'])}"))
    elif frozen:
        checks.append(c.CheckRecord(name="re-run with the frozen parameters fixed", status="NOT_RUN",
                                    detail="the model is not available here (runs done elsewhere)"))
    params = out.get("parameters") or []
    ps = space_to_contract(params, name="varied parameters", fixed=out.get("fixed"),
                           units=model.units if model else None) if params else None
    return c.StudyResult(id=study_id, study_type="sensitivity", app=app, status="done",
                         title=title or "Sensitivity study", parameter_space=ps,
                         models=[model_to_ref(model)] if model is not None else [], metrics=metrics, checks=checks,
                         limits=["indices are shares of variance for the distributions given",
                                 "designed studies sample inputs independently",
                                 "at small Sobol sample sizes the bootstrap intervals can be too narrow"],
                         provenance=_provenance(app, study_id))


__all__ = ["space_from_contract", "space_to_contract", "is_parameter_space", "model_from_ref", "model_to_ref",
           "uncertainty_result", "sensitivity_result"]
