"""The operations tools speak ``pinneapple_data.contracts``: models travel as ``ModelRef`` (with their held-out
validation metrics and validity ranges) and every twin, anomaly or control study ends as a ``StudyResult`` whose
checks record what actually ran.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from pinneapple_data.contracts import (
    CheckRecord,
    DatasetRef,
    ModelRef,
    Provenance,
    StudyResult,
    Table,
    Variable,
    sha256_file,
)

from .plant import SENSOR_UNITS
from .rom import ControlledOpInf
from .twin import LumpedTankModel, SurrogateModel, model_from_dict


def _library_version() -> str:
    try:
        from importlib.metadata import version

        return version("pinneapple")
    except Exception:  # pragma: no cover - not installed as a distribution
        return ""


def save_model(model, path: str | Path, name: str, *, app: str = "", dataset: DatasetRef | None = None,
               trust_decision: str = "UNCHECKED") -> ModelRef:
    """Write ``model`` (a twin model or a :class:`ControlledOpInf`) as JSON at ``path`` and return its
    ``ModelRef`` (saved next to it as ``<path>.modelref.json``). Held-out metrics come from the model's own
    validation record; a model without one carries no metrics."""
    path = Path(path)
    m = model.rom if isinstance(model, SurrogateModel) else model
    payload = m.to_dict()
    path.write_text(json.dumps(payload))
    if isinstance(m, ControlledOpInf):
        inputs = [Variable(name=n, unit=SENSOR_UNITS.get(n, ""), low=m.bounds.get(n, [None, None])[0],
                           high=m.bounds.get(n, [None, None])[1]) for n in m.state_names + m.input_names]
        outputs = [Variable(name=n, unit=SENSOR_UNITS.get(n, ""), description=f"{n} one step ({m.dt:g} s) later")
                   for n in m.state_names]
        v = m.validation
        metrics = {}
        for key in ("one_step_rmse", "rollout_rmse", "rollout_rel_rmse", "rollout_max_abs_error"):
            for state, val in (v.get(key) or {}).items():
                metrics[f"{key}_{state}"] = float(val)
        model_type, framework = "rom", "pinneapple_systems.operations.rom (controlled operator inference)"
    elif isinstance(m, LumpedTankModel):
        inputs = [Variable(name=n, unit=SENSOR_UNITS.get(n, "")) for n in ["h", "T"] + m.input_names]
        outputs = [Variable(name=n, unit=SENSOR_UNITS.get(n, "")) for n in m.state_names]
        metrics, model_type = {}, "lumped"
        framework = "pinneapple_systems.operations.plant (mass and energy balances)"
    else:
        raise TypeError(f"cannot describe a {type(m).__name__} as a ModelRef")
    ref = ModelRef(name=name, model_type=model_type, uri=path.name, sha256=sha256_file(path), framework=framework,
                   inputs=inputs, outputs=outputs, dataset=dataset, metrics=metrics, metrics_split="test",
                   trust_decision=trust_decision,
                   provenance=Provenance(app=app, library_version=_library_version(),
                                         notes=str(m.validation.get("split", "")) if hasattr(m, "validation") and
                                         isinstance(m.validation, dict) else ""))
    ref.save(str(path) + ".modelref.json")
    return ref


def load_model(ref: ModelRef, root: str | Path = "."):
    """Open the model a ``ModelRef`` points to (a local JSON file, checked against its SHA-256)."""
    path = Path(ref.uri)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        raise FileNotFoundError(f"model file {path} (from ModelRef {ref.name!r}) not found")
    if ref.sha256 and sha256_file(path) != ref.sha256:
        raise ValueError(f"model file {path} does not match the SHA-256 in its ModelRef")
    return model_from_dict(json.loads(path.read_text()))


def check(name: str, passed: bool | None, value=None, threshold=None, evidence: str = "", detail: str = "",
          warn: bool = False) -> CheckRecord:
    """A ``CheckRecord``: ``passed=None`` means it did not run (NOT_RUN)."""
    if passed is None:
        return CheckRecord(name=name, status="NOT_RUN", detail=detail)
    status = "PASS" if passed else ("WARN" if warn else "FAIL")
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return CheckRecord(name=name, status=status, value=value, threshold=threshold, evidence=evidence, detail=detail)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def twin_study(runtime, *, app: str = "twin", model_ref: ModelRef | None = None, title: str = "") -> StudyResult:
    """The twin's run as a ``StudyResult``: estimates table, alerts, validity checks and revalidations."""
    hist = list(runtime.history)
    names = runtime.model.state_names
    step = max(1, len(hist) // 500)
    cols: dict[str, list] = {"t": [h["t"] for h in hist[::step]]}
    for n in names:
        cols[n] = [h["state"][n] for h in hist[::step]]
        cols[f"{n}_sd"] = [h["state_sd"][n] for h in hist[::step]]
    checks = []
    d = runtime.last_drift
    checks.append(check("validity window (assess_drift)", None if d is None else d["decision"] == "ok",
                        value=None if d is None else d["decision"], threshold="ok",
                        evidence=f"drift check at t={d['t']:.0f} s on {d['n']} samples" if d else "",
                        detail="; ".join(d["reasons"]) if d else "no validity check ran yet", warn=True))
    for rv in runtime.revalidations[-1:]:
        for c in rv["checks"]:
            checks.append(check(f"revalidation rmse {c['state']}", c["passed"], value=c["rmse"], threshold=c["limit"],
                                evidence=f"open-loop replay of {rv['n_samples']} samples ending t={rv['t']:.0f} s"))
    nis = [h.get("nis") for h in hist if h.get("nis") is not None]
    if nis:
        frac = float(np.mean(np.array(nis) > runtime.nis_threshold))
        expected = 1 - runtime.rules.nis_probability
        checks.append(check("filter consistency (NIS)", frac <= 10 * expected + 0.01, value=frac,
                            threshold=10 * expected + 0.01, evidence=f"{len(nis)} analysis steps",
                            detail=f"fraction of steps above the {runtime.rules.nis_probability:.3%} chi-square "
                                   "quantile", warn=True))
    alerts = [a.to_dict() for a in runtime.alerts]
    return StudyResult(id=new_id("twin"), study_type="twin", app=app, title=title or f"Twin of {runtime.asset}",
                       models=[model_ref] if model_ref else [], table=Table(columns=cols),
                       metrics={"n_updates": runtime.n_updates, "n_alerts": sum(not a["cleared"] for a in alerts),
                                "status": runtime.status, "assimilation_ms": runtime.timing["assimilation_ms"]},
                       checks=checks,
                       limits=["one asset, one model, one data stream", "alerts are rule-based screening, not a "
                               "safety system", "EnKF assumes Gaussian noise; parameters tracked as random walks"],
                       provenance=Provenance(app=app, library_version=_library_version()),
                       metadata={"alerts": alerts[-200:], "snapshot": runtime.snapshot()})


def anomaly_study(result, *, app: str = "anomaly", benchmark: dict | None = None, title: str = "") -> StudyResult:
    """A detection run (and optionally a benchmark) as a ``StudyResult``."""
    ev = result.events
    cols = {"start": [e.start for e in ev], "end": [e.end for e in ev], "severity": [e.severity for e in ev],
            "methods": [",".join(e.methods) for e in ev], "variables": [",".join(e.variables) for e in ev],
            "explanation": [e.explanations[0] if e.explanations else "" for e in ev]}
    checks = [check(f"detector {c.name} calibrated", bool(np.isfinite(c.spread) and c.spread > 0), value=c.spread,
                    evidence=f"normal window: {result.normal_samples} samples") for c in result.channels]
    metrics: dict[str, object] = {"n_events": len(ev), "flagged_fraction": float(result.flags.mean())}
    if benchmark:
        comb = benchmark["results"].get("combined")
        if comb:
            metrics.update({"benchmark_point_precision": comb["point_precision"],
                            "benchmark_point_recall": comb["point_recall"],
                            "benchmark_event_recall": comb["event_recall"],
                            "benchmark_false_alarms_per_hour": comb["false_alarms_per_hour"]})
            checks.append(check("benchmark on labelled held-out runs", True, value=comb["point_f1"],
                                evidence=f"seeds {benchmark['seeds']}", detail=benchmark["protocol"]))
    return StudyResult(id=new_id("anomaly"), study_type="anomaly", app=app, title=title or "Anomaly detection",
                       table=Table(columns=cols), metrics=metrics, checks=checks,
                       limits=["thresholds calibrated on the declared normal window only",
                               "balances are only as good as the sensors they use"],
                       provenance=Provenance(app=app, library_version=_library_version()),
                       metadata={"events": [e.to_dict() for e in ev[:200]], "config": result.config.to_dict()})


def control_study(comparison: dict, *, app: str = "control", title: str = "") -> StudyResult:
    """A controller comparison as a ``StudyResult`` (one row per controller)."""
    runs = comparison["runs"]
    keys = ["iae_h_m_s", "iae_T_K_s", "max_violation_T_K", "max_violation_h_m", "heater_energy_kWh",
            "heater_travel_kW", "pump_travel_L_s", "mean_solve_ms"]
    cols: dict[str, list] = {"controller": [r["controller"] for r in runs.values()]}
    for k in keys:
        cols[k] = [r["metrics"][k] for r in runs.values()]
    checks = []
    for r in runs.values():
        m = r["metrics"]
        checks.append(check(f"{r['controller']}: output constraints respected",
                            m["max_violation_T_K"] <= 1e-9 and m["max_violation_h_m"] <= 1e-9,
                            value=max(m["max_violation_T_K"], m["max_violation_h_m"]), threshold=0.0,
                            evidence=f"closed-loop simulation, {len(r['trajectory']['t'])} steps"))
        if r.get("failed_solves") is not None and "MPC" in r["controller"]:
            checks.append(check(f"{r['controller']}: optimiser converged every step", r["failed_solves"] == 0,
                                value=r["failed_solves"], threshold=0, evidence="solver status per step",
                                warn=True))
    return StudyResult(id=new_id("control"), study_type="control", app=app, title=title or "Controller comparison",
                       table=Table(columns=cols), metrics={"mode": comparison["mode"]}, checks=checks,
                       limits=[comparison["mode"], "same simulated plant for every controller; a real plant adds "
                               "model mismatch the simulation does not"],
                       provenance=Provenance(app=app, library_version=_library_version()),
                       metadata={"spec": comparison["spec"], "scenario": comparison["scenario"]})
