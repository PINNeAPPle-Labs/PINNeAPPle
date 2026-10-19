"""ROM study results as the apps' open contracts (``pinneapple_data.contracts``): a ``ModelRef`` per exported
reduced model and a ``StudyResult`` for the study, so a ROM built in the ROM Lab is an input of the uncertainty,
optimisation, twin or gateway apps.

Every metric is a held-out one (the split is named); a check is reported only with the value it was decided on.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

__all__ = ["model_ref", "study_result", "rom_checks"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def model_ref(rm, *, name: str, uri: str, payload: bytes | None = None, row: dict[str, Any] | None = None,
              split: str = "test", dataset_hash: str = "", dataset_name: str = "", app: str = "rom",
              study_id: str = "", units: dict[str, str] | None = None, framework: str = "pinneapple-rom/1"):
    """``ModelRef`` of a ``ReducedModel``: inputs with their training range, outputs (the fields), the held-out
    metrics of its study row and the SHA-256 of the file at ``uri`` (``payload``: its bytes)."""
    from pinneapple_data.contracts import DatasetRef, ModelRef, Provenance, Variable
    units = units or {}
    v = rm.config["validity"]
    if rm.mode == "parametric":
        ins = [Variable(name=n, unit=units.get(n, ""), low=float(lo), high=float(hi))
               for n, lo, hi in zip(rm.config["param_names"], v["lo"], v["hi"], strict=True)]
    else:
        ins = [Variable(name="t", unit=units.get("t", ""), low=float(v["t_min"]), high=None,
                        description=f"time; trained on [{v['t_min']:g}, {v['t_max']:g}], later times are forecasts")]
    outs = [Variable(name=f["name"], unit=units.get(f["name"], ""),
                     description=f"{f['size'] // max(f['components'], 1)} {f['where']}s x {f['components']} "
                                 f"component(s)") for f in rm.config["fields"]]
    metrics: dict[str, float] = {}
    if row:
        for k_src, k in (("error_mean", "test_rel_l2_mean"), ("error_max", "test_rel_l2_max"),
                         ("error_median", "test_rel_l2_median"), ("infer_s", "inference_s_per_snapshot"),
                         ("fit_s", "fit_s")):
            if row.get(k_src) is not None and np.isfinite(row[k_src]):
                metrics[k] = float(row[k_src])
        for f, e in (row.get("field_errors") or {}).items():
            if np.isfinite(e):
                metrics[f"test_rel_l2_{f}"] = float(e)
        if row.get("speedup"):
            metrics["speedup_vs_full_order"] = float(row["speedup"])
    return ModelRef(name=name, version="0.1.0", model_type="rom", uri=uri,
                    sha256=_sha(payload) if payload is not None else None, framework=framework,
                    inputs=ins, outputs=outs, metrics=metrics, metrics_split=split,
                    dataset=DatasetRef(name=dataset_name or name, content_hash=dataset_hash) if dataset_hash else None,
                    trust_decision="UNCHECKED",
                    provenance=Provenance(app=app, study_id=study_id, source=f"rom_study: {rm.method}"
                                          + ("" if rm.rank is None else f", rank {rm.rank}"),
                                          library_version=_version()))


def rom_checks(res, tolerance: float | None) -> list:
    """Checks of a study, each with its value: the best ROM against its baseline (nearest snapshot or
    persistence), against the tolerance, and divergence of the time steppers."""
    from pinneapple_data.contracts import CheckRecord
    out = []
    ok = [r for r in res.rows if np.isfinite(r["error_mean"])]
    best = min(ok, key=lambda r: r["error_mean"]) if ok else None
    base = [r for r in ok if r["method"] in ("nearest", "persistence")]
    if best is not None and base and best["method"] not in ("nearest", "persistence"):
        ratio = base[0]["error_mean"] / max(best["error_mean"], 1e-300)
        what = "nearest training snapshot" if base[0]["method"] == "nearest" else "persistence (latest snapshot)"
        out.append(CheckRecord(name=f"best ROM beats the {what}", status="PASS" if ratio > 1 else "FAIL",
                               value=float(ratio), threshold=1.0,
                               detail=f"mean held-out error of the {what} / best ROM"))
    if best is not None and tolerance is not None:
        out.append(CheckRecord(name="held-out error within tolerance",
                               status="PASS" if best["error_mean"] <= tolerance else "FAIL",
                               value=float(best["error_mean"]), threshold=float(tolerance), detail=res.metric))
    div = [f"{r['method']} r={r['rank']}" for r in res.rows if r.get("diverged")]
    if res.rows:
        out.append(CheckRecord(name="no model diverged on the test snapshots", status="WARN" if div else "PASS",
                               value=len(div), threshold=0, detail=", ".join(div)))
    return out


def study_result(res, *, study_id: str, title: str, tolerance: float | None, models=(), artifacts=(),
                 limits=(), app: str = "rom", dataset=None):
    """``StudyResult`` of a ROM study: one table row per (method, rank), the best model, checks with values."""
    from pinneapple_data.contracts import Provenance, StudyResult, Table
    cols: dict[str, list] = {k: [] for k in ("method", "rank", "test_error_mean", "test_error_max", "fit_s",
                                              "inference_s", "speedup")}
    for r in res.rows:
        cols["method"].append(r["method"])
        cols["rank"].append(r["rank"])
        cols["test_error_mean"].append(r["error_mean"] if np.isfinite(r["error_mean"]) else None)
        cols["test_error_max"].append(r["error_max"] if np.isfinite(r["error_max"]) else None)
        cols["fit_s"].append(r["fit_s"])
        cols["inference_s"].append(r["infer_s"])
        cols["speedup"].append(r["speedup"])
    b = res.best(tolerance)
    metrics: dict[str, Any] = {"n_train": int(len(res.split.train)), "n_test": int(len(res.split.test)),
                               "split": res.split.how, "metric": res.metric, "budget_s": res.budget_s}
    if b:
        metrics.update(best_method=b["method"], best_rank=b["rank"], best_test_error=float(b["error_mean"]),
                       best_inference_s=float(b["infer_s"]))
        if b.get("speedup"):
            metrics["best_speedup"] = float(b["speedup"])
    return StudyResult(id=study_id, study_type="rom", app=app, title=title, status="done", dataset=dataset,
                       models=list(models), table=Table(columns=cols), metrics=metrics,
                       checks=rom_checks(res, tolerance), artifacts=list(artifacts), limits=list(limits),
                       provenance=Provenance(app=app, study_id=study_id, source="rom_study",
                                             library_version=_version()))


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("pinneapple")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        return ""
