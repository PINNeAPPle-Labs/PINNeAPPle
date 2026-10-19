"""The surrogate study, the regime arena and the gateway in the shared data contracts (``pinneapple_data.contracts``).

``data_from_spec``     a table + its ``DatasetSpec`` (input / output column roles) -> ``SurrogateData``
``dataset_spec``       ``SurrogateData`` -> ``DatasetSpec`` (what a study trained on)
``model_ref``          ``SurrogateModel`` -> ``ModelRef`` (validity ranges, held-out metrics, trust decision)
``study_result``       ``surrogate.StudyResult`` -> contracts ``StudyResult`` (``study_type="surrogate"``)
``arena_study_result`` regime-arena result -> contracts ``StudyResult`` (``study_type="benchmark"``)

Checks map ``supports`` / ``contradicts`` / ``not_run`` to ``PASS`` / ``FAIL`` / ``NOT_RUN``; a check is only
``PASS`` or ``FAIL`` with its measured value or the record it came from.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .data import SurrogateData, field_columns, from_table
from .model import SurrogateModel

__all__ = ["read_table_with_roles", "data_from_spec", "dataset_spec", "model_ref", "study_result", "arena_study_result", "MODEL_TYPES"]

MODEL_TYPES = {"gp": "gp", "fno": "neural_operator", "deeponet": "neural_operator", "pinn": "pinn",
               "mlp": "surrogate", "ridge": "surrogate", "gbt": "surrogate"}
_FRAMEWORK = {"gp": "scikit-learn", "ridge": "scikit-learn", "gbt": "xgboost / scikit-learn", "mlp": "pytorch",
              "fno": "pytorch", "deeponet": "pytorch", "pinn": "pytorch"}
_STATUS = {"supports": "PASS", "contradicts": "FAIL", "not_run": "NOT_RUN"}


def _c():
    from pinneapple_data import contracts
    return contracts


def read_table_with_roles(path, *, max_rows: int | None = None):
    """``(DataFrame, {column: role} or None)`` from CSV / Parquet / JSON / HDF5 / NPZ; the roles come from a
    ``DatasetSpec`` embedded in the file or in its ``.contract.json`` sidecar. HDF5 files not written by the
    contracts and NPZ files are read by ``surrogate.read_table`` (no roles)."""
    from pathlib import Path

    from .data import read_table
    path = Path(path)
    df, roles = None, None
    if path.suffix.lower() != ".npz":
        try:
            C = _c()
            df, contract = C.read_table(path)
            if isinstance(contract, C.DatasetSpec):
                roles = {c.name: c.role for c in contract.columns}
        except (ValueError, KeyError, OSError, ImportError):
            df = None
    if df is None:
        df = read_table(path)
    if max_rows is not None and len(df) > max_rows:
        raise ValueError(f"{len(df)} rows; the limit is {max_rows}")
    return df, roles


def data_from_spec(df, spec, *, group: str | None = None, grid_shape: Sequence[int] | None = None,
                   meta: Mapping[str, Any] | None = None) -> SurrogateData:
    """Inputs and outputs from the spec's column roles. A single output declared as a field (``spec.metadata
    ["fields"] = {"u": [nx, nt]}``) or given as indexed columns ``u_0, u_1, ...`` becomes a field. ``group``
    defaults to ``spec.metadata["group"]``."""
    md = dict(spec.metadata or {})
    inputs = list(spec.inputs)
    outputs = list(spec.outputs)
    fields = md.get("fields") or {}
    if len(fields) == 1 and not outputs:
        outputs = list(fields)
    names = []
    for o in outputs:                 # u_0, u_1 ... declared as outputs collapse to the field name u
        m = o.rsplit("_", 1)
        names.append(m[0] if len(m) == 2 and m[1].isdigit() and field_columns(df.columns, m[0]) else o)
    outputs = list(dict.fromkeys(names))
    inputs = [i for i in inputs if not any(i in field_columns(df.columns, o) for o in outputs)]
    group = group or md.get("group")
    if group in inputs:
        inputs.remove(group)
    shape = grid_shape or (fields.get(outputs[0]) if len(outputs) == 1 else None)
    return from_table(df, inputs, outputs, group=group, grid_shape=shape,
                      meta={"source": f"DatasetSpec {spec.name} {spec.version}", "dataset_spec": spec.name,
                            "dataset_hash": spec.content_hash(), **md.get("surrogate", {}), **dict(meta or {})})


def dataset_spec(data: SurrogateData, name: str, *, app: str = "", description: str = ""):
    c = _c()
    cols = [c.Column(name=n, role="input") for n in data.input_names]
    if data.kind == "field":
        cols += [c.Column(name=f"{data.output_names[0]}_{i}", role="output") for i in range(int(np.prod(data.grid_shape)))]
        md = {"fields": {data.output_names[0]: list(data.grid_shape)}, "coord_names": data.coord_names}
    else:
        cols += [c.Column(name=n, role="output") for n in data.output_names]
        md = {}
    if data.groups is not None:
        cols.append(c.Column(name=data.meta.get("group", "group"), role="id"))
        md["group"] = data.meta.get("group", "group")
    md["sha256"] = data.sha256()
    return c.DatasetSpec(name=name, description=description or str(data.meta.get("source", "")), columns=cols,
                         n_rows=data.n, provenance=c.Provenance(app=app, source=str(data.meta.get("source", ""))),
                         metadata=md)


def model_ref(model: SurrogateModel, *, name: str, version: str, uri: str, app: str = "",
              dataset_name: str = "") :
    c = _c()
    card = model.card
    b = card.get("validity_bounds") or {}
    inputs = [c.Variable(name=n, low=b.get(n, [None, None])[0], high=b.get(n, [None, None])[1])
              for n in model.input_names]
    outputs = [c.Variable(name=n) for n in model.output_names]
    m = card.get("metrics") or {}
    metric = card.get("metric", "error")
    metrics = {}
    for split in ("test", "shift"):
        v = (m.get(split) or {}).get("error")
        if v is not None and np.isfinite(v):
            metrics[f"{split}_{metric}"] = float(v)
    if m.get("infer_s_per_query") is not None:
        metrics["infer_s_per_query"] = float(m["infer_s_per_query"])
    trust = (card.get("trust") or {}).get("decision") or "UNCHECKED"
    ds = card.get("dataset") or {}
    method = card.get("method", "")
    return c.ModelRef(
        name=name, version=version, model_type=MODEL_TYPES.get(method, "surrogate"), uri=uri,
        sha256=card.get("model_sha256"), framework=_FRAMEWORK.get(method, ""), inputs=inputs, outputs=outputs,
        dataset=c.DatasetRef(name=dataset_name or str(ds.get("source") or ""), content_hash=card.get("dataset_sha256") or ""),
        metrics=metrics, metrics_split="test", trust_decision=trust,
        provenance=c.Provenance(app=app, source=card.get("title", "")),
        metadata={"method": method, "kind": card.get("kind"), "grid_shape": card.get("grid_shape"),
                  "metric": metric, "ood_gate": card.get("ood_gate", {}).get("rule"),
                  "metrics_note": "test: random held-out split; shift: distribution-shift split never seen in "
                                  "training"})


def _checks(trust: Mapping[str, Any]):
    c = _c()
    out = []
    for ch in trust.get("checks", []):
        st = _STATUS.get(ch.get("status"), "NOT_RUN")
        val = ch.get("value")
        ev = ch.get("source") or ""
        if st in ("PASS", "FAIL") and val is None and not ev:
            ev = "trust report of this study"
        out.append(c.CheckRecord(name=ch["name"], status=st, value=None if val is None else float(val),
                                 threshold=ch.get("limit"), evidence=ev, detail=ch.get("reason", "")))
    return out


def _limits_text(lim: Mapping[str, Any], splits: Mapping[str, Any]) -> list[str]:
    out = []
    b = lim.get("validity_bounds") or {}
    if b:
        out.append("Valid inside the training ranges: " + ", ".join(f"{k} {v[0]:.4g} .. {v[1]:.4g}" for k, v in b.items()))
    if lim.get("ood_gate"):
        out.append("OOD gate: " + lim["ood_gate"]["rule"])
    if lim.get("concentration"):
        out.append(lim["concentration"][0].upper() + lim["concentration"][1:])
    if lim.get("extrapolation_ratio") is not None:
        out.append(f"Error on the distribution-shift split ({splits.get('shift_description', '')}) is "
                   f"{lim['extrapolation_ratio']:.2f} x the held-out error")
    return out


def study_result(study, *, id: str, app: str = "surrogate", ref=None, title: str = ""):
    """``study``: a ``surrogate.StudyResult``; ``ref``: the ``ModelRef`` of its model, when published."""
    c = _c()
    rows = []
    for m in study.methods:
        s = m.get("summary") or {}
        rows.append({"method": m["id"], "applicable": m["applicable"], "reason": m["reason"],
                     "val_error": s.get("val_error_mean"), "test_error": s.get("test_error_mean"),
                     "test_error_std": s.get("test_error_std"), "shift_error": s.get("shift_error_mean"),
                     "train_s": s.get("train_s"), "infer_s_per_query": s.get("infer_s_per_query"),
                     "n_params": s.get("n_params")})
    import pandas as pd
    table = c.Table.from_frame(pd.DataFrame(rows))
    best = next((m for m in study.methods if m["id"] == study.best), None)
    metrics: dict[str, Any] = {"best": study.best, "metric": study.metric}
    if best:
        metrics.update({f"best_{k}": v for k, v in (best.get("summary") or {}).items()
                        if k in ("test_error_mean", "shift_error_mean", "train_s", "infer_s_per_query")})
    return c.StudyResult(id=id, study_type="surrogate", app=app, title=title or f"Surrogate study: {study.best}",
                         dataset=dataset_spec(study.data, study.data.meta.get("dataset_spec", "dataset"), app=app),
                         models=[ref] if ref is not None else [], table=table, metrics=metrics,
                         checks=_checks(study.trust), limits=_limits_text(study.limitations, study.splits.to_dict()),
                         provenance=c.Provenance(app=app, source="pinneapple_analysis.surrogate.run_study",
                                                 created=study.created),
                         metadata={"budget": study.budget.to_dict(), "seeds": study.seeds,
                                   "splits": study.splits.to_dict(), "hardware": study.hardware})


def arena_study_result(result: Mapping[str, Any], *, id: str, app: str = "arena", title: str = ""):
    """``result``: ``ArenaResult.to_dict()``. The table has one row per method and regime."""
    c = _c()
    import pandas as pd
    rows = []
    for reg, ranks in result["rankings"].items():
        for r in ranks:
            rows.append({"regime": reg, "rank": r["rank"], "method": r["method"], "error_mean": r["error_mean"],
                         "error_std": r["error_std"], "n_seeds": r["n_seeds"]})
    checks = []
    for ch in (result.get("audit") or {}).get("checks", []):
        st = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(ch["status"], "N/A")
        checks.append(c.CheckRecord(name=f"dataset audit: {ch['id']}", status=st, evidence=ch["summary"]))
    limits = [f"{g['title']}: not run ({g['reason']})" for g in result["regimes"] if not g["available"]]
    limits += [f"{k}: {v}" for k, v in (result.get("not_run") or {}).items()]
    winner = result["overall"][0]["method"] if result.get("overall") else None
    return c.StudyResult(id=id, study_type="benchmark", app=app, title=title or "Regime arena",
                         table=c.Table.from_frame(pd.DataFrame(rows)) if rows else None,
                         metrics={"overall_winner": winner, "metric": result["protocol"]["metric"],
                                  "wall_s": result.get("wall_s")},
                         checks=checks, limits=limits,
                         provenance=c.Provenance(app=app, source="pinneapple_arena.regimes.run_regime_arena",
                                                 created=result.get("created", "")),
                         metadata={"protocol": result["protocol"], "hardware": result.get("hardware")})
