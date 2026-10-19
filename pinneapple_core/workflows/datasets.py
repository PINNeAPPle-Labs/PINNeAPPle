"""From a case table to a workflow, and from finished runs to a training dataset (``DatasetSpec`` + table).

    cases, skeleton = read_cases("campaign_cases.parquet")          # a Campaign Planner export (or any table)
    wf = workflow_from_cases("sweep", cases.to_dict("records"), template, id_column="case_id")
    summary = WorkflowRunner(root).run(wf)
    spec, path = dataset_from_run(runner, summary["id"], cases, output_step="solve", path="dataset.parquet",
                                  skeleton=skeleton)

Rows are the cases; columns are the case parameters, then the numeric outputs of ``output_step`` (nested dicts are
flattened as ``a.b``), then ``status`` and ``run_step``. Cases whose step did not finish keep empty outputs, and the
spec is marked ``partial``.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

ID_COLUMNS = ("case_id", "case", "id", "run_id")


def read_cases(path: str | Path):
    """A case table (CSV / Parquet / HDF5 / JSON) and the contract attached to it, if any (a ParameterSpace or a
    DatasetSpec skeleton from the Campaign Planner)."""
    from pinneapple_data.contracts import read_table
    df, contract = read_table(path)
    return df, contract


def id_column_of(df, contract=None) -> str | None:
    if contract is not None and getattr(contract, "kind", "") == "DatasetSpec":
        ids = contract.names("id")
        if ids and ids[0] in df.columns:
            return ids[0]
    for c in ID_COLUMNS:
        if c in df.columns:
            return c
    return None


def dataset_from_run(runner, run_id: str, cases, *, output_step: str, path: str | Path,
                     id_column: str | None = None, outputs: Sequence[str] | None = None, skeleton=None,
                     name: str | None = None):
    """Write the dataset table to ``path`` (with its sidecar) and return ``(DatasetSpec, path)``."""
    import pandas as pd

    from pinneapple_data.contracts import (
        Column,
        DatasetSpec,
        FileRef,
        Provenance,
        write_sidecar,
        write_table,
    )
    df = cases if isinstance(cases, pd.DataFrame) else pd.DataFrame(list(cases))
    idc = id_column or id_column_of(df, skeleton)
    steps = {s["step_id"]: s for s in runner.steps(run_id)}
    rows, run_steps = [], []
    for i, case in enumerate(df.to_dict("records")):
        cid = str(case[idc]) if idc else str(case.get("case_id", f"case{i:04d}"))
        cid_s = "".join(c if c.isalnum() or c in "_.-" else "_" for c in cid)
        sid = f"{cid_s}.{output_step}"
        st = steps.get(sid, {})
        flat = _flatten(st.get("outputs") or {}) if st.get("status") in ("done", "skipped") else {}
        if outputs is not None:
            flat = {k: flat.get(k) for k in outputs}
        rows.append({**case, **{k: v for k, v in flat.items() if _scalar(v)}, "status": st.get("status", "missing"),
                     "run_step": f"{run_id}/{sid}"})
        run_steps.append(f"{run_id}/{sid}")
    out = pd.DataFrame(rows)
    param_cols = list(df.columns)
    out_cols = [c for c in out.columns if c not in param_cols and c not in ("status", "run_step")]
    units: dict[str, str] = {}
    space = None
    if skeleton is not None:
        if getattr(skeleton, "kind", "") == "DatasetSpec":
            units = {c.name: c.unit for c in skeleton.columns if c.unit}
            space = skeleton.parameter_space
        elif getattr(skeleton, "kind", "") == "ParameterSpace":
            space = skeleton
            units = {p.name: p.unit for p in skeleton.parameters if p.unit}
    role = {c: "input" for c in param_cols}
    if idc:
        role[idc] = "id"
    role.update({c: "output" for c in out_cols})
    role.update({"status": "meta", "run_step": "meta"})
    n_ok = int(out["status"].isin(["done", "skipped"]).sum())
    path = Path(path)
    cols = [Column(name=str(c), role=role.get(c, "meta"), unit=units.get(c, ""), dtype=str(out[c].dtype))
            for c in out.columns]
    spec = DatasetSpec(name=name or (skeleton.name if getattr(skeleton, "kind", "") == "DatasetSpec" else run_id),
                       columns=cols, n_rows=len(out), parameter_space=space,
                       status="complete" if n_ok == len(out) else "partial",
                       provenance=Provenance(app="runner", study_id=run_id, source=f"workflow run {run_id}",
                                             run_ids=run_steps))
    write_table(out, path)                              # the table first, so its SHA-256 is final
    spec.files = [FileRef.of(path, root=path.parent, rows=len(out))]
    write_sidecar(path, spec)
    return spec, path


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _scalar(v) -> bool:
    return isinstance(v, (int, float, str, bool)) or v is None
