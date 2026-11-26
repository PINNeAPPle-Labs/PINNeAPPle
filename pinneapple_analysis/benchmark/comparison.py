"""The standard field comparison of the benchmark report (#63): one figure per case that has a planar field --
reference, surrogate and |surrogate - reference| on one shared colour scale, with the L2 error in the number.

The drawing is ``pp.viz.compare_solutions`` and the number is ``pp.viz.field_error_metrics``, the same functions the
rest of the house uses, so a benchmark figure and a benchmark table can never quote two different errors.

Where the fields come from: a trainer that has just evaluated its final model against the reference writes the two
arrays with :func:`save_comparison` into ``runs/<job id>.npz``, next to that job's result JSON -- the ndarray sidecar
the problems' own campaigns already write (``prepared/<problem>/runs/<id>.npz``). The report reads the full-budget,
seed-0 run of the case's best field method on its main metric, draws it into ``figures/comparison_<problem>.png`` and
quotes the error. A case whose field is not planar (a 1-D profile, a 3-D nodal cloud) declares it in
``Problem.comparison_absent``, and the report prints that reason instead of inventing a figure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FIGURE_TEMPLATE", "grid_coordinates", "save_comparison", "read_comparison", "select", "figures",
           "markdown"]

#: one figure per case, next to the report's other figures
FIGURE_TEMPLATE = "comparison_{problem}.png"

#: the fields a stored comparison carries
KEYS = ("x", "y", "reference", "prediction")


def grid_coordinates(shape) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` of a uniform ``(nx, ny)`` grid, flattened in the order the field itself flattens in: ``x`` runs
    slow and ``y`` fast, which is what ``field.ravel()`` of an ``(nx, ny)`` array gives and what the contour plot
    reads back into a structured panel."""
    nx, ny = int(shape[0]), int(shape[1])
    gx, gy = (np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    return X.ravel(), Y.ravel()


def save_comparison(path, reference, prediction, *, x, y, quantity: str, unit: str = "", instance: str = "",
                    source: str = "") -> Path | None:
    """Store one instance's reference and prediction, and the coordinates of both, as ``runs/<job id>.npz``.

    ``x``, ``y``, ``reference`` and ``prediction`` are the same instance: one value per point, in one traversal order
    (``x`` and ``y`` may be 2-D as long as the field is the same shape). ``quantity``/``unit`` are the colour bar's
    labels. ``path=None`` stores nothing, so a caller that has no sidecar path needs no branch."""
    if path is None:
        return None
    ref, pred = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if ref.shape != pred.shape:
        raise ValueError(f"reference {ref.shape} and prediction {pred.shape} are not the same field")
    X, Y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if not (len(X) == len(Y) == ref.size):
        raise ValueError(f"coordinates ({len(X)}, {len(Y)}) and field ({ref.size} points) do not match")
    path = Path(path)
    np.savez_compressed(path, x=X, y=Y, reference=ref.ravel().astype(np.float32),
                        prediction=pred.ravel().astype(np.float32), quantity=np.array(str(quantity)),
                        unit=np.array(str(unit or "")), instance=np.array(str(instance)),
                        source=np.array(str(source)))
    return path


def read_comparison(path) -> dict[str, Any]:
    """The stored arrays and labels of one comparison sidecar."""
    with np.load(Path(path)) as z:
        d: dict[str, Any] = {k: z[k].astype(float) for k in KEYS}
        for k in ("quantity", "unit", "instance", "source"):
            d[k] = str(z[k]) if k in z.files else ""
    return d


def _best_field_method(plan, tables, pid) -> tuple[str | None, str]:
    """The case's field method with the lowest error on the metric that judges it, and, when there is none, why.

    Integrals-only methods are left out (they have no field to draw) and so are rows that failed a gate: an INVALID
    model's picture is not evidence of anything. The metric is the problem's own -- ``main_metric_for``, so a field
    model is ranked on its field and not on an integral it does not predict -- and, when the case has no cell for it
    (a problem whose PINN instances are empty, say), the first field column the case does have on every candidate."""
    prob, t = plan.P[pid], tables.get(pid) or {}
    rows = t.get("rows") or {}
    cand = {}
    for mid, r in rows.items():
        meth = plan.M.get(mid)
        if meth is None or prob.method_outputs(meth) == "integrals":
            continue
        if str(r.get("status", "")).startswith("INVALID"):
            continue
        cand[mid] = r
    if not cand:
        return None, "every method that predicts a field failed a gate (INVALID): there is no field to trust"
    cols = prob.columns()

    def own_key(mid):
        m = prob.main_metric_for(mid) if hasattr(prob, "main_metric_for") else getattr(prob, "main_metric", None)
        m = m() if callable(m) else m
        return next((c.key for c in cols if m and (c.set, c.metric) == tuple(m)), None)

    def rank(pairs):
        scored = [(float(c["mean"]), method) for method, c in pairs
                  if c and np.isfinite(c.get("mean", float("nan")))]
        scored.sort()
        return scored[0][1] if scored else None

    best = rank([(m, (r.get("cells") or {}).get(own_key(m)) or {}) for m, r in cand.items()])
    if best is None:                      # the case has no cell for the metric that judges it: its first field column
        for col in (c for c in cols if c.kind == "field"):
            best = rank([(m, (r.get("cells") or {}).get(col.key) or {}) for m, r in cand.items()])
            if best:
                return best, ""
    if best is None:
        return None, "no method of this case has a field column to rank on: there is nothing to say which is drawn"
    return best, ""


def _npz_path(plan, pid: str, mid: str) -> Path:
    """The sidecar of the run the figure reads: full budget, seed 0, the method's first "final" instance."""
    prob, meth = plan.P[pid], plan.M[mid]
    tag = f"final__{pid}__{mid}__L1.000__s0"
    if meth.per_instance:
        inst = (prob.instances().get("final") or [("", None)])[0][0]
        tag += f"__{inst}" if inst else ""
    return plan.ws.runs / f"{tag}.npz"


def select(plan, tables) -> dict[str, dict]:
    """{problem: entry} for the report: the figure's path and the case's numbers, or the reason there is no figure.

    An entry is ``{"problem", "title", "method", "method_title", "figure", "npz", "instance", "metrics",
    "reason"}`` -- ``figure`` and ``metrics`` are None exactly when ``reason`` is not empty."""
    out = {}
    for pid, prob in plan.P.items():
        entry = {"problem": pid, "title": prob.title, "method": None, "method_title": "", "figure": None,
                 "npz": None, "instance": "", "metrics": None, "reason": ""}
        declared = getattr(prob, "comparison_absent", "")     # the case says it has no planar field: that is the row
        if declared:
            entry["reason"] = declared
            out[pid] = entry
            continue
        mid, why = _best_field_method(plan, tables, pid)
        if mid is None:
            entry["reason"] = why
            out[pid] = entry
            continue
        entry["method"], entry["method_title"] = mid, prob.method_title(plan.M[mid])
        path = _npz_path(plan, pid, mid)
        if not path.exists():
            entry["reason"] = ("no comparison field was stored with this run: it appears once the case is re-run "
                               f"at full budget ({path.name})")
            out[pid] = entry
            continue
        try:
            from pinneapple_tools.visualization.fields import field_error_metrics
            d = read_comparison(path)
            metrics = field_error_metrics(d["reference"], d["prediction"])
        except Exception as e:                                  # a stored field that cannot be read is reported
            entry["reason"] = f"the stored comparison field could not be read ({e})"
            out[pid] = entry
            continue
        entry.update(figure=FIGURE_TEMPLATE.format(problem=pid), npz=path.name, instance=d["instance"],
                     metrics=metrics)
        out[pid] = entry
    return out


def figures(plan, entries: dict[str, dict], fig_dir) -> list[str]:
    """Draw every entry that has a stored field into ``fig_dir``; returns the files written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_tools.visualization.fields import compare_solutions

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for entry in entries.values():
        if not entry.get("npz") or not entry.get("metrics"):
            continue
        d = read_comparison(plan.ws.runs / entry["npz"])
        quantity, unit = d["quantity"] or "field", d["unit"]
        c = compare_solutions(d["x"], d["y"], d["reference"], d["prediction"], quantity=quantity, unit=unit,
                              labels=("Reference", entry["method_title"]), coord_unit="",
                              suptitle=f"{entry['title']}: {entry['method_title']} against the reference",
                              source=f"{entry['npz']}" + (f", instance {entry['instance']}"
                                                           if entry["instance"] else ""))
        c.figure.savefig(fig_dir / entry["figure"], dpi=150)
        plt.close(c.figure)
        files.append(entry["figure"])
    return files


def markdown(entries: dict[str, dict]) -> list[str]:
    """The report section: one row per case -- the figure and its error, or the case's declared reason."""
    if not entries:
        return []
    drawn = [e for e in entries.values() if e.get("metrics")]
    L = [f"## Field comparison ({len(drawn)} of {len(entries)} case(s))", "",
         "One instance of every case, reference against the surrogate that scores best on it, and the absolute "
         "difference -- the two solutions on one shared colour scale, drawn with "
         "`pp.viz.compare_solutions`. The error is `pp.viz.field_error_metrics`' L2 over the reference's RMS: the "
         "number the report quotes elsewhere, not a separate estimate.", "",
         "| Case | Surrogate | Instance | Figure | Rel. L2 | L2 | Max | n |",
         "|---|---|---|---|---|---|---|---|"]
    for e in entries.values():
        if not e.get("metrics"):
            continue
        m = e["metrics"]
        L.append(f"| {e['title']} | {e['method_title']} | {e['instance'] or '-'} | `figures/{e['figure']}` | "
                 f"{100 * m['l2_relative']:.2f} % | {m['l2']:.4g} | {m['linf']:.4g} | {m['n']} |")
    missing = [e for e in entries.values() if not e.get("metrics")]
    if missing:
        L += ["", "Cases with no comparison figure, and why:", "",
              "| Case | Why there is no standard field figure |", "|---|---|"]
        L += [f"| {e['title']} | {e['reason'].replace('|', '/')} |" for e in missing]
    return L + [""]


def as_records(entries: dict[str, dict]) -> dict[str, dict]:
    """The entries as JSON-ready records for ``summary.json``."""
    out = {}
    for pid, e in entries.items():
        out[pid] = {**e, "metrics": (None if e["metrics"] is None else {k: float(v) for k, v in e["metrics"].items()})}
    return json.loads(json.dumps(out, default=str))
