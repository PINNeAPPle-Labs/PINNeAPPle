"""Regime arena: which method works best for a physical problem, per data regime, under one protocol.

``run_regime_arena(data, ...)`` trains every candidate of ``pinneapple_analysis.surrogate.methods`` that applies to
the data, with

* **equal data**: one split (fixed seed) shared by every method and every model seed;
* **equal compute**: the same ``ComputeBudget`` for every iterative method (``train_within_budget``), one-shot fits
  report the time they used;
* **the same stopping rule**: budget spent, best of ten validation checkpoints (iterative methods);
* **several seeds**: the model seed varies, the data does not; every number has its spread across seeds;

and measures, on data no method trained on, per regime:

==================  =============================================================================================
interpolation       random held-out samples inside the training distribution
few_data            the same held-out samples, training on ``few_data_fraction`` of the training samples
extrapolation       the distribution-shift set: one end of an input, or the edge of the input space
geometry_change     whole groups (geometries, cases) held out, when the data has groups
steep_fronts        fields: the held-out samples with the steepest gradients (top quartile of max |grad u|): a
                    proxy for discontinuities
==================  =============================================================================================

plus training time, inference time per query, model size, and physics violations: the share of predictions that
break the dataset's declared constraints (``pinneapple_data.audit.constraint_mask``) and, for fields with a declared
PDE, the finite-difference residual of the predictions relative to that of the reference data. The data splits are
audited independently by ``pinneapple_data.audit.audit_dataset`` (the PC451-PC453 checks of ``pinneapple check``:
finite values, duplicates, train/test leakage, declared constraints). Numerical-solver baselines (a callable
``solve(X, coords)``) are timed and scored on the same held-out samples.

Nothing is estimated: a regime that cannot be built from the data is reported as unavailable with the reason.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget, hardware_record, time_per_query
from pinneapple_analysis.surrogate.data import SurrogateData
from pinneapple_analysis.surrogate.methods import (
    CANDIDATES,
    TrainContext,
    applicable,
    fit_candidate,
)
from pinneapple_analysis.surrogate.pde import parse_pde
from pinneapple_analysis.surrogate.splits import make_splits
from pinneapple_analysis.surrogate.study import _sample_error

__all__ = ["REGIMES", "ArenaResult", "run_regime_arena", "arena_markdown", "NOT_IN_LIBRARY"]

REGIMES = {
    "interpolation": "random held-out samples inside the training distribution",
    "few_data": "the same held-out samples, trained on a fraction of the training data",
    "extrapolation": "held-out samples beyond the training range (one end of an input, or the edge of the space)",
    "geometry_change": "whole geometries / cases never seen in training",
    "steep_fronts": "held-out fields with the steepest gradients (top quartile of max |grad u|)",
}

#: Methods named in the arena's scope that the regime arena does not run, and why.
NOT_IN_LIBRARY = {
    "gnn": "graph networks (MeshGraphNet) need mesh connectivity; parametric tables and regular grids have none. "
           "pp.benchmark runs MeshGraphNet on the airfoil mesh fields.",
    "r_oams": "R-OAMS is not in the PINNeAPPle library.",
}


@dataclass
class ArenaResult:
    problem: dict[str, Any]
    protocol: dict[str, Any]
    regimes: list[dict[str, Any]]
    methods: list[dict[str, Any]]
    rankings: dict[str, list[dict[str, Any]]]
    overall: list[dict[str, Any]]
    baselines: list[dict[str, Any]]
    audit: dict[str, Any]
    created: str
    hardware: dict[str, Any]
    wall_s: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"problem": self.problem, "protocol": self.protocol, "regimes": self.regimes, "methods": self.methods,
                "rankings": self.rankings, "overall": self.overall, "baselines": self.baselines, "audit": self.audit,
                "created": self.created, "hardware": self.hardware, "wall_s": self.wall_s, "notes": self.notes,
                "not_run": NOT_IN_LIBRARY}

    def to_markdown(self, title: str = "") -> str:
        return arena_markdown(self.to_dict(), title)


def arena_markdown(r: dict[str, Any], title: str = "") -> str:
    """A comparative report from ``ArenaResult.to_dict()`` (also from a stored result file)."""
    f = lambda v: "—" if v is None else f"{v:.4g}"  # noqa: E731
    p, pr, hw = r["problem"], r["protocol"], r.get("hardware", {})
    budget = ", ".join(f"{v:g} {k}" for k, v in pr["budget"].items() if v is not None)
    out = [f"# {title or 'Regime arena'}", "",
           f"Run {r['created']} on {hw.get('cpu', '?')} ({hw.get('logical_cpus', '?')} logical CPUs, torch "
           f"{hw.get('torch', '?')}), {r.get('wall_s', 0):.0f} s in total. Dataset sha256 `{pr['dataset_sha256'][:16]}`, "
           f"{p['n_samples']} samples ({p['kind']}), inputs {', '.join(p['inputs'])}, outputs {', '.join(p['outputs'])}.",
           "", "## Protocol", "", f"- Metric: {pr['metric']} — {pr['metric_definition']}.",
           f"- Budget per training run: {budget}; seeds {pr['seeds']}; split seed {pr['split_seed']}.",
           f"- {pr['equal_data']}.", f"- Stopping: {pr['stopping']}.", f"- Hyper-parameters: {pr['hyperparameters']}.",
           *[f"- {n}" for n in r.get("notes", [])], "", "## Regimes", ""]
    for g in r["regimes"]:
        out.append(f"- **{g['title']}**: " + (f"{g['description']} ({g['n_train']} training, {g['n_eval']} evaluated)"
                                              if g["available"] else f"not run — {g['reason']}"))
    out += ["", "## Rankings (mean ± std over seeds; lower is better)", ""]
    regs = list(r["rankings"])
    out += ["| method | " + " | ".join(x.replace("_", " ") for x in regs) + " | mean rank |",
            "|---" * (len(regs) + 2) + "|"]
    mr = {o["method"]: o["mean_rank"] for o in r["overall"]}
    for m in sorted([m for m in r["methods"] if m["applicable"]], key=lambda m: mr.get(m["id"], 1e9)):
        cells = []
        for g in regs:
            x = next((y for y in r["rankings"][g] if y["method"] == m["id"]), None)
            cells.append("—" if x is None else f"{f(x['error_mean'])} ± {f(x['error_std'])} (#{x['rank']})")
        out.append(f"| {m['title']} | " + " | ".join(cells) + f" | {f(mr.get(m['id']))} |")
    out += ["", "## Cost and physics", "", "| method | train s | inference s/query | model bytes | parameters | "
            "constraint violations | PDE residual / data |", "|---|---|---|---|---|---|---|"]
    for m in r["methods"]:
        if not m["applicable"]:
            continue
        c = m.get("cost", {})
        v = "; ".join(f"{k}: {100 * x:.1f} %" for k, x in m.get("violations", {}).items()) or "—"
        out.append(f"| {m['title']} | {f(c.get('train_s'))} | {f(c.get('infer_s_per_query'))} | "
                   f"{f(c.get('size_bytes'))} | {f(c.get('n_params'))} | {v} | {f(m.get('pde_residual_ratio'))} |")
    for b in r.get("baselines", []):
        out.append(f"| baseline: {b['id']} ({b.get('description', '')}) | — | {f(b.get('s_per_query'))} | — | — | "
                   + ("; ".join(f"{k}: {100 * x:.1f} %" for k, x in b.get("violations", {}).items()) or "—")
                   + f" | {f(b.get('pde_residual_ratio'))} |")
        out.append(f"|  error of the baseline (interpolation): {f(b.get('error_mean'))} | | | | | | |")
    out += ["", "## Not run", ""]
    out += [f"- {m['title']}: {m['reason']}" for m in r["methods"] if not m["applicable"]]
    out += [f"- {k}: {v}" for k, v in r.get("not_run", {}).items()]
    a = r.get("audit", {})
    out += ["", f"## Dataset audit: {a.get('decision', '—')}", "", f"{a.get('tool', '')}", ""]
    out += [f"- {c['status']}: {c['id']} — {c['summary']}" for c in a.get("checks", [])]
    return "\n".join(out) + "\n"


def _steepness(data: SurrogateData, idx: np.ndarray) -> np.ndarray:
    Y = data.Y[idx]
    g = np.zeros(len(idx))
    for ax in range(1, Y.ndim):
        g = np.maximum(g, np.abs(np.diff(Y, axis=ax)).reshape(len(idx), -1).max(1))
    return g


def _audit(data: SurrogateData, tr: np.ndarray, te: np.ndarray, constraints: Sequence[str]) -> dict[str, Any]:
    from pinneapple_data.audit import audit_dataset

    def cols(idx):
        d = {n: data.X[idx, j] for j, n in enumerate(data.input_names)}
        if data.kind == "field":
            d[data.output_names[0]] = data.Y[idx]
        else:
            d.update({n: data.Y[idx, j] for j, n in enumerate(data.output_names)})
        return d
    try:
        card = audit_dataset({"train": cols(tr), "test": cols(te)}, name=data.meta.get("name", "dataset"),
                             inputs=data.input_names, constraints=list(constraints))
    except Exception as e:  # reported, never hidden
        return {"decision": "not run", "reason": f"{type(e).__name__}: {e}", "checks": []}
    return {"decision": card.decision, "tool": "pinneapple_data.audit.audit_dataset (pinneapple check PC451-PC453)",
            "checks": [{"id": c.id, "status": c.status, "summary": c.summary} for c in card.checks]}


def _violations(data: SurrogateData, pred: np.ndarray, X: np.ndarray, constraints: Sequence[str]) -> dict[str, float]:
    from pinneapple_data.audit import constraint_mask
    out = {}
    for expr in constraints:
        if data.kind == "field":
            cols = {data.output_names[0]: pred.reshape(len(pred), -1)}
        else:
            cols = {n: pred[:, j] for j, n in enumerate(data.output_names)}
        cols.update({n: (X[:, j:j + 1] if data.kind == "field" else X[:, j]) for j, n in enumerate(data.input_names)})
        try:
            ok = np.asarray(constraint_mask(expr, cols), bool).reshape(len(pred), -1).all(1)
        except Exception:
            continue
        out[expr] = float(1.0 - ok.mean())
    return out


def run_regime_arena(data: SurrogateData, *, methods: Sequence[str] | str = "auto",
                     budget: ComputeBudget | Mapping | float = 10.0, seeds: int | Sequence[int] = 3,
                     regimes: Sequence[str] | str = "auto", few_data_fraction: float = 0.25,
                     split_seed: int = 0, extrapolation: Any = None, pde: Mapping[str, Any] | None = None,
                     constraints: Sequence[str] | None = None,
                     baselines: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] | None = None,
                     options: Mapping[str, Mapping[str, Any]] | None = None, threads: int = 1,
                     log: Callable[[str], None] | None = None) -> ArenaResult:
    """See the module docstring. ``extrapolation``: the shift split for the extrapolation regime
    (``data.meta['shift']`` or ``"edge"`` by default; see ``splits.make_splits``); ``constraints`` default to
    ``data.meta['constraints']``; ``pde`` to ``data.meta['pde']``."""
    import torch
    t_start = time.perf_counter()
    log = log or (lambda s: None)
    if not isinstance(budget, ComputeBudget):
        budget = ComputeBudget(wall_s=float(budget)) if isinstance(budget, (int, float)) else ComputeBudget(**budget)
    seeds = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    constraints = list(constraints if constraints is not None else data.meta.get("constraints", []))
    decl = pde if pde is not None else data.meta.get("pde")
    P = parse_pde(decl, data.coord_names, data.input_names) if data.kind == "field" else None
    options = options or {}
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, int(threads)))
    try:
        # ---------------------------------------------------------------- splits (shared by every method)
        ext = extrapolation or data.meta.get("shift") or "edge"
        base = make_splits(data, shift=ext, seed=split_seed)
        has_groups = data.groups is not None and len(np.unique(data.groups)) >= 5
        geo = make_splits(data, shift="group", seed=split_seed) if has_groups else None
        few = base.few_data(few_data_fraction, seed=split_seed)
        avail = {"interpolation": None, "few_data": None,
                 "extrapolation": None if len(base.shift) else "no extrapolation set",
                 "geometry_change": None if geo is not None else "the data has no groups (geometry or case labels)",
                 "steep_fronts": None if data.kind == "field" else "scalar outputs: no field to have a front"}
        wanted = list(REGIMES) if regimes == "auto" else list(regimes)
        steep = np.zeros(0, int)
        if data.kind == "field":
            ev = np.concatenate([base.test, base.shift])
            s = _steepness(data, ev)
            steep = ev[s >= np.quantile(s, 0.75)]
        evals = {"interpolation": base.test, "few_data": base.test, "extrapolation": base.shift,
                 "geometry_change": geo.shift if geo is not None else np.zeros(0, int), "steep_fronts": steep}
        trains = {"interpolation": "base", "few_data": "few", "extrapolation": "base", "geometry_change": "geo",
                  "steep_fronts": "base"}
        split_of = {"base": base, "few": few, "geo": geo}
        reg_rows = []
        for r in REGIMES:
            sp = split_of[trains[r]]
            reason = avail[r] or (None if r in wanted else "not selected")
            desc = REGIMES[r]
            if r == "extrapolation":
                desc = base.shift_description
            elif r == "geometry_change" and geo is not None:
                desc = geo.shift_description
            elif r == "few_data":
                desc = f"{len(few.train)} of {len(base.train)} training samples ({few_data_fraction:.0%}); " \
                       "same held-out samples as interpolation"
            reg_rows.append({"id": r, "title": r.replace("_", " "), "description": desc, "available": reason is None,
                             "reason": reason or "", "n_train": int(len(sp.train)) if sp is not None else 0,
                             "n_eval": int(len(evals[r]))})
        active = [r["id"] for r in reg_rows if r["available"]]
        needed = sorted({trains[r] for r in active})
        # ---------------------------------------------------------------- methods
        flat = lambda idx: data.Y[idx].reshape(len(idx), -1)  # noqa: E731
        reasons = applicable(data.kind, len(few.train) if "few" in needed else len(base.train), data.grid_shape,
                             P is not None)
        mids = list(CANDIDATES) if methods == "auto" else list(methods)
        rows = []
        for mid in CANDIDATES:
            c = CANDIDATES[mid]
            row = {"id": mid, "title": c.title, "family": c.family, "iterative": c.iterative,
                   "applicable": reasons[mid] is None and mid in mids,
                   "reason": reasons[mid] or ("" if mid in mids else "not selected"), "results": {}, "runs": []}
            rows.append(row)
            if not row["applicable"]:
                continue
            per_regime: dict[str, list[float]] = {r: [] for r in active}
            cost = {"train_s": [], "infer_s_per_query": [], "size_bytes": [], "n_params": []}
            viol: dict[str, list[float]] = {}
            pde_ratio: list[float] = []
            failures = []
            for seed in seeds:
                for tk in needed:
                    sp = split_of[tk]
                    ctx = TrainContext(data.X[sp.train], flat(sp.train), data.X[sp.val], flat(sp.val), budget, seed,
                                       data.grid_shape, data.coords, P, dict(options.get(mid, {})))
                    scale = ctx.Ytr.std(0) + 1e-12
                    log(f"{mid} seed {seed} train set {tk}")
                    try:
                        fitted, wall = fit_candidate(mid, ctx)
                    except Exception as e:
                        failures.append({"seed": seed, "train_set": tk, "error": f"{type(e).__name__}: {e}"})
                        continue
                    for r in active:
                        if trains[r] != tk or not len(evals[r]):
                            continue
                        idx = evals[r]
                        err = _sample_error(data.kind, fitted.predict(data.X[idx]), flat(idx), scale)
                        per_regime[r].append(float(err.mean()))
                    if tk == "base":
                        x1 = data.X[base.test[:1]]
                        cost["train_s"].append(float(fitted.info.get("train_wall_s", wall)))
                        cost["infer_s_per_query"].append(
                            time_per_query(lambda f=fitted, x=x1: f.predict(x), repeats=10, warmup=2)["median_s"])
                        cost["size_bytes"].append(float(fitted.size_bytes()))
                        cost["n_params"].append(float(fitted.n_params))
                        ev = np.concatenate([base.test, base.shift])
                        pred = fitted.predict(data.X[ev])
                        for k, v in _violations(data, pred, data.X[ev], constraints).items():
                            viol.setdefault(k, []).append(v)
                        if P is not None and len(base.test):
                            shape = (len(base.test), *data.grid_shape)
                            rt = P.fd_residual(flat(base.test).reshape(shape), data.coords, data.X[base.test])
                            rp = P.fd_residual(fitted.predict(data.X[base.test]).reshape(shape), data.coords,
                                               data.X[base.test])
                            pde_ratio.append(float(rp.mean() / max(rt.mean(), 1e-30)))
            for r in active:
                v = per_regime[r]
                row["results"][r] = {"errors": v, "n_ok": len(v),
                                     "error_mean": float(np.mean(v)) if v else None,
                                     "error_std": float(np.std(v, ddof=1)) if len(v) > 1 else (0.0 if v else None)}
            row["cost"] = {k: (float(np.mean(v)) if v else None) for k, v in cost.items()}
            row["cost_std"] = {k: (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in cost.items() if v}
            row["violations"] = {k: float(np.mean(v)) for k, v in viol.items()}
            row["pde_residual_ratio"] = float(np.mean(pde_ratio)) if pde_ratio else None
            row["failures"] = failures
        # ---------------------------------------------------------------- baselines
        bl_rows = []
        for name, solve in (baselines or {}).items():
            idx = base.test
            t0 = time.perf_counter()
            try:
                pred = np.asarray(solve(data.X[idx], data.coords), float)
            except Exception as e:
                bl_rows.append({"id": name, "error": f"{type(e).__name__}: {e}"})
                continue
            dt = (time.perf_counter() - t0) / max(len(idx), 1)
            scale = flat(base.train).std(0) + 1e-12
            err = _sample_error(data.kind, pred, flat(idx), scale)
            row = {"id": name, "description": getattr(solve, "description", ""), "regime": "interpolation",
                   "error_mean": float(err.mean()), "s_per_query": dt, "n_eval": int(len(idx)),
                   "violations": _violations(data, pred, data.X[idx], constraints)}
            if P is not None:
                shape = (len(idx), *data.grid_shape)
                rt = P.fd_residual(flat(idx).reshape(shape), data.coords, data.X[idx])
                rp = P.fd_residual(pred.reshape(shape), data.coords, data.X[idx])
                row["pde_residual_ratio"] = float(rp.mean() / max(rt.mean(), 1e-30))
            bl_rows.append(row)
        # ---------------------------------------------------------------- rankings
        rankings = {}
        for r in active:
            ok = [m for m in rows if m["applicable"] and m["results"].get(r, {}).get("error_mean") is not None]
            ok.sort(key=lambda m: m["results"][r]["error_mean"])
            rankings[r] = [{"rank": i + 1, "method": m["id"], "title": m["title"],
                            "error_mean": m["results"][r]["error_mean"], "error_std": m["results"][r]["error_std"],
                            "n_seeds": m["results"][r]["n_ok"]} for i, m in enumerate(ok)]
        overall = []
        for m in rows:
            ranks = [next(x["rank"] for x in rankings[r] if x["method"] == m["id"]) for r in rankings
                     if any(x["method"] == m["id"] for x in rankings[r])]
            if ranks:
                overall.append({"method": m["id"], "title": m["title"], "mean_rank": float(np.mean(ranks)),
                                "regimes_ranked": len(ranks)})
        overall.sort(key=lambda x: x["mean_rank"])
        metric = "rel_l2" if data.kind == "field" else "nrmse"
        protocol = {"metric": metric, "metric_definition": (
            "relative L2 error per held-out field, averaged" if metric == "rel_l2" else
            "RMSE / std of the training targets per output, averaged over outputs and held-out samples"),
            "budget": budget.to_dict(), "seeds": seeds, "split_seed": split_seed, "few_data_fraction": few_data_fraction,
            "stopping": "iterative methods: until the budget is spent, best of 10 validation checkpoints "
                        "(evaluation time not charged); one-shot fits: fitted once, time reported",
            "equal_data": "one split shared by every method and seed; the model seed varies, the data does not",
            "hyperparameters": "each method's defaults in pinneapple_analysis.surrogate.methods; no per-method search",
            "threads": int(threads), "dataset_sha256": data.sha256()}
        notes = []
        if any(m["applicable"] and not m["iterative"] for m in rows):
            notes.append("One-shot fits (ridge, gp, gbt) use the time they need, which can be less than the budget; "
                         "their train_s is that time.")
        result = ArenaResult(
            problem={**data.describe(), "constraints": constraints, "pde": decl},
            protocol=protocol, regimes=reg_rows, methods=rows, rankings=rankings, overall=overall,
            baselines=bl_rows, audit=_audit(data, base.train, np.concatenate([base.test, base.shift]), constraints),
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"), hardware=hardware_record(),
            wall_s=time.perf_counter() - t_start, notes=notes)
        return result
    finally:
        torch.set_num_threads(old)
