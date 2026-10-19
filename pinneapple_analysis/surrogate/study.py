"""Surrogate study: train every applicable candidate on the same data and budget, measure on unseen data, pick one.

``run_study(data, ...)``

1. splits once (``make_splits``): train / validation / held-out test, plus a distribution-shift set never seen in
   training (whole groups, the edge of the input space, or one end of an input);
2. lists every candidate with the reason it does or does not apply (fields vs scalars; PINN only with a declared
   PDE; GP sample limit);
3. trains each applicable candidate for each seed under the same ``ComputeBudget``
   (``pinneapple_analysis.cost.equal_budget``) and measures, in physical units, the held-out and shift errors,
   training time, inference time per query, parameter count and model size;
4. selects the method with the lowest **validation** error (the test and shift sets never choose), and packages it
   as a ``SurrogateModel`` with its limitations (validity bounds, OOD gate, where errors concentrate) and a trust
   report (``pinneapple_analysis.trust.TrustReport``) whose checks all ran on held-out data.

Error metrics: scalars, ``nrmse`` = RMSE / standard deviation of the training targets, per output then averaged;
fields, ``rel_l2`` = ||pred - true|| / ||true|| per sample, averaged.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget, hardware_record, time_per_query
from pinneapple_analysis.trust.trust_report import (
    CONTRADICTS,
    NOT_RUN,
    SUPPORTS,
    Check,
    TrustReport,
)

from .data import SurrogateData
from .limits import OODGate, limitations
from .methods import CANDIDATES, NonNegative, TrainContext, applicable, fit_candidate
from .model import SurrogateModel
from .pde import parse_pde
from .splits import Splits, make_splits

__all__ = ["run_study", "StudyResult", "evaluate", "DEFAULT_TARGETS"]

DEFAULT_TARGETS = {"nrmse": 0.10, "rel_l2": 0.05}


def _sample_error(kind: str, pred: np.ndarray, true: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if kind == "field":
        num = np.linalg.norm(pred - true, axis=1)
        return num / np.maximum(np.linalg.norm(true, axis=1), 1e-12)
    return np.sqrt((((pred - true) / scale) ** 2).mean(1))


def evaluate(kind: str, pred: np.ndarray, true: np.ndarray, scale: np.ndarray,
             output_names: Sequence[str]) -> dict[str, Any]:
    """Metrics of flattened predictions ``(n, p)`` against the truth, ``scale`` the training targets' std."""
    if len(true) == 0:
        return {}
    err = pred - true
    out: dict[str, Any] = {"n": int(len(true))}
    if kind == "field":
        rel = _sample_error(kind, pred, true, scale)
        out.update(rel_l2=float(rel.mean()), rel_l2_median=float(np.median(rel)), rel_l2_max=float(rel.max()),
                   rmse=float(np.sqrt((err ** 2).mean())), max_abs=float(np.abs(err).max()),
                   nrmse=float(np.sqrt((err ** 2).mean()) / max(float(scale.mean()), 1e-12)))
        out["error"] = out["rel_l2"]
    else:
        per = {}
        for j, n in enumerate(output_names):
            e = err[:, j]
            ss = float(((true[:, j] - true[:, j].mean()) ** 2).sum())
            per[n] = {"rmse": float(np.sqrt((e ** 2).mean())), "nrmse": float(np.sqrt((e ** 2).mean()) / scale[j]),
                      "mae": float(np.abs(e).mean()), "max_abs": float(np.abs(e).max()),
                      "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else None}
        out["per_output"] = per
        out["nrmse"] = float(np.mean([v["nrmse"] for v in per.values()]))
        r2 = [v["r2"] for v in per.values() if v["r2"] is not None]
        out["r2_min"] = float(min(r2)) if r2 else None
        out["error"] = out["nrmse"]
    return out


@dataclass
class StudyResult:
    data: SurrogateData
    splits: Splits
    budget: ComputeBudget
    seeds: list[int]
    methods: list[dict[str, Any]]
    best: str | None
    model: SurrogateModel | None
    limitations: dict[str, Any]
    trust: dict[str, Any]
    metric: str
    created: str
    hardware: dict[str, Any]
    pde: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def ranking(self, split: str = "test") -> list[dict[str, Any]]:
        rows = [m for m in self.methods if m["applicable"] and m.get("summary")]
        key = f"{split}_error_mean"
        rows = [r for r in rows if r["summary"].get(key) is not None]
        return [{"method": r["id"], "title": r["title"], "error": r["summary"][key],
                 "spread": r["summary"].get(f"{split}_error_std")} for r in sorted(rows, key=lambda r: r["summary"][key])]

    def to_markdown(self) -> str:
        """The study as a report: data, protocol, comparison, chosen model, trust checks, limitations."""
        f = lambda v: "—" if v is None else f"{v:.4g}"  # noqa: E731
        d, sp, lim = self.data.describe(), self.splits.to_dict(), self.limitations
        out = [f"# Surrogate study: {', '.join(d['outputs'])} from {', '.join(d['inputs'])}", "",
               f"Created {self.created}. Dataset sha256 `{d['sha256'][:16]}`, {d['n_samples']} samples ({d['kind']}"
               + (f", grid {d['grid_shape']}" if d["grid_shape"] else "") + f"). Source: {d['meta'].get('source', '—')}.",
               "", "## Protocol", "",
               f"- Splits: {sp['n_train']} train, {sp['n_val']} validation, {sp['n_test']} held-out test, "
               f"{sp['n_shift']} distribution shift ({sp['shift_description'] or 'none'}); split seed {sp['seed']}.",
               f"- Budget per training run: {_budget_text(self.budget)}; seeds {self.seeds}.",
               f"- Metric: {self.metric} (lower is better), in physical units on data no method trained on.",
               *[f"- {n}" for n in self.notes],
               f"- Hardware: {self.hardware.get('cpu', '?')}, {self.hardware.get('logical_cpus', '?')} logical CPUs, "
               f"torch {self.hardware.get('torch', '?')}.", "", "## Comparison", "",
               "| method | family | validation | test (± seeds) | shift | train s | inference s/query | parameters |",
               "|---|---|---|---|---|---|---|---|"]
        for m in self.methods:
            s = m.get("summary") or {}
            if not m["applicable"]:
                out.append(f"| {m['title']} | {m['family']} | not run: {m['reason']} | | | | | |")
                continue
            out.append(f"| {m['title']}{' (chosen)' if m['id'] == self.best else ''} | {m['family']} | "
                       f"{f(s.get('val_error_mean'))} | {f(s.get('test_error_mean'))} ± {f(s.get('test_error_std'))} | "
                       f"{f(s.get('shift_error_mean'))} | {f(s.get('train_s'))} | {f(s.get('infer_s_per_query'))} | "
                       f"{f(s.get('n_params'))} |")
        out += ["", f"## Trust: {self.trust.get('decision', '—')}", ""]
        for c in self.trust.get("checks", []):
            out.append(f"- {c['status']}: {c['name']} (value {f(c['value'])}"
                       + (f", limit {f(c['limit'])}" if c.get("limit") is not None else "") + ")"
                       + (f" — {c['reason']}" if c.get("reason") else ""))
        out += ["", "## Limitations", ""]
        for k, (a, b) in (lim.get("validity_bounds") or {}).items():
            out.append(f"- valid for {k} in [{a:.4g}, {b:.4g}]")
        if lim.get("ood_gate"):
            out.append(f"- OOD gate: {lim['ood_gate']['rule']}")
        ge = lim.get("gate_effect") or {}
        if ge:
            out.append(f"- held-out samples flagged by the gate: {ge.get('n_flagged')} (mean error "
                       f"{f(ge.get('mean_error_flagged'))} vs {f(ge.get('mean_error_inside'))} inside)")
        if lim.get("concentration"):
            out.append(f"- {lim['concentration']}")
        if lim.get("extrapolation_ratio") is not None:
            out.append(f"- shift error / held-out error = {lim['extrapolation_ratio']:.2f}")
        if lim.get("pde_residual"):
            out.append(f"- PDE residual of predictions / of the reference data = {lim['pde_residual']['ratio']:.3g}")
        return "\n".join(out) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.data.describe(), "splits": self.splits.to_dict(), "budget": self.budget.to_dict(),
                "seeds": self.seeds, "metric": self.metric, "methods": self.methods, "best": self.best,
                "ranking_test": self.ranking("test"), "ranking_shift": self.ranking("shift"),
                "limitations": self.limitations, "trust": self.trust, "created": self.created,
                "hardware": self.hardware, "pde": self.pde, "notes": self.notes,
                "model_card": self.model.card if self.model else None}


def _budget_text(b: ComputeBudget) -> str:
    unit = {"wall_s": "s wall clock", "flops": "FLOPs", "steps": "optimiser steps"}
    return ", ".join(f"{v:g} {unit[k]}" for k, v in b.to_dict().items() if v is not None)


def _mean_std(v: list[float]) -> tuple[float | None, float | None]:
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _trust(metric: str, target: float, best: dict, lim: dict, pde_check: dict | None, n_seeds: int) -> TrustReport:
    s = best["summary"]
    te, sh = s.get("test_error_mean"), s.get("shift_error_mean")
    checks = [Check(f"held-out {metric} <= {target:g}", SUPPORTS if te is not None and te <= target else CONTRADICTS,
                    value=te, limit=target, source="held-out test split", critical=False)]
    r2 = s.get("test_r2_min")
    if r2 is not None:
        checks.append(Check("beats the mean predictor on every output (R^2 > 0)", SUPPORTS if r2 > 0 else CONTRADICTS,
                            value=r2, limit=0.0, source="held-out test split", critical=True))
    if sh is None or te is None:
        checks.append(Check("distribution-shift error <= 3 x held-out error", NOT_RUN,
                            reason="no distribution-shift split"))
    else:
        ratio = sh / max(te, 1e-30)
        checks.append(Check("distribution-shift error <= 3 x held-out error", SUPPORTS if ratio <= 3 else CONTRADICTS,
                            value=ratio, limit=3.0, source=lim.get("shift_description", "shift split")))
    if n_seeds >= 2:
        cv = (s.get("test_error_std") or 0.0) / max(te or 1e-30, 1e-30)
        checks.append(Check("seed-to-seed spread of the held-out error <= 50 %", SUPPORTS if cv <= 0.5 else CONTRADICTS,
                            value=cv, limit=0.5, source=f"{n_seeds} seeds", weight=0.5))
    else:
        checks.append(Check("seed-to-seed spread of the held-out error <= 50 %", NOT_RUN, weight=0.5,
                            reason="one seed"))
    if pde_check is not None:
        r = pde_check["ratio"]
        checks.append(Check("PDE residual of predictions <= 2 x that of the reference data",
                            SUPPORTS if r <= 2 else CONTRADICTS, value=r, limit=2.0,
                            source="finite differences on the held-out fields"))
    return TrustReport(checks, model=best["id"])


def _trust_dict(rep: TrustReport) -> dict[str, Any]:
    return {"decision": rep.decisao, "score": rep.score, "coverage": rep.cobertura,
            "checks": [{"name": c.name, "status": c.status, "value": c.value, "limit": c.limit,
                        "critical": c.critical, "source": c.source, "reason": c.reason} for c in rep.checks],
            "rules": "any critical check failing -> REJECT; any other failing, coverage < 60 % or score < 0.8 -> "
                     "REVIEW; else APPROVED (pinneapple_analysis.trust.TrustReport)"}


def run_study(data: SurrogateData, *, methods: Sequence[str] | str = "auto", budget: ComputeBudget | Mapping |
              float = 10.0, seeds: Sequence[int] | int = 1, splits: Splits | None = None,
              split_options: Mapping[str, Any] | None = None, pde: Mapping[str, Any] | None = None,
              target: float | None = None, options: Mapping[str, Mapping[str, Any]] | None = None,
              threads: int = 1, log: Callable[[str], None] | None = None) -> StudyResult:
    """See the module docstring. ``budget``: a ``ComputeBudget``, a dict or seconds of wall clock per training run;
    ``seeds``: a count or explicit seeds; ``pde``: a declaration (``pde.py``) or ``data.meta['pde']``;
    ``target``: the held-out error a deployable model must reach (default 0.10 nRMSE, 0.05 relative L2)."""
    import torch
    log = log or (lambda s: None)
    if not isinstance(budget, ComputeBudget):
        budget = ComputeBudget(wall_s=float(budget)) if isinstance(budget, (int, float)) else ComputeBudget(**budget)
    seeds = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(threads)))
    try:
        return _run(data, methods, budget, seeds, splits, split_options, pde, target, options or {}, log)
    finally:
        torch.set_num_threads(old_threads)


def nonnegative_mask(data: SurrogateData) -> np.ndarray | None:
    """Boolean mask over the flattened outputs named in ``data.meta["nonnegative"]``: output names for scalar data,
    channel names (``meta["channels"]``, the last grid axis) for fields. None when nothing is declared."""
    names = list(data.meta.get("nonnegative") or [])
    if not names:
        return None
    if data.kind == "field":
        chans = list(data.meta.get("channels") or [])
        unknown = [n for n in names if n not in chans]
        if unknown:
            raise ValueError(f"meta['nonnegative'] names {unknown}, not channels of the field ({chans})")
        m = np.zeros(data.grid_shape, bool)
        m[..., [chans.index(n) for n in names]] = True
        return m.ravel()
    unknown = [n for n in names if n not in data.output_names]
    if unknown:
        raise ValueError(f"meta['nonnegative'] names {unknown}, not outputs ({data.output_names})")
    return np.isin(np.asarray(data.output_names), names)


def _run(data, methods, budget, seeds, splits, split_options, pde, target, options, log) -> StudyResult:
    kind, metric = data.kind, ("rel_l2" if data.kind == "field" else "nrmse")
    target = float(target if target is not None else DEFAULT_TARGETS[metric])
    sp = splits or make_splits(data, seed=seeds[0], **dict(split_options or {}))
    decl = pde if pde is not None else data.meta.get("pde")
    P = parse_pde(decl, data.coord_names, data.input_names) if kind == "field" else None
    flat = lambda idx: data.Y[idx].reshape(len(idx), -1)  # noqa: E731
    Xtr, Ytr, Xva, Yva = data.X[sp.train], flat(sp.train), data.X[sp.val], flat(sp.val)
    Xte, Yte, Xsh, Ysh = data.X[sp.test], flat(sp.test), data.X[sp.shift], flat(sp.shift)
    scale = Ytr.std(0) + 1e-12
    nonneg = nonnegative_mask(data)
    if nonneg is not None and P is not None:
        raise ValueError("non-negative outputs are trained on their square root, which a declared PDE does not hold "
                         "for: drop meta['nonnegative'] or the PDE")
    Ytr_fit, Yva_fit = (NonNegative.forward_target(Ytr, nonneg), NonNegative.forward_target(Yva, nonneg)) \
        if nonneg is not None else (Ytr, Yva)
    reasons = applicable(kind, len(sp.train), data.grid_shape, P is not None)
    wanted = list(CANDIDATES) if methods == "auto" else list(methods)
    unknown = [m for m in wanted if m not in CANDIDATES]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; available {list(CANDIDATES)}")
    rows, fits = [], {}
    for mid in CANDIDATES:
        c = CANDIDATES[mid]
        row = {"id": mid, "title": c.title, "family": c.family, "iterative": c.iterative,
               "applicable": reasons[mid] is None and mid in wanted,
               "reason": reasons[mid] or ("" if mid in wanted else "not selected"), "seeds": []}
        rows.append(row)
        if not row["applicable"]:
            continue
        for seed in seeds:
            ctx = TrainContext(Xtr, Ytr_fit, Xva, Yva_fit, budget, seed, data.grid_shape, data.coords, P,
                               dict(options.get(mid, {})))
            log(f"{mid} seed {seed}: training")
            try:
                fitted, wall = fit_candidate(mid, ctx)
            except Exception as e:  # a method that fails is reported, not hidden
                row["seeds"].append({"seed": seed, "error": f"{type(e).__name__}: {e}"})
                continue
            if nonneg is not None:                      # errors and timings below are of the squared prediction
                fitted = NonNegative(fitted, nonneg)
            rec = {"seed": seed, "train_s": float(fitted.info.get("train_wall_s", wall)),
                   "fit_wall_s": float(wall), "n_params": int(fitted.n_params), "size_bytes": fitted.size_bytes(),
                   "info": fitted.info,
                   "val": evaluate(kind, fitted.predict(Xva), Yva, scale, data.output_names) if len(Xva) else {},
                   "test": evaluate(kind, fitted.predict(Xte), Yte, scale, data.output_names),
                   "shift": evaluate(kind, fitted.predict(Xsh), Ysh, scale, data.output_names) if len(Xsh) else {}}
            x1 = Xte[:1]
            rec["infer_s_per_query"] = time_per_query(lambda f=fitted, x=x1: f.predict(x), repeats=15, warmup=2)[
                "median_s"]
            xb = Xte[: min(100, len(Xte))]
            rec["infer_s_per_query_batch"] = time_per_query(lambda f=fitted, x=xb: f.predict(x), repeats=5, warmup=1,
                                                            n_per_call=len(xb))["median_s"]
            row["seeds"].append(rec)
            fits[(mid, seed)] = fitted
        ok = [s for s in row["seeds"] if "error" not in s]
        summ: dict[str, Any] = {"n_ok": len(ok), "n_failed": len(row["seeds"]) - len(ok)}
        for split in ("val", "test", "shift"):
            m, s = _mean_std([r[split].get("error") for r in ok if r[split]])
            summ[f"{split}_error_mean"], summ[f"{split}_error_std"] = m, s
        r2 = [r["test"].get("r2_min") for r in ok if r["test"].get("r2_min") is not None]
        summ["test_r2_min"] = float(min(r2)) if r2 else None
        for k in ("train_s", "infer_s_per_query", "infer_s_per_query_batch", "n_params", "size_bytes"):
            summ[k] = _mean_std([r[k] for r in ok])[0]
        row["summary"] = summ
    cand = [r for r in rows if r["applicable"] and r.get("summary", {}).get("val_error_mean") is not None]
    if not cand:
        cand = [r for r in rows if r["applicable"] and r.get("summary", {}).get("test_error_mean") is not None]
        sel = "test_error_mean"
    else:
        sel = "val_error_mean"
    notes = [f"Selection by {sel.split('_')[0]} error; held-out test and shift sets never choose the model."]
    if not cand:
        return StudyResult(data, sp, budget, seeds, rows, None, None, {}, {}, metric,
                           datetime.now(timezone.utc).isoformat(timespec="seconds"), hardware_record(), decl,
                           notes + ["No candidate could be trained."])
    best = min(cand, key=lambda r: r["summary"][sel])
    ok_seeds = [s for s in best["seeds"] if "error" not in s]
    bseed = min(ok_seeds, key=lambda s: (s["val"] or s["test"])["error"])["seed"]
    fitted = fits[(best["id"], bseed)]
    gate = OODGate(Xtr, data.input_names)
    X_eval = np.concatenate([Xte, Xsh])
    Y_eval = np.concatenate([Yte, Ysh])
    P_eval = fitted.predict(X_eval)
    err = _sample_error(kind, P_eval, Y_eval, scale)
    sets = {"test": np.r_[np.ones(len(Xte), bool), np.zeros(len(Xsh), bool)],
            "shift": np.r_[np.zeros(len(Xte), bool), np.ones(len(Xsh), bool)]}
    lim = limitations(gate, X_eval, err, sets, Xtr, field_err=np.abs(P_eval - Y_eval) if kind == "field" else None,
                      coords=data.coords, coord_names=data.coord_names, metric=metric)
    lim["shift_description"] = sp.shift_description
    pde_check = None
    if P is not None and len(Xte):
        shape = (len(Xte), *data.grid_shape)
        r_true = P.fd_residual(Yte.reshape(shape), data.coords, Xte)
        r_pred = P.fd_residual(fitted.predict(Xte).reshape(shape), data.coords, Xte)
        pde_check = {"residual_pred_mean": float(r_pred.mean()), "residual_data_mean": float(r_true.mean()),
                     "ratio": float(r_pred.mean() / max(r_true.mean(), 1e-30))}
        lim["pde_residual"] = pde_check
    rep = _trust(metric, target, best, lim, pde_check, len(seeds))
    trust = _trust_dict(rep)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    s = best["summary"]
    card = {"title": f"{best['title']} surrogate of {', '.join(data.output_names)}", "dataset_sha256": data.sha256(),
            "dataset": {k: v for k, v in data.describe().items() if k != "meta"} | {"source": data.meta.get("source")},
            "metric": metric, "target": target, "seed": bseed,
            "metrics": {"test": next(x["test"] for x in ok_seeds if x["seed"] == bseed),
                        "shift": next(x["shift"] for x in ok_seeds if x["seed"] == bseed),
                        "test_error_mean": s["test_error_mean"], "shift_error_mean": s["shift_error_mean"],
                        "infer_s_per_query": s["infer_s_per_query"], "train_s": s["train_s"]},
            "splits": sp.to_dict(), "budget": budget.to_dict(), "limitations": lim, "trust": trust, "pde": decl,
            "created": created}
    model = SurrogateModel(fitted, gate, input_names=data.input_names, output_names=data.output_names,
                           grid_shape=data.grid_shape, coords=data.coords, coord_names=data.coord_names, card=card)
    if nonneg is not None:
        notes.append(f"{', '.join(data.meta['nonnegative'])}: trained on the square root and predicted as its "
                     "square, so never negative; every error is measured on the predicted values themselves.")
    if any(r["applicable"] and not r["iterative"] for r in rows):
        notes.append("One-shot fits (ridge, gp, gbt) are fitted once and report the time they used; the iterative "
                     f"methods each used the full budget ({_budget_text(budget)}).")
    return StudyResult(data, sp, budget, seeds, rows, best["id"], model, lim, trust, metric, created,
                       hardware_record(), decl, notes)
