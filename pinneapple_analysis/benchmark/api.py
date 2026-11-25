"""``pp.benchmark``: prepare -> validate -> run -> report, with correctness gates and an advisor.

    import pinneapple as pp
    res = pp.benchmark.run(problems=["burgers", "darcy"], methods=["pinn", "fno", "deeponet"],
                           budget=pp.ComputeBudget(wall_s=300), budget_levels=[1/9, 1/3, 1], seeds=3, n_trials=4,
                           equal_flops=True, n_workers=4, out_dir="results/my_benchmark")
    res.table(); res.report(); res.diagnostics
"""
from __future__ import annotations

import json
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

from .advisor import Diagnosis, apply_suggestions, diagnose
from .checks import architecture_gates, determinism_check, row_status, seed_spread, training_gates
from .comparison import as_records
from .comparison import figures as comparison_figures
from .comparison import select as select_comparison
from .config import BenchmarkConfig
from .registry import Check, get_method, get_problem
from .report import (
    build_tables,
    cell,
    collect,
    collect_studies,
    figures,
    markdown,
    studies_figure,
    to_dataframe,
    write_csv,
)
from .runner import Plan, Workspace, _jdump, execute_jobs, prepare_problem, require_prepared
from .validation import trust_card

__all__ = ["run", "prepare", "report", "load", "BenchmarkResult", "BenchmarkConfig"]


def _config(config=None, **kwargs) -> BenchmarkConfig:
    if isinstance(config, BenchmarkConfig):
        cfg = config
    elif isinstance(config, (str, Path)):
        cfg = BenchmarkConfig.from_yaml(config)
    elif isinstance(config, dict):
        cfg = BenchmarkConfig.from_dict(config)
    else:
        cfg = BenchmarkConfig()
    names = {f.name for f in dc_fields(BenchmarkConfig)}
    if kwargs:
        d = cfg.to_dict()
        d.update({k: v for k, v in kwargs.items() if k in names})
        unknown = set(kwargs) - names
        if unknown:
            raise TypeError(f"unknown arguments {sorted(unknown)}")
        objs_p = [p for p in kwargs.get("problems", cfg.problems) if not isinstance(p, str)]
        objs_m = [m for m in kwargs.get("methods", cfg.methods) if not isinstance(m, str)]
        cfg = BenchmarkConfig.from_dict(d)
        if objs_p:
            cfg.problems = list(kwargs["problems"])
        if objs_m:
            cfg.methods = list(kwargs["methods"])
    return cfg


def _objects(cfg) -> tuple[dict, dict, dict]:
    P = {}
    for p in cfg.problems:
        prob = get_problem(p)
        P[prob.id] = prob.configure(cfg.problem_options.get(prob.id), cfg.smoke)
    M = {m.id if not isinstance(m, str) else m: get_method(m) for m in cfg.methods}
    custom = {**{k: v for k, v in P.items() if not isinstance(_orig(cfg.problems, k), str)},
              **{k: v for k, v in M.items() if not isinstance(_orig(cfg.methods, k), str)}}
    return P, M, custom


def _orig(lst, key):
    for x in lst:
        if (x if isinstance(x, str) else getattr(x, "id", None)) == key:
            return x
    return key


def prepare(config=None, out_dir: str | Path = "benchmark_out", *, force: bool = False, adopt: bool = False, log=print,
            **kwargs) -> dict:
    """Create and validate the artifacts (solver config, geometry, data, cost, hashes) in ``out_dir/prepared/``,
    for the user to inspect before any training. ``adopt``: use artifacts you placed there yourself (they are hashed
    and checked the same way). Returns {problem: manifest}."""
    cfg = _config(config, **kwargs)
    ws = Workspace(out_dir, cfg)
    P, _, _ = _objects(cfg)
    return {pid: prepare_problem(ws, prob, n_workers=cfg.n_workers, force=force, adopt=adopt, log=log)
            for pid, prob in P.items()}


def run(config=None, out_dir: str | Path = "benchmark_out", *, force: bool = False, prepare_missing: bool = True,
        stages=("search", "final", "flops", "study"), log=print, **kwargs) -> BenchmarkResult:
    """Run the benchmark (see the module docstring). Re-running with the same ``out_dir`` resumes from the cache;
    a different configuration is refused unless ``force``. Missing artifacts are prepared first
    (``prepare_missing``); stale or failed ones stop the run unless ``allow_failed_validation``."""
    cfg = _config(config, **kwargs)
    ws = Workspace(out_dir, cfg)
    ws.guard(force)
    P, M, custom = _objects(cfg)
    for pid, prob in P.items():
        prob.bind(ws.prepared(pid))
    banners = []
    for pid, prob in P.items():
        if prepare_missing and not (ws.prepared(pid) / "manifest.json").exists():
            prepare_problem(ws, prob, n_workers=cfg.n_workers, log=log)
        man = require_prepared(ws, prob, cfg.allow_failed_validation)
        if man.get("status") != "ok":
            banners.append(f"validation of {pid} FAILED and was overridden (allow_failed_validation)")
    plan = Plan(ws, cfg, P, M)
    if cfg.skip_checks:
        banners.append("CHECKS SKIPPED (skip_checks=True): no architecture, training or determinism gate ran")
        arch = []
    else:
        log("architecture gates")
        arch = architecture_gates(plan)
        _jdump(ws.root / "validation" / "architecture.json", [c.to_dict() for c in arch])
    blocked = {c.subject for c in arch if c.passed is False and c.severity == "error"}
    if blocked:
        log(f"blocked by the architecture gates (not trained): {sorted(blocked)}")
    plan.M = {m: meth for m, meth in M.items()}
    run_pairs = Plan(ws, cfg, P, M)
    run_pairs.pairs = lambda: ((p, m, pr, me) for p, m, pr, me in Plan.pairs(plan) if f"{p}/{m}" not in blocked)
    if "search" in stages:
        execute_jobs(ws, run_pairs.search_jobs(), n_workers=cfg.n_workers, objects=custom or None, log=log)
        _jdump(ws.root / "search.json", {f"{p}/{m}": plan.best(p, m).to_dict() for p, m, _, _ in run_pairs.pairs()})
    if "final" in stages:
        execute_jobs(ws, run_pairs.final_jobs(), n_workers=cfg.n_workers, objects=custom or None, log=log)
    if "flops" in stages and any(cfg.equal_flops_for(p) for p in P):
        execute_jobs(ws, run_pairs.flops_jobs(), n_workers=cfg.n_workers, objects=custom or None, log=log)
    if "study" in stages and (cfg.budget_study or cfg.ablations):
        execute_jobs(ws, run_pairs.study_jobs() + run_pairs.ablation_jobs(), n_workers=cfg.n_workers,
                     objects=custom or None, log=log)
    res = BenchmarkResult(ws, cfg, plan, arch, banners, blocked)
    if not cfg.skip_checks:
        res.spot_check_determinism(log)
    res.report()
    return res


def load(out_dir: str | Path) -> BenchmarkResult:
    """Open a finished (or partial) benchmark folder."""
    cfg = BenchmarkConfig.from_yaml(Path(out_dir) / "config.yaml")
    ws = Workspace(out_dir, cfg)
    P, M, _ = _objects(cfg)
    for pid, prob in P.items():
        prob.bind(ws.prepared(pid))
    arch_p = ws.root / "validation" / "architecture.json"
    arch = [Check(**d) for d in json.loads(arch_p.read_text())] if arch_p.exists() else []
    banners = []
    if cfg.skip_checks:
        banners.append("CHECKS SKIPPED (skip_checks=True): no architecture, training or determinism gate ran")
    for pid in P:
        mp = ws.prepared(pid) / "manifest.json"
        if mp.exists() and json.loads(mp.read_text()).get("status") != "ok":
            banners.append(f"validation of {pid} FAILED and was overridden (allow_failed_validation)")
    blocked = {c.subject for c in arch if c.passed is False and c.severity == "error"}
    return BenchmarkResult(ws, cfg, Plan(ws, cfg, P, M), arch, banners, blocked)


def report(out_dir: str | Path) -> BenchmarkResult:
    res = load(out_dir)
    res.report()
    return res


class BenchmarkResult:
    def __init__(self, ws: Workspace, cfg: BenchmarkConfig, plan: Plan, arch: list[Check], banners: list[str],
                 blocked: set):
        self.ws, self.config, self.plan, self.arch, self.banners, self.blocked = ws, cfg, plan, arch, banners, blocked
        self.out_dir = ws.root
        self._det: list[Check] = []
        det_p = ws.root / "validation" / "determinism.json"
        if det_p.exists():
            self._det = [Check(**d) for d in json.loads(det_p.read_text())]
        self._built = None

    def spot_check_determinism(self, log=print) -> None:
        for pid, mid, _, _meth in self.plan.pairs():
            if f"{pid}/{mid}" in self.blocked:
                continue
            log(f"determinism spot-check: {pid}/{mid}")
            try:
                self._det = [determinism_check(self.plan, pid, mid)]
            except Exception as e:
                self._det = [Check("determinism (same seed, same steps, same loss)", "training", f"{pid}/{mid}", False,
                                   detail=f"crashed: {e!r}")]
            break
        _jdump(self.ws.root / "validation" / "determinism.json", [c.to_dict() for c in self._det])

    # ------------------------------------------------------------------ assembly
    def _build(self):
        if self._built is not None:
            return self._built
        col = collect(self.plan)
        train, per_pair = ([], {}) if self.config.skip_checks else training_gates(self.ws, self.plan)
        spread = []
        for (pid, mid), lv in col["extracted"].items():
            prob = self.plan.P[pid]
            main = prob.main_metric_for(mid) if hasattr(prob, "main_metric") else None
            if main and 1.0 in lv:
                vals = [cell({0: d}, *main)["mean"] for d in lv[1.0].values() if cell({0: d}, *main)]
                spread.append(seed_spread(vals, f"{pid}/{mid}", f"{main[1]} on {main[0]}"))
        train = train + spread + self._det
        statuses = {}
        for pid, mid, _, _ in self.plan.pairs():
            subj = f"{pid}/{mid}"
            statuses[subj] = row_status([c for c in self.arch if c.subject == subj],
                                        [c for c in train if c.subject in (subj, "all")], self.config.skip_checks, False)
            if subj in self.blocked:
                statuses[subj] = "INVALID"
        tables = build_tables(self.plan, col, statuses)
        validation = {}
        for pid in self.plan.P:
            p = self.ws.root / "validation" / f"{pid}.json"
            validation[pid] = json.loads(p.read_text()) if p.exists() else {"status": "not prepared"}
        all_checks = []
        for v in validation.values():
            all_checks += [Check(**c) for c in v.get("checks", []) + v.get("validation", [])]
        all_checks += self.arch + train
        loss_terms = {}
        for p in self.ws.runs.glob("final__*__L1.000__s0*.json"):
            rec = json.loads(p.read_text())
            if rec.get("ok") and rec["result"].get("loss_components"):
                loss_terms[(rec["spec"]["problem"], rec["spec"]["method"])] = rec["result"]["loss_components"]
        diags = diagnose(self.config, self.plan, col["extracted"], all_checks, loss_terms=loss_terms)
        trust = {}
        for pid in self.plan.P:
            cs = [c for c in all_checks if c.subject == pid or c.subject.startswith(pid + "/")]
            st = [r["status"] for r in tables[pid]["rows"].values()]
            tier = "D" if not st or all(s == "INVALID" for s in st) else (
                "C" if any(s in ("INVALID", "did not converge") for s in st) else
                ("B" if any("warn" in s for s in st) else "A"))
            lim = [d.finding for d in diags if d.subject.startswith(pid) and d.severity != "info"]
            try:
                trust[pid] = trust_card(self.ws.root, cs, tier, lim or ["see the Diagnostics section"])
            except Exception as e:
                trust[pid] = {"error": {"title": "trust card", "question": "", "status": "failed",
                                        "evidence": [repr(e)]}}
        self._built = {"col": col, "tables": tables, "checks": all_checks, "validation": validation,
                       "diagnoses": diags, "trust": trust, "per_pair": per_pair, "studies": collect_studies(self.plan),
                       "comparison": select_comparison(self.plan, tables)}
        return self._built

    # ------------------------------------------------------------------ outputs
    @property
    def diagnostics(self) -> list[Diagnosis]:
        return self._build()["diagnoses"]

    @property
    def checks(self) -> list[Check]:
        return self._build()["checks"]

    def table(self):
        return to_dataframe(self.plan, self._build()["tables"])

    def to_markdown(self) -> str:
        b = self._build()
        return markdown(self.plan, b["tables"], b["checks"], b["validation"], b["trust"], b["diagnoses"], self.banners,
                        b["studies"], b["comparison"])

    def to_csv(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.ws.root / "table.csv")
        write_csv(path, self.plan, self._build()["tables"])
        return path

    def report(self) -> dict[str, Any]:
        b = self._build()
        figs = figures(self.plan, b["col"], b["tables"], self.ws.root / "figures", self.banners)
        figs += studies_figure(self.plan, b["studies"], self.ws.root / "figures")
        figs += comparison_figures(self.plan, b["comparison"], self.ws.root / "figures")
        (self.ws.root / "table.md").write_text(self.to_markdown())
        self.to_csv()
        summary = {"config": self.config.to_dict(), "banners": self.banners,
                   "tables": {pid: {**t, "rows": {m: {k: v for k, v in r.items()} for m, r in t["rows"].items()}}
                              for pid, t in b["tables"].items()},
                   "checks": [c.to_dict() for c in b["checks"]], "validation": b["validation"],
                   "trust_card": b["trust"], "diagnostics": [d.to_dict() for d in b["diagnoses"]], "studies": b["studies"],
                   "comparison": as_records(b["comparison"]), "figures": figs}
        _jdump(self.ws.root / "summary.json", summary)
        return summary

    def apply_suggestions(self, path: str | Path | None = None) -> BenchmarkConfig:
        return apply_suggestions(self.config, self.diagnostics, path or self.ws.root / "config.suggested.yaml")
