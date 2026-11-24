"""The benchmark machinery: the out_dir workspace, the prepare stage, the job cache and the search, final and
equal-FLOPs stages, executed as one process per job (``threads_per_run`` threads each, ``n_workers`` at a time).

out_dir/
  config.yaml          the resolved configuration (its result hash guards the cache)
  hardware.json        hardware_record() of the machine
  prepared/<problem>/  solver config, data, geometry, cost, hashes (manifest.json), checks.json
  validation/          problem.json per problem: solver verification, data checks, PINNeAPPle validation stack
  runs/                one JSON per job: the cache
  table.md, table.csv, summary.json, figures/
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_analysis.cost.equal_budget import (
    ComputeBudget,
    equal_budget_search,
    hardware_record,
)

from .config import BenchmarkConfig
from .registry import Check, get_method, get_problem

__all__ = ["Workspace", "ConfigMismatch", "PreparationError", "prepare_problem", "execute_jobs", "run_job"]


class ConfigMismatch(RuntimeError):
    pass


class PreparationError(RuntimeError):
    pass


def _jdump(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(path)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Check):
        return o.to_dict()
    return str(o)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Workspace:
    def __init__(self, out_dir: str | Path, config: BenchmarkConfig):
        self.root = Path(out_dir)
        self.cfg = config
        for d in ("prepared", "validation", "runs", "figures", "records"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def records(self) -> Path:
        """One verifiable record per job (#38): what ran, on what, the files by hash, and how to re-run it."""
        return self.root / "records"

    def record_path(self, job_id: str) -> Path:
        return self.records / f"{job_id}.record.json"

    def records_for(self, problem_id: str, method_id: str) -> list[str]:
        """The records behind one (problem, method) row of the report, relative to the workspace."""
        tag = f"__{problem_id}__{method_id}__"
        return sorted(str(p.relative_to(self.root)) for p in self.records.glob("*.record.json") if tag in p.name)

    def prepared(self, pid: str) -> Path:
        return self.root / "prepared" / pid

    def guard(self, force: bool) -> None:
        """Refuse to mix results of two configurations in one out_dir (unless ``force``)."""
        p = self.root / "config.yaml"
        if p.exists():
            old = BenchmarkConfig.from_yaml(p)
            changed = old.result_hash() != self.cfg.result_hash() or not self.cfg.extends_studies(old)
            if changed and not force:
                diff = {k: (v, self.cfg.to_dict()[k]) for k, v in old.to_dict().items() if self.cfg.to_dict()[k] != v}
                raise ConfigMismatch(f"{self.root} holds results of another configuration (changed: {sorted(diff)}); "
                                     "use another out_dir or force=True / --force to mix them")
        self.cfg.to_yaml(p)
        _jdump(self.root / "hardware.json", hardware_record())

    def result(self, job_id: str) -> dict | None:
        p = self.runs / f"{job_id}.json"
        return json.loads(p.read_text()) if p.exists() else None


# ====================================================================== prepare
def _problem_hash(problem) -> str:
    key = {"id": problem.id, "options": problem.solver_config(), "smoke": getattr(problem, "smoke", False)}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def prepare_problem(ws: Workspace, problem, *, n_workers: int = 1, force: bool = False, adopt: bool = False,
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    """Create (or reuse) the problem's artifacts, then verify the solver, check the data and run the validation
    stack on them. Returns the manifest (status ok / failed).

    ``adopt``: artifacts already placed in ``prepared/<problem>/`` (your own data, or data generated elsewhere with
    this configuration) are hashed and used instead of being generated; the manifest records them as adopted, and
    they go through exactly the same checks."""
    out = ws.prepared(problem.id)
    out.mkdir(parents=True, exist_ok=True)
    problem.bind(out)
    mpath = out / "manifest.json"
    h = _problem_hash(problem)
    man = json.loads(mpath.read_text()) if mpath.exists() else None
    fresh = man is not None and man.get("config_hash") == h and man.get("generated") and not force and all(
        (out / f).exists() and _sha_file(out / f) == s for f, s in man.get("hashes", {}).items())
    if man is not None and man.get("config_hash") != h and not force:
        raise ConfigMismatch(f"{out} was prepared with another configuration of {problem.id}; prepare with force")
    files = problem.artifact_files(out) if adopt else None
    if not fresh and adopt and files and all((out / f).exists() for f in files):
        log(f"prepare {problem.id}: adopting {len(files)} existing artifacts")
        cost = json.loads((out / "cost.json").read_text()) if (out / "cost.json").exists() else None
        man = {"problem": problem.id, "config_hash": h, "solver_config": problem.solver_config(), "generated": True,
               "adopted": True, "files": files, "cost": cost, "hashes": {f: _sha_file(out / f) for f in files}}
        if hasattr(problem, "complete_adopted"):
            problem.complete_adopted(out)
        (out / "solver_config.json").write_text(json.dumps(problem.solver_config(), indent=1, default=str))
        fresh = True
    if not fresh:
        log(f"prepare {problem.id}: generating data")
        t0 = time.time()
        gen = problem.prepare(out, n_workers=n_workers)
        man = {"problem": problem.id, "config_hash": h, "solver_config": problem.solver_config(), "generated": True,
               "files": gen.get("files", []), "cost": gen.get("cost"), "wall_s": time.time() - t0,
               "hashes": {f: _sha_file(out / f) for f in gen.get("files", []) if (out / f).is_file()}}
        (out / "solver_config.json").write_text(json.dumps(problem.solver_config(), indent=1, default=str))
    log(f"prepare {problem.id}: verification, data checks, validation")
    checks = problem.verify_solver() + problem.data_checks(out)
    try:
        val = problem.validation(out)
    except Exception as e:                                       # a validation item that crashes is a failed item
        val = [Check("validation stack", "validation", problem.id, False, detail=f"crashed: {e!r}")]
    blocking = [c for c in checks + val if c.passed is False and c.severity == "error"]
    man.update(status="failed" if blocking else "ok", checked=time.strftime("%Y-%m-%d %H:%M:%S"),
               blocking=[c.name for c in blocking])
    _jdump(out / "checks.json", [c.to_dict() for c in checks])
    _jdump(ws.root / "validation" / f"{problem.id}.json", {"problem": problem.id, "checks": [c.to_dict() for c in checks],
                                                          "validation": [c.to_dict() for c in val],
                                                          "status": man["status"]})
    _jdump(mpath, man)
    return man





def require_prepared(ws: Workspace, problem, allow_failed: bool) -> dict:
    mpath = ws.prepared(problem.id) / "manifest.json"
    if not mpath.exists():
        raise PreparationError(f"{problem.id} is not prepared: run pp.benchmark.prepare (or the CLI's prepare) first")
    man = json.loads(mpath.read_text())
    if man.get("config_hash") != _problem_hash(problem):
        raise PreparationError(f"{problem.id}: prepared artifacts are stale (configuration changed); prepare again")
    for f, s in man.get("hashes", {}).items():
        if not (ws.prepared(problem.id) / f).exists() or _sha_file(ws.prepared(problem.id) / f) != s:
            raise PreparationError(f"{problem.id}: prepared file {f} is missing or changed since it was validated")
    if man.get("status") != "ok" and not allow_failed:
        raise PreparationError(f"{problem.id}: validation failed ({man.get('blocking')}); fix it or pass "
                               "allow_failed_validation=True (the table then says so)")
    return man


# ====================================================================== jobs
def _job_budget(spec) -> ComputeBudget:
    return ComputeBudget(**spec["budget"])


def run_job(spec: dict, out_dir: str, objects: dict | None = None) -> str:
    """Run one job in this process and write runs/<id>.json (the result, or the traceback)."""
    from pinneapple_analysis.cost.equal_budget import hardware_record as hw
    threads = int(spec.get("threads", 1))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass
    runs = Path(out_dir) / "runs"
    path = runs / f"{spec['id']}.json"
    if path.exists():
        if not (Path(out_dir) / "records" / f"{spec['id']}.record.json").exists():
            write_job_record(spec, out_dir)                       # a cached result earns its record too
        return spec["id"] + " (cached)"
    t0 = time.time()
    try:
        objects = objects or {}
        prob = objects.get(spec["problem"]) or get_problem(spec["problem"])
        prob.configure(spec.get("problem_options"), spec.get("smoke", False))
        prob.bind(Path(out_dir) / "prepared" / spec["problem"])
        meth = objects.get(spec["method"]) or get_method(spec["method"])
        prepared = Path(out_dir) / "prepared" / spec["problem"]
        kw = dict(spec.get("instance") or {})
        res = meth.train(prob, prepared, spec["cfg"], spec["seed"], _job_budget(spec),
                         final=spec["stage"] != "search", ckpt=str(runs / f"{spec['id']}.ckpt.pt"), **kw)
        out = {"spec": spec, "result": res, "ok": True}
    except Exception:                                            # failures are recorded, never hidden
        out = {"spec": spec, "ok": False, "error": traceback.format_exc()}
    out.update(hardware=hw(), job_wall_s=time.time() - t0, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    _jdump(path, out)
    write_job_record(spec, out_dir)
    return spec["id"] + (" ok" if out["ok"] else " FAILED")


def write_job_record(spec: dict, out_dir: str, *, root: Path | None = None) -> Path:
    """Write ``records/<job id>.record.json``: the evidence behind one published number of the benchmark (#38).

    It names the job's parameters (problem, method, stage, seed, budget, instance), the environment and commit, the
    files the job produced with their sha256, and the command that re-executes it. ``python -m
    pinneapple_core.run_record verify <out_dir>`` checks every record of a run in one call.
    """
    from pinneapple_core import run_record

    root = Path(root or out_dir)
    job_id = spec["id"]
    files = [root / "runs" / f"{job_id}.json"]
    ckpt = root / "runs" / f"{job_id}.ckpt.pt"
    if ckpt.exists():                                            # a checkpoint is an output too, when there is one
        files.append(ckpt)
    rec = run_record.record(
        "benchmark-run", job_id, root=root, outputs=files,
        command=[sys.executable, "-m", "pinneapple_analysis.benchmark", "run", "--config", str(root / "config.yaml"),
                 "--out", str(root), "--force"],
        parameters={k: v for k, v in spec.items() if k != "id"},
        note=f"runs/{job_id}.json is cached: delete it to re-execute this job (the command above resumes the rest)",
    )
    return rec.write(root / "records" / f"{job_id}.record.json")


def execute_jobs(ws: Workspace, jobs: list[dict], *, n_workers: int, objects: dict | None = None,
                 log=print) -> None:
    todo = [j for j in jobs if ws.result(j["id"]) is None]
    log(f"{len(jobs)} jobs, {len(todo)} to run")
    todo.sort(key=lambda j: -float(j["budget"].get("wall_s", 0) or 0))   # long jobs first
    if not todo:
        return
    if objects or n_workers <= 1:                                # user objects cannot cross a process boundary
        for i, j in enumerate(todo):
            log(f"[{i + 1}/{len(todo)}] " + run_job(j, str(ws.root), objects))
        return
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(n_workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = [ex.submit(run_job, j, str(ws.root)) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            log(f"[{i + 1}/{len(todo)}] {f.result()}")


def lvl(x: float) -> str:
    return f"L{x:.3f}"


class Plan:
    """The job specs of one benchmark: ids compatible with experiment 09 of PINNeAPPle-Benchmark."""

    def __init__(self, ws: Workspace, cfg: BenchmarkConfig, problems: dict, methods: dict):
        self.ws, self.cfg, self.P, self.M = ws, cfg, problems, methods

    def _base(self, pid, mid, cfg, seed, budget, stage, **extra):
        return {"problem": pid, "method": mid, "cfg": cfg, "seed": seed, "budget": budget, "stage": stage,
                "threads": self.cfg.threads_per_run, "smoke": self.cfg.smoke,
                "problem_options": self.cfg.problem_options.get(pid), **extra}

    def pairs(self):
        for pid, prob in self.P.items():
            for mid, meth in self.M.items():
                if meth.supports(prob):
                    yield pid, mid, prob, meth

    def space(self, prob, meth) -> dict:
        """The method's grid for this problem, with ``method_options[method]`` overriding entries (a list replaces
        the grid of that hyper-parameter, a scalar fixes it)."""
        sp = dict(meth.space(prob))
        for k, v in (self.cfg.method_options.get(meth.id) or {}).items():
            sp[k] = list(v) if isinstance(v, (list, tuple)) else [v]
        return sp

    def search_configs(self, prob, meth) -> list[dict]:
        seen = []
        equal_budget_search(self.space(prob, meth), lambda c: seen.append(dict(c)) or 0.0, n_trials=self.cfg.n_trials,
                            seed=0)
        return seen

    def search_jobs(self) -> list[dict]:
        jobs = []
        for pid, mid, prob, meth in self.pairs():
            b = self.cfg.budget_for(pid).scaled(self.cfg.search_fraction).to_dict()
            b = {k: v for k, v in b.items() if v is not None}
            inst = prob.instances()["val"] if meth.per_instance else [("", None)]
            for i, c in enumerate(self.search_configs(prob, meth)):
                for _, ref in inst[:1]:
                    jobs.append({"id": f"search__{pid}__{mid}__{i}", **self._base(pid, mid, c, 0, b, "search",
                                                                                     instance=ref)})
        return jobs

    def best(self, pid, mid):
        prob, meth = self.P[pid], self.M[mid]
        cfgs = self.search_configs(prob, meth)
        scores = {}
        for i, c in enumerate(cfgs):
            r = self.ws.result(f"search__{pid}__{mid}__{i}")
            scores[json.dumps(c, sort_keys=True)] = (prob.search_score(mid, r["result"]) if r and r.get("ok")
                                                     else float("nan"))
        return equal_budget_search(self.space(prob, meth), lambda c: scores[json.dumps(c, sort_keys=True)],
                                   n_trials=self.cfg.n_trials, seed=0)

    def final_jobs(self) -> list[dict]:
        jobs = []
        for pid, mid, prob, meth in self.pairs():
            cfg = self.best(pid, mid).best_config
            B = self.cfg.budget_for(pid)
            inst = prob.instances()
            for lv in self.cfg.budget_levels:
                b = {k: v for k, v in B.scaled(lv).to_dict().items() if v is not None}
                for s in range(self.cfg.seeds):
                    tag = f"final__{pid}__{mid}__{lvl(lv)}__s{s}"
                    base = self._base(pid, mid, cfg, s, b, "final", level=lv)
                    if not meth.per_instance:
                        jobs.append({"id": tag, **base})
                        continue
                    sel = list(inst.get("final", []))
                    if lv == 1.0:
                        sel += inst.get("full_only", []) + (inst.get("seed0_only", []) if s == 0 else [])
                    for suffix, ref in sel:
                        jobs.append({"id": f"{tag}__{suffix}", **base, "instance": ref})
        return jobs

    def study_jobs(self) -> list[dict]:
        """``budget_study``: the best configuration, seed 0, on the study instances, at each listed wall clock."""
        jobs = []
        for pid, per in (self.cfg.budget_study or {}).items():
            if pid not in self.P:
                continue
            prob, base = self.P[pid], self.cfg.budget_for(pid).wall_s
            for mid, budgets in per.items():
                meth = self.M.get(mid)
                if meth is None or not meth.supports(prob):
                    continue
                cfg = self.best(pid, mid).best_config
                inst = prob.study_instances() if meth.per_instance else [("", None)]
                for B in budgets:
                    B = min(float(B), 6.0) if self.cfg.smoke else float(B)      # smoke: a pipeline check only
                    for suffix, ref in inst:
                        jid = f"study__{pid}__{mid}__B{int(B)}" + (f"__{suffix}" if suffix else "")
                        jobs.append({"id": jid, **self._base(pid, mid, cfg, 0, {"wall_s": B}, "study",
                                                             level=B / base if base else None, instance=ref)})
        return jobs

    def ablation_jobs(self) -> list[dict]:
        """``ablations``: the best configuration with each listed change, seed 0, full budget, the audit instance."""
        jobs = []
        for pid, per in (self.cfg.ablations or {}).items():
            if pid not in self.P:
                continue
            prob = self.P[pid]
            b = {k: v for k, v in self.cfg.budget_for(pid).to_dict().items() if v is not None}
            for mid, variants in per.items():
                meth = self.M.get(mid)
                if meth is None or not meth.supports(prob):
                    continue
                best = self.best(pid, mid).best_config
                suffix, ref = (prob.audit_instance() or ("", None)) if meth.per_instance else ("", None)
                for name, change in variants.items():
                    jid = f"diag__{pid}__{mid}__{name}" + (f"__{suffix}" if suffix else "")
                    jobs.append({"id": jid, **self._base(pid, mid, {**best, **change}, 0, b, "diag", variant=name,
                                                         instance=ref)})
        return jobs

    def flops_target(self, pid) -> tuple[float | None, dict]:
        used = {}
        for mid, meth in self.M.items():
            if not meth.supports(self.P[pid]):
                continue
            first = (self.P[pid].instances().get("final") or [("", None)])[0][0]
            f = []
            for s in range(self.cfg.seeds):
                tag = f"final__{pid}__{mid}__L1.000__s{s}" + (f"__{first}" if meth.per_instance and first else "")
                r = self.ws.result(tag)
                if r and r.get("ok"):
                    f.append(r["result"]["train"]["flops"])
            if f:
                used[mid] = float(np.median(f))
        return (min(used.values()) if used else None), used

    def flops_jobs(self) -> list[dict]:
        jobs, targets = [], {}
        for pid, prob in self.P.items():
            if not self.cfg.equal_flops_for(pid):
                continue
            F, used = self.flops_target(pid)
            if F is None:
                continue
            targets[pid] = {"flops": F, "median_flops_used_in_full_wall_budget": used}
            wall = self.cfg.budget_for(pid).wall_s
            for mid, meth in self.M.items():
                if not meth.supports(prob):
                    continue
                cfg = self.best(pid, mid).best_config
                first = (prob.instances().get("final") or [("", None)])[0]
                for s in range(self.cfg.seeds):
                    b = {"flops": F, **({"wall_s": 3 * wall} if wall else {})}
                    spec = {"id": f"flops__{pid}__{mid}__s{s}", **self._base(pid, mid, cfg, s, b, "flops")}
                    if meth.per_instance:
                        spec["instance"] = first[1]
                    jobs.append(spec)
        _jdump(self.ws.root / "flops_targets.json", targets)
        return jobs
