"""Benchmark harnesses as flows: the equal-budget benchmark (``examples/benchmark``, :func:`pinneapple_analysis.
benchmark.run`) and the benchmark suite of ``examples/benchmark_suite`` (:mod:`pinneapple_tools.benchmark_suite`):
the Arena experiment from the example's YAML config (02), and the transfer-learning
and meta-learning benchmarks on parametric PDE families (13). The multi-architecture arena of 12 is the existing
``benchmark.arena`` node."""
from __future__ import annotations

import json
import math
from pathlib import Path

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact
from .examples import example_dir


def _example_file(spec: str) -> Path:
    """``<example>/<relative path>`` -> a file inside that example folder."""
    ex, _, rel = spec.partition("/")
    root = example_dir(ex)
    p = (root / rel).resolve()
    if root not in p.parents or not p.is_file():
        raise StepFailed(f"{spec!r} is not a file of examples/{ex}")
    return p


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _figures(ctx, paths, caption=""):
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".png" and p.is_file():
            ctx.report["figures"].append({"path": str(p), "name": p.stem, "caption": caption or p.stem})


# ------------------------------------------------------------------------------------------------ examples/benchmark
@node("benchmark.equal_budget", "Equal-budget benchmark (pp.benchmark.run)", category="Validation",
      outputs=[Port("metrics", "metrics"), Port("report", "report", required=False)],
      params=[Param("problems", "list", ["burgers"], help="pp.benchmark.problems()"),
              Param("methods", "list", ["pinn", "fno", "deeponet"], help="pp.benchmark.methods()"),
              Param("wall_s", "float", 300.0, minimum=0.1, help="ComputeBudget: wall-clock seconds per training run"),
              Param("budget_levels", "list", [1 / 9, 1 / 3, 1.0], help="fractions of the budget (error-vs-budget)"),
              Param("seeds", "int", 3, minimum=1), Param("n_trials", "int", 4, minimum=1),
              Param("search_fraction", "float", 1 / 3, minimum=0.0, maximum=1.0),
              Param("equal_flops", "bool", True, help="the extra equal-FLOPs arm"),
              Param("n_workers", "int", 4, minimum=1), Param("threads_per_run", "int", 1, minimum=1),
              Param("smoke", "bool", False, help="tiny data and budgets: a pipeline check in minutes")],
      calls="pinneapple_analysis.benchmark.run (pp.benchmark.run), ComputeBudget, BenchmarkResult.table / "
            "diagnostics / report")
def equal_budget(ctx, inputs, params):
    """Every method on every problem under the same ComputeBudget, the same hyper-parameter search effort and
    seeds; the result table (mean over seeds at the full budget), the correctness gates and the advisor's
    diagnostics. The run folder is the node's (config.yaml, prepared/, runs/, table.md, summary.json, figures/)."""
    import pinneapple as pp
    out = ctx.path("benchmark")
    res = pp.benchmark.run(problems=list(params["problems"]), methods=list(params["methods"]),
                           budget=pp.ComputeBudget(wall_s=float(params["wall_s"])),
                           budget_levels=[float(x) for x in params["budget_levels"]], seeds=int(params["seeds"]),
                           n_trials=int(params["n_trials"]), search_fraction=float(params["search_fraction"]),
                           equal_flops=bool(params["equal_flops"]), n_workers=int(params["n_workers"]),
                           threads_per_run=int(params["threads_per_run"]), smoke=bool(params["smoke"]),
                           out_dir=str(out), log=ctx.log)
    df = res.table()
    cols = [str(c) for c in df.columns]
    ctx.table("benchmark table (mean over seeds, full budget)", cols,
              [[(_num(v) if _num(v) is not None else (None if v is None else str(v))) for v in row]
               for row in df.itertuples(index=False)])
    for row in df.to_dict("records"):
        key = f"{row['problem']}.{row['method']}"
        ctx.metric(f"{key}.status", row["status"])
        for c, v in row.items():
            if c not in ("problem", "method", "status") and _num(v) is not None:
                ctx.metric(f"{key}.{c}", _num(v))
    diags = res.diagnostics
    ctx.table("diagnostics and suggestions", ["severity", "subject", "finding", "suggestion"],
              [[d.severity, d.subject, d.finding, d.suggestion] for d in diags])
    for d in diags:
        if d.severity == "error":
            ctx.warn(f"[{d.subject}] {d.finding}")
    for b in res.banners:
        ctx.warn(b)
    summ = json.loads((out / "summary.json").read_text())
    _figures(ctx, summ.get("figures") or [])
    ctx.summary(f"{len(df)} rows ({', '.join(params['methods'])} on {', '.join(params['problems'])}), "
                f"{len(diags)} diagnostics" + (" (smoke sizes)" if params["smoke"] else ""))
    return {"metrics": ctx.report["metrics"], "report": artifact(out / "summary.json", "json")}


# ------------------------------------------------------------------------------------------------ benchmark_suite
def _patched_yaml(ctx, spec: str, epochs: int, models_key="models", train_key="train"):
    import yaml
    cfg = yaml.safe_load(_example_file(spec).read_text())
    if epochs:
        for m in cfg.get(models_key, []):
            m.setdefault(train_key, {})["epochs"] = int(epochs)
    return cfg


@node("benchmark.arena_yaml", "Arena experiment from a YAML config (benchmark suite)", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("config", "str", "benchmark_suite/configs/experiment_burgers_1d.yaml",
                    help="<example>/<path> of an Arena experiment YAML"),
              Param("epochs", "int", 0, minimum=0, help="override every model's epochs (0: as in the file)")],
      calls="pinneapple_tools.benchmark_suite.runner.run_arena_yaml.run_arena_experiment")
def arena_yaml(ctx, inputs, params):
    """The problem preset, its reference data from the built-in solver, every model of the config trained on the
    same points; per model the best validation loss, the metrics on the reference points and the time."""
    import yaml

    from pinneapple_tools.benchmark_suite.runner.run_arena_yaml import run_arena_experiment
    cfg = _patched_yaml(ctx, params["config"], int(params["epochs"]))
    p = ctx.path("experiment.yaml")
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    res = run_arena_experiment(p, out_dir=ctx.path("arena"))
    rows = []
    for mid, m in res["summary"]["models"].items():
        ev = m.get("eval_metrics") or {}
        ctx.metric(f"{mid}.best_val", _num(m.get("best_val")))
        ctx.metric(f"{mid}.elapsed_s", _num(m.get("elapsed_sec")))
        for k, v in ev.items():
            if _num(v) is not None:
                ctx.metric(f"{mid}.{k}", _num(v))
        rows.append([mid, _num(m.get("best_val"))] + [_num(ev.get(k)) for k in ("mse", "rel_l2", "r2")] +
                    [round(float(m.get("elapsed_sec") or 0), 1)])
    ctx.table(res["experiment_name"], ["model", "best val", "mse", "rel_l2", "r2", "seconds"], rows)
    _figures(ctx, (res.get("visualizations") or {}).values())
    ctx.summary(f"{len(rows)} models trained on {res['summary'].get('problem_id')}")
    return {"metrics": ctx.report["metrics"]}


_SCEN = ["burgers_nu", "heat_alpha", "wave_c"]


@node("benchmark.transfer", "Transfer-learning benchmark on parametric PDE families", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("scenarios", "list", _SCEN, help=f"from {_SCEN}"),
              Param("epochs_source", "int", 3000, minimum=1), Param("epochs_finetune", "int", 1000, minimum=1)],
      calls="pinneapple_tools.benchmark_suite.transfer_benchmark.TransferBenchmarkPipeline.default / run")
def transfer(ctx, inputs, params):
    """Pre-train on a source member of each family, fine-tune on targets with each freezing strategy (and from
    scratch); relative L2 per scenario, target and strategy."""
    from pinneapple_tools.benchmark_suite.transfer_benchmark import TransferBenchmarkPipeline
    bad = [s for s in params["scenarios"] if s not in _SCEN]
    if bad:
        raise StepFailed(f"unknown scenarios {bad}; known: {_SCEN}")
    pipe = TransferBenchmarkPipeline.default(scenarios=list(params["scenarios"]),
                                             epochs_source=int(params["epochs_source"]),
                                             epochs_finetune=int(params["epochs_finetune"]), device="cpu")
    res = pipe.run(verbose=False)
    rows = []
    for r in res:
        k = f"{r.scenario_name}.{r.target_label.split(' ')[0]}.{r.strategy}"
        for m, v in r.metrics.items():
            if _num(v) is not None:
                ctx.metric(f"{k}.{m}", _num(v))
        rows.append([r.scenario_name, r.target_label, r.strategy, _num(r.metrics.get("rel_l2")),
                     r.n_params_trainable, round(r.elapsed_s, 2)])
    ctx.table("transfer learning", ["scenario", "target", "strategy", "rel. L2", "trainable", "seconds"], rows)
    ctx.summary(f"{len(rows)} fine-tuning runs over {len(params['scenarios'])} scenarios")
    return {"metrics": ctx.report["metrics"]}


@node("benchmark.meta", "Meta-learning benchmark (MAML / Reptile, K-shot adaptation)", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("families", "list", _SCEN, help=f"from {_SCEN}"),
              Param("n_meta_epochs", "int", 500, minimum=1),
              Param("k_shots", "list", [1, 5, 10, 20, 50, 100, 200], help="adaptation steps evaluated")],
      calls="pinneapple_tools.benchmark_suite.meta_benchmark.MetaBenchmarkPipeline.default / run")
def meta(ctx, inputs, params):
    """Meta-train on each PDE family, adapt K steps to held-out members against training from scratch; relative
    L2 per family, algorithm and K."""
    from pinneapple_tools.benchmark_suite.meta_benchmark import MetaBenchmarkPipeline
    bad = [s for s in params["families"] if s not in _SCEN]
    if bad:
        raise StepFailed(f"unknown families {bad}; known: {_SCEN}")
    pipe = MetaBenchmarkPipeline.default(families=list(params["families"]), n_meta_epochs=int(params["n_meta_epochs"]),
                                         device="cpu")
    pipe.config.k_shots = tuple(int(k) for k in params["k_shots"])
    res = pipe.run(verbose=False)
    agg: dict[tuple, list] = {}
    for r in res:
        agg.setdefault((r.family_name, r.algorithm, r.k_shot), []).append(_num(r.metrics.get("rel_l2")))
    rows = []
    for (fam, alg, k), vs in sorted(agg.items()):
        vs = [v for v in vs if v is not None]
        mean = sum(vs) / len(vs) if vs else None
        ctx.metric(f"{fam}.{alg}.k{k}.rel_l2", mean, "mean over the family's evaluation members")
        rows.append([fam, alg, k, mean, len(vs)])
    ctx.table("meta-learning (mean relative L2 over the evaluation members)",
              ["family", "algorithm", "K", "rel. L2", "members"], rows)
    ctx.summary(f"{len(res)} adaptation runs over {len(params['families'])} families")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="benchmark.equal_budget", title="Equal-budget benchmark of PINN, FNO and DeepONet", domain="benchmarks",
        example="benchmark",
        keywords={"equal budget": 4, "equal-budget": 4, "equal compute": 3, "equal flops": 3,
                  "benchmark harness": 3, "same budget": 2},
        min_score=3, nodes=("benchmark.equal_budget",),
        uncited="the benchmark protocol of docs/guides/running_a_benchmark.md",
        template="benchmark/01_equal_budget", quick=True, description="examples/benchmark/01_equal_budget.py"))
    register_capability(Capability(
        id="benchmark.arena_yaml", title="Arena experiment from a YAML config", domain="benchmarks",
        example="benchmark_suite",
        keywords={"arena experiment": 4, "yaml config": 2, "mlp sizes": 3, "compare mlp": 3},
        min_score=4, nodes=("benchmark.arena_yaml",), references=("raissi2019pinn",),
        template="benchmark_suite/02_arena_from_yaml", quick=True,
        description="examples/benchmark_suite/02_arena_from_yaml.py"))
    register_capability(Capability(
        id="benchmark.arena_suite", title="PINN arena: benchmark architectures on six physics problems", domain="benchmarks",
        example="benchmark_suite",
        keywords={"pinn arena": 4, "architectures": 1, "benchmark suite": 3, "rank architectures": 3,
                  "multi-architecture": 3},
        min_score=4, nodes=("benchmark.arena",), references=("raissi2019pinn",),
        template="benchmark_suite/12_physics_benchmark_suite", quick=True,
        description="examples/benchmark_suite/12_physics_benchmark_suite.py"))
    register_capability(Capability(
        id="benchmark.transfer_meta", title="Transfer learning and meta-learning on parametric PDE families",
        domain="benchmarks", example="benchmark_suite",
        keywords={"transfer learning": 3, "meta-learning": 3, "meta learning": 3, "maml": 3, "reptile": 3,
                  "k-shot": 3, "fine-tune": 1},
        min_score=5, nodes=("benchmark.transfer", "benchmark.meta"), references=("finn2017maml", "nichol2018reptile"),
        template="benchmark_suite/13_transfer_meta_benchmark", quick=True,
        description="examples/benchmark_suite/13_transfer_meta_benchmark.py"))


_register_capabilities()
