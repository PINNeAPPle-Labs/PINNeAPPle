"""The Arena pipelines of ``examples/arena_pipelines``: the dataset registry (00), the physics and time-series
benchmark specs (01, 02, :mod:`pinneapple_tools.benchmark_suite`) and the multi-model Arena run from a config
(04, 05, :class:`pinneapple_arena.Arena`). Each run writes its figures and tables into the node's folder."""
from __future__ import annotations

import copy
from pathlib import Path

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact
from .benchmarks import _example_file, _figures, _num


@node("arena.datasets", "Dataset registry: list and load the datasets", category="Data",
      outputs=[Port("metrics", "metrics")],
      params=[Param("ids", "list", ["burgers_1d", "heat_1d", "lorenz63", "spring_mass_1dof", "naca0012",
                                    "cylinder_2d"], help="datasets loaded and described"),
              Param("categories", "list", ["physics", "geometry", "timeseries"])],
      calls="pinneapple_data.datasets.dataset_ids, load_dataset")
def datasets(ctx, inputs, params):
    """The dataset ids per category and, for each requested id, the arrays it loads (or the error)."""
    from pinneapple_data.datasets import dataset_ids, load_dataset
    for cat in params["categories"]:
        ids = list(dataset_ids(cat))
        ctx.metric(f"n_{cat}", len(ids))
        ctx.table(f"{cat} datasets", ["id"], [[i] for i in ids])
    rows, ok = [], 0
    for did in params["ids"]:
        ctx.check_stop()
        try:
            data = load_dataset(did)
        except Exception as exc:                  # a dataset that needs the network or an optional library
            rows.append([did, "error", str(exc)[:120]])
            continue
        ok += 1
        keys = ", ".join(f"{k}:{tuple(v.shape) if hasattr(v, 'shape') else len(v)}" for k, v in data.items()
                         if k not in ("description", "source", "pde") and hasattr(v, "__len__"))
        rows.append([did, "ok", keys])
    ctx.metric("loaded", ok)
    ctx.metric("failed", len(rows) - ok)
    ctx.table("load test", ["dataset", "status", "arrays"], rows)
    ctx.summary(f"{ok} of {len(rows)} datasets loaded")
    return {"metrics": ctx.report["metrics"]}


def _spec_report(ctx, rep, label):
    rows = []
    for mid, r in rep.model_results.items():
        if r.error_message:
            ctx.warn(f"{label}/{mid}: {r.error_message}")
        for k, v in (r.metrics or {}).items():
            if _num(v) is not None:
                ctx.metric(f"{mid}.{k}", _num(v))
        ctx.metric(f"{mid}.training_time_s", _num(r.training_time_s))
        for k, v in (r.param_estimates or {}).items():
            if _num(v) is not None:
                ctx.metric(f"{mid}.estimate.{k}", _num(v))
        rows.append([mid, r.rank, r.n_params, round(float(r.training_time_s or 0), 1)] +
                    [", ".join(f"{k}={v:.3g}" for k, v in (r.metrics or {}).items() if _num(v) is not None)])
    ctx.table(f"{label}: models", ["model", "rank", "parameters", "seconds", "metrics"], rows)
    if rep.best_model:
        ctx.metric("best_model", rep.best_model)
    _figures(ctx, rep.plots_saved or [])
    p = rep.save(ctx.path(f"{label}_report.json"))
    ctx.summary(f"{label}: {len(rows)} models, best {rep.best_model}")
    return {"metrics": ctx.report["metrics"], "report": artifact(p, "json")}


_SPEC_OUT = [Port("metrics", "metrics"), Port("report", "report", required=False)]


@node("arena.physics_spec", "Physics benchmark spec (PINN models on a physics problem)", category="Validation",
      outputs=_SPEC_OUT,
      params=[Param("label", "str", "burgers_1d", help="name of the run (metric prefix)"),
              Param("spec", "dict", {"problem": "burgers_1d", "models": ["vanilla_pinn"], "epochs": 1500},
                    help="keyword arguments of PhysicsBenchmarkSpec (problem, load_generate_data, source, metrics, "
                         "collocation_points, models, inverse, inverse_variables, epochs, lr, n_col, n_bc, n_ic, "
                         "hidden, seed, plots)")],
      calls="pinneapple_tools.benchmark_suite.PhysicsBenchmarkSpec (physics_pipeline.PhysicsPipeline).run")
def physics_spec(ctx, inputs, params):
    """Every model of the spec on the problem (forward or inverse), metrics against the reference, a leaderboard
    and, for an inverse run, the identified parameters."""
    from pinneapple_tools.benchmark_suite import PhysicsBenchmarkSpec
    kw = dict(params["spec"])
    kw["output_dir"] = str(ctx.path("physics"))
    try:
        spec = PhysicsBenchmarkSpec(**kw)
    except TypeError as exc:
        raise StepFailed(f"spec: {exc}") from None
    return _spec_report(ctx, spec.run(), params["label"])


@node("arena.timeseries_spec", "Time-series benchmark spec (forecasting models on a series)", category="Time series",
      outputs=_SPEC_OUT,
      params=[Param("label", "str", "lorenz63", help="name of the run (metric prefix)"),
              Param("spec", "dict", {"source": "lorenz63", "models": ["lstm", "tcn", "nbeats"], "horizon": 20,
                                     "lookback": 50, "epochs": 60},
                    help="keyword arguments of TimeSeriesBenchmarkSpec (source, models, metrics, horizon, lookback, "
                         "test_size, target_cols, epochs, lr, seed, plots)")],
      calls="pinneapple_tools.benchmark_suite.TimeSeriesBenchmarkSpec.run")
def timeseries_spec(ctx, inputs, params):
    """Every forecasting model of the spec on the same lookback / horizon split, metrics on the test part."""
    from pinneapple_tools.benchmark_suite import TimeSeriesBenchmarkSpec
    kw = dict(params["spec"])
    kw["output_dir"] = str(ctx.path("timeseries"))
    try:
        spec = TimeSeriesBenchmarkSpec(**kw)
    except TypeError as exc:
        raise StepFailed(f"spec: {exc}") from None
    return _spec_report(ctx, spec.run(), params["label"])


@node("arena.run", "Arena: several models on one physics problem (PINN, SIREN, FNO, MeshGraphNet, ...)",
      category="Validation", outputs=[Port("metrics", "metrics")],
      params=[Param("config", "dict", {}, help="an ArenaConfig as a dict (problem, models, output); used when "
                                               "config_file is empty"),
              Param("config_file", "str", "", help="<example>/<path> of an Arena YAML or JSON config"),
              Param("epochs", "int", 0, minimum=0, help="override every model's epochs (0: as configured)"),
              Param("physics_aware", "bool", False, help="also rank with the physics residual (flag outliers)")],
      calls="pinneapple_arena.Arena (ArenaConfig.from_dict / from_yaml / from_json).run, physics_aware_summary")
def arena_run(ctx, inputs, params):
    """Data from the problem's reference, every model trained and evaluated on the same points; per model the
    L2 and relative error of each field, the training time and (PINN family) the final physics residual."""
    import json

    import yaml

    from pinneapple_arena import Arena, ArenaConfig
    if params["config_file"]:
        p = _example_file(params["config_file"])
        d = yaml.safe_load(p.read_text()) if p.suffix.lower() in (".yaml", ".yml") else json.loads(p.read_text())
    elif params["config"]:
        d = copy.deepcopy(params["config"])
    else:
        raise StepFailed("give a config or a config_file")
    if int(params["epochs"]):
        for m in d.get("models", []):
            m.setdefault("training", {})["epochs"] = int(params["epochs"])
    out = ctx.path("arena")
    d.setdefault("output", {})["dir"] = str(out)
    d["output"]["show"] = False
    arena = Arena(ArenaConfig.from_dict(d), device="cpu")
    arena.run(physics_aware=bool(params["physics_aware"]))
    rows = []
    fields = None
    for tres, eres in zip(arena.train_results, arena.results, strict=True):
        m = eres["metrics"]
        fields = fields or sorted({k.split("_", 1)[1] for k in m if k.startswith("rel_")})
        for k, v in m.items():
            if _num(v) is not None:
                ctx.metric(f"{tres.name}.{k}", _num(v))
        ctx.metric(f"{tres.name}.train_time_s", _num(tres.train_time))
        if tres.physics_residual is not None:
            ctx.metric(f"{tres.name}.physics_residual", _num(tres.physics_residual),
                       "the PDE-residual loss term at the end of training")
        rows.append([tres.name] + [_num(m.get(f"rel_{f}")) for f in fields] +
                    [round(float(tres.train_time), 1), _num(tres.physics_residual)])
    ctx.table(f"Arena: {d['problem']['name']}", ["model"] + [f"rel. L2 {f}" for f in fields or []] +
              ["seconds", "physics residual"], rows)
    if params["physics_aware"]:
        pa = arena.physics_aware_summary()
        if pa.get("flagged_high_residual"):
            ctx.warn(f"high physics residual: {pa['flagged_high_residual']}")
    _figures(ctx, sorted(Path(out).rglob("*.png")))
    ctx.summary(f"{len(rows)} models on {d['problem']['name']}")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="arena.datasets", title="List and load the dataset registry", domain="benchmarks",
        example="arena_pipelines",
        keywords={"dataset registry": 4, "list datasets": 4, "available datasets": 3, "load datasets": 3},
        min_score=3, nodes=("arena.datasets",), uncited="a listing of pinneapple_data's registry",
        template="arena_pipelines/00_list_datasets", description="examples/arena_pipelines/00_list_datasets.py"))
    register_capability(Capability(
        id="arena.physics_spec", title="Physics benchmark spec: PINNs on Burgers, heat (forward and inverse), "
                                       "Kovasznay", domain="benchmarks", example="arena_pipelines",
        keywords={"physics benchmark spec": 5, "physicsbenchmarkspec": 5, "leaderboard": 2,
                  "recover diffusivity": 3},
        min_score=4, nodes=("arena.physics_spec",), references=("raissi2019pinn",),
        template="arena_pipelines/01_physics_benchmark", quick=True,
        description="examples/arena_pipelines/01_physics_benchmark.py"))
    register_capability(Capability(
        id="arena.timeseries_spec", title="Time-series benchmark spec: LSTM, TCN, N-BEATS, GRU, Transformer",
        domain="benchmarks", example="arena_pipelines",
        keywords={"time-series benchmark": 4, "timeseries benchmark": 4, "lstm": 2, "tcn": 2, "nbeats": 3,
                  "n-beats": 3, "sunspots": 3, "air passengers": 3},
        min_score=5, nodes=("arena.timeseries_spec",), references=("hyndman2021fpp",),
        template="arena_pipelines/02_timeseries_benchmark", quick=True,
        description="examples/arena_pipelines/02_timeseries_benchmark.py"))
    register_capability(Capability(
        id="arena.run", title="Arena: PINN, SIREN, FNO and MeshGraphNet on one physics problem",
        domain="benchmarks", example="arena_pipelines",
        keywords={"arena": 3, "meshgraphnet": 1, "siren": 1, "fno": 1, "compare models": 2, "kovasznay": 1},
        min_score=5, nodes=("arena.run",), references=("raissi2019pinn", "li2021fno", "pfaff2021meshgraphnet"),
        template="arena_pipelines/04_arena_from_yaml", quick=True,
        description="examples/arena_pipelines/04_arena_from_yaml.py"))


_register_capabilities()
