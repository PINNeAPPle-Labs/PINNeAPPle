"""Physics world model (``examples/worldmodel/01_generate_train_evaluate``): trajectory datasets from the
built-in scenarios (:func:`pinneapple_worldmodel.generate_datasets`), one specialist FNO world model per scenario
(:func:`pinneapple_worldmodel.train_specialist`) and the library's benchmark tasks
(:class:`pinneapple_worldmodel.benchmark.PhysicsBenchmark`)."""
from __future__ import annotations

import time

from ..registry import Param, Port, node


@node("worldmodel.specialists", "Physics world model: generate, train specialists, benchmark",
      category="Neural operators", outputs=[Port("metrics", "metrics")],
      params=[Param("scenarios", "list", ["burgers_1d", "heat_2d"]), Param("n_train", "int", 80, minimum=1),
              Param("epochs", "int", 30, minimum=1), Param("n_modes", "int", 8, minimum=1),
              Param("width", "int", 32, minimum=4), Param("depth", "int", 3, minimum=1),
              Param("batch_size", "int", 16, minimum=1), Param("lr", "float", 1e-3)],
      calls="pinneapple_worldmodel.generate_datasets + pinneapple_worldmodel.train_specialist + "
            "pinneapple_worldmodel.benchmark.PhysicsBenchmark (BUILTIN_TASKS of the trained scenarios)")
def specialists(ctx, inputs, params):
    """Per scenario: parameters of the specialist and, per benchmark task of that scenario, next-step and rollout
    RMSE, relative L2, the divergence step and the overall score."""
    from pinneapple_worldmodel import generate_datasets, train_specialist
    from pinneapple_worldmodel.benchmark import BUILTIN_TASKS, PhysicsBenchmark
    scen = [str(s) for s in params["scenarios"]]
    t0 = time.perf_counter()
    cat = generate_datasets(scenarios=scen, sources=["solver"], n_samples=int(params["n_train"]),
                            output_dir=str(ctx.path("data")), device="cpu", validate=False, verbose=False)
    ctx.metric("generate_seconds", round(time.perf_counter() - t0, 1))
    t0 = time.perf_counter()
    epochs = int(params["epochs"])
    zoo = train_specialist(catalog=cat, output_dir=ctx.path("checkpoints"), epochs=epochs,
                           batch_size=int(params["batch_size"]), lr=float(params["lr"]), device="cpu",
                           patience=max(epochs // 3, 5), n_modes=int(params["n_modes"]), width=int(params["width"]),
                           depth=int(params["depth"]), rollout_steps=1, run_benchmark=False)
    ctx.metric("train_seconds", round(time.perf_counter() - t0, 1))
    bench = PhysicsBenchmark(device="cpu", verbose=False)
    rows = []
    for name in scen:
        try:
            entry = zoo.get_entry(name)
        except KeyError:
            ctx.warn(f"no specialist was trained for {name}")
            continue
        ctx.metric(f"{name}.n_params", int(entry.n_params))
        tasks = [k for k, t in BUILTIN_TASKS.items() if t.scenario_name == name]
        for task, r in bench.run(entry.model, tasks=tasks, model_name=name).items():
            for k in ("next_step_rmse", "rollout_rmse", "rel_l2", "div_step"):
                ctx.metric(f"{name}.{task}.{k}", float(getattr(r, k)))
            ctx.metric(f"{name}.{task}.score", float(r.overall_score()))
            rows.append([name, task, round(r.next_step_rmse, 5), round(r.rollout_rmse, 5), round(r.rel_l2, 4),
                         r.div_step, round(r.overall_score(), 3)])
    ctx.table("benchmark", ["scenario", "task", "next-step RMSE", "rollout RMSE", "rel. L2", "div. step", "score"], rows)
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="worldmodel.specialists", title="A physics world model: specialist FNOs trained on simulated trajectories",
        domain="neural operators", example="worldmodel",
        keywords={"world model": 4, "worldmodel": 4, "specialist": 2, "trajectories": 1, "fno": 1},
        requires_all=(("world model", "worldmodel"),), min_score=3, nodes=("worldmodel.specialists",),
        references=("li2021fno",), template="worldmodel/01_generate_train_evaluate", quick=True,
        description="examples/worldmodel/01_generate_train_evaluate.py"))


_register_capabilities()
