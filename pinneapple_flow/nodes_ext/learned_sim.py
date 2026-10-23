"""Learned mesh simulators on public data (``examples/meshgraphnet/02_cylinder_flow_deepmind``): MeshGraphNet on a
byte prefix of DeepMind's ``cylinder_flow`` TFRecords (:mod:`pinneapple_data.datasets.deepmind_mgn`), trained and
rolled out with :mod:`pinneapple_neural.architectures.graphnn.mgn_dynamics`."""
from __future__ import annotations

import time
from pathlib import Path

from ..registry import Param, Port, node
from .examples import examples_root


def _tfrecord_available():
    try:
        import tfrecord  # noqa: F401
    except ImportError:
        return False, "needs the tfrecord reader (pip install tfrecord)"
    return True, ""


@node("gnn.mgn_cylinder_flow", "MeshGraphNet on DeepMind cylinder flow (vortex shedding)", category="Neural operators",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("train_mb", "int", 400, minimum=1, help="byte prefix of train.tfrecord (whole trajectories)"),
              Param("valid_mb", "int", 60, minimum=1), Param("steps", "int", 3000, minimum=1),
              Param("hidden", "int", 64, minimum=4), Param("message_passing", "int", 6, minimum=1),
              Param("rollout", "int", 50, minimum=1), Param("lr", "float", 1e-3), Param("lr_final", "float", 1e-5),
              Param("noise", "float", 0.003), Param("seed", "int", 0),
              Param("data_dir", "str", "examples/meshgraphnet/_data/cylinder_flow",
                    help="download cache, relative to the repository (the example's own)")],
      calls="pinneapple_data.datasets.deepmind_mgn.download_prefix / read_trajectories / cylinder_flow_to_tensors + "
            "pinneapple_neural.architectures.graphnn.mgn_dynamics.MeshDynamicsMGN / train_dynamics / eval_rollout / "
            "frozen_ic_rmse", available=_tfrecord_available)
def mgn_cylinder_flow(ctx, inputs, params):
    """Next-step velocity of vortex shedding behind a cylinder on irregular triangle meshes, rolled out
    autoregressively on the validation trajectories; against the velocity frozen at t = 0."""
    import torch

    from pinneapple_data.datasets.deepmind_mgn import (
        CYLINDER_TYPE_MAP,
        cylinder_flow_to_tensors,
        download_prefix,
        read_trajectories,
    )
    from pinneapple_neural.architectures.graphnn.mgn_dynamics import (
        MeshDynamicsMGN,
        eval_rollout,
        frozen_ic_rmse,
        train_dynamics,
    )
    from pinneapple_neural.trainer.parallel import best_device
    torch.manual_seed(int(params["seed"]))
    dev = best_device(allow_mps=False)
    data_dir = Path(str(params["data_dir"]))
    if not data_dir.is_absolute():
        data_dir = examples_root().parent / data_dir
    download_prefix("cylinder_flow", "train", data_dir, int(params["train_mb"]) * 1_000_000)
    download_prefix("cylinder_flow", "valid", data_dir, int(params["valid_mb"]) * 1_000_000)
    tr = [cylinder_flow_to_tensors(t, CYLINDER_TYPE_MAP) for t in read_trajectories(data_dir, "train", 10_000)]
    va = [cylinder_flow_to_tensors(t, CYLINDER_TYPE_MAP) for t in read_trajectories(data_dir, "valid", 10_000)]
    model = MeshDynamicsMGN(vel_dim=2, num_node_types=4, edge_dim=3, hidden_dim=int(params["hidden"]), n_layers=2,
                            n_message_passing=int(params["message_passing"]), activation="relu")
    model.fit_stats([v for _, v, _ in tr], [p for _, _, p in tr], [g for g, _, _ in tr])
    t0 = time.perf_counter()
    hist = train_dynamics(model, tr, steps=int(params["steps"]), lr=float(params["lr"]),
                          lr_final=float(params["lr_final"]), device=dev, noise_std=float(params["noise"]),
                          seed=int(params["seed"]))
    ctx.metric("train_seconds", round(time.perf_counter() - t0, 1))
    m = eval_rollout(model, va, n_steps=int(params["rollout"]), device=dev)
    ctx.metric("rollout_rmse", m["rollout_rmse"], f"{m['n_steps']}-step free rollout, NORMAL nodes")
    ctx.metric("one_step_rmse", m["one_step_rmse"])
    ctx.metric("frozen_ic_baseline_rmse", frozen_ic_rmse(va, n_steps=int(params["rollout"])), "velocity frozen at t = 0")
    ctx.metric("train_traj", len(tr))
    ctx.metric("valid_traj", len(va))
    ctx.metric("params", sum(p.numel() for p in model.parameters()))
    if hist:
        ctx.metric("final_train_loss", float(hist[-1]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    g, v, _ = va[0]
    n = min(int(params["rollout"]), v.shape[0] - 1)
    pred, _ = model.rollout(v.to(dev), g.to(dev), n)
    raw = read_trajectories(data_dir, "valid", 1)[0]
    tri = mtri.Triangulation(raw["mesh_pos"][:, 0], raw["mesh_pos"][:, 1], raw["cells"])
    fig, ax = plt.subplots(3, 1, figsize=(8, 7))
    for k, (title, f) in enumerate([("ground truth |v|", v[n].norm(dim=-1)),
                                    ("MeshGraphNet |v|", pred[n].detach().cpu().norm(dim=-1)),
                                    ("abs. error", (pred[n].detach().cpu() - v[n]).norm(dim=-1))]):
        c = ax[k].tripcolor(tri, f.numpy(), shading="gouraud")
        ax[k].set_title(f"{title} (step {n})")
        ax[k].set_aspect("equal")
        fig.colorbar(c, ax=ax[k])
    fig.tight_layout()
    return {"metrics": ctx.report["metrics"], "figure": ctx.figure(fig, "rollout", "velocity magnitude after the rollout")}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="gnn.mgn_cylinder_flow", title="MeshGraphNet on DeepMind's cylinder_flow", domain="neural operators",
        example="meshgraphnet",
        keywords={"meshgraphnet": 3, "cylinder_flow": 3, "cylinder flow": 2, "vortex shedding": 2, "deepmind": 2,
                  "graph network": 1.5, "mesh": 0.5},
        requires_all=(("meshgraphnet", "graph network", "deepmind", "cylinder_flow"),), min_score=3.5,
        nodes=("gnn.mgn_cylinder_flow", "examples.compare"), references=("pfaff2021meshgraphnet",),
        template="meshgraphnet/02_cylinder_flow_deepmind", quick=True,
        description="examples/meshgraphnet/02_cylinder_flow_deepmind.py"))


_register_capabilities()
