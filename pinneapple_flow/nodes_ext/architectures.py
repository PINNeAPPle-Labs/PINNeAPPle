"""The model registry examples (``examples/architectures``): the registry and catalog listing (00, 01, the Group-B
check of 60), building registry models by name with a forward pass on dummy tensors (00, 60) and POD compression
with a DMD rollout on a synthetic linear system (30)."""
from __future__ import annotations

from ..registry import Param, Port, node


def _registry():
    from pinneapple_models import ModelRegistry, register_all
    register_all()
    return ModelRegistry


def _short(s, n: int = 90) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "..."


@node("architectures.registry", "Model registry and catalog listing", category="Learning",
      outputs=[Port("metrics", "metrics")],
      params=[Param("families", "list", [], help="families whose full model list is shown (empty: every model)"),
              Param("specs", "list", ["fno", "vanilla_pinn", "pod", "gnn", "informer"],
                    help="models whose registry spec (family, tags, input kind) is shown"),
              Param("catalog", "list", [], help="ModelCatalog families to list (first 10 names each), e.g. "
                                                "[autoencoders, pinns, neural_operators, rom, graphnn, transformers]")],
      calls="pinneapple_models.register_all + ModelRegistry.families / list / spec + ModelCatalog")
def registry_listing(ctx, inputs, params):
    """The registry's families and models, a few model specs and the catalog by family. This is what the registry
    tour and the model list print."""
    reg = _registry()
    fams = list(reg.families())
    names = list(reg.list())
    ctx.metric("families", len(fams))
    ctx.metric("total_models", len(names))
    ctx.table("families", ["family", "models"], [[f, len(reg.list(family=f))] for f in fams])
    wanted = [str(f) for f in params["families"]]
    if not wanted:
        ctx.table("every registered model", ["model"], [[n] for n in names])
    for f in wanted:
        ms = list(reg.list(family=f))
        ctx.metric(f"{f}.models", len(ms))
        ctx.table(f"family {f}", ["model"], [[n] for n in ms])
    rows = []
    for name in params["specs"]:
        try:
            s = reg.spec(str(name))
        except Exception as exc:                       # the registry raises KeyError (or its own) for unknown names
            ctx.warn(f"{name}: not found ({exc})")
            continue
        rows.append([s.name, s.family, ", ".join(s.tags), s.input_kind, _short(s.description)])
    if rows:
        ctx.table("specs", ["model", "family", "tags", "input kind", "description"], rows)
    if params["catalog"]:
        from pinneapple_models import ModelCatalog
        cat = ModelCatalog()
        ctx.table("catalog (first 10 per family)", ["family", "models"],
                  [[f, ", ".join(getattr(cat, str(f)).list()[:10])] for f in params["catalog"]])
    ctx.summary(f"{len(names)} models in {len(fams)} families")
    return {"metrics": ctx.report["metrics"]}


@node("architectures.forward", "Build a registry model and run a forward pass", category="Learning",
      outputs=[Port("metrics", "metrics")],
      params=[Param("model", "str", "fno", help="registry name (ModelRegistry.list())"),
              Param("arguments", "dict", {}, help="constructor arguments, e.g. {in_channels: 2, width: 32}"),
              Param("inputs", "list", [[4, 1, 64]], help="one shape per positional input tensor"),
              Param("distribution", "enum", "randn", choices=["randn", "rand"],
                    help="dummy inputs: standard normal or uniform on [0, 1)"),
              Param("fit", "bool", False, help="call model.fit(first input) first (POD, DMD, ...)"),
              Param("return_loss", "bool", False, help="forward with return_loss=True and report its losses"),
              Param("input_gradient", "bool", False, help="backpropagate sum(y) and report the input-gradient norm"),
              Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_models.ModelRegistry.build + model(*inputs) (+ model.fit)")
def forward(ctx, inputs, params):
    """One registry model built by name and run on dummy tensors. Reports the output shape, the parameter count,
    the losses it returns and, on request, the input-gradient norm (gradient flow for PINN use)."""
    import torch

    from pinneapple_core.workflows import StepFailed
    reg = _registry()
    torch.manual_seed(int(params["seed"]))
    try:
        m = reg.build(str(params["model"]), **dict(params["arguments"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise StepFailed(f"{params['model']}: {exc}") from None
    gen = torch.randn if params["distribution"] == "randn" else torch.rand
    xs = [gen(*[int(v) for v in shape]) for shape in params["inputs"]]
    if not xs:
        raise StepFailed("give at least one input shape")
    if params["fit"]:
        m.fit(xs[0])
    if params["input_gradient"]:
        for x in xs:
            x.requires_grad_(True)
    kw = {"return_loss": True} if params["return_loss"] else {}
    out = m(*xs, **kw)
    y = out.y if hasattr(out, "y") else out
    ctx.metric("parameters", int(sum(p.numel() for p in m.parameters())))
    ctx.metric("input_shapes", " ".join(str(tuple(x.shape)) for x in xs))
    ctx.metric("output_shape", str(tuple(y.shape)))
    for k, v in (getattr(out, "losses", None) or {}).items():
        if params["return_loss"] and torch.is_tensor(v) and v.numel() == 1:
            ctx.metric(f"loss.{k}", float(v.detach()))
    if params["input_gradient"]:
        y.sum().backward()
        ctx.metric("input_gradient_norm", float(xs[0].grad.norm()), "norm of d sum(y) / d x (non-zero: gradients "
                                                                    "flow to the inputs)")
    ctx.summary(f"{params['model']}: {', '.join(str(tuple(x.shape)) for x in xs)} -> {tuple(y.shape)}")
    return {"metrics": ctx.report["metrics"]}


@node("architectures.pod_dmd", "POD compression + DMD rollout of a linear system", category="Learning",
      outputs=[Port("metrics", "metrics")],
      params=[Param("trajectories", "int", 16, minimum=1), Param("steps", "int", 60, minimum=2),
              Param("state_dim", "int", 128, minimum=2), Param("rank", "int", 12, minimum=1),
              Param("noise", "float", 0.01, minimum=0.0), Param("seed", "int", 0, minimum=0,
                                                                  help="seed of the linear operator")],
      calls="pinneapple_neural.architectures.rom.toy_systems.make_linear_system / generate_sequences + "
            "rom.pod.POD (fit, encode, decode) + rom.dmd.DynamicModeDecomposition")
def pod_dmd(ctx, inputs, params):
    """POD reconstruction and POD + DMD rollout errors on a synthetic linear system. Trajectories of a contractive
    random linear system are compressed by POD (rank r), advanced by DMD on the POD coefficients and decoded."""
    import torch

    from pinneapple_neural.architectures.rom.dmd import DynamicModeDecomposition
    from pinneapple_neural.architectures.rom.pod import POD
    from pinneapple_neural.architectures.rom.toy_systems import (
        generate_sequences,
        make_linear_system,
    )
    torch.manual_seed(0)
    B, T, D, r = (int(params[k]) for k in ("trajectories", "steps", "state_dim", "rank"))
    A = make_linear_system(D, seed=int(params["seed"]))
    X = generate_sequences(B, T, D, A, noise=float(params["noise"]))
    pod = POD(r=r, center=True)
    pod.fit(X)
    out_pod = pod(X, return_loss=True)
    ctx.metric("pod_recon_mse", float(out_pod.losses["mse"]), f"rank {r}, {B} x {T} snapshots of dimension {D}")
    with torch.no_grad():
        a_seq = pod.encode(X).reshape(B, T, r)
    dmd = DynamicModeDecomposition(r=r, center=True)
    dmd.fit(a_seq)
    with torch.no_grad():
        a_hat = dmd(a_seq, return_loss=True).y
        X_hat = pod.decode(a_hat.reshape(B * T, r), shape=(B, T, D))
        mse = torch.mean((X_hat - X) ** 2).item()
    ctx.metric("rollout_mse", float(mse), "POD + DMD rollout decoded to the state space")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="architectures.registry_tour", title="Tour of the model registry: families, models, forward passes",
        domain="neural architectures", example="architectures",
        keywords={"model registry": 4, "registry tour": 4, "list models": 3, "model catalog": 3,
                  "registered models": 3, "architectures": 1},
        min_score=3, nodes=("architectures.registry", "architectures.forward"),
        references=("kovachki2023operator", "raissi2019pinn"), template="architectures/00_registry_tour", quick=False,
        description="examples/architectures/00_registry_tour.py"))
    register_capability(Capability(
        id="architectures.pod_dmd", title="POD compression and a DMD rollout of a linear dynamical system",
        domain="neural architectures", example="architectures",
        keywords={"pod dmd": 4, "pod + dmd": 4, "dmd rollout": 4, "dynamic mode decomposition": 3,
                  "proper orthogonal decomposition": 2},
        min_score=4, nodes=("architectures.pod_dmd",), references=("berkooz1993pod", "schmid2010dmd"),
        template="architectures/30_rom_pod_dmd", quick=False, description="examples/architectures/30_rom_pod_dmd.py"))


_register_capabilities()
