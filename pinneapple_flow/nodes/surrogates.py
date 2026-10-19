"""Surrogate nodes: neural operators and hybrid models with an editable architecture, HPO, fine-tuning, UQ."""
from __future__ import annotations

import numpy as np

from pinneapple_core.workflows import StepFailed

from .. import learn
from ..registry import Param, Port, node
from ..runtime import load_artifact
from ._common import save_solution
from .data import dataset_of


def _meta(v) -> dict:
    return {k: v.get(k) for k in ("param_names", "coords", "fields", "ranges", "input_kind") if k in v}


def _report_fit(ctx, model, cfg, d, info, n_fields):
    m = info["metrics"]
    for k in ("train_rel_l2", "val_rel_l2", "test_rel_l2"):
        ctx.metric(k, m[k], "mean relative L2 over the samples of that split (held out from training for val/test)")
    ctx.metric("n_parameters", m["n_parameters"])
    ctx.metric("train_seconds", round(info["seconds"], 2), "wall clock on this machine")
    g = learn.layer_graph(model)
    g["config"] = cfg
    ctx.view("architecture", g)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    te = info["splits"]["test"]
    with torch.no_grad():
        pred = model(torch.tensor(np.asarray(d["A"][te], np.float32))).numpy()
    true = np.asarray(d["Y"][te])
    shape = [int(s) for s in d["grid_shape"]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    h = [r["val_rel_l2"] for r in info["history"]]
    ax[0].semilogy(h)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("validation rel. L2")
    k = 0
    if len(shape) == 1:
        x = np.asarray(d["grid"])[:, 0]
        ax[1].plot(x, true[k][: len(x)], "k-", label="reference")
        ax[1].plot(x, pred[k][: len(x)], "r--", label="surrogate")
        ax[1].legend(fontsize=8)
        ax[1].set_title("a held-out test sample", fontsize=9)
    else:
        err = (pred[k] - true[k]).reshape(*shape, n_fields)[..., 0]
        err2 = err if err.ndim == 2 else err[..., -1]
        im = ax[1].imshow(err2.T, origin="lower", cmap="coolwarm")
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("error on a held-out test sample", fontsize=9)
    ctx.figure(fig, "surrogate", "training curve and a held-out test sample")
    rows = [[int(i), float(np.linalg.norm(pred[j] - true[j]) / max(np.linalg.norm(true[j]), 1e-12))]
            for j, i in enumerate(te[:20])]
    ctx.table("test samples", ["sample", "relative L2"], rows)


def _train_out(ctx, model, cfg, d, info, n_fields, src):
    path = ctx.path("model.pt")
    learn.save(model, cfg, path, n_fields=n_fields, meta={**_meta(src), "grid_shape": [int(s) for s in
                                                                                       d["grid_shape"]],
                                                         "lo": d.get("lo", np.array([])).tolist(),
                                                         "hi": d.get("hi", np.array([])).tolist()})
    from ..runtime import artifact
    mdl = artifact(path, "pt")
    mdl.update({"kind_model": "surrogate", "surrogate": cfg.get("kind")})
    import torch
    te = info["splits"]["test"][0]
    with torch.no_grad():
        y = model(torch.tensor(np.asarray(d["A"][te:te + 1], np.float32))).numpy()[0]
    coords = src.get("coords") or [f"x{i}" for i in range(np.asarray(d["grid"]).shape[1])]
    shape = [int(s) for s in d["grid_shape"]]
    grid = np.asarray(d["grid"])
    axes = {c: np.unique(grid[:, i]) for i, c in enumerate(coords)}
    if all(len(axes[c]) == s for c, s in zip(coords, shape, strict=False)):
        sol = save_solution(ctx, "prediction", axes, y.reshape(*shape, n_fields), coords,
                            src.get("fields") or ["u"], "surrogate", f"surrogate on test sample {te}")
    else:
        sol = None
    return mdl, sol


_ARCH = [Param("kind", "enum", "deeponet", choices=list(learn.KINDS),
               help="deeponet / fno (neural operators), mlp, pod_mlp (ROM + NN hybrid)"),
         Param("architecture", "dict", {}, help="blocks and sizes; empty = the suggested architecture "
                                                 "(see the preview)"),
         Param("epochs", "int", 600, minimum=1, maximum=100000), Param("lr", "float", 2e-3, minimum=1e-6),
         Param("batch", "int", 32, minimum=1, maximum=4096), Param("seed", "int", 0, minimum=0)]


def _cfg(params, d):
    cfg = dict(params["architecture"] or {})
    cfg.setdefault("kind", params["kind"])
    if cfg["kind"] != params["kind"]:
        cfg = {"kind": params["kind"]}
    base = learn.default_config(params["kind"], np.asarray(d["A"]).shape[1], np.asarray(d["Y"]).shape[1])
    return {**base, **cfg}


@node("surrogate.train", "Neural operator / surrogate", category="Learning",
      inputs=[Port("data", "dataset")],
      outputs=[Port("model", "model"), Port("solution", "solution", required=False), Port("metrics", "metrics")],
      params=list(_ARCH),
      calls="pinneapple_flow.learn.fit over DeepONet / FourierNeuralOperator / POD (pinneapple_neural)")
def surrogate(ctx, inputs, params):
    """Train a surrogate on a dataset: the suggested architecture (preview: layer graph + parameter count) or the
    edited one; errors on held-out validation and test samples."""
    src = inputs["data"]
    d = dataset_of(src)
    n_fields = len(src.get("fields") or ["u"])
    cfg = _cfg(params, d)
    rid = f"surrogate.{cfg['kind']}" if cfg["kind"] in ("deeponet", "fno", "pod_mlp", "mlp") else "surrogate.mlp"
    ctx.decide(rid, f"{cfg['kind']} surrogate", cfg["kind"])
    ctx.auto("architecture.default_sizes", "suggested architecture", "architecture", value=cfg)
    every = max(1, int(params["epochs"]) // 10)

    def progress(ep, v):
        ctx.check_stop()
        if ep % every == 0:
            ctx.log(f"epoch {ep}: val rel L2 {v:.4g}")
    try:
        model, info = learn.fit(cfg, d, epochs=int(params["epochs"]), lr=float(params["lr"]),
                                batch=int(params["batch"]), seed=int(params["seed"]), n_fields=n_fields,
                                callback=progress)
    except ValueError as exc:
        raise StepFailed(str(exc)) from None
    _report_fit(ctx, model, cfg, d, info, n_fields)
    mdl, sol = _train_out(ctx, model, cfg, d, info, n_fields, src)
    ctx.summary(f"{cfg['kind']}: test rel. L2 {info['metrics']['test_rel_l2']:.3g}, "
                f"{info['metrics']['n_parameters']} parameters")
    out = {"model": mdl, "metrics": info["metrics"]}
    if sol:
        out["solution"] = sol
    return out


@node("surrogate.hpo", "Hyperparameter search", category="Learning",
      inputs=[Port("data", "dataset")],
      outputs=[Port("model", "model"), Port("params", "params"), Port("metrics", "metrics")],
      params=[Param("kind", "enum", "pod_mlp", choices=list(learn.KINDS)),
              Param("architecture", "dict", {}),
              Param("space", "dict", {"lr": [3e-4, 1e-2, "log"], "width": [16, 128], "depth": [1, 4]},
                    help="{name: [low, high(, 'log')] or [choices]}: lr, epochs, batch, width, depth, activation, "
                         "rank, modes, hidden"),
              Param("n_trials", "int", 12, minimum=1, maximum=500),
              Param("trial_epochs", "int", 200, minimum=1), Param("final_epochs", "int", 800, minimum=1),
              Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_neural.trainer.adaptive_sweep.run_adaptive_sweep (Optuna TPE or random search)")
def hpo(ctx, inputs, params):
    """Search architecture and training settings on the validation split, then retrain the best with the full
    budget and report its held-out test error (the test split is never used in the search)."""
    src = inputs["data"]
    d = dataset_of(src)
    n_fields = len(src.get("fields") or ["u"])
    cfg = _cfg(params, d)
    ctx.decide("hpo.search", "TPE when Optuna is installed, else random search; validation split only")
    ctx.auto("hpo.ranges", f"search ranges {params['space']}", "space", value=params["space"])
    try:
        trials, backend = learn.tune(cfg, d, params["space"], n_trials=int(params["n_trials"]),
                                     epochs=int(params["trial_epochs"]), seed=int(params["seed"]),
                                     n_fields=n_fields, log=ctx.log)
    except ValueError as exc:
        raise StepFailed(str(exc)) from None
    ok = [t for t in trials if "error" not in t and t.get("value") is not None]
    if not ok:
        raise StepFailed("every trial failed: " + "; ".join(str(t.get("error")) for t in trials[:3]))
    best = ok[0]
    ctx.metric("search_backend", backend)
    ctx.metric("trials", len(trials))
    ctx.metric("best_trial_val_rel_l2", best["value"])
    keys = sorted({k for t in ok for k in t["params"]})
    ctx.table("trials (best first, validation rel. L2)", [*keys, "val rel L2"],
              [[t["params"].get(k) for k in keys] + [t["value"]] for t in ok])
    final_cfg, tr = learn.apply_trial(cfg, best["params"])
    model, info = learn.fit(final_cfg, d, epochs=int(params["final_epochs"]), lr=float(tr.get("lr", 2e-3)),
                            batch=int(tr.get("batch", 32)), seed=int(params["seed"]), n_fields=n_fields)
    _report_fit(ctx, model, final_cfg, d, info, n_fields)
    mdl, _ = _train_out(ctx, model, final_cfg, d, info, n_fields, src)
    ctx.summary(f"best of {len(trials)} trials ({backend}): test rel. L2 {info['metrics']['test_rel_l2']:.3g}")
    return {"model": mdl, "params": {"best": best["params"], "config": final_cfg}, "metrics": info["metrics"]}


def _load_surrogate(v):
    obj = load_artifact(v)
    if not isinstance(obj, dict) or "config" not in obj:
        raise StepFailed("expected a surrogate model (from a surrogate or HPO node)")
    return obj, learn.load(obj)


@node("train.finetune", "Fine-tuning", category="Learning",
      inputs=[Port("model", "model"), Port("data", "dataset")],
      outputs=[Port("model", "model"), Port("metrics", "metrics")],
      params=[Param("epochs", "int", 200, minimum=1), Param("lr", "float", 3e-4, minimum=1e-7),
              Param("freeze", "enum", "none", choices=["none", "all_but_last"]),
              Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_flow.learn.fit(init_from=model) with frozen layers")
def finetune(ctx, inputs, params):
    """Continue training a surrogate on new data (all layers, or only the last one), reporting the error on the new
    data's held-out split before and after."""
    import torch
    obj, model = _load_surrogate(inputs["model"])
    src = inputs["data"]
    d = dataset_of(src)
    splits = learn.split(len(d["A"]), seed=int(params["seed"]))
    with torch.no_grad():
        before = learn.rel_l2_rows(model(torch.tensor(np.asarray(d["A"][splits[2]], np.float32))).numpy(),
                                   np.asarray(d["Y"][splits[2]]))
    if params["freeze"] == "all_but_last":
        lin = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in lin[-1].parameters():
            p.requires_grad_(True)
    cfg = obj["config"]
    tuned, info = learn.fit(cfg, d, epochs=int(params["epochs"]), lr=float(params["lr"]),
                            n_fields=obj.get("n_fields", 1), splits=splits, init_from=model,
                            seed=int(params["seed"]))
    ctx.metric("test_rel_l2_before", before, "on the new data's test split")
    ctx.metric("test_rel_l2_after", info["metrics"]["test_rel_l2"], "on the same split")
    path = ctx.path("model.pt")
    learn.save(tuned, cfg, path, n_fields=obj.get("n_fields", 1), meta=obj.get("meta"))
    from ..runtime import artifact
    return {"model": artifact(path, "pt"), "metrics": ctx.report["metrics"]}


@node("uq.sensitivity", "Sensitivity (Sobol)", category="Analysis",
      inputs=[Port("model", "model")],
      outputs=[Port("metrics", "metrics")],
      params=[Param("quantity", "enum", "mean", choices=["mean", "max", "l2"],
                    help="scalar of the predicted field whose variance is decomposed"),
              Param("n", "int", 1024, minimum=64, maximum=65536), Param("seed", "int", 0, minimum=0)],
      calls="pp.sobol_indices on the surrogate over the training ranges")
def sensitivity(ctx, inputs, params):
    """Sobol indices (first order and total) of a quantity of the field with respect to the surrogate's input
    parameters, over the ranges it was trained on (with bootstrap confidence intervals)."""
    import torch

    import pinneapple as pp
    obj, model = _load_surrogate(inputs["model"])
    meta = obj.get("meta") or {}
    names, lo, hi = meta.get("param_names"), meta.get("lo"), meta.get("hi")
    if not names or not lo or len(names) > 20:
        raise StepFailed("Sobol indices need a surrogate of a parametric sweep (named parameters with ranges)")
    space = {k: (float(a), float(b)) for k, a, b in zip(names, lo, hi, strict=True)}
    q = params["quantity"]

    def fn(P):
        with torch.no_grad():
            Y = model(torch.tensor(np.asarray(P, np.float32))).numpy()
        return {"mean": Y.mean(1), "max": Y.max(1), "l2": np.sqrt((Y ** 2).mean(1))}[q]
    ctx.decide("uq.sobol", f"Sobol indices of the field {q} over the training ranges", space)
    res = pp.sobol_indices(fn, space, n=int(params["n"]), seed=int(params["seed"]))
    S1, ST = np.atleast_2d(res.S1)[0], np.atleast_2d(res.ST)[0]
    ctx.table(f"Sobol indices of the field {q}", ["parameter", "S1", "ST"],
              [[k, float(a), float(b)] for k, a, b in zip(names, S1, ST, strict=True)])
    for k, a, b in zip(names, S1, ST, strict=True):
        ctx.metric(f"S1_{k}", float(a))
        ctx.metric(f"ST_{k}", float(b))
    ctx.metric("model_evaluations", int(res.n_evals))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(names))
    ax.bar(x - 0.18, S1, 0.36, label="first order")
    ax.bar(x + 0.18, ST, 0.36, label="total")
    ax.set_xticks(x, names)
    ax.legend(fontsize=8)
    ctx.figure(fig, "sobol", "Sobol indices")
    return {"metrics": ctx.report["metrics"]}
