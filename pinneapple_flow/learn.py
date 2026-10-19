"""Surrogates for flow nodes: build (MLP blocks, DeepONet, FNO, POD + MLP), preview as a layer graph, train, tune.

The architectures are the library's (:class:`~pinneapple_neural.architectures.neural_operators.deeponet.DeepONet`,
:class:`~pinneapple_neural.architectures.neural_operators.fno.FourierNeuralOperator`,
:class:`~pinneapple_neural.architectures.rom.pod.POD`); this module wires them to one dataset format, normalises
inputs and outputs, and measures held-out errors.

Operator dataset (``.npz``): ``A`` (samples, m) inputs (parameters or an input function at m sensors), ``Y``
(samples, points * fields) outputs on a fixed grid, ``grid`` (points, d) the query coordinates.

    cfg = {"kind": "pod_mlp", "rank": 8, "blocks": [{"type": "dense", "width": 64, "activation": "tanh"}] * 2}
    preview = architecture(cfg, in_dim=2, out_dim=1600)           # layer graph + parameter count, no training
    model, info = fit(cfg, data, epochs=800)                       # info["metrics"]["test_rel_l2"]
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

KINDS = ("mlp", "deeponet", "fno", "pod_mlp")
BLOCK_TYPES = ("dense", "residual", "dropout", "layernorm")
ACTIVATIONS = ("tanh", "gelu", "relu", "silu", "sin")


def default_config(kind: str, in_dim: int, out_dim: int) -> dict[str, Any]:
    """The architecture a surrogate node suggests for ``kind`` and the data's sizes."""
    width = 64 if out_dim <= 4096 else 128
    dense = {"type": "dense", "width": width, "activation": "gelu"}
    if kind == "mlp":
        return {"kind": "mlp", "blocks": [dict(dense), dict(dense), dict(dense)]}
    if kind == "deeponet":
        return {"kind": "deeponet", "hidden": width, "modes": 32, "depth": 3}
    if kind == "fno":
        return {"kind": "fno", "width": 32, "modes": 12, "layers": 4}
    if kind == "pod_mlp":
        return {"kind": "pod_mlp", "rank": min(12, max(2, in_dim * 4)), "blocks": [dict(dense), dict(dense)]}
    raise ValueError(f"unknown surrogate kind {kind!r}; known: {KINDS}")


def _torch():
    import torch
    return torch


def _activation(name: str):
    nn = _torch().nn

    class Sin(nn.Module):
        def forward(self, x):
            return _torch().sin(x)
    return {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "sin": Sin}[name]()


def blocks_net(in_dim: int, out_dim: int, blocks: list[dict[str, Any]]):
    """A ``torch.nn.Sequential`` from editable blocks: ``dense`` (width, activation), ``residual`` (width,
    activation: x + f(x)), ``dropout`` (p), ``layernorm``; a final linear layer maps to ``out_dim``."""
    torch = _torch()
    nn = torch.nn

    class Residual(nn.Module):
        def __init__(self, w, act):
            super().__init__()
            self.lin, self.act = nn.Linear(w, w), _activation(act)

        def forward(self, x):
            return x + self.act(self.lin(x))

    if len(blocks) > 32:
        raise ValueError("at most 32 blocks")
    layers, d = [], int(in_dim)
    for b in blocks:
        t = b.get("type", "dense")
        if t not in BLOCK_TYPES:
            raise ValueError(f"block type {t!r} is not one of {BLOCK_TYPES}")
        act = b.get("activation", "tanh")
        if act not in ACTIVATIONS:
            raise ValueError(f"activation {act!r} is not one of {ACTIVATIONS}")
        if t == "dense":
            w = int(b.get("width", 64))
            if not 1 <= w <= 4096:
                raise ValueError("block width must be 1..4096")
            layers += [nn.Linear(d, w), _activation(act)]
            d = w
        elif t == "residual":
            if int(b.get("width", d)) != d:
                layers += [nn.Linear(d, int(b["width"]))]
                d = int(b["width"])
            layers.append(Residual(d, act))
        elif t == "dropout":
            layers.append(nn.Dropout(float(b.get("p", 0.1))))
        elif t == "layernorm":
            layers.append(nn.LayerNorm(d))
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


def build(cfg: dict[str, Any], in_dim: int, out_dim: int, *, n_fields: int = 1, grid: np.ndarray | None = None):
    """The torch surrogate for ``cfg`` mapping ``(B, in_dim)`` to ``(B, out_dim)`` (normalised units)."""
    torch = _torch()
    nn = torch.nn
    kind = cfg.get("kind")

    class Surrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.register_buffer("a_mean", torch.zeros(in_dim))
            self.register_buffer("a_std", torch.ones(in_dim))
            self.register_buffer("y_mean", torch.zeros(out_dim))
            self.register_buffer("y_std", torch.ones(out_dim))
            if kind == "mlp":
                self.core = blocks_net(in_dim, out_dim, cfg.get("blocks") or default_config("mlp", 1, 1)["blocks"])
            elif kind == "deeponet":
                from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
                if grid is None:
                    raise ValueError("DeepONet needs the query grid")
                self.register_buffer("grid", torch.tensor(np.asarray(grid, np.float32)))
                self.core = DeepONet(in_dim, self.grid.shape[1], n_fields, hidden=int(cfg.get("hidden", 64)),
                                     modes=int(cfg.get("modes", 32)), depth=int(cfg.get("depth", 3)))
            elif kind == "fno":
                from pinneapple_neural.architectures.neural_operators.fno import (
                    FourierNeuralOperator,
                )
                if out_dim != in_dim * n_fields:
                    raise ValueError("FNO maps a function on the output grid to fields on it: the input must be "
                                     "sampled at the output points (use DeepONet or POD+MLP for parameter inputs)")
                self.core = FourierNeuralOperator(1, n_fields, width=int(cfg.get("width", 32)),
                                                  modes=int(cfg.get("modes", 12)), layers=int(cfg.get("layers", 4)),
                                                  use_grid=True)
            elif kind == "pod_mlp":
                r = int(cfg.get("rank", 8))
                self.register_buffer("basis", torch.zeros(out_dim, r))
                self.core = blocks_net(in_dim, r, cfg.get("blocks") or default_config("pod_mlp", 1, 1)["blocks"])
            else:
                raise ValueError(f"unknown surrogate kind {kind!r}; known: {KINDS}")

        def forward_normalised(self, a):
            if self.kind == "deeponet":
                y = self.core(a, self.grid).y               # (B, N, F)
                return y.reshape(len(a), -1)
            if self.kind == "fno":
                y = self.core(a[:, None, :]).y              # (B, F, N)
                return y.transpose(1, 2).reshape(len(a), -1)
            if self.kind == "pod_mlp":
                return self.core(a) @ self.basis.T
            return self.core(a)

        def forward(self, a):
            an = (a - self.a_mean) / self.a_std
            return self.forward_normalised(an) * self.y_std + self.y_mean

    return Surrogate()


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def layer_graph(model) -> dict[str, Any]:
    """Leaf layers in order with their parameter counts, grouped by top-level block; edges in execution order
    within a group (the DeepONet branch and trunk are two groups that meet at the contraction)."""
    nodes, edges = [], []
    prev_in_group: dict[str, str] = {}
    for name, mod in model.named_modules():
        if not name or any(True for _ in mod.children()):
            continue
        if name.startswith("core."):
            name = name[5:]
        group = name.split(".")[0] if "." in name else "core"
        if group.isdigit():
            group = "core"
        n_par = int(sum(p.numel() for p in mod.parameters(recurse=False)))
        extra = ""
        for attr in ("in_features", "out_features", "in_channels", "out_channels", "p", "normalized_shape"):
            if hasattr(mod, attr):
                extra += f"{attr}={getattr(mod, attr)} "
        nodes.append({"id": name, "type": type(mod).__name__, "params": n_par, "group": group,
                      "detail": extra.strip()})
        if group in prev_in_group:
            edges.append([prev_in_group[group], name])
        prev_in_group[group] = name
    basis = getattr(model, "basis", None)
    if basis is not None and getattr(model, "kind", "") == "pod_mlp":          # the POD reconstruction after the MLP
        n_out, r = (int(x) for x in basis.shape)
        nodes.append({"id": "pod_basis", "type": "POD reconstruction", "params": 0, "group": "core",
                      "detail": f"in_modes={r} out_features={n_out} (basis fixed from the training snapshots, "
                                f"{n_out * r} values, not trained)"})
        if "core" in prev_in_group:
            edges.append([prev_in_group["core"], "pod_basis"])
    return {"nodes": nodes, "edges": edges, "total_params": count_parameters(model),
            "groups": list(dict.fromkeys(n["group"] for n in nodes))}


def architecture(cfg: dict[str, Any], in_dim: int, out_dim: int, *, n_fields: int = 1,
                 grid: np.ndarray | None = None) -> dict[str, Any]:
    """Preview of an architecture without training: layer graph, parameter count, the config used."""
    if cfg.get("kind") == "deeponet" and grid is None:
        grid = np.zeros((max(1, out_dim // max(1, n_fields)), 1))
    model = build(cfg, in_dim, out_dim, n_fields=n_fields, grid=grid)
    g = layer_graph(model)
    g["config"] = cfg
    g["in_dim"], g["out_dim"] = int(in_dim), int(out_dim)
    return g


def split(n: int, val: float = 0.15, test: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nt, nv = max(1, int(round(n * test))), max(1, int(round(n * val)))
    return idx[nt + nv:], idx[nt:nt + nv], idx[:nt]


def rel_l2_rows(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean over samples of ||pred - true|| / ||true||."""
    num = np.linalg.norm(pred - true, axis=1)
    den = np.maximum(np.linalg.norm(true, axis=1), 1e-12)
    return float(np.mean(num / den))


def fit(cfg: dict[str, Any], data: dict[str, np.ndarray], *, epochs: int = 500, lr: float = 2e-3,
        batch: int = 32, seed: int = 0, n_fields: int = 1, splits=None,
        callback: Callable[[int, float], None] | None = None, init_from=None) -> tuple[Any, dict[str, Any]]:
    """Train the surrogate ``cfg`` on ``data`` (``A``, ``Y``, ``grid``) with Adam and a cosine schedule, keeping the
    weights with the best validation error. Returns ``(model, info)``: per-split relative L2, history, seconds."""
    torch = _torch()
    torch.manual_seed(seed)
    A = np.asarray(data["A"], np.float32)
    Y = np.asarray(data["Y"], np.float32)
    grid = data.get("grid")
    tr, va, te = splits if splits is not None else split(len(A), seed=seed)
    model = build(cfg, A.shape[1], Y.shape[1], n_fields=n_fields, grid=grid)
    if init_from is not None:
        model.load_state_dict(init_from.state_dict())
    else:
        model.a_mean.copy_(torch.tensor(A[tr].mean(0)))
        model.a_std.copy_(torch.tensor(A[tr].std(0) + 1e-8))
        model.y_mean.copy_(torch.tensor(Y[tr].mean(0)))
        model.y_std.copy_(torch.tensor(np.full(Y.shape[1], Y[tr].std() + 1e-8, np.float32)))
        if cfg.get("kind") == "pod_mlp":
            from pinneapple_neural.architectures.rom.pod import POD
            Yn = (torch.tensor(Y[tr]) - model.y_mean) / model.y_std
            pod = POD(r=min(int(cfg.get("rank", 8)), len(tr)), center=False).fit(Yn)
            basis = torch.zeros_like(model.basis)
            basis[:, :pod.basis_.shape[1]] = pod.basis_
            model.basis.copy_(basis)
    At, Yt = torch.tensor(A), torch.tensor(Y)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(seed)
    best, best_state, history = float("inf"), None, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            yn = (Yt[b] - model.y_mean) / model.y_std
            pred = model.forward_normalised((At[b] - model.a_mean) / model.a_std)
            loss = ((pred - yn) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            v = rel_l2_rows(model(At[va]).numpy(), Y[va])
        history.append({"epoch": ep, "train_mse": tot / len(tr), "val_rel_l2": v})
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        if callback is not None:
            callback(ep, v)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(At).numpy()
    metrics = {f"{k}_rel_l2": rel_l2_rows(pred[ix], Y[ix]) for k, ix in (("train", tr), ("val", va), ("test", te))}
    metrics["n_parameters"] = count_parameters(model)
    return model, {"metrics": metrics, "history": history, "seconds": time.perf_counter() - t0,
                   "splits": {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}}


def save(model, cfg: dict[str, Any], path, *, n_fields: int = 1, meta: dict | None = None) -> None:
    torch = _torch()
    torch.save({"config": cfg, "in_dim": int(model.a_mean.numel()), "out_dim": int(model.y_mean.numel()),
                "n_fields": n_fields, "state": model.state_dict(), "meta": meta or {}}, path)


def load(obj: dict[str, Any]):
    """A surrogate from what :func:`save` wrote (already loaded with ``torch.load``)."""
    grid = obj["state"].get("grid")
    model = build(obj["config"], obj["in_dim"], obj["out_dim"], n_fields=obj.get("n_fields", 1),
                  grid=None if grid is None else grid.numpy())
    model.load_state_dict(obj["state"])
    model.eval()
    return model


def search_space(spec: dict[str, Any]) -> dict[str, tuple]:
    """``{"lr": [1e-4, 1e-2, "log"], "width": [16, 128], "activation": ["tanh", "gelu"]}`` ->
    :class:`~pinneapple_neural.trainer.adaptive_sweep.AdaptiveSweepConfig` parameter specs."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v) and v[-1] != "log":
            out[k] = ("categorical", (v,))
        elif isinstance(v, list) and len(v) >= 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                         for x in v[:2]):
            out[k] = ("int", tuple(v[:2]) + (("log",) if v[-1] == "log" else ()))
        elif isinstance(v, list) and len(v) >= 2:
            out[k] = ("float", (float(v[0]), float(v[1])) + (("log",) if v[-1] == "log" else ()))
        else:
            raise ValueError(f"search space entry {k!r}: [low, high(, 'log')] or a list of choices")
    return out


def apply_trial(cfg: dict[str, Any], trial: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a trial's values into architecture changes (width, depth, activation, rank, modes, hidden) and
    training options (lr, epochs, batch)."""
    cfg = {**cfg, "blocks": [dict(b) for b in cfg.get("blocks", [])]}
    train = {}
    for k, v in trial.items():
        if k in ("lr", "epochs", "batch"):
            train[k] = v
        elif k in ("width", "activation") and cfg.get("blocks"):
            for b in cfg["blocks"]:
                if b.get("type") in ("dense", "residual"):
                    b["width" if k == "width" else "activation"] = v
        elif k == "depth" and cfg.get("blocks"):
            dense = next((b for b in cfg["blocks"] if b.get("type") == "dense"),
                         {"type": "dense", "width": 64, "activation": "gelu"})
            cfg["blocks"] = [dict(dense) for _ in range(int(v))]
        else:
            cfg[k] = v
    return cfg, train


def tune(cfg: dict[str, Any], data: dict[str, np.ndarray], space: dict[str, Any], *, n_trials: int = 12,
         epochs: int = 200, seed: int = 0, n_fields: int = 1, log: Callable[[str], None] = print):
    """Hyperparameter search (Optuna TPE when installed, else random search, via
    :func:`pinneapple_neural.trainer.adaptive_sweep.run_adaptive_sweep`) minimising the validation error on fixed
    splits. Returns ``(trials sorted best first, backend name)``."""
    from pinneapple_neural.trainer.adaptive_sweep import AdaptiveSweepConfig, run_adaptive_sweep
    splits = split(len(data["A"]), seed=seed)

    def trial(**kw):
        c, tr = apply_trial(cfg, kw)
        _, info = fit(c, data, epochs=int(tr.get("epochs", epochs)), lr=float(tr.get("lr", 2e-3)),
                      batch=int(tr.get("batch", 32)), seed=seed, n_fields=n_fields, splits=splits)
        log(f"trial {kw} -> val rel L2 {info['metrics']['val_rel_l2']:.4g}")
        return info["metrics"]["val_rel_l2"]

    try:
        import optuna  # noqa: F401
        backend = "optuna-tpe"
    except ImportError:
        backend = "random-search"
    res = run_adaptive_sweep(trial, AdaptiveSweepConfig(param_space=search_space(space), n_trials=n_trials,
                                                        seed=seed, pruning=False))
    return res, backend
