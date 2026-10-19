"""Train the airfoil surrogates on OpenFOAM runs and measure them on shapes they never saw.

Two models from :mod:`.surrogate`: the :class:`~.surrogate.MLPEnsemble` on (shape, alpha) -> (Cl, log Cd, Cm),
bootstrapped so its spread is an uncertainty, and the :class:`~.surrogate.AeroGNN` (MeshGraphNet; Pfaff et al.,
"Learning Mesh-Based Simulation with Graph Networks", ICLR 2021) on the near-wall O-grid cells, which predicts the
fields p, Ux, Uy, log(1 + nut/nu) and the coefficients.

The split is by shape, not by run (:func:`split`): a held-out shape is unseen at every angle, so the test error is the
error on a new design. The NACA reference polars are a second, independent test.

    recs = load_runs(["runs/"])                       # run records of pinneapple_design.aero.dataset
    bundle, metrics = train_surrogates(recs, gnn_minutes=60)
    torch.save(bundle, "surrogate.pt")                # what pinneapple_design.aero.optimize.Engine.load reads

The CLI is ``apps/aero_optimizer/tools/train_surrogate.py``.
"""
from __future__ import annotations

import glob
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch

from .dataset import MAX_CL_OSC, USABLE, steady_values
from .surrogate import (
    FIELDS,
    AeroGNN,
    Layout,
    MLPEnsemble,
    coef_from_targets,
    coef_targets,
    edge_features,
    node_features,
)

COEF_WEIGHTS = (1.0, 2.0, 1.0)            # loss weights on (Cl, log Cd, Cm): drag matters most

__all__ = ["load_runs", "split", "coefs", "metrics", "train_mlp", "predict_mlp", "graph_data", "train_gnn",
           "predict_gnn", "train_surrogates", "MAX_CL_OSC", "USABLE"]


def load_runs(dirs: Sequence[str], max_osc: float = MAX_CL_OSC) -> list[dict[str, Any]]:
    """Usable run records from ``<dir>/runs/*.json`` of each directory, with their steady coefficients.

    Keeps runs whose ``status`` is in ``USABLE`` and that have coefficients; unsteady runs are averaged or dropped
    by :func:`.dataset.steady_values`. Each record gets ``npz``, the path of its cell fields.
    """
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "runs", "*.json"))):
            with open(f) as fh:
                r = json.load(fh)
            if r.get("status") not in USABLE or "Cl" not in r:
                continue
            r["npz"] = f[:-5] + ".npz"
            recs.append(r)
    return steady_values(recs, max_osc)


def split(recs: Sequence[dict[str, Any]], frac: float = 0.15, seed: int = 3):
    """``(train, test, reference)`` with whole shapes held out.

    ``round(frac * n)`` (at least one) of the ``set == "lhs"`` shapes go to test with all their angles. Runs of any
    other set except "reference" (e.g. "verify", the active-learning runs) always train; "reference" runs (the NACA
    polars) are returned apart.
    """
    shapes = sorted({r["shape_id"] for r in recs if r["set"] == "lhs"})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(shapes, max(1, int(round(frac * len(shapes)))), replace=False))
    tr = [r for r in recs if r["set"] != "reference" and r["shape_id"] not in test]
    te = [r for r in recs if r["set"] != "reference" and r["shape_id"] in test]
    ref = [r for r in recs if r["set"] == "reference"]
    return tr, te, ref


def coefs(recs: Iterable[dict[str, Any]]) -> np.ndarray:
    """(n, 3) array of (Cl, Cd, Cm)."""
    return np.array([[r["Cl"], r["Cd"], r["Cm"]] for r in recs])


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Per coefficient (columns Cl, Cd, Cm): mean and max absolute error and R^2 = 1 - SS_res / SS_tot;
    for Cd also the mean and 90th-percentile relative error in percent (``mape``, ``p90``)."""
    out = {}
    for k, i in (("Cl", 0), ("Cd", 1), ("Cm", 2)):
        t, p = true[:, i], pred[:, i]
        err = p - t
        out[k] = {"mae": float(np.abs(err).mean()), "max": float(np.abs(err).max()),
                  "r2": float(1 - (err ** 2).sum() / ((t - t.mean()) ** 2).sum())}
        if k == "Cd":
            rel = np.abs(err) / t
            out[k].update(mape=float(100 * rel.mean()), p90=float(100 * np.percentile(rel, 90)))
    return out


# ------------------------------------------------------------------ MLP ensemble
def train_mlp(tr: Sequence[dict[str, Any]], k: int = 5, width: int = 96, epochs: int = 6000, seed: int = 0):
    """Train ``k`` MLPs on bootstrap resamples of ``tr`` (full batch, Adam + cosine schedule).

    Targets are (Cl, log Cd, Cm), standardised. Returns ``(model, norm)`` with ``norm = {"mean", "std"}`` of the
    targets.
    """
    X = MLPEnsemble.inputs(np.array([r["shape"] for r in tr]), np.array([r["alpha"] for r in tr]))
    Y = coef_targets(*coefs(tr).T)
    mean, std = Y.mean(0), Y.std(0)
    Yn = ((Y - mean) / std).astype(np.float32)
    torch.manual_seed(seed)
    model = MLPEnsemble(k, width)
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(X), len(X)) for _ in range(k)]
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Yn)
    w = torch.tensor(COEF_WEIGHTS)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sum(((net(Xt[b]) - Yt[b]) ** 2 * w).mean() for net, b in zip(model.nets, boots, strict=True))
        loss.backward()
        opt.step()
        sched.step()
    return model, {"mean": mean, "std": std}


def predict_mlp(model: MLPEnsemble, norm: dict[str, np.ndarray], recs: Sequence[dict[str, Any]]):
    """Ensemble mean and spread of (Cl, Cd, Cm) for the runs ``recs``: two (n, 3) arrays."""
    X = MLPEnsemble.inputs(np.array([r["shape"] for r in recs]), np.array([r["alpha"] for r in recs]))
    with torch.no_grad():
        y = model(torch.from_numpy(X)).numpy() * norm["std"] + norm["mean"]
    cl, cd, cm = coef_from_targets(y)
    return np.stack([cl.mean(0), cd.mean(0), cm.mean(0)], 1), np.stack([cl.std(0), cd.std(0), cm.std(0)], 1)


# ------------------------------------------------------------------ graph network
def graph_data(lay: Layout, recs: Sequence[dict[str, Any]]):
    """Node features, edge features, node targets (p, Ux, Uy, log(1 + nut/nu)) and coefficient targets of each run.

    Reads each run's ``npz`` cell fields (indexed by ``lay.cell_index``). Shapes: (n, N, F), (n, E, 3), (n, N, 4),
    (n, 3).
    """
    X, E, Yf, Yc = [], [], [], []
    for r in recs:
        s = np.array(r["shape"])
        pos = lay.positions(s)
        X.append(node_features(lay, s, r["alpha"], pos))
        E.append(edge_features(lay, pos))
        z = np.load(r["npz"])
        nu = 1.0 / r["reynolds"]
        f = np.stack([z["p"], z["Ux"], z["Uy"], np.log1p(z["nut"] / nu)], 1)[lay.cell_index]
        Yf.append(f)
        Yc.append(coef_targets(r["Cl"], r["Cd"], r["Cm"]))
    return np.stack(X), np.stack(E), np.stack(Yf).astype(np.float32), np.stack(Yc).astype(np.float32)


def train_gnn(tr: Sequence[dict[str, Any]], te: Sequence[dict[str, Any]], minutes: float | None = 60.0,
              seed: int = 0, batch: int = 4, log: Callable[[str], Any] = print, *, epochs: int | None = None,
              hidden: int = 48, mp: int = 8, layout: Layout | None = None, checkpoint_dir: str | None = None,
              checkpoint_every: int = 5):
    """Train the graph network on ``tr`` for a wall-clock budget of ``minutes`` and/or a number of ``epochs``.

    The learning rate follows a cosine from 1e-3 to 3e-5 over the budget (the larger of the elapsed time and epoch
    fractions). Every 5 epochs and at the end the Cd error on ``te`` is logged. Returns
    ``(model, norm, history, layout)``.

    With ``checkpoint_dir``, the full training state (weights, optimiser, epoch, elapsed time, history, shuffling RNG)
    is written to ``checkpoint_dir/gnn_last.pt`` every ``checkpoint_every`` epochs, and a later call with the same
    settings resumes from it: an interrupted run continues to the same weights as an uninterrupted one. A
    checkpoint written with other settings is refused.
    """
    if minutes is None and epochs is None:
        raise ValueError("give minutes, epochs or both")
    lay = layout or Layout()
    X, E, Yf, Yc = graph_data(lay, tr)
    Xv, Ev, Yfv, Ycv = graph_data(lay, te)
    fm, fs = Yf.reshape(-1, 4).mean(0), Yf.reshape(-1, 4).std(0)
    cm_, cs = Yc.mean(0), Yc.std(0)
    norm = {"field_mean": fm, "field_std": fs, "coef_mean": cm_, "coef_std": cs}
    Yfn, Ycn = (Yf - fm) / fs, (Yc - cm_) / cs
    torch.manual_seed(seed)
    model = AeroGNN(X.shape[-1], hidden=hidden, mp=mp)
    ei = torch.from_numpy(lay.edge_index)
    wall = torch.arange(len(lay.i_sel))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0, ep, hist = time.time(), 0, []
    budget = None if minutes is None else minutes * 60
    rng = np.random.default_rng(seed)
    settings = {"minutes": minutes, "epochs": epochs, "seed": seed, "batch": batch, "hidden": hidden, "mp": mp,
                "n_train": len(tr), "train_ids": sorted(str(r.get("id", i)) for i, r in enumerate(tr))}
    ckpt = os.path.join(checkpoint_dir, "gnn_last.pt") if checkpoint_dir else None
    if ckpt and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st["settings"] != settings:
            raise ValueError(f"{ckpt} was written with other training settings; remove it or change checkpoint_dir")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        ep, hist = st["epoch"], st["history"]
        rng.bit_generator.state = st["rng"]
        t0 = time.time() - st["elapsed"]
        log(f"  gnn resumed at epoch {ep} from {ckpt}")

    def save_checkpoint():
        os.makedirs(checkpoint_dir, exist_ok=True)
        tmp = ckpt + ".tmp"
        torch.save({"settings": settings, "model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep,
                    "history": hist, "rng": rng.bit_generator.state, "elapsed": time.time() - t0}, tmp)
        os.replace(tmp, ckpt)                                      # never leaves a half-written checkpoint
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))         # noqa: E731

    def done():
        return (budget is not None and time.time() - t0 >= budget) or (epochs is not None and ep >= epochs)

    while not done():
        frac = max((time.time() - t0) / budget if budget else 0.0, ep / epochs if epochs else 0.0)
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * frac)))
        model.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for b in range(0, len(perm), batch):
            idx = perm[b:b + batch]
            y, c = model(T(X[idx]), ei, T(E[idx]), wall)
            lf = ((y - T(Yfn[idx])) ** 2).mean()
            lc = (((c - T(Ycn[idx])) ** 2) * torch.tensor(COEF_WEIGHTS)).mean()
            loss = lf + lc
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        ep += 1
        if ep % 5 == 0 or done():
            model.eval()
            with torch.no_grad():
                _, cv = model(T(Xv), ei, T(Ev), wall)
            pv = coef_from_targets(cv.numpy() * cs + cm_)
            cdr = float(np.mean(np.abs(pv[1] - coef_from_targets(Ycv)[1]) / coef_from_targets(Ycv)[1]) * 100)
            hist.append({"epoch": ep, "minutes": (time.time() - t0) / 60, "train_loss": tot / len(X), "test_cd_mape": cdr})
            log(f"  gnn epoch {ep}  {hist[-1]['minutes']:.1f} min  loss {tot / len(X):.4f}  test Cd err {cdr:.2f} %")
        if ckpt and (ep % checkpoint_every == 0 or done()):
            save_checkpoint()
    return model, norm, hist, lay


def predict_gnn(model: AeroGNN, norm: dict[str, np.ndarray], lay: Layout, recs: Sequence[dict[str, Any]]):
    """(n, 3) predicted (Cl, Cd, Cm) for ``recs`` and, per field, the RMSE against the runs' fields and their range."""
    X, E, Yf, Yc = graph_data(lay, recs)
    ei, wall = torch.from_numpy(lay.edge_index), torch.arange(len(lay.i_sel))
    out_c, out_f = [], []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(X), 16):
            y, c = model(torch.from_numpy(X[b:b + 16]), ei, torch.from_numpy(E[b:b + 16]), wall)
            out_c.append(c.numpy() * norm["coef_std"] + norm["coef_mean"])
            out_f.append(y.numpy() * norm["field_std"] + norm["field_mean"])
    C = np.concatenate(out_c)
    F = np.concatenate(out_f)
    cl, cd, cm = coef_from_targets(C)
    ferr = {}
    for k, name in enumerate(FIELDS):
        e = F[..., k] - Yf[..., k]
        ferr[name] = {"rmse": float(np.sqrt((e ** 2).mean())), "range": float(np.ptp(Yf[..., k]))}
    return np.stack([cl, cd, cm], 1), ferr


def _points(rs, p, sd=None):
    out = []
    for i, r in enumerate(rs):
        d = {"id": r["id"], "alpha": r["alpha"], "true": coefs([r])[0].tolist(), "pred": p[i].tolist()}
        if sd is not None:
            d["sd"] = sd[i].tolist()
        out.append(d)
    return out


def train_surrogates(recs: Sequence[dict[str, Any]], gnn_minutes: float = 60.0, gnn_from: str | None = None,
                     log: Callable[[str], Any] = print, *, mlp_k: int = 5, mlp_width: int = 96,
                     mlp_epochs: int = 6000, gnn_epochs: int | None = None,
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split, train both surrogates and measure them. Returns ``(bundle, metrics)``.

    ``bundle`` is the dict :meth:`.optimize.Engine.load` reads (``torch.save`` it); ``metrics`` holds the counts,
    :func:`metrics` per model on the held-out shapes (``*_test``) and the NACA references (``*_reference``), the
    per-run points and the graph network's training history. ``gnn_from`` reuses the graph network (and its
    metrics, from the ``.metrics.json`` next to it) of an earlier bundle; ``gnn_minutes = 0`` (and no
    ``gnn_epochs``) skips it.
    """
    tr, te, ref = split(recs)
    log(f"{len(tr)} training runs ({len({r['shape_id'] for r in tr})} shapes), {len(te)} test runs "
        f"({len({r['shape_id'] for r in te})} shapes), {len(ref)} NACA reference runs")
    t = time.time()
    mlp, mnorm = train_mlp(tr, mlp_k, mlp_width, mlp_epochs)
    log(f"MLP ensemble trained in {time.time() - t:.0f} s")
    res: dict[str, Any] = {"extra_runs": len([r for r in tr if r["set"] == "verify"]),
                           "counts": {"train": len(tr), "test": len(te), "reference": len(ref),
                                      "train_shapes": len({r['shape_id'] for r in tr}),
                                      "test_shapes": len({r['shape_id'] for r in te})}}
    for name, rs in (("test", te), ("reference", ref)):
        if not rs:
            continue
        p, sd = predict_mlp(mlp, mnorm, rs)
        res[f"mlp_{name}"] = metrics(coefs(rs), p)
        res[f"mlp_{name}_points"] = _points(rs, p, sd)
    log(json.dumps({k: v for k, v in res.items() if not k.endswith("points")}, indent=1))
    bundle: dict[str, Any] = {"mlp_state": mlp.state_dict(), "mlp_k": mlp_k, "mlp_width": mlp_width, "mlp_norm": mnorm,
                              "gnn_state": None,
                              "meta": {"reynolds": recs[0]["reynolds"], "trained": time.strftime("%Y-%m-%d"),
                                       "runs": len(recs), "test_shapes": sorted({r["shape_id"] for r in te})}}
    if gnn_from:
        old = torch.load(gnn_from, map_location="cpu", weights_only=False)
        bundle.update({k: old[k] for k in ("gnn_state", "gnn_norm", "gnn_node_in", "gnn_hidden", "gnn_mp")})
        with open(gnn_from.replace(".pt", ".metrics.json")) as f:
            om = json.load(f)
        res.update({k: v for k, v in om.items() if k.startswith("gnn")})
    elif gnn_minutes > 0 or gnn_epochs:
        hidden, mp = 48, 8
        gnn, gnorm, hist, lay = train_gnn(tr, te, gnn_minutes if gnn_minutes > 0 else None, log=log,
                                          epochs=gnn_epochs, hidden=hidden, mp=mp)
        bundle.update(gnn_state=gnn.state_dict(), gnn_norm=gnorm, gnn_node_in=gnn.gnn.node_in_dim, gnn_hidden=hidden,
                      gnn_mp=mp)
        res["gnn_history"] = hist
        for name, rs in (("test", te), ("reference", ref)):
            if not rs:
                continue
            p, ferr = predict_gnn(gnn, gnorm, lay, rs)
            res[f"gnn_{name}"] = metrics(coefs(rs), p)
            res[f"gnn_{name}_fields"] = ferr
            res[f"gnn_{name}_points"] = _points(rs, p)
        log(json.dumps({k: v for k, v in res.items() if k.startswith("gnn") and not k.endswith(("points", "history"))},
                       indent=1))
    return bundle, res
