"""Train the surrogates on the OpenFOAM runs and measure them on designs they never saw.

    python apps/aero_optimizer/tools/train_surrogate.py DATA_DIR OUT.pt [--gnn-minutes 60] [--extra DIR ...]

Split by shape (not by run): 15 % of the LHS shapes are held out with all their angles; the four NACA airfoils are
a second, independent test. Writes OUT.pt (both models + normalisation) and OUT.metrics.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.surrogate import (FIELDS, AeroGNN, Layout, MLPEnsemble, coef_from_targets,  # noqa: E402
                                              coef_targets, edge_features, node_features)

USABLE = ("converged", "steady-ish", "unsteady")


def load_runs(dirs):
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "runs", "*.json"))):
            r = json.load(open(f))
            if r.get("status") not in USABLE or "Cl" not in r:
                continue
            if r["status"] == "unsteady":                        # oscillating: use the mean of the kept snapshots
                H = np.array(r["history"])[-5:]
                r["Cl"], r["Cd"], r["Cm"] = (float(H[:, k].mean()) for k in (1, 2, 3))
            r["npz"] = f[:-5] + ".npz"
            recs.append(r)
    return recs


def split(recs, frac=0.15, seed=3):
    shapes = sorted({r["shape_id"] for r in recs if r["set"] != "reference"})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(shapes, max(1, int(round(frac * len(shapes)))), replace=False))
    tr = [r for r in recs if r["set"] != "reference" and r["shape_id"] not in test]
    te = [r for r in recs if r["set"] != "reference" and r["shape_id"] in test]
    ref = [r for r in recs if r["set"] == "reference"]
    return tr, te, ref


def metrics(true, pred):
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


def coefs(recs):
    return np.array([[r["Cl"], r["Cd"], r["Cm"]] for r in recs])


# ------------------------------------------------------------------ MLP ensemble
def train_mlp(tr, k=5, width=96, epochs=6000, seed=0):
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
    w = torch.tensor([1.0, 2.0, 1.0])                              # drag matters most
    for ep in range(epochs):
        opt.zero_grad()
        loss = sum(((net(Xt[b]) - Yt[b]) ** 2 * w).mean() for net, b in zip(model.nets, boots))
        loss.backward()
        opt.step()
        sched.step()
    return model, {"mean": mean, "std": std}


def predict_mlp(model, norm, recs):
    X = MLPEnsemble.inputs(np.array([r["shape"] for r in recs]), np.array([r["alpha"] for r in recs]))
    with torch.no_grad():
        y = model(torch.from_numpy(X)).numpy() * norm["std"] + norm["mean"]
    cl, cd, cm = coef_from_targets(y)
    return np.stack([cl.mean(0), cd.mean(0), cm.mean(0)], 1), np.stack([cl.std(0), cd.std(0), cm.std(0)], 1)


# ------------------------------------------------------------------ graph network
def graph_data(lay, recs):
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


def train_gnn(tr, te, minutes, seed=0, batch=4, log=print):
    lay = Layout()
    X, E, Yf, Yc = graph_data(lay, tr)
    Xv, Ev, Yfv, Ycv = graph_data(lay, te)
    fm, fs = Yf.reshape(-1, 4).mean(0), Yf.reshape(-1, 4).std(0)
    cm_, cs = Yc.mean(0), Yc.std(0)
    norm = {"field_mean": fm, "field_std": fs, "coef_mean": cm_, "coef_std": cs}
    Yfn, Ycn = (Yf - fm) / fs, (Yc - cm_) / cs
    torch.manual_seed(seed)
    model = AeroGNN(X.shape[-1])
    ei = torch.from_numpy(lay.edge_index)
    wall = torch.arange(len(lay.i_sel))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0, ep, hist = time.time(), 0, []
    budget = minutes * 60
    rng = np.random.default_rng(seed)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))         # noqa: E731
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * frac)))
        model.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for b in range(0, len(perm), batch):
            idx = perm[b:b + batch]
            y, c = model(T(X[idx]), ei, T(E[idx]), wall)
            lf = ((y - T(Yfn[idx])) ** 2).mean()
            lc = (((c - T(Ycn[idx])) ** 2) * torch.tensor([1.0, 2.0, 1.0])).mean()
            loss = lf + lc
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        ep += 1
        if ep % 5 == 0 or time.time() - t0 >= budget:
            model.eval()
            with torch.no_grad():
                _, cv = model(T(Xv), ei, T(Ev), wall)
            pv = coef_from_targets(cv.numpy() * cs + cm_)
            cdr = float(np.mean(np.abs(pv[1] - coef_from_targets(Ycv)[1]) / coef_from_targets(Ycv)[1]) * 100)
            hist.append({"epoch": ep, "minutes": (time.time() - t0) / 60, "train_loss": tot / len(X), "test_cd_mape": cdr})
            log(f"  gnn epoch {ep}  {hist[-1]['minutes']:.1f} min  loss {tot / len(X):.4f}  test Cd err {cdr:.2f} %")
    return model, norm, hist, lay


def predict_gnn(model, norm, lay, recs):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--gnn-minutes", type=float, default=60)
    a = ap.parse_args()
    recs = load_runs([a.data] + a.extra)
    tr, te, ref = split(recs)
    print(f"{len(tr)} training runs ({len({r['shape_id'] for r in tr})} shapes), {len(te)} test runs "
          f"({len({r['shape_id'] for r in te})} shapes), {len(ref)} NACA reference runs", flush=True)
    t = time.time()
    mlp, mnorm = train_mlp(tr)
    print(f"MLP ensemble trained in {time.time() - t:.0f} s", flush=True)
    res = {"counts": {"train": len(tr), "test": len(te), "reference": len(ref),
                      "train_shapes": len({r['shape_id'] for r in tr}), "test_shapes": len({r['shape_id'] for r in te})}}
    for name, rs in (("test", te), ("reference", ref)):
        p, sd = predict_mlp(mlp, mnorm, rs)
        res[f"mlp_{name}"] = metrics(coefs(rs), p)
        res[f"mlp_{name}_points"] = [{"id": r["id"], "alpha": r["alpha"], "true": coefs([r])[0].tolist(),
                                      "pred": p[i].tolist(), "sd": sd[i].tolist()} for i, r in enumerate(rs)]
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("points")}, indent=1), flush=True)
    bundle = {"mlp_state": mlp.state_dict(), "mlp_k": 5, "mlp_width": 96, "mlp_norm": mnorm, "gnn_state": None,
              "meta": {"reynolds": recs[0]["reynolds"], "trained": time.strftime("%Y-%m-%d"),
                       "runs": len(recs), "test_shapes": sorted({r["shape_id"] for r in te})}}
    if a.gnn_minutes > 0:
        gnn, gnorm, hist, lay = train_gnn(tr, te, a.gnn_minutes)
        bundle.update(gnn_state=gnn.state_dict(), gnn_norm=gnorm, gnn_node_in=gnn.gnn.node_in_dim, gnn_hidden=48, gnn_mp=8)
        res["gnn_history"] = hist
        for name, rs in (("test", te), ("reference", ref)):
            p, ferr = predict_gnn(gnn, gnorm, lay, rs)
            res[f"gnn_{name}"] = metrics(coefs(rs), p)
            res[f"gnn_{name}_fields"] = ferr
            res[f"gnn_{name}_points"] = [{"id": r["id"], "alpha": r["alpha"], "true": coefs([r])[0].tolist(),
                                          "pred": p[i].tolist()} for i, r in enumerate(rs)]
        print(json.dumps({k: v for k, v in res.items() if k.startswith("gnn") and not k.endswith(("points", "history"))}, indent=1))
    torch.save(bundle, a.out)
    json.dump(res, open(a.out.replace(".pt", ".metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
