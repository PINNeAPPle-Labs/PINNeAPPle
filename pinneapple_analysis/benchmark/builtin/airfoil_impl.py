"""Airfoil benchmark implementation (moved from PINNeAPPle-Benchmark experiment 09): MLP ensemble, MeshGraphNet,
POD + MLP fields, potential-flow and RANS PINNs on the aero app's 2-D airfoil (OpenFOAM v1912 simpleFoam,
k-omega SST, Re 4e6). The data directory is set by ``set_data_dir`` (the problem's prepared folder).
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from pinneapple_analysis.cost.equal_budget import (
    ComputeBudget,
    cosine_lr,
    set_lr,
    time_per_query,
    train_within_budget,
)
from pinneapple_design.aero.geometry import surfaces, wall_loop
from pinneapple_design.aero.mesh import GridSpec, _distances, cell_centres, grid_points
from pinneapple_design.aero.surrogate import (
    AeroGNN,
    Layout,
    MLPEnsemble,
    coef_from_targets,
    coef_targets,
    edge_features,
    node_features,
)
from pinneapple_physics.pinn_solver.factory.pinn_factory import (
    PINN,
    NeuralNetwork,
    PINNFactory,
    PINNProblemSpec,
)

from ..comparison import save_comparison

USABLE = ("converged", "steady-ish", "unsteady")
MAX_CL_OSC = 0.1                                                   # Cl swing over the last 500 iterations
_DIRS = {"air": None, "ood": None}


def set_data_dir(prepared) -> None:
    from pathlib import Path
    _DIRS["air"], _DIRS["ood"] = Path(prepared), Path(prepared) / "ood"


def load_runs(dirs):
    """The aero app's run filter (apps/aero_optimizer/tools/train_surrogate.py): usable status, deep stall dropped,
    oscillating runs averaged over their last snapshots."""
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "runs", "*.json"))):
            r = json.load(open(f))
            if r.get("status") not in USABLE or "Cl" not in r:
                continue
            if r["status"] == "unsteady" and r.get("Cl_osc", 9) > MAX_CL_OSC:
                continue
            if r["status"] == "unsteady":
                H = np.array(r["history"])[-5:]
                r["Cl"], r["Cd"], r["Cm"] = (float(H[:, k].mean()) for k in (1, 2, 3))
            r["npz"] = f[:-5] + ".npz"
            recs.append(r)
    return recs


def split(recs, frac=0.15, seed=3):
    """The aero app's split: 15 % of the LHS shapes held out with all their angles; the NACA polars apart."""
    shapes = sorted({r["shape_id"] for r in recs if r["set"] == "lhs"})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(shapes, max(1, int(round(frac * len(shapes)))), replace=False))
    tr = [r for r in recs if r["set"] != "reference" and r["shape_id"] not in test]
    te = [r for r in recs if r["set"] != "reference" and r["shape_id"] in test]
    ref = [r for r in recs if r["set"] == "reference"]
    return tr, te, ref


SPACES = {
    "mlp": {"width": [64, 96, 128], "lr": [1e-3, 3e-3], "wd": [1e-6, 1e-4]},
    "mgn": {"hidden": [32, 48], "mp": [6, 8], "lr": [5e-4, 1e-3]},
    "pinn": {"width": [64, 128], "depth": [4, 6], "lr": [1e-3, 3e-3], "n_colloc": [2048, 4096]},
    "pinn_rans": {"width": [64, 128], "depth": [4, 6], "lr": [1e-3, 3e-3], "n_colloc": [2048, 4096]},
    "mlp_fields": {"modes": [16, 32], "width": [64, 128], "lr": [1e-3, 3e-3]},
}
R_FAR = 20.0
RE = 4e6
NU = 1 / RE
# RANS PINN options, off by default (the benchmark's runs use the defaults). Tried against the zero-flow collapse of the
# data-free RANS PINN (PINNeAPPle-Benchmark experiment 09): an exact far field ("hard") and a share of collocation points
# even in log r. In 300 s on NACA 2412 at 4 deg neither moved the velocity error (0.96-0.98 against 0.96 without them).
RANS_FAR = "soft"
RANS_MIX = 0.0
DELTA_W = 0.02                                                   # RANS PINN: no-slip factor tanh(d / DELTA_W)
KUTTA_W, GAMMA_LR = 10.0, 1e-2                                   # fixed before the runs (see README)


def splits():
    recs = load_runs([str(_DIRS["air"])])
    tr, te, ref = split(recs)
    shapes = sorted({r["shape_id"] for r in tr})
    rng = np.random.default_rng(5)
    val_shapes = set(rng.choice(shapes, min(10, max(1, len(shapes) // 5)), replace=False))   # 10 of the 85 shapes
    out = {"train": [r for r in tr if r["shape_id"] not in val_shapes],
           "val": [r for r in tr if r["shape_id"] in val_shapes], "train_all": tr, "test": te, "naca": ref}
    if (_DIRS["ood"] / "runs").exists():                          # the OOD runs: never used for training
        out["ood"] = sorted(load_runs([str(_DIRS["ood"])]), key=lambda r: r["id"])
    return out


def eval_splits(final: bool, S: dict):
    return tuple(k for k in ("test", "naca", "ood") if k in S) if final else ("val",)


def coefs(recs):
    return np.array([[r["Cl"], r["Cd"], r["Cm"]] for r in recs])


def coef_scores(true: np.ndarray, pred: np.ndarray) -> dict:
    return {"cl_abs_err": np.abs(pred[:, 0] - true[:, 0]).tolist(),
            "cd_rel_err": (np.abs(pred[:, 1] - true[:, 1]) / true[:, 1]).tolist(),
            "cm_abs_err": np.abs(pred[:, 2] - true[:, 2]).tolist(),
            "cl_abs_err_mean": float(np.abs(pred[:, 0] - true[:, 0]).mean()),
            "cd_rel_err_mean": float((np.abs(pred[:, 1] - true[:, 1]) / true[:, 1]).mean()),
            "cm_abs_err_mean": float(np.abs(pred[:, 2] - true[:, 2]).mean())}


def field_scores(true: np.ndarray, pred: np.ndarray) -> dict:
    """Relative L2 per run of the velocity (Ux, Uy together) and of the pressure, on the graph nodes."""
    u = np.linalg.norm((pred[..., 1:3] - true[..., 1:3]).reshape(len(true), -1), axis=1) / np.linalg.norm(
        true[..., 1:3].reshape(len(true), -1), axis=1)
    p = np.linalg.norm(pred[..., 0] - true[..., 0], axis=1) / np.linalg.norm(true[..., 0], axis=1)
    return {"u_rel_l2": u.tolist(), "p_rel_l2": p.tolist(), "u_rel_l2_mean": float(u.mean()),
            "p_rel_l2_mean": float(p.mean())}


# ------------------------------------------------------------------------------------------------- MLP
def train_mlp(cfg: dict, seed: int, budget: ComputeBudget, *, final: bool) -> dict:
    S = splits()
    tr = S["train_all"] if final else S["train"]
    X = MLPEnsemble.inputs(np.array([r["shape"] for r in tr]), np.array([r["alpha"] for r in tr]))
    Y = coef_targets(*coefs(tr).T)
    mean, std = Y.mean(0), Y.std(0)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(((Y - mean) / std).astype(np.float32))
    torch.manual_seed(seed)
    model = MLPEnsemble(5, cfg["width"])
    rng = np.random.default_rng(seed)
    boots = [torch.from_numpy(rng.integers(0, len(X), len(X))) for _ in range(5)]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    w = torch.tensor([1.0, 2.0, 1.0])

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        loss = sum(((net(Xt[b]) - Yt[b]) ** 2 * w).mean() for net, b in zip(model.nets, boots, strict=False))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)

    def predict(recs):
        Xq = torch.from_numpy(MLPEnsemble.inputs(np.array([r["shape"] for r in recs]), np.array([r["alpha"] for r in recs])))
        with torch.no_grad():
            y = model(Xq).numpy() * std + mean
        cl, cd, cm = coef_from_targets(y)
        return np.stack([cl.mean(0), cd.mean(0), cm.mean(0)], 1)

    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()),
           "n_train_runs": len(tr)}
    for name in eval_splits(final, S):
        out[name] = coef_scores(coefs(S[name]), predict(S[name]))
        out[name]["ids"] = [r["id"] for r in S[name]]
    x1 = torch.from_numpy(MLPEnsemble.inputs(np.array([S["val"][0]["shape"]]), np.array([S["val"][0]["alpha"]])))
    out["infer_batch1"] = time_per_query(lambda: model(x1), repeats=50, warmup=5)
    return out


# ------------------------------------------------------------------------------------------------- MeshGraphNet
def graph_data(lay, recs):
    X, E, Yf, Yc = [], [], [], []
    for r in recs:
        s = np.array(r["shape"])
        pos = lay.positions(s)
        X.append(node_features(lay, s, r["alpha"], pos))
        E.append(edge_features(lay, pos))
        z = np.load(r["npz"])
        nu = 1.0 / r["reynolds"]
        Yf.append(np.stack([z["p"], z["Ux"], z["Uy"], np.log1p(z["nut"] / nu)], 1)[lay.cell_index])
        Yc.append(coef_targets(r["Cl"], r["Cd"], r["Cm"]))
    return np.stack(X), np.stack(E), np.stack(Yf).astype(np.float32), np.stack(Yc).astype(np.float32)


def train_mgn(cfg: dict, seed: int, budget: ComputeBudget, *, final: bool, batch: int = 4,
              comparison_path=None) -> dict:
    S = splits()
    tr = S["train_all"] if final else S["train"]
    lay = Layout()
    X, E, Yf, Yc = graph_data(lay, tr)
    fm, fs = Yf.reshape(-1, 4).mean(0), Yf.reshape(-1, 4).std(0)
    cm_, cs = Yc.mean(0), Yc.std(0)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))      # noqa: E731
    Xt, Et, Yfn, Ycn = T(X), T(E), T((Yf - fm) / fs), T((Yc - cm_) / cs)
    torch.manual_seed(seed)
    model = AeroGNN(X.shape[-1], hidden=cfg["hidden"], mp=cfg["mp"])
    ei, wall = torch.from_numpy(lay.edge_index), torch.arange(len(lay.i_sel))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    cw = torch.tensor([1.0, 2.0, 1.0])
    state = {"perm": rng.permutation(len(X)), "pos": 0}

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        if state["pos"] + batch > len(X):
            state["perm"], state["pos"] = rng.permutation(len(X)), 0
        idx = torch.from_numpy(state["perm"][state["pos"]:state["pos"] + batch])
        state["pos"] += batch
        y, c = model(Xt[idx], ei, Et[idx], wall)
        loss = ((y - Yfn[idx]) ** 2).mean() + (((c - Ycn[idx]) ** 2) * cw).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    model.train()
    run = train_within_budget(step, budget)
    model.eval()
    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in model.parameters()), "n_train_runs": len(tr)}
    for name in eval_splits(final, S):
        Xq, Eq, Yfq, _ = graph_data(lay, S[name])
        C, F = [], []
        with torch.no_grad():
            for b in range(0, len(Xq), 8):
                y, c = model(T(Xq[b:b + 8]), ei, T(Eq[b:b + 8]), wall)
                C.append(c.numpy() * cs + cm_)
                F.append(y.numpy() * fs + fm)
        F = np.concatenate(F)
        cl, cd, cm = coef_from_targets(np.concatenate(C))
        out[name] = coef_scores(coefs(S[name]), np.stack([cl, cd, cm], 1))
        out[name].update(field_scores(Yfq, F))
        out[name]["ids"] = [r["id"] for r in S[name]]
        if name == "test" and final and len(F):          # the report's standard comparison figure (#63)
            pos = lay.positions(np.array(S[name][0]["shape"]))
            save_comparison(comparison_path, Yfq[0, :, 0], F[0, :, 0], x=pos[:, 0], y=pos[:, 1], quantity="p",
                            unit="Pa", instance=S[name][0]["id"])
    r0 = S["val"][0]
    s0 = np.array(r0["shape"])

    def one_design():                                             # features from the shape, then the network
        pos = lay.positions(s0)
        x = torch.from_numpy(node_features(lay, s0, r0["alpha"], pos))[None]
        e = torch.from_numpy(edge_features(lay, pos))[None]
        return model(x, ei, e, wall)
    out["infer_batch1"] = time_per_query(one_design, repeats=20, warmup=3)
    return out


# ------------------------------------------------------------------------------------------------- POD + MLP fields
def train_mlp_fields(cfg: dict, seed: int, budget: ComputeBudget, *, final: bool,
                     comparison_path=None) -> dict:
    """Field baseline for a like-for-like field comparison with the MeshGraphNet: POD of the training fields (p, Ux,
    Uy on the same graph nodes), and an MLP from the shape parameters and the angle to the POD coefficients."""
    S = splits()
    tr = S["train_all"] if final else S["train"]
    lay = Layout()

    def fields(recs):
        return np.stack([np.stack([np.load(r["npz"])[k] for k in ("p", "Ux", "Uy")], 1)[lay.cell_index]
                         for r in recs]).astype(np.float32)               # (n, nodes, 3)
    Y = fields(tr)
    fm, fs = Y.mean(0), Y.reshape(-1, 3).std(0)
    Z = ((Y - fm) / fs).reshape(len(Y), -1)
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    K = min(cfg["modes"], len(Z))
    basis = Vt[:K]
    coef = Z @ basis.T
    cm_, cs = coef.mean(0), coef.std(0) + 1e-8
    X = torch.from_numpy(MLPEnsemble.inputs(np.array([r["shape"] for r in tr]), np.array([r["alpha"] for r in tr])))
    T = torch.from_numpy(((coef - cm_) / cs).astype(np.float32))
    torch.manual_seed(seed)
    w = cfg["width"]
    net = nn.Sequential(nn.Linear(8, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
                        nn.Linear(w, K))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    wts = torch.from_numpy((sv[:K] / sv[0]).astype(np.float32))          # weight coefficients by their energy

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        loss = (((net(X) - T) ** 2) * wts).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    basis_t = torch.from_numpy(basis.astype(np.float32))

    def predict(recs):
        Xq = torch.from_numpy(MLPEnsemble.inputs(np.array([r["shape"] for r in recs]), np.array([r["alpha"] for r in recs])))
        with torch.no_grad():
            c = net(Xq).numpy() * cs + cm_
        return (c @ basis).reshape(len(recs), -1, 3) * fs + fm

    out = {"train": run.to_dict(), "n_params": sum(p.numel() for p in net.parameters()) + basis.size,
           "n_train_runs": len(tr), "pod_energy_kept": float((sv[:K] ** 2).sum() / (sv ** 2).sum())}
    for name in eval_splits(final, S):
        true, pred = fields(S[name]), predict(S[name])
        out[name] = field_scores(true, pred)
        out[name]["ids"] = [r["id"] for r in S[name]]
        if name == "test" and final:                     # the report's standard comparison figure (#63)
            pos = lay.positions(np.array(S[name][0]["shape"]))
            save_comparison(comparison_path, true[0, :, 0], pred[0, :, 0], x=pos[:, 0], y=pos[:, 1], quantity="p",
                            unit="Pa", instance=S[name][0]["id"])
    r0 = S["val"][0]
    x1 = torch.from_numpy(MLPEnsemble.inputs(np.array([r0["shape"]]), np.array([r0["alpha"]])))
    out["infer_batch1"] = time_per_query(lambda: (net(x1) @ basis_t), repeats=50, warmup=5)
    return out


# ------------------------------------------------------------------------------------------------- PINNs
def find_case(case: str) -> dict:
    S = splits()
    return next(r for k in ("val", "test", "naca", "ood") for r in S.get(k, []) if r["id"] == case)


def collocation_pool(shape, d_min: float) -> torch.Tensor:
    """Cell centres of the case's own O-grid (clustered at the wall, leading and trailing edges) inside r < R."""
    P = grid_points(shape, GridSpec())
    cc = cell_centres(P)                                          # (nj, ni, 2)
    d = _distances(GridSpec().first, GridSpec().radius, GridSpec().nj)
    dc = 0.5 * (d[:-1] + d[1:])
    pts = cc[(dc > d_min) & (dc < R_FAR - 1)].reshape(-1, 2)
    pts = pts[np.linalg.norm(pts - [0.25, 0], axis=1) < R_FAR - 0.5]
    return torch.tensor(pts, dtype=torch.float32)


def _features(x, y):
    r2 = (x - 0.25) ** 2 + y ** 2
    r = torch.sqrt(r2)
    return torch.cat([2 * (x - 0.25) / (1 + r), 2 * y / (1 + r), torch.log(r2) / 6], 1), r2   # bounded, O(1) near the body


class PotentialNet(nn.Module):
    """psi = U (y cos a - x sin a) + Gamma/(4 pi) ln r^2 + (1 - r^2/R^2) N(features), Gamma trainable."""

    def __init__(self, cfg: dict, alpha_deg: float):
        super().__init__()
        self.mlp = NeuralNetwork(num_inputs=3, num_outputs=1, num_layers=cfg["depth"], num_neurons=cfg["width"],
                                 activation=nn.Tanh())
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.a = math.radians(alpha_deg)

    def forward(self, X):
        x, y = X[:, :1], X[:, 1:2]
        feats, r2 = _features(x, y)
        base = y * math.cos(self.a) - x * math.sin(self.a) + self.gamma / (4 * math.pi) * torch.log(r2)
        return base + (1 - r2 / R_FAR ** 2) * self.mlp(feats)


def wall_distance(x, y, p0: torch.Tensor, e: torch.Tensor):
    """Distance to the wall polygon (segments p0 -> p0 + e), differentiable almost everywhere."""
    px, py = x - p0[:, 0], y - p0[:, 1]
    t = ((px * e[:, 0] + py * e[:, 1]) / (e[:, 0] ** 2 + e[:, 1] ** 2)).clamp(0, 1)
    dx, dy = px - t * e[:, 0], py - t * e[:, 1]
    return torch.sqrt(dx * dx + dy * dy + 1e-14).min(dim=1, keepdim=True).values


class RansNet(nn.Module):
    """(u, v, p) with no-slip exact on the wall: u = tanh(d / DELTA_W) (cos a + N_u), v likewise, p = N_p."""

    def __init__(self, cfg: dict, alpha_deg: float, wall: np.ndarray):
        super().__init__()
        self.mlp = NeuralNetwork(num_inputs=3, num_outputs=3, num_layers=cfg["depth"], num_neurons=cfg["width"],
                                 activation=nn.Tanh())
        self.a = math.radians(alpha_deg)
        w = torch.tensor(wall, dtype=torch.float32)
        self.register_buffer("p0", w)
        self.register_buffer("e", torch.roll(w, -1, 0) - w)
        self.hard_far = cfg.get("far", RANS_FAR) == "hard"

    def forward(self, X):
        x, y = X[:, :1], X[:, 1:2]
        feats, r2 = _features(x, y)
        n = self.mlp(feats)
        if self.hard_far:   # free stream exact on the far circle (cfg "far": "hard"); see RANS_FAR
            n = n * (1 - r2 / (R_FAR - 1) ** 2).clamp(min=0.0)
        damp = torch.tanh(wall_distance(x, y, self.p0, self.e) / DELTA_W)
        return torch.cat([damp * (math.cos(self.a) + n[:, :1]), damp * (math.sin(self.a) + n[:, 1:2]), n[:, 2:3]], 1)


class EddyViscosity:
    """Prandtl mixing length with van Driest damping, evaluated from the current network and frozen (detached) for the
    step (a Picard linearisation): nu_t = l^2 |S|, l = min(kappa d (1 - exp(-d+/26)), 0.085 delta), with the friction
    velocity and the boundary-layer thickness of a turbulent flat plate at the chord Reynolds number."""

    def __init__(self, model, p0, e):
        self.model, self.p0, self.e = model, p0, e
        cf = 0.0576 * RE ** -0.2
        self.u_tau, self.delta = math.sqrt(cf / 2), 0.37 * RE ** -0.2
        self._key, self._val = None, None

    def _eval(self, x, y):
        key = (x.data_ptr(), x.shape[0])
        if key != self._key:
            xx, yy = x.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(True)
            out = self.model(xx, yy)
            u, v = out[:, :1], out[:, 1:2]
            ux, uy = torch.autograd.grad(u.sum(), (xx, yy), create_graph=True)
            vx, vy = torch.autograd.grad(v.sum(), (xx, yy), create_graph=True)
            s = torch.sqrt(2 * ux ** 2 + 2 * vy ** 2 + (uy + vx) ** 2 + 1e-12)
            with torch.no_grad():
                d = wall_distance(xx, yy, self.p0, self.e)
                lm = torch.minimum(0.41 * d * (1 - torch.exp(-d * self.u_tau / NU / 26)),
                                   torch.full_like(d, 0.085 * self.delta))
            nut = lm ** 2 * s
            gx, gy = torch.autograd.grad(nut.sum(), (xx, yy))
            self._key, self._val = key, (nut.detach(), gx.detach(), gy.detach())
        return self._val

    def nut(self, x, y):
        return self._eval(x, y)[0]

    def nutx(self, x, y):
        return self._eval(x, y)[1]

    def nuty(self, x, y):
        return self._eval(x, y)[2]


POTENTIAL_RESIDUAL = "Derivative(psi(x, y), (x, 2)) + Derivative(psi(x, y), (y, 2))"


def rans_residuals() -> list[str]:
    """Steady incompressible RANS with an eddy viscosity nut(x, y) and its gradient (nutx, nuty): continuity and the
    x and y momentum equations, div((nu + nut)(grad u + grad u^T)) written out (the PINN's residuals)."""
    u, v, p = "u(x, y)", "v(x, y)", "p(x, y)"
    D = lambda f, *w: f"Derivative({f}, {', '.join(w)})"          # noqa: E731
    ux, uy, vx, vy = D(u, "x"), D(u, "y"), D(v, "x"), D(v, "y")
    lap = lambda f: f"({D(f, '(x, 2)')} + {D(f, '(y, 2)')})"     # noqa: E731
    return [f"{ux} + {vy}",
            f"{u}*{ux} + {v}*{uy} + {D(p, 'x')} - (nu + nut(x, y))*{lap(u)} - 2*nutx(x, y)*{ux}"
            f" - nuty(x, y)*({uy} + {vx})",
            f"{u}*{vx} + {v}*{vy} + {D(p, 'y')} - (nu + nut(x, y))*{lap(v)} - nutx(x, y)*({uy} + {vx})"
            f" - 2*nuty(x, y)*{vy}"]


def _velocity(model, x, y, kind="potential"):
    out = model(x, y)
    if kind == "rans":
        return out[:, :1], out[:, 1:2], out[:, 2:3]
    gx, gy = torch.autograd.grad(out.sum(), (x, y), create_graph=True)
    return gy, -gx, 0.5 * (1 - (gy ** 2 + gx ** 2))               # u = psi_y, v = -psi_x, Bernoulli


def _req(t):
    return t.clone().requires_grad_(True)


def train_pinn(cfg: dict, seed: int, budget: ComputeBudget, *, case: str, kind: str = "potential",
               ckpt: str | None = None) -> dict:
    """A PINN for one airfoil at one angle, no simulation data. kind = "potential" or "rans" (see module docstring).

    ``ckpt``: path for checkpoints every 5 minutes of training; an existing checkpoint is resumed (same budget)."""
    rec = find_case(case)
    shape, alpha = np.array(rec["shape"]), float(rec["alpha"])
    a = math.radians(alpha)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    if kind == "potential":
        pool = collocation_pool(shape, 1e-3)
        wl = torch.tensor(wall_loop(shape, 200, 4).xy, dtype=torch.float32)
        # Kutta: the dividing streamline psi = C leaves the trailing edge along the bisector of its two surfaces
        xs = np.array([0.98, 1.0])
        yu, yl = surfaces(shape, xs)
        te = np.array([1.0, 0.5 * (yu[1] + yl[1])])
        tang = np.array([1.0, 0.5 * ((yu[1] - yu[0]) + (yl[1] - yl[0])) / 0.02])
        tang /= np.linalg.norm(tang)
        wake = torch.tensor(te + np.linspace(0.005, 0.1, 8)[:, None] * tang, dtype=torch.float32)
        spec = PINNProblemSpec(pde_residuals=[POTENTIAL_RESIDUAL],
                               conditions=[{"name": "wall", "equation": "psi(x, y) - C", "weight": cfg.get("wall_w", 10.0)},
                                           {"name": "kutta_wake", "equation": "psi(x, y) - C",
                                            "weight": cfg.get("kutta_w", KUTTA_W)}],
                               independent_vars=["x", "y"], dependent_vars=["psi"], inverse_params=["C"])
        model = PINN(PotentialNet(cfg, alpha), inverse_params_names=["C"], initial_guesses={"C": 0.0})
        special = [model.net.gamma]
        lr_special = GAMMA_LR

        def batch():
            c = pool[torch.randint(0, len(pool), (cfg["n_colloc"],), generator=g)]
            return {"collocation": (_req(c[:, :1]), _req(c[:, 1:])),
                    "conditions": [(_req(wl[:, :1]), _req(wl[:, 1:])), (_req(wake[:, :1]), _req(wake[:, 1:]))]}
    else:
        pool = collocation_pool(shape, 0.0)
        wall = wall_loop(shape, 100, 4).xy
        model = PINN(RansNet(cfg, alpha, wall))
        nut = EddyViscosity(model, model.net.p0, model.net.e)
        u, v, p = "u(x, y)", "v(x, y)", "p(x, y)"
        spec = PINNProblemSpec(
            pde_residuals=rans_residuals(),
            conditions=[{"name": "far_u", "equation": f"{u} - ca"}, {"name": "far_v", "equation": f"{v} - sa"},
                        {"name": "far_p", "equation": f"{p}"}],
            independent_vars=["x", "y"], dependent_vars=["u", "v", "p"],
            constants={"nu": NU, "ca": math.cos(a), "sa": math.sin(a)},
            exogenous={"nut": nut.nut, "nutx": nut.nutx, "nuty": nut.nuty})
        th = torch.linspace(0, 2 * math.pi, 257)[:-1]
        far = torch.stack([0.25 + (R_FAR - 1) * torch.cos(th), (R_FAR - 1) * torch.sin(th)], 1)
        special, lr_special = [], 0.0

        mix = float(cfg.get("colloc_mix", RANS_MIX))

        def batch():
            n_mix = int(mix * cfg["n_colloc"])
            c = pool[torch.randint(0, len(pool), (cfg["n_colloc"] - n_mix,), generator=g)]
            if n_mix:   # extra points even in log r over the outer flow, which the wall-clustered O-grid barely samples
                rr = torch.exp(torch.empty(n_mix, 1).uniform_(math.log(0.55), math.log(R_FAR - 1), generator=g))
                tt = torch.empty(n_mix, 1).uniform_(0, 2 * math.pi, generator=g)
                c = torch.cat([c, torch.cat([0.25 + rr * torch.cos(tt), rr * torch.sin(tt)], 1)])
            f = (_req(far[:, :1]), _req(far[:, 1:]))
            return {"collocation": (_req(c[:, :1]), _req(c[:, 1:])), "conditions": [f, f, f]}
    loss_fn = PINNFactory(spec).generate_loss_function()
    rest = [q for q in model.parameters() if all(q is not t for t in special)]
    groups = [{"params": rest, "lr": cfg["lr"]}] + ([{"params": special, "lr": lr_special}] if special else [])
    opt = torch.optim.Adam(groups)
    lbfgs_from = 1.0 - float(cfg.get("lbfgs", 0.0))
    state = {"lbfgs": None}
    resume = None
    if ckpt and os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        g.set_state(st["gen"])
        resume = st["run"]

    def step(progress):
        b = batch()
        if progress >= lbfgs_from:                                # optional L-BFGS phase at the end of the budget
            if state["lbfgs"] is None:
                state["lbfgs"] = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, history_size=50,
                                                   line_search_fn="strong_wolfe")

            def closure():
                state["lbfgs"].zero_grad(set_to_none=True)
                lo, _ = loss_fn(model, b)
                lo.backward()
                return lo
            return state["lbfgs"].step(closure).detach()
        f = cosine_lr(progress / lbfgs_from if lbfgs_from < 1 else progress, 1.0)
        opt.param_groups[0]["lr"] = cfg["lr"] * f
        if special:
            opt.param_groups[1]["lr"] = lr_special * f
        loss, _ = loss_fn(model, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    def save(run_state):
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(), "run": run_state},
                       ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)

    run = train_within_budget(step, budget, checkpoint=save if ckpt else None, checkpoint_every_s=300.0, resume=resume)
    # loss components at the end (for the audit of the loss balance)
    comps = loss_fn(model, batch())[1]
    # forces from the wall: pressure, and for RANS the viscous shear, 1e-5 chord outside each wall face
    loop = wall_loop(shape, 200, 4).xy
    a_, b_ = loop, np.roll(loop, -1, 0)
    e = b_ - a_
    L = np.linalg.norm(e, axis=1)
    n_body = np.stack([e[:, 1], -e[:, 0]], 1) / L[:, None]
    mid = torch.tensor(0.5 * (a_ + b_) - 1e-5 * n_body, dtype=torch.float32)
    mx, my = _req(mid[:, :1]), _req(mid[:, 1:])
    uw, vw, pw = _velocity(model, mx, my, kind)
    F = (pw.detach().numpy() * n_body * L[:, None]).sum(0)
    if kind == "rans":
        t_hat = torch.tensor(e / L[:, None], dtype=torch.float32)
        ut = uw * t_hat[:, :1] + vw * t_hat[:, 1:]
        gx, gy = torch.autograd.grad(ut.sum(), (mx, my))
        dn = -(gx.numpy().ravel() * n_body[:, 0] + gy.numpy().ravel() * n_body[:, 1])   # derivative along the outward normal
        tau = NU * dn                                             # nu_t = 0 at the wall
        F = F + (tau[:, None] * e / L[:, None] * L[:, None]).sum(0)
    cd_w = float(F @ np.array([math.cos(a), math.sin(a)]) / 0.5)
    cl_w = float(F @ np.array([-math.sin(a), math.cos(a)]) / 0.5)
    # fields on the same graph nodes as the MeshGraphNet
    lay = Layout()
    pos = torch.tensor(lay.positions(shape), dtype=torch.float32)
    px, py = _req(pos[:, :1]), _req(pos[:, 1:])
    u, v, pf = (t.detach().numpy().ravel() for t in _velocity(model, px, py, kind))
    z = np.load(rec["npz"])
    true_f = np.stack([z["p"], z["Ux"], z["Uy"]], 1)[lay.cell_index][None]
    pred_f = np.stack([pf, u, v], 1)[None]
    true_c = coefs([rec])
    if kind == "potential":
        gamma = float(model.net.gamma.detach())
        cl, extra = 2 * gamma, {"gamma": gamma, "cl_kutta_joukowski": 2 * gamma,
                                "wall_constant": float(model.inverse_params["C"].detach())}
    else:
        cl, extra = cl_w, {}
    pred_c = np.array([[cl, max(cd_w, 1e-12), np.nan]])
    out = {"train": run.to_dict(), "n_params": sum(q.numel() for q in model.parameters()), "case": case, "kind": kind,
           "cl_wall_integral": cl_w, "cd_wall_integral": cd_w, "true": true_c[0].tolist(), "loss_components": comps,
           "result": {**coef_scores(true_c, pred_c), **field_scores(true_f, pred_f)}, **extra}
    out["infer_eval_fields"] = time_per_query(lambda: _velocity(model, _req(pos[:, :1]), _req(pos[:, 1:]), kind),
                                              repeats=5, no_grad=False)
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    return out
