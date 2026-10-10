"""Bumper-beam surrogate: reproduction of the weekend Transolver + PINNeAPPle improvements.

Target (as in the reproduced write-up): x-displacement u_x of every node at 101 frames
(0..100 ms), with an extra loss term on the reference node 1806.

Split: the write-up's 10 held-out experiments are the test set (Exp 5,15,16,19,30,33,37,83,88,96);
from the remaining 90, 10 fixed runs are validation (early stopping / model selection) and 80
are training. Test is evaluated once, at the end.

Models
  A. Reproduction  -- TransolverLite (the write-up's block, d=64, 4 heads, 8 slices, 3 blocks),
                      inputs [x,y,z (0-1), t_DP600, t_DP1000], loss MSE(field) + 2 MSE(node 1806)
  B. Improved      -- PINNeAPPle native Transolver (Physics-Attention), per-node own gauge and
                      material one-hot, u(t=0)=0 imposed, 3-seed deep ensemble
  C. Classical     -- POD + Gaussian-process regression on (DV1, DV2); nearest neighbour
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from pinneapple_neural.architectures.neural_operators.transolver import Transolver, TransolverLite  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
RES = ROOT / ("results_smoke" if SMOKE else "results")
RES.mkdir(exist_ok=True)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TEST_IDS = [5, 15, 16, 19, 30, 33, 37, 83, 88, 96]
VAL_IDS = [3, 12, 24, 41, 50, 58, 67, 74, 91, 99]
OUT = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(RES / "log.txt", "a") as fh:
        fh.write(s + "\n")


def save():
    (RES / "results.json").write_text(json.dumps(OUT, indent=1, default=float))


class Data:
    def __init__(self):
        f = h5py.File(ROOT / "data" / "training_data.hdf5", "r")
        self.ids = sorted(int(k.split("_")[1]) for k in f["outputs"].keys())
        e0 = f"Exp_{self.ids[0]}"
        self.xyz = f[f"inputs/{e0}/nodes"][()]
        self.part = f[f"inputs/{e0}/part_ids"][()]
        elems = f[f"inputs/{e0}/elements"][()]
        self.ref = int(f.attrs["reference_node_index"])
        self.dv = np.array([[f[f"inputs/Exp_{i}/thickness_dv1"][()], f[f"inputs/Exp_{i}/thickness_dv2"][()]] for i in self.ids],
                           np.float32)
        self.ux = np.stack([f[f"outputs/Exp_{i}/displacement_field"][:, :, 0] for i in self.ids]).astype(np.float32)  # R,N,T
        self.energy_err = [float(f[f"outputs/Exp_{i}"].attrs["energy_error_pct_max"]) for i in self.ids]
        self.t = f[f"outputs/{e0}/time"][()]
        f.close()
        # node -> material group from element part ids (parts with prop 2/7 = DP1000, 1/6 = DP600)
        dp1000_parts = {6, 7, 8, 9, 10, 17, 18, 19, 20, 21}
        dp600_parts = {4, 5, 15, 16}
        self.mat = np.zeros((len(self.xyz), 3), np.float32)  # [DP600, DP1000, other]
        for e, p in zip(elems, self.part):
            nodes = e[e >= 0]
            col = 1 if p in dp1000_parts else (0 if p in dp600_parts else 2)
            self.mat[nodes, col] = 1
        self.mat[self.mat.sum(1) == 0, 2] = 1
        lo, hi = self.xyz.min(0), self.xyz.max(0)
        self.xyzn = ((self.xyz - lo) / (hi - lo)).astype(np.float32)
        idx = {i: k for k, i in enumerate(self.ids)}
        self.test = [idx[i] for i in TEST_IDS if i in idx]
        self.val = [idx[i] for i in VAL_IDS if i in idx]
        self.train = [k for k in range(len(self.ids)) if k not in self.test and k not in self.val]
        if SMOKE:
            self.train = self.train[:6]
        self.mu = float(self.ux[self.train].mean())
        self.sd = float(self.ux[self.train].std())
        # node-wise scaling (training runs only) used by the improved model
        self.mu_n = self.ux[self.train].mean((0, 2))[:, None].astype(np.float32)
        self.sd_n = (self.ux[self.train].std((0, 2))[:, None] + 1e-2).astype(np.float32)

    def norm(self, kind):
        return (self.mu, self.sd) if kind == "lite" else (self.mu_n, self.sd_n)

    def feats_lite(self, r):
        """Write-up features: [x,y,z normalised, t_DP600, t_DP1000] broadcast to every node."""
        d1, d2 = self.dv[r]
        return np.concatenate([self.xyzn, np.full((len(self.xyz), 1), d2), np.full((len(self.xyz), 1), d1)], 1)

    def feats_improved(self, r):
        """Per-node own gauge (0 for non-shell/rigid nodes) + material one-hot + both design variables."""
        d1, d2 = self.dv[r]
        own = self.mat[:, 0] * d2 + self.mat[:, 1] * d1
        return np.concatenate([self.xyzn, own[:, None], self.mat, np.full((len(self.xyz), 1), d1),
                               np.full((len(self.xyz), 1), d2)], 1).astype(np.float32)


def batches(D, runs, kind, rng, bs):
    order = rng.permutation(runs)
    for i in range(0, len(order), bs):
        rr = order[i:i + bs]
        f = D.feats_lite if kind == "lite" else D.feats_improved
        x = torch.tensor(np.stack([f(r) for r in rr]), device=DEV)
        mu, sd = D.norm(kind)
        y = torch.tensor((D.ux[rr] - mu) / sd, device=DEV)
        yield x, y


def train(D, kind, seed, epochs, lr, patience, bs=2):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = D.ux.shape[2]
    if kind == "lite":
        m = TransolverLite(in_dim=5, out_dim=T, d_model=64, n_heads=4, n_slices=8, n_blocks=3, dropout=0.1).to(DEV)
    else:
        m = Transolver(in_dim=9, out_dim=T, dim=128, depth=4, heads=8, slices=32, dropout=0.0).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4 if kind != "lite" else 0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if kind != "lite" else None
    best, state, bad, hist, t0 = 1e9, None, 0, [], time.time()
    for ep in range(epochs):
        m.train()
        tl = 0.0
        for x, y in batches(D, D.train, kind, rng, bs):
            p = forward(m, x, kind)
            loss = ((p - y) ** 2).mean() + 2.0 * ((p[:, D.ref] - y[:, D.ref]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tl += float(loss)
        if sch:
            sch.step()
        v = evaluate(D, m, D.val, kind)
        hist.append({"epoch": ep + 1, "train_loss": tl, "val_field_rel_l2": v["field_rel_l2"], "val_node_rmse": v["node_rmse_mm"]})
        score = v["field_rel_l2"] + v["node_rmse_mm"] / max(1e-9, np.abs(D.ux[D.val][:, D.ref]).std())
        if score < best:
            best, bad, state = score, 0, {k: x.detach().clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
        if (ep + 1) % 10 == 0:
            log(f"  [{kind} s{seed}] ep {ep + 1}: train {tl:.4f} val relL2 {v['field_rel_l2']:.4f} node RMSE {v['node_rmse_mm']:.3f} mm ({time.time() - t0:.0f}s)")
        if bad >= patience:
            log(f"  early stop at {ep + 1}")
            break
    m.load_state_dict(state)
    return m, {"history": hist, "train_s": time.time() - t0, "epochs": len(hist),
               "n_params": sum(p.numel() for p in m.parameters())}


Z0 = None  # (N,1) normalised value of u = 0 per node, set once the data scaling is known


def forward(m, x, kind):
    y = m(x).y
    if kind != "lite":  # u(t=0) = 0 exactly (physical zero == (0 - mu)/sd in normalised units)
        y = torch.cat([Z0.expand(y.shape[0], -1, 1), y[..., 1:]], -1)
    return y


@torch.no_grad()
def predict(D, m, runs, kind):
    m.eval()
    f = D.feats_lite if kind == "lite" else D.feats_improved
    out = []
    for r in runs:
        mu, sd = D.norm(kind)
        y = forward(m, torch.tensor(f(r)[None], device=DEV), kind)[0].float().cpu().numpy() * sd + mu
        out.append(y)
    return np.stack(out)


def evaluate(D, m, runs, kind):
    return scores(D, predict(D, m, runs, kind), runs)


def scores(D, P, runs):
    Y = D.ux[runs]
    e = P - Y
    node_e = e[:, D.ref]
    out = {"field_rel_l2": float(np.linalg.norm(e) / np.linalg.norm(Y)),
           "field_rmse_mm": float(np.sqrt((e ** 2).mean())),
           "node_rmse_mm": float(np.sqrt((node_e ** 2).mean())),
           "node_max_abs_mm": float(np.abs(node_e).max()),
           "node_r2": float(1 - (node_e ** 2).sum() / ((Y[:, D.ref] - Y[:, D.ref].mean()) ** 2).sum()),
           "node_peak_err_mm": float(np.abs(P[:, D.ref].min(1) - Y[:, D.ref].min(1)).mean()),
           "node_final_err_mm": float(np.abs(P[:, D.ref, -1] - Y[:, D.ref, -1]).mean()),
           "per_run_node_rmse_mm": {int(D.ids[r]): float(np.sqrt((node_e[k] ** 2).mean())) for k, r in enumerate(runs)}}
    # error restricted to the deformable shell structure (excludes the rigid/other group)
    shell = D.mat[:, 2] == 0
    out["shell_rel_l2"] = float(np.linalg.norm(e[:, shell]) / np.linalg.norm(Y[:, shell]))
    return out


# ---------------------------------------------------------------------------
# classical baselines on the two design variables
# ---------------------------------------------------------------------------

def pod_gp(D, train_runs, test_runs, energy=0.9999):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    Y = D.ux[train_runs].reshape(len(train_runs), -1)
    mean = Y.mean(0)
    U, S, Vt = np.linalg.svd(Y - mean, full_matrices=False)
    r = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), energy) + 1)
    C = (Y - mean) @ Vt[:r].T
    X = D.dv[train_runs]
    preds, stds = [], []
    for j in range(r):
        k = ConstantKernel(1.0) * Matern(length_scale=[0.5, 0.5], nu=2.5) + WhiteKernel(1e-4)
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(X, C[:, j])
        mu, sd = gp.predict(D.dv[test_runs], return_std=True)
        preds.append(mu); stds.append(sd)
    P = (np.stack(preds, 1) @ Vt[:r] + mean).reshape(len(test_runs), *D.ux.shape[1:])
    Pstd = np.sqrt((np.stack(stds, 1) ** 2) @ (Vt[:r] ** 2)).reshape(len(test_runs), *D.ux.shape[1:])
    return P, Pstd, r


def nearest(D, train_runs, test_runs):
    d = ((D.dv[test_runs][:, None] - D.dv[train_runs][None]) ** 2).sum(-1)
    return D.ux[np.array(train_runs)[d.argmin(1)]]


# ---------------------------------------------------------------------------

def main():
    global Z0
    D = Data()
    Z0 = torch.tensor(-D.mu_n / D.sd_n, device=DEV)
    log(f"runs={len(D.ids)} train={len(D.train)} val={len(D.val)} test={len(D.test)} nodes={len(D.xyz)} T={D.ux.shape[2]} dev={DEV}")
    OUT["data"] = {"n_runs": len(D.ids), "nodes": len(D.xyz), "frames": int(D.ux.shape[2]),
                   "energy_error_pct_max": {"max": max(D.energy_err), "median": float(np.median(D.energy_err))},
                   "train_ids": [D.ids[k] for k in D.train], "val_ids": [D.ids[k] for k in D.val],
                   "test_ids": [D.ids[k] for k in D.test], "ux_scale_mm": D.sd,
                   "design_test": D.dv[D.test].tolist(), "design_train": D.dv[D.train].tolist()}
    ep = 3 if SMOKE else 400
    preds = {}
    # A. reproduction
    t0 = time.time()
    mA, infoA = train(D, "lite", 0, ep, 1e-3, patience=60)
    preds["Reproduction: TransolverLite (write-up)"] = predict(D, mA, D.test, "lite")
    OUT["reproduction"] = {"info": infoA, "test": scores(D, preds["Reproduction: TransolverLite (write-up)"], D.test)}
    t = time.time(); predict(D, mA, D.test[:1], "lite"); OUT["reproduction"]["inference_s_per_run"] = time.time() - t
    log("A:", json.dumps({k: v for k, v in OUT["reproduction"]["test"].items() if k != "per_run_node_rmse_mm"}))
    save()
    # B. improved ensemble
    members, infos = [], []
    for s in range(1 if SMOKE else 3):
        m, info = train(D, "improved", s, 2 if SMOKE else 300, 5e-4, patience=40)
        members.append(predict(D, m, D.test, "improved"))
        infos.append(info)
        if s == 0:
            t = time.time(); predict(D, m, D.test[:1], "improved"); inf_b = time.time() - t
            torch.save(m.state_dict(), RES / "improved_seed0.pt")
    Pb = np.mean(members, 0)
    preds["Improved: PINNeAPPle Transolver ensemble"] = Pb
    OUT["improved"] = {"info": infos, "test": scores(D, Pb, D.test), "inference_s_per_run": inf_b,
                       "members_test": [scores(D, p, D.test) for p in members], "ensemble_std": np.std(members, 0).mean()}
    log("B:", json.dumps({k: v for k, v in OUT["improved"]["test"].items() if k != "per_run_node_rmse_mm"}))
    save()
    # C. classical
    Pg, Pgs, r = pod_gp(D, D.train + D.val, D.test)
    preds["POD + Gaussian process (DV1, DV2)"] = Pg
    preds["Nearest design point"] = nearest(D, D.train + D.val, D.test)
    OUT["classical"] = {"pod_modes": r, "pod_gp": scores(D, Pg, D.test), "nearest": scores(D, preds["Nearest design point"], D.test)}
    # leave-one-out on all 100 runs for the GP (cheap) -> distribution, not a single split
    if not SMOKE:
        loo = []
        for k in range(len(D.ids)):
            rest = [j for j in range(len(D.ids)) if j != k]
            P, _, _ = pod_gp(D, rest, [k])
            loo.append(np.sqrt(((P[0, D.ref] - D.ux[k, D.ref]) ** 2).mean()))
        OUT["classical"]["pod_gp_loo_node_rmse_mm"] = {"median": float(np.median(loo)), "p95": float(np.percentile(loo, 95)),
                                                       "max": float(np.max(loo)), "all": loo}
    log("C:", OUT["classical"]["pod_gp"]["node_rmse_mm"], OUT["classical"]["nearest"]["node_rmse_mm"])
    save()
    # uncertainty + VeriPhysics
    veriphysics(D, preds, members, Pg, Pgs)
    np.savez_compressed(RES / "test_node_histories.npz", t=D.t, truth=D.ux[D.test][:, D.ref],
                        ids=np.array([D.ids[k] for k in D.test]),
                        **{k.split(":")[0].replace(" ", "_").replace("+", "").replace("(", "").replace(")", "").replace(",", ""):
                           v[:, D.ref] for k, v in preds.items()})
    np.savez_compressed(RES / "test_fields_final.npz", xyz=D.xyz, truth=D.ux[D.test][:, :, [30, 60, 100]],
                        improved=Pb[:, :, [30, 60, 100]], lite=preds["Reproduction: TransolverLite (write-up)"][:, :, [30, 60, 100]],
                        mat=D.mat)
    save()


def veriphysics(D, preds, members, Pg, Pgs):
    import torch as T
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
    from pinneapple_analysis.verification.physics_confidence_score import CalibrationSummary, compute_physics_confidence
    from pinneapple_data.physics_case import PhysicsCase
    from pinneapple_pdb.benchmarks import BenchmarkEntry, register_benchmark

    tr = np.arange(D.ux.shape[2], dtype=np.float32)
    xs = np.array([(D.ids[r], t) for r in D.test for t in tr], np.float32)
    ys = D.ux[D.test][:, D.ref].reshape(-1, 1).astype(np.float32)

    @register_benchmark("openradioss_bumper_node1806_heldout")
    def _b():
        return BenchmarkEntry(name="openradioss_bumper_node1806_heldout",
                              description="u_x(t) at node 1806, 10 held-out OpenRadioss runs (write-up test IDs)",
                              reference_source="OpenRadioss latest-20260728, Bumper Beam example (Altair, CC BY-NC 4.0)",
                              x_vars=("exp", "frame"), y_vars=("ux",), reference_x=xs, reference_y=ys)

    out = {}
    sd_ens = np.std(members, 0) if len(members) > 1 else None
    for name, P in preds.items():
        case = PhysicsCase(name=name, results={"exp": xs[:, 0], "frame": xs[:, 1], "ux": P[:, D.ref].reshape(-1)},
                           reference_benchmark="openradioss_bumper_node1806_heldout")
        b = case.validate_against_benchmark()
        cal = None
        std = sd_ens if name.startswith("Improved") else (Pgs if name.startswith("POD") else None)
        if std is not None:
            yp, yt = T.tensor(P[:, D.ref, 1:].ravel()), T.tensor(D.ux[D.test][:, D.ref, 1:].ravel())
            s = T.tensor(np.maximum(std[:, D.ref, 1:].ravel(), 1e-6))
            ece = CalibrationMetrics.expected_calibration_error(yp, yt, s)
            cov = CalibrationMetrics.coverage_at_level(yp, yt, s, alpha=0.1)
            cal = CalibrationSummary(ece=ece, coverage=cov, target_coverage=0.9, sharpness=float(s.mean()))
        sc = compute_physics_confidence(benchmark_comparison=b, calibration_metrics=cal)
        out[name] = {"overall_score": sc.overall_score, "coverage": sc.coverage, "rel_l2": b.relative_l2_error,
                     "components": [c.__dict__ for c in sc.components]}
        log(sc.summary())
    out["not_run"] = {"physics_guardrail": "no compiled PDE ProblemSpec exists for explicit shell-FE crash dynamics",
                      "numerical_convergence": "single mesh supplied by the example; no refinement study was run",
                      "geometry_ood": "geometry is identical across the DoE (only gauges vary)"}
    OUT["veriphysics"] = out
    save()


if __name__ == "__main__":
    main()
