"""Robust Bekker-Wong terramechanics surrogate with PINNeAPPle.

Stages
  1. audit      -- check every physics constraint of the original PINN against the solver
  2. original   -- re-run the ORIGINAL example (frozen copy, git HEAD) unchanged
  3. robust2d   -- same (slip, sinkage) task: data-only MLP vs. robust constrained PINN
                   (hard R2/R4 by construction + verified soft R3/R5), deep ensembles
  4. stress     -- label noise, small data, sinkage extrapolation
  5. parametric -- 8-D surrogate over soil parameters (in-distribution + out-of-distribution)
  6. rover      -- operating-point checks: sinkage for the wheel load, drawbar pull
  7. speed      -- quad vs batched Gauss-Legendre vs surrogate
  8. veriphysics-- PhysicsGuardrail + convergence + calibration + benchmark -> confidence score
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PINN_ROOT = ROOT.parents[2] / "PINNeAPPle"
sys.path.insert(0, str(PINN_ROOT))

from pinneapple_simulation.numerical_solvers.bekker_wong import BekkerWongSolver, bekker_wong_forces_torch  # noqa
from pinneapple_simulation.particle_dynamics.terramechanics import SoilParams, WheelParams  # noqa
from pinneapple_neural.architectures.modified_mlp import ModifiedMLP  # noqa

SMOKE = os.environ.get("SMOKE") == "1"
RES = ROOT / ("results_smoke" if SMOKE else "results")
RES.mkdir(exist_ok=True)
torch.set_num_threads(4)
SOIL, WHEEL = SoilParams(), WheelParams()
SOLVER = BekkerWongSolver(SOIL, WHEEL)
S_RANGE, Z_RANGE = (0.0, 0.75), (0.002, 0.058)
R_, B_ = WHEEL.R, WHEEL.b
W_WHEEL = WHEEL.weight_per_wheel(SOIL.g)
OUT = {}
ORIG_TORCH = None  # differentiable wrapper of the original model (for the guardrail)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(RES / "log.txt", "a") as fh:
        fh.write(s + "\n")


def save():
    (RES / "results.json").write_text(json.dumps(OUT, indent=1, default=float))


def lhs(n, lo, hi, seed):
    from scipy.stats import qmc
    return qmc.scale(qmc.LatinHypercube(d=len(lo), seed=seed).random(n), lo, hi)


def solve2d(P):
    return SOLVER.forces_batch(P[:, 0], P[:, 1], n_gl=48)


def metrics(Yp, Yt):
    names = ("Fx", "Fz", "My")
    out = {}
    for j, n in enumerate(names):
        e = Yp[:, j] - Yt[:, j]
        out[n] = {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
                  "rel_L2": float(np.linalg.norm(e) / np.linalg.norm(Yt[:, j])), "max_abs": float(np.abs(e).max())}
    return out


# ---------------------------------------------------------------------------
# 1. audit
# ---------------------------------------------------------------------------

def stage_audit():
    ss = np.linspace(0, 0.75, 76)
    out = {}
    for label, zr in (("preset_domain", Z_RANGE), ("extended_z_to_0.1m", (0.002, 0.10))):
        zs = np.linspace(*zr, 57)
        S, Z = np.meshgrid(ss, zs)
        Y = solve2d(np.c_[S.ravel(), Z.ravel()]).reshape(len(zs), len(ss), 3)
        Fx, Fz, My = Y[..., 0], Y[..., 1], Y[..., 2]
        th1 = np.arccos(1 - zs / R_)[:, None]
        mc_orig = SOIL.c * B_ * R_ * math.pi + Fz * SOIL.tan_phi - Fx
        mc_arc = SOIL.c * B_ * R_ * th1 + Fz * SOIL.tan_phi - Fx
        dFx = np.diff(Fx, axis=1)[:, ss[1:] <= 0.4]
        out[label] = {
            "R1_Fx_at_s0_range_N": [float(Fx[:, 0].min()), float(Fx[:, 0].max())],
            "R2_original_area_piBR_min_margin_N": float(mc_orig.min()),
            "R2_original_area_ever_active": bool((mc_orig < 0.05 * np.abs(Fx).max()).any()),
            "R2_arc_area_min_margin_N": float(mc_arc.min()), "R2_arc_violations": int((mc_arc < 0).sum()),
            "R3_monotone_s_le_0.4_violations": int((dFx < 0).sum()),
            "R4_min_margin_Nm": float((My - R_ * Fx).min()), "R4_violations": int((My < R_ * Fx).sum()),
            "R5_dFz_dz_violations": int((np.diff(Fz, axis=0) < 0).sum()),
            "Fz_range_N": [float(Fz.min()), float(Fz.max())], "n_points": int(Fx.size),
        }
    # fraction of the preset domain that the 40 kg lunar rover can actually reach
    zW = [SOLVER.sinkage_from_load(W_WHEEL, s) for s in np.linspace(0, 0.75, 16)]
    out["operating_sinkage_for_wheel_load_m"] = {"W_N": W_WHEEL, "z_min": float(min(zW)), "z_max": float(max(zW))}
    OUT["audit"] = out
    log("audit:", json.dumps(out, indent=1))
    save()


# ---------------------------------------------------------------------------
# 2. original model (frozen copy)
# ---------------------------------------------------------------------------

def load_original_module():
    """Exec the frozen original example against the frozen original preset residuals."""
    pre_src = (ROOT / "original" / "terramechanics_preset_ORIGINAL.py").read_text()
    pre_src = pre_src.split("# ---------------------------------------------------------------------------\n# Preset factory")[0]
    pre_src = pre_src.replace("from .registry import register_preset\n", "").replace(
        "from ..spec import PDETermSpec, ProblemSpec\n", "").replace("from ..conditions import InitialCondition\n", "")
    pre = types.ModuleType("orig_preset")
    exec(compile(pre_src, "orig_preset", "exec"), pre.__dict__)
    src = (ROOT / "original" / "terramechanics_rover_pinn_ORIGINAL.py").read_text()
    src = src.replace("from pinneapple_physics.pde_environment.presets.terramechanics import TerramechanicsResiduals",
                      "TerramechanicsResiduals = __orig_residuals__")
    src = src.replace('OUT_DIR = Path(__file__).parent / "outputs"', f'OUT_DIR = Path(r"{RES / "original_outputs"}")')
    src = src.replace("from pinneapple_train import best_device, maybe_compile", "raise ImportError")
    mod = types.ModuleType("orig_example")
    mod.__dict__["__orig_residuals__"] = pre.TerramechanicsResiduals
    mod.__dict__["__file__"] = str(ROOT / "original" / "terramechanics_rover_pinn_ORIGINAL.py")
    exec(compile(src, "orig_example", "exec"), mod.__dict__)
    return mod


def stage_original(test_P, test_Y):
    mod = load_original_module()
    torch.manual_seed(42)
    t0 = time.time()
    model, nx, ny, hist, raw = mod.train(epochs=200 if SMOKE else 4000)
    tt = time.time() - t0
    model = model.cpu().eval()

    def f(P):
        with torch.no_grad():
            return ny.inverse(model(torch.tensor(nx.transform(P.astype(np.float32)))).numpy())

    global ORIG_TORCH

    class OrigTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, x):
            return ny.inverse_torch(self.m(nx.transform_torch(x)))

    ORIG_TORCH = OrigTorch()
    Yp = f(test_P)
    s0 = np.c_[np.zeros(50), np.linspace(*Z_RANGE, 50)]
    OUT["original"] = {"train_seconds": tt, "n_train_points": int(len(raw["slip"])),
                       "best_val_mse_normalised": float(min(hist["val"])),
                       "test": metrics(Yp, test_Y),
                       "fx_at_zero_slip_pred_vs_solver": {"pred_mean_abs": float(np.abs(f(s0)[:, 0]).mean()),
                                                          "solver_mean_abs": float(np.abs(solve2d(s0)[:, 0]).mean()),
                                                          "rmse": float(np.sqrt(((f(s0)[:, 0] - solve2d(s0)[:, 0]) ** 2).mean()))},
                       "final_physics_losses": {k: hist[k][-1] for k in ("r1", "r2", "r3", "r4")}}
    OUT["original"]["constraint_violations"] = constraint_check(f)
    log("original:", json.dumps(OUT["original"], indent=1))
    np.save(RES / "original_test_pred.npy", Yp)
    save()
    return f


def constraint_check(f, n=20000, seed=5):
    """Violation rates of the VERIFIED constraints on fresh points (finite differences)."""
    P = lhs(n, [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]], seed)
    Y = f(P)
    th1 = np.arccos(1 - P[:, 1] / R_)
    r2 = Y[:, 0] > SOIL.c * B_ * R_ * th1 + Y[:, 1] * SOIL.tan_phi
    r4 = Y[:, 2] < R_ * Y[:, 0]
    m = P[:, 0] <= 0.39
    d = 1e-3
    r3 = (f(P[m] + [d, 0])[:, 0] - Y[m, 0]) < 0
    r5 = (f(P + [0, 1e-4])[:, 1] - Y[:, 1]) < 0
    return {"R2_mohr_coulomb": float(r2.mean()), "R3_monotone_slip": float(r3.mean()),
            "R4_torque": float(r4.mean()), "R5_monotone_sinkage": float(r5.mean())}


# ---------------------------------------------------------------------------
# 3. robust surrogate (2-D)
# ---------------------------------------------------------------------------

class RobustTerraNet(nn.Module):
    """(inputs) -> (Fx, Fz, My) with R2 and R4 satisfied by construction.

        Fz = exp(g1) * F_ref                              (positive)
        Fx = c*b*R*theta_1 + Fz*tan(phi) - softplus(g2)*F_ref   (<= Mohr-Coulomb limit, R2)
        My = R*Fx + softplus(g3)*R*F_ref                  (>= R*Fx, R4)

    Soil parameters (c, tan phi) are inputs in the parametric variant, constants in 2-D.
    """

    def __init__(self, in_dim, hard=True, hidden=128, layers=5, F_ref=50.0):
        super().__init__()
        # Random-Fourier-feature bandwidth: sigma=1.0 was tuned for the 2-D case (slip, sinkage). The projection
        # B @ x has variance growing with in_dim, so the same sigma makes the feature map far too high-frequency
        # once the 6 soil parameters are added (in_dim=8): confirmed by a diagnostic sweep, val rel-L2 stuck at
        # ~0.9 (worse than a plain MLP without RFF, or a gradient-boosted baseline, both ~0.02-0.07) at sigma=1.0,
        # dropping to ~0.02-0.07 by sigma=0.1. sigma=0.1 for in_dim>2, unchanged for the 2-D case.
        sigma = 1.0 if in_dim <= 2 else 0.1
        self.net = ModifiedMLP(in_dim=in_dim, out_dim=3, hidden_dim=hidden, n_layers=layers, n_fourier=32, sigma=sigma)
        self.hard, self.F_ref = hard, F_ref

    def forward(self, xn, z, c, tan_phi):
        g = self.net(xn).y
        if self.hard == "scaled":   # same output scaling as the robust model, but no constraint imposed
            Fz = torch.exp(g[:, 0].clamp(max=8)) * self.F_ref
            return torch.stack([Fz * g[:, 1], Fz, R_ * Fz * g[:, 2]], 1)
        if not self.hard:
            return g * self.F_ref
        Fz = torch.exp(g[:, 0].clamp(max=8)) * self.F_ref
        th1 = torch.acos(torch.clamp(1 - z / R_, -1 + 1e-7, 1 - 1e-7))
        Fx = c * B_ * R_ * th1 + Fz * tan_phi - F.softplus(g[:, 1]) * self.F_ref
        My = R_ * Fx + F.softplus(g[:, 2]) * R_ * self.F_ref
        return torch.stack([Fx, Fz, My], 1)


class Surrogate:
    """Normalisation + ensemble wrapper. Inputs P: (N, d) physical; columns 0,1 = slip, sinkage;
    parametric columns 2.. = c, phi_deg, K, k_c, k_phi, n."""

    def __init__(self, lo, hi, hard, parametric=False):
        self.lo, self.hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
        self.hard, self.parametric, self.members = hard, parametric, []
        self.sigma_scale = 1.0

    def xn(self, P):
        return torch.tensor(2 * (P - self.lo) / (self.hi - self.lo) - 1, dtype=torch.float32)

    def soil_cols(self, P):
        P = torch.as_tensor(P, dtype=torch.float32)
        if self.parametric:
            return P[:, 1], P[:, 2], torch.tan(torch.deg2rad(P[:, 3]))
        return P[:, 1], torch.tensor(SOIL.c), torch.tensor(SOIL.tan_phi)

    def forward(self, m, P, xn=None):
        z, c, tp = self.soil_cols(P)
        return m(self.xn(P) if xn is None else xn, z, c, tp)

    def predict(self, P, return_std=False):
        with torch.no_grad():
            Ys = np.stack([np.concatenate([self.forward(m, P[i:i + 8192]).numpy() for i in range(0, len(P), 8192)])
                           for m in self.members])
        return (Ys.mean(0), Ys.std(0) * self.sigma_scale) if return_std else Ys.mean(0)


def rel_loss(Yp, Yt):
    """Load-relative error: forces scaled by the true normal load (+1 N), torque by R*(Fz+1)."""
    sc = (Yt[:, 1:2].abs() + 1.0)
    e = (Yp - Yt) / torch.cat([sc, sc, R_ * sc], 1)
    return (e ** 2).mean()


def train_member(sur: Surrogate, Ptr, Ytr, Pva, Yva, *, physics, seed, epochs, lr=2e-3, bs=256, tag=""):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = RobustTerraNet(Ptr.shape[1], hard=sur.hard)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    steps = epochs * math.ceil(len(Ptr) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    Yt_tr, Yt_va = torch.tensor(Ytr, dtype=torch.float32), torch.tensor(Yva, dtype=torch.float32)
    best, state, t0 = 1e9, None, time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(Ptr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = rel_loss(sur.forward(m, Ptr[b]), Yt_tr[b])
            if physics:
                # verified soft constraints on fresh collocation points (no labels)
                Pc = Ptr[rng.integers(0, len(Ptr), bs)].copy()
                Pc[:, 0] = rng.uniform(0, 0.4, bs)
                xn = sur.xn(Pc).requires_grad_(True)
                Yc = sur.forward(m, Pc, xn)
                g_fx = torch.autograd.grad(Yc[:, 0].sum(), xn, create_graph=True)[0][:, 0]
                g_fz = torch.autograd.grad(Yc[:, 1].sum(), xn, create_graph=True)[0][:, 1]
                sc = Yc[:, 1].detach().abs() + 1
                loss = loss + (F.relu(-g_fx / sc) ** 2).mean() + (F.relu(-g_fz / sc) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
        with torch.no_grad():
            v = float(rel_loss(sur.forward(m, Pva), Yt_va))
        if v < best:
            best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
        if (ep + 1) % max(1, epochs // 5) == 0:
            log(f"  {tag} seed{seed} ep{ep + 1}: val rel-MSE {v:.3e} ({time.time() - t0:.0f}s)")
    m.load_state_dict(state)
    m.eval()
    return m, best, time.time() - t0


def fit_surrogate(Ptr, Ytr, Pva, Yva, lo, hi, *, hard, physics, n_members=5, epochs=300, parametric=False, tag="",
                  min_steps=0, lr=2e-3):
    sur = Surrogate(lo, hi, hard, parametric)
    if min_steps:
        epochs = max(epochs, math.ceil(min_steps / math.ceil(len(Ptr) / 256)))
    info = []
    for s in range(n_members):
        m, v, t = train_member(sur, Ptr, Ytr, Pva, Yva, physics=physics, seed=s, epochs=epochs, tag=tag, lr=lr)
        sur.members.append(m)
        info.append({"val_rel_mse": v, "train_s": t})
    # variance recalibration on validation (one scalar, never on test)
    mu, sd = sur.predict(Pva, return_std=True)
    z = (mu - Yva) / np.maximum(sd, 1e-9)
    sur.sigma_scale = float(np.sqrt(np.mean(z ** 2)))
    return sur, info


def stage_robust2d(test_P, test_Y, f_orig):
    n_tr, n_va = (300, 100) if SMOKE else (2000, 300)
    ep = 20 if SMOKE else 300
    Ptr = lhs(n_tr, [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]], 1).astype(np.float32)
    Pva = lhs(n_va, [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]], 2).astype(np.float32)
    Ytr, Yva = solve2d(Ptr), solve2d(Pva)
    lo, hi = [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]]
    res = {"n_train": n_tr, "n_val": n_va, "n_test": len(test_P)}
    models = {}
    for name, hard, phys in (("Data-only MLP (ModifiedMLP)", False, False),
                             ("Robust PINN (hard R2,R4 + soft R3,R5)", True, True)):
        sur, info = fit_surrogate(Ptr, Ytr, Pva, Yva, lo, hi, hard=hard, physics=phys, epochs=ep, tag=name[:12])
        mu, sd = sur.predict(test_P, return_std=True)
        res[name] = {"test": metrics(mu, test_Y), "members": info, "sigma_scale": sur.sigma_scale,
                     "constraint_violations": constraint_check(sur.predict),
                     "calibration": calibration(mu, sd, test_Y)}
        models[name] = sur
        log(name, json.dumps(res[name]["test"]), res[name]["constraint_violations"])
    OUT["robust2d"] = res
    save()
    torch.save({"lo": lo, "hi": hi, "states": [m.state_dict() for m in models["Robust PINN (hard R2,R4 + soft R3,R5)"].members],
                "sigma_scale": models["Robust PINN (hard R2,R4 + soft R3,R5)"].sigma_scale}, RES / "robust2d_ensemble.pt")
    return models, (Ptr, Ytr, Pva, Yva)


def calibration(mu, sd, Y):
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
    out = {}
    for j, n in enumerate(("Fx", "Fz", "My")):
        a, b, s = (torch.tensor(v[:, j]) for v in (mu, Y, sd))
        out[n] = {"ece": CalibrationMetrics.expected_calibration_error(a, b, s),
                  "coverage90": CalibrationMetrics.coverage_at_level(a, b, s, alpha=0.1),
                  "mean_sigma": float(sd[:, j].mean())}
    return out


# ---------------------------------------------------------------------------
# 4. stress tests
# ---------------------------------------------------------------------------

def stage_stress(test_P, test_Y, data):
    Ptr, Ytr, Pva, Yva = data
    lo, hi = [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]]
    ep = 15 if SMOKE else 300
    rng = np.random.default_rng(3)
    out = {}
    cases = {
        "label_noise_3pct": (Ptr, Ytr * (1 + 0.03 * rng.standard_normal(Ytr.shape)).astype(np.float32), Pva, Yva, test_P, test_Y),
        "small_data_150": (Ptr[:150], Ytr[:150], Pva, Yva, test_P, test_Y),
    }
    mz_tr, mz_va, mz_te = Ptr[:, 1] <= 0.04, Pva[:, 1] <= 0.04, test_P[:, 1] > 0.045
    cases["extrapolate_sinkage_train_le_0.04_test_gt_0.045"] = (Ptr[mz_tr], Ytr[mz_tr], Pva[mz_va], Yva[mz_va],
                                                               test_P[mz_te], test_Y[mz_te])
    for cname, (a, b, c, d, e, f) in cases.items():
        out[cname] = {}
        for name, hard, phys in (("Data-only MLP", False, False), ("Robust PINN", True, True)):
            sur, _ = fit_surrogate(a, b, c, d, lo, hi, hard=hard, physics=phys, n_members=3, epochs=ep, tag=cname[:10],
                                   min_steps=200 if SMOKE else 2400)
            mu, sd = sur.predict(e, return_std=True)
            out[cname][name] = {"test": metrics(mu, f), "constraint_violations": constraint_check(sur.predict, n=5000),
                                "calibration": calibration(mu, sd, f)}
            log(cname, name, json.dumps(out[cname][name]["test"]))
    OUT["stress"] = out
    save()


# ---------------------------------------------------------------------------
# 5. parametric surrogate over soils
# ---------------------------------------------------------------------------

PAR_NAMES = ("slip", "z", "c", "phi_deg", "K", "k_c", "k_phi", "n")
PAR_LO = [0.0, 0.002, 200.0, 25.0, 0.008, 0.0, 4.0e5, 0.8]
PAR_HI = [0.75, 0.05, 3000.0, 42.0, 0.03, 3000.0, 1.6e6, 1.2]
OOD_LO = [0.0, 0.002, 3000.0, 42.0, 0.03, 3000.0, 1.6e6, 1.2]
OOD_HI = [0.75, 0.05, 4500.0, 48.0, 0.045, 4500.0, 2.4e6, 1.35]


def solve_param(P, n_gl=96):
    P = np.asarray(P, np.float64)
    out = []
    for i in range(0, len(P), 4096):
        Q = torch.tensor(P[i:i + 4096])
        out.append(bekker_wong_forces_torch(Q[:, 0], Q[:, 1], R=R_, b=B_, c=Q[:, 2], tan_phi=torch.tan(torch.deg2rad(Q[:, 3])),
                                            K=Q[:, 4], k_c=Q[:, 5], k_phi=Q[:, 6], n=Q[:, 7], n_gl=n_gl).numpy())
    return np.concatenate(out).astype(np.float32)


def stage_parametric():
    n_tr, n_va, n_te, n_ood = (1000, 300, 1000, 500) if SMOKE else (30000, 3000, 10000, 4000)
    ep = 10 if SMOKE else 120
    Ptr = lhs(n_tr, PAR_LO, PAR_HI, 11).astype(np.float32)
    Pva = lhs(n_va, PAR_LO, PAR_HI, 12).astype(np.float32)
    Pte = lhs(n_te, PAR_LO, PAR_HI, 13).astype(np.float32)
    # OOD: every soil parameter outside the training box simultaneously
    Pood = lhs(n_ood, OOD_LO, OOD_HI, 14).astype(np.float32)
    t0 = time.time()
    Ytr, Yva, Yte, Yood = (solve_param(P) for P in (Ptr, Pva, Pte, Pood))
    gen_s = time.time() - t0
    # verify the batched solver against adaptive quad on random parametric points
    chk = []
    for p in Pte[:60]:
        sol = BekkerWongSolver(SoilParams(c=float(p[2]), phi_deg=float(p[3]), K=float(p[4]), k_c=float(p[5]),
                                          k_phi=float(p[6]), n=float(p[7])), WHEEL)
        chk.append(np.array(sol.forces(float(p[0]), float(p[1]))))
    chk = np.array(chk)
    ver = np.abs(solve_param(Pte[:60]) - chk).max(0) / np.abs(chk).max(0)
    res = {"ranges": {"in": dict(zip(PAR_NAMES, zip(PAR_LO, PAR_HI))), "ood": dict(zip(PAR_NAMES, zip(OOD_LO, OOD_HI)))},
           "n": {"train": n_tr, "val": n_va, "test": n_te, "ood": n_ood}, "data_gen_seconds": gen_s,
           "gl96_vs_quad_max_rel_err": ver.tolist()}
    for name, hard, phys in (("Data-only MLP (linear outputs)", False, False),
                             ("Data-only MLP (physically scaled outputs)", "scaled", False),
                             ("Robust PINN", True, True)):
        sur, info = fit_surrogate(Ptr, Ytr, Pva, Yva, PAR_LO, PAR_HI, hard=hard, physics=phys, n_members=5,
                                  epochs=ep, parametric=True, tag="param-" + name[:10], lr=1e-3)
        r = {"members": info}
        for split, P, Y in (("in_distribution", Pte, Yte), ("out_of_distribution", Pood, Yood)):
            mu, sd = sur.predict(P, return_std=True)
            r[split] = {"test": metrics(mu, Y), "calibration": calibration(mu, sd, Y),
                        "mean_rel_sigma": float((sd[:, 1] / (np.abs(mu[:, 1]) + 1)).mean())}
        # OOD detection from ensemble spread (AUROC of relative std on Fz)
        s_in = (sur.predict(Pte, True)[1][:, 1] / (np.abs(Yte[:, 1]) + 1))
        s_ood = (sur.predict(Pood, True)[1][:, 1] / (np.abs(Yood[:, 1]) + 1))
        r["ood_auroc_from_spread"] = auroc(s_in, s_ood)
        res[name] = r
        log("parametric", name, json.dumps({k: r[k]["test"] for k in ("in_distribution", "out_of_distribution")}))
    OUT["parametric"] = res
    save()


def auroc(neg, pos):
    x = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    order = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# 6-8. rover, speed, veriphysics
# ---------------------------------------------------------------------------

def stage_rover(models, f_orig):
    slips = np.linspace(0.02, 0.7, 18)
    z_true = np.array([SOLVER.sinkage_from_load(W_WHEEL, s) for s in slips])
    fx_true = np.array([SOLVER.forces(s, z)[0] for s, z in zip(slips, z_true)])
    out = {"W_N": W_WHEEL}

    def solve_z(f, s):
        lo, hi = Z_RANGE[0], Z_RANGE[1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(np.array([[s, mid]], np.float32))[0, 1] > W_WHEEL:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    fs = {"Original PINN": f_orig, **{k: v.predict for k, v in models.items()}}
    for name, f in fs.items():
        zp = np.array([solve_z(f, s) for s in slips])
        fxp = np.array([f(np.array([[s, z]], np.float32))[0, 0] for s, z in zip(slips, zp)])
        out[name] = {"sinkage_rmse_mm": float(np.sqrt(((zp - z_true) ** 2).mean()) * 1e3),
                     "drawbar_pull_rmse_N": float(np.sqrt(((fxp - fx_true) ** 2).mean())),
                     "curve": {"slip": slips.tolist(), "z_pred": zp.tolist(), "fx_pred": fxp.tolist()}}
    out["solver"] = {"slip": slips.tolist(), "z": z_true.tolist(), "fx": fx_true.tolist()}
    OUT["rover"] = out
    log("rover:", {k: (v["sinkage_rmse_mm"], v["drawbar_pull_rmse_N"]) for k, v in out.items() if isinstance(v, dict) and "sinkage_rmse_mm" in v})
    save()


def stage_speed(models, f_orig):
    P = lhs(20000, [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]], 21).astype(np.float32)
    t = time.time(); [SOLVER.forces(float(p[0]), float(p[1])) for p in P[:200]]; quad = (time.time() - t) / 200
    t = time.time(); SOLVER.forces_batch(P[:, 0], P[:, 1], n_gl=48); gl = (time.time() - t) / len(P)
    t = time.time(); SOLVER.forces_batch(P[:, 0], P[:, 1], n_gl=16); gl16 = (time.time() - t) / len(P)
    sur = models["Robust PINN (hard R2,R4 + soft R3,R5)"]
    t = time.time(); sur.predict(P); ens = (time.time() - t) / len(P)
    one = Surrogate(sur.lo, sur.hi, True); one.members = sur.members[:1]
    t = time.time()
    for p in P[:500]:
        one.predict(p[None])
    single = (time.time() - t) / 500
    t = time.time(); f_orig(P); orig = (time.time() - t) / len(P)
    OUT["speed_us_per_eval"] = {"quad_adaptive": quad * 1e6, "gauss_legendre_48_batched": gl * 1e6,
                                "gauss_legendre_16_batched": gl16 * 1e6, "robust_ensemble5_batched": ens * 1e6,
                                "robust_single_member_one_query": single * 1e6, "original_pinn_batched": orig * 1e6}
    log("speed:", OUT["speed_us_per_eval"])
    save()


def stage_veriphysics(models, f_orig, test_P, test_Y):
    from pinneapple_analysis.verification.convergence import richardson_extrapolate
    from pinneapple_analysis.verification.physics_confidence_score import CalibrationSummary, compute_physics_confidence
    from pinneapple_data.physics_case import PhysicsCase
    from pinneapple_llm.guardrail import PhysicsGuardrail
    from pinneapple_pdb.benchmarks import BenchmarkEntry, register_benchmark
    from pinneapple_physics.pde_environment.presets.registry import get_preset

    @register_benchmark("bekker_wong_grc1_heldout_lhs")
    def _b():
        return BenchmarkEntry(name="bekker_wong_grc1_heldout_lhs",
                              description="Bekker-Wong (GRC-1, R=0.125 m, b=0.06 m) on an independent LHS test set",
                              reference_source="pinneapple BekkerWongSolver (adaptive quad == GL48 to 1e-13)",
                              x_vars=("slip", "sinkage"), y_vars=("Fx", "Fz", "My"),
                              reference_x=test_P.astype(np.float32), reference_y=test_Y.astype(np.float32))

    # numerical convergence of the reference quadrature for a non-integer sinkage exponent
    sol = BekkerWongSolver(SoilParams(n=0.8), WHEEL)
    Pc = lhs(200, [0.0, 0.004], [0.75, 0.058], 31)
    fvals = [sol.forces_batch(Pc[:, 0], Pc[:, 1], n_gl=k)[:, 0] for k in (8, 16, 32)]
    conv = richardson_extrapolate(*fvals, r=2.0)
    spec = get_preset("bekker_wong_surrogate_2d")
    out = {"convergence_gl_n0.8": {"observed_order": conv.observed_order, "gci_fine": conv.gci_fine,
                                    "asymptotic_ratio": conv.asymptotic_ratio}}

    class Wrap(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return torch.as_tensor(self.f(x.detach().numpy()), dtype=torch.float32)

    class TorchWrap(nn.Module):  # differentiable wrapper so the guardrail's autograd residual works
        def __init__(self, sur):
            super().__init__()
            self.sur = sur
            self.members = nn.ModuleList(sur.members)  # registered so the guardrail sees parameters

        def forward(self, x):
            xn = 2 * (x - torch.tensor(self.sur.lo)) / torch.tensor(self.sur.hi - self.sur.lo) - 1
            ys = [m(xn, x[:, 1], torch.tensor(SOIL.c), torch.tensor(SOIL.tan_phi)) for m in self.members]
            return torch.stack(ys).mean(0)

    orig_mod = ORIG_TORCH
    cands = {"Robust PINN (hard R2,R4 + soft R3,R5)": TorchWrap(models["Robust PINN (hard R2,R4 + soft R3,R5)"]),
             "Data-only MLP (ModifiedMLP)": TorchWrap(models["Data-only MLP (ModifiedMLP)"])}
    if orig_mod is not None:
        cands["Original PINN"] = orig_mod
    for name, tm in cands.items():
        torch.manual_seed(0)
        np.random.seed(0)
        g = PhysicsGuardrail(spec, residual_threshold=1e-2, n_check_points=4096).check(
            tm, reference_x=test_P, reference_y=test_Y, reference_rmse_threshold=1.0)
        with torch.no_grad():
            Yp = tm(torch.tensor(test_P, dtype=torch.float32)).numpy()
        case = PhysicsCase(name=name, results={"slip": test_P[:, 0], "sinkage": test_P[:, 1], "Fx": Yp[:, 0],
                                               "Fz": Yp[:, 1], "My": Yp[:, 2]}, reference_benchmark="bekker_wong_grc1_heldout_lhs")
        bench = case.validate_against_benchmark()
        cal = None
        if name in OUT.get("robust2d", {}):
            c = OUT["robust2d"][name]["calibration"]["Fx"]
            cal = CalibrationSummary(ece=c["ece"], coverage=c["coverage90"], target_coverage=0.9, sharpness=c["mean_sigma"])
        sc = compute_physics_confidence(guardrail_report=g, convergence_result=conv, calibration_metrics=cal,
                                        benchmark_comparison=bench)
        out[name] = {"guardrail": [c.__dict__ for c in g.checks], "trustworthy": g.trustworthy,
                     "overall_score": sc.overall_score, "coverage": sc.coverage,
                     "components": [c.__dict__ for c in sc.components], "rel_l2": bench.relative_l2_error}
        log(sc.summary())
    out["not_run"] = {"geometry_ood": "no geometry input; the parametric OOD test (stage 5) covers input-space shift"}
    OUT["veriphysics"] = out
    save()


def main():
    t0 = time.time()
    stage_audit()
    test_P = lhs(400 if SMOKE else 5000, [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]], 99).astype(np.float32)
    test_Y = solve2d(test_P)
    np.savez_compressed(RES / "test_set.npz", P=test_P, Y=test_Y)
    f_orig = stage_original(test_P, test_Y)
    models, data = stage_robust2d(test_P, test_Y, f_orig)
    stage_rover(models, f_orig)
    stage_speed(models, f_orig)
    stage_veriphysics(models, f_orig, test_P, test_Y)
    stage_stress(test_P, test_Y, data)
    stage_parametric()
    OUT["runtime_s"] = time.time() - t0
    save()
    log("done", OUT["runtime_s"])


def rerun(stages):
    """Re-run selected stages on top of the saved results (same seeds, same data generation)."""
    OUT.update(json.loads((RES / "results.json").read_text()))
    z = np.load(RES / "test_set.npz")
    test_P, test_Y = z["P"], z["Y"]
    n_tr, n_va = (300, 100) if SMOKE else (2000, 300)
    Ptr = lhs(n_tr, [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]], 1).astype(np.float32)
    Pva = lhs(n_va, [S_RANGE[0], Z_RANGE[0]], [S_RANGE[1], Z_RANGE[1]], 2).astype(np.float32)
    data = (Ptr, solve2d(Ptr), Pva, solve2d(Pva))
    if "stress" in stages:
        OUT["stress_first_attempt_epoch_matched"] = OUT.get("stress")
        stage_stress(test_P, test_Y, data)
    if "parametric" in stages:
        OUT["parametric_first_attempt"] = OUT.get("parametric")
        stage_parametric()
    save()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rerun(sys.argv[1:])
    else:
        main()
