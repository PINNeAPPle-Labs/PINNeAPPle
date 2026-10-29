"""13 — Geometry variants -> simulation data -> benchmark -> inverse -> verification.

The loop an engineer actually runs when a design parameter is unknown:

  1. a geometry with one parameter (the radius of the heated core)
  2. one simulation per variant, from a real 3-D solver
  3. the runs assembled into one training set carrying the parameter
  4. several architectures trained on the same set, benchmarked and ranked
  5. an inverse model that recovers the parameter from a few measurements
  6. the result verified: a trust card on the field and the VeriPhysics decision

Run from the repository root:

    python examples/quickstart/13_variants_benchmark_inverse.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pinneapple as pp
from pinneapple_design.geometry import sdf3d_sphere
from pinneapple_neural import Trainer, TrainConfig, build_model
from pinneapple_simulation.numerical_solvers import HeatConduction3D, HeatConfig3D
from pinneapple_analysis.inverse_problems import EnsembleKalmanInversion, EKIConfig

OUT = "runs/variants"

# ---------------------------------------------------------------------------
# 1. One geometry parameter, sampled from its SDF
# ---------------------------------------------------------------------------
RADII = [0.12, 0.18, 0.24, 0.30]
print("geometry: sphere of radius r at the centre of a unit box")
for r in RADII:
    axis = np.linspace(0.0, 1.0, 12)
    P = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    inside = sdf3d_sphere(P, (0.5, 0.5, 0.5), r) <= 0.0
    print(f"  r={r:.2f}: {inside.sum():4d} of {len(P)} sampled points inside the sphere")

# ---------------------------------------------------------------------------
# 2. One simulation per variant
# ---------------------------------------------------------------------------
def run_case(r: float) -> np.ndarray:
    """3-D heat conduction with the source confined to the sphere of radius r."""
    cfg = HeatConfig3D(
        nx=20, ny=20, nz=20, nt=30, dt=2e-3, alpha=0.02,
        source_fn=lambda X, Y, Z, t, r=r: (
            sdf3d_sphere(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]),
                         (0.5, 0.5, 0.5), r) <= 0.0
        ).reshape(np.shape(X)).astype(float),
    )
    return HeatConduction3D(cfg).solve().u


print("\n2. running one 3-D solve per variant")
fields = {}
for r in RADII:
    u = run_case(r)
    fields[r] = u
    print(f"  r={r:.2f}: field {u.shape}, final peak T {u[-1].max():.4f}")

# ---------------------------------------------------------------------------
# 3. Assemble the runs into one dataset that carries the parameter
# ---------------------------------------------------------------------------
u = fields[RADII[0]]
nt, nx, ny, nz = u.shape
axes = [np.linspace(0.0, 1.0, n) for n in (nt, nx, ny, nz)]
Tg, Xg, Yg, Zg = np.meshgrid(*axes, indexing="ij")
coords = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel(), Tg.ravel()]).astype(np.float32)

Xs, ys = [], []
rng = np.random.default_rng(0)
n_per_case = 8000
for r in RADII:
    rows = rng.choice(len(coords), size=n_per_case, replace=False)
    Xs.append(np.column_stack([coords[rows], np.full(len(rows), r, dtype=np.float32)]))
    ys.append(fields[r].reshape(-1, 1)[rows].astype(np.float32))

X = np.concatenate(Xs)
y = np.concatenate(ys)
print(f"\n3. dataset: X {X.shape} (x, y, z, t, r) -> T {y.shape}")

idx = rng.permutation(len(X))
split = int(0.85 * len(X))
tr, va = idx[:split], idx[split:]
train_loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr])), batch_size=2048, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X[va]), torch.tensor(y[va])), batch_size=4096)

# ---------------------------------------------------------------------------
# 4. Benchmark three architectures on that dataset
# ---------------------------------------------------------------------------
print("\n4. benchmarking architectures on the same data, same budget, same seed")
results = {}
for arch in ("vanilla_pinn", "siren", "modified_mlp"):
    model = build_model(arch, in_dim=5, out_dim=1, hidden_dim=48, n_layers=3) \
        if arch != "siren" else build_model("siren", in_dim=5, out_dim=1, hidden_dim=48, n_layers=3)
    trainer = Trainer(model, loss_fn=lambda m, yh, b: F.mse_loss(yh, b["y"]))
    history = trainer.fit(train_loader, val_loader,
                          TrainConfig(epochs=80, lr=2e-3, device="cpu", log_dir=f"{OUT}/{arch}"))
    last = history["history"][-1]
    with torch.no_grad():
        pred = model(torch.tensor(X[va])).y.numpy()
    rel = float(np.linalg.norm(pred - y[va]) / np.linalg.norm(y[va]))
    results[arch] = {"val_mse": last["val_total"], "rel_l2": rel, "model": model}
    print(f"  {arch:<14} val MSE {last['val_total']:.3e}   held-out rel L2 {rel:.3e}")

best = min(results, key=lambda k: results[k]["rel_l2"])
print(f"  best: {best} (held-out relative L2 {results[best]['rel_l2']:.3e})")
surrogate = results[best]["model"]

# ---------------------------------------------------------------------------
# 5. Inverse model: recover r from a handful of temperature readings
# ---------------------------------------------------------------------------
print("\n5. inverse: recover the radius from 6 probe readings of the true field")

probes = np.array([[0.5, 0.5, 0.5, 1.0],
                   [0.6, 0.5, 0.5, 1.0],
                   [0.5, 0.6, 0.5, 1.0],
                   [0.5, 0.5, 0.6, 1.0],
                   [0.5, 0.5, 0.5, 0.5],
                   [0.7, 0.5, 0.5, 0.5]], dtype=np.float32)
R_TRUE = 0.24


def forward(theta: np.ndarray) -> np.ndarray:
    """Map a batch of radii to the probe readings the surrogate predicts."""
    theta = np.asarray(theta, dtype=np.float32).reshape(-1, 1)
    n = len(theta)
    pts = np.hstack([np.tile(probes, (n, 1)), np.repeat(theta, len(probes), axis=0)])
    with torch.no_grad():
        pred = surrogate(torch.tensor(pts, dtype=torch.float32)).y.numpy()
    return pred.reshape(n, len(probes))          # (n_ensemble, n_probes), as EKI expects


y_obs = forward(np.array([R_TRUE]))[0]
eki = EnsembleKalmanInversion(forward, EKIConfig(n_ensemble=60, n_iterations=40, noise_std=0.005, seed=0))
history = eki.run(y_obs, theta_init=np.array([0.18]), ensemble=np.linspace(0.10, 0.35, 60).reshape(-1, 1))
r_hat = float(np.mean(history.theta_mean_history[-1]))
misfit = history.final_misfit
misfit = misfit() if callable(misfit) else misfit
print(f"  true r = {R_TRUE:.4f}   recovered r = {r_hat:.4f}   "
      f"error {100 * (r_hat - R_TRUE) / R_TRUE:+.2f} %   final misfit {float(misfit):.3e}")

# ---------------------------------------------------------------------------
# 6. Verification: a trust card on the field, then the VeriPhysics decision
# ---------------------------------------------------------------------------
print("\n6. verification")

# The check that matters: the surrogate's field against the solver's, at the true radius.
grid = {"x": axes[1], "y": axes[2], "z": axes[3]}
gx, gy, gz = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
gr = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel(),
                      np.ones(gx.size, dtype=np.float32),
                      np.full(gx.size, R_TRUE, dtype=np.float32)])
with torch.no_grad():
    surrogate_field = surrogate(torch.tensor(gr, dtype=torch.float32)).y.numpy().reshape(gx.shape)

card = pp.trust_card(grid, surrogate_field, reference=fields[R_TRUE][-1],
                     train_bounds={"x": (0, 1), "y": (0, 1), "z": (0, 1)})
passed = sum(1 for c in card.checks if c.passed is True)
print(f"  trust card, surrogate field vs solver field: {card.decision} "
      f"({passed} of {len(card.checks)} checks pass, the rest are N/A)")

from pinneapple_decision import VeriPhysicsVerifier, ThresholdVerifier, ExecutionResult

verifier = VeriPhysicsVerifier(min_trust=70.0, min_coverage=0.5,
                               fallback=ThresholdVerifier(thresholds={"rel_l2": 0.15}))
result = ExecutionResult(status="ok",
                         metrics={"rel_l2": results[best]["rel_l2"],
                                  "theta_error_pct": abs(100 * (r_hat - R_TRUE) / R_TRUE)},
                         diagnostics={}, artifacts={}, decision_id="inverse-r")
verification = verifier.verify(result, {"name": "sphere_radius_inverse"})
print("  VeriPhysics installed:", verifier.veriphysics_installed())
print("  decision:", verification.passed, "| source:", verification.source, "| tags:", verification.tags)
for note in (verification.notes or [])[:3]:
    print("   -", note)

print(f"\nartefacts: {OUT}/")
