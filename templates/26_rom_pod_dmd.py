"""26_rom_pod_dmd.py — Reduced Order Models: POD and DMD.

Demonstrates:
- POD: Proper Orthogonal Decomposition via SVD (reconstruction error vs. rank)
- DynamicModeDecomposition: linear system identification and forecasting (rollout)
- HAVOK: Hankel Alternative View of Koopman (delay embedding). The current class fits the linear model on the
  delay coordinates and replays the training window from its first state; it has no forecast past the data, so
  the template reports that replay error instead of a forecast. Without HAVOK's forcing term the linear core
  drifts over the 140-step window, so expect a large replay error (about 1.0 here).

All three live in ``pinneapple_neural.architectures.rom``; they take snapshots as rows, (time, space).
- ROM reconstruction error and future-state prediction
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneapple_neural.architectures.rom import POD, DynamicModeDecomposition

try:
    from pinneapple_neural.architectures.rom import HAVOK
    _HAVOK = True
except ImportError:
    _HAVOK = False


# ---------------------------------------------------------------------------
# Synthetic dataset: 2D wave equation snapshots
# u(x,t) = Σₙ aₙ(t) φₙ(x)   where  aₙ(t) = cos(ωₙ t + ψₙ)
# This produces a low-rank spatiotemporal field of travelling waves (ideal for POD/DMD).
# ---------------------------------------------------------------------------

NX       = 100
N_MODES  = 4
N_SNAPS  = 200
DT_SIM   = 0.05
N_PRED   = 40     # future steps to predict with DMD


def generate_snapshots(nx: int = NX, n_modes: int = N_MODES,
                       n_snaps: int = N_SNAPS) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial modes ψ (nx,) and snapshots X (nx, n_snaps)."""
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.arange(n_snaps) * DT_SIM
    X = np.zeros((nx, n_snaps), dtype=np.float32)
    for n in range(1, n_modes + 1):
        omega = 0.5 * n
        amp   = 1.0 / n
        phase = n * 0.3
        # travelling wave sin(n x / 2 - omega t + phase): two POD modes per n, and an exactly linear,
        # oscillating dynamics that DMD can identify (a standing wave with only cos(omega t) cannot be:
        # its snapshots do not contain the sin(omega t) partner a first-order linear model needs)
        X    += amp * np.sin(n * x[:, None] / 2 - omega * t[None, :] + phase)
    return x, X.astype(np.float32)


def main():
    np.random.seed(0)
    print("Generating snapshot matrix ...")
    x_grid, X = generate_snapshots()    # X: (nx, n_snaps)

    # Add small noise
    X_noisy = X + np.random.normal(0, 0.02, X.shape).astype(np.float32)

    # Split: train on first 160, predict next 40
    n_train = N_SNAPS - N_PRED
    X_train = X_noisy[:, :n_train]
    X_test  = X[:, n_train:]             # clean ground truth for evaluation

    # =========================================================================
    # 1) POD reconstruction
    # =========================================================================
    print("\n[1] POD reconstruction ...")
    snaps = torch.tensor(X_train.T)                     # (time, space): one snapshot per row
    for n_comp in [2, 4, 8]:
        pod = POD(r=n_comp, center=True).fit(snaps)
        err = pod.reconstruction_error(snaps)["relative_l2"]
        sv2 = pod.sv_ ** 2
        ev_ratio = float(sv2.sum() / (torch.linalg.svdvals(snaps - snaps.mean(0)) ** 2).sum())
        print(f"  r={n_comp:2d}  recon error={err:.4e}  explained variance={ev_ratio:.4f}")

    # Full POD for visualisation (r=4)
    pod4 = POD(r=4, center=True).fit(snaps)
    X_pod_recon = pod4.decode(pod4.encode(snaps)).numpy().T   # back to (space, time)

    # =========================================================================
    # 2) DMD for future prediction
    # =========================================================================
    print("\n[2] DMD forecasting ...")
    dmd = DynamicModeDecomposition(r=2 * N_MODES, center=False).fit(snaps)

    # Predict future states: roll out N_PRED steps from the last training snapshot
    roll = dmd.rollout(snaps[-1:], N_PRED)              # (1, N_PRED + 1, space), starts with x0
    X_dmd_pred = roll[0, 1:].numpy().T                  # (space, N_PRED)
    dmd_err = np.sqrt(((X_dmd_pred - X_test)**2).mean())
    print(f"  DMD forecast RMSE = {dmd_err:.4e}  (fitted on the noisy snapshots)")
    # Same fit on the clean snapshots: exact up to round-off. Noise biases the DMD eigenvalues (damping),
    # a known weakness of plain DMD; total-least-squares or forward-backward DMD reduce it.
    clean = torch.tensor(X[:, :n_train].T)
    roll_c = DynamicModeDecomposition(r=2 * N_MODES, center=False).fit(clean).rollout(clean[-1:], N_PRED)
    print(f"  DMD forecast RMSE = {np.sqrt(((roll_c[0, 1:].numpy().T - X_test) ** 2).mean()):.4e}  "
          f"(fitted on the clean snapshots)")

    # =========================================================================
    # 3) HAVOK (Hankel-based Koopman, if available)
    # =========================================================================
    if _HAVOK:
        print("\n[3] HAVOK (Hankel-DMD) ...")
        # Use a single sensor time series
        sensor_signal = X_train[NX // 2, :]    # single sensor at x = π
        sig = torch.tensor(sensor_signal[:, None])          # (time, 1)
        havok = HAVOK(delays=20, r=5).fit(sig)
        out = havok(sig)                                    # replay of the training window in delay space
        xhat = out.extras["xhat"].reshape(-1).numpy()       # last delay coordinate = the signal
        truth = sensor_signal[-len(xhat):]
        h_err = np.sqrt(((xhat - truth) ** 2).mean())
        print(f"  HAVOK replay RMSE over the training window = {h_err:.4e}  ({len(xhat)} steps)")

    # =========================================================================
    # Visualisation
    # =========================================================================
    t_full = np.arange(N_SNAPS) * DT_SIM
    t_train = t_full[:n_train]
    t_test  = t_full[n_train:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Snapshot matrix (space-time heatmap)
    im = axes[0, 0].imshow(X, aspect="auto", extent=[0, t_full[-1], 0, 2 * np.pi],
                            origin="lower", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    plt.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title("Ground truth snapshot matrix u(x,t)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x")

    # Panel 2: POD reconstruction at midpoint
    axes[0, 1].plot(x_grid, X_train[:, 80],      "k-",  label="Truth")
    axes[0, 1].plot(x_grid, X_pod_recon[:, 80],  "r--", label="POD (r=4) recon")
    axes[0, 1].set_title("POD snapshot reconstruction (t=4.0 s)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: DMD prediction for a single x
    ix = NX // 3
    axes[1, 0].plot(t_train, X_train[ix, :],  "b-",  label="Train")
    axes[1, 0].plot(t_test,  X_test[ix, :],   "k--", label="Truth (test)")
    axes[1, 0].plot(t_test,  X_dmd_pred[ix, :], "r-", label="DMD forecast")
    axes[1, 0].set_title(f"DMD forecast at x={x_grid[ix]:.2f}")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: POD singular values
    sv = pod4.sv_.numpy()
    axes[1, 1].bar(range(1, len(sv) + 1), sv, color="steelblue")
    axes[1, 1].set_title("POD singular values")
    axes[1, 1].set_xlabel("Mode index")
    axes[1, 1].set_ylabel("Singular value")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("26_rom_pod_dmd_result.png", dpi=120)
    print("\nSaved 26_rom_pod_dmd_result.png")


if __name__ == "__main__":
    main()
