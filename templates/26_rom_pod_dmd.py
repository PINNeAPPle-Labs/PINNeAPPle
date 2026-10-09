"""26_rom_pod_dmd.py — Reduced Order Models: POD, DMD, and HAVOK."""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneapple_neural.architectures.rom.dmd import DynamicModeDecomposition
from pinneapple_neural.architectures.rom.havok import HAVOK
from pinneapple_neural.architectures.rom.pod import POD

NX = 100
N_MODES = 4
N_SNAPS = 200
DT_SIM = 0.05
N_PRED = 40


def generate_snapshots(
    nx: int = NX, n_modes: int = N_MODES, n_snaps: int = N_SNAPS
) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial grid and snapshots shaped (nx, n_snaps)."""
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.arange(n_snaps) * DT_SIM
    X = np.zeros((nx, n_snaps), dtype=np.float32)

    for n in range(1, n_modes + 1):
        omega = 0.5 * n
        spatial_mode = np.sin(n * x / 2)
        amplitude = 1.0 / n
        phase = n * 0.3
        X += amplitude * np.outer(
            spatial_mode, np.cos(omega * t + phase)
        )

    return x, X


def relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute relative L2 reconstruction error."""
    denominator = torch.linalg.vector_norm(target).clamp_min(1e-12)
    return (torch.linalg.vector_norm(prediction - target) / denominator).item()


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Generating snapshot matrix ...")
    x_grid, X = generate_snapshots()
    X_noisy = X + np.random.normal(0, 0.02, X.shape).astype(np.float32)

    n_train = N_SNAPS - N_PRED

    # Models expect snapshots in (time, features) order.
    X_train = torch.from_numpy(X_noisy[:, :n_train].T).to(device)
    X_test = torch.from_numpy(X[:, n_train:].T).to(device)

    # 1. POD reconstruction
    print("\n[1] POD reconstruction ...")
    for n_comp in (2, 4, 8):
        pod = POD(r=n_comp, center=True).to(device)
        pod.fit(X_train)
        X_recon = pod.decode(pod.encode(X_train))
        error = relative_error(X_recon, X_train)
        variance = pod.explained_variance_ratio_.sum().item()
        print(
            f"  r={n_comp:2d}  reconstruction error={error:.4e}  "
            f"explained variance={variance:.4f}"
        )

    pod4 = POD(r=4, center=True).to(device)
    pod4.fit(X_train)
    X_pod_recon = pod4.decode(pod4.encode(X_train))

    # 2. DMD forecast
    print("\n[2] DMD forecasting ...")
    dmd = DynamicModeDecomposition(r=10, center=True).to(device)
    dmd.fit(X_train)
    forecast = dmd.rollout(X_train[-1:, :], steps=N_PRED)
    X_dmd_pred = forecast[0, 1:, :]
    dmd_rmse = torch.sqrt(torch.mean((X_dmd_pred - X_test) ** 2)).item()
    print(f"  DMD forecast RMSE = {dmd_rmse:.4e}")

    # 3. HAVOK reconstruction of the training sensor signal
    print("\n[3] HAVOK reconstruction ...")
    sensor_train = X_train[:, NX // 2 : NX // 2 + 1]
    havok = HAVOK(delays=20, r=5, decode_mode="last").to(device)
    havok.fit(sensor_train)
    havok_output = havok(sensor_train)
    havok_recon = havok_output.extras["xhat"]
    havok_rmse = torch.sqrt(
        torch.mean((havok_recon - sensor_train[19:]) ** 2)
    ).item()
    print(f"  HAVOK embedded reconstruction RMSE = {havok_rmse:.4e}")

    # Convert tensors for plotting.
    X_np = X
    X_train_np = X_train.detach().cpu().numpy().T
    X_test_np = X_test.detach().cpu().numpy().T
    X_pod_np = X_pod_recon.detach().cpu().numpy().T
    X_dmd_np = X_dmd_pred.detach().cpu().numpy().T
    sv = pod4.sv_.detach().cpu().numpy()
    havok_np = havok_recon.detach().cpu().numpy().squeeze()
    havok_time = np.arange(19, n_train) * DT_SIM

    t_full = np.arange(N_SNAPS) * DT_SIM
    t_train = t_full[:n_train]
    t_test = t_full[n_train:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    image = axes[0, 0].imshow(
        X_np,
        aspect="auto",
        extent=[0, t_full[-1], 0, 2 * np.pi],
        origin="lower",
        cmap="RdBu_r",
        vmin=-1.5,
        vmax=1.5,
    )
    fig.colorbar(image, ax=axes[0, 0])
    axes[0, 0].set_title("Ground truth snapshot matrix u(x,t)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x")

    axes[0, 1].plot(x_grid, X_train_np[:, 80], "k-", label="Noisy input")
    axes[0, 1].plot(x_grid, X_pod_np[:, 80], "r--", label="POD (r=4)")
    axes[0, 1].set_title("POD snapshot reconstruction")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    ix = NX // 3
    axes[1, 0].plot(t_train, X_train_np[ix], "b-", label="Train")
    axes[1, 0].plot(t_test, X_test_np[ix], "k--", label="Ground truth")
    axes[1, 0].plot(t_test, X_dmd_np[ix], "r-", label="DMD forecast")
    axes[1, 0].set_title(f"DMD forecast at x={x_grid[ix]:.2f}")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(range(1, len(sv) + 1), sv)
    axes[1, 1].set_title("POD singular values")
    axes[1, 1].set_xlabel("Mode index")
    axes[1, 1].set_ylabel("Singular value")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("26_rom_pod_dmd_result.png", dpi=120)
    plt.close(fig)
    print("\nSaved 26_rom_pod_dmd_result.png")
    print(
        f"HAVOK reconstructed {len(havok_np)} embedded observations "
        f"over {len(havok_time)} time points."
    )


if __name__ == "__main__":
    main()
