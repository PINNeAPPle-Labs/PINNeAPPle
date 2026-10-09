"""23_model_export.py — ONNX and TorchScript export with pinneapple_tools.model_export.

Demonstrates:
- export_onnx: export a trained PINN to ONNX with a dynamic batch axis
- export_torchscript: trace a model to TorchScript (.pt)
- A post-export check: the exported model must reproduce the PyTorch outputs (max abs difference)
- Latency comparison: Python torch vs. ONNX Runtime vs. TorchScript

ONNX needs ``pip install onnx`` (and ``onnxruntime`` for the check and the latency); without them those steps are
skipped with a message. Files go to a temporary folder.
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np

import tempfile
from pathlib import Path

from pinneapple_tools.model_export import export_onnx, export_torchscript


# ---------------------------------------------------------------------------
# Evaluation model: trained Poisson PINN
# (same architecture as template 01, weights trained here for standalone use)
# ---------------------------------------------------------------------------

class PoissonPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        phi = xy[:, 0:1] * (1 - xy[:, 0:1]) * xy[:, 1:2] * (1 - xy[:, 1:2])
        return phi * self.net(xy)


def quick_train(model: nn.Module, device, n_epochs: int = 2000) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        opt.zero_grad()
        xy = torch.rand(1024, 2, device=device, requires_grad=True)
        u  = model(xy)
        g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        f    = -2 * math.pi**2 * torch.sin(math.pi * xy[:, 0:1]) * \
                torch.sin(math.pi * xy[:, 1:2])
        (u_xx + u_yy - f).pow(2).mean().backward()
        opt.step()


def benchmark_inference(fn, x: torch.Tensor, n: int = 200) -> float:
    """Return mean latency in ms over n runs."""
    for _ in range(10):      # warmup
        fn(x)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t0) / n * 1000


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")     # ONNX Runtime typically on CPU for comparison
    print(f"Device: {device}")

    # --- Train ---------------------------------------------------------------
    print("Training Poisson PINN ...")
    model = PoissonPINN().to(device)
    quick_train(model, device, n_epochs=2000)
    model.eval()
    print("Training complete.")

    x_dummy = torch.rand(64, 2, device=device)

    out_dir = Path(tempfile.mkdtemp(prefix="pinneapple_export_"))
    onnx_path = str(out_dir / "23_poisson.onnx")
    ts_path = str(out_dir / "23_poisson_scripted.pt")
    with torch.no_grad():
        y_ref = model(x_dummy)

    # --- ONNX export ---------------------------------------------------------
    print("\nExporting to ONNX ...")
    try:
        export_onnx(model, onnx_path, x_dummy, input_names=["input"], output_names=["output"], opset_version=17,
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print(f"  Saved: {onnx_path}")
    except Exception as e:                       # onnx / onnxscript not installed
        onnx_path = None
        print(f"  ONNX export skipped ({type(e).__name__}: {e})")

    # --- TorchScript export --------------------------------------------------
    print("Exporting to TorchScript ...")
    export_torchscript(model, ts_path, example_input=x_dummy)       # traced with the example input
    print(f"  Saved: {ts_path}")

    # --- Validation: the exported models reproduce the PyTorch outputs -------
    print("\nValidating exports ...")
    ts_model = torch.jit.load(ts_path)
    ts_model.eval()
    with torch.no_grad():
        ts_err = float((ts_model(x_dummy) - y_ref).abs().max())
    print(f"  TorchScript max |diff| = {ts_err:.2e}  ({'ok' if ts_err < 1e-5 else 'MISMATCH'})")
    sess = None
    if onnx_path:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            onnx_err = float(abs(sess.run(None, {"input": x_dummy.numpy()})[0] - y_ref.numpy()).max())
            print(f"  ONNX max |diff|        = {onnx_err:.2e}  ({'ok' if onnx_err < 1e-4 else 'MISMATCH'})")
        except ImportError:
            print("  onnxruntime not installed — skipping the ONNX check.")

    # --- Latency benchmark ---------------------------------------------------
    print("\nBenchmarking inference latency (batch=64) ...")
    x_bench = torch.rand(64, 2, device=device)

    # PyTorch
    lat_torch = benchmark_inference(lambda x: model(x), x_bench)

    # TorchScript
    lat_ts = benchmark_inference(lambda x: ts_model(x), x_bench)

    # ONNX Runtime
    if sess is not None:
        lat_onnx = benchmark_inference(lambda x: sess.run(None, {"input": x.numpy()}), x_bench)
    else:
        lat_onnx = float("nan")

    print(f"\n  PyTorch:      {lat_torch:.3f} ms")
    print(f"  TorchScript:  {lat_ts:.3f} ms")
    print(f"  ONNX Runtime: {lat_onnx:.3f} ms")

    speedup_ts   = lat_torch / lat_ts   if lat_ts   > 0 else float("nan")
    speedup_onnx = lat_torch / lat_onnx if lat_onnx > 0 else float("nan")
    print(f"\n  TorchScript speedup:  {speedup_ts:.2f}x")
    print(f"  ONNX Runtime speedup: {speedup_onnx:.2f}x")


if __name__ == "__main__":
    main()
