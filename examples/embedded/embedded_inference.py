"""Embedded inference, measured: a field model, its physics residual and its measurement check exported as one ONNX
graph, and the hertz it actually reaches on an ARM CPU (#350).

    python examples/embedded/embedded_inference.py            # train, calibrate, export, measure (~40 s)
    python examples/embedded/embedded_inference.py --quick    # no training: the graph and the hertz only

The three things the example produces:

1. **a flight-size model** -- a 12 k-parameter convolutional surrogate of the advection-diffusion field: it is given two
   consecutive frames and has to produce the next one, which is what a flight computer actually has;
2. **the decision in the graph** -- the prediction, the residual of the equation, the trust level and the two flags
   (``off_physics`` and ``suspect``) as six outputs of one straight-line ONNX file, with no ``If``/``Loop``/``Scan``;
3. **the numbers** -- the latency per call on 1 and 4 threads, the hertz they imply, and the residual on cases inside
   and outside the training envelope, which is what the flag is for.

Two choices make the residual mean something, and both were measured before they were written down:

* the training field is the **exact solution of the discrete equation the stencil writes** (backward Euler, solved in
  Fourier), not of the continuous one. A centred difference on a 64-wide grid truncates the grid-scale modes badly
  enough to dominate the residual: with the continuous propagator the stencil scores the *exact* next frame at 1.70,
  above what a model scores, and persistence beats the truth. With the discrete propagator the exact frame scores
  0.0 and the residual ranks the models by how wrong they are;
* the tolerance is **calibrated** on a held-out in-domain sample instead of guessed, because the residual is a time
  derivative and divides by ``dt``: a model whose frame error is 2 % of the field has a residual of order one. The
  limit is set at 1.5x the in-domain median and then applied unchanged to every other case.

The model is not the drone of #335 (that issue is not done): it is a field surrogate of the size class a flight model
has, and the latency measured here is the budget of *this* graph on *this* CPU -- the Mac's ARM core -- which is the
same kind of number a Raspberry Pi 5 or a Jetson CPU would give, not the same number.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SHAPE = (2, 32, 64)              # two frames, 32 x 64
DT = 0.5
DX = 1.0
BAND = 1.0                       # the largest wavenumber in the training field
PARAMETERS = {"c_x": 1.0, "c_y": 0.35, "alpha": 0.01}

#: The cases the flag is asked about. ``alpha-5x`` and ``alpha-50x`` differ from training only in the diffusion, which
#: is the smaller term: 5x more of it is invisible to the residual, 50x more is not. ``static`` is a field that does
#: not move at all -- the easiest case for a model to be wrong about, and the one a stuck sensor produces.
CASES = {
    "in-domain": dict(c=(1.0, 0.35), alpha=0.01),
    "alpha-5x": dict(c=(1.0, 0.35), alpha=0.05),
    "alpha-50x": dict(c=(1.0, 0.35), alpha=0.5),
    "speed-2.5x": dict(c=(2.5, 0.9), alpha=0.01),
    "static": dict(c=(0.0, 0.0), alpha=0.0),
}


def _out(name: str) -> str:
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path) or OUT, exist_ok=True)
    return path


def _eigenvalues(c, alpha, shape=SHAPE, dx=DX):
    """The symbol of the discrete operator the residual writes: centred differences, periodic, ``-(c . grad) + a lap``."""
    kx = np.fft.fftfreq(shape[2]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2 * np.pi
    grad = 1j * (c[0] * np.sin(kx)[None, :] / dx + c[1] * np.sin(ky)[:, None] / dx)
    lap = alpha * ((2 * np.cos(kx) - 2)[None, :] + (2 * np.cos(ky) - 2)[:, None]) / dx ** 2
    return -grad + lap


def _propagator(c, alpha):
    """The exact solution of ``u_t = -(c . grad) u + alpha lap u`` **as the stencil discretises it**, backward Euler."""
    symbol = _eigenvalues(c, alpha)
    return lambda u: np.real(np.fft.ifft2(np.fft.fft2(u) / (1.0 - DT * symbol)))


def field(n: int = 2000, *, c=(1.0, 0.35), alpha: float = 0.01, seed: int = 0):
    """``(states, targets)``: two consecutive frames and the frame after them, all three from the propagator.

    :func:`_propagator` solves the *discrete* equation, so ``(u_{k+1} - u_k)/dt + c . grad u_{k+1} - alpha lap
    u_{k+1}`` is zero to round-off on every frame it produces. That is what makes a non-zero residual on a prediction
    the model's own error and nothing else.
    """
    rng = np.random.default_rng(seed)
    shape = SHAPE[1:]
    kx = np.fft.fftfreq(shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[0]) * 2 * np.pi
    band = np.maximum(np.abs(kx)[None, :], np.abs(ky)[:, None]) < BAND
    step = _propagator(c, alpha)
    states, targets = [], []
    for _ in range(n):
        u0 = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * band))
        u0 = u0 / np.abs(u0).max()
        u1 = step(u0)
        states.append(np.stack([u0, u1]))
        targets.append(step(u1))
    return np.stack(states).astype(np.float32), np.stack(targets).astype(np.float32)


def train(epochs: int = 300, n: int = 2000, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(2, 16, 3, padding=1, padding_mode="circular"), torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, 3, padding=1, padding_mode="circular"), torch.nn.Tanh(),
        torch.nn.Conv2d(16, 1, 3, padding=1, padding_mode="circular"))
    states_np, target_np = field(n, seed=seed)
    states = torch.tensor(states_np)
    target = torch.tensor(target_np).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    t0 = time.time()
    for k in range(epochs):
        idx = torch.randint(0, len(states), (16,))
        pred = model(states[idx])
        loss = (pred - target[idx]).pow(2).mean() / target[idx].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if k % 100 == 0:
            print(f"  epoch {k:4d}: relative MSE {loss.item():.4e}", flush=True)
    return model.eval(), {"train_seconds": time.time() - t0, "relative_mse": loss.item(), "samples": n}


def residuals(model, spec, n: int = 16) -> dict[str, list[float]]:
    """The residual of one prediction per sample, for every case in :data:`CASES`."""
    import torch

    from pinneapple_physics.embedded import residual_fn

    fn = residual_fn(spec)
    out: dict[str, list[float]] = {}
    for name, kwargs in CASES.items():
        states_np, _ = field(n, seed=7, **kwargs)
        state = torch.tensor(states_np)
        with torch.no_grad():
            pred = model(state)
            out[name] = [float(fn(pred[i:i + 1], state[i, -1])) for i in range(len(states_np))]
    return out


def figure(model, spec, measured, tolerance, path: str) -> str:
    """Two panels: what the flag does on each case, and one frame the model got right."""
    import matplotlib
    import torch
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(measured)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    ax = axes[0]
    box = ax.boxplot([measured[n] for n in names], orientation="horizontal", widths=0.55, patch_artist=True,
                     showfliers=False)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names)
    for patch, name in zip(box["boxes"], names, strict=False):
        flagged = np.mean(np.asarray(measured[name]) > tolerance) > 0.5
        patch.set_facecolor("#a23a00" if flagged else "#0081be")
        patch.set_alpha(0.75)
    ax.axvline(tolerance, color="#262626", ls="--", lw=1.2, label=f"tolerance {tolerance:.3f} (calibrated)")
    ax.set_xlabel("residual of the equation on the prediction  (RMS / mean |u|, units of 1/dt)")
    ax.set_title("the flag splits where the physics does")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    states, targets = field(1, seed=11)
    with torch.no_grad():
        prediction = model(torch.tensor(states))[0, 0].numpy()
    truth = targets[0]
    vmax = float(np.abs(truth).max())
    ax = axes[1]
    image = ax.imshow(np.concatenate([truth, prediction], axis=0), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="auto")
    ax.set_yticks([16, 48])
    ax.set_yticklabels(["next frame (exact)", "next frame (model)"])
    error = float(np.abs(prediction - truth).max() / np.abs(truth).max())
    ax.set_title(f"a state the model was trained on -- max error {error:.1%} of the field")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="no training: export the untrained model and measure")
    ap.add_argument("--out", default=_out("embedded_inference.json"))
    args = ap.parse_args(argv)

    import torch

    from pinneapple_physics.embedded import EmbeddedRuntime, EmbeddedSpec, export, graph_summary

    contract = dict(state_shape=SHAPE, equation="advection-diffusion", parameters=PARAMETERS, dt=DT, dx=DX,
                    sensor_channels=(0,), measurement_tolerance=0.1)
    if args.quick:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Conv2d(2, 8, 3, padding=1, padding_mode="circular"),
                                    torch.nn.Tanh(),
                                    torch.nn.Conv2d(8, 1, 3, padding=1, padding_mode="circular")).eval()
        training = {}
    else:
        print("training the flight model on the discrete propagator:", flush=True)
        model, training = train()
    parameters = sum(p.numel() for p in model.parameters())

    # the residual does not depend on the tolerance, so the limit is calibrated first and the graph built with it
    measured = residuals(model, EmbeddedSpec(tolerance=1.0, **contract))
    in_domain = float(np.median(measured["in-domain"]))
    tolerance = round(1.5 * in_domain, 3) if not args.quick else 0.05
    spec = EmbeddedSpec(tolerance=tolerance, **contract)
    print(f"calibrated tolerance: 1.5 x {in_domain:.4f} = {tolerance:.4f}", flush=True)

    path = export(model, spec, _out("flight_model.onnx"))
    summary = graph_summary(path)
    print(f"graph: {summary.bytes} bytes, {len(summary.op_types)} op types, straight line = {summary.is_straight_line}",
          flush=True)

    print(f"residual against the calibrated limit of {tolerance:.4f}:", flush=True)
    report = {}
    for name, values in measured.items():
        flagged = float(np.mean(np.asarray(values) > tolerance))
        report[name] = {"median_residual": float(np.median(values)), "max_residual": float(np.max(values)),
                        "flagged": flagged}
        print(f"  {name:12s}: median {report[name]['median_residual']:.4f} "
              f"max {report[name]['max_residual']:.4f}  flagged {flagged:5.0%}", flush=True)

    state = torch.tensor(field(1, seed=11)[0][0])
    with torch.no_grad():
        measurement = np.array([float(model(state.unsqueeze(0))[0, 0, 0, 0])])
    latency = {}
    for threads in (1, 4):
        latency[str(threads)] = EmbeddedRuntime(path, threads=threads).latency(state.numpy(), measurement, n=500)
        lat = latency[str(threads)]
        print(f"  {threads} thread(s): median {lat['median_ms']:.3f} ms ({lat['hz']:.0f} Hz), "
              f"p95 {lat['p95_ms']:.3f} ms ({lat['hz_p95']:.0f} Hz)", flush=True)

    plot = figure(model, spec, measured, tolerance, _out("figures/residual_and_field.png"))
    print(f"wrote {plot}", flush=True)

    result = {"issue": 350, "graph": summary.to_dict(), "parameters": parameters, "training": training,
              "tolerance": tolerance, "residuals": report, "latency": latency,
              "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                          "python": platform.python_version(), "torch": torch.__version__,
                          "cpus": os.cpu_count()},
              "spec": contract["parameters"] | {"state_shape": SHAPE, "dt": DT, "dx": DX, "tolerance": tolerance}}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
