"""A PINNeAPPle surrogate drives the twin: turntable clip of a field the network predicts at any instant.

    python examples/unity/04_surrogate_turntable.py          # trains, checks, records media/04_*.mp4

The pipeline, in the order it runs:

    closed-form slab solution  ->  training data (x, t) -> T
    PINNeAPPle Trainer         ->  a small MLP surrogate (two inputs, one output), weights saved to the twin folder
    trust card                 ->  the predicted field against the heat equation, its walls and the closed form
    turntable render           ->  8 s, camera orbiting 360 deg, the field evaluated *continuously* in time

That last point is what a baked field cannot do: the six time steps of the twin are gone, and the surface is
coloured at 30 frames per second straight from the surrogate -- the same role the network plays inside Unity (via
Sentis / Inference Engine, see docs/guides/unity_integration.md).

No Unity and no GPU: matplotlib draws the mesh, ffmpeg assembles the frames. The numbers on screen are the ones the
trust card approved.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import pinneapple as pp  # noqa: E402
from pinneapple_neural import TrainConfig, Trainer  # noqa: E402
from pinneapple_physics.closed_form.heat import slab_transient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
LIGHT = np.array([0.35, -0.45, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

L = 0.4                 # spool length [m]
ALPHA = 4e-6            # thermal diffusivity [m^2/s]
T_HOT, T_COLD, T0 = 180.0, 30.0, 30.0
T_END = 3600.0          # last time [s]
T_MIN = 0.02 * L * L / 4e-6     # the window the surrogate is trained and drawn in [s] (after the t = 0 jump)


# ----------------------------------------------------------------------------------------------- the twin's geometry
def _spool_scene():
    """The demo 01 twin: the spool plus its fin, and the two sensors."""
    spec = importlib.util.spec_from_file_location(
        "unity_demo_01", os.path.join(HERE, "01_engineering_twin_to_unity.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo.build_twin()


# ---------------------------------------------------------------------------------------------------- the surrogate
def train_surrogate(samples: int = 6000, epochs: int = 1500, seed: int = 0,
                    lr: float = 5e-4):
    """MLP (x/L, log t) -> (T - T0) / (T_HOT - T0), trained on the closed-form solution.

    The sample window starts at ``T_MIN``, not at 0: at t = 0 the analytic field is discontinuous at both walls, and
    a network asked to fit that jump spends its capacity on a singularity nobody sees (the animation starts after
    it). Time is log-uniform inside the window, because diffusion is fast at the start and slow later -- linear
    sampling would spend almost every point on the part of the curve that is already flat.
    """
    g = np.random.default_rng(seed)
    x = g.uniform(0.0, L, samples)
    t = T_MIN * (T_END / T_MIN) ** g.uniform(0.0, 1.0, samples)
    T = slab_transient(x, t, L, ALPHA, T_HOT, T_COLD, T0)
    X = torch.tensor(np.stack([x / L, np.log(t / T_MIN) / np.log(T_END / T_MIN)], 1), dtype=torch.float32)
    Y = torch.tensor(((T - T0) / (T_HOT - T0))[:, None], dtype=torch.float32)

    torch.manual_seed(seed)
    model = architecture()

    def loss(m, y_hat, batch):
        return torch.mean((y_hat - batch["y"]) ** 2)

    order = torch.randperm(samples)
    batches = [(X[order[i:i + 512]], Y[order[i:i + 512]]) for i in range(0, samples, 512)]
    with tempfile.TemporaryDirectory() as log_dir:          # never write a runs/ folder into the checkout
        cfg = TrainConfig(epochs=epochs, lr=lr, log_dir=log_dir, run_name="unity-surrogate",
                          device="cpu", seed=seed, grad_clip=1.0)
        out = Trainer(model, loss_fn=loss).fit(batches, cfg=cfg)
    model.eval()

    return model, predictor(model), out


def architecture():
    """The network the demo uses: two inputs (x/L, log t), three hidden layers, one output."""
    return torch.nn.Sequential(
        torch.nn.Linear(2, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )


def predictor(model):
    """``(x, t) -> T [C]``, the way Unity evaluates the exported network."""
    model.eval()

    def predict(x_query, t_query):
        tq = np.maximum(np.asarray(t_query, float).ravel(), T_MIN)
        xt = np.stack([np.asarray(x_query, float).ravel() / L,
                       np.log(tq / T_MIN) / np.log(T_END / T_MIN)], 1)
        with torch.no_grad():
            y = model(torch.tensor(xt, dtype=torch.float32)).numpy().ravel()
        return T0 + (T_HOT - T0) * y

    return predict


def load_surrogate(path: str):
    """A fit saved by an earlier run, re-checked before it is used (the card is cheap; trusting a file is not)."""
    model = architecture()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model, predictor(model)


def check_surrogate(predict) -> tuple[dict, float]:
    """The predicted field against the heat equation, its walls and the closed form."""
    xs = np.linspace(0.0, L, 41)
    ts = np.linspace(T_MIN, T_END, 25)                      # the smooth window (no t = 0 jump)
    pred = np.stack([predict(xs, np.full_like(xs, t)) for t in ts], axis=1)      # (nx, nt): x first, t second
    ref = slab_transient(xs[:, None], ts[None, :], L, ALPHA, T_HOT, T_COLD, T0)
    rel = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
    card = pp.trust_card({"x": xs, "t": ts}, pred, pde="heat", alpha=ALPHA,
                         dirichlet={"x_min": T_HOT, "x_max": T_COLD})
    return card, rel


# ------------------------------------------------------------------------------------------------------- the movie
def _faces(vertices, faces, values, cmap, norm, grey=None):
    tri = vertices[faces]
    if grey is not None:
        colors = np.tile(np.append(grey, 1.0), (len(tri), 1))
    else:
        colors = cmap(norm(values[faces].mean(axis=1)))
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    colors[:, :3] *= (0.5 + 0.5 * np.abs(n @ LIGHT))[:, None]
    return Poly3DCollection(tri, facecolors=colors, edgecolors="none", linewidths=0)


def record(scene, predict, movie: str, *, seconds: float = 8.0, fps: int = 30,
           elev: float = 20.0, zoom: float = 1.35) -> str:
    """A turntable: the camera orbits 360 deg while the field follows the surrogate in time.

    Left: the twin, coloured per face by the network at that instant. Right: the sensor traces over the whole clip
    with a marker at the current time, so the clip shows a number rising *and* the field it comes from.
    """
    os.makedirs(MEDIA, exist_ok=True)
    frames = os.path.join(tempfile.gettempdir(), "pinneapple_turntable")
    os.makedirs(frames, exist_ok=True)
    spool = next(p for p in scene.parts if p.name == "spool")
    fin = next(p for p in scene.parts if p.name == "fin")
    allv = np.concatenate([p.vertices for p in scene.parts])
    mn, mx = allv.min(0), allv.max(0)
    span = float((mx - mn).max())
    pad = 0.12 * span                                        # room for the sensor labels, which used to clip
    cmap, norm = plt.get_cmap("turbo"), Normalize(T_COLD, T_HOT)
    n = int(round(seconds * fps))
    times = T_MIN + (T_END - T_MIN) * np.linspace(0.0, 1.0, n)
    # each sensor reads the surrogate at the vertex nearest to it, for every instant of the clip
    sens = []
    for sensor in scene.sensors:
        d = np.linalg.norm(spool.vertices - np.asarray(sensor.position, float), axis=1)
        k = int(np.argmin(d))
        sens.append((sensor, spool.vertices[k, 0], predict(np.full(n, spool.vertices[k, 0]), times)))
    for frame in range(n):
        u = frame / (n - 1)
        t = float(times[frame])
        field = predict(spool.vertices[:, 0], np.full(len(spool.vertices), t))
        fig = plt.figure(figsize=(11.0, 6.0), dpi=110)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.02)
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax.add_collection3d(_faces(spool.vertices, spool.faces, field, cmap, norm))
        ax.add_collection3d(_faces(fin.vertices, fin.faces, None, cmap, norm, grey=(0.78, 0.66, 0.45)))
        for sensor, _x_s, series in sens:
            value = float(series[frame])
            ax.scatter(*sensor.position, c="gold", s=60, depthshade=False, edgecolors="black", linewidths=0.6)
            ax.text(*sensor.position, f" {sensor.id} {value:.0f} C", fontsize=9, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.set_xlim(mn[0] - pad, mx[0] + pad)
        ax.set_ylim(mn[1] - pad, mx[1] + pad)
        ax.set_zlim(mn[2] - pad, mx[2] + pad)
        ax.set_box_aspect(mx - mn, zoom=zoom)
        ax.view_init(elev=elev, azim=-90.0 + 360.0 * u)
        ax.set_axis_off()
        ax.set_title("PINNeAPPle surrogate -> twin", fontsize=12, pad=0)

        chart = fig.add_subplot(gs[0, 1])
        for sensor, _x, series in sens:
            chart.plot(times / 60.0, series, lw=2.0 if sensor.series is None else 1.4, label=sensor.id)
            chart.plot([t / 60.0], [series[frame]], "o", ms=7, mec="black", mew=0.8)
        chart.axvline(t / 60.0, color="0.45", lw=0.8, ls=":")
        chart.set_xlim(times[0] / 60.0, times[-1] / 60.0)
        chart.set_ylim(T_COLD - 5, T_HOT + 5)
        chart.set_xlabel("time [min]", fontsize=9)
        chart.set_ylabel("temperature [C]", fontsize=9)
        chart.tick_params(labelsize=8)
        chart.legend(fontsize=8, loc="lower right", frameon=False)
        chart.grid(alpha=0.25, lw=0.5)
        chart.set_title(f"t = {t / 60:4.1f} min    frame {frame + 1}/{n}", fontsize=10)
        for side in ("top", "right"):
            chart.spines[side].set_visible(False)

        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.55, pad=0.0, aspect=28)
        cb.set_label("temperature [C]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.subplots_adjust(left=0.0, right=0.97, top=0.92, bottom=0.08)
        fig.savefig(os.path.join(frames, f"f{frame:04d}.png"))
        plt.close(fig)
    dest = os.path.join(MEDIA, movie)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(frames, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", dest], check=True)
    for name in os.listdir(frames):
        os.remove(os.path.join(frames, name))
    os.rmdir(frames)
    print(f"{movie}: {n} frames, {seconds:.0f} s at {fps} fps, {os.path.getsize(dest) // 1024} kB, "
          f"{span:.2f} m across")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="train the surrogate, check it, record the turntable")
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3, help="attempts before giving up on a bad fit")
    ap.add_argument("--no-movie", action="store_true", help="train and check only")
    ap.add_argument("--resume", action="store_true", help="reuse _out/surrogate.pt instead of retraining")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    scene = _spool_scene()
    print(f"twin: {scene.title} ({len(scene.parts)} parts, {len(scene.times)} baked time steps)")

    # A fit is not accepted because it ran: it is accepted because the field it predicts passes the checks below.
    # With Adam at a fixed rate on a small net, a run occasionally diverges after it converged, so a seed that does
    # not reach the target is retried -- and the retries are printed, not hidden.
    weights = os.path.join(HERE, "_out", "surrogate.pt")
    card = rel = None
    model = predict = None
    if a.resume and os.path.exists(weights):
        model, predict = load_surrogate(weights)
        card, rel = check_surrogate(predict)
        print(f"reusing {os.path.relpath(weights, HERE)}: relative L2 = {rel:.3%}")
    for attempt, seed in enumerate(range(a.seeds), start=1):
        if predict is not None:
            break
        print(f"training the surrogate (attempt {attempt}): {a.samples} samples, {a.epochs} epochs, seed {seed} ...")
        model, predict, out = train_surrogate(a.samples, a.epochs, seed=seed)
        print(f"  last training loss (normalised MSE): {out['history'][-1]['train_total']:.3e}")
        card, rel = check_surrogate(predict)
        print(f"  surrogate vs closed form on a 41 x 25 grid: relative L2 = {rel:.3%}")
        if rel < 0.02 and card.decision == "APPROVED":
            break
    if card is None:
        raise SystemExit("the surrogate never trained")
    for check in card.checks:
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[check.passed]
        value = "" if check.value is None else f"  value {check.value:.4g}"
        print(f"  {mark}  {check.name}{value}")
    print(f"  decision: {card.decision}")

    os.makedirs(os.path.dirname(weights), exist_ok=True)
    if not a.resume:
        torch.save(model.state_dict(), weights)
    print(f"  weights: {os.path.relpath(weights, HERE)}  (export_onnx of this model is what Sentis runs in Unity)")
    if card.decision != "APPROVED":
        print(f"  the card refused this fit ({card.decision}): the clip below would show a field nobody signed off on")

    if a.no_movie:
        return 0 if card.decision == "APPROVED" else 1
    dest = record(scene, predict, "04_surrogate_turntable.mp4", seconds=a.seconds, fps=a.fps)
    print(dest)
    return 0 if card.decision == "APPROVED" else 1


if __name__ == "__main__":
    sys.exit(main())
