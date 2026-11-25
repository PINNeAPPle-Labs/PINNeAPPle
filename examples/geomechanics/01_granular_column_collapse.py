"""Granular column collapse with the material point method, against the experimental runout law.

Columns of dry sand (friction angle 30°, cohesionless) of width L0 = 0.2 m and height H0 = a L0, against a back
wall on a rough floor, are released at t = 0 (pinneapple_simulation.particle_dynamics.geo_mpm: MLS-MPM, plane-strain
Drucker-Prager matched to Mohr-Coulomb). The final runout is compared with the scaling Lube et al. (2005) measured
for two-dimensional collapses: (L_inf - L0)/L0 = 1.2 a for a < 1.8 and 1.9 a^(2/3) for a > 2.8 (a transition in
between).

The frames are written in GeoTaichi's layout (particles/MPMParticle*.npz), read back through the GeoTaichi bridge
and turned into PhysicalSamples, the same path a GeoTaichi run of the same problem takes.

About 6 minutes on one core (dx = 1 cm). Writes docs/assets/geomechanics/column_collapse.png and .gif and
examples/geomechanics/_out/collapse_metrics.json.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinneapple_simulation.external_solvers.geotaichi import frames_to_samples, read_geotaichi
from pinneapple_simulation.particle_dynamics.geo_mpm import GeoMPM, Sand, box, runout

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "geomechanics")
os.makedirs(ASSETS, exist_ok=True)
L0, PHI = 0.2, 30.0
ASPECTS = (0.5, 1.0, 2.0, 3.0)


def lube(a):
    """Lube et al. (2005): the fit of the regime a belongs to; in the 1.8-2.8 transition, the mean of both."""
    a = np.asarray(a, float)
    return np.where(a < 1.8, 1.2 * a, np.where(a > 2.8, 1.9 * a ** (2 / 3), 0.5 * (1.2 * a + 1.9 * a ** (2 / 3))))


res = {}
for a in ASPECTS:
    sim = GeoMPM(domain=(L0 * (2 + 2.5 * a), a * L0 + 0.1), dx=0.01, floor="sticky", walls="slip")
    sim.add_body(box((0, 0), (L0, a * L0)), Sand(friction_deg=PHI))
    run_dir = os.path.join(OUT, f"a{a:g}")
    sim.run(t_end=0.6 + 0.4 * a, save_every=0.05, out=run_dir)
    frames = read_geotaichi(run_dir)                         # the GeoTaichi reader on our own frames
    last = frames[-1]
    res[a] = {"frames": frames, "dL": (runout(last["position"]) - L0) / L0,
              "H_end": float(np.quantile(last["position"][:, 1], 0.995)),
              "v_end": float(np.linalg.norm(last["velocity"], axis=1).max())}
    print(f"a = {a}: (L-L0)/L0 = {res[a]['dL']:.2f} (Lube {float(lube(a)):.2f}), final height "
          f"{res[a]['H_end'] / L0:.2f} L0, max speed at the end {res[a]['v_end']:.3f} m/s", flush=True)
samples = frames_to_samples(res[1.0]["frames"], source="pinneapple.geo_mpm")
metrics = {f"a={a:g}": {"runout_dL_over_L0": r["dL"], "lube_2005": float(lube(a)),
                        "ratio_to_lube": r["dL"] / float(lube(a)), "final_height_over_L0": r["H_end"] / L0,
                        "max_speed_at_end": r["v_end"]} for a, r in res.items()}
metrics["samples_per_run"] = len(samples)
json.dump(metrics, open(os.path.join(OUT, "collapse_metrics.json"), "w"), indent=1)

fig = plt.figure(figsize=(12, 6.5))
gs = fig.add_gridspec(len(ASPECTS), 2, width_ratios=(1.6, 1), hspace=0.6)
for i, a in enumerate(ASPECTS):
    ax = fig.add_subplot(gs[i, 0])
    f0, f1 = res[a]["frames"][0], res[a]["frames"][-1]
    ax.scatter(f0["position"][:, 0], f0["position"][:, 1], s=0.4, c="0.8")
    ax.scatter(f1["position"][:, 0], f1["position"][:, 1], s=0.4, c="#c8833a")
    ax.axvline(L0 * (1 + float(lube(a))), color="k", ls="--", lw=1)
    ax.set_xlim(0, L0 * (2 + 2.5 * max(ASPECTS)))
    ax.set_ylim(0, max(0.25, a * L0 + 0.02))
    ax.set_aspect("equal")
    ax.set_title(f"a = {a:g}: initial (grey), final (sand); dashed: Lube et al. runout", fontsize=9)
    ax.tick_params(labelsize=7)
ax = fig.add_subplot(gs[:, 1])
a1, a2 = np.linspace(0.3, 1.8, 50), np.linspace(2.8, 3.3, 20)
ax.plot(a1, 1.2 * a1, "k--", label="Lube et al. (2005): 1.2 a, a < 1.8")
ax.plot(a2, 1.9 * a2 ** (2 / 3), "k:", lw=2, label="1.9 a^(2/3), a > 2.8")
ax.plot(ASPECTS, [res[a]["dL"] for a in ASPECTS], "o-", color="#c8833a", label=f"MPM, φ = {PHI:g}°")
ax.set_xlabel("aspect ratio a = H0 / L0")
ax.set_ylabel("(L∞ - L0) / L0")
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(os.path.join(ASSETS, "column_collapse.png"), dpi=110, bbox_inches="tight")

from PIL import Image  # noqa: E402

from pinneapple_tools.visualization.style import use_theme  # noqa: E402

use_theme("report")   # one theme for every figure of the library (#918)

frames = res[2.0]["frames"]
imgs = []
for fr in frames:
    f, ax = plt.subplots(figsize=(7, 1.9))
    sp = np.linalg.norm(fr["velocity"], axis=1)
    ax.scatter(fr["position"][:, 0], fr["position"][:, 1], s=1.2, c=sp, cmap="inferno", vmin=0, vmax=2.0)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 0.45)
    ax.set_aspect("equal")
    ax.set_title(f"granular column collapse, a = 2, t = {float(fr['time']):.2f} s (colour: speed, 0-2 m/s)",
                 fontsize=8)
    ax.tick_params(labelsize=6)
    f.canvas.draw()
    imgs.append(np.asarray(f.canvas.buffer_rgba())[..., :3])
    plt.close(f)
pil = [Image.fromarray(i).convert("P", palette=Image.ADAPTIVE, colors=64) for i in imgs]
pil[0].save(os.path.join(ASSETS, "column_collapse.gif"), save_all=True, append_images=pil[1:], duration=80, loop=0)
print(json.dumps(metrics, indent=1))
