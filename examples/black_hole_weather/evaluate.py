"""Evaluate the black-hole weather forecaster: trust horizon, physics check without ground truth, figures, 3D twin.

    python examples/black_hole_weather/evaluate.py --runs data/bh --model runs/bh_one --test PL0SS0.1 --split test \\
        --out runs/bh_one/eval --twin

* Iterative rollouts from many start times in the test segment (as forecasts from many initial dates).
* Per lead time: MAE of the forecast, of persistence and of the time-mean flow, and the anomaly correlation;
  the trust horizon is where the correlation drops below 0.6 or the forecast stops beating persistence.
* Mass envelope: from the training runs, the fastest the mass in the window ever changed; the lead at which a
  rollout first breaks it is a trust limit found *without* the truth, compared here with the true horizon.
* Figures (skill, slices) and, with ``--twin``, a PINNeAPPle-Twin3D scene (simulation | forecast, with the
  forecast-error and mass-check sensors) plus a GIF recorded from the viewer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import load_run  # noqa: E402

from pinneapple_physics.blackhole.forecast import (MassEnvelope, lead_time_scores, load_forecaster, rollout,  # noqa: E402
                                                   trust_horizon, window_mass)


def cell_volumes(grid, codec):
    rf, tf = grid["r_faces"], grid["theta_faces"]
    vr = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0
    vt = np.cos(tf[:-1]) - np.cos(tf[1:])
    return codec.crop(2 * math.pi * vr[:, None] * vt[None, :])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--split", default="test", choices=["test", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=20, help="rollout length in blocks of 5 frames")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--twin-blocks", type=int, default=12)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)

    model, codec, ck = load_forecaster(os.path.join(a.model, "forecaster.pt"))
    with open(os.path.join(a.model, "setup.json")) as f:
        setup = json.load(f)
    stride, k = setup["stride"], ck["train"]["frames"]
    run = load_run(os.path.join(a.runs, a.test))
    rho = run["frames"][::stride, 0]
    t = run["t"][::stride]
    dt = float(np.diff(t).mean())
    x = codec.encode(rho)
    n = len(x)
    seg0 = int(0.8 * n) if (a.split == "test" and setup["one_sim"]) else int(0.2 * n)
    vol = cell_volumes(run["grid"], codec)

    # time-mean flow and mass envelope from the training runs
    train_x, masses = [], []
    for name in setup["train"]:
        rr = load_run(os.path.join(a.runs, name))["frames"][::stride, 0]
        if setup["one_sim"]:
            rr = rr[:int(0.7 * len(rr))]
        train_x.append(codec.encode(rr))
        masses.append(window_mass(codec.crop(rr), vol))
    mean_state = np.concatenate(train_x).mean(0)
    env = MassEnvelope.fit(masses, dt)

    # rollouts from several start times
    L = a.blocks * k
    starts = np.linspace(seg0, n - L - k, a.starts).astype(int) if n - L - k > seg0 else np.array([seg0])
    sc_all, viol, rolls = [], [], {}
    w = vol / vol.mean()
    for s0 in starts:
        x0 = x[s0:s0 + k]
        truth = x[s0 + k:s0 + k + L]
        pred = rollout(model, x0, a.blocks)[:len(truth)]
        sc_all.append(lead_time_scores(pred, truth, x0[-1], mean_state))
        m_pred = window_mass(codec.decode(pred), vol)
        m0 = window_mass(codec.decode(x0[-1:]), vol)[0]
        viol.append(env.first_violation(m_pred, dt, m0=m0))
        rolls[int(s0)] = (pred, truth)
    keys = ["mae", "persistence", "climatology", "acc"]
    mean_sc = {kk: np.mean([s[kk] for s in sc_all], 0) for kk in keys}
    hz = trust_horizon(mean_sc, dt)
    per_start = [trust_horizon(s, dt) for s in sc_all]
    lead = dt * (1 + np.arange(L))
    mass_lead = [None if v is None else float(lead[v]) for v in viol]
    summary = {"test": a.test, "frame_dt": dt, "block_dt": dt * k, "starts": [int(s) for s in starts],
               "horizon_mean": hz, "horizon_per_start": per_start, "mass_check_first_violation": mass_lead,
               "mass_rate_max": env.rate_max, "one_block_mae": float(mean_sc["mae"][:k].mean()),
               "one_block_persistence": float(mean_sc["persistence"][:k].mean()),
               "scores": {kk: v.tolist() for kk, v in mean_sc.items()}, "lead": lead.tolist()}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({kk: summary[kk] for kk in ("horizon_mean", "mass_check_first_violation", "one_block_mae",
                                                 "one_block_persistence")}, indent=1))

    skill_figure(lead, mean_sc, sc_all, hz, mass_lead, dt * k, os.path.join(a.out, "skill.png"), a.test)
    s_first = int(starts[0])
    pred, truth = rolls[s_first]
    slices_figure(run["grid"], codec, pred, truth, dt, k, os.path.join(a.out, "slices.png"))
    if a.twin:
        make_twin(run, codec, pred, truth, x[s_first:s_first + k], env, dt, k, a, vol, mean_sc)


def skill_figure(lead, mean_sc, sc_all, hz, mass_lead, block_dt, path, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for s in sc_all:
        ax[0].plot(lead, s["mae"], color="#d95f02", alpha=0.15, lw=1)
        ax[1].plot(lead, s["acc"], color="#d95f02", alpha=0.15, lw=1)
    ax[0].plot(lead, mean_sc["mae"], color="#d95f02", lw=2.4, label="U-Net forecast")
    ax[0].plot(lead, mean_sc["persistence"], color="#555", lw=1.6, ls="--", label="persistence")
    ax[0].plot(lead, mean_sc["climatology"], color="#1b9e77", lw=1.6, ls=":", label="time-mean flow")
    ax[0].set_ylabel("MAE of normalised log density")
    ax[1].plot(lead, mean_sc["acc"], color="#d95f02", lw=2.4)
    ax[1].axhline(0.6, color="#555", lw=1, ls="--")
    ax[1].set_ylabel("anomaly correlation")
    ax[1].set_ylim(-0.2, 1.02)
    for a_ in ax:
        a_.axvline(hz["acc"], color="#7570b3", lw=1.4)
        a_.set_xlabel("lead time [GM/c$^3$]")
        a_.grid(alpha=0.25)
        for b in np.arange(block_dt, lead[-1] + 1e-9, block_dt):
            a_.axvline(b, color="#ddd", lw=0.5, zorder=0)
    ml = [m for m in mass_lead if m is not None]
    if ml:
        ax[1].scatter(ml, [0.0] * len(ml), marker="v", color="#e7298a", zorder=5, label="mass check fires")
        ax[1].legend(loc="lower left", frameon=False)
    ax[0].legend(frameon=False)
    ax[1].text(hz["acc"], 1.0, f"  trust horizon {hz['acc']:.0f} M", color="#7570b3", va="top")
    fig.suptitle(f"Black-hole weather forecast skill ({name}, {len(sc_all)} start times)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def slices_figure(grid, codec, pred, truth, dt, k, path, leads=(1, 3, 6, 10)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rf = grid["r_faces"][:codec.r_cells + 1]
    tf = grid["theta_faces"][codec.theta_trim:len(grid["theta_faces"]) - codec.theta_trim]
    Rf = rf[:, None] * np.sin(tf)[None]
    Zf = rf[:, None] * np.cos(tf)[None]
    leads = [b for b in leads if b * k <= len(pred)]
    fig, ax = plt.subplots(2, len(leads), figsize=(3.1 * len(leads), 6.2))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for j, b in enumerate(leads):
        i = b * k - 1
        for row, (arr, lab) in enumerate([(truth, "simulation"), (pred, "U-Net")]):
            lg = np.log10(codec.decode(arr[i]))
            ax[row, j].pcolormesh(Rf, Zf, lg, vmin=codec.log_min, vmax=codec.log_max, cmap="inferno")
            ax[row, j].set_aspect("equal")
            ax[row, j].set_xlim(0, 60)
            ax[row, j].set_ylim(-45, 45)
            ax[row, j].set_xticks([])
            ax[row, j].set_yticks([])
            if row == 0:
                ax[row, j].set_title(f"+{(i + 1) * dt:.0f} M")
            if j == 0:
                ax[row, j].set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_twin(run, codec, pred, truth, x0, env, dt, k, a, vol, mean_sc):
    from pinneapple_physics.blackhole.capture import capture_twin
    from pinneapple_physics.blackhole.twin import accretion_scene

    nb = min(a.twin_blocks * k, len(pred))
    # frames: the k input frames (both panels equal), then the forecast
    sim = np.concatenate([x0, truth[:nb]])
    fc = np.concatenate([x0, pred[:nb]])
    full = lambda z: _uncrop(np.log10(codec.decode(z)), codec, run["grid"])
    times = dt * (np.arange(len(sim)) - (k - 1))
    err = np.concatenate([np.zeros(k), np.abs(pred[:nb] - truth[:nb]).mean((1, 2))])
    hz_err = float(np.interp(0.6, mean_sc["acc"][::-1], mean_sc["mae"][::-1]))     # MAE where ACC = 0.6
    m = window_mass(codec.decode(fc), vol)
    rate = np.concatenate([[0.0], np.abs(np.diff(np.log(m))) / dt])
    sensors = [
        {"id": "forecast error", "panel": "forecast", "series": err, "unit": "", "label": "forecast error (vs simulation)",
         "envelope": (0.0, hz_err), "quantity": "MAE log-density"},
        {"id": "mass check", "panel": "forecast", "series": rate * 1e3, "unit": "1e-3/M", "dx": 25.0,
         "label": "|d ln M / dt| (no ground truth)", "envelope": (0.0, env.margin * env.rate_max * 1e3)},
    ]
    sc = accretion_scene(run["grid"], {"simulation": full(sim), "forecast": full(fc)}, times, r_view=60.0,
                         title="Black hole weather: simulation vs U-Net forecast", sensors=sensors)
    folder = os.path.join(a.out, "twin")
    sc.export(folder)
    frames = capture_twin(folder, range(len(times)), field="log10_density", cmap="inferno", view="1,0.75,-1",
                          zoom=0.78, size=(1280, 640))
    _gif(frames, times, err, hz_err, rate, env, os.path.join(a.out, "twin_forecast.gif"), k)


def _uncrop(z, codec, grid):
    """Cropped (T, rc, nth - 2 trim) -> full (T, nr, nth) by padding with the window edge values."""
    nr, nt = len(grid["r"]), len(grid["theta"])
    out = np.empty(z.shape[:1] + (nr, nt), np.float32)
    out[:, :codec.r_cells, codec.theta_trim:nt - codec.theta_trim] = z
    out[:, codec.r_cells:] = out[:, codec.r_cells - 1:codec.r_cells]
    out[:, :, :codec.theta_trim] = out[:, :, codec.theta_trim:codec.theta_trim + 1]
    out[:, :, nt - codec.theta_trim:] = out[:, :, nt - codec.theta_trim - 1:nt - codec.theta_trim]
    return out


def _gif(frames, times, err, hz_err, rate, env, path, k):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    out = []
    for i, p in enumerate(frames):
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        d.text((W * 0.22, 18), "SIMULATION", fill=(235, 235, 235), font=font, anchor="ma")
        d.text((W * 0.72, 18), "U-NET FORECAST", fill=(235, 235, 235), font=font, anchor="ma")
        lab = "input frames" if times[i] <= 0 else f"lead +{times[i]:.0f} GM/c^3"
        d.text((24, H - 70), lab, fill=(235, 235, 235), font=font)
        trust = err[i] <= hz_err
        mass_ok = rate[i] <= env.margin * env.rate_max
        bar_w = int((W - 48) * min(1.0, err[i] / (2 * hz_err)))
        d.rectangle([24, H - 34, W - 24, H - 22], outline=(120, 120, 120))
        d.rectangle([24, H - 34, 24 + bar_w, H - 22], fill=(46, 160, 67) if trust else (215, 58, 73))
        d.text((W - 24, H - 70), ("trusted" if trust else "beyond trust horizon") +
               ("  |  mass check ok" if mass_ok else "  |  mass check: VIOLATED"),
               fill=(46, 160, 67) if trust and mass_ok else (215, 58, 73), font=small, anchor="ra")
        out.append(im.convert("P", palette=Image.ADAPTIVE, colors=200))
    durations = [600 if t <= 0 else 260 for t in times]
    durations[-1] = 1600
    out[0].save(path, save_all=True, append_images=out[1:], duration=durations, loop=0, optimize=True)


if __name__ == "__main__":
    main()
