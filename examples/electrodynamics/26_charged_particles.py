"""Charged particles in electric and magnetic fields (issue #1106): six cases, each checked against its closed form,
and an animation of a proton trapped in an Earth-like dipole field (the radiation-belt picture).

    python examples/electrodynamics/26_charged_particles.py            # figures + JSON, about 1-2 min
    python examples/electrodynamics/26_charged_particles.py --movie    # also the radiation-belt MP4s (needs ffmpeg)
    python examples/electrodynamics/26_charged_particles.py --quick    # shorter runs (CI)

Everything is computed by ``pinneapple_simulation.electromagnetics.particles`` (relativistic Boris pusher):

1. cyclotron motion of a 1 MeV electron: radius and frequency against gamma m v/(qB) and qB/(gamma m);
2. E x B drift (a cycloid) against E/B;
3. grad-B drift against (m v^2 / 2qB) |grad B|/B;
4. a magnetic bottle: which protons escape against the loss cone arcsin sqrt(B_min/B_max);
5. a Penning trap: cyclotron, magnetron and axial frequencies from the motion against the closed forms;
6. a 180-degree mass spectrometer: landing points of C-12, C-13 and C-14 against 2r.

Outputs in ``results/``: ``26_charged_particles.png`` (16:9 panel), ``26_charged_particles.json`` (every number),
``26_radiation_belt.png`` (hero still) and, with ``--movie``, ``26_radiation_belt_16x9.mp4`` / ``_4x5.mp4``.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinneapple_simulation.electromagnetics import particles as P  # noqa: E402

OUT = Path(__file__).parent / "results"
BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"
C1, C2, C3 = "#58a6ff", "#f0883e", "#3fb950"


def style(ax, title, three_d=False):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, loc="left")
    ax.tick_params(colors=MUTED, labelsize=7)
    if three_d:
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0, 0, 0, 0))
            a.label.set_color(MUTED)
    else:
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)


def note(ax, text, three_d=False):
    kw = dict(transform=ax.transAxes, color=FG, fontsize=8, va="bottom", ha="left",
              bbox=dict(facecolor="#161b22", edgecolor="#30363d", boxstyle="round,pad=0.3"))
    (ax.text2D if three_d else ax.text)(0.02, 0.02, text, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res = {}

    cyc = P.cyclotron_study(1.0e6, 1.0, "electron", steps_per_turn=200, turns=2 if a.quick else 5)
    dr = P.drift_study(periods=8 if a.quick else 20)
    bot = P.magnetic_bottle_study(n_particles=16 if a.quick else 40, bounces=1.5 if a.quick else 3.0)
    pen = P.penning_study(V0=900.0, x0_m=(1e-4, 0.0, 5e-5), v0_m_s=(0.0, 3.0e3, 0.0), periods=20 if a.quick else 40)
    ms = P.mass_spectrometer_study()
    res["cyclotron"] = {k: v for k, v in cyc.items() if k not in ("t", "x")}
    res["drifts"] = {k: v for k, v in dr.items() if k not in ("t", "x_exb", "x_gradb")}
    res["bottle"] = {"loss_cone_deg": bot["loss_cone_deg"], "mirror_ratio": bot["mirror_ratio"],
                     "agreement": bot["agreement"], "mu_relative_variation": bot["mu_relative_variation"],
                     "n_particles": int(len(bot["pitch_deg"])), "n_escaped": int(bot["escaped"].sum())}
    res["penning"] = {"exact_rad_s": pen["exact"], "measured_rad_s": pen["measured"]}
    res["mass_spectrometer"] = {"B_T": ms["B"], "energy_eV": ms["energy_eV"],
                                "ions": [{k: v for k, v in i.items() if k != "x"} for i in ms["ions"]],
                                "separation_m": ms["separation_m"]}

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle("Charged particles in E and B fields, computed with PINNeAPPle (Boris pusher) and checked",
                 color=FG, fontsize=15, x=0.02, ha="left", y=0.975)

    ax = fig.add_subplot(2, 3, 1)
    x = cyc["x"]
    rn = cyc["radius_nonrelativistic_m"]
    th = np.linspace(0, 2 * np.pi, 200)
    gc = x[:, :2].mean(0)
    ax.plot(x[:, 0] * 1e3, x[:, 1] * 1e3, color=C1, lw=1.5, label="1 MeV electron (Boris)")
    ax.plot((gc[0] + rn * np.cos(th)) * 1e3, (gc[1] + rn * np.sin(th)) * 1e3, "--", color=MUTED, lw=1,
            label="non-relativistic radius")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor=FG, edgecolor="#30363d", loc="upper right")
    style(ax, "1 · Cyclotron motion, B = 1 T")
    note(ax, f"γ = {cyc['gamma']:.3f}\nradius {cyc['radius_m'] * 1e3:.4f} mm (exact {cyc['radius_exact_m'] * 1e3:.4f})\n"
             f"ω {cyc['frequency_rad_s']:.4e} (exact {cyc['frequency_exact_rad_s']:.4e}) rad/s\n"
             f"energy drift {cyc['energy_rel_drift']:.0e}")

    ax = fig.add_subplot(2, 3, 2)
    xe = dr["x_exb"]
    ax.plot(xe[:, 0], xe[:, 1], color=C2, lw=1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    style(ax, "2 · E × B drift (cycloid), E perpendicular to B")
    note(ax, f"drift {dr['exb_drift_m_s']:.1f} m/s\nexact E/B = {dr['exb_drift_exact_m_s']:.1f} m/s")

    ax = fig.add_subplot(2, 3, 3)
    xg = dr["x_gradb"]
    ax.plot(xg[:, 0], xg[:, 1], color=C3, lw=1.0)
    ax.set_xlabel("x [m]  (|B| grows to the right)")
    ax.set_ylabel("y [m]")
    style(ax, "3 · Grad-B drift")
    note(ax, f"drift {dr['gradb_drift_m_s']:.1f} m/s\nexact {dr['gradb_drift_exact_m_s']:.1f} m/s"
             f"\nr_L / L_∇B = {dr['larmor_over_grad_scale']:.3f}")

    ax = fig.add_subplot(2, 3, 4)
    xs, tb = bot["x"], bot["t"]
    zm = 2.0
    for k in range(xs.shape[1]):
        esc = bot["escaped"][k]
        z = xs[:, k, 2]
        if esc:                                   # frozen after escaping: draw only up to the exit
            j = int(np.argmax(np.abs(z) > zm)) + 1 or len(z)
            ax.plot(tb[:j] * 1e6, z[:j], color=C2, lw=1.0, alpha=0.9)
        else:
            ax.plot(tb * 1e6, z, color=C1, lw=0.6, alpha=0.7)
    for zz in (-zm, zm):
        ax.axhline(zz, color=MUTED, ls="--", lw=0.8)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("t [µs]")
    ax.set_ylabel("z [m]  (mirrors at ±2 m)")
    style(ax, "4 · Magnetic bottle: trapped (blue), escaped (orange)")
    note(ax, f"mirror ratio {bot['mirror_ratio']:.1f}, loss cone {bot['loss_cone_deg']:.2f}°\n"
             f"escape matches the cone for {bot['agreement']:.0%} of {len(bot['pitch_deg'])} protons\n"
             f"μ varies {bot['mu_relative_variation']:.1%} along an orbit")

    ax = fig.add_subplot(2, 3, 5)
    xp = pen["x"]
    n1 = min(len(xp), int(2 * math.pi / pen["exact"]["magnetron"] / (pen["t"][1] - pen["t"][0])) + 1)
    ax.plot(xp[:n1, 0] * 1e6, xp[:n1, 1] * 1e6, color=C1, lw=0.8)      # one magnetron turn
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    style(ax, "5 · Penning trap seen from above (proton, 1 T, 900 V, d = 5 mm)")
    ex, me = pen["exact"], pen["measured"]
    note(ax, "\n".join(f"{n}: {me[k] / 2 / math.pi / 1e3:,.2f} kHz (exact {ex[k] / 2 / math.pi / 1e3:,.2f})"
                       for k, n in (("modified_cyclotron", "cyclotron f+"), ("axial", "axial fz"),
                                    ("magnetron", "magnetron f-"))))

    ax = fig.add_subplot(2, 3, 6)
    for ion, col in zip(ms["ions"], (C1, C3, C2), strict=False):
        ax.plot(ion["x"][:, 0] * 100, ion["x"][:, 1] * 100, color=col, lw=1.4, label=f"C-{ion['mass_u']:.0f}")
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor=FG, edgecolor="#30363d", loc="upper left")
    style(ax, "6 · Mass spectrometer, 2 keV ions, 0.5 T")
    note(ax, "\n".join(f"C-{i['mass_u']:.0f}: lands {i['landing_m'] * 100:.3f} cm (2r = {i['landing_exact_m'] * 100:.3f})"
                       for i in ms["ions"]))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "26_charged_particles.png", dpi=110, facecolor=BG)
    plt.close(fig)

    # radiation belt
    belt = P.dipole_trap_study(duration_s=40.0 if a.quick else 160.0, save_every=40)
    res["radiation_belt"] = {k: v for k, v in belt.items() if k not in ("t", "x")}
    hero(belt, OUT / "26_radiation_belt.png")
    if a.movie:
        for size, name in (((16, 9), "16x9"), ((8, 10), "4x5")):
            movie(belt, OUT / f"26_radiation_belt_{name}.mp4", size)
    (OUT / "26_charged_particles.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"cyclotron_radius_rel_err": abs(cyc["radius_m"] / cyc["radius_exact_m"] - 1),
                      "exb_rel_err": abs(dr["exb_drift_m_s"] / dr["exb_drift_exact_m_s"] - 1),
                      "gradb_rel_err": abs(dr["gradb_drift_m_s"] / dr["gradb_drift_exact_m_s"] - 1),
                      "bottle_agreement": bot["agreement"],
                      "belt_bounce_s": [belt["bounce_period_s"], belt["bounce_period_approx_s"]],
                      "belt_drift_s": [belt["drift_period_s"], belt["drift_period_approx_s"]]}, indent=1))


def _earth(ax):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#1f6feb", alpha=0.85,
                    linewidth=0, shade=True)
    for lon in range(0, 360, 45):
        p = P.dipole_field_line(2.0, longitude_deg=lon) / P.EARTH_RADIUS
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#8b949e", lw=0.5, alpha=0.5)


def _frame(fig, belt, k, trail):
    fig.clf()
    ax = fig.add_axes([0.0, 0.08, 1.0, 0.84], projection="3d")
    ax.set_facecolor(BG)
    ax.set_axis_off()
    _earth(ax)
    x = belt["x"] / P.EARTH_RADIUS
    j0 = max(0, k - trail)
    seg = x[j0:k + 1]
    n = len(seg)
    for i in range(max(1, n - 1)):
        a = (i + 1) / n
        ax.plot(seg[i:i + 2, 0], seg[i:i + 2, 1], seg[i:i + 2, 2], color=(0.94, 0.53, 0.24, a), lw=0.8)
    ax.scatter(*x[k], color="#ffd33d", s=18)
    lim = 2.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim * 0.7, lim * 0.7)
    ax.set_box_aspect((1, 1, 0.7), zoom=1.35)
    ax.view_init(elev=22, azim=-60 + 40 * k / max(1, len(x) - 1))
    t = belt["t"][k]
    fig.text(0.03, 0.95, "A proton trapped in Earth's magnetic field", color=FG, fontsize=15, va="top")
    fig.text(0.03, 0.91, "10 MeV, L = 2, computed with PINNeAPPle (relativistic Boris pusher, dipole field)",
             color=MUTED, fontsize=9, va="top")
    fig.text(0.03, 0.05,
             f"t = {t:6.1f} s   ·   gyration {belt['gyro_period_s'] * 1e3:.1f} ms   ·   "
             f"bounce {belt['bounce_period_s']:.2f} s (dipole formula {belt['bounce_period_approx_s']:.2f})   ·   "
             f"drift {belt['drift_period_s']:.0f} s around Earth ({belt['drift_period_approx_s']:.0f}), "
             f"{belt['drift_direction']}", color=FG, fontsize=8.5)


def hero(belt, path):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    _frame(fig, belt, len(belt["x"]) - 1, trail=len(belt["x"]))
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def movie(belt, path, size, seconds=14, fps=24):
    n = seconds * fps
    idx = np.linspace(0, len(belt["x"]) - 1, n).astype(int)
    fig = plt.figure(figsize=size, facecolor=BG, dpi=120)
    w, h = (int(size[0] * 120) // 2 * 2, int(size[1] * 120) // 2 * 2)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20",
                           "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    trail = max(40, len(belt["x"]) // 6)
    for k in idx:
        _frame(fig, belt, int(k), trail)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:h, :w]
        ff.stdin.write(buf.tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)


if __name__ == "__main__":
    main()
