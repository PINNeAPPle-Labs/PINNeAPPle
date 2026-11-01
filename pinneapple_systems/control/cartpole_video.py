"""Animation of the cart-pole study: the pendulum on its cart, the angle trace and the poles, as an MP4 and a still.

    from pinneapple_systems.control.cartpole_video import cartpole_animation
    r = cartpole_animation("out/", push=-40.0)        # out/cartpole.mp4, out/cartpole_poster.png, plus the metrics

Every number drawn on the frames comes from :func:`~.cartpole.cartpole_study` run inside this call. The layout is a
4:5 portrait card on a dark background:
* the cart and pendulum, followed by the camera, with a fading trail coloured by the controller in charge (amber for
  the swing-up, teal for the LQR) and an arrow during the push;
* the angle against time;
* the open-loop and closed-loop poles in the s-plane.

Needs matplotlib and ffmpeg (on the PATH, or through ``imageio-ffmpeg``).
"""
from __future__ import annotations

import math
import os
import shutil
from typing import Any

import numpy as np

from .cartpole import CartPoleParams, cartpole_study

__all__ = ["cartpole_animation"]

BG, FG, MUTED, GRID = "#0b0f17", "#e8edf5", "#8a96a8", "#1c2433"
AMBER, TEAL, RED = "#ffb547", "#2ee6c5", "#ff5d6c"


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def cartpole_animation(out_dir: str, push: float = -25.0, t_push: float = 12.0, a_swing: float = 3.0,
                       t_end: float = 16.0, fps: int = 20, width_px: int = 720, title: str = "Balancing the unbalanceable"
                       ) -> dict[str, Any]:
    """Run the cart-pole study and render it as an animation.

    Parameters
    ----------
    out_dir : str
        Folder for ``cartpole.mp4`` and ``cartpole_poster.png``.
    push : float [rad/s^2]
        Angular-acceleration push, 0.1 s long, at ``t_push``.
    t_push : float [s]
        Time of the push.
    a_swing : float [m/s^2]
        Cart acceleration limit while pumping energy.
    t_end : float [s]
        Simulated duration.
    fps : int [1/s]
        Frames per simulated second (the video plays in real time).
    width_px : int [px]
        Frame width; the height is 5/4 of it.
    title : str
        Title drawn on the frames.

    Returns
    -------
    files : list of str
        File names written in ``out_dir``.
    frames : int
        Frames in the video.
    unstable_pole : float [1/s]
        Open-loop pole at upright.
    doubling_time : float [s]
        Doubling time of an error at upright.
    free_fall_time_to_90deg : float [s]
        No control: from 1 deg to 90 deg.
    catch_time : float [s]
        When the LQR took over.
    push_peak_deg : float [deg]
        Largest angle after the push.
    push_recovery : float [s]
        Back within 2 deg after the push.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch

    os.makedirs(out_dir, exist_ok=True)
    P = CartPoleParams(a_swing=float(a_swing))
    r = cartpole_study(P, t_end=float(t_end), t_push=float(t_push), push=float(push))
    t, X, M = r["t"], r["x"], r["mode"]
    TH = np.radians(np.asarray(r["angle_deg"], float))
    L = P.length
    dt = float(t[1] - t[0])
    step = max(1, int(round(1.0 / (fps * dt))))
    idx = np.arange(0, len(t), step)
    t_catch = r["catch_time"]
    s = width_px / 1080.0                                       # font scale against the 1080 px design

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100 * s, facecolor=BG)
    fig.text(0.06, 0.955, title, color=FG, fontsize=34, weight="bold")
    fig.text(0.06, 0.925, "Inverted pendulum on a cart · swing-up + LQR · simulated with PINNeAPPle", color=MUTED,
             fontsize=16)
    ax = fig.add_axes([0.04, 0.47, 0.92, 0.36])
    ax.set_facecolor(BG)
    half = 1.15
    ax.set_ylim(-0.62, 0.68)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-8, 8], [-0.12, -0.12], color=GRID, lw=3, solid_capstyle="round")
    for xx in np.arange(-8, 8.01, 0.25):
        major = abs(xx - round(xx)) < 1e-9
        ax.plot([xx, xx], [-0.15, -0.09], color="#3a4a66" if major else GRID, lw=1.5)
        if major:
            ax.text(xx, -0.22, f"{xx:+.0f} m", color="#3a4a66", fontsize=11, ha="center")
    cart = FancyBboxPatch((0, 0), 0.36, 0.16, boxstyle="round,pad=0.0,rounding_size=0.04", fc="#243046",
                          ec="#3a4a66", lw=1.5)
    ax.add_patch(cart)
    rod, = ax.plot([], [], lw=5, color=FG, solid_capstyle="round", zorder=4)
    bob = Circle((0, 0), 0.055, color=FG, zorder=5)
    ax.add_patch(bob)
    trail = ax.scatter([], [], s=6, zorder=3)
    phase_txt = fig.text(0.06, 0.885, "", color=AMBER, fontsize=24, weight="bold", va="top")
    phase_sub = fig.text(0.06, 0.855, "", color=MUTED, fontsize=15, va="top")
    clock = fig.text(0.94, 0.885, "", color=MUTED, fontsize=16, ha="right", va="top", family="monospace")
    arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=4, mutation_scale=30))
    ax2 = fig.add_axes([0.08, 0.20, 0.52, 0.22])
    ax3 = fig.add_axes([0.68, 0.20, 0.27, 0.22])
    for a in (ax2, ax3):
        a.set_facecolor(BG)
        for sp in a.spines.values():
            sp.set_color(GRID)
        a.tick_params(colors=MUTED)
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(-190, 190)
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.axhline(0, color=GRID, lw=1)
    ax2.set_xlabel("time (s)", color=MUTED)
    ax2.set_ylabel("angle from upright (°)", color=MUTED)
    if math.isfinite(t_catch):
        ax2.axvspan(t_catch, t[-1], color=TEAL, alpha=0.06)
    ax2.axvline(t_push, color=RED, lw=1, ls=":")
    l1, = ax2.plot([], [], color=AMBER, lw=2.2)
    l2, = ax2.plot([], [], color=TEAL, lw=2.2)
    ol = np.array([r["unstable_pole"], -r["unstable_pole"], 0.0, 0.0])
    cl = np.array([[p.real, p.imag] for p in r["closed_loop_poles"]])
    lim = max(9.0, float(np.max(np.abs(cl[:, 0]))) * 1.15)
    ax3.set_xlim(-lim, 6)
    ax3.set_ylim(-5, 5)
    ax3.axvline(0, color=GRID, lw=1)
    ax3.axhline(0, color=GRID, lw=1)
    ax3.axvspan(0, 6, color=RED, alpha=0.08)
    ax3.set_title("poles (1/s)", color=MUTED, fontsize=13, loc="left")
    ax3.scatter(ol, 0 * ol, marker="x", s=110, color=RED, lw=3, zorder=5)
    ax3.text(ol.max(), 1.0, f"+{ol.max():.2f}", color=RED, fontsize=12, ha="center")
    cls = ax3.scatter([], [], s=70, color=TEAL, zorder=6)
    fig.text(0.06, 0.13, f"Upright is unstable: pole at +{r['unstable_pole']:.2f} 1/s  →  an error doubles every "
             f"{r['doubling_time'] * 1000:.0f} ms.", color=FG, fontsize=16)
    fig.text(0.06, 0.10, f"Without control: 1° becomes 90° in {r['free_fall_time_to_90deg']:.2f} s.   With control: "
             f"caught at {t_catch:.2f} s, push of {r['push_peak_deg']:.0f}° recovered in {r['push_recovery']:.2f} s.",
             color=MUTED, fontsize=13.5)
    fig.text(0.06, 0.04, "pinneapple_systems.control · linearize → stability → lqr → closed-loop simulation",
             color=MUTED, fontsize=12)

    def draw(i):
        ti, p, th = t[i], X[i, 0], TH[i]
        cart.set_x(p - 0.18)
        cart.set_y(-0.10)
        ax.set_xlim(p - half, p + half)
        bx, by = p + L * np.sin(th), -0.02 + L * np.cos(th)
        col = TEAL if M[i] == "lqr" else AMBER
        rod.set_data([p, bx], [-0.02, by])
        rod.set_color(col)
        bob.center = (bx, by)
        bob.set_color(col)
        j0 = max(0, i - int(1.6 / dt))
        js = np.arange(j0, i + 1, 4)
        tx, ty = X[js, 0] + L * np.sin(TH[js]), -0.02 + L * np.cos(TH[js])
        alpha = np.linspace(0.05, 0.9, len(js))
        trail.set_offsets(np.c_[tx, ty])
        trail.set_color([matplotlib.colors.to_rgba(TEAL if M[j] == "lqr" else AMBER, a) for j, a in zip(js, alpha, strict=True)])
        if ti < t_catch:
            phase_txt.set_text("1 · Pumping energy")
            phase_txt.set_color(AMBER)
            phase_sub.set_text(f"energy-based swing-up, cart acceleration ≤ {P.a_swing:g} m/s²")
        elif ti < t_push:
            phase_txt.set_text("2 · LQR catches it")
            phase_txt.set_color(TEAL)
            phase_sub.set_text("linearised at upright · gains from the Riccati equation")
        else:
            phase_txt.set_text("3 · Push rejected")
            phase_txt.set_color(TEAL)
            phase_sub.set_text(f"peak {r['push_peak_deg']:.0f}°, back within 2° in {r['push_recovery']:.2f} s")
        clock.set_text(f"t = {ti:5.2f} s")
        arrow.set_visible(t_push <= ti < t_push + 0.5)
        if arrow.get_visible():
            sgn = 1.0 if push < 0 else -1.0
            arrow.xy = (bx + sgn * 0.08, by)
            arrow.set_position((bx + sgn * 0.5, by))
        deg = np.degrees(TH[:i + 1]).copy()
        deg[np.where(np.abs(np.diff(deg)) > 180)[0]] = np.nan
        tt = t[:i + 1]
        m = tt < t_catch
        l1.set_data(tt[m], deg[m])
        l2.set_data(tt[~m], deg[~m])
        cls.set_offsets(cl if ti >= t_catch else np.empty((0, 2)))

    files = []
    ff = _ffmpeg()
    W = int(round(width_px / 2) * 2)
    H = int(round(W * 1.25 / 2) * 2)
    if ff:
        import subprocess
        mp4 = os.path.join(out_dir, "cartpole.mp4")
        proc = subprocess.Popen([ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
                                 f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-crf", "21", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
        fig.set_size_inches(W / fig.dpi, H / fig.dpi)
        for i in idx:
            draw(i)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            if buf.shape[0] != H or buf.shape[1] != W:
                buf = np.ascontiguousarray(np.pad(buf, ((0, max(0, H - buf.shape[0])), (0, max(0, W - buf.shape[1])),
                                                         (0, 0)))[:H, :W])
            proc.stdin.write(buf.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0:
            files.append("cartpole.mp4")
    k = int(np.searchsorted(t[idx], min(t_catch + 0.25, t[-1]) if math.isfinite(t_catch) else t[-1] / 2))
    draw(idx[min(k, len(idx) - 1)])
    fig.savefig(os.path.join(out_dir, "cartpole_poster.png"), facecolor=BG)
    files.append("cartpole_poster.png")
    plt.close(fig)
    return {"files": files, "frames": int(len(idx)) if ff else 0, "unstable_pole": r["unstable_pole"],
            "doubling_time": r["doubling_time"], "free_fall_time_to_90deg": r["free_fall_time_to_90deg"],
            "catch_time": t_catch, "push_peak_deg": r["push_peak_deg"], "push_recovery": r["push_recovery"]}
