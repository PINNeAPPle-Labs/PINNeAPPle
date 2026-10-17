"""Figures and animations for electromagnetic fields: field maps with Poynting arrows, wave-propagation movies
(GIF with Pillow, MP4 with ffmpeg) and radiation patterns.

Fields are 2-D arrays indexed ``[ix, iy]`` (the layout of :mod:`pinneapple_simulation.electromagnetics.fdtd`);
``extent = (x0, x1, y0, y1)`` in metres. ``outline`` is a boolean or fractional material map drawn as a contour at
half filling, so the geometry stays visible on every frame.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = ["field_map", "field_movie", "radiation_pattern"]


def _limits(data, symmetric: bool, clip: float):
    a = np.asarray(data, float)
    if symmetric:
        v = float(np.quantile(np.abs(a), clip)) or 1.0
        return -v, v
    return float(np.quantile(a, 1 - clip)), float(np.quantile(a, clip))


def field_map(field, extent: Sequence[float], path=None, *, outline=None, vectors=None, title: str = "",
              label: str = "", cmap: str = "RdBu_r", symmetric: bool = True, clip: float = 0.995, stride: int = 0,
              ax=None):
    """Draw one 2-D field (``[ix, iy]``) with an optional material outline and a vector field
    ``vectors = (vx, vy)`` (e.g. the time-averaged Poynting vector) as arrows every ``stride`` nodes. Returns the
    matplotlib figure; saves it when ``path`` is given."""
    import matplotlib.pyplot as plt

    f = np.asarray(field, float)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 5.2 * (extent[3] - extent[2]) / max(extent[1] - extent[0], 1e-300)
                                        if extent[1] > extent[0] else 5.2))
    else:
        fig = ax.figure
    lo, hi = _limits(f, symmetric, clip)
    im = ax.imshow(f.T, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
    if outline is not None:
        o = np.asarray(outline, float)
        xs = np.linspace(extent[0], extent[1], o.shape[0])
        ys = np.linspace(extent[2], extent[3], o.shape[1])
        ax.contour(xs, ys, o.T, levels=[0.5], colors="k", linewidths=1.0)
    if vectors is not None:
        vx, vy = (np.asarray(v, float) for v in vectors)
        st = stride or max(1, f.shape[0] // 24)
        xs = np.linspace(extent[0], extent[1], f.shape[0])[::st]
        ys = np.linspace(extent[2], extent[3], f.shape[1])[::st]
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        ax.quiver(X, Y, vx[::st, ::st], vy[::st, ::st], color="k", alpha=0.7, pivot="mid")
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    if path is not None and own:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def field_movie(frames, extent: Sequence[float], path, *, times=None, outline=None, title: str = "",
                label: str = "", cmap: str = "RdBu_r", fps: int = 20, clip: float = 0.995, dpi: int = 90) -> Path:
    """Animate a list of 2-D fields with a fixed symmetric colour scale (set from all frames) and the geometry
    outline; ``.gif`` is written with Pillow, ``.mp4`` with ffmpeg."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as mplanim
    import matplotlib.pyplot as plt

    frames = [np.asarray(f, float) for f in frames]
    lo, hi = _limits(np.stack(frames), True, clip)
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * (extent[3] - extent[2]) / (extent[1] - extent[0]) + 0.6))
    im = ax.imshow(frames[0].T, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
    if outline is not None:
        o = np.asarray(outline, float)
        ax.contour(np.linspace(extent[0], extent[1], o.shape[0]), np.linspace(extent[2], extent[3], o.shape[1]),
                   o.T, levels=[0.5], colors="k", linewidths=1.0)
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    txt = ax.set_title(title)

    def update(k):
        im.set_data(frames[k].T)
        if times is not None:
            txt.set_text(f"{title}  t = {times[k] * 1e9:.3f} ns" if times[k] < 1e-6 else f"{title}  t = {times[k]:.3g} s")
        return (im, txt)

    anim = mplanim.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = "ffmpeg" if path.suffix.lower() == ".mp4" else "pillow"
    anim.save(str(path), writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return path


def radiation_pattern(theta, values, path=None, *, reference=None, labels: Sequence[str] = ("computed", "analytic"),
                      db: bool = True, floor_db: float = -40.0, title: str = ""):
    """Polar plot of a far-field pattern (normalised to its maximum), optionally against a reference; ``db``
    plots ``20 log10 |F|`` clipped at ``floor_db``."""
    import matplotlib.pyplot as plt

    th = np.asarray(theta, float)

    def norm(v):
        v = np.abs(np.asarray(v))
        v = v / v.max()
        return np.maximum(20 * np.log10(np.maximum(v, 1e-12)), floor_db) - floor_db if db else v

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(th, norm(values), lw=2, label=labels[0])
    if reference is not None:
        ax.plot(th, norm(reference), "k--", lw=1.2, label=labels[1])
    if db:
        ticks = np.arange(0, -floor_db + 1, 10)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t + floor_db:.0f} dB" for t in ticks], fontsize=7)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
