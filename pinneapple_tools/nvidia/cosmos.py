"""Cosmos physics-plausibility benchmark: where a generated video stops being physical.

A world foundation model (NVIDIA Cosmos, or any video model) generates a *future* of a scene. No one watching a
clip can tell when the generated frames left physics behind, so this module measures it: per-frame residuals of the
conservation laws the scene obeys, a limit calibrated on clips that *are* physical, and the first frame that breaks
it -- the **trust horizon of the generated future**, the same idea the trust layer applies to a forecast, with the
frame index as the lead.

The generated clips here are made, not downloaded (no network, no Cosmos weights on this machine): each one starts
from a *stated* physical scene, rendered to the same kind of grey frames a model would output, and the non-physical
ones apply one named violation at a known frame. That gives a **labelled** set, which is the only way to report a
detection rate and a horizon instead of an impression:

``clip_bouncing_ball``
    a rigid body under gravity, bounces with restitution ``e`` (the scene of
    :func:`pinneapple_worldmodel.physical_world.bouncing_ball_scene`, which also supplies the exact labels).
    Violations: ``"reverse"`` (the ball reverses horizontally without an impulse), ``"mass_gain"`` (it grows),
    ``"gravity_off"`` (acceleration stops after the violation frame), ``"teleport"`` (it jumps and comes back).
``clip_channel_flow``
    a tracer-seeded channel: incompressible flow, no force along the channel. ``widening=True`` gives a
    *decelerating* flow (``u = Q/h(x)`` with ``v`` chosen so that ``div u = 0``), which is physical and must pass.
    Violations: ``"mass_appears"`` (new tracers fade in), ``"momentum_gain"`` (the flow speeds up with no force),
    ``"compress"`` (the streamlines converge, so mass piles up).

Residuals, all measured from the pixels only:

===================  =========================================================================
``acceleration``     rigid: the fitted vertical acceleration against ``g``, on free-flight frames
``area``             rigid: the object's area against its median (mass appearing, growing, vanishing)
``impulse``          rigid: a change of velocity that gravity does not explain (a jump, a reversal)
``mass``             fluid: the photometric tracer mass against the first frame
``divergence``       fluid: ``mean|div u| L / U`` from Horn-Schunck optical flow (mass piling up)
``momentum``         fluid: ``mean|Du/Dt| L / U^2``, the material acceleration, where no force acts
===================  =========================================================================

Limits are calibrated on the clean clips of the same scene (``calibrate``: the highest residual a physical clip
reached, times a margin) -- never on the violated ones, which is what makes the detection rate a measurement.

``judge_video`` takes a decoded clip (an MP4 from Cosmos, through :func:`frames_from_video`) and returns the same
verdict, so the benchmark and a real generated clip are judged by one function.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["COSMOS_EXTRA", "Clip", "ClipVerdict", "BenchmarkReport", "optical_flow", "render_discs",
           "clip_bouncing_ball", "clip_channel_flow", "violations", "rigid_residuals", "fluid_residuals",
           "residuals", "calibrate", "judge", "labelled_set", "benchmark", "frames_from_video", "judge_video"]

COSMOS_EXTRA = 'pip install "pinneapple[cosmos]"'
G = 9.81
CHECKS = ("acceleration", "area", "impulse", "mass", "divergence", "momentum")


# ---------------------------------------------------------------------------------------------- clips
@dataclass
class Clip:
    """A clip as frames plus what it is: the stated scene, the violation it carries (``None`` = physical) and the
    first frame at which the violation is visible (``at``, the label the detection is scored against)."""
    name: str
    frames: np.ndarray                  # (T, H, W) float in [0, 1]
    dt: float                           # s between frames
    kind: str                           # "rigid" | "fluid"
    scene: str = ""
    violation: str | None = None
    at: int | None = None
    px_per_m: float | None = None
    scored: tuple[str, ...] = ()        # the checks this scene's law makes meaningful
    labels: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.frames))

    def physical(self) -> bool:
        return self.violation is None


def render_discs(cx, cy, r, H: int, W: int, *, bg: float = 0.22, fg: float = 0.88, gain: float = 1.0,
                 texture: float = 0.03, blur: float = 0.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Frames ``(T, H, W)`` with one antialiased disk per frame at ``(cx, cy)`` with radius ``r`` (pixels), on a
    textured background with optional blur and sensor noise: the same rendering a generated clip is judged in."""
    from scipy import ndimage
    cx, cy, r = np.broadcast_arrays(np.asarray(cx, float), np.asarray(cy, float), np.asarray(r, float))
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    base = bg + texture * (0.5 * np.sin(xx / 5.0 + 1.3) * np.cos(yy / 6.5))
    frames = []
    for x, y, rad in zip(cx.ravel(), cy.ravel(), r.ravel(), strict=True):
        d = np.hypot(xx - x, yy - y)
        disk = np.clip(rad + 0.5 - d, 0.0, 1.0)
        img = base * (1.0 - disk) + fg * disk
        img = gain * img
        if blur > 0:
            k = np.exp(-0.5 * (np.arange(-3, 4) / blur) ** 2)
            k /= k.sum()
            img = ndimage.convolve1d(ndimage.convolve1d(img, k, axis=0, mode="constant"), k, axis=1, mode="constant")
        if noise > 0:
            img = img + rng.normal(0, noise, img.shape)
        frames.append(np.clip(img, 0, 1))
    return np.asarray(frames, np.float32)


def clip_bouncing_ball(*, seed: int = 0, T: int = 48, fps: float = 24.0, H: int = 64, W: int = 96,
                       violation: str | None = None, at: int | None = None) -> Clip:
    """A rigid body under gravity with bounces, and optionally one labelled violation applied from frame ``at``.

    The physical clip comes from :func:`pinneapple_worldmodel.physical_world.bouncing_ball_scene` (exact event
    trajectory, randomized camera and texture, exact labels). A violation edits the *labelled state* -- position,
    radius -- and renders it again with the same camera and texture, so the pixels differ only in the physics.
    """
    from pinneapple_worldmodel.physical_world import bouncing_ball_scene
    v = bouncing_ball_scene(seed=seed, T=T, fps=fps, H=H, W=W)
    lab = v.labels
    cx = np.asarray(lab["cx_px"], float).copy()
    cy = np.asarray(lab["cy_px"], float).copy()
    r = np.full(len(cx), float(lab["radius_m"]) * v.px_per_m)
    at = at if at is not None else max(3, len(cx) // 2)
    if violation is not None:
        if not 2 < at < len(cx) - 2:
            raise ValueError(f"violation frame {at} outside the clip (2..{len(cx) - 3})")
        if violation == "reverse":                       # horizontal velocity flips at `at`: needs an impulse
            cx[at:] = cx[at] - (cx[at:] - cx[at])
        elif violation == "mass_gain":                   # the object grows: mass from nowhere
            r[at:] = r[at] * (1.0 + 0.06 * np.arange(1, len(cx) - at + 1))
        elif violation == "gravity_off":                 # acceleration stops: gravity switched off
            v0 = cy[at] - cy[at - 1]
            cy[at:] = cy[at] + v0 * np.arange(0, len(cx) - at)
        elif violation == "teleport":                    # a jump and back: two impulses, no force
            cy[at:at + 4] = cy[at:at + 4] - 0.25 * v.px_per_m
        else:
            raise ValueError(f"unknown violation {violation!r} for a rigid clip "
                             f"(have {violations('rigid')})")
    px = dict(v.params)
    frames = render_discs(cx, cy, r, H, W, bg=float(px["bg"]), fg=float(px["fg"]), gain=float(px["gain"]),
                          texture=float(px["texture"]), blur=float(px["blur"]), noise=float(px["noise"]),
                          seed=seed + 1000)
    return Clip(name=f"ball{seed}" + (f"-{violation}@{at}" if violation else ""), frames=frames, dt=v.dt,
                kind="rigid", scene=f"bouncing_ball(seed={seed}, e={lab['restitution']:.2f})",
                violation=violation, at=None if violation is None else at, px_per_m=v.px_per_m,
                scored=("acceleration", "area", "impulse"),
                labels={"cx_px": cx, "cy_px": cy, "r_px": r, "g": float(lab["g"]),
                        "radius_m": float(lab["radius_m"]), "restitution": float(lab["restitution"])})


def _channel_state(T: int, H: int, W: int, *, widening: bool, seed: int, n_x: int = 9, n_y: int = 5,
                   u0: float = 1.2, beta: float = 1.0) -> dict[str, np.ndarray]:
    """Tracer positions of a channel flow, one array per frame.

    ``h(x) = h0 (1 + beta x / W)`` is the streamtube height, ``u(x) = u0 h0 / h(x)`` the streamwise velocity (so the
    discharge ``u h`` is constant and the flow *decelerates* when the channel widens) and ``v = u h'/h (y - y0)``
    closes ``div u = 0`` exactly: mass is conserved by construction, and with no force along the channel the material
    acceleration is zero in the uniform case.
    """
    rng = np.random.default_rng(seed)
    y0, h0 = 0.18 * H, 0.62 * H
    xs = np.linspace(0.04 * W, 0.46 * W, n_x)
    X, Yf = [], []
    for x in xs:
        for _ in range(n_y):
            X.append(x + rng.uniform(-0.012, 0.012) * W)
            Yf.append(rng.uniform(0.15, 0.85))                       # fraction of the local height, from the bed
    x = np.asarray(X, float)
    f = np.asarray(Yf, float)
    P = np.zeros((T, len(x), 2))
    for k in range(T):
        h = h0 * (1.0 + beta * x / W) if widening else np.full_like(x, h0)
        u = u0 * h0 / h
        v = (u * (beta * h0 / W if widening else 0.0) / h) * (f * h)
        P[k, :, 0] = x
        P[k, :, 1] = y0 + f * h
        x = x + u * (1.0 / 24.0) * 24.0 * 0.0 + u                 # one frame of advection (dt folded into u0)
        f = f + (v / h)                                            # the fraction of the local height is carried along
        f = np.clip(f, 0.05, 0.95)
    return {"x": P[:, :, 0], "y": P[:, :, 1], "y0": np.array(y0), "h0": np.array(h0)}


def clip_channel_flow(*, seed: int = 0, T: int = 48, fps: float = 24.0, H: int = 64, W: int = 96,
                      widening: bool = False, violation: str | None = None, at: int | None = None) -> Clip:
    """A tracer-seeded channel flow (physical), and optionally one labelled violation from frame ``at``."""
    st = _channel_state(T, H, W, widening=widening, seed=seed)
    x, y = st["x"].copy(), st["y"].copy()
    at = at if at is not None else max(3, T // 2)
    extra = np.zeros((T, 0, 2))
    if violation == "mass_appears":                    # new tracers fade in: mass from nowhere
        src = np.random.default_rng(seed + 7)
        n_new = 14
        xe = src.uniform(0.15 * W, 0.85 * W, n_new)
        ye = src.uniform(0.18 * H + 0.2 * 0.62 * H, 0.18 * H + 0.8 * 0.62 * H, n_new)
        extra = np.zeros((T, n_new, 2))
        extra[:, :, 0] = np.clip(xe[:, None] + 1.1 * np.arange(T)[None, :], 0, W - 1).T
        extra[:, :, 1] = np.repeat(ye[None, :], T, axis=0)
        extra = extra[at - at:]                        # all frames; the fade is applied when rendering
    elif violation == "momentum_gain":                 # the flow speeds up with no force behind it
        x = st["x"].copy()
        gain = 1.0 + 0.03 * np.arange(T)
        x[:, :] = st["x"][0][None, :] + (st["x"] - st["x"][0][None, :]) * gain[:, None]
        x[:at] = st["x"][:at]
    elif violation == "compress":                      # streamlines converge: mass piles up
        x = st["x"].copy()
        ramp = np.clip((np.arange(T) - at) / 3.0, 0.0, None)
        x[:, :] = st["x"][0][None, :] + (st["x"] - st["x"][0][None, :]) * (1.0 - 0.55 * ramp)[:, None] ** 0
        x = st["x"] * (1.0 - 0.055 * ramp)[:, None]
        x[:at] = st["x"][:at]
        y = st["y"] * (1.0 - 0.030 * ramp)[:, None]
        y[:at] = st["y"][:at]
    elif violation is not None:
        raise ValueError(f"unknown violation {violation!r} for a fluid clip (have {violations('fluid')})")
    r = 2.2
    frames = _render_tracers(x, y, extra, r, H, W, fade_from=None if violation != "mass_appears" else at, seed=seed)
    return Clip(name=f"channel{'-wide' if widening else ''}{-(seed and 100) +  0 if False else ''}"
                     + (f"-{violation}@{at}" if violation else ""),
                frames=frames, dt=1.0 / fps, kind="fluid",
                scene=f"channel_flow(widening={widening})", violation=violation,
                at=None if violation else None if violation is None else at,
                scored=("mass", "divergence") if widening else ("mass", "divergence", "momentum"),
                labels={"u0_px": 1.2, "widening": widening, "n_tracers": x.shape[1] + (14 if violation == "mass_appears" else 0)})


def _render_tracers(x, y, extra, r: float, H: int, W: int, *, fade_from: int | None, seed: int) -> np.ndarray:
    """Tracer frames: bright disks on a dark background (the mass a generator would have to keep constant)."""
    T = x.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    rng = np.random.default_rng(seed + 11)
    base = 0.06 + 0.02 * (0.5 * np.sin(xx / 5.0) * np.cos(yy / 6.5))
    frames = np.zeros((T, H, W), np.float32)
    for k in range(T):
        img = base.copy()
        for cx, cy in zip(x[k], y[k], strict=True):
            img = np.maximum(img, 0.90 * np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0))
        if extra.shape[1]:
            w = 1.0 if fade_from is None else float(np.clip((k - fade_from + 1) / 4.0, 0.0, 1.0))
            for cx, cy in zip(extra[k, :, 0], extra[k, :, 1], strict=True):
                img = np.maximum(img, w * 0.90 * np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0))
        frames[k] = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    return frames


def violations(kind: str) -> tuple[str, ...]:
    """The named violations available for a scene kind (``"rigid"`` or ``"fluid"``)."""
    if kind == "rigid":
        return ("reverse", "mass_gain", "gravity_off", "teleport")
    if kind == "fluid":
        return ("mass_appears", "momentum_gain", "compress")
    raise ValueError(f"unknown scene kind {kind!r} (have 'rigid', 'fluid')")


# ---------------------------------------------------------------------------------------------- residuals
def optical_flow(f0, f1, *, alpha: float = 1.0, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Horn-Schunck dense optical flow ``(u, v)`` in pixels per frame between two frames.

    The classic variational estimator (Horn & Schunck, 1981): ``u, v`` minimise the brightness-constancy residual
    plus ``alpha^2 |grad u|^2``, solved by the Jacobi iteration on the 4-neighbour mean. It is not the best flow
    available, but it needs nothing outside NumPy/SciPy and it is the estimator this benchmark's numbers were
    measured with.
    """
    from scipy import ndimage
    a, b = np.asarray(f0, float), np.asarray(f1, float)
    Iy, Ix = np.gradient(a)
    It = b - a
    k = alpha ** 2 + Ix ** 2 + Iy ** 2
    u = np.zeros_like(a)
    v = np.zeros_like(a)
    for _ in range(int(iters)):
        ubar = ndimage.uniform_filter(u, 3, mode="nearest")
        vbar = ndimage.uniform_filter(v, 3, mode="nearest")
        u = ubar - Ix * (Ix * ubar + Iy * vbar + It) / k
        v = vbar - Iy * (Ix * ubar + Iy * vbar + It) / k
    return u, -v                                    # -v: image rows grow downwards, the physical y grows upwards


def _length_scale(frames) -> float:
    """The scale a flow residual is normalised by: 1 % of the frame width, so the numbers are dimensionless."""
    return 0.1 * float(np.asarray(frames).shape[-1])


def rigid_residuals(clip: Clip) -> dict[str, np.ndarray]:
    """Per-frame residuals of a rigid-body clip, from the tracked object only.

    ``acceleration``: ``|a_y + g| / g`` on free-flight frames (the frames within one of a floor contact are left out
    -- the bounce is not free flight); ``area``: ``|A - median A| / median A``; ``impulse``: the frame-to-frame
    velocity change that gravity does not explain, ``|dv - g dt| / v_ref`` with ``v_ref`` the median speed. Frames
    where the object is not tracked are ``nan`` (and counted by ``index``).
    """
    from pinneapple_worldmodel.physical_world import track_object
    frames = np.asarray(clip.frames, float)
    tr = track_object(frames)
    ppm = float(clip.px_per_m or 1.0)
    T = len(frames)
    x = tr["cx"] / ppm
    y = -tr["cy"] / ppm
    a = tr["area"]
    ok = np.isfinite(x) & np.isfinite(y)
    dt = clip.dt
    # split at the floor contacts: local minima of the height, whose neighbours are not free flight
    contact = np.zeros(T, bool)
    for i in range(1, T - 1):
        if ok[i - 1:i + 2].all() and y[i] < y[i - 1] and y[i] <= y[i + 1]:
            contact[max(i - 1, 0):min(i + 2, T)] = True
    acc = np.full(T, np.nan)
    for i in range(1, T - 1):
        if ok[i - 1:i + 2].all() and not contact[i]:
            acc[i] = abs((y[i + 1] - 2 * y[i] + y[i - 1]) / dt ** 2 + G) / G
    med_a = float(np.nanmedian(a[ok])) if ok.any() else 1.0
    area = np.abs(a - med_a) / max(med_a, 1e-9)
    vx = np.full(T, np.nan)
    vy = np.full(T, np.nan)
    vx[1:] = np.diff(x) / dt
    vy[1:] = np.diff(y) / dt
    v_ref = float(np.nanmedian(np.hypot(vx, vy))) or 1.0
    impulse = np.zeros(T)
    dvx = np.full(T, np.nan)
    dvy = np.full(T, np.nan)
    dvx[1:] = np.diff(vx)
    dvy[1:] = np.diff(vy)
    impulse = np.hypot(dvx, dvy + G * dt) / v_ref
    impulse[~np.isfinite(impulse)] = 0.0
    return {"acceleration": acc, "area": area, "impulse": impulse}


def fluid_residuals(clip: Clip) -> dict[str, np.ndarray]:
    """Per-frame residuals of a fluid clip, from the frames and their Horn-Schunck optical flow.

    ``mass``: ``|M_k - M_0| / M_0`` with ``M`` the photometric tracer mass (the brightness above the background,
    summed over the frame); ``divergence``: ``mean|div u| L / U`` (mass piling up or thinning out);
    ``momentum``: ``mean|Du/Dt| L / U^2``, the material acceleration, which is zero where no force acts.
    """
    F = np.asarray(clip.frames, float)
    T = len(F)
    L = _length_scale(F)
    bg = np.median(F, axis=0)
    mass = np.array([float(np.clip(F[k] - bg, 0, None).sum()) for k in range(T)])
    mass = np.abs(mass - mass[0]) / max(mass[0], 1e-9)
    div = np.zeros(T)
    mom = np.zeros(T)
    prev = None
    for k in range(T - 1):
        u, v = optical_flow(F[k], F[k + 1])
        U = float(np.sqrt(np.mean(u ** 2 + v ** 2))) or 1e-9
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        div[k + 1] = float(np.mean(np.abs(du_dx + dv_dy))) * L / U
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        conv = u * du_dx + v * du_dy                 # u . grad u, streamwise component
        convv = u * dv_dx + v * dv_dy
        if prev is not None:
            up, vp = prev
            acc = np.hypot(u - up, v - vp) + np.hypot(conv, convv)
        else:
            acc = np.hypot(conv, convv)
        mom[k + 1] = float(np.mean(acc)) * L / (U ** 2)
        prev = (u, v)
    return {"mass": mass, "divergence": div, "momentum": mom}


def residuals(clip: Clip) -> dict[str, np.ndarray]:
    """The residuals of a clip's law: :func:`rigid_residuals` or :func:`fluid_residuals`, by ``clip.kind``."""
    if clip.kind == "rigid":
        out = rigid_residuals(clip)
    elif clip.kind == "fluid":
        out = fluid_residuals(clip)
    else:
        raise ValueError(f"unknown clip kind {clip.kind!r}")
    return {k: v for k, v in out.items() if not clip.scored or k in clip.scored}


# ---------------------------------------------------------------------------------------------- scoring
@dataclass
class ClipVerdict:
    """What the residuals say about one clip: the first frame that breaks a calibrated limit (the trust horizon of
    the generated future), which check broke it, and -- on a violated clip -- how far that is from the label."""
    clip: Clip
    per_frame: dict[str, np.ndarray]
    limits: dict[str, float]
    crossed: dict[str, int | None] = field(default_factory=dict)

    @property
    def first_flagged(self) -> int | None:
        hits = [k for k in self.crossed.values() if k is not None]
        return min(hits) if hits else None

    @property
    def flagged_by(self) -> str | None:
        k = self.first_flagged
        if k is None:
            return None
        for name, when in self.crossed.items():
            if when == k:
                return name
        return None

    @property
    def detected(self) -> bool:
        return self.first_flagged is not None

    @property
    def horizon_frames(self) -> int | None:
        """Frames of the generated future that the calibrated limits certify as physical."""
        return self.first_flagged

    @property
    def horizon_s(self) -> float | None:
        k = self.first_flagged
        return None if k is None else float(k * self.clip.dt)

    @property
    def lag_frames(self) -> int | None:
        """``first_flagged - at``: 0 = flagged on the first violating frame, negative = flagged late."""
        if self.clip.at is None or self.first_flagged is None:
            return None
        return int(self.first_flagged - self.clip.at)

    @property
    def certified_frames(self) -> int | None:
        """How much of a *physical* clip was certified (``None`` for a violated clip)."""
        return self.first_flagged if self.clip.physical() else None

    def to_dict(self) -> dict[str, Any]:
        return {"clip": self.clip.name, "scene": self.clip.scene, "kind": self.clip.kind,
                "violation": self.clip.violation, "at": self.clip.at,
                "first_flagged": self.first_flagged, "flagged_by": self.flagged_by,
                "horizon_frames": self.horizon_frames, "horizon_s": self.horizon_s,
                "lag_frames": self.lag_frames,
                "peak": {k: float(np.nanmax(v)) if np.size(v) else 0.0 for k, v in self.per_frame.items()}}


def calibrate(clean: Sequence[ClipVerdict] | Sequence[Clip], *, quantile: float = 0.99,
              margin: float = 1.5, floor: float = 0.05) -> dict[str, float]:
    """One limit per check from clips that are physical: the ``quantile`` of every residual they reached, times
    ``margin``, never below ``floor``. Calibrating on the clean clips and scoring the violated ones is what makes
    the detection rate a measurement rather than a threshold fitted to the answer."""
    vals: dict[str, list[float]] = {}
    for item in clean:
        per_frame = item.per_frame if isinstance(item, ClipVerdict) else residuals(item)
        for name, arr in per_frame.items():
            a = np.asarray(arr, float)
            a = a[np.isfinite(a)]
            if a.size:
                vals.setdefault(name, []).append(float(np.quantile(a, quantile)))
    return {name: max(floor, margin * max(v)) for name, v in vals.items()}


def judge(clip: Clip, limits: dict[str, float]) -> ClipVerdict:
    """Score one clip against calibrated ``limits``: the first frame any check goes over its limit."""
    per_frame = residuals(clip)
    crossed: dict[str, int | None] = {}
    for name, arr in per_frame.items():
        lim = limits.get(name)
        hits = np.nonzero(np.asarray(arr, float) > lim)[0] if lim is not None else []
        crossed[name] = int(hits[0]) if len(hits) else None
    return ClipVerdict(clip, per_frame, dict(limits), crossed)


@dataclass
class BenchmarkReport:
    """The labelled set judged: detection rate, lag, horizons, false alarms, and the table the docs repeat."""
    verdicts: list[ClipVerdict]
    limits: dict[str, float]
    calibration: list[str] = field(default_factory=list)

    def of(self, kind: str | None = None) -> list[ClipVerdict]:
        return [v for v in self.verdicts if kind is None or v.clip.kind == kind]

    @property
    def violated(self) -> list[ClipVerdict]:
        return [v for v in self.verdicts if not v.clip.physical()]

    @property
    def clean(self) -> list[ClipVerdict]:
        return [v for v in self.verdicts if v.clip.physical()]

    def detection_rate(self) -> float:
        """Fraction of the violated clips in which a calibrated limit was broken at all."""
        v = self.violated
        return float(np.mean([x.detected for x in v])) if v else float("nan")

    def false_alarm_rate(self) -> float:
        """Fraction of the *physical* clips that were flagged anyway (the cost of the limits being tight)."""
        c = self.clean
        return float(np.mean([x.detected for x in c])) if c else float("nan")

    def lags(self) -> list[int]:
        return [x.lag_frames for x in self.violated if x.lag_frames is not None]

    def attribution_rate(self) -> float:
        """Fraction of the detected violated clips where the check that fired is the one the violation breaks."""
        want = {"reverse": "impulse", "teleport": "impulse", "mass_gain": "area", "gravity_off": "acceleration",
                "mass_appears": "mass", "momentum_gain": "momentum", "compress": "divergence"}
        got = [(want.get(x.clip.violation or "", ""), x.flagged_by) for x in self.violated if x.detected]
        return float(np.mean([w == g for w, g in got])) if got else float("nan")

    def table(self) -> list[dict[str, Any]]:
        return [v.to_dict() for v in self.verdicts]

    def summary(self) -> dict[str, Any]:
        lags = self.lags()
        cert = [v.certified_frames for v in self.clean if v.certified_frames is not None]
        return {"clips": len(self.verdicts), "violated": len(self.violated), "physical": len(self.clean),
                "detection_rate": self.detection_rate(), "false_alarm_rate": self.false_alarm_rate(),
                "attribution_rate": self.attribution_rate(),
                "lag_frames": {"min": min(lags) if lags else None, "median": float(np.median(lags)) if lags else None,
                               "max": max(lags) if lags else None},
                "certified_frames_of_physical_clips": cert,
                "limits": dict(self.limits)}

    def horizon(self, *, unit: str = "frames") -> Any:
        """The benchmark as a :class:`pinneapple_analysis.trust.record.Horizon`: the median frame at which the
        violated clips stopped being physical is the useful lead, the clip length the no-skill one."""
        from pinneapple_analysis.trust.record import Horizon, TrustValue
        src = "pinneapple_tools.nvidia.cosmos.benchmark"
        hs = [v.horizon_frames for v in self.violated if v.horizon_frames is not None]
        clip_len = max((v.clip.n for v in self.verdicts), default=0)
        useful = (TrustValue.measured(float(np.median(hs)), source=src, unit=unit,
                                      method=f"first frame any calibrated residual exceeded its limit, over "
                                             f"{len(hs)} labelled clips",
                                      reference=float(clip_len), reference_source="the length of the clips judged")
                  if hs else TrustValue.not_measured("no violated clip broke a calibrated limit", source=src))
        requested = (TrustValue.measured(float(clip_len), source=src, unit=unit, method="the clips judged")
                     if clip_len else TrustValue.not_measured("no clips were judged", source=src))
        no_skill = (TrustValue.measured(float(clip_len), source=src, unit=unit,
                                        method="the whole generated clip is the lead a consumer would ask for")
                    if clip_len else TrustValue.not_measured("no clips were judged", source=src))
        return Horizon(useful, no_skill, requested, unit=unit, source=src)


def labelled_set(*, seeds: Sequence[int] = (0, 1), T: int = 48, fps: float = 24.0) -> tuple[list[Clip], list[Clip]]:
    """``(clean, violated)``: the labelled clips the benchmark is scored on. Clean clips calibrate the limits and
    measure the false-alarm rate; violated clips carry one named violation at a known frame."""
    clean: list[Clip] = []
    bad: list[Clip] = []
    for s in seeds:
        clean.append(clip_bouncing_ball(seed=s, T=T, fps=fps))
        for k, kind in enumerate(violations("rigid")):
            bad.append(clip_bouncing_ball(seed=s, T=T, fps=fps, violation=kind, at=T // 2 + k))
    clean.append(clip_channel_flow(seed=0, T=T, fps=fps))
    clean.append(clip_channel_flow(seed=1, T=T, fps=fps))
    clean.append(clip_channel_flow(seed=0, T=T, fps=fps, widening=True))
    for s in (0, 1):
        for k, kind in enumerate(violations("fluid")):
            bad.append(clip_channel_flow(seed=s, T=T, fps=fps, violation=kind, at=T // 2 + k))
    bad.append(clip_channel_flow(seed=0, T=T, fps=fps, widening=True, violation="momentum_gain", at=T // 2))
    return clean, bad


def benchmark(*, seeds: Sequence[int] = (0, 1), T: int = 48, fps: float = 24.0,
              clips: tuple[Sequence[Clip], Sequence[Clip]] | None = None) -> BenchmarkReport:
    """Run the labelled set: calibrate the limits on the clean clips, then judge every clip."""
    clean, bad = clips if clips is not None else labelled_set(seeds=seeds, T=T, fps=fps)
    limits = calibrate(clean)
    verdicts = [judge(c, limits) for c in list(clean) + list(bad)]
    return BenchmarkReport(verdicts, limits, calibration=[c.name for c in clean])


# ---------------------------------------------------------------------------------------------- real clips
def frames_from_video(path: str, *, max_frames: int | None = None, fps: float | None = None) -> tuple[np.ndarray, float]:
    """Decode a video file (an MP4 out of Cosmos, a capture) into grey frames ``(T, H, W)`` and the frame period.

    Needs ``imageio`` (``pip install "pinneapple[cosmos]"``) with its ffmpeg plugin: the codecs are not part of the
    core install.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"decoding a generated clip needs imageio: {COSMOS_EXTRA} (imageio + imageio-ffmpeg)") from e
    rd = imageio.get_reader(os.fspath(path))
    meta = rd.get_meta_data() or {}
    frames = []
    for i, fr in enumerate(rd):
        if max_frames is not None and i >= max_frames:
            break
        a = np.asarray(fr, float)
        if a.ndim == 3:
            a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
        frames.append(a / 255.0)
    rd.close()
    dt = 1.0 / float(fps or meta.get("fps") or 24.0)
    return np.asarray(frames, np.float32), dt


def judge_video(path: str, *, kind: str = "rigid", px_per_m: float | None = None, dt: float | None = None,
                limits: dict[str, float] | None = None, calibration: Sequence[Clip] | None = None) -> ClipVerdict:
    """Judge a decoded clip by the same residuals: give ``limits`` or a set of physical ``calibration`` clips.

    A clip from a world model is judged exactly as a labelled one -- that is the point of calibrating on physical
    clips -- but its pixel scale (``px_per_m``) and frame period have to come with it, or the rigid residuals are
    not in metres. Without either the limits or calibration clips the verdict is refused, not guessed.
    """
    frames, dt2 = frames_from_video(path)
    clip = Clip(name=os.path.basename(path), frames=frames, dt=dt or dt2, kind=kind, scene="generated clip",
                px_per_m=px_per_m, scored=("acceleration", "area", "impulse") if kind == "rigid"
                else ("mass", "divergence", "momentum"))
    if limits is None:
        if not calibration:
            raise ValueError("judging a generated clip needs the limits calibrated on physical clips: pass "
                             "limits=calibrate(...) or calibration=[physical clips]")
        limits = calibrate(calibration)
    return judge(clip, limits)
