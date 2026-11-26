"""Physics checks per domain, for models that were trained somewhere else: does the prediction still obey the
structure the domain demands, where no truth is available?

A pre-trained model is accurate on its training distribution and quiet outside it; the checks here ask the three
questions a domain asks of any field, and answer them with a number that needs no reference:

* **weather** — :func:`advection_residual`: a tracer that is advected satisfies ``dc/dt + u.grad c = 0``, so the part
  of its 6-hour change that the wind cannot explain is a measure of how far the prediction has left the dynamics.
  :func:`global_drift`: the area-weighted integral of a conserved total (dry static energy ``cp T + g z``) must not
  wander: its drift per day, in units of the field's own spatial spread, is a model-independent number that a
  physical forecast keeps near the one the analysis has.
* **flood masks** — :func:`hand` (height above nearest drainage) and :func:`flood_mask_score`: water on a cell that
  the terrain does not drain into is impossible; a mask that wets terrain far above the water surface, or that wets a
  downstream cell while leaving its upstream donors dry, is inadmissible whatever the model.
* **video** — :func:`temporal_residual`: between two frames of a physical scene, intensity moves with the scene
  (brightness constancy), so the part of the frame-to-frame change that no displacement within ``max_speed`` can
  explain is the flicker, the reshuffle or the teleporting content.

Every check is a :class:`~pinneapple_physics.physics_audit.monitor.ValidityCheck` (value, tolerance, the evidence
string of the monitor) and :func:`as_prediction_check` turns it into the ``(score, reason)`` callable that
``PhysicsEnsemble(prediction_checks=...)`` tilts the weights with.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .physics_audit.monitor import ValidityCheck

__all__ = ["EARTH_RADIUS", "sphere_gradient", "advection_residual", "advection_check",
           "global_drift", "dry_static_energy", "energy_check",
           "flow_directions", "hand", "flood_mask_score", "hand_check",
           "lucas_kanade", "temporal_residual", "temporal_continuity_check",
           "as_prediction_check"]

EARTH_RADIUS = 6.371e6                      # m
CP_AIR = 1004.0                             # J kg-1 K-1


# ----------------------------------------------------------------------------------------------- weather

def _lat_axis(lat: np.ndarray) -> tuple[np.ndarray, float]:
    la = np.deg2rad(np.asarray(lat, float))
    return la, float(np.sign(lat[0] - lat[-1]))         # +1 when the first row is the north (index grows southward)


def sphere_gradient(field: np.ndarray, lat: Sequence[float], lon: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Eastward and northward gradient on a latitude-longitude grid in 1/m: central differences, periodic in
    longitude, one-sided at the first and last row; the metric ``R cos(lat)`` divides the eastward difference, so a
    gradient near a pole is not inflated by the converging meridians."""
    f = np.asarray(field, float)
    la, sign = _lat_axis(np.asarray(lat, float))
    dlam = np.deg2rad(float(np.abs(np.diff(np.asarray(lon, float))).mean()))
    dphi = np.deg2rad(float(np.abs(np.diff(np.asarray(lat, float))).mean()))
    dx = EARTH_RADIUS * np.cos(la) * dlam
    dfdx = (np.roll(f, -1, -1) - np.roll(f, 1, -1)) / (2 * dx)[..., None]
    dfdphi = np.gradient(f, axis=-2) / (EARTH_RADIUS * dphi)        # per index step, northward positive
    return dfdx, sign * dfdphi


def advection_residual(field: np.ndarray, u: np.ndarray, v: np.ndarray, hours: float,
                       lat: Sequence[float], lon: Sequence[float], weights: np.ndarray | None = None) -> float:
    """Relative part of the time change of ``field`` that the advection by ``(u, v)`` does not explain.

    ``field``: ``(steps + 1, lat, lon)``; ``u``, ``v``: the advecting wind in m/s, ``(steps + 1, lat, lon)`` or
    ``(lat, lon)`` (one analysis wind for the whole window). ``hours`` is the interval between two steps. The value is
    ``||dc/dt + u.grad c|| / max(||dc/dt||, ||u.grad c||)`` with the area weights ``weights`` (cosine of latitude by
    default): 0 for a field that is exactly advected, ~1 for a change with no advective structure -- or for a tracer
    that sits still under a wind that should carry it, which is the same failure from the other side -- and it needs
    no reference field. The denominator is the larger of the two terms so that a field which does not change at all is
    judged against the advection that should have moved it, instead of dividing by zero.
    """
    c = np.asarray(field, float)
    if c.ndim != 3 or c.shape[0] < 2:
        raise ValueError("field must be (steps + 1, lat, lon) with at least two steps")
    dt = float(hours) * 3600.0
    w = np.cos(_lat_axis(np.asarray(lat, float))[0]) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    u = np.broadcast_to(u[None] if u.ndim == 2 else u, c.shape)
    v = np.broadcast_to(v[None] if v.ndim == 2 else v, c.shape)
    num = den_t = den_a = 0.0
    for k in range(1, c.shape[0]):
        tend = (c[k] - c[k - 1]) / dt
        gx, gy = sphere_gradient(c[k], lat, lon)
        gx0, gy0 = sphere_gradient(c[k - 1], lat, lon)
        gx, gy = 0.5 * (gx + gx0), 0.5 * (gy + gy0)                 # gradient at the middle of the interval
        adv = 0.5 * (u[k] + u[k - 1]) * gx + 0.5 * (v[k] + v[k - 1]) * gy
        res = tend + adv
        num += float(np.sum(res ** 2 * w[:, None]))
        den_t += float(np.sum(tend ** 2 * w[:, None]))
        den_a += float(np.sum(adv ** 2 * w[:, None]))
    den = max(den_t, den_a)
    return 0.0 if den <= 0.0 else float(np.sqrt(num / den))


def advection_check(wind_of: Callable[[Any], tuple], tolerance: float, lat: Sequence[float], lon: Sequence[float],
                    hours: float = 6.0, select: Callable[[np.ndarray, Any], np.ndarray] | None = None,
                    weights: np.ndarray | None = None, name: str = "advection_consistency") -> ValidityCheck:
    """``wind_of(query) -> (u, v)`` is the wind that should advect the prediction of that query (the analysis wind at
    the initial time is legitimate: it is known before the forecast). ``select(prediction, query)`` picks the field
    out of the prediction when it is not one already."""
    def fn(p: np.ndarray, q: Any) -> float:
        x = np.asarray(select(p, q) if select is not None else p, float)
        u, v = wind_of(q)
        return advection_residual(x, u, v, hours, lat, lon, weights)
    return ValidityCheck(name, "residual", fn, tolerance, False,
                         lambda val: f"{100 * val:.3g} % of the change unexplained by advection "
                                     f"(tolerance {100 * tolerance:.3g} %)")


def dry_static_energy(temperature: np.ndarray, geopotential: np.ndarray) -> np.ndarray:
    """``cp T + g z`` (:math:`\\Phi = g z` given as geopotential): the total energy of a dry parcel that a forecast
    must not create or destroy globally."""
    return CP_AIR * np.asarray(temperature, float) + np.asarray(geopotential, float)


def global_drift(field: np.ndarray, lat: Sequence[float], hours: float = 6.0,
                 weights: np.ndarray | None = None) -> float:
    """Change per day of the area-weighted mean of ``field``, in units of its own initial spatial spread
    (area-weighted standard deviation about that mean). A conserved total gives ~0; a forecast that creates energy or
    mass shows it here without any reference, and the scale-free value lets a temperature and a geopotential be
    compared."""
    x = np.asarray(field, float)
    if x.ndim != 3 or x.shape[0] < 2:
        raise ValueError("field must be (steps + 1, lat, lon) with at least two steps")
    w = np.cos(_lat_axis(np.asarray(lat, float))[0]) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    gm = (x.mean(axis=-1) * w[None, :]).sum(axis=-1)               # (steps + 1,) global area-weighted mean
    spread = float(np.sqrt(np.sum(w[:, None] * (x[0] - gm[0]) ** 2)))
    per_day = (float(gm[-1] - gm[0]) / (float(hours) * 3600.0 * (x.shape[0] - 1))) * 86400.0
    return float(abs(per_day) / max(spread, 1e-12))


def energy_check(tolerance: float, lat: Sequence[float], hours: float = 6.0,
                 field_fn: Callable[[np.ndarray, Any], np.ndarray] | None = None, weights: np.ndarray | None = None,
                 name: str = "energy_drift") -> ValidityCheck:
    """Drift of the global total of the prediction (:func:`global_drift`); ``field_fn`` builds the conserved total out
    of the prediction and the query (``dry_static_energy`` of two channels, for instance)."""
    def fn(p: np.ndarray, q: Any) -> float:
        x = np.asarray(field_fn(p, q) if field_fn is not None else p, float)
        return global_drift(x, lat, hours, weights)
    return ValidityCheck(name, "conservation", fn, tolerance, False,
                         lambda val: f"{val:.3g} spreads per day (tolerance {tolerance:.3g})")


# ----------------------------------------------------------------------------------------------- flood masks

def flow_directions(terrain: np.ndarray) -> np.ndarray:
    """Index of the steepest lower 8-neighbour of every cell, flattened; ``-1`` where the cell is a pit (no lower
    neighbour). Edges are not connected across the boundary (the grid is a patch of terrain, not a sphere)."""
    z = np.asarray(terrain, float)
    ny, nx = z.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    best = np.full(z.shape, -1, dtype=np.int64)
    drop = np.zeros(z.shape)
    for dy, dx, dist in ((0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
                         (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))):
        src_z = z[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
        dst_z = z[max(0, dy):ny - max(0, -dy), max(0, dx):nx - max(0, -dx)]
        dst_i = idx[max(0, dy):ny - max(0, -dy), max(0, dx):nx - max(0, -dx)]
        slope = (src_z - dst_z) / dist
        here = (slice(max(0, -dy), ny - max(0, dy)), slice(max(0, -dx), nx - max(0, dx)))
        take = slope > drop[here]
        drop[here] = np.where(take, slope, drop[here])
        best[here] = np.where(take, dst_i, best[here])
    best[drop <= 0] = -1
    return best.ravel()


def hand(terrain: np.ndarray, drainage: np.ndarray) -> np.ndarray:
    """Height above nearest drainage (m): for every cell, the elevation difference along the descending path
    (steepest descent, :func:`flow_directions`) to the first cell of the drainage network it reaches. ``NaN`` where the
    path ends in a pit that is not on the network: a filled DEM and a connected stream network reach everywhere, and a
    cell that reaches nothing is reported rather than given a made-up height."""
    z = np.asarray(terrain, float)
    dr = np.asarray(drainage, bool).ravel()
    nxt = flow_directions(z)
    flat = z.ravel()
    out = np.where(dr, 0.0, np.nan)
    for start in range(flat.size):
        if np.isfinite(out[start]):
            continue
        path, seen, i = [], set(), start
        base = -1
        while True:                                  # walk downstream to the network or to a cell already resolved
            if i < 0 or i in seen:
                break
            if np.isfinite(out[i]):
                base = i
                break
            seen.add(i)
            path.append(i)
            i = int(nxt[i])
        if base >= 0:                                # every cell on the path is that height above the same outlet
            for j in path:
                out[j] = flat[j] - flat[base] + out[base]
    return out.reshape(z.shape)


def flood_mask_score(mask: np.ndarray, terrain: np.ndarray, drainage: np.ndarray, depth: float,
                     check_upstream: bool = True) -> float:
    """Admissibility of a flood mask in [0, 1] (1 = admissible), from two violations of water-on-terrain:

    * **above the water** — a wet cell whose height above the nearest drainage exceeds ``depth``: the water would have
      to sit above the terrain that drains away;
    * **dry upstream of wet** — a dry cell that drains directly into a wet cell and lies below the water surface: a
      disconnected mask, water without the cells that feed it.

    Normalised by the number of wet cells (a mask with none is admissible and uninformative: 1)."""
    m = np.asarray(mask, bool)
    z = np.asarray(terrain, float)
    h = hand(z, drainage)
    wet = int(m.sum())
    if wet == 0:
        return 1.0
    bad = np.zeros(z.shape, bool)
    bad |= m & (np.nan_to_num(h, nan=depth + 1.0) > depth)
    if check_upstream:
        nxt = flow_directions(z).reshape(z.shape)
        for cell in np.argwhere(m).tolist():                       # a wet neighbour's donors must be wet as well
            y, x = cell
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < z.shape[0] and 0 <= xx < z.shape[1] and nxt[yy, xx] == y * z.shape[1] + x:
                    bad[yy, xx] |= (not m[yy, xx]) and (np.nan_to_num(h[yy, xx], nan=depth + 1.0) < depth)
    return float(1.0 - min(bad.sum() / wet, 1.0))


def hand_check(terrain: np.ndarray, drainage: np.ndarray, depth: float, min_score: float = 1.0,
               select: Callable[[np.ndarray, Any], np.ndarray] | None = None,
               name: str = "water_on_terrain") -> ValidityCheck:
    """``flood_mask_score`` as a validity check: the value is ``1 - score`` (tolerance ``1 - min_score``), so it reads
    like the other checks — 0 is a mask the terrain admits."""
    def fn(p: np.ndarray, q: Any) -> float:
        m = np.asarray(select(p, q) if select is not None else p) > 0.5
        return 1.0 - flood_mask_score(m, terrain, drainage, depth)
    return ValidityCheck(name, "constitutive", fn, 1.0 - min_score, False,
                         lambda val: f"{100 * val:.3g} % of the wet cells inadmissible "
                                     f"(tolerance {100 * (1 - min_score):.3g} %)")


# ----------------------------------------------------------------------------------------------- video

def _box_sum(x: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)^2 window, periodic in both axes (a frame is a scene, not a tile)."""
    y = np.cumsum(np.pad(x, ((0, 0), (r + 1, r), (0, 0)), mode="wrap"), axis=-2)
    y = y[..., 2 * r + 1:, :] - y[..., :-2 * r - 1, :]
    z = np.cumsum(np.pad(y, ((0, 0), (0, 0), (r + 1, r)), mode="wrap"), axis=-1)
    return z[..., 2 * r + 1:] - z[..., :-2 * r - 1]


def lucas_kanade(frames: np.ndarray, window: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense one-step displacement between consecutive frames (Lucas-Kanade with a ``window`` half-size box filter):
    ``(u, v, residual)`` where ``u`` is along the last axis, ``v`` along the first, both in pixels per frame, and
    ``residual`` is ``|dI/dt + u dI/dx + v dI/dy|`` — the part of the change that the motion does not explain. The
    gradients are averaged over the two frames of each pair, the discretisation that keeps the linearisation honest
    for displacements of a pixel or two (the difference against the single-frame gradient is large: on a smooth
    texture a one-pixel translation gives a residual of 0.16 here and 0.43 there)."""
    f = np.asarray(frames, float)
    if f.ndim != 3 or f.shape[0] < 2:
        raise ValueError("frames must be (n_frames, height, width) with at least two frames")
    r = int(window)
    dx = 0.5 * (np.roll(f, -1, -1) - np.roll(f, 1, -1))
    dy = 0.5 * (np.roll(f, -1, -2) - np.roll(f, 1, -2))
    dt = np.diff(f, axis=0)
    ax, ay, at = 0.5 * (dx[:-1] + dx[1:]), 0.5 * (dy[:-1] + dy[1:]), dt
    sxx = _box_sum(ax * ax, r) + 1e-9
    syy = _box_sum(ay * ay, r) + 1e-9
    sxy = _box_sum(ax * ay, r)
    sxt = _box_sum(ax * at, r)
    syt = _box_sum(ay * at, r)
    det = sxx * syy - sxy * sxy
    good = np.abs(det) > 1e-6
    u = np.where(good, -(syy * sxt - sxy * syt) / np.where(good, det, 1.0), 0.0)
    v = np.where(good, -(sxx * syt - sxy * sxt) / np.where(good, det, 1.0), 0.0)
    resid = np.abs(at + u * ax + v * ay)
    return u, v, resid


def temporal_residual(frames: np.ndarray, max_speed: float | None = None, window: int = 3) -> dict[str, float]:
    """How much of a video's frame-to-frame change the motion cannot explain, plus the two things a physical sequence
    does not do: move faster than ``max_speed`` (pixels per frame) and change its flow from one frame to the next.

    ``flow_residual``: ``||dI/dt + u.grad I|| / ||dI/dt||`` over the frames (0 for a scene that moves rigidly, 1 when
    no displacement explains the change — a flicker, a gain change, shuffled frames); ``teleport_fraction``: fraction
    of pixels whose estimated speed is above ``max_speed``; ``jerk``: mean change of the flow between frames divided
    by the mean speed (a scene that moves smoothly keeps it near 0, a reshuffled one does not)."""
    u, v, resid = lucas_kanade(frames, window)
    f = np.asarray(frames, float)
    change = float(np.sqrt(np.mean(np.diff(f, axis=0) ** 2)))
    speed = np.hypot(u, v)
    out = {"flow_residual": float(np.sqrt(np.mean(resid ** 2)) / max(change, 1e-12)),
           "mean_speed": float(speed.mean()),
           "teleport_fraction": float(np.mean(speed > max_speed)) if max_speed is not None else 0.0}
    if u.shape[0] > 1:
        du = np.hypot(*(np.diff(w, axis=0) for w in (u, v)))
        out["jerk"] = float(du.mean() / max(speed.mean(), 1e-6))
    else:
        out["jerk"] = 0.0
    return out


def temporal_continuity_check(tolerance: float = 0.25, max_speed: float | None = None, window: int = 3,
                              name: str = "temporal_continuity") -> ValidityCheck:
    """``max(flow_residual, teleport_fraction)`` against ``tolerance``: the value is 0 for a sequence whose change is a
    motion within ``max_speed`` and 1 when nothing about the change is motion."""
    def fn(p: np.ndarray, q: Any) -> float:
        r = temporal_residual(p, max_speed, window)
        return float(max(r["flow_residual"], r["teleport_fraction"]))
    return ValidityCheck(name, "constitutive", fn, tolerance, False,
                         lambda val: f"{100 * val:.3g} % of the change unexplained by motion "
                                     f"(tolerance {100 * tolerance:.3g} %)")


# ----------------------------------------------------------------------------------------------- to the ensemble

def as_prediction_check(check: ValidityCheck, strength: float = 1.0) -> Callable[[np.ndarray, Any], tuple]:
    """The ``(score, reason)`` callable ``PhysicsEnsemble(prediction_checks=...)`` expects, from a validity check:
    a score of 1 inside the tolerance, falling linearly to 0 at ``strength`` times the tolerance (so a check that is
    twice over its limit takes the expert out of the case, and one slightly over only tilts its weight)."""
    def fn(prediction: np.ndarray, query: Any) -> tuple[float, str]:
        res = check(prediction, query)
        if not np.isfinite(res.value):
            return 0.0, res.detail
        score = 1.0 - res.value / max(check.tolerance * strength, 1e-300)
        return float(np.clip(score, 0.0, 1.0)), f"{check.name}: {res.detail}"
    return fn
