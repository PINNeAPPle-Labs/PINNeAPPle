"""Point features of an AirfRANS coordinate network, and the physics residuals of the constrained variant.

Every feature is scaled per case — by the chord and by the freestream magnitude — so the same weights transfer across
cases and the network sees O(1) numbers, with the wall-normal coordinate kept as a separate logarithmic channel because
the boundary layer is a few percent of the chord thick.
"""
from __future__ import annotations

import numpy as np

__all__ = ["WALL_FRACTION", "SHAPE_BINS", "SHAPE_FEATURES", "chord", "shape_descriptor", "point_features",
           "scaled_targets", "sample_indices", "FEATURE_NAMES"]

#: share of each training sample taken from the wall and the boundary-layer band
WALL_FRACTION = 0.25

#: bins of the chord the airfoil's upper and lower surfaces are sampled on for the global shape descriptor
SHAPE_BINS = 32

#: length of :func:`shape_descriptor` — the per-case channels appended when a model is shape-conditioned
SHAPE_FEATURES = 2 * SHAPE_BINS

#: the input channels, in the order :func:`point_features` returns them
FEATURE_NAMES = ("x_over_chord", "y_over_chord", "sdf_over_chord", "log_sdf", "normal_x", "normal_y", "surface",
                 "freestream_x", "freestream_y", "log_inlet_velocity", "bias")


def chord(airfoil_points: np.ndarray) -> float:
    """Chord length of the case, from the extent of its airfoil point cloud."""
    span = np.ptp(np.asarray(airfoil_points, float), axis=0)
    return float(max(span[0], 1e-9))


def shape_descriptor(airfoil_points: np.ndarray, bins: int = SHAPE_BINS) -> np.ndarray:
    """The airfoil's upper and lower surfaces on a uniform chord grid, in chords — the case's global shape.

    A coordinate network sees one point at a time, and every airfoil of the family passes through the same region of
    the plane: without this descriptor two cases are indistinguishable at the same ``(x, y, n)``, while their wall
    shear differs by a factor of several. Thirty-two samples an edge are enough to separate the family, and they cost
    nothing at inference — the airfoil's own points travel with every case."""
    points = np.asarray(airfoil_points, float)
    c = chord(points)
    x, y = points[:, 0] / c, points[:, 1] / c
    index = np.clip(np.rint(x * (bins - 1)).astype(int), 0, bins - 1)
    upper, lower = np.full(bins, np.nan), np.full(bins, np.nan)
    for i in range(bins):
        mask = index == i
        if mask.any():
            upper[i], lower[i] = y[mask].max(), y[mask].min()
    empty = np.isnan(upper)
    if empty.any() and not empty.all():
        filled = np.flatnonzero(~empty)
        upper[empty] = np.interp(np.flatnonzero(empty), filled, upper[filled])
        lower[empty] = np.interp(np.flatnonzero(empty), filled, lower[filled])
    return np.nan_to_num(np.concatenate([upper, lower]), nan=0.0).astype(np.float32)


def point_features(features: np.ndarray, airfoil_points: np.ndarray, shape: bool = False) -> np.ndarray:
    """The eleven input channels from the AirfRANS per-point features ``(x, y, sdf, n_x, n_y, surface, ux, uy)``.

    The signed distance enters twice: divided by the chord, and as a signed logarithm, which is what lets a network
    resolve the first few percent of the chord where the wall gradients live."""
    features = np.asarray(features, float)
    x, y, sdf, nx, ny, surface, ux, uy = (features[:, i] for i in range(8))
    c = chord(airfoil_points)
    u_inf = np.hypot(ux, uy)
    u_inf = np.where(u_inf > 0, u_inf, 1.0)
    scaled = sdf / c
    block = np.stack([x / c, y / c, scaled, np.log1p(np.abs(scaled)) * np.sign(scaled), nx, ny, surface,
                      ux / u_inf, uy / u_inf, np.log(u_inf), np.ones_like(x)], axis=1).astype(np.float32)
    if shape:
        block = np.concatenate([block, np.tile(shape_descriptor(airfoil_points), (len(block), 1))], axis=1)
    return block


def scaled_targets(targets: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Velocity in units of the freestream and kinematic pressure in units of its dynamic head."""
    features = np.asarray(features, float)
    u_inf = np.hypot(features[:, 6], features[:, 7])
    u_inf = np.where(u_inf > 0, u_inf, 1.0)[:, None]
    out = np.asarray(targets, float).copy()
    out[:, :2] /= u_inf
    out[:, 2] /= u_inf[:, 0] ** 2
    return out.astype(np.float32)


def sample_indices(features: np.ndarray, per_case: int, rng: np.random.Generator,
                   wall_fraction: float = WALL_FRACTION, band: float = 0.02) -> np.ndarray:
    """Indices for one case's training sample, oversampling the wall and the boundary-layer band.

    The airfoil curve is under one percent of the nodes and every wall-derived quantity is read from it, so a uniform
    sample would leave the region that decides the force metrics almost unseen."""
    features = np.asarray(features, float)
    n, sdf, surface = len(features), features[:, 2], features[:, 5]
    wall = np.flatnonzero(surface > 0.5)
    near_wall = np.flatnonzero((surface <= 0.5) & (np.abs(sdf) < band))
    take_wall = int(per_case * wall_fraction) if wall.size else 0
    take_band = int(per_case * wall_fraction) if near_wall.size else 0
    parts = []
    if take_wall:
        parts.append(rng.choice(wall, size=min(take_wall, wall.size), replace=False))
    if take_band:
        parts.append(rng.choice(near_wall, size=min(take_band, near_wall.size), replace=False))
    parts.append(rng.choice(n, size=min(max(per_case - take_wall - take_band, 1), n), replace=False))
    return np.sort(np.unique(np.concatenate(parts)))
