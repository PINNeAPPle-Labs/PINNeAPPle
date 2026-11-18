"""Liner wear: the impulse wear model, and fitting its coefficient to a measured (laser-scanned) profile.

A mill liner loses thickness where the charge keeps hitting it, and DEM is used for exactly one reason there: to
turn a liner profile measured after a campaign into a wear law that predicts the next one.  The model here is the
impulse form of Archard's wear law, which is what DEM wear studies use when the liner is discretised into patches
(Cleary 1998, *Predicting charge motion, power draw, segregation and wear in ball mills using discrete element
methods*, Minerals Engineering 11:1061-1080):

    depth_patch = k_w * I_patch / (H * A_patch)

with ``I`` the normal impulse the patch received over the run [N s], ``H`` the liner hardness [Pa] and ``A`` the
patch area [m^2].  ``k_w`` is a dimensionless wear coefficient; it is the one number a measured profile is used to
calibrate, and :func:`fit_wear_law` does that by least squares through the origin and reports the ``R^2`` of the fit
so a poor calibration is visible rather than assumed.

The impulse form assumes the mean sliding speed at the contact is comparable between the calibration campaign and
the prediction -- true for a mill run at the same speed and fill, and the reason the fit is reported per speed band
rather than once for a whole circuit.

    >>> from pinneapple_simulation.granular.wear import fit_wear_law
    >>> import numpy as np
    >>> impulse = np.array([1.0, 2.0, 3.0]); areas = np.full(3, 0.01); hardness = 2.0e9
    >>> depth = 3.0e-9 * impulse / (hardness * areas)
    >>> round(fit_wear_law(impulse, areas, depth, hardness)["k_wear"], 9)
    3e-09
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["WearTracker", "fit_wear_law", "patch_areas", "linear_wear_profile"]


def patch_areas(wall) -> np.ndarray:
    """Area of every wall plane [m^2] (the barrel facets of a :class:`Wall`, which set their own areas)."""
    areas = getattr(wall, "areas", None)
    if areas is None:
        raise ValueError("this wall carries no plane areas: build it with Wall.faceted_cylinder, or pass areas "
                         "explicitly to WearTracker")
    return np.asarray(areas, dtype=float)


@dataclass
class WearTracker:
    """Accumulates the impulse each patch sees and converts it to a wear depth."""

    wall_index: int = 0
    hardness: float = 2.0e9                       # liner hardness [Pa] (a cast steel liner is ~2-4 GPa)
    k_wear: float = 1.0e-9                         # wear coefficient, calibrated per liner material
    areas: np.ndarray | None = None
    impulse: np.ndarray = field(default_factory=lambda: np.zeros(0))
    include_obstacles: bool = True                 # lifters wear too; they are reported as extra patches

    def reset(self) -> None:
        self.impulse = np.zeros(0)

    def update(self, world) -> np.ndarray:
        """Add the impulse of the last step (already cumulative in the world) and return the per-patch depth."""
        wall = world.walls[self.wall_index]
        base = world._plane_offsets[self.wall_index]
        n_planes = len(wall.offsets)
        impulse = np.array(world.plane_impulse[base:base + n_planes], dtype=float)
        if self.include_obstacles:
            extra = [world.obstacle_impulse[k] for k, (w, _) in enumerate(world.obstacle_index)
                     if w == self.wall_index]
            if extra:
                impulse = np.r_[impulse, np.asarray(extra, dtype=float)]
        self.impulse = impulse
        return self.depths(world)

    def areas_of(self, world) -> np.ndarray:
        if self.areas is not None:
            return np.asarray(self.areas, dtype=float)
        wall = world.walls[self.wall_index]
        areas = patch_areas(wall)
        if self.include_obstacles:
            extra = [float(o.shape.area) for o in wall.obstacles]
            if extra:
                areas = np.r_[areas, np.asarray(extra, dtype=float)]
        return areas

    def depths(self, world=None) -> np.ndarray:
        if world is None:
            if self.areas is None:
                raise ValueError("WearTracker.depths needs either the world (to read the patch areas) or areas=")
            areas = np.asarray(self.areas, dtype=float)
        else:
            areas = self.areas_of(world)
        return self.k_wear * self.impulse / (self.hardness * np.maximum(areas, 1.0e-30))


def fit_wear_law(impulse: np.ndarray, areas: np.ndarray, depth: np.ndarray,
                 hardness: float) -> dict:
    """Least-squares wear coefficient through the origin, with the fit quality.

    ``x = I / (H A)`` has units of metres, so ``k_wear = sum(x y) / sum(x^2)`` is dimensionless; ``r2`` is computed
    on the same one-parameter model and a value well below one says the scanned profile is not explained by the
    impact distribution alone (a different liner material, a changed speed, or a patch too coarse to resolve).
    """
    impulse = np.asarray(impulse, dtype=float)
    areas = np.asarray(areas, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if not (len(impulse) == len(areas) == len(depth)):
        raise ValueError("impulse, areas and depth must have the same length (one entry per patch)")
    x = impulse / (hardness * np.maximum(areas, 1.0e-30))
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        return {"k_wear": 0.0, "r2": 0.0, "patches": int(len(x))}
    k = float(np.sum(x * depth) / denom)
    residual = float(np.sum((depth - k * x) ** 2))
    total = float(np.sum(depth ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else 0.0
    return {"k_wear": k, "r2": float(r2), "patches": int(len(x)),
            "max_residual": float(np.max(np.abs(depth - k * x))) if len(x) else 0.0}


def linear_wear_profile(areas: np.ndarray, rate: np.ndarray, hours: float) -> np.ndarray:
    """A synthetic ``target`` liner profile from per-patch wear rates [m/h] -- the stand-in for a laser scan."""
    return np.asarray(rate, dtype=float) * float(hours)
