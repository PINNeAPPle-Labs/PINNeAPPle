"""Explicit 31/8 feature schemas for bracket DeepONet experiments.

This module implements a *reconstructed* feature convention. The original
author's exact preprocessing, family vocabulary and length/force scales were
not provided. It must not be called an exact reproduction.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

FAMILIES = ("L", "G", "U", "Z", "other_04", "other_05",
            "other_06", "other_07", "other_08", "other_09",
            "other_10", "other_11")
GEOMETRY_KEYS = ("width", "length", "height", "thickness",
                 "hole_diameter", "offset", "gusset_thickness",
                 "fillet_radius", "hole_spacing", "rib_count", "auxiliary")
BRANCH_FEATURES = (
    "force_x_scaled", "force_y_scaled", "force_z_scaled",
    "moment_x_scaled", "moment_y_scaled", "moment_z_scaled",
    "log_young_pa", "poisson",
) + tuple(f"family_{name}" for name in FAMILIES) + tuple(f"geometry_{k}_mm" for k in GEOMETRY_KEYS)
TRUNK_FEATURES = ("x_over_length", "y_over_length", "z_over_length",
                  "distance_to_fixed_m", "distance_to_loaded_m",
                  "distance_to_concave_edge_m",
                  "fixed_proximity", "loaded_proximity")
assert len(BRANCH_FEATURES) == 31 and len(TRUNK_FEATURES) == 8


def branch_features(family: str, geometry_mm: dict, young_pa: float,
                    poisson: float, force_n: np.ndarray, moment_nm: np.ndarray,
                    force_scale_n: float = 100., length_scale_m: float = 0.1) -> np.ndarray:
    if family not in FAMILIES or young_pa <= 0 or force_scale_n <= 0 or length_scale_m <= 0:
        raise ValueError("Invalid family or normalization reference")
    f = np.asarray(force_n, dtype=float).reshape(3) / force_scale_n
    m = np.asarray(moment_nm, dtype=float).reshape(3) / (force_scale_n * length_scale_m)
    family_hot = np.zeros(len(FAMILIES))
    family_hot[FAMILIES.index(family)] = 1
    geometry = np.array([float(geometry_mm.get(key, 0.)) for key in GEOMETRY_KEYS])
    features = np.concatenate((f, m, [np.log(young_pa), poisson], family_hot, geometry))
    if not np.all(np.isfinite(features)):
        raise ValueError("Nonfinite branch features")
    return features.astype(np.float32)


def trunk_features(points_m: np.ndarray, fixed_surface_points_m: np.ndarray,
                   loaded_surface_points_m: np.ndarray,
                   concave_edge_points_m: np.ndarray | None = None,
                   length_scale_m: float = 0.1, proximity_radius_m: float = 0.01) -> np.ndarray:
    points = np.asarray(points_m, dtype=float)
    fixed = np.asarray(fixed_surface_points_m, dtype=float)
    loaded = np.asarray(loaded_surface_points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [P,3]")
    if fixed.ndim != 2 or loaded.ndim != 2 or fixed.shape[1] != 3 or loaded.shape[1] != 3:
        raise ValueError("Surface points must be [N,3]")
    if not len(fixed) or not len(loaded) or length_scale_m <= 0 or proximity_radius_m <= 0:
        raise ValueError("Surface samples and scales must be nonempty and positive")
    df = cKDTree(fixed).query(points)[0]
    dl = cKDTree(loaded).query(points)[0]
    if concave_edge_points_m is None:
        raise ValueError("Concave-edge samples required; do not silently substitute zero")
    edge = np.asarray(concave_edge_points_m, dtype=float)
    if edge.ndim != 2 or edge.shape[1] != 3 or len(edge) == 0:
        raise ValueError("concave_edge_points_m must be nonempty [E,3]")
    de = cKDTree(edge).query(points)[0]
    r = proximity_radius_m
    features = np.column_stack((points / length_scale_m, df, dl, de,
                                r / (r + df), r / (r + dl)))
    if not np.all(np.isfinite(features)):
        raise ValueError("Nonfinite trunk features")
    return features.astype(np.float32)
