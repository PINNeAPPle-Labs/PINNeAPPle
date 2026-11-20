"""The wall quantities AirfRANS scores and the benchmark derives from the field: slope, wall shear, forces.

The benchmark never reads a submitted wall shear or a submitted force: :mod:`airfrans` re-integrates both from the
velocity and pressure the participant writes, through ``Simulation.wallshearstress`` and ``Simulation.force``. That
makes two things worth knowing before designing a model:

* the wall shear is ``2ν S·n`` at the surface nodes, with ``S`` the symmetric part of the velocity gradient that
  VTK's point-based least squares reconstructs from the *whole* field around each wall node; and
* every force coefficient is that same reconstruction, integrated along the airfoil polyline.

Both are therefore linear in the near-wall velocity gradient, and the gradient is the one quantity a point-wise
coordinate network has to get right on a mesh whose first cell sits at ``y⁺ ≈ 0.2``. This module owns the truth side
(computed once per case with the pinned stack and cached as a compact ``.npz``) and the differentiable reconstruction
that lets a training loss see the same numbers the evaluator will.

Two conventions are fixed here and used everywhere downstream:

``slope``
    the wall-normal derivative of the tangential velocity, divided by the case's ``U∞/c`` and multiplied by ``-1``, so
    it is dimensionless and points along the flow at the wall. It equals ``-wallShearStress/ν`` — see
    :func:`slope_from_wall_shear`.

``quadrature``
    the airfoil polyline exactly as ``pyvista`` builds it: one segment per cell, cell values the mean of the two
    endpoint values, and the cell's own length and normal. :func:`curve_coefficients` reproduces
    ``Simulation.force_coefficient`` from predicted fields to the last few digits, which
    ``tests/test_airfrans_benchmark.py`` asserts.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

__all__ = ["WallTruthMissing", "wall_case_truth", "write_wall_store", "load_wall_case",
           "slope_from_wall_shear", "curve_coefficients", "surface_rank", "curve_projection"]


class WallTruthMissing(RuntimeError):
    """The optional AirfRANS/VTK stack is not installed: only the cache writer needs it."""


def _simulation():
    try:
        from airfrans.simulation import Simulation
    except ImportError as error:                      # pragma: no cover - depends on the machine
        raise WallTruthMissing(
            "the wall-truth cache needs the pinned participant stack (airfrans, pyvista, vtk); training and "
            f"prediction only need the cache. The import failed with: {error}") from error
    return Simulation


def slope_from_wall_shear(wall_shear: np.ndarray, nu: float, inlet_velocity: float, chord: float) -> np.ndarray:
    """The dimensionless wall slope the network predicts, from a wall-shear vector.

    A tangential near-wall profile ``u_t(d)`` gives ``2ν S·n = ν (∂u_t/∂d) t̂`` with ``d`` the distance *away* from the
    wall, so the slope vector is ``+wallShearStress/ν``, pointing along the flow: the skin friction the flow exerts on
    the surface. Dividing by ``U∞/c`` puts it in the network's own units (a velocity per unit of ``sdf/c``), which is
    what makes one weight set work across cases with different freestreams.

    The sign was measured, not assumed. Blending the true field into a linear near-wall layer with
    ``∂u_t/∂d = +τ/ν`` reproduces the evaluator's wall shear to 0.9 % and 1.7 % on two cases; the opposite sign gives
    200 %, because it points the near-wall velocity against the flow.
    """
    return np.asarray(wall_shear, float) * (1.0 / nu) * (chord / inlet_velocity)


def wall_case_truth(dataset_root: str, case_id: str) -> dict[str, Any]:
    """One case's scored wall truth: slope at every surface node, the airfoil quadrature, and the coefficients.

    The slope is stored in the *internal mesh* order of the surface nodes, so a training sample that keeps the native
    point order lines up with it index for index; the quadrature is stored in the airfoil's own order, which is the
    order ``Simulation.force`` integrates in."""
    Simulation = _simulation()
    sim = Simulation(root=dataset_root, name=case_id)
    surface = np.asarray(sim.surface, bool)
    airfoil_points = np.asarray(sim.airfoil_position, float)
    chord = float(max(np.ptp(airfoil_points[:, 0]), 1e-9))
    inlet = float(sim.inlet_velocity)
    nu = float(sim.NU)

    # over_airfoil=False keeps the native internal-mesh order (zeros off the surface): exactly what a point sample needs
    wall_shear_surface = np.asarray(sim.wallshearstress(over_airfoil=False, reference=True), float)[surface]
    slope = slope_from_wall_shear(wall_shear_surface, nu, inlet, chord)

    cells = np.asarray(sim.airfoil.lines, np.int64).reshape(-1, 3)[:, 1:]
    polydata = sim.airfoil.ptc(pass_point_data=False)
    lengths = np.asarray(polydata.cell_data["Length"], float).ravel()
    cell_normals = np.asarray(polydata.cell_data["Normals"], float)[:, :2]
    if len(lengths) != len(cells):
        raise RuntimeError(f"{case_id}: {len(lengths)} cells of lengths for {len(cells)} segments")

    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient(reference=True)
    return {
        "case": np.array([case_id]),
        "surface": np.flatnonzero(surface).astype(np.int64),
        "slope": slope.astype(np.float32),
        "cells": cells.astype(np.int32),
        "lengths": lengths.astype(np.float32),
        "cell_normals": cell_normals.astype(np.float32),
        "cd": np.float32(cd), "cdp": np.float32(cdp), "cdv": np.float32(cdv),
        "cl": np.float32(cl), "clp": np.float32(clp), "clv": np.float32(clv),
        "inlet_velocity": np.float32(inlet), "nu": np.float32(nu), "chord": np.float32(chord),
        "rho": np.float32(sim.RHO), "aoa_rad": np.float32(sim.angle_of_attack),
    }


def write_wall_store(dataset_root: str, case_ids: Iterable[str], out_dir: str, *, workers: int = 1,
                     progress=None) -> dict[str, Any]:
    """Cache :func:`wall_case_truth` for every case, skipping the ones already written.

    The stack is the slow part (about 1.7 s a case: a mesh read plus two gradient reconstructions), so the writer is
    resumable and parallel over cases."""
    case_ids = list(case_ids)
    os.makedirs(out_dir, exist_ok=True)
    pending = [c for c in case_ids if not os.path.exists(os.path.join(out_dir, f"{c}.npz"))]
    done, failures = 0, []
    if workers > 1 and pending:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(wall_case_truth, dataset_root, c): c for c in pending}
            for future, case in futures.items():
                try:
                    _save(out_dir, future.result())
                except Exception as error:            # reported, never skipped silently
                    failures.append({"case_id": case, "error": f"{type(error).__name__}: {error}"})
                done += 1
                if progress:
                    progress(done, len(pending))
    else:
        for case in pending:
            try:
                _save(out_dir, wall_case_truth(dataset_root, case))
            except Exception as error:
                failures.append({"case_id": case, "error": f"{type(error).__name__}: {error}"})
            done += 1
            if progress:
                progress(done, len(pending))
    record = {"requested": len(case_ids), "written": done - len(failures), "skipped": len(case_ids) - len(pending),
              "failures": failures[:10], "failed": len(failures)}
    with open(os.path.join(out_dir, "wall.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    return record


def _save(out_dir: str, arrays: dict[str, Any]) -> None:
    target = os.path.join(out_dir, f"{str(arrays['case'][0])}.npz")
    np.savez_compressed(target, **arrays)


def load_wall_case(wall_store: str, case_id: str) -> dict[str, Any]:
    """The cached wall truth of one case, with the scalars read back as floats."""
    with np.load(os.path.join(wall_store, f"{case_id}.npz")) as data:
        record = {name: np.array(data[name]) for name in data.files}
    for name in ("cd", "cdp", "cdv", "cl", "clp", "clv", "inlet_velocity", "nu", "chord", "rho", "aoa_rad"):
        record[name] = float(record[name])
    return record


def surface_rank(features: np.ndarray) -> np.ndarray:
    """For every node of the internal mesh, its position among the surface nodes, or ``-1`` elsewhere.

    The wall truth is stored compactly over the surface nodes; this is the index that maps a sampled node onto it."""
    surface = np.asarray(features, float)[:, 5] > 0.5
    rank = np.full(len(surface), -1, np.int64)
    rank[surface] = np.arange(int(surface.sum()))
    return rank


def curve_coefficients(pressure: np.ndarray, slope: np.ndarray, cells: np.ndarray, lengths: np.ndarray,
                       cell_normals: np.ndarray, *, cos_alpha: float, sin_alpha: float,
                       nu: float, inlet_velocity: float, chord: float) -> dict[str, float]:
    """The four force coefficients the evaluator would derive from a predicted curve pressure and wall slope.

    ``pressure`` is the kinematic pressure divided by ``U∞²`` and ``slope`` the slope of :func:`slope_from_wall_shear`,
    both at the airfoil's own points. Works on numpy and on torch tensors alike: the training loss calls it with
    tensors so the gradient flows back through the surface pressure and the slope head."""
    p_cell = 0.5 * (pressure[cells[:, 0]] + pressure[cells[:, 1]])
    normal_x = cell_normals[:, 0]
    # drag along (cos α, sin α); lift along (−sin α, cos α) — the basis row order of Simulation.force_coefficient.
    # force_p = ρ Σ p n L and the ρ cancels against the ½ρU∞² normalisation, leaving 2 Σ p n L / U∞².
    drag_dot_p = p_cell * (normal_x * cos_alpha + cell_normals[:, 1] * sin_alpha)
    lift_dot_p = p_cell * (-normal_x * sin_alpha + cell_normals[:, 1] * cos_alpha)
    # the viscous force is the wall-shear vector itself, +ν times the physical slope, integrated along the polyline
    drag_dot_v = slope[:, 0] * cos_alpha + slope[:, 1] * sin_alpha
    lift_dot_v = -slope[:, 0] * sin_alpha + slope[:, 1] * cos_alpha
    s_drag = 0.5 * (drag_dot_v[cells[:, 0]] + drag_dot_v[cells[:, 1]])
    s_lift = 0.5 * (lift_dot_v[cells[:, 0]] + lift_dot_v[cells[:, 1]])

    # both fields are already divided by their own scale (p by U∞², the slope by U∞/c), so only the ½ρU∞²
    # normalisation and the ν of the wall shear are left
    scale_p = 2.0
    scale_v = 2.0 * nu / (inlet_velocity * chord)
    cdp = (drag_dot_p * lengths).sum() * scale_p
    clp = (lift_dot_p * lengths).sum() * scale_p
    cdv = (s_drag * lengths).sum() * scale_v
    clv = (s_lift * lengths).sum() * scale_v
    if isinstance(cdp, (int, float, np.floating, np.integer)):
        return {"cdp": float(cdp), "clp": float(clp), "cdv": float(cdv), "clv": float(clv),
                "cd": float(cdp + cdv), "cl": float(clp + clv)}
    return {"cdp": cdp, "clp": clp, "cdv": cdv, "clv": clv, "cd": cdp + cdv, "cl": clp + clv}


def curve_features(airfoil_points: np.ndarray, airfoil_normals: np.ndarray, freestream: Sequence[float],
                   inlet_velocity: float, shape: bool = False, panel: bool = False,
                   panel_pressure: bool = True) -> np.ndarray:
    """The network's eleven input channels evaluated on the airfoil curve itself (``sdf = 0``, ``surface = 1``).

    ``panel`` adds the case's inviscid field on those same nodes, which is what a panel-conditioned model expects: the
    force term reads the curve through the same network as the field, so the two feature blocks have to describe the
    same channels or the force term cannot run at all."""
    from .features import point_features

    points = np.asarray(airfoil_points, float)
    normals = np.asarray(airfoil_normals, float)[:, :2]
    count = len(points)
    raw = np.concatenate([points, np.zeros((count, 1)), normals, np.ones((count, 1)),
                          np.tile(np.asarray(freestream, float).reshape(1, 2) / inlet_velocity, (count, 1))], axis=1)
    return point_features(raw, points, shape=shape, panel=panel, panel_pressure=panel_pressure)


def curve_projection(slope: np.ndarray, features: np.ndarray, bins: int = 33, smooth: bool = True) -> np.ndarray:
    """Replace every node's wall slope by the mean of its chord bin, on its own surface.

    The readout predicts a slope per node and gets the *shape* of the wall shear wrong — measured on the packaged
    entries, the per-case error is uniform, a least-squares scale does not remove it, and a field that is 2.5× worse
    near the wall moves the metric by a single point. What the truth looks like instead is a smooth curve along the
    chord: binning the true slope into 33 cells per surface reproduces it to 12 % (65 cells: 7.3 %), so two short curves
    per case carry more than the readout manages to predict. Projecting the prediction onto that class keeps its level
    and removes the node-to-node noise the evaluator's gradient reconstruction amplifies.

    ``features`` is a batch in the order of :data:`~.features.FEATURE_NAMES`: ``x/c`` (column 0) gives the position along
    the chord, the freestream unit vector (7:9) the chord frame, and the wall normals (4:6) the surface a node is on.
    """
    slope = np.asarray(slope, float)
    features = np.asarray(features, float)
    freestream = features[:, 7:9]
    normal_axis = np.stack([-freestream[:, 1], freestream[:, 0]], axis=1)
    side = np.sign(np.sum(features[:, 4:6] * normal_axis, axis=1))
    along = np.clip(features[:, 0], 0.0, 1.0 - 1e-9)
    index = (along * bins).astype(np.int64) + (side > 0).astype(np.int64) * bins

    # the curve is *estimated* on the wall nodes and then *evaluated* at every node: the slope only acts where the wall
    # layer's weight is non-zero, which is the boundary-layer band just off the surface, not the surface itself
    on_wall = features[:, 6] > 0.5
    total = np.zeros((2 * bins, 2))
    count = np.zeros(2 * bins)
    np.add.at(total, index[on_wall], slope[on_wall])
    np.add.at(count, index[on_wall], 1.0)
    curve = total / np.maximum(count, 1.0)[:, None]
    centres = (np.arange(bins) + 0.5) / bins
    projected = slope.copy()
    along = np.clip(features[:, 0], 0.0, 1.0)
    for side_index, side_mask in ((0, side <= 0), (1, side > 0)):
        rows = curve[side_index * bins:(side_index + 1) * bins]
        known = np.flatnonzero(count[side_index * bins:(side_index + 1) * bins] > 0)
        if known.size == 0:
            continue
        target = side_mask
        if smooth:
            # linear in the chord between bin centres: a step function at 33 bins puts a jump exactly where the
            # evaluator's least-squares gradient looks, which is why the piecewise-constant version is not the default
            for component in range(2):
                projected[target, component] = np.interp(along[target], centres[known], rows[known, component])
        else:
            outside = np.clip((along[target] * bins).astype(int), 0, bins - 1)
            projected[target] = rows[outside]
    return projected
