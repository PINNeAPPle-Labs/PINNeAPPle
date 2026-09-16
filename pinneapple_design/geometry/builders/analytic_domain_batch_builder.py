"""Analytic (mesh-free) domain-batch builders for axis-aligned canonical
shapes: boxes/rectangles and cylinders.

Why this exists alongside :class:`STLDomainBatchBuilder`
----------------------------------------------------------
``STLDomainBatchBuilder`` is the right tool when the real geometry is an
arbitrary mesh -- but it always re-centers the loaded mesh on its bounding
box's centroid (see its ``_normalize_trimesh``), which silently breaks any
``value_fn`` that assumes literal, un-shifted domain coordinates (e.g.
``channel_flow_3d``'s inlet profile computes ``y*(H-y)`` assuming
``y in [0, H]``, not a re-centered ``[-H/2, H/2]``). For presets whose
domain is *exactly* an axis-aligned box or a circular-cross-section
cylinder -- i.e. genuinely representable by ``spec.domain_bounds`` alone,
no mesh file needed -- sampling the box/cylinder analytically avoids that
pitfall entirely and needs no ``trimesh``/``rtree`` optional dependency.

This is still REAL geometry, not a mock: the sampled points are genuine
interior/boundary points of the literal solid the preset's own docstring
describes (a box of the stated dimensions, a cylinder of the stated
radius/length), and boundary condition targets are computed by calling
each ``ConditionSpec``'s own ``mask()``/``values()`` -- exactly the same
mechanism :class:`STLDomainBatchBuilder` uses -- so nothing about the
preset's declared physics is bypassed or approximated.

See ``pinneapple_physics/pde_environment/presets/tag_geometry.py`` for the
per-preset table of which faces each tag names (the actual, preset-by-
preset justification), and ``examples/pde_environment/07_tagged_presets_
real_geometry.py`` for end-to-end usage through ``solve_pde``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pinneapple_physics.pde_environment.spec import ProblemSpec

FaceSpec = Tuple[str, str]  # (coord_name, "min" | "max")


def _sample_free_columns(
    rng: np.random.Generator,
    coords: Sequence[str],
    bounds: Dict[str, Tuple[float, float]],
    n: int,
    fixed: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Sample ``n`` points over ``coords``, uniform within ``bounds`` for
    every coordinate except those pinned to an exact value via ``fixed``
    (used to place points exactly on a box face)."""
    fixed = fixed or {}
    cols = []
    for c in coords:
        if c in fixed:
            cols.append(np.full((n,), float(fixed[c]), dtype=np.float32))
        else:
            lo, hi = bounds[c]
            cols.append(rng.uniform(float(lo), float(hi), size=n).astype(np.float32))
    return np.stack(cols, axis=1)


def _apply_conditions(
    spec: ProblemSpec,
    Xb: np.ndarray,
    ctx: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build ``y_bc`` targets and per-condition boolean masks by calling
    every ``dirichlet``/``neumann``/``robin`` condition's own
    ``mask()``/``values()`` on the sampled boundary points ``Xb``. Works
    uniformly for ``selector_type="tag"`` (reads ``ctx["tag_masks"]``) and
    ``selector_type="callable"`` (self-selecting, e.g. an ``x==0`` face)
    conditions alike.

    Important, found by hand while validating the tag-geometry fixtures:
    ``compile_problem``'s loss_fn (``pinneapple_physics/pinn_solver/
    compiler/compile.py``) uses whatever is in ``batch["y_bc"]`` (sliced to
    the condition's own ``mask_<name>``) as the target for EVERY condition
    kind once an explicit ``y_bc`` is supplied in the batch -- it only
    falls back to calling ``cond.values()`` itself when ``y_bc`` is
    entirely absent (the ``solve_pde()`` auto-sampling path). This means
    Neumann/Robin conditions need their target written into ``y_bc`` too,
    not just Dirichlet ones -- ``STLDomainBatchBuilder._targets_from_
    conditions`` only fills ``y_bc`` for ``kind == "dirichlet"``, which is
    a latent gap that happens to never surface in the existing examples
    (``04_heat3d_stl_box.py`` has no Neumann tag condition at all;
    ``03_ns2d_channel_tags.py`` uses an all-zero ``y_bc`` array that
    happens to be correct only because its one Neumann target, dp/dn=0,
    genuinely is zero). Filling it for every kind here avoids silently
    feeding NaN targets into Neumann/Robin conditions with a nonzero or
    position-dependent target."""
    n_fields = len(spec.fields)
    y_bc = np.full((Xb.shape[0], n_fields), np.nan, dtype=np.float32)
    cond_masks: Dict[str, np.ndarray] = {}

    for cond in spec.conditions:
        if cond.kind not in ("dirichlet", "neumann", "robin"):
            continue
        m = np.asarray(cond.mask(Xb, ctx), dtype=bool)
        cond_masks[f"mask_{cond.name}"] = m
        if np.any(m):
            vals = cond.values(Xb[m], ctx)
            vals = np.asarray(vals, dtype=np.float32)
            if vals.ndim == 1:
                vals = vals[:, None]
            for j, fname in enumerate(cond.fields):
                if fname in spec.fields:
                    y_bc[m, list(spec.fields).index(fname)] = vals[:, j]

    return y_bc, cond_masks


def sample_box_tag_batch(
    spec: ProblemSpec,
    tag_faces: Dict[str, List[FaceSpec]],
    *,
    n_col: int = 20_000,
    n_bc_per_face: int = 3_000,
    seed: int = 7,
    user_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample a real box-domain batch for a ``ProblemSpec`` whose
    ``domain_bounds`` literally describe an axis-aligned box (or rectangle,
    or hyper-rectangle -- works for any number of coordinates).

    Parameters
    ----------
    tag_faces : dict tag_name -> list of (coord_name, "min"|"max")
        Which face(s) of the box each ``selector_type="tag"`` condition's
        tag refers to (union if more than one). Coordinates that never
        appear here (e.g. a time coordinate ``t``) are treated as "free":
        sampled uniformly over their own ``domain_bounds`` range for every
        interior AND boundary point, never used to define a face -- this
        is exactly what ``examples/pde_environment/03_ns2d_channel_tags.py``
        already does for ``t`` in ``ns_incompressible_2d``.

    Returns a dict with the same keys ``solve_pde`` accepts directly via
    ``**batch`` (``x_col`` is included too, for standalone
    ``compile_problem`` loss checks like ``examples/pde_environment/
    04_heat3d_stl_box.py`` does -- strip it before calling ``solve_pde``,
    which sample its own collocation points from ``spec.domain_bounds``).
    """
    rng = np.random.default_rng(seed)
    coords = list(spec.coords)
    bounds = spec.domain_bounds
    missing = [c for c in coords if c not in bounds]
    if missing:
        raise ValueError(
            f"sample_box_tag_batch: spec.domain_bounds is missing bounds for coordinate(s) "
            f"{missing} (spec={spec.name!r}). Every coordinate needs a (min, max) bound."
        )

    unique_faces = sorted({f for faces in tag_faces.values() for f in faces})
    for (c, side) in unique_faces:
        if c not in coords:
            raise ValueError(f"sample_box_tag_batch: face coordinate {c!r} is not in spec.coords={coords}")
        if side not in ("min", "max"):
            raise ValueError(f"sample_box_tag_batch: face side must be 'min' or 'max', got {side!r}")

    Xc = _sample_free_columns(rng, coords, bounds, n_col)

    face_X: Dict[FaceSpec, np.ndarray] = {}
    face_N: Dict[FaceSpec, np.ndarray] = {}
    for (c, side) in unique_faces:
        lo, hi = bounds[c]
        val = lo if side == "min" else hi
        X = _sample_free_columns(rng, coords, bounds, n_bc_per_face, fixed={c: val})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(c)] = -1.0 if side == "min" else 1.0
        face_X[(c, side)] = X
        face_N[(c, side)] = N

    Xb = np.concatenate([face_X[f] for f in unique_faces], axis=0)
    Nb = np.concatenate([face_N[f] for f in unique_faces], axis=0)

    offsets: Dict[FaceSpec, Tuple[int, int]] = {}
    off = 0
    for f in unique_faces:
        n = face_X[f].shape[0]
        offsets[f] = (off, off + n)
        off += n

    tag_masks: Dict[str, np.ndarray] = {}
    for tag, faces in tag_faces.items():
        m = np.zeros((Xb.shape[0],), dtype=bool)
        for f in faces:
            s, e = offsets[f]
            m[s:e] = True
        tag_masks[tag] = m

    ctx: Dict[str, Any] = dict(user_ctx or {})
    ctx.setdefault("tag_masks", {})
    ctx["tag_masks"].update(tag_masks)
    ctx.setdefault(
        "bounds",
        {
            "min": np.array([bounds[c][0] for c in coords], dtype=np.float32),
            "max": np.array([bounds[c][1] for c in coords], dtype=np.float32),
        },
    )
    ctx.setdefault("mesh_info", {"shape": "analytic_box", "n_faces": len(unique_faces)})

    y_bc, cond_masks = _apply_conditions(spec, Xb, ctx)

    batch: Dict[str, Any] = {
        "x_col": Xc,
        "x_bc": Xb,
        "y_bc": y_bc,
        "n_bc": Nb,
        "ctx": ctx,
        **cond_masks,
    }
    return batch


def sample_cylinder_tag_batch(
    spec: ProblemSpec,
    *,
    axis_coord: str,
    cross_coords: Tuple[str, str],
    radius: float,
    tag_faces: Dict[str, str],
    n_col: int = 20_000,
    n_bc_per_face: int = 3_000,
    seed: int = 7,
    user_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample a real cylinder-domain batch (circular cross-section,
    extruded along ``axis_coord``) -- e.g. ``pipe_flow_3d``.

    Parameters
    ----------
    axis_coord : the coordinate running along the cylinder's axis (its
        ``domain_bounds`` gives the pipe length).
    cross_coords : the two coordinates spanning the circular cross-section
        (their ``domain_bounds`` must be symmetric, ``(-radius, radius)``).
    radius : the pipe radius (must match ``cross_coords``' bounds).
    tag_faces : dict tag_name -> one of "inlet" (axis_coord == min),
        "outlet" (axis_coord == max), "wall" (lateral cylindrical surface
        r == radius). Any other coordinate present in ``spec.coords``
        (e.g. none, for ``pipe_flow_3d``) is treated as free/uniform, same
        as :func:`sample_box_tag_batch`.
    """
    rng = np.random.default_rng(seed)
    coords = list(spec.coords)
    bounds = spec.domain_bounds
    c0, c1 = cross_coords
    R = float(radius)

    def _sample_disk(n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Rejection sampling for a uniform disk of radius R.
        pts = np.zeros((0, 2), dtype=np.float32)
        while pts.shape[0] < n:
            cand = rng.uniform(-R, R, size=(max(n, 256), 2)).astype(np.float32)
            keep = cand[:, 0] ** 2 + cand[:, 1] ** 2 <= R ** 2
            pts = np.concatenate([pts, cand[keep]], axis=0)
        return pts[:n, 0], pts[:n, 1]

    def _sample_ring(n: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
        return R * np.cos(theta), R * np.sin(theta)

    # ---- interior collocation points: uniform along the axis, uniform
    # (rejection-sampled) within the disk cross-section, free elsewhere. ----
    ax_lo, ax_hi = bounds[axis_coord]
    x_ax = rng.uniform(float(ax_lo), float(ax_hi), size=n_col).astype(np.float32)
    y0, y1 = _sample_disk(n_col)
    free_coords = [c for c in coords if c not in (axis_coord, c0, c1)]
    Xc_cols = {axis_coord: x_ax, c0: y0, c1: y1}
    for c in free_coords:
        lo, hi = bounds[c]
        Xc_cols[c] = rng.uniform(float(lo), float(hi), size=n_col).astype(np.float32)
    Xc = np.stack([Xc_cols[c] for c in coords], axis=1)

    # ---- boundary faces: inlet disk, outlet disk, lateral wall ----
    face_kinds = set(tag_faces.values())
    face_X: Dict[str, np.ndarray] = {}
    face_N: Dict[str, np.ndarray] = {}

    def _free_others(n: int, overrides: Dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for c in coords:
            if c in overrides:
                cols.append(overrides[c])
            else:
                lo, hi = bounds[c]
                cols.append(rng.uniform(float(lo), float(hi), size=n).astype(np.float32))
        return np.stack(cols, axis=1)

    if "inlet" in face_kinds:
        y0, y1 = _sample_disk(n_bc_per_face)
        X = _free_others(n_bc_per_face, {axis_coord: np.full(n_bc_per_face, float(ax_lo), dtype=np.float32), c0: y0, c1: y1})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(axis_coord)] = -1.0
        face_X["inlet"], face_N["inlet"] = X, N

    if "outlet" in face_kinds:
        y0, y1 = _sample_disk(n_bc_per_face)
        X = _free_others(n_bc_per_face, {axis_coord: np.full(n_bc_per_face, float(ax_hi), dtype=np.float32), c0: y0, c1: y1})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(axis_coord)] = 1.0
        face_X["outlet"], face_N["outlet"] = X, N

    if "wall" in face_kinds:
        n = n_bc_per_face
        x_ax_w = rng.uniform(float(ax_lo), float(ax_hi), size=n).astype(np.float32)
        y0, y1 = _sample_ring(n)
        X = _free_others(n, {axis_coord: x_ax_w, c0: y0, c1: y1})
        N = np.zeros((n, len(coords)), dtype=np.float32)
        N[:, coords.index(c0)] = y0 / R
        N[:, coords.index(c1)] = y1 / R
        face_X["wall"], face_N["wall"] = X, N

    ordered = sorted(face_X.keys())
    Xb = np.concatenate([face_X[k] for k in ordered], axis=0)
    Nb = np.concatenate([face_N[k] for k in ordered], axis=0)

    offsets: Dict[str, Tuple[int, int]] = {}
    off = 0
    for k in ordered:
        n = face_X[k].shape[0]
        offsets[k] = (off, off + n)
        off += n

    tag_masks: Dict[str, np.ndarray] = {}
    for tag, kind in tag_faces.items():
        m = np.zeros((Xb.shape[0],), dtype=bool)
        s, e = offsets[kind]
        m[s:e] = True
        tag_masks[tag] = m

    ctx: Dict[str, Any] = dict(user_ctx or {})
    ctx.setdefault("tag_masks", {})
    ctx["tag_masks"].update(tag_masks)
    ctx.setdefault(
        "bounds",
        {
            "min": np.array([bounds[c][0] for c in coords], dtype=np.float32),
            "max": np.array([bounds[c][1] for c in coords], dtype=np.float32),
        },
    )
    ctx.setdefault("mesh_info", {"shape": "analytic_cylinder", "radius": R})

    y_bc, cond_masks = _apply_conditions(spec, Xb, ctx)

    return {
        "x_col": Xc,
        "x_bc": Xb,
        "y_bc": y_bc,
        "n_bc": Nb,
        "ctx": ctx,
        **cond_masks,
    }
