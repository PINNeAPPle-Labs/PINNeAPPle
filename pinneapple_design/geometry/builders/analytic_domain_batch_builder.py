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


# ---------------------------------------------------------------------------
# Second follow-up pass (2026-09-17): three more analytic shapes, added for
# the same reason as the two above -- genuinely real, published/standard
# geometry that is fully described by a preset's own parameters, sampled
# exactly (not via a mesh file). See tag_geometry.py for which preset uses
# which and, for each, the exact published formula/source cited in code.
# ---------------------------------------------------------------------------

def sample_axisymmetric_wall_tag_batch(
    spec: ProblemSpec,
    *,
    axis_coord: str,
    radial_coord: str,
    r_wall_fn,
    dr_wall_fn,
    tag_faces: Dict[str, str],
    n_col: int = 20_000,
    n_bc_per_face: int = 3_000,
    seed: int = 7,
    user_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample a real axisymmetric (r, z) domain whose wall radius varies
    along the axis as ``r_wall_fn(z)`` (e.g. a conical nozzle) -- a
    genuine, non-cylindrical generalization of :func:`sample_cylinder_tag_batch`
    needed because the wall is not at a constant radius.

    Parameters
    ----------
    r_wall_fn, dr_wall_fn : callables ``z (np.ndarray) -> r`` / ``dr/dz``,
        the wall contour and its exact derivative (used for the wall's
        outward normal; for a straight cone this is just a constant slope).
    tag_faces : dict tag_name -> one of "inlet" (axis_coord==min, disk of
        radius r_wall_fn(min)), "outlet" (axis_coord==max, disk of radius
        r_wall_fn(max)), "axis" (radial_coord==0, the symmetry line),
        "wall" (radial_coord==r_wall_fn(axis_coord), the nozzle contour).
    """
    rng = np.random.default_rng(seed)
    coords = list(spec.coords)
    bounds = spec.domain_bounds
    ax_lo, ax_hi = bounds[axis_coord]
    free_coords = [c for c in coords if c not in (axis_coord, radial_coord)]

    def _free_others(n: int, overrides: Dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for c in coords:
            if c in overrides:
                cols.append(overrides[c])
            else:
                lo, hi = bounds[c]
                cols.append(rng.uniform(float(lo), float(hi), size=n).astype(np.float32))
        return np.stack(cols, axis=1)

    # ---- interior collocation: uniform in z, uniform in r in [0, r_wall(z)]
    # -- this treats (r, z) as a literal 2D computational plane (matching
    # how the preset already declares coords=("r","z") and a plain
    # rectangular domain_bounds for r), not a 3D-volume-weighted sample. ----
    z_col = rng.uniform(float(ax_lo), float(ax_hi), size=n_col).astype(np.float32)
    r_wall_col = np.asarray(r_wall_fn(z_col), dtype=np.float32)
    r_col = (rng.uniform(0.0, 1.0, size=n_col).astype(np.float32)) * r_wall_col
    Xc = _free_others(n_col, {axis_coord: z_col, radial_coord: r_col})

    face_kinds = set(tag_faces.values())
    face_X: Dict[str, np.ndarray] = {}
    face_N: Dict[str, np.ndarray] = {}

    if "inlet" in face_kinds:
        r_in = float(r_wall_fn(np.array([ax_lo], dtype=np.float32))[0])
        r = rng.uniform(0.0, 1.0, size=n_bc_per_face).astype(np.float32) * r_in
        X = _free_others(n_bc_per_face, {axis_coord: np.full(n_bc_per_face, float(ax_lo), dtype=np.float32), radial_coord: r})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(axis_coord)] = -1.0
        face_X["inlet"], face_N["inlet"] = X, N

    if "outlet" in face_kinds:
        r_out = float(r_wall_fn(np.array([ax_hi], dtype=np.float32))[0])
        r = rng.uniform(0.0, 1.0, size=n_bc_per_face).astype(np.float32) * r_out
        X = _free_others(n_bc_per_face, {axis_coord: np.full(n_bc_per_face, float(ax_hi), dtype=np.float32), radial_coord: r})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(axis_coord)] = 1.0
        face_X["outlet"], face_N["outlet"] = X, N

    if "axis" in face_kinds:
        z = rng.uniform(float(ax_lo), float(ax_hi), size=n_bc_per_face).astype(np.float32)
        X = _free_others(n_bc_per_face, {axis_coord: z, radial_coord: np.zeros(n_bc_per_face, dtype=np.float32)})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(radial_coord)] = -1.0
        face_X["axis"], face_N["axis"] = X, N

    if "wall" in face_kinds:
        z = rng.uniform(float(ax_lo), float(ax_hi), size=n_bc_per_face).astype(np.float32)
        r = np.asarray(r_wall_fn(z), dtype=np.float32)
        X = _free_others(n_bc_per_face, {axis_coord: z, radial_coord: r})
        dr = np.asarray(dr_wall_fn(z), dtype=np.float32)
        # Outward normal to the fluid region {radial < r_wall(axis)}:
        # implicit g(r,z) = r - r_wall(z), fluid is g<0, outward = +grad(g)
        # normalized = (1, -dr/dz) / ||.||.
        norm = np.sqrt(1.0 + dr ** 2)
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(radial_coord)] = 1.0 / norm
        N[:, coords.index(axis_coord)] = -dr / norm
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
    ctx.setdefault("mesh_info", {"shape": "analytic_axisymmetric_wall"})

    y_bc, cond_masks = _apply_conditions(spec, Xb, ctx)

    return {"x_col": Xc, "x_bc": Xb, "y_bc": y_bc, "n_bc": Nb, "ctx": ctx, **cond_masks}


def sample_annulus_tag_batch(
    spec: ProblemSpec,
    tag_faces: Dict[str, str],
    *,
    cross_coords: Tuple[str, str],
    inner_radius: float,
    outer_radius: float,
    n_col: int = 20_000,
    n_bc_per_face: int = 3_000,
    seed: int = 7,
    user_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample a real annular (ring) cross-section domain: ``inner_radius <=
    sqrt(x^2+y^2) <= outer_radius``. ``tag_faces``: tag_name -> "inner" (the
    r=inner_radius circle) or "outer" (the r=outer_radius circle)."""
    rng = np.random.default_rng(seed)
    coords = list(spec.coords)
    bounds = spec.domain_bounds
    c0, c1 = cross_coords
    Ri, Ro = float(inner_radius), float(outer_radius)
    free_coords = [c for c in coords if c not in (c0, c1)]

    def _free_others(n: int, overrides: Dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for c in coords:
            if c in overrides:
                cols.append(overrides[c])
            else:
                lo, hi = bounds[c]
                cols.append(rng.uniform(float(lo), float(hi), size=n).astype(np.float32))
        return np.stack(cols, axis=1)

    # ---- interior collocation: rejection-sample the annulus within the
    # bounding square [-Ro, Ro]^2. ----
    pts = np.zeros((0, 2), dtype=np.float32)
    while pts.shape[0] < n_col:
        cand = rng.uniform(-Ro, Ro, size=(max(n_col, 512), 2)).astype(np.float32)
        r2 = cand[:, 0] ** 2 + cand[:, 1] ** 2
        keep = (r2 <= Ro ** 2) & (r2 >= Ri ** 2)
        pts = np.concatenate([pts, cand[keep]], axis=0)
    pts = pts[:n_col]
    Xc = _free_others(n_col, {c0: pts[:, 0], c1: pts[:, 1]})

    def _ring(n: int, R: float) -> Tuple[np.ndarray, np.ndarray]:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
        return R * np.cos(theta), R * np.sin(theta)

    face_kinds = set(tag_faces.values())
    face_X: Dict[str, np.ndarray] = {}
    face_N: Dict[str, np.ndarray] = {}

    if "inner" in face_kinds:
        x0, y0 = _ring(n_bc_per_face, Ri)
        X = _free_others(n_bc_per_face, {c0: x0, c1: y0})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        # Outward normal of the SOLID annulus at its inner boundary points
        # toward the center (into the hole).
        N[:, coords.index(c0)] = -x0 / Ri
        N[:, coords.index(c1)] = -y0 / Ri
        face_X["inner"], face_N["inner"] = X, N

    if "outer" in face_kinds:
        x0, y0 = _ring(n_bc_per_face, Ro)
        X = _free_others(n_bc_per_face, {c0: x0, c1: y0})
        N = np.zeros((n_bc_per_face, len(coords)), dtype=np.float32)
        N[:, coords.index(c0)] = x0 / Ro
        N[:, coords.index(c1)] = y0 / Ro
        face_X["outer"], face_N["outer"] = X, N

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
    ctx.setdefault("mesh_info", {"shape": "analytic_annulus", "inner_radius": Ri, "outer_radius": Ro})

    y_bc, cond_masks = _apply_conditions(spec, Xb, ctx)

    return {"x_col": Xc, "x_bc": Xb, "y_bc": y_bc, "n_bc": Nb, "ctx": ctx, **cond_masks}


def sample_box_with_curve_tag_batch(
    spec: ProblemSpec,
    *,
    shape_coords: Tuple[str, str],
    box_tag_faces: Optional[Dict[str, List[FaceSpec]]] = None,
    curve_tags: Optional[Dict[str, Any]] = None,
    inside_body_fn: Optional[Any] = None,
    n_col: int = 20_000,
    n_bc_per_face: int = 3_000,
    n_bc_curve: int = 3_000,
    seed: int = 7,
    user_ctx: Optional[Dict[str, Any]] = None,
    max_reject_iters: int = 50,
) -> Dict[str, Any]:
    """Sample a 2D box domain (``shape_coords``, e.g. ("x","y")) with zero or
    more solid bodies/curves embedded in it -- e.g. an airfoil/blade/car-body
    silhouette that is NOT a face of the bounding box.

    Parameters
    ----------
    box_tag_faces : same format as :func:`sample_box_tag_batch`'s
        ``tag_faces`` -- outer-boundary tags (inlet/outlet/walls/...).
    curve_tags : dict tag_name -> callable(n, rng) -> (n, 2) array of points
        on that tag's curve, in ``shape_coords`` order. The curve is treated
        as a zero-thickness Dirichlet wall unless ``inside_body_fn`` also
        excludes its interior from collocation sampling.
    inside_body_fn : optional callable(X: (N,2)) -> bool array, True where a
        point is inside a solid body -- used to reject collocation points
        (and, defensively, any box-face point) that would otherwise fall
        inside it. When None, the curve is a zero-thickness (no exclusion)
        wall, appropriate for a thin cambered-plate blade model.
    """
    rng = np.random.default_rng(seed)
    coords = list(spec.coords)
    bounds = spec.domain_bounds
    c0, c1 = shape_coords
    box_tag_faces = box_tag_faces or {}
    curve_tags = curve_tags or {}

    def _inside(X2: np.ndarray) -> np.ndarray:
        if inside_body_fn is None:
            return np.zeros((X2.shape[0],), dtype=bool)
        return np.asarray(inside_body_fn(X2), dtype=bool)

    # ---- interior collocation, rejecting points inside any embedded body ----
    Xc = _sample_free_columns(rng, coords, bounds, n_col)
    if inside_body_fn is not None:
        shape_idx = [coords.index(c0), coords.index(c1)]
        bad = _inside(Xc[:, shape_idx])
        it = 0
        while np.any(bad) and it < max_reject_iters:
            n_bad = int(bad.sum())
            repl = _sample_free_columns(rng, coords, bounds, n_bad)
            Xc[bad] = repl
            bad = _inside(Xc[:, shape_idx])
            it += 1

    # ---- outer box faces (same construction as sample_box_tag_batch) ----
    unique_faces = sorted({f for faces in box_tag_faces.values() for f in faces})
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

    segments_X: List[np.ndarray] = []
    segments_N: List[np.ndarray] = []
    segment_names: List[Any] = []
    for f in unique_faces:
        segments_X.append(face_X[f])
        segments_N.append(face_N[f])
        segment_names.append(f)

    # ---- embedded curve tags ----
    curve_offsets: Dict[str, Tuple[int, int]] = {}
    running = sum(x.shape[0] for x in segments_X)
    for tag, fn in curve_tags.items():
        pts2 = np.asarray(fn(n_bc_curve, rng), dtype=np.float32)
        X = np.zeros((pts2.shape[0], len(coords)), dtype=np.float32)
        X[:, coords.index(c0)] = pts2[:, 0]
        X[:, coords.index(c1)] = pts2[:, 1]
        # any coordinate beyond the 2 shape coords is sampled uniformly free
        for c in coords:
            if c not in (c0, c1):
                lo, hi = bounds[c]
                X[:, coords.index(c)] = rng.uniform(float(lo), float(hi), size=pts2.shape[0]).astype(np.float32)
        N = np.zeros((pts2.shape[0], len(coords)), dtype=np.float32)  # unused (Dirichlet-only curve tags)
        segments_X.append(X)
        segments_N.append(N)
        curve_offsets[tag] = (running, running + X.shape[0])
        running += X.shape[0]

    Xb = np.concatenate(segments_X, axis=0) if segments_X else np.zeros((0, len(coords)), dtype=np.float32)
    Nb = np.concatenate(segments_N, axis=0) if segments_N else np.zeros((0, len(coords)), dtype=np.float32)

    offsets: Dict[FaceSpec, Tuple[int, int]] = {}
    off = 0
    for f in unique_faces:
        n = face_X[f].shape[0]
        offsets[f] = (off, off + n)
        off += n

    tag_masks: Dict[str, np.ndarray] = {}
    for tag, faces in box_tag_faces.items():
        m = np.zeros((Xb.shape[0],), dtype=bool)
        for f in faces:
            s, e = offsets[f]
            m[s:e] = True
        tag_masks[tag] = m
    for tag, (s, e) in curve_offsets.items():
        m = np.zeros((Xb.shape[0],), dtype=bool)
        m[s:e] = True
        tag_masks[tag] = m

    ctx: Dict[str, Any] = dict(user_ctx or {})
    ctx.setdefault("tag_masks", {})
    ctx["tag_masks"].update(tag_masks)
    ctx.setdefault("mesh_info", {"shape": "analytic_box_with_curve", "curve_tags": list(curve_tags.keys())})

    y_bc, cond_masks = _apply_conditions(spec, Xb, ctx)

    return {"x_col": Xc, "x_bc": Xb, "y_bc": y_bc, "n_bc": Nb, "ctx": ctx, **cond_masks}
