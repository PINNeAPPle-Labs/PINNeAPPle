"""PINNeAPPle results as OpenUSD stages for NVIDIA Omniverse (authored and checked with ``usd-core``).

Omniverse apps (USD Composer, Kit, Isaac Sim) open OpenUSD files. This module writes a result as a USD stage with the
physics fields kept as data, not only as colours, and checks the file before anyone opens it:

``FieldStage``
    ``/World`` (Z up, metres) with meshes (``add_mesh``), curves (``add_curves``, pipes, streamlines) and points
    (``add_points``, sensors). Every field is a ``primvars:<name>`` float array (vertex interpolation) with its unit
    in ``pinneapple:unit:<name>``; one field per prim is also baked into ``primvars:displayColor`` so any USD viewer
    shows it. ``add_time_samples`` keys a field (and its colours) over time, so the Omniverse timeline plays a
    transient result. ``set_trust`` stores a trust decision and its checks on a prim (``pinneapple:trust:*``), so the
    scene carries the verdict of the numbers it shows.
``validate_stage``
    re-opens the file and checks it: it opens, up axis and units, every mesh has consistent topology (indices inside
    the points, counts summing to the indices), every vertex primvar has one finite value per point (per time
    sample), extents are set, and the ``UsdUtils.ComplianceChecker`` result.
``render_stage_png``
    a preview: the meshes and curves read back from the file, drawn with matplotlib in their ``displayColor``. It is
    a check that the file holds what was written, not an Omniverse RTX render.
``kit_render_script``
    the script to run inside Omniverse Kit / USD Composer (``omni.usd``, ``omni.kit.viewport.utility``) to open the
    stage and capture an RTX viewport image. Needs an Omniverse install with an RTX GPU.
``grid_surface`` / ``extrude_profile`` / ``tube``
    triangle meshes for the common cases (a plate with a field on it, an extruded section, a pipe along a path).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["require_pxr", "FieldStage", "validate_stage", "render_stage_png", "render_stage_gif", "rasterize_topdown", "read_stage_geometry", "usd_validators", "kit_render_script", "grid_surface",
           "extrude_profile", "tube", "field_colors"]


def require_pxr():
    """``pxr`` (OpenUSD), with an install hint when it is missing."""
    try:
        import pxr
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: F401
        return pxr
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("OpenUSD is needed for USD stages: pip install usd-core (the same library Omniverse "
                          "uses; no GPU needed)") from e


def _name(s: str) -> str:
    n = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s) or "prim"
    return "m_" + n if n[0].isdigit() else n


def field_colors(values, cmap: str = "inferno", vrange: tuple[float, float] | None = None) -> np.ndarray:
    """``(n, 3)`` linear RGB of ``values`` through a matplotlib colour map (sRGB -> linear, as USD expects)."""
    import matplotlib

    v = np.asarray(values, float).ravel()
    lo, hi = vrange if vrange is not None else (np.nanmin(v), np.nanmax(v))
    t = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    rgb = np.asarray(matplotlib.colormaps[cmap](t))[:, :3]
    return rgb ** 2.2


# ---------------------------------------------------------------------------------------------- geometry
def grid_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of a structured ``(nx, ny)`` grid of points: ``(V (nx*ny, 3), F (2(nx-1)(ny-1), 3))``."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Z = np.zeros_like(X) if Z is None else np.asarray(Z, float)
    nx, ny = X.shape
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return V, F


def extrude_profile(xy: np.ndarray, length: float, n: int = 2, axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """A closed 2-D profile ``(m, 2)`` (in the plane normal to ``axis``) extruded over ``[0, length]`` in ``n``
    stations: the side wall as triangles. Vertex ``k * m + i`` is profile point ``i`` at station ``k``."""
    xy = np.asarray(xy, float)
    m = len(xy)
    s = np.linspace(0.0, length, n)
    V = np.zeros((n * m, 3))
    other = [a for a in range(3) if a != axis]
    for k, sk in enumerate(s):
        V[k * m:(k + 1) * m, other[0]] = xy[:, 0]
        V[k * m:(k + 1) * m, other[1]] = xy[:, 1]
        V[k * m:(k + 1) * m, axis] = sk
    F = []
    for k in range(n - 1):
        for i in range(m):
            a, b = k * m + i, k * m + (i + 1) % m
            F += [[a, b, b + m], [a, b + m, a + m]]
    return V, np.asarray(F, np.int64)


def tube(path: np.ndarray, radius: float, n_around: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A tube of ``radius`` around a polyline ``path (p, 3)``: vertex ``k * n_around + j`` is ring ``k``."""
    P = np.asarray(path, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = np.cross(T, ref)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = np.cross(T, N)
    ang = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    V = (P[:, None, :] + radius * (np.cos(ang)[None, :, None] * N[:, None, :]
                                   + np.sin(ang)[None, :, None] * B[:, None, :])).reshape(-1, 3)
    F = []
    for k in range(len(P) - 1):
        for j in range(n_around):
            a, b = k * n_around + j, k * n_around + (j + 1) % n_around
            F += [[a, b, b + n_around], [a, b + n_around, a + n_around]]
    return V, np.asarray(F, np.int64)


# ---------------------------------------------------------------------------------------------- authoring
class FieldStage:
    """A USD stage holding PINNeAPPle fields. ``save()`` writes it (``.usda`` text or ``.usd`` / ``.usdc``)."""

    def __init__(self, path: str, title: str = "PINNeAPPle result", *, meters_per_unit: float = 1.0,
                 fps: float = 24.0):
        require_pxr()
        from pxr import Sdf, Usd, UsdGeom
        self.path = os.fspath(path)
        if os.path.exists(self.path):
            os.remove(self.path)
        self.stage = Usd.Stage.CreateNew(self.path)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, meters_per_unit)
        self.stage.SetTimeCodesPerSecond(fps)
        world = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World"))
        self.stage.SetDefaultPrim(world.GetPrim())
        self.stage.GetRootLayer().documentation = f"{title}: written by PINNeAPPle (pinneapple_tools.nvidia)"
        self.title = title
        self._times: list[float] = []

    # -- helpers
    def _prim_path(self, name: str, parent: str = "/World") -> str:
        return f"{parent}/{_name(name)}"

    def xform(self, name: str, translate=(0, 0, 0), parent: str = "/World") -> str:
        from pxr import Gf, UsdGeom
        x = UsdGeom.Xform.Define(self.stage, self._prim_path(name, parent))
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
        return str(x.GetPath())

    def _fields(self, prim, fields: Mapping[str, Any], units: Mapping[str, str] | None, n: int, time=None):
        from pxr import Sdf, UsdGeom, Vt
        api = UsdGeom.PrimvarsAPI(prim)
        for k, vals in fields.items():
            v = np.asarray(vals, float).ravel()
            if v.size != n:
                raise ValueError(f"field {k!r} has {v.size} values for {n} points")
            pv = api.GetPrimvar(_name(k))
            if not pv:
                pv = api.CreatePrimvar(_name(k), Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.FloatArray(v.astype(np.float32).tolist()), *([] if time is None else [time]))
            if units and k in units:
                prim.CreateAttribute(f"pinneapple:unit:{_name(k)}", Sdf.ValueTypeNames.String).Set(units[k])

    def _colour(self, gprim, values, cmap, vrange, time=None):
        from pxr import Gf, UsdGeom, Vt
        rgb = field_colors(values, cmap, vrange)
        pv = gprim.GetDisplayColorPrimvar() or gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.SetInterpolation(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in rgb]), *([] if time is None else [time]))

    @staticmethod
    def _extent(V):
        from pxr import Gf, Vt
        lo, hi = V.min(0), V.max(0)
        return Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])

    # -- prims
    def add_mesh(self, name: str, V, F, *, fields: Mapping[str, Any] | None = None,
                 units: Mapping[str, str] | None = None, color_by: str | None = None, cmap: str = "inferno",
                 vrange: tuple[float, float] | None = None, color=(0.6, 0.6, 0.6), parent: str = "/World",
                 double_sided: bool = True) -> str:
        """A triangle mesh with per-vertex ``fields``; ``color_by`` bakes one field into ``displayColor``."""
        from pxr import Gf, UsdGeom, Vt
        V, F = np.asarray(V, float).reshape(-1, 3), np.asarray(F, np.int64).reshape(-1, 3)
        if F.size and (F.min() < 0 or F.max() >= len(V)):
            raise ValueError("face indices outside the points")
        m = UsdGeom.Mesh.Define(self.stage, self._prim_path(name, parent))
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in V]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(F.ravel().tolist()))
        m.CreateExtentAttr(self._extent(V))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDoubleSidedAttr(double_sided)
        fields = dict(fields or {})
        self._fields(m.GetPrim(), fields, units, len(V))
        if color_by:
            self._colour(m, fields[color_by], cmap, vrange)
            m.GetPrim().CreateAttribute("pinneapple:colorBy", _sdf().ValueTypeNames.String).Set(color_by)
        else:
            m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, color))]))
        return str(m.GetPath())

    def add_curves(self, name: str, paths: Sequence[np.ndarray], *, width: float = 0.05,
                   fields: Mapping[str, Sequence[np.ndarray]] | None = None, units: Mapping[str, str] | None = None,
                   color_by: str | None = None, cmap: str = "viridis", vrange=None, parent: str = "/World") -> str:
        """Linear ``BasisCurves`` (one per path); ``fields``: name -> one array per path (one value per point)."""
        from pxr import Gf, UsdGeom, Vt
        P = np.concatenate([np.asarray(p, float).reshape(-1, 3) for p in paths])
        c = UsdGeom.BasisCurves.Define(self.stage, self._prim_path(name, parent))
        c.CreateTypeAttr(UsdGeom.Tokens.linear)
        c.CreateCurveVertexCountsAttr(Vt.IntArray([len(p) for p in paths]))
        c.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in P]))
        c.CreateWidthsAttr(Vt.FloatArray([float(width)]))
        c.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        c.CreateExtentAttr(self._extent(P))
        flat = {k: np.concatenate([np.asarray(x, float).ravel() for x in v]) for k, v in (fields or {}).items()}
        self._fields(c.GetPrim(), flat, units, len(P))
        if color_by:
            self._colour(c, flat[color_by], cmap, vrange)
        return str(c.GetPath())

    def add_points(self, name: str, P, *, width: float = 0.05, fields: Mapping[str, Any] | None = None,
                   units: Mapping[str, str] | None = None, color=(1.0, 0.2, 0.1), parent: str = "/World") -> str:
        """Points (sensors, probes) with per-point ``fields``."""
        from pxr import Gf, UsdGeom, Vt
        P = np.asarray(P, float).reshape(-1, 3)
        p = UsdGeom.Points.Define(self.stage, self._prim_path(name, parent))
        p.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, x)) for x in P]))
        p.CreateWidthsAttr(Vt.FloatArray([float(width)] * len(P)))
        p.CreateExtentAttr(self._extent(P))
        p.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, color))]))
        self._fields(p.GetPrim(), dict(fields or {}), units, len(P))
        return str(p.GetPath())

    def add_time_samples(self, prim_path: str, field: str, times: Sequence[float], values: Sequence[Any], *,
                         color: bool = True, cmap: str = "inferno", vrange=None) -> None:
        """Key ``field`` of ``prim_path`` at ``times`` (seconds; one array per time). With ``color`` the
        ``displayColor`` follows, on one colour range over all times."""
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(prim_path)
        g = UsdGeom.Gprim(prim)
        n = len(UsdGeom.PointBased(prim).GetPointsAttr().Get())
        allv = np.concatenate([np.asarray(v, float).ravel() for v in values])
        vr = vrange if vrange is not None else (float(allv.min()), float(allv.max()))
        fps = self.stage.GetTimeCodesPerSecond()
        for t, v in zip(times, values, strict=True):
            tc = float(t) * fps
            self._fields(prim, {field: v}, None, n, time=tc)
            if color:
                self._colour(g, v, cmap, vr, time=tc)
            self._times.append(tc)
        self.stage.SetStartTimeCode(min(self._times))
        self.stage.SetEndTimeCode(max(self._times))

    def add_rigid_animation(self, name: str, V, F, times: Sequence[float], positions, quats_xyzw, *,
                            color=(0.7, 0.7, 0.72), colors_over_time=None, parent: str = "/World") -> str:
        """A rigid part moving over time: an ``Xform`` with time-sampled translate and orient ops (a recorded
        simulation, e.g. a robot link from the robot sim) and the mesh ``V, F`` in its local frame. ``colors_over_time``
        (one RGB per time, linear) keys the part's ``displayColor`` (e.g. a stress utilisation)."""
        from pxr import Gf, UsdGeom, Vt
        x = UsdGeom.Xform.Define(self.stage, self._prim_path(name, parent))
        top, oop = x.AddTranslateOp(), x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        fps = self.stage.GetTimeCodesPerSecond()
        for k, t in enumerate(times):
            tc = float(t) * fps
            top.Set(Gf.Vec3d(*map(float, positions[k])), tc)
            qx, qy, qz, qw = map(float, quats_xyzw[k])
            oop.Set(Gf.Quatd(qw, qx, qy, qz), tc)
            self._times.append(tc)
        mp = self.add_mesh("geom", V, F, color=color, parent=str(x.GetPath()))
        if colors_over_time is not None:
            g = UsdGeom.Mesh(self.stage.GetPrimAtPath(mp))
            attr = g.GetDisplayColorAttr()
            for k, t in enumerate(times):
                attr.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, colors_over_time[k]))]), float(t) * fps)
        self.stage.SetStartTimeCode(min(self._times))
        self.stage.SetEndTimeCode(max(self._times))
        return str(x.GetPath())

    def set_trust(self, prim_path: str, decision: str, checks: Sequence[Mapping[str, Any]] = (),
                  **info: Any) -> None:
        """Store a trust decision (APPROVED / REVIEW / REJECT) and its checks on a prim."""
        Sdf = _sdf()
        prim = self.stage.GetPrimAtPath(prim_path)
        prim.CreateAttribute("pinneapple:trust:decision", Sdf.ValueTypeNames.String).Set(str(decision))
        prim.CreateAttribute("pinneapple:trust:checks", Sdf.ValueTypeNames.String).Set(
            json.dumps(list(checks), default=_jsonable))
        for k, v in info.items():
            prim.CreateAttribute(f"pinneapple:info:{_name(k)}", Sdf.ValueTypeNames.String).Set(
                v if isinstance(v, str) else json.dumps(v, default=_jsonable))

    def save(self) -> str:
        self.stage.GetRootLayer().Save()
        return self.path


def _sdf():
    from pxr import Sdf
    return Sdf


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------------------------- checking
def validate_stage(path: str, *, required_primvars: Mapping[str, Sequence[str]] | None = None,
                   compliance: bool = True) -> dict[str, Any]:
    """Re-open a USD file and check it. Returns ``{"ok", "problems", "prims", "meshes", "curves", "points",
    "time_samples", "up_axis", "meters_per_unit", "compliance"}``. ``required_primvars``: prim path -> primvar
    names that must be there."""
    require_pxr()
    from pxr import Usd, UsdGeom
    problems: list[str] = []
    stage = Usd.Stage.Open(os.fspath(path))
    if stage is None:
        return {"ok": False, "problems": [f"{path}: does not open"]}
    up, mpu = UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage)
    if not stage.GetDefaultPrim():
        problems.append("no default prim")
    counts = {"prims": 0, "meshes": 0, "curves": 0, "points": 0}
    tsamples = 0
    for prim in stage.Traverse():
        counts["prims"] += 1
        tsamples += sum(len(a.GetTimeSamples()) for a in prim.GetAttributes())   # fields, colours, poses
        if not prim.IsA(UsdGeom.PointBased):
            continue
        pb = UsdGeom.PointBased(prim)
        kind = "meshes" if prim.IsA(UsdGeom.Mesh) else "curves" if prim.IsA(UsdGeom.BasisCurves) else "points"
        counts[kind] += 1
        pts = pb.GetPointsAttr().Get()
        n = len(pts) if pts is not None else 0
        if n == 0:
            problems.append(f"{prim.GetPath()}: no points")
            continue
        if not np.isfinite(np.asarray(pts, float)).all():
            problems.append(f"{prim.GetPath()}: non-finite points")
        if not pb.GetExtentAttr().HasAuthoredValue():
            problems.append(f"{prim.GetPath()}: no extent")
        if kind == "meshes":
            m = UsdGeom.Mesh(prim)
            fc = np.asarray(m.GetFaceVertexCountsAttr().Get(), int)
            fi = np.asarray(m.GetFaceVertexIndicesAttr().Get(), int)
            if fc.sum() != fi.size:
                problems.append(f"{prim.GetPath()}: face counts sum {fc.sum()} != {fi.size} indices")
            if fi.size and (fi.min() < 0 or fi.max() >= n):
                problems.append(f"{prim.GetPath()}: face indices outside the {n} points")
        if kind == "curves":
            cc = np.asarray(UsdGeom.BasisCurves(prim).GetCurveVertexCountsAttr().Get(), int)
            if cc.sum() != n:
                problems.append(f"{prim.GetPath()}: curve counts sum {cc.sum()} != {n} points")
        for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if pv.GetInterpolation() != UsdGeom.Tokens.vertex:
                continue
            attr = pv.GetAttr()
            times = attr.GetTimeSamples() or [Usd.TimeCode.Default()]
            for t in times:
                v = attr.Get(t)
                if v is None:
                    continue
                a = np.asarray(v, float)
                if len(a) != n:
                    problems.append(f"{prim.GetPath()}.{pv.GetPrimvarName()}: {len(a)} values for {n} points")
                elif not np.isfinite(a).all():
                    problems.append(f"{prim.GetPath()}.{pv.GetPrimvarName()}: non-finite values")
    for p, names in (required_primvars or {}).items():
        prim = stage.GetPrimAtPath(p)
        if not prim:
            problems.append(f"{p}: missing")
            continue
        api = UsdGeom.PrimvarsAPI(prim)
        problems += [f"{p}: primvar {k!r} missing" for k in names if not api.HasPrimvar(_name(k))]
    comp = usd_validators(stage) if compliance else {"run": False}
    problems += [f"validator {e['validator']}: {e['message']}" for e in comp.get("errors", [])
                 if e["type"] == "Error"]
    return {"ok": not problems, "problems": problems, **counts, "time_samples": tsamples, "up_axis": str(up),
            "meters_per_unit": float(mpu), "start": stage.GetStartTimeCode(), "end": stage.GetEndTimeCode(),
            "compliance": comp}


def usd_validators(stage) -> dict[str, Any]:
    """Every validator OpenUSD registers (``pxr.UsdValidation``: stage metadata, composition, shading, geometry
    subsets, and the ``UsdPhysics`` rigid-body, collider, joint and articulation checkers) run on ``stage``, or the
    legacy ``UsdUtils.ComplianceChecker`` on builds without ``UsdValidation``."""
    try:
        from pxr import UsdValidation
    except ImportError:
        UsdValidation = None
    if UsdValidation is not None:
        reg = UsdValidation.ValidationRegistry()
        vals = reg.GetOrLoadAllValidators()
        errs = UsdValidation.ValidationContext(vals).Validate(stage)
        return {"run": True, "engine": "pxr.UsdValidation", "validators": sorted(v.GetMetadata().name for v in vals),
                "errors": [{"validator": e.GetValidator().GetMetadata().name if e.GetValidator() else "",
                            "type": str(e.GetType()).rsplit(".", 1)[-1], "message": e.GetMessage()} for e in errs]}
    try:  # pragma: no cover - older USD builds
        from pxr import UsdUtils
        ch = UsdUtils.ComplianceChecker(rootPackageOnly=False, skipVariants=False)
        ch.CheckCompliance(stage.GetRootLayer().realPath)
        return {"run": True, "engine": "UsdUtils.ComplianceChecker", "validators": [],
                "errors": [{"validator": "ComplianceChecker", "type": "Error", "message": m} for m in ch.GetErrors()]}
    except Exception as e:  # noqa: BLE001
        return {"run": False, "why": f"{type(e).__name__}: {e}"}


def read_stage_geometry(path: str, time: float | None = None, exclude: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Meshes and curves of a USD file as arrays: ``[{"path", "kind", "points", "faces" or "counts", "colors"}]``
    (colours from ``displayColor`` at ``time``, a time code; default: the first sample). ``exclude``: prim path
    prefixes to leave out (e.g. a ground plane)."""
    require_pxr()
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(os.fspath(path))
    when = Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()
    out = []
    for prim in stage.Traverse():
        if any(str(prim.GetPath()).startswith(e) for e in exclude):
            continue
        if prim.IsA(UsdGeom.Cube):                              # boxes (Isaac Sim colliders / visuals)
            xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(when), float)
            h = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2.0
            c = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
            P = (np.c_[c, np.ones(8)] @ xf)[:, :3]
            F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
            col = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            out.append({"path": str(prim.GetPath()), "kind": "mesh", "points": P, "faces": F,
                        "colors": None if col is None else np.asarray(col, float)})
            continue
        if prim.IsA(UsdGeom.Cylinder):                          # wheels (Isaac Sim colliders / visuals)
            cy = UsdGeom.Cylinder(prim)
            r, h, ax = cy.GetRadiusAttr().Get(), cy.GetHeightAttr().Get(), cy.GetAxisAttr().Get()
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
            k = {"X": 0, "Y": 1, "Z": 2}[ax]
            o = [i for i in range(3) if i != k]
            c = np.zeros((48, 3))
            c[:24, o], c[24:, o] = ring, ring
            c[:24, k], c[24:, k] = -h / 2, h / 2
            xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(when), float)
            P = (np.c_[c, np.ones(48)] @ xf)[:, :3]
            F = [[i, (i + 1) % 24, 24 + (i + 1) % 24] for i in range(24)] + \
                [[i, 24 + (i + 1) % 24, 24 + i] for i in range(24)] + \
                [[0, i, i + 1] for i in range(1, 23)] + [[24, 24 + i + 1, 24 + i] for i in range(1, 23)]
            col = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            out.append({"path": str(prim.GetPath()), "kind": "mesh", "points": P, "faces": np.asarray(F),
                        "colors": None if col is None else np.asarray(col, float)})
            continue
        if not prim.IsA(UsdGeom.PointBased) or prim.IsA(UsdGeom.Points):
            continue
        g = UsdGeom.Gprim(prim)
        dc = g.GetDisplayColorAttr()
        ts = dc.GetTimeSamples()
        tc = Usd.TimeCode(time if time is not None else (ts[0] if ts else Usd.TimeCode.Default().GetValue())) \
            if ts else Usd.TimeCode.Default()
        cols = dc.Get(tc if ts else when)
        xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(when), float)
        P = np.asarray(UsdGeom.PointBased(prim).GetPointsAttr().Get(), float)
        P = (np.c_[P, np.ones(len(P))] @ xf)[:, :3]
        rec = {"path": str(prim.GetPath()), "points": P, "colors": None if cols is None else np.asarray(cols, float)}
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            fc = np.asarray(m.GetFaceVertexCountsAttr().Get(), int)
            fi = np.asarray(m.GetFaceVertexIndicesAttr().Get(), int)
            tris, k = [], 0
            for c in fc:                                       # fan-triangulate
                tris += [[fi[k], fi[k + j], fi[k + j + 1]] for j in range(1, c - 1)]
                k += c
            rec.update(kind="mesh", faces=np.asarray(tris, int).reshape(-1, 3))
        else:
            rec.update(kind="curves", counts=np.asarray(UsdGeom.BasisCurves(prim).GetCurveVertexCountsAttr().Get()))
        out.append(rec)
    return out


def render_stage_png(path: str, png: str, *, time: float | None = None, elev: float = 28, azim: float = -60,
                     title: str = "", colorbar: tuple[str, float, float, str] | None = None,
                     size: tuple[float, float] = (7.5, 5.5), dpi: int = 130, flat: bool = False,
                     exclude: Sequence[str] = ()) -> str:
    """Draw the meshes and curves of a USD file (read back from disk) in their ``displayColor``; ``colorbar``:
    ``(label, lo, hi, cmap)``; ``flat``: axes scaled to the data in x, y and a squashed z (long, low scenes such
    as a pipeline). A file check, not an Omniverse RTX render."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    geo = read_stage_geometry(path, time, exclude)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(projection="3d")
    allp = []
    for g in geo:
        P = g["points"]
        allp.append(P)
        cols = g["colors"]
        lin = None if cols is None else np.clip(cols, 0, 1) ** (1 / 2.2)
        if g["kind"] == "mesh":
            F = g["faces"]
            if lin is not None and len(lin) == len(P):
                fc = lin[F].mean(1)
            else:
                fc = np.tile(lin[0] if lin is not None else [0.6, 0.6, 0.6], (len(F), 1))
            ax.add_collection3d(Poly3DCollection(P[F], facecolors=fc, edgecolors="none", linewidths=0))
        else:
            segs, cs, k = [], [], 0
            for c in g["counts"]:
                for j in range(k, k + c - 1):
                    segs.append(P[j:j + 2])
                    cs.append(lin[j] if lin is not None and len(lin) == len(P) else [0.2, 0.2, 0.2])
                k += c
            ax.add_collection3d(Line3DCollection(segs, colors=cs, linewidths=2.5))
    A = np.concatenate(allp)
    lo, hi = A.min(0), A.max(0)
    mid, r = (lo + hi) / 2, (hi - lo).max() / 2 or 1.0
    if flat:
        ext = np.maximum(hi - lo, 1e-9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((ext[0], ext[1], max(ext[2], 0.08 * ext[:2].max())))
    else:
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    from matplotlib.ticker import MaxNLocator
    for axis, n in ((ax.xaxis, 5), (ax.yaxis, 4), (ax.zaxis, 4)):
        axis.set_major_locator(MaxNLocator(n))
    ax.tick_params(labelsize=7, pad=1)
    ax.set_xlabel("x [m]", labelpad=10)
    ax.set_ylabel("y [m]", labelpad=10)
    if flat:                    # a squashed z axis has no room for ticks and a label next to the y labels
        ax.set_zticks([])
    else:
        ax.set_zlabel("z [m]", labelpad=6)
    ax.set_title(title or f"{os.path.basename(path)} (read back with OpenUSD)", fontsize=10)
    if colorbar:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        lab, a, b, cmap = colorbar
        # pad: the 3-D axes draw the z label outside their box, where a default colorbar would sit
        fig.colorbar(cm.ScalarMappable(Normalize(a, b), cmap), ax=ax, shrink=0.6, label=lab, pad=0.12 if not flat else 0.05)
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png


def kit_render_script(usd_path: str, png_path: str, *, frames: int = 60) -> str:
    """Python to run inside Omniverse Kit / USD Composer (Window > Script Editor, or ``kit --exec``): opens the stage
    and writes an RTX viewport capture. Needs Omniverse with an RTX GPU; not runnable with ``usd-core`` alone. Save
    it next to the stage: the paths are resolved from the script's own folder."""
    return f'''# Run inside NVIDIA Omniverse Kit / USD Composer (needs an RTX GPU). Written by pinneapple_tools.nvidia.
import asyncio
import os

import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

HERE = os.path.dirname(os.path.abspath(__file__))


async def main():
    ctx = omni.usd.get_context()
    await ctx.open_stage_async(os.path.join(HERE, {os.path.basename(usd_path)!r}))
    for _ in range({int(frames)}):                       # let RTX converge
        await omni.kit.app.get_app().next_update_async()
    capture_viewport_to_file(get_active_viewport(), os.path.join(HERE, {os.path.basename(png_path)!r}))
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

asyncio.ensure_future(main())
'''


def render_stage_gif(path: str, gif: str, times: Sequence[float], *, fps: float = 6.0, label=None, **kw) -> str:
    """An animated GIF of a time-sampled stage: :func:`render_stage_png` at each time code in ``times``;
    ``label(t)`` gives each frame's title."""
    import tempfile

    from PIL import Image
    frames = []
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(times):
            p = os.path.join(d, f"f{i:04d}.png")
            render_stage_png(path, p, time=t, title=label(t) if label else f"time code {t:g}", **kw)
            frames.append(Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128))
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                       optimize=True)
    return gif


def rasterize_topdown(path: str, H: int, W: int, *, bounds=None, time: float | None = None,
                      exclude: Sequence[str] = (), background=(0.0, 0.0, 0.0), view: str = "top") -> np.ndarray:
    """An orthographic view of the meshes of a USD file, ``(H, W, 3)`` sRGB in [0, 1]: every triangle is filled
    with its vertices' ``displayColor`` interpolated (barycentric), nearest on top. ``view``: ``"top"`` (looking down
    -z: x right, y up) or ``"side"`` (looking along +y: x right, z up). ``bounds``: ``(left, right, bottom, top)`` of
    the view in those two axes (default: the meshes' extent). A camera for synthetic data, not an RTX render."""
    geo = [g for g in read_stage_geometry(path, time, exclude) if g["kind"] == "mesh"]
    if view == "side":
        geo = [dict(g, points=g["points"][:, [0, 2, 1]] * [1, 1, -1]) for g in geo]
    elif view != "top":
        raise ValueError(f"view must be 'top' or 'side', got {view!r}")
    P = np.concatenate([g["points"] for g in geo])
    xmin, xmax, ymin, ymax = bounds if bounds is not None else (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(),
                                                                 P[:, 1].max())
    img = np.tile(np.asarray(background, float), (H, W, 1))
    zbuf = np.full((H, W), -np.inf)
    for g in geo:
        V, F = g["points"], g["faces"]
        C = g["colors"]
        C = np.tile(C[0] if C is not None else [0.6, 0.6, 0.6], (len(V), 1)) if C is None or len(C) != len(V) else C
        C = np.clip(C, 0, 1) ** (1 / 2.2)
        u = (V[:, 0] - xmin) / (xmax - xmin) * (W - 1)
        v = (ymax - V[:, 1]) / (ymax - ymin) * (H - 1)
        for f in F:
            x0, x1, x2 = u[f]
            y0, y1, y2 = v[f]
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            ia, ib = int(max(np.floor(min(x0, x1, x2)), 0)), int(min(np.ceil(max(x0, x1, x2)), W - 1))
            ja, jb = int(max(np.floor(min(y0, y1, y2)), 0)), int(min(np.ceil(max(y0, y1, y2)), H - 1))
            if ia > ib or ja > jb:
                continue
            xx, yy = np.meshgrid(np.arange(ia, ib + 1), np.arange(ja, jb + 1))
            l0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
            l1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
            if not inside.any():
                continue
            z = l0 * V[f[0], 2] + l1 * V[f[1], 2] + l2 * V[f[2], 2]
            top = inside & (z > zbuf[yy, xx])
            if not top.any():
                continue
            col = l0[..., None] * C[f[0]] + l1[..., None] * C[f[1]] + l2[..., None] * C[f[2]]
            img[yy[top], xx[top]] = col[top]
            zbuf[yy[top], xx[top]] = z[top]
    return img
