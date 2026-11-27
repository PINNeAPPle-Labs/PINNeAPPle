"""Drive an OpenUSD stage from a PINNeAPPle forecast: time-sampled attributes, a streamer, and a reader.

A forecast is a field per part, a pose per part and a series per sensor, one value per forecast step. This module
turns it into the USD an Omniverse app (USD Composer, Kit, Isaac Sim) or ``usdview`` plays back, and it does it the
two ways a live link needs:

``LiveLinkStreamer``
    an open stage that grows one time code per :meth:`~LiveLinkStreamer.push`: after step ``k`` the file on disk holds
    samples ``0..k`` and ``endTimeCode = k``. That is the live-link half that does not need Kit -- a writer process
    streams steps into the stage while a viewer opens the same file.
``write_forecast_stage``
    the same writer, run to the end, for a forecast that is already known.
``read_forecast_stage`` / ``play_stage``
    a reader that re-opens the file **with ``pxr`` alone** and plays the time samples back as frames: poses, the
    colour ramp, the fields and the sensor values, step by step. This is the part that is proved in the tests; it
    needs no Omniverse install.
``KitLiveLink`` / ``kit_livelink_script``
    the Kit half: applying the same samples to a stage that is already open in an Omniverse Kit application
    (``omni.usd``, ``omni.kit.app``). Kit is an Omniverse launcher / USD Composer install, not a wheel, so this half
    raises with the extra named when ``omni`` is absent and the script it would run is written out instead
    (``kit_livelink_script``): declared, not executed here.

What is authored per forecast step, for every part:

===============================  ===========================================================
``xformOp:translate``            the part's position (``ForecastPart.position``), or the
                                 tracked extremum of its field (:func:`follow_field`)
``xformOp:scale``                its scale (``ForecastPart.scale``), e.g. the magnitude of the
                                 field at that step
``primvars:<field>``             the per-vertex field, one array per step
``primvars:displayColor``        the field through a matplotlib colour map (the colour ramp),
                                 one range over the whole forecast
===============================  ===========================================================

and for every sensor a ``pinneapple:value`` time series on a sphere, whose ``displayColor`` turns red at the steps
that leave its envelope. Layer metadata (``customLayerData``) carries the format tag, the title, the source, the
length and time units, the real time values of the steps (the time codes are step indices) and the trust record of
the forecast that produced it, so the meaning of the numbers survives the hand-off.

The writer needs ``pxr`` (``pip install "pinneapple[omniverse]"``, ``usd-core``): no GPU, no Kit.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .omniverse import field_colors, require_pxr

__all__ = ["LIVELINK_FORMAT", "OMNIVERSE_EXTRA", "require_omniverse", "ForecastPart", "SensorTrack", "ForecastTrack",
           "track_from_scene", "follow_field", "LiveLinkStreamer", "write_forecast_stage", "ForecastPlayback",
           "read_forecast_stage", "play_stage", "kit_available", "KitLiveLink", "kit_livelink_script"]

LIVELINK_FORMAT = "pinneapple-livelink/1"
OMNIVERSE_EXTRA = 'pip install "pinneapple[omniverse]"'
KIT_HINT = (f"the Omniverse Kit live link needs Kit (omni.usd), which ships with the Omniverse launcher / "
            f"USD Composer, not with pip. The OpenUSD side that writes, streams and reads the stage is installed by "
            f"{OMNIVERSE_EXTRA} (usd-core) and needs no GPU; run kit_livelink_script() inside Kit to apply the "
            f"time samples to the stage that is open there.")
_IN_ENVELOPE = (0.10, 0.75, 0.20)
_OUT_OF_ENVELOPE = (0.90, 0.10, 0.10)


def require_omniverse() -> Any:
    """``pxr`` (OpenUSD), with the install extra named when it is missing."""
    try:
        return require_pxr()
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"the Omniverse live link needs OpenUSD: {OMNIVERSE_EXTRA} (usd-core; no GPU, "
                          f"no Kit needed to write, stream and read a stage)") from e


# ---------------------------------------------------------------------------------------------- the forecast
@dataclass
class ForecastPart:
    """One mesh of the forecast: geometry in its own frame, fields over time, and an optional pose over time."""
    name: str
    vertices: np.ndarray                        # (V, 3)
    faces: np.ndarray                           # (F, 3)
    fields: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (T, V) transient or (V,) steady
    field_units: dict[str, str] = field(default_factory=dict)
    position: np.ndarray | None = None          # (T, 3) or (3,): xformOp:translate time samples
    scale: np.ndarray | None = None             # (T,) or (T, 3): xformOp:scale time samples
    color: Sequence[float] = (0.6, 0.6, 0.62)   # displayColor when there is no field to colour by
    cmap: str = "inferno"
    color_by: str | None = None                 # field baked into displayColor (default: the first field)

    def color_field(self) -> str | None:
        if self.color_by is not None:
            return self.color_by
        return next(iter(self.fields), None)


@dataclass
class SensorTrack:
    """A sensor of the forecast: a position, a series of values and the envelope the series is judged against."""
    name: str
    position: Sequence[float]
    series: np.ndarray                          # (T,)
    unit: str = ""
    envelope: tuple[float, float] | None = None


@dataclass
class ForecastTrack:
    """A forecast ready to stream: ``T`` steps of ``times``, parts with per-step fields and poses, and sensors."""
    title: str
    times: Sequence[float]
    parts: list[ForecastPart]
    sensors: list[SensorTrack] = field(default_factory=list)
    time_unit: str = "s"
    length_unit: str = "m"
    source: str = ""
    trust: Mapping[str, Any] | None = None      # the Trust record of the forecast, written to layer metadata

    def __len__(self) -> int:
        return int(len(np.asarray(self.times)))

    @property
    def n_steps(self) -> int:
        return len(self)

    def part(self, name: str) -> ForecastPart:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"no part {name!r} in the forecast (have {[p.name for p in self.parts]})")

    def check(self) -> list[str]:
        """Problems that would make the stage wrong: the fields, the poses and the series must have ``T`` steps."""
        n, problems = self.n_steps, []
        if n < 1:
            problems.append("the forecast has no steps")
        for p in self.parts:
            for name, arr in p.fields.items():
                a = np.asarray(arr, float)
                if a.ndim == 1:
                    continue
                if a.ndim != 2 or a.shape != (n, len(p.vertices)):
                    problems.append(f"{p.name}.{name}: expected ({n}, {len(p.vertices)}) for {n} steps, got {a.shape}")
                elif not np.isfinite(a).all():
                    problems.append(f"{p.name}.{name}: non-finite values")
            for what, arr in (("position", p.position), ("scale", p.scale)):
                if arr is None:
                    continue
                a = np.asarray(arr, float)
                if not np.isfinite(a).all():
                    problems.append(f"{p.name}.{what}: non-finite values")
                elif what == "position":
                    if not (a.shape == (3,) or a.shape == (n, 3)):
                        problems.append(f"{p.name}.position: expected (3,) or ({n}, 3), got {a.shape}")
                elif a.ndim and a.shape[0] not in (n, 3):
                    problems.append(f"{p.name}.scale: expected a scalar, (3,), ({n},) or ({n}, 3), got {a.shape}")
            cf = p.color_field()
            if cf is not None and cf not in p.fields:
                problems.append(f"{p.name}: color_by {cf!r} is not one of its fields {sorted(p.fields)}")
        for s in self.sensors:
            if np.asarray(s.series, float).size != n:
                problems.append(f"sensor {s.name}: {np.asarray(s.series).size} values for {n} steps")
        return problems


def track_from_scene(scene: Any) -> ForecastTrack:
    """A :class:`ForecastTrack` from a :class:`pinneapple_twin3d.scene.Scene` (:class:`~pinneapple_twin3d.scene.Part`
    fields become transient primvars, recorded motion becomes the pose, sensors keep their series and envelope)."""
    times = np.arange(1) if scene.times is None else np.asarray(scene.times, float)
    parts = []
    for p in scene.parts:
        motion = getattr(p, "motion", None)
        parts.append(ForecastPart(
            name=p.name, vertices=p.vertices, faces=p.faces,
            fields={k: np.asarray(v, float) for k, v in p.fields.items()},
            field_units={k: scene.field_units.get(k, "") for k in p.fields},
            position=None if motion is None else np.asarray(motion, float)[:, :3],
            color=tuple(p.color)))
    sensors = [SensorTrack(s.id, np.asarray(s.position, float),
                           np.full(len(times), np.nan) if s.series is None else np.asarray(s.series, float),
                           unit=s.unit, envelope=None if s.envelope is None else tuple(s.envelope))
               for s in scene.sensors]
    return ForecastTrack(title=scene.title, times=times, parts=parts, sensors=sensors, time_unit=scene.time_unit,
                         length_unit=scene.length_unit, source=scene.source,
                         trust=_trust_dict(getattr(scene, "trust", None)))


def _trust_dict(trust: Any) -> Mapping[str, Any] | None:
    if trust is None:
        return None
    if isinstance(trust, Mapping):
        return dict(trust)
    for attr in ("to_dict", "summary"):
        fn = getattr(trust, attr, None)
        if callable(fn):
            try:
                out = fn()
            except TypeError:
                continue
            if isinstance(out, Mapping):
                return dict(out)
    return {"summary": str(trust)}


def _marker(name: str, radius: float, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A small octahedron at ``center``: the marker a tracked extremum is drawn with."""
    c = np.asarray(center, float)
    v = c + radius * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return v, f


def follow_field(track: ForecastTrack, part: str, *, field: str | None = None, quantile: float = 0.9,
                 marker: str | None = None, radius: float = 0.02, cmap: str = "hot") -> ForecastPart:
    """Add a marker mesh that follows where the forecast is extreme, and return it.

    At every step the marker's ``xformOp:translate`` is the centroid of the vertices in the top ``quantile`` of the
    field (where the forecast says the hot spot is) and its ``xformOp:scale`` is the extremum normalised over the
    forecast, so the marker both moves and grows with the field. This is the "position and scale driven by a
    PINNeAPPle forecast" pair: without it a stage shows a colour ramp that changes in place and nothing moves.
    """
    p = track.part(part)
    name = field or p.color_field()
    if name is None or name not in p.fields:
        raise KeyError(f"{part}: no field to follow (fields: {sorted(p.fields)})")
    a = np.asarray(p.fields[name], float)
    V = np.asarray(p.vertices, float)
    if a.ndim == 1:
        a = np.repeat(a[None, :], track.n_steps, axis=0)
    pos = np.zeros((len(a), 3))
    ext = np.zeros(len(a))
    for k, row in enumerate(a):
        thr = np.quantile(row, quantile)
        hot = row >= thr
        w = np.clip(row - row.min(), 0.0, None)
        w = np.where(hot, w, 0.0)
        w = w / w.sum() if w.sum() > 0 else np.full(len(row), 1.0 / len(row))
        pos[k] = w @ V
        ext[k] = float(row.max())
    lo, hi = float(np.min(ext)), float(np.max(ext))
    scale = 1.0 + 3.0 * (ext - lo) / (hi - lo if hi > lo else 1.0)
    v, f = _marker(name, radius)
    name_out = marker or f"{part} hot spot"
    if any(q.name == name_out for q in track.parts):
        raise ValueError(f"a part named {name_out!r} is already in the forecast")
    mp = ForecastPart(name=name_out, vertices=v, faces=f, position=pos, scale=scale,
                      color=(1.0, 0.35, 0.1), cmap=cmap)
    track.parts.append(mp)
    return mp


# ---------------------------------------------------------------------------------------------- writing
class LiveLinkStreamer:
    """An OpenUSD stage that grows one time code per :meth:`push`.

    ``LiveLinkStreamer(path, track)`` declares the prims and their attributes (no samples yet), then ``push(k)``
    authors step ``k`` and saves: the file on disk always ends at the last step streamed, which is what a viewer
    that re-opens it sees. ``write_forecast_stage`` is ``stream`` + ``close``.
    """

    def __init__(self, path: str, track: ForecastTrack, *, meters_per_unit: float = 1.0,
                 save_every: int = 1):
        require_omniverse()
        from pxr import Sdf, Usd, UsdGeom
        problems = track.check()
        if problems:
            raise ValueError("the forecast cannot be streamed: " + "; ".join(problems))
        self.path = os.fspath(path)
        self.track = track
        self.pushed = 0
        self.save_every = max(1, int(save_every))
        self.stage = Usd.Stage.CreateNew(self.path)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, float(meters_per_unit))
        self.stage.SetTimeCodesPerSecond(1.0)                    # one time code per forecast step
        world = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World"))
        self.stage.SetDefaultPrim(world.GetPrim())
        layer = self.stage.GetRootLayer()
        layer.documentation = f"{track.title}: streamed by PINNeAPPle (pinneapple_tools.nvidia.livelink)"
        layer.customLayerData = self._metadata()
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(0)
        self._primvars: dict[str, Any] = {}
        self._ops: dict[str, tuple[Any, Any]] = {}
        self._colors: dict[str, Any] = {}
        self._sensors: dict[str, Any] = {}
        self._author_prims()
        layer.Save()

    # -- layout
    def _metadata(self) -> dict[str, Any]:
        from pxr import Vt
        t = self.track
        md: dict[str, Any] = {"pinneapple_format": LIVELINK_FORMAT, "title": str(t.title),
                             "source": str(t.source), "time_unit": str(t.time_unit),
                             "length_unit": str(t.length_unit), "steps": int(t.n_steps),
                             "time_values": Vt.DoubleArray([float(x) for x in np.asarray(t.times, float)])}
        if t.trust is not None:
            md["pinneapple_trust"] = json.dumps(t.trust, default=_jsonable, sort_keys=True)
        return md

    def _author_prims(self) -> None:
        from pxr import Gf, Sdf, UsdGeom, Vt
        for p in self.track.parts:
            V = np.asarray(p.vertices, float).reshape(-1, 3)
            F = np.asarray(p.faces, np.int64).reshape(-1, 3)
            if F.size and (F.min() < 0 or F.max() >= len(V)):
                raise ValueError(f"{p.name}: face indices outside the {len(V)} points")
            mesh = UsdGeom.Mesh.Define(self.stage, f"/World/{_safe(p.name)}")
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, x)) for x in V]))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(F.ravel().tolist()))
            lo, hi = V.min(0) if len(V) else np.zeros(3), V.max(0) if len(V) else np.zeros(3)
            mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mp = mesh.GetPrim()
            mp.CreateAttribute("pinneapple:name", Sdf.ValueTypeNames.String).Set(str(p.name))
            mp.CreateAttribute("pinneapple:source", Sdf.ValueTypeNames.String).Set(str(self.track.source))
            api = UsdGeom.PrimvarsAPI(mp)
            for fname, arr in p.fields.items():
                pv = api.CreatePrimvar(_safe(fname), Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
                pv.SetInterpolation(UsdGeom.Tokens.vertex)
                mp.CreateAttribute(f"pinneapple:unit:{_safe(fname)}", Sdf.ValueTypeNames.String).Set(
                    str(p.field_units.get(fname, "")))
                self._primvars[f"{p.name}\x00{fname}"] = pv
            cf = p.color_field()
            dc = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            base = np.asarray(p.color, float) if cf is None else np.full(3, 0.5)
            dc.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, base))] * max(len(V), 1)))
            if cf is not None:
                self._colors[p.name] = dc
                mp.CreateAttribute("pinneapple:colorBy", Sdf.ValueTypeNames.String).Set(str(cf))
            if p.position is not None:
                tr = mesh.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            else:
                tr = None
            sc = mesh.AddScaleOp(UsdGeom.XformOp.PrecisionFloat) if p.scale is not None else None
            if tr is not None or sc is not None:
                self._ops[p.name] = (tr, sc)
        for s in self.track.sensors:
            name = _safe("sensor_" + s.name)
            sph = UsdGeom.Sphere.Define(self.stage, f"/World/{name}")
            sph.CreateRadiusAttr(0.01)
            sph.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*[float(x) for x in np.asarray(s.position, float)[:3]]))
            sp = sph.GetPrim()
            sp.CreateAttribute("pinneapple:sensor_id", Sdf.ValueTypeNames.String).Set(str(s.name))
            sp.CreateAttribute("pinneapple:unit", Sdf.ValueTypeNames.String).Set(str(s.unit))
            val = sp.CreateAttribute("pinneapple:value", Sdf.ValueTypeNames.Double)
            if s.envelope is not None:
                sp.CreateAttribute("pinneapple:envelope", Sdf.ValueTypeNames.Double2).Set(
                    Gf.Vec2d(float(s.envelope[0]), float(s.envelope[1])))
            dc = sph.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
            dc.Set(Vt.Vec3fArray([Gf.Vec3f(*_IN_ENVELOPE)]))
            self._sensors[s.name] = (val, dc)

    # -- streaming
    def _colour_range(self, part: ForecastPart, name: str) -> tuple[float, float]:
        a = np.asarray(part.fields[name], float)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        return (lo, hi if hi > lo else lo + 1.0)

    def push(self, k: int) -> float:
        """Author step ``k`` and save; returns its time code."""
        from pxr import Gf, Vt
        if not 0 <= k < self.track.n_steps:
            raise IndexError(f"step {k} outside the forecast (0..{self.track.n_steps - 1})")
        tc = float(k)
        for p in self.track.parts:
            for fname, arr in p.fields.items():
                a = np.asarray(arr, float)
                row = a if a.ndim == 1 else a[k]
                pv = self._primvars[f"{p.name}\x00{fname}"]
                pv.Set(Vt.FloatArray([float(x) for x in row]), tc)
            cf = p.color_field()
            if cf is not None:
                a = np.asarray(p.fields[cf], float)
                row = a if a.ndim == 1 else a[k]
                lo, hi = self._colour_range(p, cf)
                rgb = field_colors(row, p.cmap, (lo, hi))
                self._colors[p.name].Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in rgb]), tc)
            tr, sc = self._ops.get(p.name, (None, None))
            if tr is not None:
                pos = np.asarray(p.position, float).reshape(-1, 3)
                tr.Set(Gf.Vec3d(*map(float, pos[k if len(pos) > 1 else 0])), tc)
            if sc is not None:
                s = np.asarray(p.scale, float)
                v = s if (s.ndim == 0 or s.shape == (3,)) else s[k]   # a vector of 3 is constant, not per step
                sc.Set(Gf.Vec3f(*[float(x) for x in np.broadcast_to(v, (3,))]), tc)
        for s in self.track.sensors:
            val, dc = self._sensors[s.name]
            x = float(np.asarray(s.series, float)[k])
            val.Set(x, tc)
            inside = True if s.envelope is None else bool(s.envelope[0] <= x <= s.envelope[1])
            dc.Set(Vt.Vec3fArray([Gf.Vec3f(*(_IN_ENVELOPE if inside else _OUT_OF_ENVELOPE))]), tc)
        self.stage.SetEndTimeCode(k)
        self.pushed += 1
        if self.pushed % self.save_every == 0:
            self.stage.GetRootLayer().Save()
        return tc

    def stream(self, every: int = 1) -> "LiveLinkStreamer":
        """Push every step (or every ``every``-th); returns ``self``."""
        for k in range(0, self.track.n_steps, max(1, int(every))):
            self.push(k)
        return self

    def close(self) -> str:
        """Save and return the path."""
        self.stage.GetRootLayer().Save()
        return self.path


def write_forecast_stage(track: ForecastTrack, path: str, *, meters_per_unit: float = 1.0) -> str:
    """Write the whole forecast as one stage (``.usda`` text or ``.usdc`` binary) and return the path."""
    link = LiveLinkStreamer(path, track, meters_per_unit=meters_per_unit)
    link.stream()
    return link.close()


# ---------------------------------------------------------------------------------------------- reading back
@dataclass
class ForecastPlayback:
    """A stage read back with ``pxr`` alone: the time samples of every prim, step by step.

    ``parts``: name -> ``{"translate": (T, 3) | None, "scale": (T, 3) | None, "fields": {name: (T, V)},
    "colors": (T, V, 3) | None}``; ``sensors``: name -> ``{"value": (T,), "position": (3,), "unit", "envelope"}``;
    ``times``: the real time of each step, from the layer metadata (the time codes are step indices).
    """
    path: str
    times: np.ndarray
    parts: dict[str, dict[str, Any]]
    sensors: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    time_codes: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(len(self.time_codes))

    def frame(self, k: int) -> dict[str, Any]:
        """Everything the stage holds at step ``k``: poses, the field of every part and the sensor values."""
        out: dict[str, Any] = {"step": int(k), "time": float(self.times[k]), "parts": {}, "sensors": {}}
        for name, d in self.parts.items():
            fr = {"fields": {f: a[k] for f, a in d["fields"].items()}}
            for what in ("translate", "scale"):
                if d.get(what) is not None:
                    fr[what] = d[what][k]
            if d.get("colors") is not None:
                fr["color"] = d["colors"][k]
            out["parts"][name] = fr
        for name, d in self.sensors.items():
            out["sensors"][name] = {"value": float(d["value"][k]), "unit": d.get("unit", ""),
                                    "in_envelope": None if d.get("envelope") is None
                                    else bool(d["envelope"][0] <= d["value"][k] <= d["envelope"][1])}
        return out

    def frames(self) -> list[dict[str, Any]]:
        return [self.frame(k) for k in range(self.n_steps)]

    def check(self) -> dict[str, Any]:
        """What the file must satisfy to be played: every series has one value per time code, all finite, the
        ``displayColor`` of every part has one colour per vertex, and the sensors are inside their sphere."""
        problems: list[str] = []
        for name, d in self.parts.items():
            for f, a in d["fields"].items():
                if a.shape[0] != self.n_steps:
                    problems.append(f"{name}.{f}: {a.shape[0]} samples for {self.n_steps} time codes")
                elif not np.isfinite(a).all():
                    problems.append(f"{name}.{f}: non-finite values")
            for what in ("translate", "scale"):
                if d.get(what) is not None and (d[what].shape[0] != self.n_steps
                                                or not np.isfinite(d[what]).all()):
                    problems.append(f"{name}.{what}: bad time samples {d[what].shape}")
        for name, d in self.sensors.items():
            v = d["value"]
            if v.shape[0] != self.n_steps:
                problems.append(f"sensor {name}: {v.shape[0]} samples for {self.n_steps} time codes")
            elif not np.isfinite(v).all():
                problems.append(f"sensor {name}: non-finite values")
        return {"ok": not problems, "problems": problems, "steps": self.n_steps, "parts": sorted(self.parts),
                "sensors": sorted(self.sensors), "format": self.metadata.get("pinneapple_format", "")}


def read_forecast_stage(path: str) -> ForecastPlayback:
    """Re-open a stage written by :class:`LiveLinkStreamer` and read every time sample out of it, with ``pxr``."""
    require_omniverse()
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(os.fspath(path))
    if stage is None:
        raise ValueError(f"{path}: does not open as a USD stage")
    md = dict(stage.GetRootLayer().customLayerData or {})
    times_md = np.asarray(list(md.get("time_values", [])), float)
    parts: dict[str, dict[str, Any]] = {}
    sensors: dict[str, dict[str, Any]] = {}
    codes: set[float] = set()
    for prim in stage.Traverse():
        path_ = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh) and prim.GetAttribute("pinneapple:name").HasAuthoredValue():
            name = prim.GetAttribute("pinneapple:name").Get()
            fields: dict[str, np.ndarray] = {}
            api = UsdGeom.PrimvarsAPI(prim)
            for pv in api.GetPrimvars():
                pvn = str(pv.GetPrimvarName())
                if pvn == "displayColor":
                    continue
                samples = _samples(pv.GetAttr())
                if samples is not None:
                    fields[pvn] = np.asarray(samples, float)
                    codes.update(pv.GetAttr().GetTimeSamples())
            colors = None
            dc = UsdGeom.Gprim(prim).GetDisplayColorAttr()
            dsamp = _samples(dc)
            if dsamp is not None:
                colors = np.asarray(dsamp, float)
                codes.update(dc.GetTimeSamples())
            out: dict[str, Any] = {"fields": fields, "colors": colors,
                                   "translate": _xform_samples(prim, "xformOp:translate"),
                                   "scale": _xform_samples(prim, "xformOp:scale"),
                                   "path": path_}
            parts[name] = out
        elif prim.IsA(UsdGeom.Sphere) and prim.GetAttribute("pinneapple:sensor_id").HasAuthoredValue():
            name = prim.GetAttribute("pinneapple:sensor_id").Get()
            val = prim.GetAttribute("pinneapple:value")
            samples = _samples(val)
            if samples is not None:
                codes.update(val.GetTimeSamples())
            env = prim.GetAttribute("pinneapple:envelope").Get()
            pos = prim.GetAttribute("xformOp:translate").Get()
            sensors[name] = {"value": np.asarray(samples if samples is not None else [], float),
                             "position": np.asarray(pos, float) if pos is not None else np.zeros(3),
                             "unit": str(prim.GetAttribute("pinneapple:unit").Get() or ""),
                             "envelope": None if env is None else (float(env[0]), float(env[1])),
                             "path": path_}
    tc = np.asarray(sorted(codes), float)
    times = times_md[: len(tc)] if times_md.size >= len(tc) else tc
    return ForecastPlayback(os.fspath(path), times, parts, sensors, md, tc)


def _samples(attr) -> list[Any] | None:
    """The values of ``attr`` at every time code it is keyed at (``None`` when it has no samples)."""
    from pxr import Usd
    codes = attr.GetTimeSamples()
    if not codes:
        return None
    return [attr.Get(Usd.TimeCode(t)) for t in codes]


def _xform_samples(prim, name: str) -> np.ndarray | None:
    from pxr import Usd
    attr = prim.GetAttribute(name)
    if not attr or not attr.GetTimeSamples():
        return None
    return np.asarray([list(attr.Get(Usd.TimeCode(t))) for t in attr.GetTimeSamples()], float)


def play_stage(path: str, *, echo=print, check: bool = True) -> list[str]:
    """Play a stage back step by step: one line per step with the pose, the tracked field and the sensors.

    This is the reader the demo uses -- it opens the file with ``pxr`` and nothing else, so what it prints is what an
    Omniverse viewport would show on the timeline.
    """
    pb = read_forecast_stage(path)
    if check:
        r = pb.check()
        echo(f"{os.path.basename(path)}: {r['steps']} steps, {len(pb.parts)} part(s), {len(pb.sensors)} sensor(s)"
             + ("" if r["ok"] else f" -- PROBLEMS: {'; '.join(r['problems'])}"))
    lines = []
    for k in range(pb.n_steps):
        f = pb.frame(k)
        bits = [f"[{k + 1}/{pb.n_steps}] t={f['time']:.4g}"]
        for name, d in f["parts"].items():
            if "translate" in d:
                bits.append(f"{name} at ({d['translate'][0]:.3g}, {d['translate'][1]:.3g}, {d['translate'][2]:.3g})"
                            + (f" scale {np.mean(d['scale']):.3g}" if "scale" in d else ""))
        for name, d in f["sensors"].items():
            flag = "" if d["in_envelope"] is None else (" ok" if d["in_envelope"] else " OUT")
            bits.append(f"{name}={d['value']:.4g}{(' ' + d['unit']) if d['unit'] else ''}{flag}")
        line = " | ".join(bits)
        lines.append(line)
        if echo is not None:
            echo(line)
    return lines


# ---------------------------------------------------------------------------------------------- the Kit half
def kit_available() -> bool:
    """Whether an Omniverse Kit session is importable (``omni.usd``): False on a plain ``usd-core`` install."""
    try:
        import omni.usd  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any failure of a Kit import means there is no Kit here
        return False


class KitLiveLink:
    """The Kit half of the live link: apply the streamed time samples to the stage open in an Omniverse Kit app.

    Needs Kit (``omni.usd``, ``omni.kit.app``), which is installed with the Omniverse launcher / USD Composer and is
    not a pip package: constructing this without Kit raises :class:`ImportError` naming the extra that installs the
    OpenUSD side and pointing at :func:`kit_livelink_script`, which writes the script that runs inside Kit.
    """

    def __init__(self, stage_path: str, *, layer_path: str | None = None, fps: float = 1.0, live: bool = True):
        if not kit_available():
            raise ImportError(KIT_HINT)
        import omni.kit.app  # noqa: F401  (pragma: no cover - Kit only)
        import omni.usd  # noqa: F401
        self.stage_path = stage_path
        self.layer_path = layer_path
        self.fps = float(fps)
        self.live = bool(live)


def kit_livelink_script(stage_path: str = "/World/Twin", layer_path: str = "forecast.usda",
                        fps: float = 1.0) -> str:
    """The script to run inside Omniverse Kit: opens ``layer_path`` under ``stage_path``, plays its time codes and
    (``live=True``) re-reads the layer every frame, so a writer streaming samples into the same file drives the live
    stage. Declared, not executed here: it needs Kit and an RTX machine."""
    return f'''"""Apply a PINNeAPPle forecast streamed to a USD layer inside Omniverse Kit (issue #411).

Needs an Omniverse Kit application (USD Composer, or a custom Kit app) with an RTX GPU. Run it from the Script
Editor or as a `--exec` script; it does not need the PINNeAPPle package itself, only `pxr` (shipped with Kit).
"""
import omni.kit.app
import omni.usd
from pxr import Sdf, Usd, UsdGeom

LAYER = {layer_path!r}          # written by pinneapple_tools.nvidia.livelink.LiveLinkStreamer
TARGET = {stage_path!r}         # where the forecast is attached in the open stage
FPS = {fps!r}                   # time codes per second (one time code per forecast step)


def open_live_layer():
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    root = stage.GetRootLayer()
    sub = Sdf.Layer.FindOrOpen(LAYER)
    root.subLayerPaths.append(sub.identifier)
    prim = stage.GetPrimAtPath(TARGET)
    if not prim:
        prim = UsdGeom.Xform.Define(stage, TARGET).GetPrim()
    ctx.set_current_edit_target(root)


async def _play(_):
    import omni.kit.app
    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()
    start, end = int(stage.GetStartTimeCode() or 0), int(stage.GetEndTimeCode() or 0)
    if end <= start:                       # nothing streamed yet: wait for the writer
        return
    for code in range(start, end + 1):
        omni.usd.get_context().get_stage().SetTimeCode(code)   # the stage evaluates every time sample here
        await app.next_update_async()


_open = open_live_layer
omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(_play, name="pinneapple live link")
'''


def _safe(name: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name)) or "prim"
    return f"p_{s}" if s[0].isdigit() else s


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
