"""Time animation of a :class:`~.scene.Scene`: rigid bodies moving along a trajectory, per-frame arrows, trails,
per-vertex fields and head-up numbers, for the browser viewer (:mod:`.web`) and Blender (:mod:`.blender`).

    from pinneapple_tools.visualization.studio import Scene, Surface
    from pinneapple_tools.visualization.studio.animation import Animation
    sc = Scene([Surface("runway", Vr, Fr, "concrete"), Surface("body", Vb, Fb, "paint", group="body")])
    anim = Animation(t)                                   # times (T,) in seconds
    anim.add_track("body", positions, quaternions)        # (T, 3), (T, 4) [w, x, y, z]; body vertices in body frame
    anim.add_vectors("forces", origins, vectors, values=mag, attach="body", label="Force [N]")
    anim.add_trail("path", positions)
    anim.add_hud("V", speed, "Speed", "m/s")
    sc.animation = anim.resample(200)                     # scene.json stays small
    web_viewer(sc, "viewer/")                             # play / pause, time slider, follow camera

Every array is in the scene's axes (``Scene.axes``, e.g. ``"z_up"``). A track moves the surfaces whose
``Surface.group`` equals its group: their vertices are given in the body frame and sit at frame k at
``p_world = R(q_k) v + positions[k]``. Quaternions are scalar first, ``[w, x, y, z]``, rotating body to scene axes.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .compose import default_range, stops


# ---------------------------------------------------------------------- rotations
def matrix_from_quat(q) -> np.ndarray:
    """Rotation matrix (3, 3) (or (..., 3, 3)) of unit quaternion(s) ``[w, x, y, z]`` (normalised here)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_from_matrix(R) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` (w >= 0) of a rotation matrix (3, 3) or a stack (..., 3, 3)
    (Shepperd's method: the largest of the four squared components is taken first, so it is stable)."""
    R = np.asarray(R, float)
    shp = R.shape[:-2]
    M = R.reshape(-1, 3, 3)
    out = np.empty((len(M), 4))
    for i, m in enumerate(M):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        k = int(np.argmax([tr, m[0, 0], m[1, 1], m[2, 2]]))
        if k == 0:
            s = 2 * np.sqrt(max(1 + tr, 0.0))
            q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif k == 1:
            s = 2 * np.sqrt(max(1 + m[0, 0] - m[1, 1] - m[2, 2], 0.0))
            q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif k == 2:
            s = 2 * np.sqrt(max(1 + m[1, 1] - m[0, 0] - m[2, 2], 0.0))
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(max(1 + m[2, 2] - m[0, 0] - m[1, 1], 0.0))
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        q = np.asarray(q)
        q /= np.linalg.norm(q)
        out[i] = -q if q[0] < 0 else q
    return out.reshape(shp + (4,))


def slerp(q0, q1, f: float) -> np.ndarray:
    """Spherical interpolation between unit quaternions ``[w, x, y, z]`` (shortest arc), f in [0, 1]."""
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:                                        # nearly equal: linear, then normalise
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(min(1.0, d))
    return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)


def _continuous(q: np.ndarray) -> np.ndarray:
    """Normalise and flip signs so consecutive quaternions lie in the same hemisphere (no long-way interpolation)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return q


def _decimals_of(fmt: str) -> int:
    m = re.search(r"\.(\d+)[fe%]", fmt)
    return int(m.group(1)) if m else 2


# ---------------------------------------------------------------------- the animation
@dataclass
class Animation:
    """Per-frame data on top of a static :class:`~.scene.Scene`; see the module docstring."""
    times: np.ndarray
    tracks: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)   # group -> {"positions", "quaternions"}
    vectors: list[dict[str, Any]] = field(default_factory=list)
    trails: list[dict[str, Any]] = field(default_factory=list)
    surface_fields: dict[str, dict[str, Any]] = field(default_factory=dict)  # surface name -> {"values", ...}
    hud: list[dict[str, Any]] = field(default_factory=list)
    camera: str = "fixed"                       # "fixed" | "follow" (the viewer starts with the chase camera on)
    follow_group: str | None = None          # default: the first track
    follow_offset: Sequence[float] | None = None   # camera position relative to the body (None: from its size)
    follow_frame: str = "body"                  # "body": the offset turns with the body; "world": fixed direction

    def __post_init__(self):
        self.times = np.asarray(self.times, float).ravel()
        if len(self.times) < 1 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be a non-empty, strictly increasing sequence")
        if self.camera not in ("fixed", "follow"):
            raise ValueError("camera must be 'fixed' or 'follow'")
        if self.follow_frame not in ("body", "world"):
            raise ValueError("follow_frame must be 'body' or 'world'")

    @property
    def n_frames(self) -> int:
        return len(self.times)

    def _check(self, a, shape: tuple[int, ...], what: str) -> np.ndarray:
        a = np.asarray(a, float)
        ok = (a.ndim == 1 + len(shape) and a.shape[0] == self.n_frames
              and all(want in (-1, got) for got, want in zip(a.shape[1:], shape, strict=False)))
        if not ok:
            raise ValueError(f"{what}: expected shape ({self.n_frames}, {', '.join('M' if s == -1 else str(s) for s in shape)}), got {a.shape}")
        return a

    # ------------------------------------------------------------------ building
    def add_track(self, group: str, positions, quaternions=None) -> Animation:
        """Rigid motion of the surfaces with ``Surface.group == group``: positions (T, 3), quaternions (T, 4)
        ``[w, x, y, z]`` (None: no rotation)."""
        p = self._check(positions, (3,), "positions")
        q = (np.tile([1.0, 0, 0, 0], (self.n_frames, 1)) if quaternions is None
             else self._check(quaternions, (4,), "quaternions"))
        self.tracks[group] = {"positions": p, "quaternions": _continuous(q.copy())}
        if self.follow_group is None:
            self.follow_group = group
        return self

    def add_vectors(self, name: str, origins, vectors, values=None, label: str = "",
                    color_range: tuple[float, float] | None = None, attach: str | None = None,
                    color: str = "#f97316", cmap: str = "jet") -> Animation:
        """Arrows per frame: origins and vectors (T, M, 3) already scaled to display length; ``values`` (T, M)
        colour them on ``cmap`` ("jet", or the diverging "coolwarm" / "bwr"; range: ``color_range``, else min..max
        over all frames for jet and symmetric about zero for a diverging map), otherwise ``color``.
        ``attach=group``: given in that track's body frame, they move with it."""
        stops(cmap)
        o = self._check(origins, (-1, 3), "origins")
        v = self._check(vectors, (-1, 3), "vectors")
        if o.shape != v.shape:
            raise ValueError(f"origins {o.shape} and vectors {v.shape} differ")
        s = None if values is None else np.asarray(values, float).reshape(o.shape[:2])
        if attach is not None and attach not in self.tracks:
            raise ValueError(f"attach={attach!r}: add_track({attach!r}, ...) first")
        rng = color_range if color_range is not None or s is None else default_range(s, cmap)
        self.vectors.append({"name": name, "origins": o, "vectors": v, "values": s, "label": label,
                             "range": None if rng is None else tuple(map(float, rng)), "attach": attach,
                             "color": color, "cmap": cmap})
        return self

    def add_trail(self, name: str, points, color: str = "#f59e0b", values=None, label: str = "",
                  color_range: tuple[float, float] | None = None, cmap: str = "jet", radius: float | None = None,
                  alpha: float = 0.8, trail_seconds: float | None = 8.0, fade: bool = True) -> Animation:
        """A path (T, 3) drawn behind the body up to the current time (e.g. the track's positions): only its last
        ``trail_seconds`` (None: all of it), its opacity falling from ``alpha`` at the current point to 0 at the
        tail when ``fade``. ``values`` (T,) colour it point by point on ``cmap`` (with a colour bar titled
        ``label``), otherwise it is ``color``. ``radius``: tube radius in Blender (default 0.003 x the tracked
        body's size)."""
        stops(cmap)
        P = self._check(points, (3,), "trail points")
        v = None if values is None else np.asarray(values, float).ravel()
        if v is not None and len(v) != self.n_frames:
            raise ValueError(f"trail values: expected {self.n_frames}, got {len(v)}")
        rng = None if v is None else tuple(map(float, color_range or default_range(v, cmap)))
        self.trails.append({"name": name, "points": P, "color": color, "values": v, "label": label, "range": rng,
                            "cmap": cmap, "radius": None if radius is None else float(radius), "alpha": float(alpha),
                            "seconds": None if trail_seconds is None else float(trail_seconds), "fade": bool(fade)})
        return self

    def add_lines(self, name: str, points, color: str = "#f59e0b", **kw) -> Animation:
        """Alias of :meth:`add_trail`: one point per frame, shown progressively."""
        return self.add_trail(name, points, color, **kw)

    def add_surface_field(self, surface: str, values, label: str = "",
                          color_range: tuple[float, float] | None = None, cmap: str = "jet") -> Animation:
        """A per-vertex scalar per frame (T, n_vertices) on the Surface named ``surface``, on ``cmap``."""
        stops(cmap)
        a = np.asarray(values, float)
        if a.ndim != 2 or a.shape[0] != self.n_frames:
            raise ValueError(f"values: expected ({self.n_frames}, n_vertices), got {a.shape}")
        rng = color_range or default_range(a, cmap)
        self.surface_fields[surface] = {"values": a, "label": label, "range": tuple(map(float, rng)), "cmap": cmap}
        return self

    def add_hud(self, key: str, values, label: str = "", unit: str = "", fmt: str = "{:.1f}") -> Animation:
        """A number per frame (T,) shown in the overlay as ``label: value unit``."""
        v = np.asarray(values, float).ravel()
        if len(v) != self.n_frames:
            raise ValueError(f"hud {key!r}: expected {self.n_frames} values, got {len(v)}")
        self.hud = [h for h in self.hud if h["key"] != key] + [
            {"key": key, "values": v, "label": label or key, "unit": unit, "fmt": fmt}]
        return self

    # ------------------------------------------------------------------ poses
    def _track(self, group: str | None):
        g = group if group is not None else (self.follow_group or next(iter(self.tracks), None))
        if g not in self.tracks:
            raise KeyError(f"no track {g!r}; tracks: {list(self.tracks)}")
        return self.tracks[g]

    def pose(self, group: str | None = None, k: int = 0) -> np.ndarray:
        """4x4 body-to-scene transform of ``group`` at frame k."""
        tr = self._track(group)
        M = np.eye(4)
        M[:3, :3] = matrix_from_quat(tr["quaternions"][k])
        M[:3, 3] = tr["positions"][k]
        return M

    def frame(self, k: int) -> dict[str, np.ndarray]:
        """The 4x4 transform of every track at frame k."""
        return {g: self.pose(g, k) for g in self.tracks}

    def at(self, t: float, group: str | None = None) -> np.ndarray:
        """4x4 transform of ``group`` at time t: positions linear, rotations slerp (clamped to the time range)."""
        tr = self._track(group)
        f = self._fractional(t)
        k = min(int(np.floor(f)), self.n_frames - 1)
        a = f - k
        M = np.eye(4)
        if k >= self.n_frames - 1 or a == 0:
            return self.pose(group, k)
        M[:3, :3] = matrix_from_quat(slerp(tr["quaternions"][k], tr["quaternions"][k + 1], a))
        M[:3, 3] = (1 - a) * tr["positions"][k] + a * tr["positions"][k + 1]
        return M

    def _fractional(self, t: float) -> float:
        return float(np.interp(t, self.times, np.arange(self.n_frames)))

    def transform(self, group: str, k: int, points) -> np.ndarray:
        """Body-frame points (N, 3) of ``group`` in scene coordinates at frame k."""
        M = self.pose(group, k)
        return np.asarray(points, float).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]

    def vectors_world(self, i: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Origins and vectors (M, 3) of vector set i at frame k in scene coordinates."""
        vs = self.vectors[i]
        o, v = vs["origins"][k], vs["vectors"][k]
        if vs["attach"]:
            M = self.pose(vs["attach"], k)
            o, v = o @ M[:3, :3].T + M[:3, 3], v @ M[:3, :3].T
        return o, v

    def trail_window(self, i: int, k: int, seconds: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the points of trail i shown at frame k (its last ``trail_seconds``, or ``seconds``) and
        their opacity factor in [0, 1] (1 at frame k, falling linearly to 0 at the tail end when the trail fades)."""
        tr = self.trails[i]
        sec = seconds if seconds is not None else tr.get("seconds")
        t = self.times
        j0 = 0 if sec is None else int(np.searchsorted(t, t[k] - sec, side="left"))
        idx = np.arange(min(j0, k), k + 1)
        if not tr.get("fade", False) or len(idx) < 2:
            return idx, np.ones(len(idx))
        span = sec if sec is not None else (t[k] - t[idx[0]]) or 1.0
        return idx, np.clip(1.0 - (t[k] - t[idx]) / span, 0.0, 1.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Extent of the track and trail points (not the bodies themselves)."""
        P = [tr["positions"] for tr in self.tracks.values()] + [t["points"] for t in self.trails]
        if not P:
            return np.zeros(3), np.zeros(3)
        A = np.concatenate(P)
        return A.min(0), A.max(0)

    # ------------------------------------------------------------------ size
    def resample(self, n_frames: int) -> Animation:
        """The same animation on ``n_frames`` evenly spaced times (positions, vectors, fields and HUD values linear,
        rotations slerp). Use it to keep scene.json small, or to get a fixed frame rate for a video."""
        n = int(n_frames)
        if n < 1:
            raise ValueError("n_frames must be >= 1")
        tn = np.linspace(self.times[0], self.times[-1], n) if n > 1 else self.times[:1].copy()
        f = np.interp(tn, self.times, np.arange(self.n_frames))
        k0 = np.minimum(np.floor(f).astype(int), self.n_frames - 1)
        k1 = np.minimum(k0 + 1, self.n_frames - 1)
        a = f - k0

        def lerp(A):
            A = np.asarray(A, float)
            w = a.reshape((-1,) + (1,) * (A.ndim - 1))
            return (1 - w) * A[k0] + w * A[k1]

        out = Animation(tn, camera=self.camera, follow_group=self.follow_group, follow_offset=self.follow_offset,
                        follow_frame=self.follow_frame)
        for g, tr in self.tracks.items():
            Q = tr["quaternions"]
            q = np.array([slerp(Q[i], Q[j], w) for i, j, w in zip(k0, k1, a, strict=False)])
            out.tracks[g] = {"positions": lerp(tr["positions"]), "quaternions": _continuous(q)}
        for vs in self.vectors:
            out.vectors.append({**vs, "origins": lerp(vs["origins"]), "vectors": lerp(vs["vectors"]),
                                "values": None if vs["values"] is None else lerp(vs["values"])})
        for tr in self.trails:
            out.trails.append({**tr, "points": lerp(tr["points"]),
                               "values": None if tr.get("values") is None else lerp(tr["values"])})
        for s, sf in self.surface_fields.items():
            out.surface_fields[s] = {**sf, "values": lerp(sf["values"])}
        for h in self.hud:
            out.hud.append({**h, "values": lerp(h["values"])})
        return out

    # ------------------------------------------------------------------ export
    def to_dict(self, decimals: int = 4, surfaces: Sequence[Any] | None = None) -> dict[str, Any]:
        """Compact JSON-able dict (floats rounded to ``decimals``; quaternions to 6). ``surfaces``: the scene's
        surfaces, to record which glTF nodes (indices, in order) each track moves and where each field goes."""
        r = lambda a, nd=decimals: np.round(np.asarray(a, float), nd).tolist()  # noqa: E731

        def finite(a, nd=decimals):
            a = np.round(np.asarray(a, float), nd)
            return [[None if not np.isfinite(x) else float(x) for x in row] for row in a] if a.ndim == 2 else \
                [None if not np.isfinite(x) else float(x) for x in a]

        names = [s.name for s in surfaces] if surfaces is not None else None
        groups = [getattr(s, "group", "") for s in surfaces] if surfaces is not None else None
        d: dict[str, Any] = {
            "times": r(self.times, 6),
            "tracks": [{"group": g, "positions": r(tr["positions"]), "quaternions": r(tr["quaternions"], 6),
                        **({"surfaces": [i for i, gg in enumerate(groups) if gg == g]} if groups is not None else {})}
                       for g, tr in self.tracks.items()],
            "vectors": [{"name": vs["name"], "label": vs["label"], "attach": vs["attach"], "color": vs["color"],
                         "cmap": vs.get("cmap", "jet"),
                         "range": None if vs["range"] is None else list(vs["range"]),
                         "origins": r(vs["origins"]), "vectors": r(vs["vectors"]),
                         "values": None if vs["values"] is None else finite(vs["values"].reshape(self.n_frames, -1))}
                        for vs in self.vectors],
            "trails": [{"name": t["name"], "color": t["color"], "points": r(t["points"]),
                        "values": None if t.get("values") is None else finite(t["values"]), "label": t.get("label", ""),
                        "range": None if t.get("range") is None else list(t["range"]), "cmap": t.get("cmap", "jet"),
                        "radius": t.get("radius"), "alpha": t.get("alpha", 1.0), "seconds": t.get("seconds"),
                        "fade": t.get("fade", False)} for t in self.trails],
            "fields": [{"surface": s, "label": sf["label"], "range": list(sf["range"]), "cmap": sf.get("cmap", "jet"),
                        **({"index": names.index(s)} if names is not None and s in names else {}),
                        "values": finite(sf["values"])} for s, sf in self.surface_fields.items()],
            "hud": [{"key": h["key"], "label": h["label"], "unit": h["unit"], "fmt": h["fmt"],
                     "nd": _decimals_of(h["fmt"]), "values": finite(h["values"], 6)} for h in self.hud],
            "camera": self.camera, "follow_group": self.follow_group,
            "follow_offset": None if self.follow_offset is None else r(self.follow_offset),
            "follow_frame": self.follow_frame,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Animation:
        """Inverse of :meth:`to_dict` (up to the rounding)."""
        a = cls(d["times"], camera=d.get("camera", "fixed"), follow_group=d.get("follow_group"),
                follow_offset=d.get("follow_offset"), follow_frame=d.get("follow_frame", "body"))
        for tr in d.get("tracks", []):
            a.tracks[tr["group"]] = {"positions": np.asarray(tr["positions"], float),
                                     "quaternions": np.asarray(tr["quaternions"], float)}
        nan = lambda x: None if x is None else np.asarray(x, dtype=float)  # noqa: E731  (None entries -> NaN)
        for vs in d.get("vectors", []):
            a.vectors.append({"name": vs["name"], "label": vs.get("label", ""), "attach": vs.get("attach"),
                              "color": vs.get("color", "#f97316"), "cmap": vs.get("cmap", "jet"),
                              "range": None if vs.get("range") is None else tuple(vs["range"]),
                              "origins": np.asarray(vs["origins"], float), "vectors": np.asarray(vs["vectors"], float),
                              "values": nan(vs.get("values"))})
        for t in d.get("trails", []):
            a.trails.append({"name": t["name"], "color": t.get("color", "#f59e0b"),
                             "points": np.asarray(t["points"], float), "values": nan(t.get("values")),
                             "label": t.get("label", ""), "range": None if t.get("range") is None else tuple(t["range"]),
                             "cmap": t.get("cmap", "jet"), "radius": t.get("radius"), "alpha": t.get("alpha", 1.0),
                             "seconds": t.get("seconds"), "fade": t.get("fade", False)})
        for f in d.get("fields", []):
            a.surface_fields[f["surface"]] = {"values": nan(f["values"]), "label": f.get("label", ""),
                                              "range": tuple(f["range"]), "cmap": f.get("cmap", "jet")}
        for h in d.get("hud", []):
            a.hud.append({"key": h["key"], "label": h["label"], "unit": h.get("unit", ""), "fmt": h.get("fmt", "{:.1f}"),
                          "values": nan(h["values"])})
        return a

    def hud_lines(self, k: int, time_fmt: str = "t = {:.2f} s", keys: Sequence[str] | None = None) -> list[str]:
        """The overlay text of frame k: the time, then ``label: value unit`` per HUD entry (only ``keys``, in that
        order, when given)."""
        out = [time_fmt.format(self.times[k])]
        hs = self.hud if keys is None else [h for key in keys for h in self.hud if h["key"] == key]
        for h in hs:
            v = h["values"][k]
            out.append(f"{h['label']}: {h['fmt'].format(v) if np.isfinite(v) else '-'}{(' ' + h['unit']) if h['unit'] else ''}")
        return out
