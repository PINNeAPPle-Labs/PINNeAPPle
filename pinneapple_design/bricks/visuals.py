"""Pictures of the loop: the evolution as a timeline, the twin3d web viewer, GIF/MP4 frames, Blender renders and
building instructions.

A :class:`Frame` is a list of placed bricks (pose, colour, optional stud utilisation) plus red markers on joints that
broke. :func:`evolution_timeline` builds the frames of a search: for each generation, its best build at rest
coloured by stud utilisation (the load path), then one rejected candidate of that generation replayed by Newton
after the joints the statics check says fail are released. The same frames go to

* :func:`timeline_scene` -> the twin3d three.js viewer (time slider, play, field = stud utilisation, captions),
* :func:`render_frames` / :func:`write_gif` / :func:`write_mp4` (software renderer, no GPU),
* and the final build to :func:`render_blender` / :func:`turntable_blender` (Cycles) and
  :func:`write_instructions` (step-by-step layers, part list, an LDraw file with steps).
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .assembly import Assembly, Brick
from .mesh import box_mesh, brick_mesh
from .parts import PALETTE, PITCH_MM, PLATE_MM, colour, part

UTIL_FIELD = "stud utilisation"
BROKEN_VALUE = 1.2              # utilisation shown on bricks whose joint broke (above every intact value)
_ID = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class Item:
    brick: Brick
    pos: np.ndarray             # centre [mm]
    quat: np.ndarray = field(default_factory=lambda: _ID.copy())
    value: float = float("nan")     # stud utilisation (nan: draw the brick colour)


@dataclass
class Marker:
    size: tuple[float, float]   # (sx, sy) of the joint patch [mm]
    pos: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: _ID.copy())


@dataclass
class Frame:
    items: list[Item]
    caption: str = ""
    short: str = ""
    kind: str = "build"         # build | replay | final
    generation: int = 0
    markers: list[Marker] = field(default_factory=list)
    scenery: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)    # static boxes (tables) [mm]


def _rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def build_frame(asm: Assembly, values: Sequence[float] | None = None, **kw) -> Frame:
    """A frame of ``asm`` at rest; ``values``: stud utilisation per brick (colours the bricks by load path)."""
    items = [Item(b, b.com_mm(), _ID.copy(), float(values[i]) if values is not None else float("nan"))
             for i, b in enumerate(asm.bricks)]
    return Frame(items, **kw)


def scenery(asms: Sequence[Assembly], thickness: float = 80.0, margin: float = 48.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Table blocks under the builds (their ground regions near the parts, or one block for unlimited ground), as
    tall as the drop to the floor in :mod:`.dynamics` by default."""
    los, his = zip(*(a.bounds_mm() for a in asms if len(a)), strict=True)
    lo, hi = np.min(los, 0) - margin, np.max(his, 0) + margin
    ground = asms[0].ground
    rects = [(lo[0], hi[0], lo[1], hi[1])] if ground.regions is None else []
    for r in ground.regions or []:
        x0, x1, y0, y1 = (v * PITCH_MM for v in r)
        x0, x1, y0, y1 = max(x0, lo[0]), min(x1, hi[0]), max(y0, lo[1]), min(y1, hi[1])
        if x1 > x0 and y1 > y0:
            rects.append((x0, x1, y0, y1))
    return [(np.array([x0, y0, -thickness]), np.array([x1, y1, 0.0])) for x0, x1, y0, y1 in rects]


def utilisation(asm: Assembly, loads=(), joint=None) -> np.ndarray:
    """Worst stud utilisation of each brick's joints (0 = no stud in tension, 1 = at the clutch limit)."""
    from .statics import analyse_assembly
    r = analyse_assembly(asm, loads, joint)
    if r.status == "mechanism":
        return np.zeros(len(asm))
    return np.clip(np.nan_to_num(r.body_utilisation), 0, BROKEN_VALUE)


def replay_frames(asm: Assembly, *, loads=(), joint=None, frames: int = 14, duration: float = 0.6,
                  caption: str = "", short: str = "", generation: int = 0) -> list[Frame]:
    """Progressive failure (statics) then a Newton replay with the failed joints released: one frame per sample,
    broken joints marked in red and their bricks at :data:`BROKEN_VALUE`."""
    from .dynamics import replay
    from .statics import progressive_failure
    rounds = progressive_failure(asm, loads, joint)
    broken = rounds[-1].broken
    rp = replay(asm, broken, duration=duration, frames=frames, loads=loads, joint=joint)
    conns = {(c.lower, c.upper): c for c in asm.connections()}
    hurt = {i for p in broken for i in p if i >= 0}
    out = []
    for k in range(len(rp.t)):
        items = [Item(b, rp.pos[k, i], rp.quat[k, i], BROKEN_VALUE if i in hurt else 0.0)
                 for i, b in enumerate(asm.bricks)]
        marks = []
        for lo, up in broken:
            c = conns.get((lo, up))
            if c is None:
                continue
            x0, x1, y0, y1 = (v * PITCH_MM for v in c.rect)
            centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, c.z * PLATE_MM])
            rel = centre - rp.pos[0, up]
            R = _rot(rp.quat[k, up])
            marks.append(Marker((x1 - x0, y1 - y0), rp.pos[k, up] + R @ rel, rp.quat[k, up]))
        out.append(Frame(items, caption, short, "replay", generation, marks))
    return out


def evolution_timeline(result, goal, *, generations: Sequence[int] | None = None, hold: int = 3,
                       replay: bool = True, frames_per_replay: int = 14, duration: float = 0.6,
                       final_hold: int = 6, metric: str | None = None) -> list[Frame]:
    """Frames of a :class:`~.search.SearchResult`: per generation the best build (coloured by stud utilisation) and,
    when ``replay``, one rejected candidate of that generation collapsing in Newton."""
    gens = list(range(len(result.generations))) if generations is None else list(generations)
    out: list[Frame] = []
    scen = scenery([c.asm for c in result.history if len(c.asm)])
    unit = "mm"
    for g in gens:
        best = result.best_of_generation(g)
        if best is None:
            pool = [result.history[i] for i in result.generations[g]]
            best = min(pool, key=lambda c: c.ev.violation)
        val = _metric(best.ev, metric)
        status = "stands" if best.ev.feasible else best.ev.status.replace("_", " ")
        short = f"gen {g}: {val} {unit} with {best.ev.n_bricks} parts, {status}" if val else f"gen {g}: {status}"
        u = utilisation(best.asm, goal.loads(best.asm), goal.joint)
        for _ in range(hold):
            out.append(build_frame(best.asm, u, caption=f"generation {g}, best: {best.ev.explanation}", short=short,
                                   kind="build", generation=g))
        if replay:
            fails = [c for c in result.failures_of_generation(g) if c.ev.status in ("joint_failure", "mechanism")]
            if fails:
                f = max(fails, key=lambda c: (c.ev.score, -c.ev.violation))
                why = f.ev.explanation
                out += replay_frames(f.asm, loads=goal.loads(f.asm), joint=goal.joint, frames=frames_per_replay,
                                     duration=duration, caption=f"generation {g}, rejected ({f.edit}): {why}",
                                     short=f"gen {g}: rejected, {f.ev.status.replace('_', ' ')}", generation=g)
    best = result.best
    for f in out:
        f.scenery = scen
    if best is not None:
        u = utilisation(best.asm, goal.loads(best.asm), goal.joint)
        val = _metric(best.ev, metric)
        for _ in range(final_hold):
            out.append(build_frame(best.asm, u, caption=f"best build: {best.ev.explanation}",
                                   short=f"best: {val} mm, {best.ev.n_bricks} parts" if val else "best build",
                                   kind="final", generation=gens[-1] if gens else 0, scenery=scen))
    return out


def _metric(ev, metric: str | None) -> str:
    key = metric or next((k for k in ("reach_mm", "height_mm") if k in ev.metrics), None)
    if key is None or key not in ev.metrics:
        return ""
    return f"{ev.metrics[key]:.0f}"


# ---------------------------------------------------------------------------------------------- meshes per frame
_MESH_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _local_mesh(b: Brick, segments: int = 10) -> tuple[np.ndarray, np.ndarray]:
    key = (b.part.name, b.rot % 2, segments)
    if key not in _MESH_CACHE:
        probe = Brick(b.part, 0, 0, 0, b.rot)
        _MESH_CACHE[key] = brick_mesh(probe, segments=segments, local=True)
    return _MESH_CACHE[key]


def _marker_mesh(size, thick: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = size
    return box_mesh((-sx / 2 - 0.3, -sy / 2 - 0.3, -thick / 2), (sx / 2 + 0.3, sy / 2 + 0.3, thick / 2))


def frame_bounds(frames: Sequence[Frame], margin: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    pts = [it.pos for f in frames for it in f.items] + [c for f in frames[:1] for box in f.scenery for c in box]
    r = max((float(np.linalg.norm(_local_mesh(it.brick)[0], axis=1).max()) for f in frames[:1] for it in f.items),
            default=10.0)
    P = np.array(pts)
    lo, hi = P.min(0) - r - margin, P.max(0) + r + margin
    lo[2] = max(lo[2], -margin)
    return lo, hi


# ---------------------------------------------------------------------------------------------- twin3d viewer
def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def timeline_scene(frames: Sequence[Frame], title: str = "self-modifying brick build", source: str = ""):
    """A twin3d :class:`~pinneapple_twin3d.Scene` (millimetres, z up) with one time step per frame: brick motion,
    visibility, the stud utilisation field and the captions. Export it with ``scene.export(folder)``."""
    from pinneapple_twin3d import Scene
    T = len(frames)
    sc = Scene(title, length_unit="mm", times=np.arange(T, dtype=float), time_unit="frame", source=source)
    need: dict[tuple, int] = {}
    for f in frames:
        cnt: dict[tuple, int] = {}
        for it in f.items:
            k = ("brick", it.brick.part.name, it.brick.rot % 2, it.brick.colour)
            cnt[k] = cnt.get(k, 0) + 1
        for m in f.markers:
            k = ("marker", round(m.size[0], 1), round(m.size[1], 1))
            cnt[k] = cnt.get(k, 0) + 1
        for k, v in cnt.items():
            need[k] = max(need.get(k, 0), v)
    for k, (lo, hi) in enumerate(frames[0].scenery if frames else []):
        V, F = box_mesh(lo, hi)
        sc.add_part(f"table_{k}", V, F, group="table", color=tuple(_lin((0.62, 0.55, 0.47))))
    pools: dict[tuple, list[dict]] = {}
    for k, n in need.items():
        for j in range(n):
            if k[0] == "brick":
                proto = Brick(part(k[1]), 0, 0, 0, k[2])
                V, F = _local_mesh(proto)
                rgb, group = _lin(colour(k[3]).rgb), "bricks"
            else:
                V, F = _marker_mesh((k[1], k[2]))
                rgb, group = _lin((0.86, 0.08, 0.08)), "broken joints"
            name = f"{k[0]}_{'_'.join(str(v) for v in k[1:])}_{j}"
            sc.add_part(name, V, F, group=group, color=tuple(rgb))
            pools.setdefault(k, []).append({"name": name, "nv": len(V), "pos": np.zeros((T, 3)),
                                            "quat": np.tile(_ID, (T, 1)), "vis": np.zeros(T),
                                            "val": np.zeros(T)})
    for t, f in enumerate(frames):
        used: dict[tuple, int] = {}
        for it in f.items:
            k = ("brick", it.brick.part.name, it.brick.rot % 2, it.brick.colour)
            slot = pools[k][used.get(k, 0)]
            used[k] = used.get(k, 0) + 1
            slot["pos"][t], slot["quat"][t], slot["vis"][t] = it.pos, it.quat, 1.0
            slot["val"][t] = 0.0 if np.isnan(it.value) else it.value
        for m in f.markers:
            k = ("marker", round(m.size[0], 1), round(m.size[1], 1))
            slot = pools[k][used.get(k, 0)]
            used[k] = used.get(k, 0) + 1
            slot["pos"][t], slot["quat"][t], slot["vis"][t] = m.pos, m.quat, 1.0
            slot["val"][t] = BROKEN_VALUE
    for slots in pools.values():
        for s in slots:
            vis = s["vis"]
            if vis.any():                      # park hidden slots on a visible pose so bounds stay tight
                first = int(np.argmax(vis))
                for t in range(T):
                    if not vis[t]:
                        s["pos"][t], s["quat"][t] = s["pos"][first], s["quat"][first]
            sc.add_motion(s["name"], s["pos"], s["quat"])
            sc.add_visibility(s["name"], vis)
            sc.add_field(s["name"], UTIL_FIELD, np.repeat(s["val"][:, None], s["nv"], 1), unit="1")
    sc.set_step_labels([f.caption for f in frames])
    return sc


# ---------------------------------------------------------------------------------------------- software frames
class _Body:
    def __init__(self, V, F, mat):
        self.vertices, self.faces, self._mat = V, F, mat

    def face_materials(self):
        return [self._mat] * len(self.faces)


class _Spec:
    def __init__(self, mats, camera):
        self._mats, self.camera = mats, camera

    def material_table(self):
        return self._mats


class _Geom:
    def __init__(self, bodies, mats, camera):
        self.bodies, self.spec = bodies, _Spec(mats, camera)


def _jet_bucket(v: float, n: int = 12) -> str:
    return f"util_{int(np.clip(round(v / BROKEN_VALUE * (n - 1)), 0, n - 1))}"


def _materials(n: int = 12) -> dict:
    from pinneapple_tools.visualization.studio.colormap import jet
    mats = {f"c_{c}": {"color": list(colour(c).rgb), "metallic": 0.0, "roughness": 0.3} for c in PALETTE}
    for k in range(n):
        t = k / (n - 1)
        mats[f"util_{k}"] = {"color": list(jet(min(1.0, t * BROKEN_VALUE))), "metallic": 0.0, "roughness": 0.45}
    mats["marker"] = {"color": [0.9, 0.06, 0.06], "metallic": 0.0, "roughness": 0.4, "emissive": [0.55, 0.0, 0.0]}
    mats["ghost"] = {"color": [0.82, 0.83, 0.85], "metallic": 0.0, "roughness": 0.6}
    mats["table"] = {"color": [0.62, 0.55, 0.47], "metallic": 0.0, "roughness": 0.8}
    return mats


def frame_geometry(frame: Frame, *, colour_by: str = "auto", ghost: set[int] | None = None):
    """The frame as a duck-typed geometry for :func:`pinneapple_design.geometry_library.render.render_geometry`.

    ``colour_by``: ``"brick"`` (true colours), ``"utilisation"`` (load path), ``"auto"`` (utilisation for build
    frames, true colours for replays). ``ghost``: item indices drawn light grey (earlier building steps)."""
    bodies = []
    for i, it in enumerate(frame.items):
        V, F = _local_mesh(it.brick)
        Vw = V @ _rot(it.quat).T + it.pos
        mode = colour_by if colour_by != "auto" else ("brick" if frame.kind == "replay" else "utilisation")
        if ghost and i in ghost:
            mat = "ghost"
        elif mode == "utilisation" and not np.isnan(it.value):
            mat = _jet_bucket(it.value)
        else:
            mat = f"c_{it.brick.colour}"
        bodies.append(_Body(Vw, F, mat))
    for m in frame.markers:
        V, F = _marker_mesh(m.size)
        bodies.append(_Body(V @ _rot(m.quat).T + m.pos, F, "marker"))
    for lo, hi in frame.scenery:
        bodies.append(_Body(*box_mesh(lo, hi), "table"))
    return _Geom(bodies, _materials(), {"azimuth": 235.0, "elevation": 22.0})


def render_frames(frames: Sequence[Frame], *, size=(960, 540), azimuth: float = 250.0, elevation: float = 20.0,
                  colour_by: str = "auto", ssaa: int = 1, labels: bool = True) -> list[np.ndarray]:
    """Software-rendered RGB frames with one fixed camera framing every frame."""
    from pinneapple_design.geometry_library.render import render_geometry
    lo, hi = frame_bounds(frames)
    out = []
    for f in frames:
        im = render_geometry(frame_geometry(f, colour_by=colour_by), size=size, azimuth=azimuth, elevation=elevation,
                             ssaa=ssaa, frame=(lo, hi), label=(f.short or f.caption) if labels else None, margin=0.92)
        mode = colour_by if colour_by != "auto" else ("brick" if f.kind == "replay" else "utilisation")
        if labels:
            _legend(im, "utilisation" if mode == "utilisation" else ("broken" if f.markers else None))
        out.append(np.asarray(im))
    return out


def _legend(im, kind: str | None) -> None:
    """Colour key in the lower-left corner: the stud-utilisation bar, or the red broken-joint swatch."""
    if kind is None:
        return
    from PIL import ImageDraw

    from pinneapple_design.geometry_library.render import _font
    from pinneapple_tools.visualization.studio.colormap import jet
    W, H = im.size
    d = ImageDraw.Draw(im)
    font = _font(max(11, H // 34))
    x0, y0 = int(0.03 * W), int(0.90 * H)
    if kind == "utilisation":
        w, h = int(0.22 * W), max(8, H // 50)
        for i in range(w):
            c = jet(min(1.0, i / (w - 1)))
            d.line([(x0 + i, y0), (x0 + i, y0 + h)], fill=tuple(int(255 * v) for v in c))
        d.text((x0, y0 - int(1.6 * h) - 4), "stud tension / clutch", fill=(40, 44, 52), font=font)
        d.text((x0, y0 + h + 2), "0", fill=(40, 44, 52), font=font)
        d.text((x0 + w - 8, y0 + h + 2), "1", fill=(40, 44, 52), font=font)
    else:
        s = max(10, H // 40)
        d.rectangle([x0, y0, x0 + s, y0 + s], fill=(230, 16, 16))
        d.text((x0 + s + 6, y0 - 2), "joint that let go (clutch exceeded)", fill=(40, 44, 52), font=font)


def write_gif(path: str, images: Sequence[np.ndarray], fps: float = 8.0) -> str:
    from pinneapple_tools.visualization.dashboard.static import write_gif as _gif
    return _gif(path, images, fps)


def write_mp4(path: str, images: Sequence[np.ndarray], fps: float = 8.0) -> str:
    """H.264 MP4 through imageio-ffmpeg when installed, else the ``ffmpeg`` program."""
    imgs = [np.ascontiguousarray(np.asarray(i)[..., :3]) for i in images]
    h, w = imgs[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    imgs = [i[:h2, :w2] for i in imgs]
    try:
        import imageio.v2 as iio
        iio.mimsave(path, imgs, fps=fps, codec="libx264", macro_block_size=1, pixelformat="yuv420p")
        return path
    except ImportError:
        pass
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("writing MP4 needs imageio[ffmpeg] or the ffmpeg program")
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, input=b"".join(i.tobytes() for i in imgs), check=True)
    return path


# ---------------------------------------------------------------------------------------------- Blender
def blender_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("bpy") is not None


def render_blender(asm: Assembly, out: str, *, values: Sequence[float] | None = None, view="iso",
                   samples: int = 64, size=(1280, 720), title: str | None = None,
                   tables: Sequence[tuple[np.ndarray, np.ndarray]] | None = None, **kw) -> str:
    """Cycles render of the build in its brick colours, or coloured by ``values`` per brick (stud utilisation).
    ``tables``: table boxes [mm] drawn with the build (default: :func:`scenery` of the build; ``[]`` for none)."""
    import tempfile

    from pinneapple_tools.visualization.studio.blender import render
    from pinneapple_tools.visualization.studio.scene import Surface

    from .mesh import studio_scene
    tables = scenery([asm], thickness=24.0, margin=24.0) if tables is None else tables

    def add_tables(sc):
        for k, (lo, hi) in enumerate(tables):
            V, F = box_mesh(lo, hi)
            sc.surfaces.append(Surface(f"table_{k}", V / 1000.0, F, material="table"))
    if values is None:
        sc, mats = studio_scene(asm)
        add_tables(sc)
        mats["table"] = {"color": [0.62, 0.55, 0.47], "metallic": 0.0, "roughness": 0.8}
        glb = os.path.join(tempfile.mkdtemp(), "build.glb")
        with open(glb, "wb") as f:
            f.write(sc.to_glb(materials=mats))
        return render(glb, out, view=view, samples=samples, size=size, title=title, colorbar_on=False, **kw)
    sc, _ = studio_scene(asm, fields={UTIL_FIELD: values})
    add_tables(sc)                                           # no field on the tables: drawn grey
    sc.labels[UTIL_FIELD] = "stud utilisation (tension / clutch)"
    return render(sc, out, field=UTIL_FIELD, field_range=(0.0, 1.0), view=view, samples=samples, size=size,
                  title=title or "stud utilisation", **kw)


def turntable_blender(asm: Assembly, folder: str, *, frames: int = 24, elevation: float = 0.45, samples: int = 32,
                      size=(960, 540), gif: str | None = None, mp4: str | None = None, fps: float = 12.0) -> list[str]:
    """``frames`` Cycles renders around the build (one per view direction); returns the PNG paths and, with
    ``gif`` / ``mp4``, also writes the turntable as an animation."""
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    out = []
    for k in range(frames):
        a = 2 * np.pi * k / frames
        out.append(render_blender(asm, os.path.join(folder, f"turntable_{k:03d}.png"),
                                  view=(float(np.cos(a)), float(np.sin(a)), elevation), samples=samples, size=size))
    if gif or mp4:
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in out]
        if gif:
            write_gif(gif, imgs, fps)
        if mp4:
            write_mp4(mp4, imgs, fps)
    return out


# ---------------------------------------------------------------------------------------------- instructions
def building_steps(asm: Assembly) -> list[list[int]]:
    """Brick indices per building step: one step per layer (plate height), bottom first."""
    layers: dict[int, list[int]] = {}
    for i, b in enumerate(asm.bricks):
        layers.setdefault(b.z, []).append(i)
    return [sorted(v, key=lambda i: (asm.bricks[i].x, asm.bricks[i].y)) for _, v in sorted(layers.items())]


def write_instructions(asm: Assembly, folder: str, *, title: str = "Building instructions", size=(800, 520)) -> str:
    """Step images (earlier steps in grey, the new parts in colour), a part list, an LDraw file with one STEP per
    layer and ``instructions.md`` tying them together. Returns the Markdown path."""
    from pinneapple_design.geometry_library.render import render_geometry

    from .ldraw import write_ldr
    os.makedirs(folder, exist_ok=True)
    steps = building_steps(asm)
    full = build_frame(asm)
    lo, hi = frame_bounds([full])
    lines = [f"# {title}", "", f"{len(asm)} parts, {len(steps)} steps, "
             f"{asm.height_mm():.1f} mm tall, {asm.mass() * 1e3:.1f} g.", "", "## Part list", "",
             "| part | colour | count |", "|---|---|---|"]
    for k, n in asm.part_counts().items():
        p, c = k.split(" ")
        lines.append(f"| {p} | {c} | {n} |")
    lines += ["", "## Steps", ""]
    done: set[int] = set()
    for s, ids in enumerate(steps, 1):
        shown = sorted(done | set(ids))
        sub = Assembly([asm.bricks[i] for i in shown], asm.ground)
        ghost = {j for j, i in enumerate(shown) if i in done}
        geom = frame_geometry(build_frame(sub), colour_by="brick", ghost=ghost)
        img = os.path.join(folder, f"step_{s:02d}.png")
        render_geometry(geom, img, size=size, azimuth=235, elevation=28, ssaa=2, frame=(lo, hi), label=f"step {s}")
        new = {}
        for i in ids:
            k = f"{asm.bricks[i].part.name} ({asm.bricks[i].colour})"
            new[k] = new.get(k, 0) + 1
        lines += [f"### Step {s}", "", f"![step {s}](step_{s:02d}.png)", "",
                  "Add: " + ", ".join(f"{n} x {k}" for k, n in new.items()), ""]
        done |= set(ids)
    write_ldr(asm, os.path.join(folder, "build.ldr"))
    lines += ["The same steps as an LDraw file: [build.ldr](build.ldr) (opens in LDraw-compatible editors).", ""]
    path = os.path.join(folder, "instructions.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
