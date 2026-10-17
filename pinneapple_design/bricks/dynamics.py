"""Rigid-body replay of a brick build with NVIDIA Newton (:mod:`pinneapple_simulation.external_solvers.newton_physics`).

Every cluster of parts held together by intact stud joints is one rigid body made of the parts' boxes (welded to the
world when a baseplate holds it); released joints become ordinary box-box contacts with friction.
The ground is Newton's plane, or static table boxes for ground regions (then the plane is a floor below them, so
parts that leave the table fall onto it). Newton handles box-box contacts with Coulomb friction (XPBD by default).

The static analysis (:mod:`.statics`) decides which joints fail; this module shows what happens next. Newton has
no breakable joints, so a joint either holds for the whole replay or is released from the start.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .assembly import Assembly
from .parts import PITCH_MM, G, JointModel

FLOOR_DEPTH_M = 0.08          # the floor under table regions [m]


def table_boxes(asm: Assembly, margin_mm: float = 80.0) -> list[tuple[float, float, float, float]]:
    """The ground regions clipped to the build's neighbourhood, (x0, x1, y0, y1) in millimetres."""
    lo, hi = asm.bounds_mm()
    box = (lo[0] - margin_mm, hi[0] + margin_mm, lo[1] - margin_mm, hi[1] + margin_mm)
    out = []
    for r in asm.ground.regions or []:
        x0, x1, y0, y1 = (v * PITCH_MM for v in r)
        x0, x1 = max(x0, box[0]), min(x1, box[1])
        y0, y1 = max(y0, box[2]), min(y1, box[3])
        if x1 > x0 and y1 > y0:
            out.append((x0, x1, y0, y1))
    return out


def clusters(asm: Assembly, broken: Sequence[tuple[int, int]] = ()) -> tuple[list[list[int]], set[int]]:
    """Groups of bricks held together by intact stud joints, and the groups welded to a baseplate."""
    broken = {tuple(p) for p in broken}
    parent = list(range(len(asm.bricks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    grounded = set()
    for c in asm.connections():
        if not c.clutch or (c.lower, c.upper) in broken:
            continue
        if c.lower < 0:
            grounded.add(c.upper)
        else:
            parent[find(c.lower)] = find(c.upper)
    groups: dict[int, list[int]] = {}
    for i in range(len(asm.bricks)):
        groups.setdefault(find(i), []).append(i)
    out = list(groups.values())
    return out, {k for k, g in enumerate(out) if grounded & set(g)}


def rigid_scene(asm: Assembly, broken: Sequence[tuple[int, int]] = (), joint: JointModel | None = None,
                dead_loads: Sequence = ()):
    """Newton :class:`RigidScene` of the build (metres): one rigid body per cluster of bricks held by intact stud
    joints (a compound of their boxes), welded to the world when it is held by a baseplate. ``dead_loads``:
    :class:`~.statics.Load` whose downward part is a dead weight, added as a small box at the load point.

    Returns ``(scene, layout, lift)``: ``layout[k] = (body index, brick indices, local brick centres [m])``; with
    ground regions the scene is lifted by ``lift`` metres so the tables stand on Newton's ground plane, which
    becomes the floor parts fall onto."""
    from pinneapple_simulation.external_solvers.newton_physics import RigidScene
    joint = joint or JointModel()
    mu = joint.mu
    s = RigidScene()
    s.ground(mu=mu)                                          # the table, or the floor under table regions
    lift = 0.0
    if asm.ground.regions is not None:
        lift = FLOOR_DEPTH_M
        for k, (x0, x1, y0, y1) in enumerate(table_boxes(asm)):
            t = s.box(f"table_{k}", ((x1 - x0) / 2000, (y1 - y0) / 2000, lift / 2),
                      pos=((x0 + x1) / 2000, (y0 + y1) / 2000, lift / 2), mass=50.0, mu=mu)
            s.fixed(-1, t)
    groups, welded = clusters(asm, broken)
    layout = []
    for k, g in enumerate(groups):
        centres = np.array([asm.bricks[i].com_mm() / 1000.0 for i in g]) + [0.0, 0.0, lift]
        origin = centres[0]
        boxes = []
        for i, c in zip(g, centres, strict=True):
            b = asm.bricks[i]
            sx, sy, sz = b.part.size_mm if b.rot % 2 == 0 else (b.part.size_mm[1], b.part.size_mm[0], b.part.size_mm[2])
            boxes.append(((sx / 2000, sy / 2000, sz / 2000), tuple(c - origin), b.part.mass))
        for ld in dead_loads:
            if ld.body in g and ld.force[2] < 0:             # the dead weight: a 4 mm cube on the load point
                p = np.asarray(ld.point, float) / 1000.0 + [0.0, 0.0, lift + 0.002]
                boxes.append(((0.002, 0.002, 0.002), tuple(p - origin), float(-ld.force[2]) / G))
        body = s.compound(f"cluster_{k}", boxes, pos=tuple(origin), mu=mu)
        if k in welded:
            s.fixed(-1, body)
        layout.append((body, g, centres - origin))
    return s, layout, lift


@dataclass
class Replay:
    t: np.ndarray                 # (T,) [s]
    pos: np.ndarray               # (T, n, 3) brick centres [mm]
    quat: np.ndarray              # (T, n, 4) (w, x, y, z)
    broken: list[tuple[int, int]]
    max_displacement_mm: float
    moved: np.ndarray             # (n,) bool: displaced more than ``tol_mm`` by the end
    solver: str
    wall_time_s: float
    max_tilt_deg: float = 0.0     # largest rotation of any brick from its start orientation

    @property
    def collapsed(self) -> bool:
        return bool(self.moved.any())


def replay(asm: Assembly, broken: Sequence[tuple[int, int]] = (), *, duration: float = 0.8, dt: float = 1e-3,
           solver: str = "xpbd", iterations: int = 16, frames: int = 40, loads: Sequence = (), tol_mm: float = 2.0,
           joint: JointModel | None = None) -> Replay:
    """Simulate the build released at rest with ``broken`` joints let go. ``loads`` (:class:`~.statics.Load`): a
    downward load is a dead weight (a small box of that weight at the load point, falling with the part instead of
    pushing it with a constant force); horizontal parts are constant forces with their moment. Returns brick poses at
    ``frames`` evenly spaced times; raises ``FloatingPointError`` if the solver diverges."""
    from pinneapple_simulation.external_solvers.newton_physics import simulate
    from pinneapple_simulation.external_solvers.newton_physics.scene import _rot
    scene, layout, lift = rigid_scene(asm, broken, joint, loads)
    body_of = {i: body for body, g, _ in layout for i in g}
    wrenches = {}
    for ld in loads:                                         # horizontal parts: constant forces
        f = np.asarray(ld.force, float) * [1.0, 1.0, 0.0 if ld.force[2] < 0 else 1.0]
        if not f.any():
            continue
        body = body_of[ld.body]
        boxes = scene.bodies[body].shape.boxes
        m = np.array([bx[2] for bx in boxes])
        com = np.asarray(scene.bodies[body].pos) + (m[:, None] * np.array([bx[1] for bx in boxes])).sum(0) / m.sum()
        r = np.asarray(ld.point, float) / 1000.0 + [0.0, 0.0, lift] - com      # Newton: torque about the COM
        F0, T0 = wrenches.get(body, (np.zeros(3), np.zeros(3)))
        wrenches[body] = (F0 + f, T0 + np.cross(r, f))
    n_steps = int(round(duration / dt))
    every = max(1, n_steps // max(frames - 1, 1))
    tr = simulate(scene, duration=duration, dt=dt, solver=solver, iterations=iterations, record_every=every,
                  wrenches=wrenches or None)
    T = len(tr.t)
    n = len(asm.bricks)
    pos, quat = np.zeros((T, n, 3)), np.zeros((T, n, 4))
    for body, g, local in layout:
        for k in range(T):
            R = _rot(tr.quat[k, body])
            pos[k, g] = tr.pos[k, body] + local @ R.T
            quat[k, g] = tr.quat[k, body]
    pos = pos * 1000.0 - [0.0, 0.0, lift * 1000.0]
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
        raise FloatingPointError(f"Newton ({solver}, dt = {dt}) diverged; try a smaller dt")
    disp = np.linalg.norm(pos - pos[:1], axis=2)
    dots = np.clip(np.abs((quat * quat[:1]).sum(-1)), 0.0, 1.0)
    tilt = float(np.degrees(2 * np.arccos(dots)).max(initial=0.0))
    return Replay(tr.t, pos, quat, [tuple(p) for p in broken], float(disp.max(initial=0.0)),
                  disp[-1] > tol_mm, solver, tr.wall_time_s, tilt)


def harmonic_threshold(n: int = 4, length_mm: float = 63.8, *,
                       factors: Sequence[float] = (0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1),
                       duration: float = 0.8, mu: float = 0.4, tol_mm: float = 1.0, solver: str = "xpbd") -> list[dict]:
    """Newton on the harmonic stack of ``n`` smooth blocks with every offset scaled by ``f``: does the top block
    stay (closed form: stands for f <= 1, falls for f > 1)? One row per factor."""
    from pinneapple_simulation.external_solvers.newton_physics import RigidScene, simulate

    from .closed_form import harmonic_stack
    rows = []
    h, w, m = 9.6, 7.8, 0.00303
    for f in factors:
        right = harmonic_stack(n, length_mm) + length_mm
        rel = np.diff(np.concatenate([[0.0], right])) * f
        right = np.cumsum(rel)
        s = RigidScene().ground(mu=mu)                      # the floor; the table top is at z = lift
        lift = FLOOR_DEPTH_M
        tb = s.box("table", (0.1, 0.05, lift / 2), pos=(-0.1, 0, lift / 2), mass=50.0, mu=mu)
        s.fixed(-1, tb)
        for i, r in enumerate(right):
            s.box(f"b{i}", (length_mm / 2000, w / 2000, h / 2000),
                  pos=((r - length_mm / 2) / 1000, 0, lift + (i + 0.5) * h / 1000), mass=m, mu=mu)
        tr = simulate(s, duration=duration, dt=1e-3, solver=solver, iterations=16, record_every=50)
        top = tr.pos[:, -1] * 1000
        drop = float(top[0, 2] - top[-1, 2])
        rows.append({"factor": f, "overhang_mm": float(right[-1]), "closed_form_stands": f <= 1.0,
                     "newton_top_drop_mm": drop, "newton_stands": drop < tol_mm,
                     "agree": (f <= 1.0) == (drop < tol_mm)})
    return rows


def cross_check(goal, candidates: Sequence, *, max_n: int = 30, duration: float = 0.4, tol_mm: float = 2.0,
                dt: float = 2.5e-4, iterations: int = 32, seed: int = 0) -> dict:
    """Does Newton agree with the statics verdict? For up to ``max_n`` evaluated candidates (``.asm``, ``.ev``) whose
    verdict Newton can judge with every stud joint intact (``stable``: nothing should move by more than ``tol_mm``;
    ``mechanism``: something should), replay them under the goal's loads. Joint failures are left out: Newton has no
    clutch model, so it cannot confirm or refute them. The small default ``dt`` keeps XPBD's friction creep under
    a sustained sideways load (about 3 mm in 0.6 s at dt = 1 ms on a 5 g tower pushed with 10 mN) below ``tol_mm``.
    Returns the rows and the confusion counts."""
    rng = np.random.default_rng(seed)
    pool = [c for c in candidates if c.ev.status in ("stable", "mechanism") and len(c.asm)]
    stable = [c for c in pool if c.ev.status == "stable"]          # balance the two verdicts
    mech = [c for c in pool if c.ev.status == "mechanism"]
    k = max_n // 2
    pick = [stable[i] for i in rng.permutation(len(stable))[:k]] + [mech[i] for i in rng.permutation(len(mech))[:k]]
    rows = []
    for c in pick:
        rp = replay(c.asm, (), duration=duration, frames=4, loads=goal.loads(c.asm), tol_mm=tol_mm, dt=dt,
                    iterations=iterations)
        statics_stands = c.ev.status == "stable"
        rows.append({"statics": c.ev.status, "newton_moved_mm": rp.max_displacement_mm, "newton_tilt_deg": rp.max_tilt_deg,
                     "newton_stands": not rp.collapsed, "agree": statics_stands == (not rp.collapsed),
                     "n_bricks": len(c.asm), "explanation": c.ev.explanation})
    conf = {"stable_and_stays": 0, "stable_but_moves": 0, "tips_and_moves": 0, "tips_but_stays": 0}
    for r in rows:
        key = ("stable_and_stays" if r["newton_stands"] else "stable_but_moves") if r["statics"] == "stable" else \
              ("tips_but_stays" if r["newton_stands"] else "tips_and_moves")
        conf[key] += 1
    return {"rows": rows, "confusion": conf, "agreement": float(np.mean([r["agree"] for r in rows])) if rows else
            float("nan"), "tol_mm": tol_mm, "duration_s": duration, "dt": dt, "iterations": iterations}


def friction_creep(asm: Assembly, loads: Sequence, *, dts: Sequence[float] = (1e-3, 2.5e-4), duration: float = 0.6,
                   iterations: int = 32) -> list[dict]:
    """Largest displacement of a build that should hold still under ``loads``, for several time steps: a sustained
    sideways load makes XPBD friction creep by an amount that shrinks with ``dt`` (a solver artefact)."""
    out = []
    for dt in dts:
        rp = replay(asm, (), duration=duration, frames=4, loads=loads, dt=dt, iterations=iterations)
        out.append({"dt": dt, "max_displacement_mm": rp.max_displacement_mm, "max_tilt_deg": rp.max_tilt_deg,
                    "duration_s": duration})
    return out
