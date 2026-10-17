"""Static equilibrium of a brick build: contact forces, stud-joint utilisation, failure explanations.

Model (rigid parts, linear programme)
-------------------------------------
Every part is a rigid body loaded by its weight and by external point loads. Every contact patch between two
parts (or a part and the ground) is the overlap rectangle of their faces and carries

* compression ``c >= 0`` at the four corners of the patch (so the resultant can sit anywhere inside it), the
  corners moved ``JointModel.edge_inset_mm`` inwards (rounded edges; no knife-edge balance),
* at a stud joint: a clutch tension ``0 <= t_s <= clutch`` at each stud centre (the stud pulling out of its tube)
  and a horizontal force at each stud (shear and torsion through the stud walls), free as in StableLego or capped
  by ``JointModel.shear_n``;
* at a smooth contact (a table, a stud-less block): Coulomb friction at each corner, ``|f_x|, |f_y| <= mu c``.

Equilibrium of forces and moments of each body gives six equations. Among all force distributions that satisfy
them, :func:`analyse` finds the one with the smallest peak stud utilisation ``u = max t_s / clutch`` (then the
smallest total tension). By the lower-bound theorem of plasticity, if some admissible distribution keeps every
stud within its capacity, a rigid-plastic structure stands, so ``u <= 1`` means "stands" under these
assumptions and ``1 / u`` is the factor on gravity it could carry. No admissible distribution at all, even with
unbreakable studs, means a mechanism: the build tips or slides (reported with the centre of mass and the support
polygon it left). The model is optimistic: real clutch is brittle and joints are compliant, so the actual force
distribution is not the most favourable one.

Units: millimetres, newtons, kilograms.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .parts import PITCH_MM, PLATE_MM, G, JointModel


# ---------------------------------------------------------------------------------------------- structure
@dataclass
class Body:
    lo: np.ndarray            # box lower corner [mm]
    hi: np.ndarray            # box upper corner [mm]
    mass: float               # [kg]
    com: np.ndarray | None = None
    label: str = ""

    def __post_init__(self):
        self.lo, self.hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
        self.com = 0.5 * (self.lo + self.hi) if self.com is None else np.asarray(self.com, float)


@dataclass
class Interface:
    """Contact patch between ``lower`` (``-1`` = ground) and ``upper`` at height ``z`` [mm].

    ``rect``: (x0, x1, y0, y1) [mm]. ``studs``: (k, 2) stud centres [mm]; empty = smooth contact."""

    lower: int
    upper: int
    z: float
    rect: tuple[float, float, float, float]
    studs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    label: str = ""

    def __post_init__(self):
        self.studs = np.asarray(self.studs, float).reshape(-1, 2)

    @property
    def clutch(self) -> bool:
        return len(self.studs) > 0

    def corners(self, inset: float = 0.0) -> np.ndarray:
        """Corners of the patch, moved ``inset`` mm inwards (to the centre line when the patch is narrower)."""
        x0, x1, y0, y1 = self.rect
        dx, dy = min(inset, (x1 - x0) / 2), min(inset, (y1 - y0) / 2)
        x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


@dataclass
class Load:
    body: int
    point: np.ndarray          # [mm]
    force: np.ndarray          # [N]
    label: str = ""

    def __post_init__(self):
        self.point, self.force = np.asarray(self.point, float), np.asarray(self.force, float)


@dataclass
class Structure:
    bodies: list[Body]
    interfaces: list[Interface]
    loads: list[Load] = field(default_factory=list)
    gravity: float = G

    @classmethod
    def from_boxes(cls, boxes: Sequence[tuple[Sequence[float], Sequence[float]]], masses: Sequence[float], *,
                   ground_z: float = 0.0, ground_rects: Sequence[tuple[float, float, float, float]] | None = None,
                   tol: float = 1e-6) -> Structure:
        """Smooth boxes (no studs) with contacts found where a top face meets a bottom face."""
        bodies = [Body(lo, hi, m, label=f"block {i}") for i, ((lo, hi), m) in enumerate(zip(boxes, masses, strict=True))]
        ifs = []
        for i, b in enumerate(bodies):
            r = (b.lo[0], b.hi[0], b.lo[1], b.hi[1])
            if abs(b.lo[2] - ground_z) < tol:
                for g in (ground_rects or [(-np.inf, np.inf, -np.inf, np.inf)]):
                    o = _overlap(r, g)
                    if o:
                        ifs.append(Interface(-1, i, ground_z, o))
            for j, a in enumerate(bodies):
                if j != i and abs(a.hi[2] - b.lo[2]) < tol:
                    o = _overlap((a.lo[0], a.hi[0], a.lo[1], a.hi[1]), r)
                    if o:
                        ifs.append(Interface(j, i, b.lo[2], o))
        return cls(bodies, ifs)


def _overlap(a, b):
    x0, x1, y0, y1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    return (x0, x1, y0, y1) if x1 - x0 > 1e-9 and y1 - y0 > 1e-9 else None


def from_assembly(asm, loads: Sequence[Load] = (), broken: Sequence[tuple[int, int]] = ()) -> Structure:
    """The :class:`Structure` of an :class:`~pinneapple_design.bricks.assembly.Assembly` (cell-bounds boxes).

    ``broken``: (lower, upper) brick pairs whose stud joint has failed; they keep a smooth (friction) contact."""
    broken = {tuple(p) for p in broken}
    bodies = []
    for i, b in enumerate(asm.bricks):
        lo, hi = b.box_mm()
        bodies.append(Body(lo, hi, b.part.mass, label=f"{i}:{b.part.name}"))
    ifs = []
    for c in asm.connections():
        rect = tuple(v * PITCH_MM for v in c.rect)
        cells = [] if (c.lower, c.upper) in broken else c.studs
        studs = np.array([[(i + 0.5) * PITCH_MM, (j + 0.5) * PITCH_MM] for i, j in cells]).reshape(-1, 2)
        ifs.append(Interface(c.lower, c.upper, c.z * PLATE_MM, rect, studs,
                             label=f"{'ground' if c.lower < 0 else c.lower}->{c.upper}"))
    return Structure(bodies, ifs, list(loads))


# ---------------------------------------------------------------------------------------------- results
@dataclass
class StaticsResult:
    status: str                             # "stable" | "joint_failure" | "mechanism" | "empty"
    utilisation: float                      # peak stud tension / clutch (inf for a mechanism)
    interface_utilisation: np.ndarray       # (n_interfaces,) peak stud tension / clutch per interface
    interface_compression: np.ndarray       # (n_interfaces,) total compression [N]
    interface_tension: np.ndarray           # (n_interfaces,) total stud tension [N]
    body_load: np.ndarray                   # (n_bodies,) compression carried through each body's bottom [N]
    body_utilisation: np.ndarray            # (n_bodies,) worst utilisation of the body's contacts
    critical: list[int]                     # interfaces at the peak utilisation (joint failure)
    explanation: str
    tipping: dict | None = None             # for a mechanism: the support polygon the centre of mass left
    clutch_n: float = 0.98
    lp_status: str = ""

    @property
    def stable(self) -> bool:
        return self.status == "stable"

    @property
    def margin(self) -> float:
        """Load factor on gravity the build can carry (1 / utilisation; inf when no stud is loaded)."""
        if self.status == "mechanism":
            return 0.0
        return float("inf") if self.utilisation <= 0 else 1.0 / self.utilisation


# ---------------------------------------------------------------------------------------------- the LP
class _LP:
    """Columns: per interface 4 corner compressions; stud interfaces add (t, fx, fy) per stud; smooth ones add
    (fx, fy) per corner. One extra column: ``u`` (peak utilisation) or ``lam`` (load factor)."""

    def __init__(self, s: Structure, joint: JointModel):
        self.s, self.joint = s, joint
        self.cols = 0
        self.blocks = []
        for itf in s.interfaces:
            blk = {"c": self._take(4)}
            if itf.clutch:
                k = len(itf.studs)
                blk["t"], blk["fx"], blk["fy"] = self._take(k), self._take(k), self._take(k)
            else:
                blk["fx"], blk["fy"] = self._take(4), self._take(4)
            self.blocks.append(blk)
        self.extra = self._take(1)[0]
        self.n = self.cols

    def _take(self, k):
        idx = np.arange(self.cols, self.cols + k)
        self.cols += k
        return idx

    def equilibrium(self):
        """A_eq (6 n_bodies x n), b_eq (gravity), L (6 n_bodies,) external-load column."""
        nb = len(self.s.bodies)
        trip: list[tuple[int, int, float]] = []
        b = np.zeros(6 * nb)
        L = np.zeros(6 * nb)

        def add_force(body, point, F, col, sign):
            if body < 0:
                return
            r = np.asarray(point, float) - self.s.bodies[body].com
            M = np.cross(r, F)
            for k in range(3):
                if F[k]:
                    trip.append((6 * body + k, col, sign * F[k]))
                if M[k]:
                    trip.append((6 * body + 3 + k, col, sign * M[k]))

        ez, ex, ey = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        for itf, blk in zip(self.s.interfaces, self.blocks, strict=True):
            corners = itf.corners(self.joint.edge_inset_mm)
            for k, col in enumerate(blk["c"]):                      # compression pushes the upper body up
                p = (*corners[k], itf.z)
                add_force(itf.upper, p, ez, col, +1)
                add_force(itf.lower, p, ez, col, -1)
            pts = itf.studs if itf.clutch else corners
            if itf.clutch:
                for k, col in enumerate(blk["t"]):                  # tension pulls the upper body down
                    p = (*pts[k], itf.z)
                    add_force(itf.upper, p, ez, col, -1)
                    add_force(itf.lower, p, ez, col, +1)
            for key, d in (("fx", ex), ("fy", ey)):
                for k, col in enumerate(blk[key]):
                    p = (*pts[k], itf.z)
                    add_force(itf.upper, p, d, col, +1)
                    add_force(itf.lower, p, d, col, -1)
        for i, body in enumerate(self.s.bodies):                    # sum of contact forces = -weight
            b[6 * i + 2] = body.mass * self.s.gravity
        for ld in self.s.loads:                                     # external loads, scaled by the extra column
            r = ld.point - self.s.bodies[ld.body].com
            L[6 * ld.body:6 * ld.body + 3] += ld.force
            L[6 * ld.body + 3:6 * ld.body + 6] += np.cross(r, ld.force)
        return _sparse(trip, 6 * nb, self.n), b, L

    def inequalities(self, mode: str):
        """Friction cones and clutch limits. ``mode``: "u" (t <= u * clutch) or "lam" (t <= clutch)."""
        trip: list[tuple[int, int, float]] = []
        rhs: list[float] = []
        r = 0
        mu, T = self.joint.mu, self.joint.clutch_n
        for itf, blk in zip(self.s.interfaces, self.blocks, strict=True):
            if itf.clutch:
                for col in blk["t"]:
                    trip.append((r, col, 1.0))
                    if mode == "u":
                        trip.append((r, self.extra, -T))
                    rhs.append(0.0 if mode == "u" else T)
                    r += 1
            else:
                for key in ("fx", "fy"):
                    for k, col in enumerate(blk[key]):
                        for sgn in (1.0, -1.0):                     # +-f - mu c <= 0
                            trip += [(r, col, sgn), (r, blk["c"][k], -mu)]
                            rhs.append(0.0)
                            r += 1
        return _sparse(trip, r, self.n), np.array(rhs)

    def bounds(self):
        lb = np.zeros(self.n)
        ub = np.full(self.n, np.inf)
        cap = self.joint.shear_n
        for itf, blk in zip(self.s.interfaces, self.blocks, strict=True):   # horizontal forces are signed
            lb[blk["fx"]] = lb[blk["fy"]] = -np.inf
            if itf.clutch and cap is not None:
                lb[blk["fx"]] = lb[blk["fy"]] = -cap
                ub[blk["fx"]] = ub[blk["fy"]] = cap
        return list(zip(lb, ub, strict=True))


def _sparse(trip, n_rows, n_cols):
    from scipy.sparse import csr_matrix
    if not trip:
        return csr_matrix((n_rows, n_cols))
    r, c, v = zip(*trip, strict=True)
    return csr_matrix((v, (r, c)), shape=(n_rows, n_cols))


def _solve(c, A_ub, b_ub, A_eq, b_eq, bounds):
    from scipy.optimize import linprog
    return linprog(c, A_ub=A_ub if A_ub.shape[0] else None, b_ub=b_ub if A_ub.shape[0] else None,
                   A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")


def analyse(structure: Structure, joint: JointModel | None = None) -> StaticsResult:
    """Equilibrium under gravity and ``structure.loads`` with the smallest peak stud utilisation."""
    joint = joint or JointModel()
    s = structure
    nb, ni = len(s.bodies), len(s.interfaces)
    if nb == 0:
        z = np.zeros(0)
        return StaticsResult("empty", 0.0, z, z, z, z, z, [], "no parts", None, joint.clutch_n)
    lp = _LP(s, joint)
    A_eq, b_eq, L = lp.equilibrium()
    # loads enter with factor 1: move them to the right-hand side
    b_eq = b_eq - L
    A_ub, b_ub = lp.inequalities("u")
    bounds = lp.bounds()
    c = np.zeros(lp.n)
    c[lp.extra] = 1.0
    res = _solve(c, A_ub, b_ub, A_eq, b_eq, bounds)
    if res.status != 0:
        tip = tipping_report(s, joint)
        msg = tip["message"] if tip else "no equilibrium: the build slides or tips even with unbreakable studs"
        inf = np.full(ni, np.nan)
        return StaticsResult("mechanism", float("inf"), inf, inf, inf, np.zeros(nb), np.full(nb, np.inf), [], msg,
                             tip, joint.clutch_n, res.message)
    u = float(res.x[lp.extra])
    # second stage: at that peak, the least total tension (a unique, physically tidy distribution)
    c2 = np.zeros(lp.n)
    for itf, blk in zip(s.interfaces, lp.blocks, strict=True):
        if itf.clutch:
            c2[blk["t"]] = 1.0
        c2[blk["c"]] = 1e-6
    b2 = list(bounds)
    b2[lp.extra] = (0.0, u * (1 + 1e-7) + 1e-12)
    res2 = _solve(c2, A_ub, b_ub, A_eq, b_eq, b2)
    x = res2.x if res2.status == 0 else res.x
    return _result(s, lp, x, u, joint, res.message)


def _result(s, lp, x, u, joint, lp_status) -> StaticsResult:
    nb, ni = len(s.bodies), len(s.interfaces)
    T = joint.clutch_n
    iu, ic, it = np.zeros(ni), np.zeros(ni), np.zeros(ni)
    for k, (itf, blk) in enumerate(zip(s.interfaces, lp.blocks, strict=True)):
        ic[k] = x[blk["c"]].sum()
        if itf.clutch:
            t = x[blk["t"]]
            it[k] = t.sum()
            iu[k] = t.max() / T if len(t) else 0.0
    bl, bu = np.zeros(nb), np.zeros(nb)
    for k, itf in enumerate(s.interfaces):
        bl[itf.upper] += ic[k]
        for b in (itf.lower, itf.upper):
            if b >= 0:
                bu[b] = max(bu[b], iu[k])
    peak = iu.max() if ni else 0.0
    critical = [k for k in range(ni) if peak > 0 and iu[k] >= peak * (1 - 1e-6)]
    if u <= 1.0 + 1e-9:
        status = "stable"
        expl = (f"stands: peak stud tension {u * T:.3f} N = {u:.0%} of the {T:g} N clutch"
                if u > 0 else "stands on compression and friction alone (no stud in tension)")
    else:
        status = "joint_failure"
        names = ", ".join(_name(s, k) for k in critical[:3])
        expl = (f"joint failure at {names}: needs {u * T:.3f} N per stud, clutch is {T:g} N "
                f"(utilisation {u:.2f})")
    return StaticsResult(status, u, iu, ic, it, bl, bu, critical, expl, None, T, lp_status)


def _name(s: Structure, k: int) -> str:
    itf = s.interfaces[k]
    a = "ground" if itf.lower < 0 else (s.bodies[itf.lower].label or f"part {itf.lower}")
    b = s.bodies[itf.upper].label or f"part {itf.upper}"
    return f"{a} / {b} (z = {itf.z:.1f} mm, {len(itf.studs)} studs)"


def capacity(structure: Structure, joint: JointModel | None = None, lam_max: float = 1e6) -> float:
    """Largest factor ``lam`` on ``structure.loads`` (gravity fixed) with every stud within its clutch and an
    equilibrium that exists (0 when even gravity alone is not carried; ``lam_max`` when the loads are carried
    without limit)."""
    joint = joint or JointModel()
    s = structure
    if not s.bodies:
        return 0.0
    lp = _LP(s, joint)
    A_eq, b_eq, L = lp.equilibrium()
    A_eq = A_eq.tolil()
    A_eq[:, lp.extra] = L.reshape(-1, 1)            # contact forces + lam * loads = -weight
    A_eq = A_eq.tocsr()
    A_ub, b_ub = lp.inequalities("lam")
    bounds = lp.bounds()
    bounds[lp.extra] = (0.0, lam_max)
    c = np.zeros(lp.n)
    c[lp.extra] = -1.0
    res = _solve(c, A_ub, b_ub, A_eq, b_eq, bounds)
    return float(res.x[lp.extra]) if res.status == 0 else 0.0


# ---------------------------------------------------------------------------------------------- tipping
def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2-D points (Andrew's monotone chain)."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, float), 9))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def signed_distance_inside(poly: np.ndarray, p: np.ndarray) -> float:
    """Distance from ``p`` to the boundary of the convex polygon ``poly`` (ccw): > 0 inside, < 0 outside."""
    p = np.asarray(p, float)
    if len(poly) == 1:
        return -float(np.linalg.norm(p - poly[0]))
    if len(poly) == 2:
        a, b = poly
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-30), 0, 1)
        return -float(np.linalg.norm(p - (a + t * (b - a))))
    d_in = np.inf
    outside = False
    d_out = np.inf
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)   # outward normal (ccw polygon)
        s = float(np.dot(p - a, nrm))
        if s > 0:
            outside = True
        d_in = min(d_in, -s)
        t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
        d_out = min(d_out, float(np.linalg.norm(p - (a + t * e))))
    return -d_out if outside else d_in


def centre_of_pressure(s: Structure, members: Sequence[int], z0: float) -> np.ndarray | None:
    """Where the resultant of the weights and loads on ``members`` crosses the plane ``z = z0`` (None if the
    resultant does not point down)."""
    Fz = sx = sy = 0.0
    forces = [(s.bodies[i].com, np.array([0, 0, -s.bodies[i].mass * s.gravity])) for i in members]
    forces += [(ld.point, ld.force) for ld in s.loads if ld.body in set(members)]
    for r, F in forces:
        Fz += F[2]
        sx += r[0] * F[2] - (r[2] - z0) * F[0]
        sy += r[1] * F[2] - (r[2] - z0) * F[1]
    if Fz >= -1e-15:
        return None
    return np.array([sx / Fz, sy / Fz])


def tipping_report(s: Structure, joint: JointModel | None = None) -> dict | None:
    """The worst support polygon left by a centre of pressure, treating stud joints as unbreakable.

    Cuts examined: the ground (per group of parts welded by studs) and every smooth contact that alone carries
    the parts above it. Returns ``{"members", "interfaces", "point", "distance_mm", "message"}`` for the worst cut
    (distance < 0: outside the polygon), or None when every centre of pressure is inside its support."""
    nb = len(s.bodies)
    inset = (joint or JointModel()).edge_inset_mm
    parent = list(range(nb))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for itf in s.interfaces:
        if itf.lower >= 0 and itf.clutch:
            parent[find(itf.lower)] = find(itf.upper)
    clusters: dict[int, list[int]] = {}
    for i in range(nb):
        clusters.setdefault(find(i), []).append(i)

    worst = None
    # cut below each cluster: the smooth or ground contacts that carry it and everything that rests on it
    above: dict[int, set[int]] = {r: set() for r in clusters}
    for itf in s.interfaces:
        if itf.lower >= 0:
            a, b = find(itf.lower), find(itf.upper)
            if a != b:
                above[a].add(b)

    def everything_on(r):
        seen, todo = {r}, [r]
        while todo:
            n = todo.pop()
            for m in above[n]:
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return seen

    for r, members in clusters.items():
        ms = set(members)
        supports = [k for k, itf in enumerate(s.interfaces) if itf.upper in ms and (itf.lower < 0 or find(itf.lower) != r)]
        if not supports:
            continue
        if any(s.interfaces[k].clutch for k in supports):
            continue                      # held by studs: tension can resist tipping, the LP decides
        load_set = sorted(i for c in everything_on(r) for i in clusters[c])
        zs = {round(s.interfaces[k].z, 6) for k in supports}
        if len(zs) != 1:
            continue                      # supports at several heights: no single polygon
        z0 = zs.pop()
        pts = np.vstack([s.interfaces[k].corners(inset) for k in supports])
        hull = convex_hull(pts)
        p = centre_of_pressure(s, load_set, z0)
        if p is None:
            continue
        d = signed_distance_inside(hull, p)
        if worst is None or d < worst["distance_mm"]:
            names = sorted({s.bodies[i].label or str(i) for i in members})
            what = "ground" if all(s.interfaces[k].lower < 0 for k in supports) else "the parts below"
            loaded = any(ld.body in set(load_set) for ld in s.loads)
            what_acts = "the resultant of the weight and loads of" if loaded else "the centre of mass of"
            worst = {"members": load_set, "interfaces": supports, "point": p.tolist(), "z_mm": z0,
                     "distance_mm": d, "hull": hull.tolist(),
                     "message": (f"tips: {what_acts} {len(load_set)} part(s) from {', '.join(names[:3])} up "
                                 f"passes {abs(d):.1f} mm {'outside' if d < 0 else 'inside'} its support on {what} "
                                 f"(z = {z0:.1f} mm)")}
    if worst is not None and worst["distance_mm"] >= 0:
        worst["message"] = worst["message"].replace("tips:", "slides or tips (resultant inside its support):")
    return worst


def analyse_assembly(asm, loads: Sequence[Load] = (), joint: JointModel | None = None,
                     broken: Sequence[tuple[int, int]] = ()) -> StaticsResult:
    """:func:`analyse` of an assembly (gravity + ``loads``), with the stud joints in ``broken`` released."""
    return analyse(from_assembly(asm, loads, broken), joint)


@dataclass
class FailureRound:
    result: StaticsResult
    broken: list[tuple[int, int]]          # all joints released so far (lower, upper brick indices)
    new: list[tuple[int, int]]             # released in this round


def progressive_failure(asm, loads: Sequence[Load] = (), joint: JointModel | None = None,
                        max_rounds: int = 50) -> list[FailureRound]:
    """Release the most utilised stud joints one round at a time until the build stands or becomes a mechanism.

    Each round solves :func:`analyse`; on a joint failure, the critical joints (those at the peak utilisation) lose
    their clutch and keep a friction contact. The last round tells how the build ends: ``stable`` (it found a new
    equilibrium after some joints let go) or ``mechanism`` (it collapses; replay it with :mod:`.dynamics`)."""
    joint = joint or JointModel()
    broken: list[tuple[int, int]] = []
    rounds = []
    for _ in range(max_rounds):
        s = from_assembly(asm, loads, broken)
        r = analyse(s, joint)
        if r.status != "joint_failure":
            rounds.append(FailureRound(r, list(broken), []))
            break
        new = [(s.interfaces[k].lower, s.interfaces[k].upper) for k in r.critical]
        broken += [p for p in new if p not in broken]
        rounds.append(FailureRound(r, list(broken), new))
    return rounds


def top_load(asm, force: Sequence[float], *, at: Sequence[float] | None = None, label: str = "load") -> Load:
    """A point load on the highest part covering ``at`` = (x, y) [mm] (default: the centre of the top part), applied
    at the top face."""
    if at is None:
        i = max(range(len(asm.bricks)), key=lambda k: (asm.bricks[k].top, -k))
        lo, hi = asm.bricks[i].box_mm()
        p = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
        return Load(i, p, np.asarray(force, float), label)
    best = None
    for k, b in enumerate(asm.bricks):
        lo, hi = b.box_mm()
        if lo[0] <= at[0] <= hi[0] and lo[1] <= at[1] <= hi[1] and (best is None or hi[2] > best[1]):
            best = (k, hi[2])
    if best is None:
        raise ValueError(f"no part covers {tuple(at)}")
    return Load(best[0], np.array([at[0], at[1], best[1]]), np.asarray(force, float), label)


def single_stud_torsion(asm, loads: Sequence[Load] = (), joint: JointModel | None = None,
                        spread_mm: float = 1.5) -> dict:
    """Does a verdict rest on single-stud joints having no torsion capacity? A lone stud is a pivot about the
    vertical in this model; re-solve with each single-stud joint given three contact points ``spread_mm`` apart
    (a little torsion and shear moment) and report both statuses."""
    s = from_assembly(asm, loads)
    before = analyse(s, joint).status
    n = 0
    for itf in s.interfaces:
        if len(itf.studs) == 1:
            c = itf.studs[0]
            itf.studs = np.array([c + [-spread_mm, 0.0], c + [spread_mm, 0.0], c + [0.0, spread_mm]])
            n += 1
    return {"single_stud_joints": n, "status": before, "status_with_torsion": analyse(s, joint).status}
