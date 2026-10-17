"""Design goals for the self-modifying loop: each one scores a build with the statics check and says why it fails.

* :class:`CantileverGoal`: the longest stable overhang past a table edge with at most N parts.
* :class:`BridgeGoal`: span a gap between two tables and carry a point load at mid-span with the fewest parts.
* :class:`TowerGoal`: the tallest tower that survives a lateral push at its top, with at most N parts.

All three are scored by :mod:`.statics` (rigid parts, stud clutch, friction); a score is only given to a build that
passes, and every failure comes with the explanation of the statics result (which joint, or where the centre of
mass left its support).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .assembly import Assembly, Ground
from .parts import PITCH_MM, JointModel
from .statics import Load, analyse, capacity, from_assembly, top_load


@dataclass
class Evaluation:
    score: float                      # larger is better (only meaningful when feasible)
    feasible: bool
    violation: float                  # 0 when feasible, larger = further from feasible
    n_bricks: int
    status: str                       # stable | joint_failure | mechanism | invalid | ...
    explanation: str
    metrics: dict = field(default_factory=dict)
    critical: list[tuple[int, int]] = field(default_factory=list)     # (lower, upper) of the critical joints

    def objectives(self, goal) -> np.ndarray:
        return goal.objectives(self)


class Goal:
    """Base class. ``evaluate`` returns an :class:`Evaluation` whose ``violation`` follows one scale for every goal:
    invalid build >= 5, unsupported parts >= 2, goal-specific geometric miss (e.g. no deck) in [2, 3), fails under
    its own weight in [1, 2), overloaded (carries itself but not the load) in [0, 1), feasible 0."""

    name = "goal"
    description = ""
    parts: Sequence[str] = ()
    planar = True                     # parts stay in the y = 0 row, no rotation
    max_bricks = 20
    x_range = (-20, 40)
    y_range = (0, 1)
    z_max = 60
    colours: Sequence[str] = ("blue", "yellow", "green", "white", "orange", "azure", "lime", "light_grey")
    # (no red: the visuals use red for joints that broke)
    target_x: float | None = None     # where "extend" edits grow toward [studs]; None = farther along +x

    def __init__(self, joint: JointModel | None = None):
        self.joint = joint or JointModel()

    # to override
    def ground(self) -> Ground:
        raise NotImplementedError

    def seed(self) -> Assembly:
        raise NotImplementedError

    def loads(self, asm: Assembly) -> list[Load]:
        return []

    def score_and_check(self, asm: Assembly) -> Evaluation:
        raise NotImplementedError

    def objectives(self, ev: Evaluation) -> np.ndarray:
        """Minimised objectives for NSGA-II selection (primary first)."""
        return np.array([-ev.score, ev.n_bricks], float)

    # shared
    def in_bounds(self, asm: Assembly) -> bool:
        for b in asm.bricks:
            x0, x1, y0, y1 = b.footprint()
            if x0 < self.x_range[0] or x1 > self.x_range[1] or y0 < self.y_range[0] or y1 > self.y_range[1]:
                return False
            if b.z < 0 or b.top > self.z_max:
                return False
        return len(asm) <= self.max_bricks

    def evaluate(self, asm: Assembly) -> Evaluation:
        n = len(asm)
        if n == 0:
            return Evaluation(0.0, False, 10.0, 0, "invalid", "empty build")
        overl = asm.overlaps()
        if overl:
            return Evaluation(0.0, False, 5.0 + len(overl), n, "invalid", f"{len(overl)} overlapping pair(s)")
        free = set(range(n)) - asm.grounded()
        if free:
            return Evaluation(0.0, False, 2.0 + len(free), n, "invalid",
                              f"{len(free)} part(s) not connected to the ground: {sorted(free)[:5]}")
        if not self.in_bounds(asm):
            return Evaluation(0.0, False, 5.0, n, "invalid", "outside the design region or too many parts")
        return self.score_and_check(asm)

    def features(self, asm: Assembly) -> np.ndarray:
        """Goal-agnostic descriptors of a build, for surrogate models."""
        if not len(asm):
            return np.zeros(14)
        lo, hi = asm.bounds_mm()
        com = asm.com_mm()
        conns = [c for c in asm.connections() if c.lower >= 0]
        studs = [len(c.studs) for c in conns]
        ground = [c for c in asm.connections() if c.lower < 0]
        garea = sum((c.rect[1] - c.rect[0]) * (c.rect[3] - c.rect[2]) for c in ground)
        nb = len(asm)
        return np.array([nb, lo[0], hi[0], hi[1] - lo[1], hi[2], com[0], com[1], com[2], asm.mass() * 1e3,
                         sum(studs), min(studs, default=0), len(conns) / nb, garea,
                         sum(b.part.h == 1 for b in asm.bricks) / nb], float)


def _failure(asm, res, n, score_if=0.0) -> Evaluation:
    """Evaluation of a build the statics check rejects."""
    s = from_assembly(asm)
    crit = [(s.interfaces[k].lower, s.interfaces[k].upper) for k in res.critical] if res.status == "joint_failure" else []
    if res.status == "mechanism":
        d = res.tipping["distance_mm"] if res.tipping else 0.0
        viol = 1.0 + 0.99 * min(1.0, max(0.0, -d) / 50.0)
    else:
        viol = 1.0 + 0.99 * (1.0 - 1.0 / max(res.utilisation, 1.0))
    return Evaluation(score_if, False, viol, n, res.status, res.explanation,
                      {"utilisation": res.utilisation}, crit)


def _overload(asm, res, n, score, lam, metrics, short) -> Evaluation:
    """A build whose load factor is below 1: the statics explanation when it fails at the full load, else ``short``.
    Violation 1 - lam when it carries its own weight (lam > 0), else that of the failure under gravity (>= 1)."""
    ev = _failure(asm, res, n, score) if not res.stable else Evaluation(score, False, 0.0, n, "overload", short)
    ev.violation = 1.0 - lam if lam > 0 else max(ev.violation, 1.0)
    ev.metrics.update(metrics)
    return ev


class CantileverGoal(Goal):
    """Reach as far past a table edge as possible with at most ``max_bricks`` 1-wide parts, under gravity.

    The table is the half-plane x < 0 (smooth, friction ``mu``); parts are joined by studs. The score is the reach
    of the farthest part end past the edge [mm]. Reference values: one block reaches L / 2, unconnected blocks one
    per level reach (L / 2) H_n (harmonic stacking); welded parts could reach about n L / 2."""

    name = "cantilever"
    description = "longest stable cantilever past a table edge"
    parts = ("brick_1x2", "brick_1x4", "brick_1x6", "brick_1x8", "plate_1x4", "plate_1x8")

    def __init__(self, max_bricks: int = 10, joint: JointModel | None = None):
        super().__init__(joint)
        self.max_bricks = max_bricks
        self.x_range = (-8 * max_bricks, 8 * max_bricks)
        self.z_max = 3 * max_bricks + 3

    def ground(self) -> Ground:
        return Ground("table", [(-1000, 0, -1000, 1000)])

    def seed(self) -> Assembly:
        a = Assembly(ground=self.ground(), name="cantilever seed")
        a.add("brick_1x4", -4, 0, 0, colour="blue")
        return a

    def score_and_check(self, asm: Assembly) -> Evaluation:
        n = len(asm)
        res = analyse(from_assembly(asm), self.joint)
        reach = max(b.footprint()[1] for b in asm.bricks) * PITCH_MM - 0.1
        if not res.stable:
            ev = _failure(asm, res, n)
            ev.metrics["reach_mm"] = reach
            return ev
        return Evaluation(reach, True, 0.0, n, "stable", res.explanation,
                          {"reach_mm": reach, "utilisation": res.utilisation, "margin": res.margin})


class BridgeGoal(Goal):
    """Span a ``gap`` (in studs) between two tables and carry ``load_n`` newtons at mid-span on the deck, with the
    fewest parts. Feasible when the statics check carries gravity plus the load (load factor >= 1)."""

    name = "bridge"
    description = "bridge a gap and carry a mid-span load with the fewest parts"
    parts = ("brick_1x2", "brick_1x4", "brick_1x6", "brick_1x8", "plate_1x4", "plate_1x6", "plate_1x8")

    def __init__(self, gap: int = 12, load_n: float = 1.0, max_bricks: int = 24, joint: JointModel | None = None):
        super().__init__(joint)
        self.gap, self.load_n, self.max_bricks = gap, load_n, max_bricks
        self.target_x = gap / 2
        self.x_range = (-10, gap + 10)
        self.z_max = 36

    def ground(self) -> Ground:
        return Ground("table", [(-1000, 0, -1000, 1000), (self.gap, 1000, -1000, 1000)])

    def seed(self) -> Assembly:
        a = Assembly(ground=self.ground(), name="bridge seed")
        a.add("brick_1x8", -8, 0, 0, colour="blue")
        a.add("brick_1x8", self.gap, 0, 0, colour="blue")
        return a

    def load_point(self) -> tuple[float, float]:
        return (self.gap * PITCH_MM / 2, 0.5 * PITCH_MM)

    def loads(self, asm: Assembly) -> list[Load]:
        try:
            return [top_load(asm, (0.0, 0.0, -self.load_n), at=self.load_point(), label="mid-span load")]
        except ValueError:
            return []

    def objectives(self, ev: Evaluation) -> np.ndarray:
        return np.array([ev.n_bricks, -min(ev.metrics.get("load_factor", 0.0), 10.0)], float)

    def score_and_check(self, asm: Assembly) -> Evaluation:
        n = len(asm)
        lds = self.loads(asm)
        if not lds:
            px = self.load_point()[0]
            d = min(min(abs(b.footprint()[0] * PITCH_MM - px), abs(b.footprint()[1] * PITCH_MM - px))
                    for b in asm.bricks)
            return Evaluation(-n, False, 2.0 + min(0.99, d / (self.gap * PITCH_MM)), n, "no_deck",
                              f"nothing covers mid-span: nearest part end is {d:.0f} mm away")
        s = from_assembly(asm, lds)
        lam = capacity(s, self.joint, lam_max=100.0)
        res = analyse(s, self.joint)
        m = {"load_factor": lam, "utilisation": res.utilisation, "load_n": self.load_n,
             "capacity_n": lam * self.load_n}
        if lam >= 1.0 - 1e-9 and res.stable:
            return Evaluation(-n, True, 0.0, n, "stable",
                              f"carries {lam * self.load_n:.3f} N at mid-span ({lam:.2f} x the {self.load_n} N load)",
                              m)
        return _overload(asm, res, n, -n, lam, m, f"carries only {lam:.2f} x the {self.load_n} N load")


class TowerGoal(Goal):
    """The tallest tower on a ``base`` x ``base`` stud footprint that survives a horizontal push ``push_n`` [N] at its
    top (and gravity) with at most ``max_bricks`` parts; it stands on a smooth table, so it can tip as well as break."""

    name = "tower"
    description = "tallest tower that survives a lateral push at the top"
    parts = ("brick_1x2", "brick_1x4", "brick_2x2", "brick_2x4", "plate_2x4", "plate_2x2", "brick_1x1")
    planar = False

    def __init__(self, max_bricks: int = 16, push_n: float = 0.01, base: int = 6, joint: JointModel | None = None):
        super().__init__(joint)
        self.max_bricks, self.push_n, self.base = max_bricks, push_n, base
        self.x_range = (0, base)
        self.y_range = (0, base)
        self.z_max = 3 * max_bricks + 3

    def ground(self) -> Ground:
        return Ground("table", None)

    def seed(self) -> Assembly:
        a = Assembly(ground=self.ground(), name="tower seed")
        a.add("brick_2x2", 2, 2, 0, colour="blue")
        return a

    def loads(self, asm: Assembly) -> list[Load]:
        return [top_load(asm, (self.push_n, 0.0, 0.0), label="lateral push")]

    def score_and_check(self, asm: Assembly) -> Evaluation:
        n = len(asm)
        h = asm.height_mm()
        s = from_assembly(asm, self.loads(asm))
        lam = capacity(s, self.joint, lam_max=1000.0)
        m = {"height_mm": h, "load_factor": lam, "push_capacity_n": lam * self.push_n}
        if lam >= 1.0 - 1e-9:
            res = analyse(s, self.joint)
            return Evaluation(h, True, 0.0, n, "stable",
                              f"{h:.1f} mm tall, survives {lam * self.push_n * 1e3:.1f} mN at the top "
                              f"({lam:.2f} x the push); {res.explanation}", {**m, "utilisation": res.utilisation})
        res = analyse(s, self.joint)
        return _overload(asm, res, n, h, lam, m, f"survives only {lam:.2f} x the push")


GOALS = {"cantilever": CantileverGoal, "bridge": BridgeGoal, "tower": TowerGoal}


def reference_cantilever(n: int, length_mm: float = 63.8) -> dict[str, float]:
    """Closed-form reaches for ``n`` parts of length ``length_mm``: a single block (L/2), harmonic stacking of
    unconnected blocks ((L/2) H_n) and the centre-of-mass bound for joined parts (n L / 2: all the length in one
    straight beam with its centre of mass on the edge; stud joints need overlaps, so real builds stay below it)."""
    from .closed_form import harmonic_overhang
    return {"single_block": length_mm / 2, "harmonic_unconnected": harmonic_overhang(n, length_mm),
            "com_bound": n * length_mm / 2}

