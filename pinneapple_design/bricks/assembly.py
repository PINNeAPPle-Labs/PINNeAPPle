"""Grid assemblies of interlocking bricks: placement, stud-tube connectivity and validity rules.

Coordinates: ``x``, ``y`` in stud pitches (8 mm), ``z`` in plate heights (3.2 mm), z up, the ground at ``z = 0``.
A brick at ``(x, y, z)`` with footprint ``sx`` x ``sy`` covers the cells ``[x, x + sx) x [y, y + sy)`` and the
plate layers ``[z, z + h)``; ``rot = 1`` turns it a quarter turn about z (``sx, sy = ny, nx``).

Two bricks are connected where the top of the lower one (``z_a + h_a = z_b``) overlaps the footprint of the upper
one: every overlapping cell is one stud-tube contact. A brick on ``z = 0`` rests on the ground, which is either
a table (smooth, friction only) or a baseplate (studs everywhere), optionally limited to some rectangles (a table
edge, the two banks of a gap).
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .parts import BRICK_MM, PITCH_MM, PLATE_MM, PartType, colour, part

Rect = tuple[float, float, float, float]          # x0, x1, y0, y1 in stud units


@dataclass
class Brick:
    part: PartType
    x: int
    y: int
    z: int
    rot: int = 0
    colour: str = "red"

    @property
    def sx(self) -> int:
        return self.part.ny if self.rot % 2 else self.part.nx

    @property
    def sy(self) -> int:
        return self.part.nx if self.rot % 2 else self.part.ny

    @property
    def top(self) -> int:
        return self.z + self.part.h

    def footprint(self) -> Rect:
        return (self.x, self.x + self.sx, self.y, self.y + self.sy)

    def cells(self) -> Iterable[tuple[int, int, int]]:
        for i in range(self.x, self.x + self.sx):
            for j in range(self.y, self.y + self.sy):
                for k in range(self.z, self.top):
                    yield (i, j, k)

    def box_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper corners of the body in millimetres (cell bounds, without the 0.1 mm clearance)."""
        lo = np.array([self.x * PITCH_MM, self.y * PITCH_MM, self.z * PLATE_MM])
        hi = np.array([(self.x + self.sx) * PITCH_MM, (self.y + self.sy) * PITCH_MM, self.top * PLATE_MM])
        return lo, hi

    def com_mm(self) -> np.ndarray:
        lo, hi = self.box_mm()
        return 0.5 * (lo + hi)

    def key(self) -> tuple:
        return (self.part.name, self.x, self.y, self.z, self.rot % 2)

    def to_dict(self) -> dict:
        return {"part": self.part.name, "x": self.x, "y": self.y, "z": self.z, "rot": self.rot % 2,
                "colour": self.colour}


@dataclass
class Ground:
    """``kind``: ``"table"`` (smooth, friction ``mu``) or ``"baseplate"`` (studs). ``regions``: rectangles in stud
    units where the ground exists (``None`` = everywhere)."""

    kind: str = "table"
    regions: list[Rect] | None = None

    def patches(self, rect: Rect) -> list[Rect]:
        if self.regions is None:
            return [rect]
        out = []
        for r in self.regions:
            o = _overlap(rect, r)
            if o is not None:
                out.append(o)
        return out

    def to_dict(self) -> dict:
        return {"kind": self.kind, "regions": None if self.regions is None else [list(r) for r in self.regions]}


@dataclass
class Connection:
    """Contact between ``lower`` (brick index, ``-1`` = ground) and ``upper`` at plate height ``z``.

    ``rect``: overlap of the two footprints [stud units]; ``studs``: the cells (stud centres at ``i + 0.5``) where a
    stud of the lower part enters the upper one (empty for a smooth contact)."""

    lower: int
    upper: int
    z: int
    rect: Rect
    studs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def clutch(self) -> bool:
        return len(self.studs) > 0


def _overlap(a: Rect, b: Rect) -> Rect | None:
    x0, x1 = max(a[0], b[0]), min(a[1], b[1])
    y0, y1 = max(a[2], b[2]), min(a[3], b[3])
    if x1 - x0 <= 1e-9 or y1 - y0 <= 1e-9:
        return None
    return (x0, x1, y0, y1)


class Assembly:
    """A list of bricks on a stud grid plus the ground they stand on.

    >>> a = Assembly()
    >>> a.add("brick_2x4", 0, 0, 0)
    0
    >>> a.add("brick_2x2", 1, 0, 3, colour="blue")
    1
    >>> [(c.lower, c.upper, len(c.studs)) for c in a.connections()]
    [(-1, 0, 0), (0, 1, 4)]
    """

    def __init__(self, bricks: Sequence[Brick] | None = None, ground: Ground | None = None, name: str = ""):
        self.bricks: list[Brick] = list(bricks or [])
        self.ground = ground or Ground()
        self.name = name

    # ------------------------------------------------------------------ editing
    def add(self, part_name, x: int, y: int, z: int, rot: int = 0, colour: str = "red") -> int:
        p = part_name if isinstance(part_name, PartType) else part(part_name)
        self.bricks.append(Brick(p, int(x), int(y), int(z), int(rot) % 2, colour))
        return len(self.bricks) - 1

    def remove(self, i: int) -> Brick:
        return self.bricks.pop(i)

    def copy(self) -> Assembly:
        return Assembly([copy.copy(b) for b in self.bricks], copy.deepcopy(self.ground), self.name)

    def __len__(self) -> int:
        return len(self.bricks)

    # ------------------------------------------------------------------ geometry
    def occupancy(self) -> dict[tuple[int, int, int], list[int]]:
        occ: dict[tuple[int, int, int], list[int]] = {}
        for i, b in enumerate(self.bricks):
            for c in b.cells():
                occ.setdefault(c, []).append(i)
        return occ

    def overlaps(self) -> list[tuple[int, int]]:
        """Pairs of bricks that share a cell (an impossible build)."""
        pairs = set()
        for ids in self.occupancy().values():
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    pairs.add((ids[a], ids[b]))
        return sorted(pairs)

    def connections(self) -> list[Connection]:
        """Ground contacts and brick-on-brick contacts, with the stud cells of each."""
        out: list[Connection] = []
        by_top: dict[int, list[int]] = {}
        for i, b in enumerate(self.bricks):
            by_top.setdefault(b.top, []).append(i)
        for i, b in enumerate(self.bricks):
            fp = b.footprint()
            if b.z == 0:
                for r in self.ground.patches(fp):
                    studs = []
                    if self.ground.kind == "baseplate" and not b.part.smooth:
                        studs = _cells(r)
                    out.append(Connection(-1, i, 0, r, studs))
            for j in by_top.get(b.z, []):
                lower = self.bricks[j]
                r = _overlap(lower.footprint(), fp)
                if r is None:
                    continue
                studs = [] if (lower.part.smooth or b.part.smooth) else _cells(r)
                out.append(Connection(j, i, b.z, r, studs))
        return out

    def graph(self) -> dict[int, set[int]]:
        """Adjacency of the contact graph (node ``-1`` is the ground)."""
        adj: dict[int, set[int]] = {i: set() for i in range(-1, len(self.bricks))}
        for c in self.connections():
            adj[c.lower].add(c.upper)
            adj[c.upper].add(c.lower)
        return adj

    def grounded(self) -> set[int]:
        """Bricks with a contact path to the ground."""
        adj, seen, todo = self.graph(), {-1}, [-1]
        while todo:
            n = todo.pop()
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return seen - {-1}

    def components(self) -> list[set[int]]:
        """Groups of bricks held together by stud joints (the rigid clusters when no joint breaks)."""
        parent = list(range(len(self.bricks)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for c in self.connections():
            if c.lower >= 0 and c.clutch:
                parent[find(c.lower)] = find(c.upper)
        groups: dict[int, set[int]] = {}
        for i in range(len(self.bricks)):
            groups.setdefault(find(i), set()).add(i)
        return list(groups.values())

    def validate(self) -> list[str]:
        """Rule violations (empty list = a buildable, supported assembly)."""
        issues = []
        for a, b in self.overlaps():
            issues.append(f"overlap: brick {a} and brick {b} share a cell")
        for i, b in enumerate(self.bricks):
            if b.z < 0:
                issues.append(f"brick {i} is below the ground")
            elif b.z == 0 and not self.ground.patches(b.footprint()) and self.ground.regions is not None:
                issues.append(f"brick {i} stands on z = 0 outside the ground regions")
        free = set(range(len(self.bricks))) - self.grounded()
        for i in sorted(free):
            issues.append(f"unsupported: brick {i} has no contact path to the ground")
        return issues

    def is_valid(self) -> bool:
        return not self.validate()

    # ------------------------------------------------------------------ summaries
    def mass(self) -> float:
        return float(sum(b.part.mass for b in self.bricks))

    def com_mm(self) -> np.ndarray:
        m = np.array([b.part.mass for b in self.bricks])
        c = np.array([b.com_mm() for b in self.bricks])
        return (m[:, None] * c).sum(0) / m.sum()

    def bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        boxes = [b.box_mm() for b in self.bricks]
        return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)

    def height_mm(self) -> float:
        return max((b.top for b in self.bricks), default=0) * PLATE_MM

    def part_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in self.bricks:
            k = f"{b.part.name} {b.colour}"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    def signature(self) -> str:
        """Order-independent hash of the bricks (colour excluded: it does not change the physics)."""
        keys = sorted(b.key() for b in self.bricks)
        return hashlib.sha1(json.dumps([keys, self.ground.to_dict()]).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {"name": self.name, "ground": self.ground.to_dict(), "bricks": [b.to_dict() for b in self.bricks]}

    @classmethod
    def from_dict(cls, d: dict) -> Assembly:
        g = d.get("ground", {})
        regions = g.get("regions")
        ground = Ground(g.get("kind", "table"), None if regions is None else [tuple(r) for r in regions])
        a = cls(ground=ground, name=d.get("name", ""))
        for b in d["bricks"]:
            a.add(b["part"], b["x"], b["y"], b["z"], b.get("rot", 0), b.get("colour", "red"))
        return a

    def __repr__(self) -> str:
        return f"Assembly({len(self.bricks)} bricks, ground={self.ground.kind}, height={self.height_mm():.1f} mm)"


def _cells(r: Rect) -> list[tuple[int, int]]:
    return [(i, j) for i in range(int(round(r[0])), int(round(r[1]))) for j in range(int(round(r[2])), int(round(r[3])))]


def colour_rgb(name: str) -> tuple[float, float, float]:
    return colour(name).rgb


def stack(part_name: str, n: int, *, offsets: Sequence[int] | None = None, colours: Sequence[str] = ("red",),
          ground: Ground | None = None) -> Assembly:
    """``n`` copies of a part stacked straight up, brick ``k`` shifted by ``offsets[k]`` studs along x."""
    a = Assembly(ground=ground)
    p = part(part_name)
    for k in range(n):
        a.add(p, offsets[k] if offsets is not None else 0, 0, k * p.h, colour=colours[k % len(colours)])
    return a


__all__ = ["Assembly", "Brick", "Ground", "Connection", "stack", "colour_rgb", "BRICK_MM", "PLATE_MM", "PITCH_MM"]
