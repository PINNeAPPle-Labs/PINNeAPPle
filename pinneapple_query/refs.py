"""Stable physical references: where a quantity lives, independent of any mesh.

A reference names a *component* and a coordinate local to it, never a node or cell number, so it survives
remeshing, refinement and a change of method:

* ``"die@0.31,0.72"``: the die's junction surface at fractional in-plane coordinates (u, v) in [0, 1];
* ``"die@0.31,0.72,0.5"``: the same point at fractional height 0.5 through the die;
* ``"ubump[12,40]"``: the microbump of block (12, 40) (likewise ``tsv[...]``, ``c4[...]``);
* ``"interface:die/tim"``: the whole die/TIM interface; ``"interface:die/tim[3:9,0:4]"`` a block range of it;
* ``"region:die[3:9,0:4]"``: the junction surface over a block range;
* ``"lid:top"``: the cooled top face.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalRef:
    component: str
    local: tuple[float, ...] | None = None     # fractional coordinates in the component frame
    index: tuple[int, ...] | None = None       # an instance of a repeated feature (block indices)
    span: tuple[tuple[int, int], ...] | None = None   # a block range (half-open)
    kind: str = "point"                        # "point" | "feature" | "region" | "interface" | "face"

    def __str__(self) -> str:
        s = self.component
        if self.kind == "interface":
            s = "interface:" + s
        elif self.kind == "region":
            s = "region:" + s
        if self.index is not None:
            s += "[" + ",".join(map(str, self.index)) + "]"
        if self.span is not None:
            s += "[" + ",".join(f"{a}:{b}" for a, b in self.span) + "]"
        if self.local is not None:
            s += "@" + ",".join(f"{x:g}" for x in self.local)
        return s


_SPAN = re.compile(r"^(?P<name>[\w/]+)\[(?P<body>[^\]]*)\]$")


def _span_or_index(name: str):
    m = _SPAN.match(name)
    if not m:
        return name, None, None
    parts = [p.strip() for p in m["body"].split(",")]
    if all(":" in p for p in parts):
        return m["name"], None, tuple(tuple(int(x) for x in p.split(":")) for p in parts)
    return m["name"], tuple(int(p) for p in parts), None


def parse_ref(where) -> PhysicalRef:
    """Parse the string forms of the module docstring (a :class:`PhysicalRef` passes through)."""
    if isinstance(where, PhysicalRef):
        return where
    s = str(where).strip()
    kind = "point"
    if s.startswith("interface:"):
        kind, s = "interface", s[len("interface:"):]
    elif s.startswith("region:"):
        kind, s = "region", s[len("region:"):]
    local = None
    if "@" in s:
        s, loc = s.split("@", 1)
        local = tuple(float(x) for x in loc.split(","))
        if any(not 0.0 <= x <= 1.0 for x in local):
            raise ValueError(f"local coordinates are fractions in [0, 1]: {where!r}")
    if ":" in s and kind == "point" and "[" not in s:
        comp, face = s.split(":", 1)
        return PhysicalRef(comp, None, None, None, "face:" + face)
    name, index, span = _span_or_index(s)
    if index is not None and kind == "point":
        kind = "feature"
    if kind == "point" and local is None:
        kind = "region"
    return PhysicalRef(name, local, index, span, kind)
