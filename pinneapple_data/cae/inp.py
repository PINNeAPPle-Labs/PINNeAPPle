"""Abaqus / CalculiX input deck (.inp): keyword blocks, nodes, elements (to VTK types), node and element sets.

``read_deck`` keeps every keyword block with its line number, so the preflight rules can point at the line of a
problem; ``deck_mesh`` turns the deck into a ``Mesh``. *INCLUDE files are resolved from the uploaded files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import CellBlock, ELEMENTS, Mesh

# Abaqus/CalculiX element families -> VTK type (corner-first ordering is the same for all of these)
_ETYPES = [
    (r"^D?C3D4[HT]?$", "tetra"), (r"^D?C3D10[A-Z]*$", "tetra10"), (r"^D?C3D8[A-Z]*$", "hexahedron"),
    (r"^D?C3D20[A-Z]*$", "hexahedron20"), (r"^D?C3D6[A-Z]*$", "wedge"), (r"^D?C3D15[A-Z]*$", "wedge15"),
    (r"^D?C3D5[A-Z]*$", "pyramid"),
    (r"^(S3|STRI3|M3D3|SFM3D3|DS3|(CPS|CPE|CAX|CPEG|DC2D|DCAX)3[A-Z]*)$", "triangle"),
    (r"^(S6|STRI65|M3D6|DS6|(CPS|CPE|CAX|DC2D|DCAX)6[A-Z]*)$", "triangle6"),
    (r"^(S4[A-Z]*|M3D4[A-Z]*|SFM3D4[A-Z]*|DS4|(CPS|CPE|CAX|CPEG|DC2D|DCAX)4[A-Z]*)$", "quad"),
    (r"^(S8[A-Z]*|M3D8[A-Z]*|DS8|(CPS|CPE|CAX|DC2D|DCAX)8[A-Z]*)$", "quad8"),
    (r"^(B31[A-Z]*|B21[A-Z]*|T3D2[A-Z]*|T2D2[A-Z]*|D)$", "line"), (r"^(B32[A-Z]*|B22|T3D3|T2D3)$", "line3"),
]


def vtk_type(abq: str) -> Optional[str]:
    t = abq.upper()
    for pat, v in _ETYPES:
        if re.match(pat, t):
            return v
    return None


@dataclass
class Block:
    keyword: str                    # upper case, without '*'
    params: Dict[str, str]          # upper-case keys, raw values
    data: List[List[str]]           # comma-split data lines
    line: int                       # 1-based line of the keyword in its file
    file: str


@dataclass
class Deck:
    blocks: List[Block] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    missing_includes: List[str] = field(default_factory=list)

    def find(self, kw: str) -> List[Block]:
        return [b for b in self.blocks if b.keyword == kw]


def _kw(line: str) -> Tuple[str, Dict[str, str]]:
    parts = [p.strip() for p in line[1:].split(",")]
    params = {}
    for p in parts[1:]:
        if not p:
            continue
        k, _, v = p.partition("=")
        params[k.strip().upper()] = v.strip()
    return parts[0].upper(), params


def read_deck(text: str, name: str = "model.inp", files: Optional[Dict[str, str]] = None, _depth: int = 0) -> Deck:
    deck = Deck()
    lines = text.splitlines()
    cur: Optional[Block] = None
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            kw, prm = _kw(s)
            if kw == "INCLUDE":
                fn = prm.get("INPUT", "").strip('"').replace("\\", "/").split("/")[-1]
                deck.includes.append(fn)
                if files and fn in files and _depth < 5:
                    sub = read_deck(files[fn], fn, files, _depth + 1)
                    deck.blocks += sub.blocks
                    deck.includes += sub.includes
                    deck.missing_includes += sub.missing_includes
                else:
                    deck.missing_includes.append(fn)
                cur = None
                continue
            cur = Block(kw, prm, [], i, name)
            deck.blocks.append(cur)
        elif cur is not None:
            cur.data.append([x.strip() for x in s.rstrip(",").split(",")])
    return deck


def _set_members(b: Block) -> List[str]:
    vals = [x for row in b.data for x in row if x]
    if "GENERATE" in b.params:
        out = []
        for row in b.data:
            nums = [int(float(x)) for x in row if x]
            if len(nums) >= 2:
                a, z = nums[0], nums[1]
                st = nums[2] if len(nums) > 2 else 1
                out += [str(k) for k in range(a, z + 1, st)]
        return out
    return vals


def resolve_sets(deck: Deck, kind: str) -> Dict[str, List[str]]:
    """NSET or ELSET name (upper case) -> member labels; sets may contain other sets' names."""
    raw: Dict[str, List[str]] = {}
    for b in deck.find(kind):
        name = b.params.get(kind, "").upper()
        raw.setdefault(name, []).extend(_set_members(b))
    if kind == "ELSET":                                                # *ELEMENT, ELSET=... adds members too
        for b in deck.find("ELEMENT"):
            if "ELSET" in b.params:
                raw.setdefault(b.params["ELSET"].upper(), []).extend(r[0] for r in b.data if r and r[0])
    if kind == "NSET":
        for b in deck.find("NODE"):
            if "NSET" in b.params:
                raw.setdefault(b.params["NSET"].upper(), []).extend(r[0] for r in b.data if r and r[0])
    out: Dict[str, List[str]] = {}

    def expand(name: str, depth: int = 0) -> List[str]:
        if name in out:
            return out[name]
        res = []
        for m in raw.get(name, []):
            if re.fullmatch(r"-?\d+", m):
                res.append(m)
            elif depth < 10 and m.upper() in raw:
                res += expand(m.upper(), depth + 1)
        out[name] = res
        return res
    for k in raw:
        expand(k)
    return out


def deck_mesh(deck: Deck) -> Mesh:
    node_ids, coords = [], []
    for b in deck.find("NODE"):
        for r in b.data:
            if len(r) >= 3 and r[0]:
                node_ids.append(int(float(r[0])))
                xyz = [float(x) if x else 0.0 for x in r[1:4]] + [0.0] * (3 - len(r[1:4]))
                coords.append(xyz[:3])
    if not node_ids:
        raise ValueError("the deck has no *NODE data")
    nid = np.array(node_ids, np.int64)
    pts = np.array(coords, float)
    lut = np.full(nid.max() + 1, -1, np.int64)
    lut[nid] = np.arange(len(nid))
    by_type: Dict[str, Tuple[List[int], List[List[int]]]] = {}
    skipped: Dict[str, int] = {}
    elset_of: Dict[int, str] = {}
    for b in deck.find("ELEMENT"):
        abq = b.params.get("TYPE", "").upper()
        vt = vtk_type(abq)
        rows: List[List[int]] = []
        cur: List[int] = []
        want = None
        for r in b.data:                                               # element rows may continue on next lines
            vals = [int(float(x)) for x in r if x]
            cur += vals
            if vt is not None:
                want = _nodes_for(vt)
                if want and len(cur) >= want + 1:
                    rows.append(cur[: want + 1])
                    cur = []
            else:
                rows.append(cur)
                cur = []
        if vt is None:
            skipped[abq or "?"] = skipped.get(abq or "?", 0) + len(rows)
            continue
        ids, conn = by_type.setdefault(vt, ([], []))
        for r in rows:
            ids.append(r[0])
            conn.append(r[1:])
            if "ELSET" in b.params:
                elset_of[r[0]] = b.params["ELSET"].upper()
    blocks = []
    for vt, (ids, conn) in by_type.items():
        c = np.array(conn, np.int64)
        if (c > lut.size - 1).any() or (lut[c] < 0).any():
            bad = sorted(set(c[(c > lut.size - 1) | (lut[np.minimum(c, lut.size - 1)] < 0)].tolist()))[:10]
            raise ValueError(f"elements reference undefined nodes: {bad}")
        blocks.append(CellBlock(vt, lut[c], np.array(ids, np.int64)))
    top = max((ELEMENTS[b.type][1] for b in blocks), default=0)
    m = Mesh(points=pts, blocks=[b for b in blocks if b.dim == top],
             boundary_blocks=[(b.type, b) for b in blocks if b.dim != top], source={"format": "Abaqus/CalculiX .inp"})
    m.source["node_ids"] = nid
    m.source["skipped_elements"] = skipped
    # element sets -> global cell indices
    order = np.concatenate([b.ids for b in m.blocks]) if m.blocks else np.zeros(0, np.int64)
    pos = {int(e): i for i, e in enumerate(order)}
    for name, members in resolve_sets(deck, "ELSET").items():
        idx = [pos[int(x)] for x in members if int(x) in pos]
        if idx:
            m.cell_sets[name] = np.array(idx, np.int64)
    for name, members in resolve_sets(deck, "NSET").items():
        idx = [int(lut[int(x)]) for x in members if int(x) < lut.size and lut[int(x)] >= 0]
        m.node_sets[name] = np.array(idx, np.int64)
    return m


def _nodes_for(vt: str) -> int:
    return {"tetra": 4, "tetra10": 10, "hexahedron": 8, "hexahedron20": 20, "wedge": 6, "wedge15": 15,
            "pyramid": 5, "triangle": 3, "triangle6": 6, "quad": 4, "quad8": 8, "line": 2, "line3": 3}[vt]
