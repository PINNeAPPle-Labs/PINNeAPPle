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


_DOF = {1: "ux", 2: "uy", 3: "uz", 4: "rx", 5: "ry", 6: "rz", 11: "T"}
_BODY_LOADS = {"GRAV", "CENTRIF", "NEWTON", "CORIO", "BX", "BY", "BZ", "BF", "BFNU"}


def _num(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def deck_conditions(deck: Deck) -> List[Dict[str, object]]:
    """The supports and loads of a deck as node ids (labels of ``*NODE``), one entry per keyword block and kind:
    ``*BOUNDARY`` (kind ``bc``: the constrained degrees of freedom and their values), ``*CLOAD`` / ``*CFLUX`` (kind
    ``load``: the sum of the nodal forces or fluxes per direction), ``*DLOAD`` / ``*DFLUX`` / ``*FILM`` on element sets or
    elements (kind ``load``: the nodes of those elements, the load type and magnitude). Body loads on whole sets (GRAV,
    CENTRIF) carry no nodes and are listed with ``nodes == []``. Each entry: ``name``, ``kind``, ``keyword``, ``line``,
    ``node_ids``, ``detail``."""
    nsets, elsets = resolve_sets(deck, "NSET"), resolve_sets(deck, "ELSET")
    el_nodes: Dict[int, List[int]] = {}
    for b in deck.find("ELEMENT"):
        vt = vtk_type(b.params.get("TYPE", ""))
        want = _nodes_for(vt) if vt else None
        cur: List[int] = []
        for r in b.data:
            cur += [int(float(x)) for x in r if x]
            if want and len(cur) >= want + 1:
                el_nodes[cur[0]] = cur[1:want + 1]
                cur = []

    def targets(t: str, sets: Dict[str, List[str]]) -> List[int]:
        t = t.strip()
        if re.fullmatch(r"-?\d+", t):
            return [int(t)]
        return [int(x) for x in sets.get(t.upper(), [])]

    out: List[Dict[str, object]] = []
    for b in deck.blocks:
        kw = b.keyword
        rows = [r for r in b.data if r and r[0]]
        if kw == "BOUNDARY" and rows:
            nodes, dofs, nonzero = [], set(), []
            for r in rows:
                nodes += targets(r[0], nsets)
                d1 = int(_num(r[1]) or 0) if len(r) > 1 else 0
                d2 = int(_num(r[2]) or d1) if len(r) > 2 and r[2] else d1
                v = _num(r[3]) if len(r) > 3 else 0.0
                for d in range(d1, d2 + 1):
                    dofs.add(d)
                    if v:
                        nonzero.append(f"{_DOF.get(d, d)} = {v:g}")
            names = ", ".join(_DOF.get(d, str(d)) for d in sorted(dofs))
            out.append({"name": "fixed " + names if not nonzero else "prescribed " + names, "kind": "bc",
                        "keyword": "*BOUNDARY", "line": b.line, "node_ids": sorted(set(nodes)),
                        "detail": (", ".join(sorted(set(nonzero))) if nonzero else names + " = 0")})
        elif kw in ("CLOAD", "CFLUX") and rows:
            nodes, tot = [], {}
            for r in rows:
                ns = targets(r[0], nsets)
                nodes += ns
                d, v = int(_num(r[1]) or 0) if len(r) > 1 else 0, _num(r[2]) if len(r) > 2 else 0.0
                tot[d] = tot.get(d, 0.0) + (v or 0.0) * len(ns)
            sym = "F" if kw == "CLOAD" else "Q"
            det = ", ".join(f"{sym}{'xyz'[d - 1] if 1 <= d <= 3 else d} = {v:.4g}" for d, v in sorted(tot.items()) if v)
            out.append({"name": f"{'load' if kw == 'CLOAD' else 'heat flux'} {det or '0'}", "kind": "load",
                        "keyword": f"*{kw}", "line": b.line, "node_ids": sorted(set(nodes)),
                        "detail": f"sum over {len(set(nodes))} nodes: {det or '0'}"})
        elif kw in ("DLOAD", "DFLUX", "FILM") and rows:
            nodes, kinds = [], []
            for r in rows:
                els = targets(r[0], elsets)
                lt = r[1].strip().upper() if len(r) > 1 else ""
                kinds.append(" ".join(x for x in r[1:3] if x))
                if lt in _BODY_LOADS:
                    continue                                             # GRAV, CENTRIF: a body load, no surface
                for e in els:
                    nodes += el_nodes.get(e, [])
            out.append({"name": f"{kw.lower()} {kinds[0]}" + (" …" if len(set(kinds)) > 1 else ""), "kind": "load",
                        "keyword": f"*{kw}", "line": b.line, "node_ids": sorted(set(nodes)),
                        "detail": "; ".join(dict.fromkeys(kinds)) + ("" if nodes else " (body load: no region)")})
    return out
