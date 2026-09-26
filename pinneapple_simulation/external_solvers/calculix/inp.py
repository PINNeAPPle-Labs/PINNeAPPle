"""CalculiX ``.inp`` decks: a small model container, a writer and a reader.

Only the keywords a linear static structural model needs are handled (``*NODE``, ``*ELEMENT``,
``*NSET``/``*ELSET``, ``*MATERIAL`` + ``*ELASTIC`` + ``*DENSITY``, ``*SOLID SECTION``,
``*BOUNDARY``, ``*CLOAD``, ``*STEP``/``*STATIC``, ``*NODE FILE``/``*EL FILE``). Everything else in
a deck written by a pre-processor is ignored by the reader, so an engineer's own deck (PrePoMax,
cgx, FreeCAD FEM) can still be read for its mesh, material, supports and point loads.

Keyword syntax: CalculiX User's Manual (G. Dhondt, www.calculix.de), section 7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalculixModel:
    nodes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    elements: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    element_type: str = "C3D8"
    nsets: Dict[str, List[int]] = field(default_factory=dict)
    material: str = "STEEL"
    E: float = 210e9
    nu: float = 0.3
    density: Optional[float] = None
    boundary: List[Tuple[str, int, int, float]] = field(default_factory=list)  # (nset|node, dof1, dof2, value)
    cloads: List[Tuple[str, int, float]] = field(default_factory=list)  # (nset|node, dof, value)
    title: str = "pinneapple model"

    def node_array(self) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.array(sorted(self.nodes), dtype=np.int64)
        return ids, np.array([self.nodes[i] for i in ids], dtype=float)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def write_inp(model: CalculixModel, path: str, *, job_outputs=("U",), stress_outputs=("S",)) -> str:
    lines = [f"** {model.title}", "*NODE, NSET=NALL"]
    for nid in sorted(model.nodes):
        x, y, z = model.nodes[nid]
        lines.append(f"{nid}, {x:.10g}, {y:.10g}, {z:.10g}")
    lines.append(f"*ELEMENT, TYPE={model.element_type}, ELSET=EALL")
    for eid in sorted(model.elements):
        conn = [str(eid)] + [str(n) for n in model.elements[eid]]
        # data lines hold at most 16 entries; continuation lines end with a comma
        rows = list(_chunks(conn, 16))
        for k, row in enumerate(rows):
            lines.append(", ".join(row) + ("," if k < len(rows) - 1 else ""))
    for name, ids in model.nsets.items():
        lines.append(f"*NSET, NSET={name}")
        for row in _chunks([str(i) for i in ids], 16):
            lines.append(", ".join(row))
    lines += [f"*MATERIAL, NAME={model.material}", "*ELASTIC", f"{model.E:.10g}, {model.nu:.10g}"]
    if model.density is not None:
        lines += ["*DENSITY", f"{model.density:.10g}"]
    lines.append(f"*SOLID SECTION, ELSET=EALL, MATERIAL={model.material}")
    if model.boundary:
        lines.append("*BOUNDARY")
        lines += [f"{t}, {a}, {b}, {v:.10g}" for t, a, b, v in model.boundary]
    lines += ["*STEP", "*STATIC"]
    if model.cloads:
        lines.append("*CLOAD")
        lines += [f"{t}, {d}, {v:.10g}" for t, d, v in model.cloads]
    if job_outputs:
        lines += ["*NODE FILE", ", ".join(job_outputs)]
    if stress_outputs:
        lines += ["*EL FILE", ", ".join(stress_outputs)]
    lines.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _params(line: str) -> Tuple[str, Dict[str, str]]:
    parts = [p.strip() for p in line.split(",")]
    kw = parts[0].upper()
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().upper()] = v.strip()
        elif p:
            params[p.upper()] = ""
    return kw, params


def read_inp(path: str) -> CalculixModel:
    """Read mesh, material, supports and point loads of a deck (``*INCLUDE`` is followed)."""
    import os

    def lines_of(p):
        base = os.path.dirname(os.path.abspath(p))
        for raw in open(p, errors="replace"):
            s = raw.strip()
            if not s or s.startswith("**"):
                continue
            if s.upper().startswith("*INCLUDE"):
                inc = _params(s)[1].get("INPUT")
                if inc:
                    yield from lines_of(os.path.join(base, inc))
                continue
            yield s

    m = CalculixModel(nodes={}, elements={}, nsets={}, boundary=[], cloads=[])
    kw, prm, pending = None, {}, []
    for s in lines_of(path):
        if s.startswith("*"):
            kw, prm = _params(s)
            if kw == "*ELEMENT":
                m.element_type = prm.get("TYPE", m.element_type).upper()
            if kw == "*MATERIAL":
                m.material = prm.get("NAME", m.material)
            if kw in ("*NSET",):
                m.nsets.setdefault(prm.get("NSET", ""), [])
            continue
        vals = [v.strip() for v in s.split(",")]
        if kw == "*NODE":
            m.nodes[int(vals[0])] = tuple(float(v) for v in (vals[1:4] + ["0", "0", "0"])[:3])
            if "NSET" in prm:
                m.nsets.setdefault(prm["NSET"], []).append(int(vals[0]))
        elif kw == "*ELEMENT":
            vals = [v for v in vals if v]
            pending += vals
            if not s.endswith(","):
                m.elements[int(pending[0])] = tuple(int(v) for v in pending[1:])
                pending = []
        elif kw == "*NSET":
            name = prm.get("NSET", "")
            if "GENERATE" in prm:
                a, b, st = (int(v) for v in (vals + ["1"])[:3])
                m.nsets[name] += list(range(a, b + 1, st))
            else:
                m.nsets[name] += [int(v) if v.lstrip("-").isdigit() else v for v in vals if v]
        elif kw == "*ELASTIC":
            m.E, m.nu = float(vals[0]), float(vals[1])
        elif kw == "*DENSITY":
            m.density = float(vals[0])
        elif kw == "*BOUNDARY":
            a = int(vals[1])
            b = int(vals[2]) if len(vals) > 2 and vals[2] else a
            v = float(vals[3]) if len(vals) > 3 and vals[3] else 0.0
            m.boundary.append((vals[0], a, b, v))
        elif kw == "*CLOAD":
            m.cloads.append((vals[0], int(vals[1]), float(vals[2])))
    return m


def expand_target(model: CalculixModel, target: str) -> List[int]:
    """Node ids behind a ``*BOUNDARY``/``*CLOAD`` target (a node id or an NSET name)."""
    t = str(target).strip()
    if t.lstrip("-").isdigit():
        return [int(t)]
    for name, ids in model.nsets.items():
        if name.upper() == t.upper():
            out = []
            for i in ids:
                out += expand_target(model, i) if isinstance(i, str) else [i]
            return out
    raise KeyError(f"unknown node set '{target}'")


def cantilever_hex_mesh(L=1.0, b=0.1, h=0.1, nx=20, ny=2, nz=2, element_type="C3D20R") -> CalculixModel:
    """Brick cantilever along x, fixed at x = 0; C3D8/C3D8I (8 nodes) or C3D20/C3D20R (20 nodes).

    Nodes set ``FIXED`` (x = 0) and ``TIP`` (x = L). Node numbering and element connectivity follow
    CalculiX's C3D8/C3D20 conventions (User's Manual §6.2.4, §6.2.10).
    """
    quad = element_type.upper().startswith("C3D20")
    m = CalculixModel(element_type=element_type.upper(), title=f"cantilever {L}x{b}x{h}")
    rx, ry, rz = (2 * nx, 2 * ny, 2 * nz) if quad else (nx, ny, nz)
    nid = {}
    k = 0
    for i in range(rx + 1):
        for j in range(ry + 1):
            for kk in range(rz + 1):
                if quad and sum(v % 2 for v in (i, j, kk)) > 1:
                    continue  # 20-node bricks: no face- or body-centre nodes
                k += 1
                nid[(i, j, kk)] = k
                m.nodes[k] = (L * i / rx, b * j / ry, h * kk / rz)
    e = 0
    s = 2 if quad else 1
    for i in range(0, rx, s):
        for j in range(0, ry, s):
            for kk in range(0, rz, s):
                c = lambda a, bb, cc: nid[(i + a, j + bb, kk + cc)]
                corners = [c(0, 0, 0), c(s, 0, 0), c(s, s, 0), c(0, s, 0),
                           c(0, 0, s), c(s, 0, s), c(s, s, s), c(0, s, s)]
                if quad:
                    mids = [c(1, 0, 0), c(2, 1, 0), c(1, 2, 0), c(0, 1, 0),
                            c(1, 0, 2), c(2, 1, 2), c(1, 2, 2), c(0, 1, 2),
                            c(0, 0, 1), c(2, 0, 1), c(2, 2, 1), c(0, 2, 1)]
                    corners += mids
                e += 1
                m.elements[e] = tuple(corners)
    m.nsets["FIXED"] = [n for (i, _, _), n in nid.items() if i == 0]
    m.nsets["TIP"] = [n for (i, _, _), n in nid.items() if i == rx]
    return m
