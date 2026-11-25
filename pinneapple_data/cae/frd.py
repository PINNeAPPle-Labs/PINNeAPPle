"""CalculiX result file (.frd, ASCII): nodes, elements and the result blocks (DISP, STRESS, CONTACT, NDTEMP, FLUX ...).

Two levels of reading, over the same fixed columns:

* :func:`read_frd` — the ``Mesh`` view: nodes, elements and the last increment of each result in ``mesh.point_data``,
  every increment listed in ``mesh.source``;
* :func:`read_blocks` — every increment of every block, which is what a *verification* needs: a case with two steps
  reads the preload apart from the pull, and a contact state has to be addressable by time.

The contact block ``CONTACT`` (six components per node, written by ``*CONTACT FILE``) has named access through
:func:`contact` — ``opening``, ``slip1``/``slip2``, ``pressure``, ``shear1``/``shear2``, the quantities the
post-processor of PrePoMax plots — and :func:`coulomb_ratio` reduces them to the ``|tau|/p`` that friction needs.

Every value is read **by column**, never by splitting on whitespace: ``ccx`` writes a value and its neighbour with no
separator between them when the first one is wide (``7.64418E-03-1.24413E+00``), so a ``split()`` reads five numbers
where the file has six and every offset after it shifts, silently and plausibly.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .model import CellBlock, Mesh

_FRD_TYPES = {1: ("hexahedron", 8), 2: ("wedge", 6), 3: ("tetra", 4), 4: ("hexahedron20", 20), 5: ("wedge15", 15),
              6: ("tetra10", 10), 7: ("triangle", 3), 8: ("triangle6", 6), 9: ("quad", 4), 10: ("quad8", 8),
              11: ("line", 2), 12: ("line3", 3)}
_UNITS_HINT = {"DISP": "length", "STRESS": "stress", "TOSTRAIN": "1", "NDTEMP": "temperature", "FLUX": "heat flux",
               "FORC": "force", "PE": "1", "ERROR": "%",
               "CONTACT": "length for opening and slip, stress for pressure and shear"}

#: the six components of the ``CONTACT`` block -> the name each one is handed out under
CONTACT_FIELDS = {"COPEN": "opening", "CSLIP1": "slip1", "CSLIP2": "slip2", "CPRESS": "pressure",
                  "CSHEAR1": "shear1", "CSHEAR2": "shear2"}


def _f(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:                                                 # "1.0-100" style exponents
        return float(s[:1] + s[1:].replace("-", "E-").replace("+", "E+")) if s else 0.0


@dataclass
class ResultBlock:
    """One result block at one increment: its ``.frd`` name, its components, and the values per node id.

    A row holds exactly ``len(components)`` values, ``nan`` where the file is shorter than its own header claims:
    ``ccx`` announces four components for ``DISP`` (the fourth being ``ALL``, the magnitude) and writes three under a
    ``*NODE FILE`` deck.
    """
    name: str
    components: list[str] = field(default_factory=list)
    values: dict[int, list[float]] = field(default_factory=dict)
    time: float | None = None

    def column(self, component: str) -> dict[int, float]:
        """``node -> value`` of one component, named either as in the file (``CSHEAR1``) or as handed out (``shear1``)."""
        keys = [k for k, v in CONTACT_FIELDS.items() if v == component]
        label = keys[0] if keys else component.upper()
        if label not in self.components:
            return {}
        k = self.components.index(label)
        return {n: (row[k] if k < len(row) else float("nan")) for n, row in self.values.items()}


def _num(s: str, j: int) -> float:
    """The ``j``-th 12-column value of a data line; ``nan`` when the line ends before it."""
    field = s[13 + 12 * j: 25 + 12 * j]
    return _f(field) if field.strip() else float("nan")


def _rows(lines: list[str], i: int, ncomp: int) -> tuple[dict[int, list[float]], int]:
    """The ``-1``/``-2`` data rows of a block from line ``i``, and the line index just after them."""
    out: dict[int, list[float]] = {}
    last = -1
    n = len(lines)
    while i < n and lines[i][:3] in (" -1", " -2"):
        s = lines[i]
        if s[:3] == " -1":
            last = int(s[3:13])
            out[last] = [_num(s, j) for j in range(ncomp)]
        elif last >= 0:                                                # a long row continues on the next -2 line
            need = ncomp - len(out[last])
            out[last] += [_num(s, j) for j in range(max(0, need))]
        i += 1
    return out, i


def _component_count(head: str) -> int:
    """The number of components a ``-4`` header announces.

    Read from its columns, and from a plain split when the columns do not line up: ``ccx`` writes the name in a
    12-character field, and a deck writer (or a hand-written fixture) may pad it to 10, which moves the count five
    columns left. Reading the number as 0 there would silently turn six contact components into one.
    """
    for slice_ in (head[13:18], head[13:19], head[13:]):
        try:
            return int(slice_)
        except ValueError:
            continue
    parts = head[13:].split()
    try:
        return int(parts[0]) if parts else 0
    except ValueError:
        return 0


def read_blocks(b: bytes) -> list[ResultBlock]:
    """Every result block of the ``.frd``, oldest first, each with the time of the increment it belongs to.

    ``ccx`` writes one row per node of the model — zero where the node carries nothing — so a block comes back whole
    and a field can be asked for by time (:func:`contact`).
    """
    lines = b.decode("latin-1").splitlines()
    i, n = 0, len(lines)
    out: list[ResultBlock] = []
    time: float | None = None
    while i < n:
        ln = lines[i]
        if ln[:7].strip() == "100CL":                                  # the header of an increment
            time = _f(ln[12:24])
            i += 1
            continue
        if ln[:3] != " -4":                                            # only a -4 line starts a block
            i += 1
            continue
        name = ln[5:13].strip()
        ncomp = _component_count(ln)
        comps: list[str] = []
        i += 1
        while i < n and lines[i][:3] == " -5":
            comps.append(lines[i][5:13].strip())
            i += 1
        comps = (comps[:ncomp] or []) if ncomp else comps
        if not comps:
            comps = [f"C{k + 1}" for k in range(ncomp)] or ["ALL"]
            ncomp = len(comps)
        rows, i = _rows(lines, i, ncomp)
        out.append(ResultBlock(name, comps, rows, time))
    return out


def read_frd(b: bytes) -> Mesh:
    """The ``Mesh`` of a ``.frd``: nodes, elements and, for each result, the last increment in ``point_data``."""
    lines = b.decode("latin-1").splitlines()
    i, n = 0, len(lines)
    nid, xyz = [], []
    elems: dict[str, list] = {}
    while i < n:
        ln = lines[i]
        tag = ln[:7].strip()
        if tag == "2C":                                                # nodes
            i += 1
            while i < n and lines[i][:3] == " -1":
                s = lines[i]
                nid.append(int(s[3:13]))
                xyz.append([_f(s[13:25]), _f(s[25:37]), _f(s[37:49])])
                i += 1
            continue
        if tag == "3C":                                                # elements
            i += 1
            while i < n and lines[i][:3] in (" -1", " -2"):
                s = lines[i]
                if s[:3] == " -1":
                    eid, et = int(s[3:13]), int(s[13:18])
                    vt, k = _FRD_TYPES.get(et, (None, 0))
                    nodes: list[int] = []
                    i += 1
                    while i < n and lines[i][:3] == " -2":
                        t = lines[i][3:]
                        nodes += [int(t[j:j + 10]) for j in range(0, len(t), 10) if t[j:j + 10].strip()]
                        i += 1
                    if vt:
                        elems.setdefault(vt, []).append((eid, nodes[:k]))
                    continue
                i += 1
            continue
        i += 1
    if not nid:
        raise ValueError("no nodes in the .frd file")
    nid_a = np.array(nid, np.int64)
    lut = np.full(nid_a.max() + 1, -1, np.int64)
    lut[nid_a] = np.arange(len(nid_a))
    blocks = [CellBlock(vt, lut[np.array([e[1] for e in lst], np.int64)], np.array([e[0] for e in lst], np.int64))
              for vt, lst in elems.items()]
    top = max((b.dim for b in blocks), default=0)
    m = Mesh(points=np.array(xyz, float), blocks=[b for b in blocks if b.dim == top],
             boundary_blocks=[(b.type, b) for b in blocks if b.dim != top], source={"format": "CalculiX .frd"})
    m.source["node_ids"] = nid_a
    seen = []
    for r in read_blocks(b):
        if not r.values or not r.components:
            continue
        ids = list(r.values)
        w = len(r.components)
        arr = np.full((len(nid_a), w), np.nan)
        arr[lut[np.array(ids)]] = np.array([r.values[n][:w] for n in ids], float)
        comps = [c for c in r.components if c.upper() != "ALL"]
        if comps[:3] == ["D1", "D2", "D3"]:                             # DISP: three displacements, not four
            arr, comps = arr[:, :3], comps[:3]
        m.point_data[r.name] = arr[:, 0] if arr.shape[1] == 1 else arr
        m.source.setdefault("components", {})[r.name] = comps
        m.units[r.name] = f"model units ({_UNITS_HINT.get(r.name, 'as in the deck')})"
        seen.append({"name": r.name, "time": r.time})
    m.source["results"] = seen
    if seen:
        m.source["time"] = seen[-1]["time"]
    return m


# ------------------------------------------------------------------------------------------------ contact results
def contact(b: bytes, until: float | None = None) -> dict[int, dict[str, float]]:
    """Contact results per node, from the last ``CONTACT`` block written: ``opening``, ``slip1``, ``slip2``,
    ``pressure``, ``shear1``, ``shear2``.

    ``until`` reads the state at a step time — the last block written at or before it — which is how a case separates
    "what the preload did" from "what the pull did". Empty when the deck asked for no ``*CONTACT FILE``: nothing was
    written to read, and that is not the same as a contact that carried nothing.

    A component the file does not carry reads as ``0.0``, so the record has always the same six keys.
    """
    blocks = [r for r in read_blocks(b) if r.name == "CONTACT"]
    if until is not None:
        blocks = [r for r in blocks if r.time is None or r.time <= until + 1e-9]
    if not blocks:
        return {}
    last = blocks[-1]
    cols = {CONTACT_FIELDS[c]: last.column(c) for c in CONTACT_FIELDS}
    out: dict[int, dict[str, float]] = {}
    for node in last.values:
        out[node] = {name: float(col.get(node, 0.0)) for name, col in cols.items()}
    return out


def coulomb_ratio(rows: dict[int, dict[str, float]], nodes: Iterable[int] | None = None,
                  min_fraction: float = 0.05) -> float | None:
    """The largest ``|tau| / p`` over the loaded nodes: ``0`` while nothing is stuck or sliding, ``mu`` at the limit.

    ``|tau| = hypot(shear1, shear2)`` and ``p`` is the contact pressure of the same node, so the number is exactly
    what the Coulomb cone compares: ``|tau| <= mu * p`` node by node. Nodes below ``min_fraction`` of the peak
    pressure are left out — there the pressure is a rounding error and the ratio it produces says nothing about
    friction. ``None`` when no node carries pressure at all (the interface never closed).
    """
    keep = set(nodes) if nodes is not None else None
    loaded = [(n, r) for n, r in rows.items() if (keep is None or n in keep) and r.get("pressure", 0.0) > 0.0]
    if not loaded:
        return None
    peak = max(r["pressure"] for _, r in loaded)
    ratios = [math.hypot(r.get("shear1", 0.0), r.get("shear2", 0.0)) / r["pressure"]
              for _, r in loaded if r["pressure"] >= min_fraction * peak]
    return max(ratios) if ratios else None


def displacement(b: bytes) -> dict[int, tuple[float, float, float]]:
    """The displacement field of the last step (``DISP`` block): node -> ``(vx, vy, vz)``."""
    last = next((r for r in reversed(read_blocks(b)) if r.name == "DISP" and len(r.components) >= 3), None)
    if last is None:
        return {}
    return {n: (v[0], v[1], v[2]) for n, v in last.values.items() if len(v) >= 3}


def stress(b: bytes) -> dict[int, dict[str, float]]:
    """The nodal stresses of the last step (``STRESS`` block) with the file's own component names.

    Tension is positive, so a pressure comes out negative — the convention the closed forms are written in.
    """
    last = next((r for r in reversed(read_blocks(b)) if r.name == "STRESS"), None)
    if last is None:
        return {}
    return {n: dict(zip(last.components, v, strict=False)) for n, v in last.values.items()}


def coordinates(b: bytes) -> dict[int, tuple[float, float, float]]:
    """The nodal coordinates the file carries, for a deck whose mesh the caller did not build."""
    lines = b.decode("latin-1").splitlines()
    out: dict[int, tuple[float, float, float]] = {}
    for line in lines:
        if line.startswith(" -1") and len(line) >= 49 and line[3:13].strip().lstrip("+-").isdigit():
            try:
                out[int(line[3:13])] = (float(line[13:25]), float(line[25:37]), float(line[37:49]))
            except ValueError:
                continue
        elif line.startswith(" -3"):
            break
    return out
