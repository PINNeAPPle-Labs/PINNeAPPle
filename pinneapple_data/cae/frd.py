"""CalculiX result file (.frd, ASCII): nodes, elements and the result blocks (DISP, STRESS, NDTEMP, FLUX, ...).

The last increment of each result is kept (``mesh.point_data``), all increments are listed in ``mesh.source``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .model import CellBlock, Mesh

_FRD_TYPES = {1: ("hexahedron", 8), 2: ("wedge", 6), 3: ("tetra", 4), 4: ("hexahedron20", 20), 5: ("wedge15", 15),
              6: ("tetra10", 10), 7: ("triangle", 3), 8: ("triangle6", 6), 9: ("quad", 4), 10: ("quad8", 8),
              11: ("line", 2), 12: ("line3", 3)}
_UNITS_HINT = {"DISP": "length", "STRESS": "stress", "TOSTRAIN": "1", "NDTEMP": "temperature", "FLUX": "heat flux",
               "FORC": "force", "PE": "1", "ERROR": "%"}


def _f(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:                                                 # "1.0-100" style exponents
        return float(s[:1] + s[1:].replace("-", "E-").replace("+", "E+")) if s else 0.0


def read_frd(b: bytes) -> Mesh:
    lines = b.decode("latin-1").splitlines()
    i, n = 0, len(lines)
    nid, xyz = [], []
    elems: Dict[str, List] = {}
    results: List[Dict] = []
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
                    nodes: List[int] = []
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
        if tag == "100CL":                                            # a result block
            step_val = _f(ln[12:24])
            i += 1
            head = lines[i]
            name = head[5:13].strip()
            ncomp = int(head[13:18])
            comps = []
            i += 1
            while i < n and lines[i][:3] == " -5":
                comps.append(lines[i][5:13].strip())
                i += 1
            ids, vals = [], []
            while i < n and lines[i][:3] in (" -1", " -2"):
                s = lines[i]
                if s[:3] == " -1":
                    ids.append(int(s[3:13]))
                    vals.append([_f(s[13 + 12 * j: 25 + 12 * j]) for j in range((len(s) - 13) // 12)])
                else:
                    vals[-1] += [_f(s[13 + 12 * j: 25 + 12 * j]) for j in range((len(s) - 13) // 12)]
                i += 1
            results.append({"name": name, "time": step_val, "components": comps[:ncomp], "ids": ids, "values": vals})
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
    for r in results:
        w = len(r["components"])
        if not r["ids"] or w == 0:
            continue
        arr = np.full((len(nid_a), w), np.nan)
        v = np.array([row[:w] + [np.nan] * (w - len(row[:w])) for row in r["values"]], float)
        arr[lut[np.array(r["ids"])]] = v
        comps = [c for c in r["components"] if c.upper() != "ALL"]
        if r["name"] == "DISP" or (w == 4 and comps[:3] in (["D1", "D2", "D3"],)):
            arr = arr[:, :3]
            comps = comps[:3]
        m.point_data[r["name"]] = arr[:, 0] if arr.shape[1] == 1 else arr
        m.source.setdefault("components", {})[r["name"]] = comps
        m.units[r["name"]] = f"model units ({_UNITS_HINT.get(r['name'], 'as in the deck')})"
        seen.append({"name": r["name"], "time": r["time"]})
    m.source["results"] = seen
    if seen:
        m.source["time"] = seen[-1]["time"]
    return m
