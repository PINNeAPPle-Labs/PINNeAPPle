"""CalculiX ``.frd`` results: reader (ASCII) and writer.

The writer is the heart of the drop-in idea: a PINN or surrogate that predicts displacements
(and stresses) on the mesh of an engineer's ``.inp`` can write a ``.frd`` that cgx, PrePoMax,
ccx2paraview or any FRD reader opens exactly like a CalculiX result. Format: CalculiX GraphiX
manual, "Result Format" (fixed-width ASCII records: ``2C`` nodes, ``3C`` elements, ``100CL``
result blocks with ``-4`` header, ``-5`` components, ``-1`` data lines, ``-3`` end).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# CalculiX element type ids in .frd element blocks (GraphiX manual): he8=1, he20=4, te4=3, te10=6
_FRD_ETYPE = {"C3D8": 1, "C3D8I": 1, "C3D8R": 1, "C3D20": 4, "C3D20R": 4, "C3D4": 3, "C3D10": 6}


@dataclass
class FrdResults:
    node_ids: np.ndarray
    coords: np.ndarray
    steps: List[Dict[str, np.ndarray]] = field(default_factory=list)  # [{"DISP": (N,3), "STRESS": (N,6)}]

    def field(self, name: str, step: int = -1) -> np.ndarray:
        return self.steps[step][name]


def _fixed_floats(s: str, width: int = 12) -> List[float]:
    return [float(s[i:i + width]) for i in range(0, len(s.rstrip()), width) if s[i:i + width].strip()]


def read_frd(path: str) -> FrdResults:
    lines = open(path, errors="replace").read().splitlines()
    ids, xyz = [], []
    steps: List[Dict[str, np.ndarray]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("2C"):
            i += 1
            while i < len(lines) and not lines[i].startswith(" -3"):
                r = lines[i]
                if r.startswith(" -1"):
                    ids.append(int(r[3:13]))
                    xyz.append(_fixed_floats(r[13:]))
                i += 1
        elif ln.strip().startswith("100C"):
            i += 1
            head = lines[i]
            name = head[5:13].strip()
            ncomp = int(head[13:18])
            i += 1
            while lines[i].startswith(" -5"):
                i += 1
            vals: Dict[int, List[float]] = {}
            last = None
            while i < len(lines) and not lines[i].startswith(" -3"):
                r = lines[i]
                if r.startswith(" -1"):
                    last = int(r[3:13])
                    vals[last] = _fixed_floats(r[13:])
                elif r.startswith(" -2") and last is not None:  # continuation of a long record
                    vals[last] += _fixed_floats(r[13:])
                i += 1
            order = {n: k for k, n in enumerate(ids)}
            arr = np.full((len(ids), ncomp), np.nan)
            for n, v in vals.items():
                if n in order:
                    arr[order[n], :len(v[:ncomp])] = v[:ncomp]
            if not steps or name in steps[-1]:
                steps.append({})
            steps[-1][name] = arr
        i += 1
    return FrdResults(np.array(ids, dtype=np.int64), np.array(xyz, dtype=float), steps)


def _rec(n: int, values) -> str:
    return " -1" + f"{n:10d}" + "".join(f"{v:12.5E}" for v in values)


def write_frd(path: str, node_ids, coords, elements: Optional[Dict[int, tuple]] = None,
              element_type: str = "C3D8", fields: Optional[Dict[str, np.ndarray]] = None,
              step_time: float = 1.0, source: str = "pinneapple") -> str:
    """Write an ASCII ``.frd`` with nodes (+ elements) and ``DISP`` (N,3) / ``STRESS`` (N,6) blocks."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    coords = np.asarray(coords, dtype=float)
    out = ["    1C" + source[:66], f"    1UUSER{'':<66}", f"    2C{len(node_ids):30d}{'':37}1"]
    out += [_rec(int(n), c) for n, c in zip(node_ids, coords)]
    out.append(" -3")
    if elements:
        et = _FRD_ETYPE.get(element_type.upper(), 1)
        out.append(f"    3C{len(elements):30d}{'':37}1")
        for eid, conn in sorted(elements.items()):
            out.append(f" -1{eid:10d}{et:5d}{0:5d}{1:5d}")
            for chunk in (conn[i:i + 10] for i in range(0, len(conn), 10)):
                out.append(" -2" + "".join(f"{n:10d}" for n in chunk))
        out.append(" -3")
    comps = {"DISP": ["D1", "D2", "D3"], "STRESS": ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"]}
    for name, arr in (fields or {}).items():
        arr = np.asarray(arr, dtype=float)
        names = comps.get(name, [f"C{k + 1}" for k in range(arr.shape[1])])
        out.append(f"  100CL  101{step_time:12.5E}{len(node_ids):12d}{'':20}0    1           1")
        out.append(f" -4  {name:<8}{arr.shape[1]:5d}    1")
        for k, c in enumerate(names):
            out.append(f" -5  {c:<8}    1    2{k + 1:5d}    0")
        out += [_rec(int(n), v) for n, v in zip(node_ids, arr)]
        out.append(" -3")
    out.append("9999")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def read_dat_displacements(path: str) -> Dict[int, np.ndarray]:
    """Nodal displacements from a ``.dat`` file (``*NODE PRINT`` output)."""
    out, on = {}, False
    for ln in open(path, errors="replace"):
        if re.match(r"\s*displacements", ln, re.I):
            on = True
            continue
        if on:
            parts = ln.split()
            if len(parts) == 4:
                try:
                    out[int(parts[0])] = np.array([float(v) for v in parts[1:]])
                    continue
                except ValueError:
                    pass
            if out and not parts:
                on = False
    return out
