"""Reading the result back: patch averages, mass flow, CSV tables and the indicators the checks use.

The numbers a compressible passage result is judged by are few, and each has a definition that
has to be stated:

``mdot``
    mass flow through the inlet patch, ``sum(phi)`` over its faces, positive into the domain and
    scaled from the sector to the whole valve (x4 for the 90 deg sector). ``phi`` is already
    ``rho U . Sf``: this is the flow the solver solved, not a reconstruction from a mean velocity.
``p_in`` / ``p_out``
    area-weighted average of the static pressure on the inlet and outlet patches. Area weighted, so
    that the fine mesh near the axis core does not outvote the rest of the face.
``T_out``
    mass-flux-weighted average of the static temperature on the outlet patch. Mass weighted, not
    area weighted: temperature is carried by the mass, and a stagnant corner must not vote.
``Ma_max``
    the largest Mach number in the domain, taken from OpenFOAM's own ``MachNo`` function object, so
    it uses the real-gas sound speed rather than ``sqrt(gamma R T)`` with an ideal-gas gamma.
``Kv``
    the flow coefficient the run implies, ``Kv = Q [m3/h] sqrt((rho/rho_water)/dP [bar])`` with Q the
    volume flow at the inlet state. :mod:`.checks` computes the IEC 60534-2-1 equivalent
    independently and compares the two: different routes to the same number, which is what makes the
    comparison worth doing. A passage that is not a valve still has a Kv, and it is the number a
    piping engineer will ask for.

Everything is written as CSV as well as returned, because the customer's next step is a spreadsheet.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .physics import peng_robinson

__all__ = ["FlowResult", "read_case", "write_tables", "axial_profile", "read_patch_geometry",
           "patch_triangles", "RHO_WATER"]

RHO_WATER = 998.2          # kg/m3, 20 C -- the reference density of the flow coefficient


# ------------------------------------------------------------------------------- OpenFOAM text
def _strip(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r"FoamFile\s*\{.*?\}", " ", text, count=1, flags=re.S)


def _numbers(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def _internal(text: str) -> str:
    """The text after ``internalField``, with OpenFOAM's ``List<scalar>/List<vector>`` removed."""
    m = re.search(r"internalField\s+(uniform|nonuniform)\s+", text)
    if m is None:
        raise ValueError("no internalField")
    return re.sub(r"^\s*List<[^>]*>", " ", text[m.end():], count=1)


def _scalar_field(text: str) -> np.ndarray:
    rest = _internal(text)
    if rest.lstrip().startswith("uniform"):
        rest = rest.lstrip()[len("uniform"):]
        if "(" in rest and rest.index("(") < 40:          # a list follows (boundary values)
            n = int(re.match(r"\s*(\d+)\s*\(", rest).group(1))
            return np.full(n, _numbers(rest)[0])
        return np.full(1, float(_numbers(rest)[0]))
    n = int(re.match(r"\s*(\d+)", rest).group(1))
    return np.array(_numbers(rest[rest.index("(") + 1:])[:n], dtype=float)


def _vector_field(text: str) -> np.ndarray:
    rest = _internal(text)
    if rest.lstrip().startswith("uniform"):
        vals = _numbers(rest)
        v = np.array(vals[:3])
        m = re.search(r"(\d+)\s*\(", rest)
        n = int(m.group(1)) if m else 1
        return np.tile(v, (n, 1))
    n = int(re.match(r"\s*(\d+)", rest).group(1))
    body = rest[rest.index("(") + 1:]
    pts = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", body)[:n]
    return np.array(pts, dtype=float)


def _section(text: str, key: str) -> str:
    m = re.search(rf"(?<![\w]){re.escape(key)}\s*\{{", text)
    if m is None:
        return ""
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1]


def _patch_values(text: str, patch: str) -> np.ndarray:
    """Values on one patch: uniform -> one value, nonuniform -> one per face."""
    sec = _section(_section(text, "boundaryField"), patch)
    if not sec:
        return np.zeros(0)
    m = re.search(r"value\s+(uniform|nonuniform)\s+", sec)
    if m is None:
        return np.zeros(0)
    rest = re.sub(r"^\s*List<[^>]*>", " ", sec[m.end():], count=1)
    if m.group(1) == "uniform":
        return np.array([float(_numbers(rest)[0])])
    n = int(re.match(r"\s*(\d+)", rest).group(1))
    return np.array(_numbers(rest[rest.index("(") + 1:])[:n], dtype=float)


# ------------------------------------------------------------------------------- mesh geometry
def _faces(path: str) -> list[list[int]]:
    text = _strip(path)
    n = int(re.match(r"\s*(\d+)", text).group(1))
    body = text[text.index("(") + 1:]
    out = []
    for line in re.findall(r"(\d+)\(([^)]*)\)", body)[:n]:
        out.append([int(v) for v in line[1].split()])
    return out


def _points(path: str) -> np.ndarray:
    text = _strip(path)
    body = text[text.index("(") + 1:]
    return np.array(re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", body),
                    dtype=float)


def read_patch_geometry(case: str) -> dict[str, dict[str, Any]]:
    """Faces, areas and centres of every boundary patch, from ``constant/polyMesh``."""
    mesh = os.path.join(case, "constant", "polyMesh")
    pts = _points(os.path.join(mesh, "points"))
    faces = _faces(os.path.join(mesh, "faces"))
    bnd = _strip(os.path.join(mesh, "boundary"))
    out: dict[str, dict[str, Any]] = {}
    for name, _kind, n, start in re.findall(
            r"(\w+)\s*\{\s*type\s+(\w+);\s*nFaces\s+(\d+);\s*startFace\s+(\d+);", bnd):
        ids = list(range(int(start), int(start) + int(n)))
        area = np.zeros(len(ids))
        centre = np.zeros((len(ids), 3))
        for k, fid in enumerate(ids):
            q = pts[faces[fid]]
            nxt = np.roll(q, -1, axis=0)
            nrm = np.array([(q[:, 1] * nxt[:, 2] - q[:, 2] * nxt[:, 1]).sum(),
                            (q[:, 2] * nxt[:, 0] - q[:, 0] * nxt[:, 2]).sum(),
                            (q[:, 0] * nxt[:, 1] - q[:, 1] * nxt[:, 0]).sum()]) / 2.0
            area[k] = np.linalg.norm(nrm)
            centre[k] = q.mean(axis=0)
        out[name] = {"type": _kind, "ids": ids, "area_m2": area, "centre_m": centre}
    return out


def patch_triangles(case: str, patch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A boundary patch as triangles, with the cell each face bounds.

    Fan-triangulated per face so a non-planar quad is honoured; the cell index lets the caller paint
    the surface with the cell's field. Used by the 3D scene, where the walls and the cut plane are
    the two surfaces worth drawing.
    """
    mesh = os.path.join(case, "constant", "polyMesh")
    pts = _points(os.path.join(mesh, "points"))
    faces = _faces(os.path.join(mesh, "faces"))
    owner_text = _strip(os.path.join(mesh, "owner"))
    owner = np.array([int(v) for v in re.findall(r"^\s*(\d+)\s*$", owner_text, flags=re.M)])
    ids = read_patch_geometry(case)[patch]["ids"]
    verts: list[np.ndarray] = []
    tris: list[list[int]] = []
    per_face: list[int] = []
    for fid in ids:
        f = faces[fid]
        base = len(verts)
        verts.extend(pts[v] for v in f)
        for j in range(1, len(f) - 1):
            tris.append([base, base + j, base + j + 1])
            per_face.append(int(owner[fid]))
    return np.array(verts), np.array(tris, dtype=np.int64), np.array(per_face, dtype=np.int64)


def cell_centres(case: str) -> np.ndarray:
    """Cell centres from the mesh itself (mean of the cell's face centres).

    Written here rather than asked from OpenFOAM so the tables and the 3D scene can be built from a
    case directory alone, with no post-processing run in between.
    """
    mesh = os.path.join(case, "constant", "polyMesh")
    pts = _points(os.path.join(mesh, "points"))
    faces = _faces(os.path.join(mesh, "faces"))
    owner_text = _strip(os.path.join(mesh, "owner"))
    n = int(re.match(r"\s*(\d+)", owner_text).group(1))
    owner = np.array([int(v) for v in re.findall(r"^\s*(\d+)\s*$", owner_text, flags=re.M)][:n])
    centre = np.array([pts[f].mean(axis=0) for f in faces])
    n_cells = int(owner.max()) + 1
    acc = np.zeros((n_cells, 3))
    cnt = np.zeros(n_cells)
    np.add.at(acc, owner, centre)
    np.add.at(cnt, owner, 1.0)
    return acc / np.maximum(cnt, 1)[:, None]


def _n_internal(case: str) -> int:
    text = _strip(os.path.join(case, "constant", "polyMesh", "neighbour"))
    return int(re.match(r"\s*(\d+)", text).group(1))


def _first_time(case: str) -> str:
    times = sorted((d for d in os.listdir(case)
                    if re.fullmatch(r"[0-9.eE+-]+", d) and os.path.isdir(os.path.join(case, d))
                    and os.path.exists(os.path.join(case, d, "p"))), key=float)
    if not times:
        raise FileNotFoundError(f"nenhum diretório de tempo com o campo p em {case}")
    return times[-1]


@dataclass
class FlowResult:
    """What one converged run says about the valve."""

    lift_mm: float
    open_fraction: float
    mdot_kg_s: float
    p_in_pa: float
    p_out_pa: float
    T_in_K: float
    T_out_K: float
    T_min_K: float
    T_max_K: float
    ma_max: float
    ma_mean: float
    u_max: float
    rho_in: float
    kv: float
    time_dir: str
    mdot_out_kg_s: float = 0.0

    @property
    def dp_pa(self) -> float:
        return self.p_in_pa - self.p_out_pa

    @property
    def dp_bar(self) -> float:
        return self.dp_pa / 1e5

    @property
    def dT_K(self) -> float:
        return self.T_out_K - self.T_in_K

    @property
    def mass_imbalance(self) -> float:
        if not self.mdot_kg_s:
            return 0.0
        return abs(self.mdot_out_kg_s - self.mdot_kg_s) / abs(self.mdot_kg_s)

    @property
    def volume_flow_m3_h(self) -> float:
        return self.mdot_kg_s / self.rho_in * 3600.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lift_mm": round(self.lift_mm, 3),
            "open_fraction": round(self.open_fraction, 4),
            "mdot_kg_s": round(self.mdot_kg_s, 6),
            "mdot_kg_h": round(self.mdot_kg_s * 3600.0, 3),
            "p_in_bar": round(self.p_in_pa / 1e5, 4),
            "p_out_bar": round(self.p_out_pa / 1e5, 4),
            "dp_bar": round(self.dp_bar, 4),
            "T_in_K": round(self.T_in_K, 4),
            "T_out_K": round(self.T_out_K, 4),
            "dT_K": round(self.dT_K, 4),
            "T_min_K": round(self.T_min_K, 3),
            "T_max_K": round(self.T_max_K, 3),
            "Ma_max": round(self.ma_max, 4),
            "u_max_m_s": round(self.u_max, 3),
            "rho_in_kg_m3": round(self.rho_in, 3),
            "Q_m3_h": round(self.volume_flow_m3_h, 4),
            "Kv_m3_h": round(self.kv, 5),
            "desbalanco_massico": round(self.mass_imbalance, 8),
        }


def read_case(case: str, passage, conditions, gas, *, time_dir: str | None = None) -> tuple[FlowResult, dict[str, np.ndarray]]:
    """Read ``case`` into a :class:`FlowResult` plus the raw fields, for the tables and the 3D scene."""
    t = time_dir or _first_time(case)
    sector_deg = float(passage.sector_deg)
    geom_meta = {"lift_mm": float(passage.meta.get("parameter_value", 0.0)),
                 "open_fraction": float(passage.meta.get("open_fraction", 1.0)),
                 "T_in_K": float(conditions.T_in_K)}
    base = os.path.join(case, t)
    p_text, T_text = _strip(os.path.join(base, "p")), _strip(os.path.join(base, "T"))
    p, T = _scalar_field(p_text), _scalar_field(T_text)
    U = _vector_field(_strip(os.path.join(base, "U")))
    ma = _scalar_field(_strip(os.path.join(base, "Ma"))) if os.path.exists(os.path.join(base, "Ma")) \
        else np.zeros_like(p)

    patches = read_patch_geometry(case)
    scale = 360.0 / sector_deg

    def weighted(patch: str, field_text: str, weights: np.ndarray) -> tuple[float, float]:
        vals = _patch_values(field_text, patch)
        a = patches[patch]["area_m2"]
        if vals.size == 1:
            vals = np.full(a.size, vals[0])
        if vals.size != a.size:
            return float(np.mean(vals)) if vals.size else float("nan"), float(a.sum())
        w = weights if weights.size == a.size else a
        return float((vals * w).sum() / max(w.sum(), 1e-30)), float(a.sum())

    p_in, _ = weighted("inlet", p_text, patches["inlet"]["area_m2"])
    p_out, _ = weighted("outlet", p_text, patches["outlet"]["area_m2"])

    phi_text = _strip(os.path.join(base, "phi")) if os.path.exists(os.path.join(base, "phi")) else ""
    mdot = mdot_out = 0.0
    if phi_text:
        mdot = -float(np.sum(_patch_values(phi_text, "inlet"))) * scale
        mdot_out = float(np.sum(_patch_values(phi_text, "outlet"))) * scale
    T_out, _ = weighted("outlet", T_text, np.abs(_patch_values(phi_text, "outlet"))
                        if phi_text else patches["outlet"]["area_m2"])

    rho_path = os.path.join(base, "rho")
    if os.path.exists(rho_path):
        rho_patch = _patch_values(_strip(rho_path), "inlet")
        rho_in = float(np.mean(rho_patch)) if rho_patch.size else float(np.mean(_scalar_field(_strip(rho_path))))
    else:
        rho_in = peng_robinson(gas, p_in, geom_meta.get("T_in_K", 300.0))["rho"]

    dp_bar = (p_in - p_out) / 1e5
    q_m3h = mdot / rho_in * 3600.0
    kv = q_m3h * math.sqrt((rho_in / RHO_WATER) / dp_bar) if dp_bar > 0 else float("nan")

    umag = np.linalg.norm(U, axis=1)
    res = FlowResult(lift_mm=float(geom_meta["lift_mm"]), open_fraction=float(geom_meta["open_fraction"]),
                     mdot_kg_s=mdot, p_in_pa=p_in, p_out_pa=p_out,
                     T_in_K=float(geom_meta.get("T_in_K", 300.0)), T_out_K=T_out,
                     T_min_K=float(T.min()), T_max_K=float(T.max()),
                     ma_max=float(ma.max()) if ma.size == p.size else float("nan"),
                     ma_mean=float(ma.mean()) if ma.size == p.size else float("nan"),
                     u_max=float(umag.max()), rho_in=rho_in, kv=kv, time_dir=t, mdot_out_kg_s=mdot_out)

    if os.path.exists(os.path.join(base, "C")):
        centres = _vector_field(_strip(os.path.join(base, "C")))
    else:
        centres = cell_centres(case)
        if len(centres) != len(p):
            centres = np.zeros((0, 3))
    fields = {"p": p, "T": T, "U": U, "Ma": ma, "centres": centres, "patches": patches,
              "Um": umag}
    return res, fields


def write_tables(out_dir: str, result: FlowResult, fields: dict[str, np.ndarray]) -> list[str]:
    """Write the CSV tables of one run and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    units = {"mdot_kg_s": "kg/s", "mdot_kg_h": "kg/h", "p_in_bar": "bar", "p_out_bar": "bar",
             "dp_bar": "bar", "T_in_K": "K", "T_out_K": "K", "dT_K": "K", "T_min_K": "K",
             "T_max_K": "K", "Ma_max": "-", "u_max_m_s": "m/s", "rho_in_kg_m3": "kg/m3",
             "Q_m3_h": "m3/h", "Kv_m3_h": "m3/h", "lift_mm": "mm", "open_fraction": "-",
             "desbalanco_massico": "-"}
    written = []
    path = os.path.join(out_dir, "indicadores.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("grandeza,valor,unidade\n")
        for k, v in result.as_dict().items():
            f.write(f"{k},{v},{units.get(k, '-')}\n")
    written.append(path)

    profile = axial_profile(fields)
    if len(profile):
        path = os.path.join(out_dir, "perfil_axial.csv")
        np.savetxt(path, profile, delimiter=",", header="x_mm,p_bar,T_K,U_m_s,Ma",
                   comments="", fmt="%.6g")
        written.append(path)

    centres = fields.get("centres")
    if centres is not None and len(centres) == len(fields["p"]):
        path = os.path.join(out_dir, "celulas.csv")
        U = fields["U"]
        arr = np.column_stack([centres, fields["p"] / 1e5, fields["T"], U, fields["Um"], fields["Ma"]])
        np.savetxt(path, arr, delimiter=",",
                   header="x_m,y_m,z_m,p_bar,T_K,Ux,Uy,Uz,U_m_s,Ma", comments="", fmt="%.6g")
        written.append(path)

    wall = fields.get("patches", {}).get("wall")
    if wall is not None:
        path = os.path.join(out_dir, "parede.csv")
        np.savetxt(path, np.column_stack([wall["centre_m"], wall["area_m2"]]), delimiter=",",
                   header="x_m,y_m,z_m,area_m2", comments="", fmt="%.6g")
        written.append(path)
    return written


def axial_profile(fields: dict[str, np.ndarray], n_bins: int = 80) -> np.ndarray:
    """Profile along x: mean p, T, |U| and Ma per x-bin — the one line of the result worth a table."""
    c = fields.get("centres")
    if c is None or len(c) == 0:
        return np.zeros((0, 5))
    x = c[:, 0] * 1000.0
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    ma = fields.get("Ma")
    rows = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 4:
            continue
        rows.append([0.5 * (edges[b] + edges[b + 1]),
                     float(np.mean(fields["p"][sel]) / 1e5),
                     float(np.mean(fields["T"][sel])),
                     float(np.mean(fields["Um"][sel])),
                     float(np.mean(ma[sel])) if ma is not None and len(ma) else 0.0])
    return np.array(rows)
