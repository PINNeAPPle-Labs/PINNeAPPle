"""Neutral physical dataset: any supported result (OpenFOAM, CalculiX, VTK/VTU, Gmsh, point tables) normalised into
one representation (geometry, mesh, coordinates, fields with quantity and unit, metadata) and written to VTK (.vtu),
HDF5, Parquet, CSV, NPZ, JSON or a PINNeAPPle dataset (UPD Zarr store, loadable with
pinneapple_data.serialization.load_zarr).

    from pinneapple_data.cae.dataset import build_dataset, export
    ds = build_dataset(mesh, unit_system="N-mm-t-s", to_si=True)
    blob = export(ds, "vtu")
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import io
import json
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import ELEMENTS, Mesh

SCHEMA = "pinneapple.physical_dataset/1"
VTK_ID = {"vertex": 1, "line": 3, "triangle": 5, "polygon": 7, "quad": 9, "tetra": 10, "hexahedron": 12, "wedge": 13,
          "pyramid": 14, "line3": 21, "triangle6": 22, "quad8": 23, "tetra10": 24, "hexahedron20": 25, "wedge15": 26,
          "pyramid13": 27, "quad9": 28, "hexahedron27": 29, "polyhedron": 42}
QUANTITIES = [  # (name pattern, quantity, SI unit)
    (r"^(u|velocity|vel|urel|u_mean)$", "velocity", "m/s"), (r"^(p|pressure|p_rgh)$", "pressure", "Pa"),
    (r"^(t|temperature|ndtemp|nt)$", "temperature", "K"), (r"^(k|tke)$", "turbulent kinetic energy", "m2/s2"),
    (r"^(epsilon)$", "turbulent dissipation rate", "m2/s3"), (r"^(omega)$", "specific dissipation rate", "1/s"),
    (r"^(nut|nutilda|alphat)$", "turbulent viscosity", "m2/s"), (r"^(disp|displacement|u_fea)$", "displacement", "m"),
    (r"^(stress|s|sigma)$", "stress", "Pa"), (r"^(tostrain|strain|pe)$", "strain", "1"),
    (r"^(flux|hfl)$", "heat flux", "W/m2"), (r"^(forc|force|rf)$", "force", "N"), (r"^(rho|density)$", "density", "kg/m3"),
]
# CalculiX/Abaqus result units per consistent unit system
UNIT_SYSTEMS = {
    "SI": {"length": "m", "DISP": "m", "STRESS": "Pa", "FORC": "N", "NDTEMP": "K", "FLUX": "W/m2", "TOSTRAIN": "1",
           "PE": "1", "to_si": {"length": 1.0, "DISP": 1.0, "STRESS": 1.0, "FORC": 1.0, "FLUX": 1.0}},
    "N-mm-t-s": {"length": "mm", "DISP": "mm", "STRESS": "MPa", "FORC": "N", "NDTEMP": "K", "FLUX": "mW/mm2",
                 "TOSTRAIN": "1", "PE": "1", "to_si": {"length": 1e-3, "DISP": 1e-3, "STRESS": 1e6, "FORC": 1.0, "FLUX": 1e3}},
}


def quantity_of(name: str) -> Tuple[Optional[str], Optional[str]]:
    n = name.strip().lower()
    for pat, q, u in QUANTITIES:
        if re.match(pat, n):
            return q, u
    return None, None


# ------------------------------------------------------------------------------- polyhedral cells -> VTK cells
def poly_cells(m: Mesh) -> Tuple[List[Tuple[str, np.ndarray]], List[List[List[int]]], np.ndarray]:
    """OpenFOAM cells as VTK cells: hexahedra, wedges, tetrahedra and pyramids are rebuilt from their faces; any other
    cell becomes a VTK polyhedron (list of faces). Returns (blocks [(type, conn)], polyhedra faces, cell order)."""
    p = m.poly
    nc, ni = p.n_cells, p.n_internal
    offs, flat = p.face_offsets, p.face_nodes
    faces = [flat[offs[i]:offs[i + 1]] for i in range(p.n_faces)]
    cf: List[List[np.ndarray]] = [[] for _ in range(nc)]
    for f, c in enumerate(p.owner):
        cf[c].append(faces[f])                                     # outward for the owner
    for f, c in enumerate(p.neighbour):
        cf[c].append(faces[f][::-1])                               # reversed: outward for the neighbour
    out: Dict[str, List[Tuple[int, List[int]]]] = {}
    polys: List[Tuple[int, List[List[int]]]] = []
    for c in range(nc):
        fl = cf[c]
        sz = sorted(len(f) for f in fl)
        typ = None
        if sz == [3, 3, 3, 3]:
            typ = "tetra"
        elif sz == [3, 3, 3, 3, 4]:
            typ = "pyramid"
        elif sz == [3, 3, 4, 4, 4]:
            typ = "wedge"
        elif sz == [4, 4, 4, 4, 4, 4]:
            typ = "hexahedron"
        conn = _shape(typ, fl) if typ else None
        if conn is None:
            polys.append((c, [list(map(int, f)) for f in fl]))
        else:
            out.setdefault(typ, []).append((c, conn))
    blocks, order = [], []
    for t in ("hexahedron", "wedge", "tetra", "pyramid"):
        if t in out:
            blocks.append((t, np.array([x[1] for x in out[t]], np.int64)))
            order += [x[0] for x in out[t]]
    order += [x[0] for x in polys]
    return blocks, [x[1] for x in polys], np.array(order, np.int64)


def _shape(typ: str, fl: List[np.ndarray]) -> Optional[List[int]]:
    nbr: Dict[int, set] = {}
    for f in fl:
        for a, b in zip(f, np.roll(f, -1)):
            nbr.setdefault(int(a), set()).add(int(b))
            nbr.setdefault(int(b), set()).add(int(a))
    if typ == "tetra":
        base = [int(x) for x in fl[0][::-1]]                       # VTK: (0,1,2) normal towards 3
        apex = [n for n in nbr if n not in base]
        return base + apex if len(apex) == 1 else None
    if typ == "pyramid":
        q = next(f for f in fl if len(f) == 4)
        base = [int(x) for x in q[::-1]]                           # normal towards the apex
        apex = [n for n in nbr if n not in base]
        return base + apex if len(apex) == 1 else None
    if typ == "wedge":
        t = next(f for f in fl if len(f) == 3)
        base = [int(x) for x in t]                                 # VTK wedge: (0,1,2) normal away from (3,4,5)
    else:
        q = fl[0]
        base = [int(x) for x in q[::-1]]                           # VTK hexahedron: (0,1,2,3) normal towards (4..7)
    top = []
    for n in base:
        other = [k for k in nbr[n] if k not in base]
        if len(other) != 1:
            return None
        top.append(other[0])
    return base + top


# ------------------------------------------------------------------------------- the dataset
def build_dataset(m: Mesh, *, unit_system: str = "auto", to_si: bool = False, rho: Optional[float] = None,
                  location: str = "auto", files: Optional[Dict[str, bytes]] = None) -> Dict[str, Any]:
    """Normalise a Mesh into the neutral dataset (a dict holding numpy arrays and metadata)."""
    src = m.source
    solver = _solver(m)
    fmt = src.get("format", "")
    fe = "CalculiX" in fmt or "Abaqus" in fmt
    if unit_system == "auto":
        unit_system = src.get("unit_system") or ("SI" if not fe else "model units")
    notes: List[str] = []
    length_unit = "m" if unit_system == "SI" or not fe else UNIT_SYSTEMS.get(unit_system, {}).get("length", "model units")
    scale_len = UNIT_SYSTEMS[unit_system]["to_si"]["length"] if to_si and unit_system in UNIT_SYSTEMS else 1.0
    if to_si and fe and unit_system not in UNIT_SYSTEMS:
        notes.append("Conversion to SI skipped: choose the deck's unit system (SI or N-mm-t-s) first.")
    pts = m.points * scale_len
    fields: Dict[str, Dict[str, Any]] = {}
    for where, store in (("point", m.point_data), ("cell", m.cell_data)):
        for name, arr in store.items():
            a = np.asarray(arr, float)
            q, si_unit = quantity_of(name)
            unit = m.units.get(name)
            comps = m.source.get("components", {}).get(name)
            if fe:
                us = UNIT_SYSTEMS.get(unit_system)
                unit = us.get(name, unit) if us else (unit or "model units")
                if to_si and us and name in us["to_si"]:
                    a = a * us["to_si"][name]
                    unit = UNIT_SYSTEMS["SI"].get(name, unit)
            if name in ("p", "p_rgh") and unit == "m2/s2":
                q = "kinematic pressure (p/ρ)"
                if rho:
                    a = a * rho
                    unit = "Pa"
                    q = "pressure"
                    notes.append(f"{name}: OpenFOAM's kinematic pressure multiplied by ρ = {rho:g} kg/m³ to give Pa.")
                else:
                    notes.append(f"{name} is OpenFOAM's kinematic pressure (m²/s², p/ρ); give ρ to convert it to Pa.")
            fields[name] = {"location": where, "components": 1 if a.ndim == 1 else int(a.shape[1]),
                            "component_names": comps or (["x", "y", "z"][: a.shape[1]] if a.ndim == 2 and a.shape[1] <= 3 else None),
                            "unit": unit, "quantity": q, "values": a,
                            "min": float(np.nanmin(a)) if a.size else None, "max": float(np.nanmax(a)) if a.size else None,
                            "mean": float(np.nanmean(a)) if a.size else None}
    cells_info = {"kind": m.kind, "cells": int(m.n_cells), "points": int(m.n_points), "element_types": m.element_counts()}
    centres = None
    if m.n_cells and (m.poly is not None or m.blocks):
        from .compare import cell_centres_volumes
        cc, vol = cell_centres_volumes(m)
        centres = cc * scale_len
        if vol is not None:
            cells_info["total_volume"] = float(vol.sum() * scale_len ** 3)
    if location == "auto":
        location = "cell" if any(f["location"] == "cell" for f in fields.values()) else "point"
    meta = {"solver": solver, "source_format": fmt, "case": src.get("root") or src.get("file"),
            "time": src.get("time"), "times_available": src.get("times"), "unit_system": unit_system,
            "length_unit": "m" if (to_si and scale_len != 1.0) else length_unit,
            "converted_to_si": bool(to_si and (scale_len != 1.0 or (fe and unit_system in UNIT_SYSTEMS))),
            "converted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "converter": "PINNeAPPle interop (pinneapple_data.cae.dataset)", "notes": notes}
    if files:
        meta["files"] = [{"name": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()[:16]}
                         for k, v in sorted(files.items()) if v]
    mn, mx = pts.min(0), pts.max(0)
    return {"schema": SCHEMA, "geometry": {"bounding_box": {"min": mn.tolist(), "max": mx.tolist(), "size": (mx - mn).tolist()},
                                           "dimension": int(m.dim) if m.dim else int(np.sum((mx - mn) > 1e-12 * max((mx - mn).max(), 1e-300))),
                                           "length_unit": meta["length_unit"]},
            "mesh": cells_info, "points": pts, "cell_centres": centres, "fields": fields, "metadata": meta,
            "location": location, "_mesh": m, "_scale": scale_len}


def _solver(m: Mesh) -> Optional[str]:
    s = m.source
    if s.get("kind") in ("openfoam", "openfoam_polymesh"):
        return "OpenFOAM" + (f" {s['application']}" if s.get("application") else "")
    if "frd" in s.get("format", ""):
        return "CalculiX"
    return s.get("solver")


def describe(ds: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe description (the schema the user sees): no arrays."""
    f = {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in ds["fields"].items()}
    return {"schema": ds["schema"], "geometry": ds["geometry"], "mesh": ds["mesh"],
            "coordinates": {"location": ds["location"], "count": int(len(_coords(ds))), "names": ["x", "y", "z"],
                            "unit": ds["geometry"]["length_unit"]},
            "fields": f, "metadata": ds["metadata"]}


def _coords(ds) -> np.ndarray:
    return ds["cell_centres"] if ds["location"] == "cell" and ds["cell_centres"] is not None else ds["points"]


def _table(ds) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Columns of the tabular exports at the chosen location (cell centres or points)."""
    loc = ds["location"]
    xyz = _coords(ds)
    cols = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    units = {"x": ds["geometry"]["length_unit"], "y": ds["geometry"]["length_unit"], "z": ds["geometry"]["length_unit"]}
    for name, f in ds["fields"].items():
        if f["location"] != loc and not (loc == "point" and f["location"] == "point"):
            continue
        v = f["values"]
        if v.ndim == 1:
            cols[name] = v
            units[name] = f["unit"] or ""
        else:
            names = f.get("component_names") or [str(i) for i in range(v.shape[1])]
            for j in range(v.shape[1]):
                cols[f"{name}_{names[j]}"] = v[:, j]
                units[f"{name}_{names[j]}"] = f["unit"] or ""
    return cols, units


# ------------------------------------------------------------------------------- writers
def export(ds: Dict[str, Any], fmt: str) -> Tuple[bytes, str, str]:
    """(content, media type, file extension)."""
    fmt = fmt.lower()
    if fmt == "vtu":
        return write_vtu(ds), "application/xml", "vtu"
    if fmt in ("h5", "hdf5"):
        return write_hdf5(ds), "application/x-hdf5", "h5"
    if fmt == "parquet":
        return write_parquet(ds), "application/vnd.apache.parquet", "parquet"
    if fmt == "csv":
        return write_csv(ds), "text/csv", "csv"
    if fmt == "npz":
        return write_npz(ds), "application/octet-stream", "npz"
    if fmt == "json":
        return write_json(ds), "application/json", "json"
    if fmt in ("pinneapple", "zarr"):
        return write_pinneapple(ds), "application/zip", "zarr.zip"
    raise ValueError(f"unknown format {fmt}: vtu, hdf5, parquet, csv, npz, json, pinneapple")


def _b64(a: np.ndarray) -> str:
    raw = np.ascontiguousarray(a).tobytes()
    return base64.b64encode(struct.pack("<I", len(raw)) + raw).decode()


def _da(name: str, a: np.ndarray, vtype: str, ncomp: int = 1) -> str:
    nc = f' NumberOfComponents="{ncomp}"' if ncomp > 1 or name == "Points" else ""
    return f'<DataArray type="{vtype}" Name="{name}"{nc} format="binary">{_b64(a)}</DataArray>\n'


def write_vtu(ds: Dict[str, Any]) -> bytes:
    m: Mesh = ds["_mesh"]
    pts = np.asarray(ds["points"], "<f8")
    conn, offsets, types, faces_stream, face_offsets = [], [], [], None, None
    order = None
    if m.poly is not None:
        blocks, polys, order = poly_cells(m)
        n_std = sum(len(c) for _, c in blocks)
        for t, c in blocks:
            conn.append(c.ravel())
            types.append(np.full(len(c), VTK_ID[t], np.uint8))
            offsets.append(np.full(len(c), c.shape[1], np.int64))
        if polys:
            fs, fo, pc, po = [], [], [], []
            pos = 0
            for pf in polys:
                nodes = sorted({n for f in pf for n in f})
                pc.append(np.array(nodes, np.int64))
                po.append(len(nodes))
                stream = [len(pf)] + [x for f in pf for x in [len(f)] + f]
                fs += stream
                pos += len(stream)
                fo.append(pos)
            conn.append(np.concatenate(pc))
            offsets.append(np.array(po, np.int64))
            types.append(np.full(len(polys), VTK_ID["polyhedron"], np.uint8))
            faces_stream = np.array(fs, np.int64)
            face_offsets = np.concatenate([np.full(n_std, -1, np.int64), np.array(fo, np.int64)])
    else:
        for b in m.blocks + [blk for _, blk in m.boundary_blocks if blk.dim == 0]:
            if b.type not in VTK_ID:
                continue
            conn.append(b.conn.ravel())
            types.append(np.full(len(b.conn), VTK_ID[b.type], np.uint8))
            offsets.append(np.full(len(b.conn), b.conn.shape[1], np.int64))
    if not conn:                                                      # point cloud: one vertex per point
        conn = [np.arange(len(pts), dtype=np.int64)]
        types = [np.full(len(pts), 1, np.uint8)]
        offsets = [np.ones(len(pts), np.int64)]
    conn = np.concatenate(conn).astype("<i8")
    off = np.cumsum(np.concatenate(offsets)).astype("<i8")
    typ = np.concatenate(types)
    ncell = len(typ)
    x = ['<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">\n',
         f'<UnstructuredGrid>\n<Piece NumberOfPoints="{len(pts)}" NumberOfCells="{ncell}">\n<Points>\n',
         _da("Points", pts, "Float64", 3), "</Points>\n<Cells>\n", _da("connectivity", conn, "Int64"),
         _da("offsets", off, "Int64"), _da("types", typ, "UInt8")]
    if faces_stream is not None:
        x += [_da("faces", faces_stream.astype("<i8"), "Int64"), _da("faceoffsets", face_offsets.astype("<i8"), "Int64")]
    x.append("</Cells>\n")
    pd, cd = [], []
    for name, f in ds["fields"].items():
        v = np.asarray(f["values"], "<f8")
        if f["location"] == "point" and len(v) == len(pts):
            pd.append(_da(name, v, "Float64", f["components"]))
        elif f["location"] == "cell" and len(v) == m.n_cells and m.n_cells == ncell:
            vv = v[order] if order is not None else v
            cd.append(_da(name, vv, "Float64", f["components"]))
    if pd:
        x += ["<PointData>\n"] + pd + ["</PointData>\n"]
    if cd:
        x += ["<CellData>\n"] + cd + ["</CellData>\n"]
    x.append("</Piece>\n</UnstructuredGrid>\n")
    meta = json.dumps(describe(ds), default=str)
    x.append(f"<!-- {SCHEMA} metadata: {meta.replace('--', '- -')} -->\n</VTKFile>\n")
    return "".join(x).encode()


def write_hdf5(ds: Dict[str, Any]) -> bytes:
    import h5py
    bio = io.BytesIO()
    m: Mesh = ds["_mesh"]
    with h5py.File(bio, "w") as h:
        h.attrs["schema"] = SCHEMA
        h.attrs["metadata"] = json.dumps(describe(ds), default=str)
        g = h.create_group("mesh")
        g.create_dataset("points", data=ds["points"], compression="gzip")
        g["points"].attrs["unit"] = ds["geometry"]["length_unit"]
        if ds["cell_centres"] is not None:
            g.create_dataset("cell_centres", data=ds["cell_centres"], compression="gzip")
        if m.poly is not None:
            p = m.poly
            for k, v in (("face_nodes", p.face_nodes), ("face_offsets", p.face_offsets), ("owner", p.owner),
                         ("neighbour", p.neighbour)):
                g.create_dataset(f"polyMesh/{k}", data=v, compression="gzip")
            pg = g.create_group("polyMesh/patches")
            for pt in p.patches:
                pg.attrs[pt["name"]] = json.dumps({k: pt[k] for k in ("type", "startFace", "nFaces")})
        for b in m.blocks:
            g.create_dataset(f"cells/{b.type}", data=b.conn, compression="gzip")
        fg = h.create_group("fields")
        for name, f in ds["fields"].items():
            d = fg.create_dataset(name, data=f["values"], compression="gzip")
            for k in ("location", "unit", "quantity"):
                if f.get(k):
                    d.attrs[k] = f[k]
            if f.get("component_names"):
                d.attrs["components"] = ",".join(f["component_names"])
    return bio.getvalue()


def write_parquet(ds: Dict[str, Any]) -> bytes:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols, units = _table(ds)
    t = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta[b"pinneapple"] = json.dumps({**describe(ds), "column_units": units}, default=str).encode()
    t = t.replace_schema_metadata(meta)
    bio = io.BytesIO()
    pq.write_table(t, bio, compression="zstd")
    return bio.getvalue()


def write_csv(ds: Dict[str, Any]) -> bytes:
    cols, units = _table(ds)
    names = list(cols)
    head = ",".join(f"{n} [{units[n]}]" if units.get(n) else n for n in names)
    arr = np.column_stack([cols[n] for n in names])
    bio = io.StringIO()
    bio.write(f"# {SCHEMA} · {ds['metadata'].get('solver') or ds['metadata'].get('source_format')} · values at "
              f"{'cell centres' if ds['location'] == 'cell' else 'points'}\n")
    np.savetxt(bio, arr, delimiter=",", header=head, comments="", fmt="%.8g")
    return bio.getvalue().encode()


def write_npz(ds: Dict[str, Any]) -> bytes:
    arrs = {"points": ds["points"], "metadata": np.array(json.dumps(describe(ds), default=str))}
    if ds["cell_centres"] is not None:
        arrs["cell_centres"] = ds["cell_centres"]
    for name, f in ds["fields"].items():
        arrs[f"{f['location']}/{name}"] = f["values"]
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrs)
    return bio.getvalue()


def write_json(ds: Dict[str, Any], max_values: int = 2_000_000) -> bytes:
    d = describe(ds)
    total = sum(f["values"].size for f in ds["fields"].values()) + _coords(ds).size
    if total <= max_values:
        d["coordinates"]["values"] = np.round(_coords(ds), 9).tolist()
        for name, f in ds["fields"].items():
            d["fields"][name]["values"] = np.round(f["values"], 9).tolist() if f["location"] == ds["location"] else None
    else:
        d["note"] = f"{total:,} values: data arrays left out of the JSON; use HDF5, Parquet or NPZ."
    return json.dumps(d, default=str).encode()


def write_pinneapple(ds: Dict[str, Any]) -> bytes:
    """A PINNeAPPle dataset: one PhysicalSample (state = coordinates and fields at the chosen location) written with
    the library's UPD Zarr store (pinneapple_data.serialization.save_zarr), zipped. Load it with
    ``load_zarr`` after unzipping; units, quantities and the source are in the sample's provenance."""
    import os
    import shutil
    import tempfile
    import zipfile
    from pinneapple_data.physical_sample import PhysicalSample
    from pinneapple_data.serialization import save_zarr
    loc = ds["location"]
    xyz = _coords(ds)
    state = {"x": xyz[:, 0].copy(), "y": xyz[:, 1].copy(), "z": xyz[:, 2].copy()}
    fields = {}
    for name, f in ds["fields"].items():
        if f["location"] == loc:
            state[name] = np.asarray(f["values"], float)
            fields[name] = {"unit": f["unit"], "quantity": f["quantity"], "components": f.get("component_names")}
    d = describe(ds)
    prov = {k: v for k, v in d["metadata"].items()}
    prov.update(schema=SCHEMA, location=loc, length_unit=ds["geometry"]["length_unit"], fields=fields,
                mesh=ds["mesh"], bounding_box=ds["geometry"]["bounding_box"])
    sample = PhysicalSample(state=state, domain={"type": "mesh" if ds["mesh"]["cells"] else "points", "location": loc},
                            schema={"units": {k: v["unit"] for k, v in fields.items()}}, provenance=prov)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "dataset.zarr")
        save_zarr([sample], root)
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    full = os.path.join(dp, fn)
                    z.write(full, os.path.relpath(full, tmp))
        shutil.rmtree(root, ignore_errors=True)
    return bio.getvalue()
