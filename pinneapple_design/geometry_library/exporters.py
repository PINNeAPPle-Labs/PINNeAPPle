"""File writers: binary STL, OBJ (+ MTL), glTF binary, OpenUSD (ASCII) and STEP (faceted B-rep).

Every writer is deterministic (same geometry -> same bytes), which lets the manifest pin each file with SHA-256.
STEP is written as an AP214 ``FACETED_BREP`` per closed body (planar triangular faces bounded by poly-loops), which
OpenCASCADE / FreeCAD / CATIA / NX import as solids; :func:`step_check` reads it back through OpenCASCADE when
``cadquery`` is installed.
"""
from __future__ import annotations

import io
import os
import struct

import numpy as np

FORMATS = ("stl", "obj", "glb", "usda", "step")
_ALIASES = {"usd": "usda", "stp": "step", "gltf": "glb"}


def normalise_format(fmt: str) -> str:
    f = fmt.lower().lstrip(".")
    f = _ALIASES.get(f, f)
    if f not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS} (or usd / stp)")
    return f


def stl_bytes(V: np.ndarray, F: np.ndarray, header: str) -> bytes:
    """Binary STL of the triangle mesh ``(V, F)``; ``header`` fills the 80-byte header."""
    tris = np.asarray(V)[np.asarray(F)].astype("<f4")
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + np.float32(1e-30)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tris
    head = header.encode()[:80].ljust(80, b" ")
    return head + struct.pack("<I", len(tris)) + rec.tobytes()


def to_stl(geom, aero_only: bool = False) -> bytes:
    V, F = geom.mesh(aero_only=aero_only)
    return stl_bytes(V, F, f"PINNeAPPle geometry library: {geom.name} ({geom.lod}), metres")


def to_obj(geom) -> bytes:
    mats = geom.scene().surfaces
    out = io.StringIO()
    out.write(f"# PINNeAPPle geometry library: {geom.name} ({geom.lod}), metres, z up\n")
    out.write(f"mtllib {geom.name}.mtl\n")
    off = 1
    for s in mats:
        out.write(f"o {s.name}\nusemtl {s.material}\n")
        np.savetxt(out, s.vertices, fmt="v %.7g %.7g %.7g")
        np.savetxt(out, s.faces + off, fmt="f %d %d %d")
        off += len(s.vertices)
    return out.getvalue().encode()


def to_mtl(geom) -> bytes:
    from pinneapple_tools.visualization.studio.scene import MATERIALS
    out = []
    for m in dict.fromkeys(s.material for s in geom.scene().surfaces):
        c = MATERIALS.get(m, {"color": [0.6, 0.6, 0.6]})["color"]
        out.append(f"newmtl {m}\nKd {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
    return "".join(out).encode()


def to_glb(geom) -> bytes:
    sc = geom.scene()
    return sc.to_glb(geom.spec.material_table())


def to_usda(geom) -> bytes:
    sc = geom.scene()
    return sc.to_usda(geom.spec.material_table()).encode()


def to_step(geom, aero_only: bool = False) -> bytes:
    """ISO 10303-21 file, AP214, one ``FACETED_BREP`` per closed body (open sheets are skipped)."""
    from . import meshops
    lines = []
    nid = [0]

    def ent(s: str) -> int:
        nid[0] += 1
        lines.append(f"#{nid[0]}={s};")
        return nid[0]

    app = ent("APPLICATION_CONTEXT('automotive design')")
    ent(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    mctx = ent(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    prod = ent(f"PRODUCT('{geom.name}','{geom.title}','',(#{mctx}))")
    form = ent(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    dctx = ent(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    pdef = ent(f"PRODUCT_DEFINITION('design','',#{form},#{dctx})")
    pds = ent(f"PRODUCT_DEFINITION_SHAPE('','',#{pdef})")
    ent(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod}))")
    lu = ent("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))")
    au = ent("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = ent("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = ent(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{lu},'distance_accuracy_value','')")
    gctx = ent(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
               f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))REPRESENTATION_CONTEXT('',''))")
    o = ent("CARTESIAN_POINT('',(0.,0.,0.))")
    dz = ent("DIRECTION('',(0.,0.,1.))")
    dx = ent("DIRECTION('',(1.,0.,0.))")
    ax = ent(f"AXIS2_PLACEMENT_3D('',#{o},#{dz},#{dx})")
    breps = []
    for b in geom.bodies:
        if not b.closed or (aero_only and not b.aero):
            continue
        V, F = meshops.weld(b.vertices, b.faces)
        pid0 = nid[0] + 1
        for x, y, z in V:
            ent(f"CARTESIAN_POINT('',({_real(x)},{_real(y)},{_real(z)}))")
        faces = []
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-300
        for (a, bb, c), nn in zip(F, fn, strict=False):
            loop = ent(f"POLY_LOOP('',(#{pid0 + a},#{pid0 + bb},#{pid0 + c}))")
            bound = ent(f"FACE_OUTER_BOUND('',#{loop},.T.)")
            e1 = V[bb] - V[a]
            e1 = e1 / (np.linalg.norm(e1) + 1e-300)
            dn = ent(f"DIRECTION('',({_real(nn[0])},{_real(nn[1])},{_real(nn[2])}))")
            de = ent(f"DIRECTION('',({_real(e1[0])},{_real(e1[1])},{_real(e1[2])}))")
            pl = ent(f"PLANE('',#{ent(f'AXIS2_PLACEMENT_3D({chr(39)}{chr(39)},#{pid0 + a},#{dn},#{de})')})")
            faces.append(ent(f"FACE_SURFACE('',(#{bound}),#{pl},.T.)"))
        shell = ent("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
        breps.append(ent(f"FACETED_BREP('{b.name}',#{shell})"))
    rep = ent("FACETED_BREP_SHAPE_REPRESENTATION('',(" + ",".join(f"#{x}" for x in [ax] + breps) + f"),#{gctx})")
    ent(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    head = ("ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('PINNeAPPle geometry library: " + geom.name +
            "'),'2;1');\nFILE_NAME('" + geom.name + ".step','2026-01-01T00:00:00',(''),(''),"
            "'PINNeAPPle','PINNeAPPle geometry library','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n")
    return (head + "\n".join(lines) + "\nENDSEC;\nEND-ISO-10303-21;\n").encode()


def _real(x: float) -> str:
    """A STEP REAL: always with a decimal point (``1.``, ``-2.5E-05``)."""
    t = f"{x:.9G}"
    if t in ("-0",):
        t = "0"
    if "E" in t:
        mant, ex = t.split("E")
        return (mant if "." in mant else mant + ".") + "E" + str(int(ex))
    return t if "." in t else t + "."


WRITERS = {"stl": to_stl, "obj": to_obj, "glb": to_glb, "usda": to_usda, "step": to_step}


def export(geom, path: str, fmt: str | None = None) -> str:
    fmt = normalise_format(fmt or os.path.splitext(path)[1])
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(WRITERS[fmt](geom))
    if fmt == "obj":
        with open(os.path.join(d, f"{geom.name}.mtl"), "wb") as f:
            f.write(to_mtl(geom))
    return path


def read_stl(path: str):
    """Binary STL -> (triangles (m, 3, 3) float32)."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n, offset=84)
    return rec["v"].copy()


def step_check(path: str) -> dict:
    """Read a STEP file back with OpenCASCADE (``cadquery``): number of solids and total volume [m^3] (OpenCASCADE
    works in millimetres)."""
    import cadquery as cq
    shape = cq.importers.importStep(path)
    solids = shape.solids().vals()
    return {"solids": len(solids), "volume": float(sum(abs(s.Volume()) for s in solids)) * 1e-9}
