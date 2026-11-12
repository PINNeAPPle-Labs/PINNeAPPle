"""3-D domains for passages that are **not** a solid of revolution, meshed with gmsh.

The hexahedral generator in :mod:`.passage` revolves one meridian profile around the flow axis. That
is exact for a nozzle, an orifice plate or a straight-through valve, and impossible for a globe
valve: the body is a rounded shell with a **bonnet branch on a perpendicular axis**, and a body of
revolution about the flow axis cannot have a perpendicular branch at all — not with a finer mesh,
not with a different profile. This module is the other backend: the domain is built as a solid in
gmsh, meshed in tetrahedra, and handed to OpenFOAM through ``gmshToFoam``.

The customer's STEP cannot be used for this. Its 30 solids are thin shells that overlap by ~0.1 mm;
``box - metal`` (OCC, checked twice, gmsh 4.15) returns one 4 466 cm³ lump — the shells leak — plus
0.06–0.11 cm³ slivers of air caught between them. The flow volume is therefore **built from the
dimensions read off the meridian section**, which is the same route the 2-D reconstruction took and
for the same reason.

Measured on the section (``CAD_MEASUREMENTS``, mm): port axis at y = -4.7, bore radius 10, flange
faces at x = -118.5 and +61.5, body outer bottom at y = -36.4, bonnet walls at x = -42.4 and -14.5
above y = 40.
"""
from __future__ import annotations

import math
import os
import re

__all__ = ["globe_valve_domain", "build_solid_mesh", "SOLID_BUILDERS", "set_patch_types"]

#: gmsh option set that has to be the same for every domain, so a mesh is reproducible
_MESH_SIZE = {"Mesh.MeshSizeMin": 0.6, "Mesh.MeshSizeMax": 6.0, "Mesh.Algorithm3D": 1,
              "Mesh.Optimize": 1, "Mesh.MshFileVersion": 2.2, "Mesh.SaveAll": 0}


def globe_valve_domain(p: dict) -> tuple[int, dict[str, list[int]]]:
    """The fluid inside a globe valve: two ports, a rounded body, a bonnet and the plug.

    ``p`` carries the dimensions in mm (the section's measurements, not guesses). The plug and the
    stem are *cut* out of the union, so the throttling curtain is whatever the plug's face leaves
    between itself and the seat plane — the parameter the study sweeps.
    """
    import gmsh

    x_in, x_out = p["x_in"], p["x_out"]
    y0 = p["y_port"]                     # altura do eixo dos bocais
    r_bore = p["r_bore"]
    x_body = p["x_body"]                 # eixo do corpo e do pescoço
    r_body = p["r_body"]                 # raio interno do corpo
    r_bonnet = p["r_bonnet"]
    y_bonnet = p["y_bonnet"]             # até onde sobe a cavidade do pescoço
    x_seat = p["x_seat"]
    lift = p["lift"]
    r_disc = p["r_disc"]
    t_disc = p["disc_thickness"]
    r_stem = p["r_stem"]

    o = gmsh.model.occ
    inlet = o.addCylinder(x_in, y0, 0.0, x_body - x_in, 0.0, 0.0, r_bore)
    outlet = o.addCylinder(x_body, y0, 0.0, x_out - x_body, 0.0, 0.0, r_bore)
    body = o.addSphere(x_body, y0, 0.0, r_body)
    bonnet = o.addCylinder(x_body, y0, 0.0, 0.0, y_bonnet, 0.0, r_bonnet)
    o.synchronize()
    dom, _ = o.fuse([(3, inlet)], [(3, outlet), (3, body), (3, bonnet)])
    o.synchronize()

    # o obturador: face a ``lift`` da sede, espessura ``t_disc`` para montante, mais a haste
    x_face = x_seat - lift
    disc = o.addCylinder(x_face - t_disc, y0, 0.0, t_disc, 0.0, 0.0, r_disc)
    stem = o.addCylinder(x_body, y0, 0.0, 0.0, y_bonnet + 20.0, 0.0, r_stem)
    # o diafragma da sede: o anel sólido que separa montante de jusante, com o furo do obturador.
    # Sem ele o disco flutua dentro da esfera e não há estrangulamento nenhum.
    ring = o.addCylinder(x_seat, y0, 0.0, p["seat_thickness"], 0.0, 0.0, r_body)
    hole = o.addCylinder(x_seat - 1.0, y0, 0.0, p["seat_thickness"] + 2.0, 0.0, 0.0, r_bore)
    o.synchronize()
    seat, _ = o.cut([(3, ring)], [(3, hole)])
    o.synchronize()
    fluid, _ = o.cut(dom, [(3, disc), (3, stem), *seat])
    o.synchronize()
    if len(fluid) != 1:
        raise RuntimeError(f"o domínio de fluido saiu em {len(fluid)} pedaços: o obturador cortou o corpo")
    tag = fluid[0][1]

    groups: dict[str, list[int]] = {"inlet": [], "outlet": [], "wall": []}
    for _, s in gmsh.model.getBoundary([(3, tag)], oriented=False):
        x0, y0b, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, abs(s))
        if abs(x0 - x_in) < 1e-3 and abs(x1 - x_in) < 1e-3:
            groups["inlet"].append(abs(s))
        elif abs(x0 - x_out) < 1e-3 and abs(x1 - x_out) < 1e-3:
            groups["outlet"].append(abs(s))
        else:
            groups["wall"].append(abs(s))
    return tag, groups


#: name -> builder, so a :class:`~.passage.PassageSpec` can name its domain in ``meta``
SOLID_BUILDERS = {"globe_valve": globe_valve_domain}


def build_solid_mesh(passage, case_dir: str, *, timeout: float = 3600.0) -> dict:
    """Hexa-dominant mesh of the 3-D domain: the surface from gmsh, the volume from snappyHexMesh.

    Three meshers were measured on this geometry before this one, and the numbers are kept because
    each cost an afternoon:

    ==========================  ==================  ============  ==========================
    malha                       não-ortogonalidade  esbelteza     resultado no solver
    ==========================  ==================  ============  ==========================
    tetraedros (gmsh)           71° / 22°           1,06          p para em 1000 iterações
    ``polyDualMesh 90``         51° / 12,5°          4,77          67 faces invertidas, idem
    ``+ -concaveMultiCells``    88° / 13,3°          12,2          20 faces invertidas, idem
    ``snappyHexMesh`` (esta)    45° / 10,3°          1,42          p converge em 3 iterações
    ==========================  ==================  ============  ==========================

    ``mesh_size`` is the cell size **at the wall**; the background block is four times coarser and
    snappyHexMesh refines it two levels on the surfaces.
    """
    import numpy as np

    meta = dict(passage.meta)
    builder = SOLID_BUILDERS[meta["solid_builder"]]
    params = dict(meta["solid"])
    params["lift"] = float(meta.get("parameter_value", params.get("lift", 4.0)))
    size = float(params.pop("mesh_size", 2.0))
    params.pop("refine_box", None)
    params.pop("refine_size", None)

    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        for k, v in _MESH_SIZE.items():
            gmsh.option.setNumber(k, v)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 3.0)
        tag, groups = builder(params)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        points = np.array(coords, dtype=float).reshape(-1, 3)
        lookup = {int(t): i for i, t in enumerate(tags)}
        triangles: dict[str, np.ndarray] = {}
        for name, surfaces in groups.items():
            out = []
            for s in surfaces:
                kinds, _, node_tags = gmsh.model.mesh.getElements(2, s)
                for kind, tags_s in zip(kinds, node_tags, strict=False):
                    if int(kind) != 2:                       # 2 = triângulo
                        continue
                    idx = np.array([lookup[int(t)] for t in tags_s], dtype=int).reshape(-1, 3)
                    out.append(idx)
            triangles[name] = np.vstack(out) if out else np.zeros((0, 3), dtype=int)
    finally:
        gmsh.finalize()

    lo = points.min(axis=0) * 1e-3
    hi = points.max(axis=0) * 1e-3
    surface_dir = os.path.join(case_dir, "constant", "triSurface")
    os.makedirs(surface_dir, exist_ok=True)
    for name, idx in triangles.items():
        _write_stl(os.path.join(surface_dir, f"{name}.stl"), points[idx] * 1e-3)

    bg = 4.0 * size * 1e-3
    margin = 1.5 * bg
    _write_block_mesh(os.path.join(case_dir, "system", "blockMeshDict"), lo, hi, bg, margin)
    # o ponto tem de estar no **fluido**, não no meio da peça: o centro do corpo cai dentro do
    # obturador, e o snappyHexMesh então escolhe a região de fora e malha a caixa inteira (medido:
    # 2,29 L em vez de 138 cm³, e o escoamento diverge porque "entrada" e "saída" dão para o ar)
    inside = ((float(params["x_in"]) + 2.0 * float(params["r_bore"])) * 1e-3,
              float(params["y_port"]) * 1e-3, 0.0)
    _write_snappy(os.path.join(case_dir, "system", "snappyHexMeshDict"), triangles, inside)

    from .case import _openfoam

    rc, log = _openfoam("blockMesh > log.blockMesh 2>&1 && snappyHexMesh -overwrite "
                        "> log.snappyHexMesh 2>&1; tail -2 log.snappyHexMesh", case_dir, timeout)
    if rc != 0 or "FOAM FATAL" in log:
        raise RuntimeError(f"snappyHexMesh falhou: {log.strip()[:400]}")
    _keep_patches(case_dir, {"inlet", "outlet", "wall"})
    set_patch_types(case_dir, {"inlet": "patch", "outlet": "patch", "wall": "wall"})
    _, check = _openfoam("checkMesh > log.checkMesh 2>&1; grep -E 'cells:|non-orthogonality|"
                         "Max skewness|Mesh OK|Failed' log.checkMesh", case_dir, 900)
    n_cells = 0
    m = re.search(r"cells:\s+(\d+)", check)
    if m:
        n_cells = int(m.group(1))
    return {"cells": n_cells, "points": len(points), "tets": 0,
            "bbox_mm": [round(v * 1e3, 2) for v in (*lo, *hi)],
            "patches": sorted(triangles), "mesher": "snappyHexMesh hex-dominante",
            "mesh_size_mm": size, "checkmesh": check.strip().replace("\n", " | ")}


def _write_stl(path: str, triangles) -> None:
    """Binary STL, in metres — snappyHexMesh does not scale, and the CAD is in millimetres."""
    import numpy as np

    tri = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(np.uint32(len(tri)).tobytes())
        for t in tri:
            f.write(np.zeros(3, dtype=np.float32).tobytes())
            f.write(t.astype(np.float32).tobytes())
            f.write(np.zeros(1, dtype=np.uint16).tobytes())


def _write_block_mesh(path: str, lo, hi, cell: float, margin: float) -> None:
    """The background box that snappyHexMesh casts into: it must cover the part plus a margin."""

    x0, y0, z0 = (float(v) - margin for v in lo)
    x1, y1, z1 = (float(v) + margin for v in hi)
    nx = max(2, math.ceil((x1 - x0) / cell))
    ny = max(2, math.ceil((y1 - y0) / cell))
    nz = max(2, math.ceil((z1 - z0) / cell))
    body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({x0:.6f} {y0:.6f} {z0:.6f})
    ({x1:.6f} {y0:.6f} {z0:.6f})
    ({x1:.6f} {y1:.6f} {z0:.6f})
    ({x0:.6f} {y1:.6f} {z0:.6f})
    ({x0:.6f} {y0:.6f} {z1:.6f})
    ({x1:.6f} {y0:.6f} {z1:.6f})
    ({x1:.6f} {y1:.6f} {z1:.6f})
    ({x0:.6f} {y1:.6f} {z1:.6f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    borda
    {{
        type patch;
        faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7));
    }}
);
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def _write_snappy(path: str, triangles, inside) -> None:
    """Castellated + snapped, no layers: the walls are adiabatic and no-slip, not boundary-layer."""
    names = [n for n, t in triangles.items() if len(t)]
    surfaces = "\n".join(
        f"        {n} {{ level (2 2); patchInfo {{ type {'wall' if n == 'wall' else 'patch'}; }} }}"
        for n in names)
    geometry = "\n".join(f"    {n}.stl {{ type triSurfaceMesh; name {n}; }}" for n in names)
    body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
{geometry}
}}

castellatedMeshControls
{{
    maxLocalCells 4000000;
    maxGlobalCells 8000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{surfaces}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh ({inside[0]:.6f} {inside[1]:.6f} {inside[2]:.6f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def set_patch_types(case_dir: str, types: dict[str, str]) -> int:
    """Give each patch the OpenFOAM type its boundary condition needs (``wall`` for the walls).

    ``gmshToFoam`` creates every patch as ``patch``: a wall function on a patch that is not of type
    ``wall`` is a fatal error, and it is the first thing the solver complains about.
    """
    path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if not os.path.exists(path):
        return 0
    text = open(path, encoding="utf-8", errors="replace").read()
    changed = 0
    for name, kind in types.items():
        pattern = (r"(\n\s*" + re.escape(name) + r"\s*\n\s*\{\s*\n\s*type\s+)\w+")
        text, n = re.subn(pattern, lambda m, k=kind: m.group(1) + k, text, count=1)
        changed += n
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return changed


def _keep_patches(case_dir: str, keep: set[str]) -> int:
    """Drop every patch the case does not know about, after snappyHexMesh has run.

    The background box leaves its own outer patch (``borda``) in ``constant/polyMesh/boundary``: the
    domain is closed by the inlet and outlet caps, so it ends with few or no faces. A field written
    before the mesh existed has one entry per field patch, and OpenFOAM refuses a mesh with a patch
    the field does not mention — so the mesh is trimmed to the three patches the case writes.

    ``boundary`` is a ``PtrList``: it carries the number of entries *before* the list, and dropping
    an entry without decrementing it leaves a file OpenFOAM refuses to read.
    """
    path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if not os.path.exists(path):
        return 0
    text = open(path, encoding="utf-8", errors="replace").read()
    dropped = 0
    for name in re.findall(r"^\s{4}(\w+)\s*\n\s*\{", text, flags=re.M):
        if name in keep:
            continue
        block = re.search(r"\n    " + re.escape(name) + r"\s*\n    \{.*?\n    \}", text, flags=re.S)
        if block:
            text = text.replace(block.group(0), "", 1)
            dropped += 1
    if not dropped:
        return 0
    head = text[:text.index("(")]
    counts = list(re.finditer(r"(?m)^\s*(\d+)\s*$", head))
    if counts:
        last = counts[-1]
        total = int(last.group(1))
        head = head[:last.start()] + str(total - dropped) + head[last.end():]
        text = head + text[text.index("("):]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return dropped
