"""Meshing the exported geometry: blockMesh background + snappyHexMesh (always available), or cfMesh's
``cartesianMesh`` when it is installed (``mesher="auto"`` picks it then).

:func:`write_internal_mesh` writes the dictionaries of an internal-flow mesh (the fluid is *inside* the closed STL,
one STL solid per patch becomes one OpenFOAM patch) and the ``Allmesh`` script; the snappyHexMesh dictionary is
:func:`pinneapple_simulation.numerical_solvers.external_flow.snappy_dict`, the one the external-flow solver uses.
External flows are meshed by :class:`~pinneapple_simulation.numerical_solvers.external_flow.ExternalFlow`'s own
dictionaries; both get the same ``Allmesh`` (:func:`allmesh_script`): blockMesh, surfaceFeatureExtract, snappyHexMesh
(decomposed with MPI when ``procs > 1``, then ``reconstructParMesh``), so the ``openfoam.mesh`` action of
:mod:`pinneapple_tools.automation` runs it and checks it with checkMesh.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MeshSpec", "write_internal_mesh", "allmesh_script", "allrun_script", "decompose_dict", "split_procs",
           "has_cfmesh", "MPIRUN"]

MPIRUN = "mpirun --allow-run-as-root --oversubscribe -np"
HDR = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n"


@dataclass
class MeshSpec:
    """``cells_across``: cells across ``length`` (a parameter name, mm, or a number in mm) near the walls;
    ``level``: snappyHexMesh surface level (the background is ``level`` times coarser by 2); ``mesher``: snappy,
    cfmesh or auto. ``max_non_ortho`` is the snappy quality limit."""
    cells_across: float = 16.0
    length: str | float = "D"
    level: int = 1
    mesher: str = "snappy"
    feature_angle: float = 150.0
    max_non_ortho: float = 65.0

    def cell_size_m(self, parameters: dict[str, float]) -> float:
        L = float(parameters[self.length]) if isinstance(self.length, str) else float(self.length)
        return L * 1e-3 / self.cells_across

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_cfmesh(bashrc: str | None = None) -> bool:
    from pinneapple_simulation.numerical_solvers.external_flow import FOAM_BASHRC
    r = subprocess.run(["bash", "-c", f'source "{bashrc or FOAM_BASHRC}" >/dev/null 2>&1; command -v cartesianMesh'],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _w(case: Path, rel: str, cls: str, body: str) -> None:
    p = case / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HDR % (cls, p.name) + body)


def split_procs(procs: int, size: np.ndarray) -> tuple[int, int, int]:
    """Split ``procs`` into (nx, ny, nz) along the longest sides of the box first."""
    n = [1, 1, 1]
    left = int(procs)
    f = 2
    factors = []
    while left > 1:
        while left % f == 0:
            factors.append(f)
            left //= f
        f += 1
    for p in sorted(factors, reverse=True):
        i = int(np.argmax(np.asarray(size, float) / np.asarray(n)))
        n[i] *= p
    return n[0], n[1], n[2]


def decompose_dict(case: Path, procs: int, size: np.ndarray) -> None:
    """hierarchical decomposition (scotch / metis can be stubs in some builds)."""
    n = split_procs(procs, size)
    _w(case, "system/decomposeParDict", "dictionary", f"numberOfSubdomains {procs};\nmethod hierarchical;\n"
       f"hierarchicalCoeffs {{ n ({n[0]} {n[1]} {n[2]}); order xyz; }}\n")


def allmesh_script(procs: int, mesher: str = "snappy") -> str:
    head = '#!/bin/bash\ncd "${0%/*}" || exit 1\nset -o pipefail\n'
    if mesher == "cfmesh":
        return head + "cartesianMesh > log.cartesianMesh 2>&1 || exit 1\n"
    s = head + ("blockMesh > log.blockMesh 2>&1 || exit 1\n"
                "surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 || exit 1\n")
    if procs > 1:
        s += (f"decomposePar -force > log.decomposePar.mesh 2>&1 || exit 1\n"
              f"{MPIRUN} {procs} snappyHexMesh -overwrite -parallel > log.snappyHexMesh 2>&1 || exit 1\n"
              "reconstructParMesh -constant > log.reconstructParMesh 2>&1 || exit 1\n"
              "rm -rf processor*\n")
    else:
        s += "snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 || exit 1\n"
    return s


def allrun_script(solver: str, procs: int, scalar: bool = False) -> str:
    """Copy ``0.orig`` to ``0`` and run ``solver`` (decomposed with MPI when ``procs > 1``, then the latest time
    reconstructed); the solver log is ``log.<solver>`` in the case. ``scalar``: then 20 steady iterations of
    ``scalarTransportFoam`` on the converged flow (log ``log.scalarTransport``)."""
    s = '#!/bin/bash\ncd "${0%/*}" || exit 1\nrm -rf 0 processor* && cp -r 0.orig 0 || exit 1\n'
    if procs > 1:
        s += (f"decomposePar -force > log.decomposePar 2>&1 || exit 1\n"
              f"{MPIRUN} {procs} {solver} -parallel > log.{solver} 2>&1 || exit 1\n"
              "reconstructPar -latestTime > log.reconstructPar 2>&1 || exit 1\n"
              "rm -rf processor*\n")
    else:
        s += f"{solver} > log.{solver} 2>&1 || exit 1\n"
    if scalar:
        s += ("L=$(foamListTimes -latestTime -noZero 2>/dev/null | tail -1)\n"
              "for f in 0/*; do [ -e \"$L/${f#0/}\" ] || cp \"$f\" \"$L\"/; done\n"
              "cp system/controlDict system/controlDict.flow\n"
              "sed -e \"s/^startFrom .*/startFrom latestTime;/\" -e \"s/^endTime .*/endTime $((L + 20));/\" "
              "-e \"s/^writeInterval .*/writeInterval 20;/\" -e \"s/^application .*/application scalarTransportFoam;/\" "
              "system/controlDict.flow > system/controlDict\n"
              "scalarTransportFoam > log.scalarTransport 2>&1 || exit 1\n"
              "N=$(foamListTimes -latestTime -noZero 2>/dev/null | tail -1)\n"
              "for f in \"$L\"/*; do [ -e \"$N/${f##*/}\" ] || cp -r \"$f\" \"$N\"/; done\n"
              "cp system/controlDict.flow system/controlDict\n")
    return s


def _script(case: Path, name: str, text: str) -> None:
    p = case / name
    p.write_text(text)
    p.chmod(0o755)


def write_internal_mesh(case: str | Path, stl: str | Path, patches: dict[str, str], inside: np.ndarray,
                        cell_size: float, spec: MeshSpec, procs: int = 1, bashrc: str | None = None) -> dict[str, Any]:
    """Dictionaries and ``Allmesh`` of an internal-flow mesh of the closed multi-solid ``stl`` (metres).

    ``patches``: STL solid -> patch type (``patch``, ``wall``, ``symmetry``); ``inside``: a point in the fluid
    (snappy's ``locationInMesh``); ``cell_size``: the cell size at the walls (m). Returns what was written."""
    from pinneapple_simulation.numerical_solvers.external_flow import snappy_dict
    case = Path(case)
    tri = case / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    shutil.copy(stl, tri / "geometry.stl")
    pts = _stl_points(tri / "geometry.stl")
    lo, hi = pts.min(0), pts.max(0)
    mesher = spec.mesher
    if mesher == "auto":
        mesher = "cfmesh" if has_cfmesh(bashrc) else "snappy"
    h0 = cell_size * 2 ** spec.level                          # background cell
    pad = 1.5 * h0
    lo_b, hi_b = lo - pad, hi + pad
    n = [max(1, int(math.ceil((b - a) / h0))) for a, b in zip(lo_b, hi_b, strict=True)]
    hi_b = lo_b + np.asarray(n) * h0                           # cubic cells
    x0, y0, z0 = (float(v) for v in lo_b)
    x1, y1, z1 = (float(v) for v in hi_b)
    _w(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices ( ({x0!r} {y0!r} {z0!r}) ({x1!r} {y0!r} {z0!r}) ({x1!r} {y1!r} {z0!r}) ({x0!r} {y1!r} {z0!r})
           ({x0!r} {y0!r} {z1!r}) ({x1!r} {y0!r} {z1!r}) ({x1!r} {y1!r} {z1!r}) ({x0!r} {y1!r} {z1!r}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
edges ();
boundary ( background {{ type patch; faces ( (0 3 2 1) (4 5 6 7) (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) ); }} );
mergePatchPairs ();
""")
    _w(case, "system/surfaceFeatureExtractDict", "dictionary",
       f"geometry.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle "
       f"{spec.feature_angle:g}; }} subsetFeatures {{ nonManifoldEdges no; openEdges yes; }} writeObj no; }}\n")
    lv = int(spec.level)
    regions = " ".join(f"{p} {{ name {p}; }}" for p in patches)
    geom = f"    geometry.stl {{ type triSurfaceMesh; name geometry; regions {{ {regions} }} }}"
    surf_regions = " ".join(f"{p} {{ level ({lv} {lv}); patchInfo {{ type {t}; }} }}" for p, t in patches.items())
    surfaces = f"        geometry {{ level ({lv} {lv}); regions {{ {surf_regions} }} }}"
    features = f'        {{ file "geometry.eMesh"; level {lv}; }}'
    refine = f"        geometry {{ mode inside; levels ((1E15 {lv})); }}"
    body = snappy_dict(geom, features, surfaces, refine, inside).replace("maxNonOrtho 65;",
                                                                           f"maxNonOrtho {spec.max_non_ortho:g};")
    _w(case, "system/snappyHexMeshDict", "dictionary", body)
    if mesher == "cfmesh":
        ren = " ".join(f"{p} {{ newName {p}; type {t}; }}" for p, t in patches.items())
        _w(case, "system/meshDict", "dictionary", f'surfaceFile "constant/triSurface/geometry.stl";\n'
           f"maxCellSize {cell_size!r};\nrenameBoundary {{ newPatchNames {{ {ren} }} }}\n")
    decompose_dict(case, procs, hi - lo)
    _script(case, "Allmesh", allmesh_script(procs, mesher))
    return {"mesher": mesher, "background_cells": n, "background_cell_m": h0, "wall_cell_m": cell_size,
            "bounds_m": [lo.tolist(), hi.tolist()], "location_in_mesh": [float(v) for v in inside],
            "patches": dict(patches), "procs": procs}


def _stl_points(path: Path) -> np.ndarray:
    return np.array([ln.split()[1:4] for ln in path.read_text().splitlines() if ln.lstrip().startswith("vertex")],
                    float)
