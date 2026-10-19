"""3-D views of the reference comparisons (pinneapple_veriphysics.mesh_views): the values are PC701's, on the mesh."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinneapple_veriphysics.mesh_views import mesh_views
from pinneapple_veriphysics.validation_lab import CONFIG_NAME, config_from_spec, run_validation

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "apps" / "validation_lab" / "examples"
HEAD = 'FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n'


def _foam_box(case: Path, n=(4, 3, 3), size=(2.0, 1.0, 1.0), fields=None):
    """A structured hexahedral polyMesh (internal faces first, then one patch per box face) and volScalarFields at
    time 1; fields: name -> function of the cell centres."""
    nx, ny, nz = n
    xs, ys, zs = (np.linspace(0, s, k + 1) for s, k in zip(size, n, strict=True))
    pid = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)  # noqa: E731
    cid = lambda i, j, k: i + nx * (j + ny * k)  # noqa: E731
    pts = [(x, y, z) for z in zs for y in ys for x in xs]
    internal, bnd = [], {f: [] for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = cid(i, j, k)
                fx = [pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i + 1, j + 1, k + 1), pid(i + 1, j, k + 1)]
                fy = [pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i + 1, j + 1, k + 1), pid(i + 1, j + 1, k)]
                fz = [pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1), pid(i, j + 1, k + 1)]
                for f, last, nb, key in ((fx, i == nx - 1, (i + 1, j, k), "xmax"), (fy, j == ny - 1, (i, j + 1, k), "ymax"),
                                         (fz, k == nz - 1, (i, j, k + 1), "zmax")):
                    (bnd[key].append((f, c)) if last else internal.append((f, c, cid(*nb))))
                if i == 0:
                    bnd["xmin"].append(([pid(0, j, k), pid(0, j, k + 1), pid(0, j + 1, k + 1), pid(0, j + 1, k)], c))
                if j == 0:
                    bnd["ymin"].append(([pid(i, 0, k), pid(i + 1, 0, k), pid(i + 1, 0, k + 1), pid(i, 0, k + 1)], c))
                if k == 0:
                    bnd["zmin"].append(([pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)], c))
    faces = [f for f, _, _ in internal] + [f for v in bnd.values() for f, _ in v]
    owner = [o for _, o, _ in internal] + [o for v in bnd.values() for _, o in v]
    pm = case / "constant" / "polyMesh"
    pm.mkdir(parents=True)
    w = lambda name, cls, body: (pm / name).write_text(HEAD.format(cls=cls, obj=name) + body)  # noqa: E731
    w("points", "vectorField", f"{len(pts)}\n(\n" + "\n".join(f"({x} {y} {z})" for x, y, z in pts) + "\n)\n")
    w("faces", "faceList", f"{len(faces)}\n(\n" + "\n".join(f"4({' '.join(map(str, f))})" for f in faces) + "\n)\n")
    w("owner", "labelList", f"{len(owner)}\n(\n" + "\n".join(map(str, owner)) + "\n)\n")
    w("neighbour", "labelList", f"{len(internal)}\n(\n" + "\n".join(str(nb) for _, _, nb in internal) + "\n)\n")
    start, body = len(internal), []
    for name, v in bnd.items():
        body.append(f"    {name}\n    {{\n        type wall;\n        nFaces {len(v)};\n        startFace {start};\n    }}")
        start += len(v)
    w("boundary", "polyBoundaryMesh", f"{len(bnd)}\n(\n" + "\n".join(body) + "\n)\n")
    C = np.array([((i + .5) * size[0] / nx, (j + .5) * size[1] / ny, (k + .5) * size[2] / nz)
                  for k in range(nz) for j in range(ny) for i in range(nx)])
    (case / "1").mkdir()
    for name, fn in (fields or {}).items():
        v = fn(C)
        bf = "\n".join(f"    {b} {{ type zeroGradient; }}" for b in bnd)
        (case / "1" / name).write_text(HEAD.format(cls="volScalarField", obj=name) + "dimensions [0 2 -2 0 0 0 0];\n"
                                       f"internalField nonuniform List<scalar>\n{len(v)}\n(\n"
                                       + "\n".join(f"{x:.10g}" for x in v) + f"\n);\nboundaryField\n{{\n{bf}\n}}\n")
    (case / "system").mkdir()
    (case / "system" / "controlDict").write_text(HEAD.format(cls="dictionary", obj="controlDict") + "application icoFoam;\n")
    return C


def _run(d: Path, spec: dict):
    (d / CONFIG_NAME).write_text(config_from_spec(spec))
    lab = run_validation(d).to_dict()
    return lab, mesh_views(d, report=lab)


def test_vtu_example_views_match_pc701_and_deform_with_the_displacement(tmp_path):
    for f in ("cantilever_rom.vtu", "cantilever_fem.vtu"):
        (tmp_path / f).write_bytes((EX / f).read_bytes())
    lab, views = _run(tmp_path, {"results": [{"path": "cantilever_rom.vtu", "units": {"D": "m", "von_mises": "MPa"},
                                              "bc": {"x=min": {"D": [0, 0, 0]}},
                                              "reference": {"path": "cantilever_fem.vtu", "rel-l2": 0.05}}]})
    pc701 = {o["subject"]: o["value"] for o in lab["outcomes"] if o["rule"] == "PC701"}
    assert pc701["D"] == pytest.approx(0.2116, abs=1e-4) and pc701["von_mises"] == pytest.approx(0.2775, abs=1e-4)
    (v,) = views
    for n in ("D", "von_mises"):
        assert v["errors"][n]["rel_l2"] == pytest.approx(pc701[n], rel=1e-9)
        assert "PC701 in the report" in v["errors"][n]["summary"][0]
    meta = v["view"].meta
    assert [f["name"] for f in meta["fields"]][:3] == ["D:pred", "D:ref", "D:error"]
    assert set(meta["deform"]) == {"prediction", "reference"}
    assert all(f["deform"] for f in meta["fields"])                     # von Mises is drawn deformed too
    bc = next(m for m in meta["markers"] if m["kind"] == "bc")
    assert bc["name"] == "x=min" and bc["n"] == 28                      # 4 x 7 nodes on the clamp face
    hs = v["errors"]["von_mises"]["hotspots"]
    assert len(hs) >= 3 and hs[0]["name"] == "R1" and hs[0]["max"] == pytest.approx(v["errors"]["von_mises"]["max"])
    assert "x=min" in hs[2]["touches"] or "x=min" in hs[3]["touches"]   # stress errors sit at the clamp
    labelled = [m["name"] for m in meta["markers"] if m["kind"] == "hotspot" and m["label"] and m["fields"][0] == "von_mises:pred"]
    assert labelled == ["R1", "R2", "R3"]                               # the rest are in the table, points only


def test_openfoam_3d_case_cell_values_and_patch_markers(tmp_path):
    ref_fn = lambda C: 1 + C[:, 0] ** 2                                   # noqa: E731
    _foam_box(tmp_path / "pred", fields={"p": lambda C: ref_fn(C) + 0.3 * (C[:, 0] > 1.5)})
    _foam_box(tmp_path / "ref", fields={"p": ref_fn})
    lab, views = _run(tmp_path, {"results": [{"path": "pred", "bc": {"xmin": {"p": 1.0}},
                                              "reference": {"path": "ref", "fields": ["p"], "rel-l2": 0.01}}]})
    (v,) = views
    e = v["errors"]["p"]
    pc701 = next(o["value"] for o in lab["outcomes"] if o["rule"] == "PC701")
    assert e["rel_l2"] == pytest.approx(pc701, rel=1e-9) and e["max"] == pytest.approx(0.3)
    assert e["hotspots"][0]["touches"] and "x=max" in e["hotspots"][0]["touches"]
    meta = v["view"].meta
    assert meta["cells"] == 36 and {f["where"] for f in meta["fields"]} == {"cell"}
    assert meta["deform"] == {}
    patch = next(m for m in meta["markers"] if m["kind"] == "bc")
    assert patch["name"] == "xmin" and patch["n"] == 9 and np.allclose(np.array(patch["points"])[:, 0], 0)


def test_two_dimensional_and_tabular_results_get_no_view(tmp_path):
    x = y = np.linspace(0, 1, 9)
    X, Y = np.meshgrid(x, y, indexing="ij")
    np.savez(tmp_path / "T.npz", x=x, y=y, T=X + Y)
    np.savez(tmp_path / "Tref.npz", x=x, y=y, T=X + Y + 0.01)
    _, views = _run(tmp_path, {"results": [{"path": "T.npz", "reference": {"path": "Tref.npz"}}]})
    assert views == []
