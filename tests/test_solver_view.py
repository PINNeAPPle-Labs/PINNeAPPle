"""Solver results as 3-D views (pinneapple_tools.visualization.studio.solver_view) with the supports and loads of a
CalculiX deck (pinneapple_data.cae.inp.deck_conditions) and the patch conditions of an OpenFOAM case
(pinneapple_data.cae.foam.boundary_conditions)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from pinneapple_data.cae import foam
from pinneapple_data.cae.inp import deck_conditions, read_deck
from pinneapple_data.cae.upload import expand
from pinneapple_tools.visualization.studio.solver_view import find_result, solver_result_view

ROOT = Path(__file__).resolve().parents[1]
BEAM = ROOT / "examples" / "calculix_cantilever"
CAVITY = ROOT / "apps" / "simulation_comparator" / "examples" / "cavity_Re100_20x20.zip"


def _cavity(tmp: Path) -> Path:
    for name, b in expand([("c.zip", CAVITY.read_bytes())]).items():
        p = tmp / "case" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    return tmp


def test_deck_conditions_named_sets_dof_ranges_and_summed_loads():
    c = deck_conditions(read_deck((BEAM / "beam.inp").read_text(), "beam.inp"))
    bc, load = c
    assert bc["kind"] == "bc" and bc["name"] == "fixed ux, uy, uz" and len(bc["node_ids"]) == 21
    assert load["kind"] == "load" and load["keyword"] == "*CLOAD" and len(load["node_ids"]) == 21
    assert "Fz = -1000" in load["name"]                                 # 21 x -47.619 N


def test_deck_conditions_prescribed_values_pressure_and_body_loads():
    deck = read_deck("""*NODE
1, 0, 0, 0
2, 1, 0, 0
3, 1, 1, 0
4, 0, 1, 0
5, 0, 0, 1
6, 1, 0, 1
7, 1, 1, 1
8, 0, 1, 1
*ELEMENT, TYPE=C3D8, ELSET=E
1, 1, 2, 3, 4, 5, 6, 7, 8
*BOUNDARY
1, 1, 3
2, 1, 1, 0.001
*STEP
*STATIC
*DLOAD
1, P2, 1e5
*DLOAD
E, GRAV, 9.81, 0, 0, -1
*END STEP
""")
    bc, p, g = deck_conditions(deck)
    assert bc["name"] == "prescribed ux, uy, uz" and bc["detail"] == "ux = 0.001" and bc["node_ids"] == [1, 2]
    assert p["kind"] == "load" and p["node_ids"] == list(range(1, 9)) and "P2 1e5" in p["detail"]
    assert g["node_ids"] == [] and "body load" in g["detail"]


def test_openfoam_boundary_conditions_from_the_first_time():
    fs = expand([("c.zip", CAVITY.read_bytes())])
    m = foam.read_polymesh(fs, "")
    bc = foam.boundary_conditions(fs, m)
    assert bc == {"movingWall": {"U": "fixedValue uniform (1 0 0)", "p": "zeroGradient"},
                  "fixedWalls": {"U": "noSlip", "p": "zeroGradient"}}           # frontAndBack is empty: left out


def test_calculix_result_view_deforms_and_marks_the_deck(tmp_path):
    for f in ("beam.inp", "beam.frd"):
        shutil.copy(BEAM / f, tmp_path / f)
    assert find_result(tmp_path) == ("calculix", tmp_path / "beam.frd")
    fv, info = solver_result_view(tmp_path)
    assert info["solver"] == "CalculiX" and info["deck"] == "beam.inp" and "DISP" in info["fields"]
    meta = fv.meta
    names = {f["name"]: f for f in meta["fields"]}
    assert names["STRESS"]["label"] == "von Mises stress" and names["STRESS"]["deform"] == "displacement"
    from pinneapple_data.cae.frd import read_frd
    u = np.asarray(read_frd((tmp_path / "beam.frd").read_bytes()).point_data["DISP"])[:, :3]
    assert meta["deform"]["displacement"]["max"] == pytest.approx(np.linalg.norm(u, axis=1).max())
    kinds = {m["name"]: (m["kind"], m["n"]) for m in meta["markers"]}
    assert kinds["fixed ux, uy, uz"] == ("bc", 21) and next(v for k, v in kinds.items() if k.startswith("load")) == ("load", 21)
    assert meta["faces"] == 336 and not meta["decimated"]
    small, _ = solver_result_view(tmp_path, max_faces=150)            # the size guide: large surfaces are decimated
    assert small.meta["decimated"] and small.meta["faces"] < meta["faces"]


def test_openfoam_result_view_cells_patches_and_2d_note(tmp_path):
    fv, info = solver_result_view(_cavity(tmp_path))
    assert info["kind"] == "openfoam" and set(info["fields"]) == {"U", "p"} and info["cells"] == 400
    kinds = {c["name"]: (c["kind"], c["role"]) for c in info["conditions"]}
    assert kinds == {"movingWall": ("bc", "driving wall"), "fixedWalls": ("bc", "")}     # all boundary conditions
    assert "2-D case" in info["notes"][0] and fv.meta["deform"] == {}
    assert "|U| = 1 m/s" in info["notes"][1] and "cell-centre values" in info["notes"][1]   # the lid moves at 1 m/s
    assert {f["where"] for f in fv.meta["fields"]} == {"cell"}
    lid = next(m for m in fv.meta["markers"] if m["name"] == "movingWall (driving wall)")
    assert lid["n"] == 20 and np.allclose(np.array(lid["points"])[:, 1], 0.1)


def test_no_result_is_an_error(tmp_path):
    assert find_result(tmp_path) is None
    with pytest.raises(ValueError, match="no CalculiX"):
        solver_result_view(tmp_path)
