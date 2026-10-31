"""Known answers for generate_mesh and the geometry.unstructured_mesh node."""
import json
import math

import numpy as np
import pytest

pytest.importorskip("gmsh")

from pinneapple_design.geometry.mesh.generate import X_OUTPUTS, generate_mesh  # noqa: E402

BL = {"first_layer": 0.01, "n_layers": 3, "growth": 1.2}


def _box_surface(tmp_path, open_faces=0):
    """A closed unit-box STL written by gmsh; ``open_faces`` triangles removed to open it."""
    import gmsh
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.25)
    gmsh.model.mesh.generate(2)
    p = tmp_path / "box.stl"
    gmsh.write(str(p))
    gmsh.finalize()
    if open_faces:
        parts = p.read_text().split("facet normal")
        p = tmp_path / "open.stl"
        p.write_text(parts[0] + "facet normal" + "facet normal".join(parts[1 + open_faces:]))
    return p


def test_box_and_rectangle_are_exact():
    box = generate_mesh("box", [1.0, 2.0, 0.5], h=0.25)
    assert box["dim"] == 3 and set(box["element_counts"]) == {"tetra"}
    assert box["volume"] == pytest.approx(1.0, rel=1e-9) and box["area"] == pytest.approx(7.0, rel=1e-9)
    rect = generate_mesh("rectangle", [2.0, 0.5], h=0.1)
    assert rect["dim"] == 2 and rect["volume"] is None and rect["area"] == pytest.approx(1.0, rel=1e-9)


def test_curved_volumes_converge_to_the_exact_value():
    exact = math.pi * 0.5 ** 2 * 2.0
    err = [abs(generate_mesh("cylinder", [0.5, 2.0], h=h)["volume"] - exact) / exact for h in (0.2, 0.1)]
    assert err[1] < 0.01 and err[1] < err[0]            # facets inscribed in the circle: below, closing in
    disk = generate_mesh("disk", [0.5], h=0.05)["area"]
    assert disk == pytest.approx(math.pi * 0.25, rel=0.01)
    sphere = generate_mesh("sphere", [0.5], h=0.1)["volume"]
    assert sphere == pytest.approx(4 / 3 * math.pi * 0.125, rel=0.03)


def test_element_count_grows_like_h_to_the_minus_dimension():
    hs = np.array([0.2, 0.1, 0.05])
    n3 = [generate_mesh("box", h=h)["n_elements"] for h in hs]
    n2 = [generate_mesh("rectangle", h=h)["n_elements"] for h in hs]
    p3 = np.polyfit(np.log(hs), np.log(n3), 1)[0]
    p2 = np.polyfit(np.log(hs), np.log(n2), 1)[0]
    assert -3.3 < p3 < -2.6 and -2.2 < p2 < -1.8, (p3, p2)


def test_quality_of_primitive_meshes():
    for shape, h in (("box", 0.1), ("cylinder", 0.1), ("sphere", 0.1), ("rectangle", 0.05), ("disk", 0.05)):
        out = generate_mesh(shape, h=h)
        assert out["sicn_min"] > 0.25 and out["gamma_min"] > 0.25, (shape, out["sicn_min"], out["gamma_min"])
        assert 0.5 * h < out["h_mean"] < 1.5 * h and out["aspect_ratio_max"] < 4


def test_boundary_layer_and_refinement():
    bl = generate_mesh("box", h=0.2, boundary_layer=BL)
    assert bl["element_counts"].get("wedge", 0) > 0 and bl["element_counts"].get("tetra", 0) > 0
    assert bl["volume"] == pytest.approx(1.0, rel=1e-4) and bl["area"] == pytest.approx(6.0, rel=1e-4)
    assert bl["h_min"] == pytest.approx(0.01, rel=1e-6) and bl["sicn_min"] > 0
    bl2 = generate_mesh("rectangle", h=0.1, boundary_layer={"first_layer": 0.005, "n_layers": 4, "growth": 1.3})
    assert bl2["h_min"] == pytest.approx(0.005, rel=1e-6) and bl2["area"] == pytest.approx(1.0, rel=1e-9)
    base = generate_mesh("box", h=0.2)
    ref = generate_mesh("box", h=0.2, refine=[{"center": [0.5, 0.5, 0.5], "radius": 0.2, "h": 0.04}])
    assert ref["h_min"] < 0.06 and ref["n_elements"] > 3 * base["n_elements"]


def test_a_layer_too_thick_for_the_edges_is_refused_clearly():
    with pytest.raises(ValueError, match="boundary layer"):
        generate_mesh("cylinder", [0.5, 1.0], h=0.08,
                      boundary_layer={"first_layer": 0.008, "n_layers": 4, "growth": 1.3})


def test_files_and_origin(tmp_path):
    out = generate_mesh("rectangle", [1.0, 1.0], h=0.2, out_dir=str(tmp_path), origin=[2.0, 3.0])
    assert set(out["files"]) == {"msh", "vtu", "glb", "view"}
    assert (tmp_path / "mesh.msh").read_text().startswith("$MeshFormat")
    assert b"<VTKFile" in (tmp_path / "mesh.vtu").read_bytes()[:200]
    assert (tmp_path / "quality.glb").read_bytes()[:4] == b"glTF"
    meta = json.loads((tmp_path / "quality.json").read_text())
    assert meta["schema"] == "pinneapple-field-view/v1"
    assert [f["name"] for f in meta["fields"]] == ["sicn", "gamma", "aspect_ratio"]
    assert meta["bounds"]["min"][:2] != [0.0, 0.0] or meta["bounds"]["max"][:2] != [1.0, 1.0]
    assert generate_mesh("rectangle", [1.0, 1.0], h=0.2)["files"] == {}


def test_stl_step_and_open_surface(tmp_path):
    closed = generate_mesh(str(_box_surface(tmp_path)), h=0.2)
    assert closed["volume"] == pytest.approx(1.0, rel=1e-6) and closed["area"] == pytest.approx(6.0, rel=1e-6)
    assert closed["backend"].startswith("mesh_from_surface")
    walls = generate_mesh(str(_box_surface(tmp_path)), h=0.2, boundary_layer=BL)
    assert walls["element_counts"].get("wedge", 0) > 0 and walls["volume"] == pytest.approx(1.0, rel=1e-4)
    with pytest.raises(ValueError, match=r"not closed \(\d+ open edges, 0 non-manifold edges\)"):
        generate_mesh(str(_box_surface(tmp_path, open_faces=4)), h=0.2)
    import gmsh
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.addBox(0, 0, 0, 1000, 500, 250)          # millimetres
    gmsh.model.occ.synchronize()
    gmsh.write(str(tmp_path / "box_mm.step"))
    gmsh.finalize()
    step = generate_mesh(str(tmp_path / "box_mm.step"), h=0.1, scale=1e-3)
    assert step["volume"] == pytest.approx(0.125, rel=1e-9) and step["area"] == pytest.approx(1.75, rel=1e-9)


def test_bad_inputs_are_refused(tmp_path):
    for kw in ({"geometry": "torus"}, {"geometry": "box", "h": 0}, {"geometry": "box", "dimensions": [1, 1]},
               {"geometry": "box", "dimensions": [1, -1, 1]}, {"geometry": str(tmp_path / "none.stl")},
               {"geometry": "box", "boundary_layer": {"first_layer": 0, "n_layers": 3}},
               {"geometry": "box", "refine": [{"center": [0, 0, 0], "radius": 0.1}]}):
        with pytest.raises(ValueError):
            generate_mesh(**kw)


def test_file_limits_urls_and_missing_gmsh(tmp_path, monkeypatch):
    stl = _box_surface(tmp_path)
    with pytest.raises(ValueError, match="above the 0.001 MB limit"):
        generate_mesh(str(stl), h=0.2, max_input_mb=0.001)
    monkeypatch.setenv("PINNEAPPLE_MESH_MAX_INPUT_MB", "0.001")
    with pytest.raises(ValueError, match="MB limit"):
        generate_mesh(str(stl), h=0.2)
    monkeypatch.delenv("PINNEAPPLE_MESH_MAX_INPUT_MB")
    with pytest.raises(ValueError, match="URLs are not fetched"):
        generate_mesh("https://example.com/part.stl", h=0.2)
    monkeypatch.setitem(__import__("sys").modules, "gmsh", None)
    with pytest.raises(RuntimeError, match="pip install gmsh"):
        generate_mesh("box", h=0.5)


def test_estimate_and_guard_against_a_huge_mesh(tmp_path):
    from pinneapple_design.geometry.mesh.generate import estimate_elements
    # the estimate is an upper bound within ~30 % of what gmsh makes
    for shape, dims, h in (("box", [1, 1, 1], 0.1), ("cylinder", [0.5, 2.0], 0.1), ("rectangle", [2, 1], 0.05),
                           ("disk", [0.5], 0.05)):
        from pinneapple_design.geometry.mesh.generate import _primitive_size
        est = estimate_elements(*_primitive_size(shape, dims), h, [], None)
        n = generate_mesh(shape, dims, h=h)["n_elements"]
        assert n <= est <= 1.35 * n, (shape, n, est)
    with pytest.raises(ValueError, match=r"use h >= 0\.0(4|5)\d* m"):
        generate_mesh("box", h=0.01, max_elements=50_000)          # 6e6 estimated: refused before meshing
    with pytest.raises(ValueError, match="max_elements"):
        generate_mesh("box", h=0.2, refine=[{"center": [0.5, 0.5, 0.5], "radius": 0.4, "h": 0.002}])
    with pytest.raises(ValueError, match="box.stl"):
        generate_mesh(str(_box_surface(tmp_path)), h=0.005, max_elements=10_000)
    assert generate_mesh("box", h=0.1, max_elements=10_000)["n_elements"] < 10_000


def test_nothing_is_written_outside_out_dir(tmp_path, monkeypatch):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    out = tmp_path / "out"
    generate_mesh(str(_box_surface(tmp_path)), h=0.25, out_dir=str(out))
    generate_mesh("cylinder", h=0.2, boundary_layer=BL, out_dir=str(out))
    assert list(work.iterdir()) == []
    assert sorted(p.name for p in out.iterdir()) == ["mesh.msh", "mesh.vtu", "quality.glb", "quality.json"]


def test_outputs_match_the_declaration():
    out = generate_mesh("box", h=0.5)
    assert [o["name"] for o in X_OUTPUTS["generate_mesh"]] == list(out)


def test_flow_node_meshes_the_domain_and_a_file(tmp_path, monkeypatch):
    import pinneapple_flow as pf
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PINNEAPPLE_FLOW_DATA", str(tmp_path))
    _box_surface(tmp_path)
    f = pf.Flow("mesh", nodes=[
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": 2, "time_dependent": False,
                                                            "x_range": [0, 2], "y_range": [-1, 0]}},
        {"id": "dom", "type": "geometry.unstructured_mesh", "params": {"shape": "domain", "h": 0.1}},
        {"id": "stl", "type": "geometry.unstructured_mesh", "params": {"shape": "file", "file": "box.stl",
                                                                        "h": 0.25}},
        {"id": "bad", "type": "geometry.unstructured_mesh", "params": {"shape": "file", "file": "../x.stl"}}],
        edges=[{"source": "geo", "source_port": "geometry", "target": "dom", "target_port": "geometry"}])
    r = pf.run(f, workdir=tmp_path / "w")
    assert r.nodes["dom"]["status"] == "done", r.nodes["dom"]["error"]
    rep = r.report("dom")
    assert rep["metrics"]["area"] == pytest.approx(2.0, rel=1e-9)
    assert rep["tables"][0]["title"] == "quality" and "scene" in rep["views"]
    assert rep["views"]["scene"]["meta"]["schema"] == "pinneapple-field-view/v1"
    m = r.load("dom", "mesh")
    assert m["summary"]["n_elements"] == rep["metrics"]["n_elements"] and "msh" in m["files"]
    assert r.report("stl")["metrics"]["volume"] == pytest.approx(1.0, rel=1e-6)
    assert r.nodes["bad"]["status"] == "failed" and "data folder" in r.nodes["bad"]["error"]


def test_polygon_and_flow_domains(tmp_path):
    """A polygon is meshed as one curve (its vertices do not set the size: straight elements of size h follow it, so
    the meshed area is that of the inscribed polygon of side h); ``enclosure`` meshes the box minus the body: area /
    volume of the box minus the polygon, the disk, the STL sphere."""
    from pinneapple_simulation.immersed.geometry import icosphere, write_stl
    n = 512
    t = 2 * np.pi * np.arange(n) / n
    V = np.column_stack([0.2 + 0.05 * np.cos(t), 0.2 + 0.05 * np.sin(t)])
    poly_area = 0.5 * n * 0.05 ** 2 * math.sin(2 * math.pi / n)
    alone = generate_mesh("polygon", vertices=V.tolist(), h=0.02)
    assert alone["dim"] == 2 and alone["area"] == pytest.approx(poly_area, rel=0.03)     # 16 chords of 0.02
    assert alone["n_elements"] < 100                     # a 512-gon meshed edge by edge would need > 512 triangles
    fine = generate_mesh("polygon", vertices=V.tolist(), h=0.002)
    assert fine["area"] == pytest.approx(poly_area, rel=5e-4)
    flow = generate_mesh("polygon", vertices=V.tolist(), enclosure=[[0, 0], [2.2, 0.41]], h=0.05,
                         refine=[{"center": [0.2, 0.2], "radius": 0.1, "h": 0.0125}], out_dir=str(tmp_path / "c"))
    assert flow["area"] == pytest.approx(2.2 * 0.41 - poly_area, rel=2e-4)          # chords of 0.0125 on the body
    disk = generate_mesh("disk", [0.25], origin=[0.5, 0.5], enclosure=[[0, 0], [1, 1]], h=0.05)
    assert disk["area"] == pytest.approx(1 - math.pi * 0.25 ** 2, rel=2e-3)
    Vs, F = icosphere(0.5, 3)
    stl = write_stl(str(tmp_path / "s.stl"), Vs, F)
    box = generate_mesh(stl, h=1.0, enclosure=[[-2, -2, -2], [3, 2, 2]],
                        refine=[{"center": [0, 0, 0], "radius": 1.0, "h": 0.25}])
    tri = Vs[F]
    v_sphere = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    assert box["dim"] == 3 and box["volume"] == pytest.approx(80 - v_sphere, rel=1e-3)
    with pytest.raises(ValueError, match="enclosure"):
        generate_mesh("disk", [0.25], enclosure=[[0, 0], [-1, 1]], h=0.1)
    with pytest.raises(ValueError, match="polygon"):
        generate_mesh("polygon", vertices=[[0, 0], [1, 0]], h=0.1)
