"""Geometry library: every entry is documented, closed, free of self-intersections, the size it says it is, and
regenerates the committed files; the loader, the exporters and the renderer work."""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import pytest

from pinneapple_design import geometry_library as gl
from pinneapple_design.geometry_library import exporters, meshops, store

NAMES = gl.list()
CATEGORIES = {"aircraft", "rotorcraft", "ground_vehicle", "rail", "marine", "space", "energy", "thermal",
              "turbomachinery", "civil", "biomedical"}


@pytest.fixture(scope="module")
def light():
    return {n: gl.build(n) for n in NAMES}


def test_the_library_has_the_first_entries_and_at_least_eight_new_ones():
    for n in ("airliner_a320", "f1_car", "road_car", "viper_rover"):
        assert n in NAMES
    new = [n for n in NAMES if not gl.get_spec(n).legacy]
    assert len(new) >= 8, new


@pytest.mark.parametrize("name", NAMES)
def test_info_is_complete(name):
    i = gl.info(name)
    assert i["units"] == "m" and i["frame"] and i["description"] and i["title"]
    assert i["category"] in CATEGORIES
    assert set(i["dimensions"]) == {"length_x", "width_y", "height_z"}
    assert i["sources"] and all(s["ref"] for s in i["sources"])
    assert len(i["physics"]) >= 2
    for p in i["parameters"]:
        assert p["doc"]
        if "range" in p:
            lo, hi = p["range"]
            assert lo <= p["default"] <= hi, p
        if "choices" in p:
            assert p["default"] in p["choices"]


@pytest.mark.parametrize("name", NAMES)
def test_light_build_is_closed_outward_and_within_budget(name, light):
    g = light[name]
    spec = gl.get_spec(name)
    assert 0 < g.n_triangles() <= spec.max_triangles["light"]
    rep = g.check()
    for body, q in rep["bodies"].items():
        assert q["finite"], body
        assert q["watertight"], (body, q)
        assert q["volume"] > 0, (body, q["volume"])
        assert q["degenerate_faces"] == 0, (body, q["degenerate_faces"])


@pytest.mark.parametrize("name", NAMES)
def test_bounding_box_matches_the_documented_dimensions(name, light):
    spec = gl.get_spec(name)
    d = light[name].dimensions()
    for k, (value, tol) in spec.dimensions.items():
        assert abs(d[k] - value) <= tol * value, (k, d[k], value, tol)


@pytest.mark.parametrize("name", NAMES)
def test_volume_and_area_are_sane(name, light):
    g = light[name]
    d = g.dimensions()
    box = d["length_x"] * d["width_y"] * d["height_z"]
    vol, area = g.volume(), g.area()
    assert 0 < vol < 1.5 * box                      # bodies may overlap (wing in fuselage) but not wildly
    assert area > 0
    # isoperimetric sanity: no body encloses more than a sphere of the same area
    for b in g.bodies:
        v, a = meshops.volume(b.vertices, b.faces), meshops.area(b.vertices, b.faces)
        assert v <= a ** 1.5 / (6 * math.sqrt(math.pi)) * 1.0001, b.name


@pytest.mark.parametrize("name", NAMES)
def test_no_self_intersections(name, light):
    g = light[name]
    exempt = gl.get_spec(name).self_intersection_exempt
    bad = {}
    for b in g.bodies:
        if b.name in exempt:
            continue
        k = meshops.self_intersections(b.vertices, b.faces)
        if k:
            bad[b.name] = k
    assert not bad, bad


@pytest.mark.parametrize("name", NAMES)
def test_regeneration_equals_the_committed_files(name, light):
    g = light[name]
    m = store.read_manifest()["geometries"][name]["files"]
    for fname in (store.file_name(name, "light", "stl"), store.file_name(name, "light", "glb")):
        path = store.REPO_DATA / name / fname
        assert path.exists(), f"{path} is not committed (python scripts/geometry_library.py light)"
        assert store.sha256(path) == m[fname]["sha256"], f"{fname}: committed file differs from the manifest"
    stored = exporters.read_stl(str(store.REPO_DATA / name / store.file_name(name, "light", "stl")))
    built = exporters.to_stl(g)
    if hashlib.sha256(built).hexdigest() == m[store.file_name(name, "light", "stl")]["sha256"]:
        return                                         # bit-identical
    V, F = g.mesh()
    tri = V[F].astype(np.float32)
    assert tri.shape == stored.shape
    scale = float(np.ptp(V, axis=0).max())
    assert np.abs(tri - stored).max() <= 1e-6 * scale + 1e-7


@pytest.mark.parametrize("name", NAMES)
def test_fine_level_of_detail(name):
    g = gl.build(name, lod="fine")
    spec = gl.get_spec(name)
    assert g.n_triangles() <= spec.max_triangles["fine"]
    assert g.check()["watertight"]
    d = g.dimensions()
    for k, (value, tol) in spec.dimensions.items():
        assert abs(d[k] - value) <= tol * value, (k, d[k], value)


def test_parameters_change_the_geometry_and_are_range_checked():
    a = gl.build("wind_turbine_nrel5mw")
    b = gl.build("wind_turbine_nrel5mw", pitch_deg=20.0, rotor_azimuth_deg=60.0)
    assert not np.allclose(a.mesh()[0][:100], b.mesh()[0][:100])
    with pytest.raises(ValueError):
        gl.build("wind_turbine_nrel5mw", hub_height=500.0)
    with pytest.raises(TypeError):
        gl.build("wind_turbine_nrel5mw", not_a_parameter=1)
    with pytest.raises(KeyError):
        gl.build("no_such_geometry")


def test_every_format_exports(tmp_path, light):
    g = light["road_car"]
    for fmt in ("stl", "glb", "obj", "usda", "step"):
        p = g.export(str(tmp_path / f"car.{fmt}"))
        assert os.path.getsize(p) > 1000
    assert (tmp_path / "road_car.mtl").exists()
    assert open(tmp_path / "car.glb", "rb").read(4) == b"glTF"
    assert open(tmp_path / "car.step").read(13) == "ISO-10303-21;"
    tri = exporters.read_stl(str(tmp_path / "car.stl"))
    assert len(tri) == g.n_triangles()


def test_step_reads_back_as_solids(tmp_path, light):
    pytest.importorskip("cadquery")
    g = light["wind_turbine_nrel5mw"]
    p = g.export(str(tmp_path / "wt.step"))
    r = exporters.step_check(p)
    assert r["solids"] == len(g.bodies)
    assert r["volume"] == pytest.approx(g.volume(), rel=1e-4)


def test_solid_union_is_one_watertight_surface(light):
    pytest.importorskip("manifold3d")
    V, F = light["wind_turbine_nrel5mw"].solid()
    q = meshops.quality(V, F)
    assert q["watertight"] and q["volume"] > 0
    assert q["volume"] <= light["wind_turbine_nrel5mw"].volume() + 1e-6


def test_load_prefers_the_verified_repository_copy_and_rebuilds_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_GEOMETRY_CACHE", str(tmp_path))
    monkeypatch.setenv("PINNEAPPLE_OFFLINE", "1")
    p = gl.load("road_car", format="stl")
    assert p == store.REPO_DATA / "road_car" / "road_car.light.stl"
    with pytest.warns(UserWarning):
        q = gl.load("road_car", format="obj", lod="fine")   # Hub only: offline -> built from the code
    assert q.parent.parent == tmp_path and q.exists()
    assert gl.load("road_car", format="usd").suffix == ".usda"
    with pytest.raises(FileNotFoundError):
        gl.load("road_car", format="step", lod="fine", allow_build=False)


def test_manifest_lists_every_entry_with_hub_files():
    m = store.read_manifest()
    assert m["hub_repo"] == "PINNeAPPle-Labs/geometry-library"
    for n in NAMES:
        files = m["geometries"][n]["files"]
        assert store.file_name(n, "fine", "stl") in files and store.file_name(n, "light", "step") in files
        for e in files.values():
            assert len(e["sha256"]) == 64 and e["bytes"] > 0


@pytest.mark.parametrize("name", NAMES)
def test_committed_files_stay_light(name):
    d = store.REPO_DATA / name
    total = sum(f.stat().st_size for f in d.iterdir())
    assert total < 3.0e6, f"{name}: {total / 1e6:.2f} MB committed"
    assert store.thumbnail_path(name).exists()


def test_render_draws_the_geometry(light):
    im = light["road_car"].render(None, size=(160, 110), ssaa=1)
    a = np.asarray(im).astype(float)
    assert a.shape == (110, 160, 3) and a.std() > 10


def test_top_level_shortcut():
    import pinneapple as pp
    assert pp.geometries.list() == NAMES
    assert pp.geometries.info("viper_rover")["category"] == "space"


def test_self_intersection_check_finds_real_crossings_and_ignores_touching():
    from pinneapple_design.geometry_library import kit
    assert meshops.self_intersections(*kit.merge([kit.box((1, 1, 1)), kit.box((1, 1, 1), (0.5, 0.5, 0.5))])) > 0
    thin = kit.box((1, 0.01, 1))
    assert meshops.self_intersections(*kit.merge([thin, kit.tf(thin, R=kit.rot((0, 0, 1), 5))])) > 0
    # far from the origin (a tower top, a ship) and with flat caps: no false positives
    assert meshops.self_intersections(*kit.cylinder((0, 0, 90), (0, 0, 95), 3, 64)) == 0
    V, F = kit.wing([{"le": (0, 0, 0), "chord": 1, "t": 0.04, "m": 0.08}, {"le": (0, 1, 0), "chord": 0.5, "t": 0.04,
                                                                           "m": 0.08}], n=40, frame=((1, 0, 0), (0, 0, 1)))
    assert meshops.self_intersections(np.round(V / 1e-7) * 1e-7, F) == 0


def test_build_is_on_the_reproducibility_grid():
    from pinneapple_design.geometry_library.core import QUANTUM
    for b in gl.build("wigley_hull").bodies:
        assert np.allclose(b.vertices / QUANTUM, np.round(b.vertices / QUANTUM), atol=1e-6)


def test_no_negative_zero_coordinates(light):
    # -0 and +0 are equal but written differently by the STL / OBJ / USD writers, and which one a coordinate that rounds
    # to zero gets depends on the SIMD path NumPy takes on the CPU (the publishing job's SHA-256 check failed on it)
    for n, g in light.items():
        for b in g.bodies:
            assert not np.any((b.vertices == 0) & np.signbit(b.vertices)), (n, b.name)


_SIMD_BUILD = """
import hashlib, sys
from pinneapple_design import geometry_library as gl
from pinneapple_design.geometry_library import exporters
g = gl.build(sys.argv[1], lod="light")
for fmt in ("stl", "obj", "usda", "glb"):
    print(fmt, hashlib.sha256(exporters.WRITERS[fmt](g)).hexdigest())
"""


@pytest.mark.parametrize("disabled", ["X86_V4", "X86_V3 X86_V4"])
def test_files_do_not_depend_on_the_cpu_simd_level(disabled, light):
    """Every machine writes the bytes of the manifest: the same build with NumPy's AVX-512 / AVX2 kernels turned off
    (NPY_DISABLE_CPU_FEATURES, as on a CI runner without them) writes identical files."""
    import platform
    import subprocess
    import sys
    if platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("x86 SIMD levels")
    name = "helicopter_h145"                           # had coordinates whose sign of zero followed the SIMD level
    env = {**os.environ, "NPY_DISABLE_CPU_FEATURES": disabled, "PYTHONWARNINGS": "ignore"}
    r = subprocess.run([sys.executable, "-c", _SIMD_BUILD, name], env=env, capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    other = dict(line.split() for line in r.stdout.splitlines())
    here = {fmt: hashlib.sha256(exporters.WRITERS[fmt](light[name])).hexdigest() for fmt in other}
    assert other == here
