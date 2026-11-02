"""Core coverage (#1129, toward 1.0) for geometry I/O and meshing: SDF meshing in 2-D and 3-D, physics point
clouds, meshio and STL round trips, the geometry asset registry, STEP through gmsh and through the exact B-rep reader.

References:
* a disc and a sphere of radius 0.5: meshed area pi/4 and volume pi/6, contours and boundary points on the circle;
* a mesh written and read back is the same mesh;
* a STEP file meshed by gmsh and read by the exact B-rep kernel gives the same surface area (two independent
  kernels).
The optional backends (gmsh, meshio, scikit-image, trimesh) are installed by the CI step that runs this file.
"""
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pinneapple_design.geometry.brep import diagnostics as BD
from pinneapple_design.geometry.brep import step as BS
from pinneapple_design.geometry.core import registry as RG
from pinneapple_design.geometry.core.mesh import MeshData
from pinneapple_design.geometry.io import cadquery_bridge as CQ
from pinneapple_design.geometry.io import meshio_bridge as MB
from pinneapple_design.geometry.io import sdf_meshing as SM
from pinneapple_design.geometry.io import step as IS
from pinneapple_design.geometry.io import stl as ST
from pinneapple_design.geometry.mesh import mesher_3d as M3

ROOT = Path(__file__).resolve().parents[1]
NOZZLE = ROOT / "examples" / "handoff_cases" / "free_surface_trough" / "geometry" / "nozzle.step"


def _circle(p):
    return np.linalg.norm(p, axis=1) - 0.5


def _tri_area(V, F):
    V, F = np.asarray(V, float), np.asarray(F)
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()


def _tet():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return MeshData(vertices=V, faces=F, normals=None)


# ------------------------------------------------------------------------------------------------ 2-D SDF
def test_sdf_contours_boundary_points_and_triangulation():
    pytest.importorskip("skimage")
    g = SM.SDFGrid2D(resolution=128)
    assert g.bounds_min == (-1.0, -1.0)
    c = SM.marching_squares_contours(_circle, grid=g)
    assert len(c) == 1 and np.abs(np.linalg.norm(c[0], axis=1) - 0.5).max() < 2.0 / 127   # within a grid cell
    b = SM.sample_boundary_points_sdf2d(_circle, n=100, rng=np.random.default_rng(0), band=0.01)
    assert b.shape == (100, 2) and np.abs(np.linalg.norm(b, axis=1) - 0.5).max() <= 0.01      # inside the band
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")
    m = SM.sdf2d_to_tri_mesh(_circle, grid=g, mesh_size=0.05)
    assert _tri_area(m.vertices, m.faces) == pytest.approx(np.pi / 4, rel=2e-3)            # polygonal disc


# ------------------------------------------------------------------------------------------------ 3-D
def test_sphere_from_an_sdf_and_its_physics_points():
    pytest.importorskip("skimage")
    pytest.importorskip("gmsh")
    sph = lambda p: np.linalg.norm(p, axis=1) - 0.5                                         # noqa: E731
    m = M3.mesh_from_sdf_3d(sph, (-0.6,) * 3, (0.6,) * 3, resolution=40, volume=True)
    assert isinstance(m, M3.Mesh3D)
    pc = M3.sample_physics_points(m, n_interior=400, n_boundary=300, n_near_wall=50, seed=0)
    assert isinstance(pc, M3.PhysicsPointCloud3D) and pc.n_interior == 400 and pc.n_boundary == 300
    assert np.abs(np.linalg.norm(pc.boundary, axis=1) - 0.5).max() < 0.02                   # on the sphere
    assert np.linalg.norm(pc.interior, axis=1).max() < 0.5 + 1e-6                           # inside it
    assert pc.all_points().shape[1] == 3 and set(pc.to_dict()) >= {"interior", "boundary"}
    surf = M3.mesh_from_sdf_3d(sph, (-0.6,) * 3, (0.6,) * 3, resolution=40, volume=False)
    assert _tri_area(surf.vertices, surf.surface_faces) == pytest.approx(np.pi, rel=0.02)   # 4 pi r^2
    assert m.tet_volumes().sum() == pytest.approx(np.pi / 6, rel=0.03)                      # 4/3 pi r^3
    vol = M3.mesh_from_surface(MeshData(vertices=surf.vertices, faces=surf.surface_faces, normals=None),
                               mesh_size=0.1, optimize=False)
    assert vol is not None and vol.tet_volumes().sum() == pytest.approx(np.pi / 6, rel=0.03)


def test_surface_from_a_point_cloud():
    """With alpha larger than every Delaunay tetrahedron, the alpha shape of a filled ball is its convex hull."""
    pytest.importorskip("trimesh")
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(400, 3))
    pts *= rng.random((400, 1)) ** (1 / 3) / np.linalg.norm(pts, axis=1, keepdims=True)  # uniform in the ball
    s = M3.surface_from_pointcloud(pts, method="alpha_shape", alpha=1e12)
    assert s is not None and np.allclose(np.asarray(s.vertices), pts, atol=1e-6)            # the input points
    assert _tri_area(s.vertices, s.faces) == pytest.approx(ConvexHull(pts).area, rel=1e-6)   # = the convex hull


# ------------------------------------------------------------------------------------------------ files
def test_meshio_round_trip_and_upd(tmp_path):
    pytest.importorskip("meshio")
    m = _tet()
    MB.save_meshio(m, tmp_path / "t.vtu", point_data={"T": np.arange(4.0)})
    back = MB.load_meshio(tmp_path / "t.vtu")
    assert np.allclose(back.vertices, m.vertices) and np.array_equal(np.asarray(back.faces), m.faces)
    upd = MB.meshio_to_upd(str(tmp_path / "t.vtu"))
    assert upd.state["vertices"].shape == (4, 3) and upd.state["faces"].shape == (4, 3)
    assert upd.state["T"].tolist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(FileNotFoundError):
        MB.load_meshio(tmp_path / "missing.vtu")


def test_stl_loader_and_the_geometry_registry(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("meshio")
    MB.save_meshio(_tet(), tmp_path / "t.stl", file_format="stl")
    m = ST.load_stl(tmp_path / "t.stl")
    assert _tri_area(m.vertices, m.faces) == pytest.approx(1.5 + np.sqrt(3) / 2)            # exact tetrahedron
    a1 = RG.load_geometry_asset(str(tmp_path / "t.stl"))
    a2 = RG.build_geometry_asset({"kind": "file", "path": str(tmp_path / "t.stl")})
    assert RG.load_geometry_asset(a1) is a1
    assert _tri_area(a1.mesh.vertices, a1.mesh.faces) == pytest.approx(_tri_area(a2.mesh.vertices, a2.mesh.faces))


def test_step_through_gmsh_and_through_the_exact_kernel_agree():
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")
    m = IS.step_to_mesh(NOZZLE, cfg=IS.StepImportConfig(mesh_size=5.0, curvature_refine=False, optimize=False))
    gmsh_area_m2 = _tri_area(m.vertices, m.faces) * 1e-6                                     # gmsh reads mm
    if not BS.available_backends():
        pytest.skip("no exact B-rep backend")
    body = BS.read_step(NOZZLE)[0]
    assert all(isinstance(f, BS.CadFace) for f in body.faces)
    exact = sum(f.area_m2 for f in body.faces)
    assert gmsh_area_m2 == pytest.approx(exact, rel=0.02)                                    # two kernels
    tess = importlib.import_module("pinneapple_design.geometry.brep.tessellate").tessellate(body, quality="draft")
    vf = BD.vertex_face(tess)
    assert len(vf) == len(tess.vertices) and vf.min() >= 0 and vf.max() < len(body.faces)
    with pytest.raises(FileNotFoundError):
        IS.step_to_mesh(NOZZLE.with_name("missing.step"))


def test_cadquery_conversion_with_a_stub_kernel(monkeypatch):
    """Without CadQuery installed (it is not in CI), the conversion is checked with a stub module: what is tested is
    this function's own work (tessellation -> PhysicalSample), not the CAD kernel."""
    monkeypatch.setitem(sys.modules, "cadquery", types.ModuleType("cadquery"))

    class Solid:
        def tessellate(self, lin, ang):
            return [tuple(v) for v in _tet().vertices], [tuple(f) for f in _tet().faces]
    s = CQ.cadquery_solid_to_upd(Solid(), units="mm", meta={"part": "tet"})
    assert s.state["vertices"].shape == (4, 3) and s.state["faces"].shape == (4, 3)
    assert s.provenance["part"] == "tet" and s.schema["units"] == {"vertices": "mm"}
