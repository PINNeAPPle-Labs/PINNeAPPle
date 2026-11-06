"""The 3-D car wind tunnel wrapper: the lattice operating point it enforces, the contract a flow reads (checks,
force history, mean-field slices) and its own limits said in the value."""
import pytest

from pinneapple_flow import wrappers as W


def test_the_mask_is_the_ahmed_body():
    mask = W._body_mask((116, 55, 31), dx=W.AHMED["length"] / 32, x0=-1.2, slant_deg=25.0, clearance=0.15)
    assert int(mask.sum()) > 1000                                  # a body, not a wall or an empty box
    solid_x = mask.any(dim=(1, 2)).nonzero().flatten()
    assert len(solid_x) == pytest.approx(32, abs=2)                # 32 cells along its length
    assert not mask[:, :, 0].any() and not mask[:, :, -1].any()    # clear of the road and of the tunnel roof


def test_the_tunnel_reports_checks_a_flow_can_read():
    r = W.car_wind_tunnel_3d(cells_per_length=16, steps=200, samples=4)
    assert r["kind"] == "cfd3d" and r["body"] == "ahmed"
    codes = {c["code"]: c for c in r["checks"]}
    assert set(codes) == {"simulate.converged", "simulate.balance"}
    assert all(c["status"] in ("pass", "fail") and "value" in c and "detail" in c for c in codes.values())
    assert r["tau"] >= W.TAU_MIN - 1e-9                            # never at the BGK stability limit
    assert len(r["cd_history"]) >= 4 and r["cd"] == pytest.approx(r["cd_history"][-1], abs=1e-3)
    nx, ny, nz = r["shape"]
    assert np_shape(r["fields"]["u (y = mid)"]) == (nx, nz)
    assert np_shape(r["fields"]["u (z = mid)"]) == (nx, ny)
    assert r["limitations"] and r["reference"]["cd"] == pytest.approx(W.AHMED["reference_cd_25deg"])


def test_a_reynolds_the_grid_cannot_hold_is_clamped_and_said_so():
    r = W.car_wind_tunnel_3d(cells_per_length=16, steps=100, samples=2, Re=1.0e6)
    assert r["re_clamped"] and r["Re"] == pytest.approx(r["re_max"], rel=1e-3) and r["Re"] < 1.0e6
    assert r["re_requested"] == 1.0e6


def np_shape(field):
    return len(field), len(field[0])


def test_the_geometry_comes_from_the_request_when_there_is_no_file():
    """The dimensions in the request build the body: a 4.5 m car is not the 1.04 m reference scaled, it is its own
    case (the tunnel follows it), and the value says which geometry ran."""
    small = W.car_wind_tunnel_3d(cells_per_length=16, steps=100, samples=2)
    big = W.car_wind_tunnel_3d(length_m=4.5, width_m=1.9, height_m=1.1, slant_deg=18.0, cells_per_length=20,
                               steps=100, samples=2)
    assert big["dims_m"] == {"length": 4.5, "width": 1.9, "height": 1.1}
    assert big["geometry"].startswith("parametric 4.5 x 1.9 x 1.1 m, slant 18")
    assert big["shape"] != small["shape"]                       # the tunnel scales with the body
    assert big["frontal_area_cells"] > small["frontal_area_cells"]


def test_a_surface_file_replaces_the_parametric_body(tmp_path):
    """An uploaded (or exported) closed surface is voxelized in place of the parametric body — the same body by both
    paths must give about the same drag, which is what makes the file path trustworthy."""
    trimesh = pytest.importorskip("trimesh")
    from pinneapple_design.geometry.bodies import ahmed_body
    path = tmp_path / "body.stl"
    v, f = ahmed_body(25.0)
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(str(path))
    parametric = W.car_wind_tunnel_3d(cells_per_length=16, steps=400, samples=8)
    surface = W.car_wind_tunnel_3d(cells_per_length=16, steps=400, samples=8, geometry=str(path))
    assert surface["body"] == "body.stl" and surface["geometry"].startswith("surface body.stl")
    # once the force settles, the same body through two rasterizations gives about the same drag
    assert surface["cd_mean"] == pytest.approx(parametric["cd_mean"], rel=0.4)
