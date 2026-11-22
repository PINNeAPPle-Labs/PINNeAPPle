"""Repair operations on surface meshes (#1494): every case measures the same ``surface_checks`` metrics before and after,
so a test states exactly what the operation fixed and what it left alone.

The meshes are synthetic -- a unit cube of twelve triangles is the smallest closed surface that shows a hole, a
duplicate face, a flipped normal and split vertices separately.
"""
import copy
import json

import numpy as np
import pytest

from pinneapple_data.cae import repair as RP
from pinneapple_data.cae.model import CellBlock, Mesh
from pinneapple_data.cae.quality import surface_checks

# the unit cube's eight corners in bit order, and its twelve outward triangles
CUBE_POINTS = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
CUBE_TRIS = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def cube(points=None, tris=None):
    pts = CUBE_POINTS if points is None else points
    conn = CUBE_TRIS if tris is None else tris
    return Mesh(points=np.array(pts, float), blocks=[CellBlock("triangle", np.array(conn, int))])


def cylinder(n=12, h=1.0):
    """An open tube: two rims of ``n`` edges, outward-facing triangles, no caps."""
    ang = 2 * np.pi * np.arange(n) / n
    pts = np.vstack([np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1),
                     np.stack([np.cos(ang), np.sin(ang), np.full(n, h)], 1)])
    j, k = (np.arange(n) + 1) % n, (np.arange(n) + 1) % n + n
    tris = np.vstack([np.stack([np.arange(n), j, k], 1), np.stack([np.arange(n), k, np.arange(n) + n], 1)])
    return Mesh(points=pts, blocks=[CellBlock("triangle", tris)])


# ------------------------------------------------------------------------------------------ a clean mesh
def test_clean_cube_is_returned_unchanged_and_says_so():
    m = cube()
    before = surface_checks(m)
    new, plan = RP.repair_surface(m)
    after = surface_checks(new)

    assert before["watertight"] and before["open_edges"] == 0
    assert plan.no_op, [s.operation for s in plan if s.changed]
    assert [s.operation for s in plan] == ["merge_vertices", "drop_degenerate_faces", "drop_duplicate_faces",
                                           "orient_consistently", "close_holes"]
    for key in ("faces", "open_edges", "nonmanifold_edges", "inconsistent_orientation_edges", "degenerate_faces",
                "duplicate_faces", "shells", "watertight"):
        assert after[key] == before[key], key
    assert after["enclosed_volume"] == pytest.approx(1.0)
    assert plan.summary["changed"] == 0 and plan.summary["fixed"] == [] and plan.summary["regressed"] == []


def test_every_result_measures_before_and_after_with_the_report_metrics():
    _, plan = RP.repair_surface(cylinder())
    for step in plan:
        assert set(step.before) == set(step.after) == set(RP._SCALAR_METRICS)
        assert step.guarantees and step.cannot_fix, step.operation
        json.dumps(step.as_dict())            # the result is a report, so it has to serialise
        assert step.kind == "surface"


# ------------------------------------------------------------------------------------------ holes
def test_cube_with_one_face_removed_is_closed_and_watertight():
    holed = cube(tris=CUBE_TRIS[2:])          # the two triangles of the z = 0 face are gone
    before = surface_checks(holed)
    assert before["open_edges"] == 4 and before["watertight"] is False    # the square rim of the missing face

    new, res = RP.close_holes(holed)
    after = surface_checks(new)
    assert after["open_edges"] == 0
    assert after["watertight"] is True
    assert after["nonmanifold_edges"] == 0
    assert after["enclosed_volume"] == pytest.approx(1.0)
    assert after["area"] == pytest.approx(6.0)
    assert res.changed == 2                    # the square hole is two triangles again
    assert res.regressed == []
    assert "no unique filling" in " ".join(res.cannot_fix)


def test_repair_surface_closes_the_hole_end_to_end():
    holed = cube(tris=CUBE_TRIS[2:])
    new, plan = RP.repair_surface(holed)
    after = surface_checks(new)
    assert after["watertight"] is True and after["open_edges"] == 0
    assert after["enclosed_volume"] == pytest.approx(1.0)
    assert plan.summary["before"]["open_edges"] == 4
    assert plan.summary["after"]["open_edges"] == 0
    assert "open_edges" in plan.summary["fixed"] and plan.summary["regressed"] == []
    assert not plan.no_op


def test_long_holes_are_fanned_from_a_new_centre_point():
    tube = cylinder(n=12)
    assert surface_checks(tube)["open_edges"] == 24     # two rims, twelve edges each
    new, res = RP.close_holes(tube, max_edges=100)
    after = surface_checks(new)
    assert after["open_edges"] == 0 and after["watertight"] is True
    assert new.n_points == tube.n_points + 2            # one centre point per rim (both loops are longer than 8)
    assert res.changed == 24                            # twelve triangles per rim
    assert after["enclosed_volume"] == pytest.approx(3.0, rel=1e-6)   # twelve-gon of circumradius 1, height 1
    assert res.regressed == []


def test_a_hole_longer_than_max_edges_is_left_open_and_counted():
    tube = cylinder(n=12)
    new, res = RP.close_holes(tube, max_edges=8)
    after = surface_checks(new)
    assert after["open_edges"] == 24 and after["watertight"] is False
    assert res.changed == 0
    assert "left open" in " ".join(res.cannot_fix)


# ------------------------------------------------------------------------------------------ orientation
def test_one_flipped_face_is_reoriented():
    tris = CUBE_TRIS.copy()
    tris[0] = tris[0][::-1]
    m = cube(tris=tris)
    before = surface_checks(m)
    assert before["inconsistent_orientation_edges"] == 3       # the three edges of that triangle
    assert before["enclosed_volume"] == pytest.approx(1.0)     # one triangle does not flip the volume

    new, res = RP.orient_consistently(m)
    after = surface_checks(new)
    assert after["inconsistent_orientation_edges"] == 0
    assert after["enclosed_volume"] == pytest.approx(1.0)
    assert res.changed == 1
    assert res.regressed == []
    assert json.dumps(res.as_dict())


def test_a_cube_turned_inside_out_is_turned_back():
    m = cube(tris=CUBE_TRIS[:, ::-1])
    assert surface_checks(m)["enclosed_volume"] == pytest.approx(-1.0)
    new, res = RP.orient_consistently(m)
    assert surface_checks(new)["enclosed_volume"] == pytest.approx(1.0)
    assert res.changed == 12                    # the whole shell, not one triangle at a time
    assert res.regressed == []


def test_separate_shells_are_oriented_independently():
    pts = np.vstack([CUBE_POINTS, CUBE_POINTS + [3.0, 0.0, 0.0]])
    good = CellBlock("triangle", CUBE_TRIS)
    bad = CellBlock("triangle", CUBE_TRIS[:, ::-1] + 8)
    m = Mesh(points=pts, blocks=[good, bad])
    before = surface_checks(m)
    assert before["shells"] == 2
    new, res = RP.orient_consistently(m)
    after = surface_checks(new)
    assert after["inconsistent_orientation_edges"] == 0
    assert res.changed == 12                    # only the inverted shell moved
    vols = _shell_volumes(new)
    assert all(v > 0 for v in vols), vols


def _shell_volumes(mesh):
    """The enclosed volume of each shell, measured on its own."""
    shell = surface_checks(mesh)["shell_of_face"]
    faces = np.concatenate([b.corners for b in mesh.blocks if b.dim == 2])
    out = []
    for s in np.unique(shell):
        p = mesh.points[faces[shell == s]]
        out.append(float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0))
    return out


# ------------------------------------------------------------------------------------------ degenerate and duplicate
def test_zero_area_face_and_duplicate_face_are_removed():
    pts = np.vstack([CUBE_POINTS, [(0.0, 0.0, 0.5)]])       # a point on the edge 0-1: triangle (0, 1, 8) is flat
    tris = np.vstack([CUBE_TRIS, CUBE_TRIS[0], [[0, 1, 8]]])
    m = cube(points=pts, tris=tris)
    before = surface_checks(m)
    assert before["degenerate_faces"] == 1
    assert before["duplicate_faces"] == 1

    no_flat, res_flat = RP.drop_degenerate_faces(m)
    assert surface_checks(no_flat)["degenerate_faces"] == 0
    assert res_flat.changed == 1

    no_dup, res_dup = RP.drop_duplicate_faces(m)
    after = surface_checks(no_dup)
    assert after["duplicate_faces"] == 0
    assert after["faces"] == before["faces"] - 1
    assert res_dup.changed == 1
    assert res_dup.regressed == []             # the area that went away was the duplicated face's, not new geometry
    for res in (res_flat, res_dup):
        assert res.regressed == [] or "area" not in res.regressed


def test_the_pipeline_removes_both_and_keeps_the_cube_closed():
    pts = np.vstack([CUBE_POINTS, [(0.0, 0.0, 0.5)]])
    tris = np.vstack([CUBE_TRIS, CUBE_TRIS[0], [[0, 1, 8]]])
    new, plan = RP.repair_surface(cube(points=pts, tris=tris))
    after = surface_checks(new)
    assert after["degenerate_faces"] == 0 and after["duplicate_faces"] == 0
    assert after["watertight"] is True
    assert after["enclosed_volume"] == pytest.approx(1.0)
    assert plan.summary["regressed"] == []


# ------------------------------------------------------------------------------------------ vertices
def test_duplicated_vertices_are_merged_and_the_topology_kept():
    pts = np.vstack([CUBE_POINTS, CUBE_POINTS + 1e-9])
    tris = CUBE_TRIS.copy()
    tris[0] = CUBE_TRIS[0] + 8                 # this one triangle is written with the copied vertices
    m = cube(points=pts, tris=tris)
    before = surface_checks(m)
    assert m.n_points == 16
    assert before["open_edges"] == 6, before           # the three edges of the split triangle stop matching
    assert before["watertight"] is False

    new, res = RP.merge_vertices(m)
    after = surface_checks(new)
    assert new.n_points == 8
    assert after["open_edges"] == 0 and after["watertight"] is True
    assert after["enclosed_volume"] == pytest.approx(1.0)
    assert after["faces"] == before["faces"]
    assert res.changed >= 8
    assert res.regressed == []


def test_merging_that_fuses_two_parts_is_reported_as_a_net_loss():
    """Two coincident cubes are the trap the docstring warns about: welding them makes every edge non-manifold."""
    pts = np.vstack([CUBE_POINTS, CUBE_POINTS])
    both = CellBlock("triangle", np.vstack([CUBE_TRIS, CUBE_TRIS + 8]))
    m = Mesh(points=pts, blocks=[both])
    assert surface_checks(m)["nonmanifold_edges"] == 0
    new, res = RP.merge_vertices(m)
    assert "nonmanifold_edges" in res.regressed
    assert res.net_loss is True
    assert "coincidence is not the same as connection" in " ".join(res.cannot_fix)


# ------------------------------------------------------------------------------------------ refusal and immutability
def test_a_volume_mesh_is_refused_with_the_reason_and_no_new_mesh():
    tetra = Mesh(points=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                 blocks=[CellBlock("tetra", np.array([[0, 1, 2, 3]]))])
    new, res = RP.merge_vertices(tetra)
    assert res.refused is True and res.reason.startswith("a volume mesh is not repaired")
    assert res.changed == 0
    assert res.cannot_fix == [res.reason]
    assert np.array_equal(new.points, tetra.points)
    assert np.array_equal(new.blocks[0].conn, tetra.blocks[0].conn)


def test_the_input_mesh_is_never_mutated():
    pts = np.vstack([CUBE_POINTS, [(0.0, 0.0, 0.5)]])
    tris = np.vstack([CUBE_TRIS[2:], CUBE_TRIS[0], [[0, 1, 8]]])
    m = cube(points=pts, tris=tris)
    m.blocks[0].ids = np.arange(len(tris))
    snapshot = copy.deepcopy(m)

    for fn in (RP.merge_vertices, RP.drop_degenerate_faces, RP.drop_duplicate_faces, RP.orient_consistently,
               RP.close_holes):
        fn(m)
    RP.repair_surface(m)

    assert np.array_equal(m.points, snapshot.points)
    assert np.array_equal(m.blocks[0].conn, snapshot.blocks[0].conn)
    assert np.array_equal(m.blocks[0].ids, snapshot.blocks[0].ids)
    assert len(m.blocks) == len(snapshot.blocks)
