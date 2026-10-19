"""Error maps (pinneapple_analysis.validation.error_map) and the apps' 3-D field views
(pinneapple_tools.visualization.studio.fieldview)."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from pinneapple_analysis.validation.error_map import (
    error_by_region,
    error_hotspots,
    error_map,
    grow_region,
    pointwise_error,
)
from pinneapple_data.cae.model import CellBlock, Mesh
from pinneapple_tools.visualization.studio.fieldview import (
    colour_range,
    displacement_factor,
    field_view,
    length_factor,
    subsample,
)


def _grid(n=21):
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]


def _box(nx=8, ny=2, nz=3, L=2.0, W=0.2, H=0.3) -> Mesh:
    xs, ys, zs = np.linspace(0, L, nx + 1), np.linspace(-W / 2, W / 2, ny + 1), np.linspace(-H / 2, H / 2, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    i, j, k = (g.ravel() for g in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    conn = np.stack([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)], 1)
    return Mesh(points=P, blocks=[CellBlock("hexahedron", conn)])


def _glb_json(b: bytes) -> dict:
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n])


# ------------------------------------------------------------------------------------------------ error maps
def test_pointwise_error_scalar_vector_and_shape_check():
    assert np.allclose(pointwise_error([1.0, -2.0], [0.5, -1.0]), [0.5, 1.0])
    assert np.allclose(pointwise_error([[3.0, 4.0, 0.0]], [[0.0, 0.0, 0.0]]), [5.0])
    with pytest.raises(ValueError, match="map one onto"):
        pointwise_error(np.zeros(3), np.zeros(4))


def test_hotspots_find_two_separate_bumps_in_order():
    P = _grid()
    err = 1.0 * np.exp(-((P[:, 0] - 0.2) ** 2 + (P[:, 1] - 0.2) ** 2) / 0.003) \
        + 0.5 * np.exp(-((P[:, 0] - 0.8) ** 2 + (P[:, 1] - 0.7) ** 2) / 0.003)
    hs = error_hotspots(P, err, n=5, quantile=0.95)
    assert len(hs) == 2
    assert hs[0]["max"] > hs[1]["max"]
    assert np.allclose(hs[0]["at"][:2], [0.2, 0.2], atol=0.051)
    assert np.allclose(hs[1]["centre"][:2], [0.8, 0.7], atol=0.06)
    assert 0.9 < hs[0]["share"] + hs[1]["share"] <= 1.0 + 1e-12
    assert error_hotspots(P, np.zeros(len(P))) == []


def test_error_by_region_and_grow_region():
    P = _grid()
    near = grow_region(P, np.where(P[:, 0] == 0.0)[0], radius=0.11)
    assert set(np.unique(P[near, 0]).round(3)) == {0.0, 0.05, 0.1}
    err = np.where(P[:, 0] <= 0.1, 1.0, 0.01)
    r = {x["name"]: x for x in error_by_region(err, {"edge": near, "rest": np.setdiff1d(np.arange(len(P)), near)})}
    assert r["edge"]["share_error"] > 0.99 and r["rest"]["share_error"] < 0.01
    assert abs(r["edge"]["share_points"] + r["rest"]["share_points"] - 1) < 1e-12
    assert r["edge"]["rms_ratio"] > 2 and r["rest"]["rms_ratio"] < 0.1
    w = np.where(P[:, 0] <= 0.1, 0.0, 1.0)                       # weights move the shares
    zero = error_by_region(err, {"edge": near}, weights=w)[0]
    assert zero["share_points"] == 0 and zero["share_error"] == 0


def test_error_map_numbers_and_summary():
    P = _grid()
    ref = 1.0 + P[:, 0]
    pred = ref + np.where(P[:, 0] <= 0.1, 0.2, 0.0)
    clamp = grow_region(P, np.where(P[:, 0] == 0.0)[0], 0.11)
    tip = grow_region(P, np.where(P[:, 0] == 1.0)[0], 0.11)
    em = error_map(P, pred, ref, regions={"clamp": clamp, "tip": tip}, unit="MPa")
    assert em.rel_l2 == pytest.approx(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
    assert em.max_error == pytest.approx(0.2)
    text = em.summary(reported={"this design": 0.05, "study mean": 0.044})
    assert text[0].startswith(f"Relative L2 error {100 * em.rel_l2:.2f} % (this design: 5.00 %; study mean: 4.40 %)")
    assert "concentrates in the clamp" in text[1]
    assert "in the clamp" in text[-1] and "MPa" in text[-1]
    d = em.to_dict()
    json.dumps(d)                                                # JSON-able: no index sets or arrays left
    assert d["regions"][0]["name"] == "clamp" and "indices" not in d["hotspots"][0]


def test_error_map_uniform_error_names_no_region():
    P = _grid()
    em = error_map(P, np.ones(len(P)) * 1.1, np.ones(len(P)), regions={"a": np.arange(50)})
    assert "No named region dominates" in em.summary()[1]


# ------------------------------------------------------------------------------------------------ field views
def test_colour_range_values_errors_and_flat():
    assert colour_range([1, 3, 2]) == {"lo": 1.0, "hi": 3.0, "max": 3.0, "clipped": False}
    e = np.r_[np.linspace(0, 1, 999), 40.0]
    r = colour_range(e, "error")
    assert r["lo"] == 0 and r["hi"] < 1.1 and r["max"] == 40 and r["clipped"]
    f = colour_range([2.0, 2.0])
    assert f["lo"] < 2 < f["hi"]
    assert colour_range([np.nan])["max"] is None
    assert len(subsample(np.zeros((1000, 3)), 10)) == 10


def test_field_view_ranges_deformation_markers_and_glb():
    m = _box()
    x = m.points[:, 0]
    fem, rom = 10 * (2 - x), 10 * (2 - x) + 0.3 * np.sin(x)
    u = np.c_[np.zeros_like(x), 0.002 * x ** 2, -0.004 * x ** 2]
    cellv = np.arange(m.n_cells, dtype=float)
    fv = field_view(m, [{"name": "fem", "values": fem, "unit": "MPa", "group": "vm", "deform": "fem"},
                        {"name": "rom", "values": rom, "unit": "MPa", "group": "vm", "deform": "fem"},
                        {"name": "err", "values": np.abs(rom - fem), "kind": "error"},
                        {"name": "cells", "values": cellv}],
                    deform={"fem": u}, markers=[{"name": "clamp", "kind": "bc", "points": m.points[x == 0]}])
    meta = fv.meta
    f = {d["name"]: d for d in meta["fields"]}
    assert f["fem"]["lo"] == f["rom"]["lo"] and f["fem"]["hi"] == f["rom"]["hi"]          # one scale per group
    assert f["fem"]["hi"] == pytest.approx(max(fem.max(), rom.max()))
    assert f["err"]["lo"] == 0 and f["err"]["kind"] == "error" and f["cells"]["where"] == "cell"
    d = meta["deform"]["fem"]
    assert d["max"] == pytest.approx(np.linalg.norm(u, axis=1).max())
    assert d["auto_scale"] == pytest.approx(0.1 * 2.0 / d["max"])
    assert meta["markers"][0]["kind"] == "bc" and meta["markers"][0]["n"] == 12
    mk = meta["markers"][0]                                       # snapped to surface vertices on the clamp face
    assert mk["label"] is True
    assert len(mk["vertices"]) == 12 and np.allclose(fv.scene.surfaces[0].vertices[mk["vertices"], 0], 0)
    assert not meta["decimated"] and meta["faces"] == meta["faces_original"]
    js = _glb_json(fv.glb())
    attrs = js["meshes"][0]["primitives"][0]["attributes"]
    assert {"_F0", "_F1", "_F2", "_F3", "_U0X", "_U0Y", "_U0Z"} <= set(attrs)
    srf = fv.scene.surfaces[0]                                    # displacement in the viewer's axes: z_up -> y_up
    V = srf.vertices
    k = int(np.argmax(V[:, 0]))
    assert srf.fields["u0y"][k] == pytest.approx(-0.004 * V[k, 0] ** 2)
    assert srf.fields["u0z"][k] == pytest.approx(-0.002 * V[k, 0] ** 2)
    json.dumps(meta)


def test_field_view_decimates_large_surfaces_and_checks_lengths():
    m = _box(60, 10, 10)
    fv = field_view(m, [{"name": "x", "values": m.points[:, 0]}], max_faces=1000)
    assert fv.meta["decimated"] and fv.meta["faces"] < fv.meta["faces_original"]
    assert fv.meta["faces"] <= 1300
    with pytest.raises(ValueError, match="values for"):
        field_view(m, [{"name": "bad", "values": np.zeros(7)}])


def test_length_units_and_displacement_detection():
    assert length_factor("mm") == pytest.approx(1e-3) and length_factor("m", "mm") == pytest.approx(1e3)
    assert length_factor("MPa") is None and length_factor(None) is None
    assert displacement_factor("displacement", "mm", 3) == pytest.approx(1e-3)
    assert displacement_factor("D", "", 3) == 1.0 and displacement_factor("D", None, 1) is None
    assert displacement_factor("U", "m/s", 3) is None                 # a velocity is not a displacement
    assert displacement_factor("T", "", 3) is None
