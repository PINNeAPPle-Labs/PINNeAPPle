"""Animation-centred dashboard: validation of the spec, the HTML payload (charts, tiles, colour bar, 3D data),
the self-contained page, and the PNG / GIF / MP4 exports."""
import json
import re
import shutil

import numpy as np
import pytest

from pinneapple_tools.visualization.dashboard import (
    Chart,
    ColorBar,
    Dashboard,
    Marker,
    Param,
    Series,
    Tile,
)
from pinneapple_tools.visualization.dashboard.html import payload, scene3d_payload


def _db(n=8, frames=True):
    t = np.linspace(0, 1, n)
    img = [np.full((36, 64, 3), 40 * (k % 6), np.uint8) for k in range(n)]
    return Dashboard(
        "Test", t, img if frames else [None] * n,
        charts=[Chart("Angle", "angle [rad]", "time [s]", [Series("a", np.sin(t)), Series("b", np.cos(t))]),
                Chart("Phase", "y [m]", "x [m]", [Series("curve", t ** 2, x=t)], [Marker("now", t, t ** 2)],
                      kind="phase", slot="left"),
                Chart("Along", "z [mm]", "distance [m]", [Series("a", t)], x=2 * t, slot="bottom")],
        tiles=[Tile("Distance", 2 * t, "m"), Tile("Mass", 450.0, "kg", "{:.0f}")],
        params=[Param("g", 1.62, "m/s²")], colorbar=ColorBar("Stress [kPa]", "inferno", 0.0, 30.0))


def test_spec_checks_lengths_and_assigns_colours_by_entity():
    db = _db()
    a = db.charts[0].series[0]
    b = db.charts[2].series[0]
    assert a.color == b.color                       # same label, same colour on every chart
    assert db.charts[0].series[1].color != a.color
    with pytest.raises(ValueError):
        Dashboard("x", [0, 1], [np.zeros((2, 2, 3))])
    with pytest.raises(ValueError):
        Dashboard("x", [0, 1], [np.zeros((2, 2, 3))] * 2, charts=[Chart("c", "y", "t", [Series("s", [1, 2, 3])])])
    with pytest.raises(ValueError):
        Chart("c", "y", "t", kind="pie")
    assert db.tiles[0].text(3) == f"{2 * db.t[3]:.2f}" and db.tiles[1].text(5) == "450"


def test_payload_has_everything_the_page_draws():
    p = payload(_db())
    assert [c["kind"] for c in p["charts"]] == ["series", "phase", "series"]
    assert len(p["charts"][0]["x"]) == 8 and p["charts"][1]["markers"][0]["label"] == "now"
    assert p["tiles"][0]["text"][-1] == "2.00" and p["tiles"][1]["text"] == ["450"]
    assert p["colorbar"]["stops"][0][0] == 0 and len(p["colorbar"]["stops"][0][1]) == 3
    json.dumps(p)


def test_scene3d_payload_round_trips():
    import base64
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    P = np.zeros((4, 3))
    Q = np.tile([1.0, 0, 0, 0], (4, 1))
    X, Y = np.meshgrid(np.arange(3.0), np.arange(2.0))
    s = scene3d_payload({"b": [(V, [[0, 1, 2]], (0.5, 0.5, 0.5))]}, {"b": (P, Q)},
                        (X, Y, X * 0, np.zeros(X.shape + (3,))))
    v = np.frombuffer(base64.b64decode(s["bodies"][0]["parts"][0]["v"]), np.float32)
    np.testing.assert_allclose(v.reshape(-1, 3), V)
    assert s["terrain"]["nx"] == 3 and s["terrain"]["ny"] == 2


def test_html_is_self_contained(tmp_path):
    path = _db().to_html(str(tmp_path / "d.html"))
    html = open(path).read()
    assert "<script type=\"module\">" in html and "colorbarHTML" in html          # studio-core inlined
    assert not re.search(r"(src|href)=\"https?://", html)                       # nothing fetched from the internet
    assert ("data:video/" in html) or ("data:image/jpeg" in html)
    if shutil.which("ffmpeg"):
        assert "data:video/webm" in html or "data:video/mp4" in html
    db = _db()
    db.scene3d = scene3d_payload({}, {}, None, None)
    html3 = open(db.to_html(str(tmp_path / "d3.html"))).read()
    assert "importmap" in html3 and "three-orbit" in html3


def test_png_and_videos(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio", reason="the GIF/MP4 writer is imageio: pip install -e '.[pyvista]'")
    from PIL import Image
    db = _db()
    p = db.to_png(str(tmp_path / "d.png"), k=3, size=(960, 540))
    im = np.asarray(Image.open(p))
    assert im.shape[:2] == (540, 960)
    assert im[5:20, 5:200].mean() < 60                                             # dark studio background
    out = db.to_video(gif=str(tmp_path / "d.gif"), mp4=str(tmp_path / "d.mp4"), size=(640, 360),
                      gif_size=(320, 180), gif_every=2)
    assert out["frames"] == 8 and (tmp_path / "d.gif").stat().st_size > 0
    if shutil.which("ffmpeg"):
        assert (tmp_path / "d.mp4").stat().st_size > 0
