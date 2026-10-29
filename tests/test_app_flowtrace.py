"""FlowTrace app (apps/flowtrace): a known shift uploaded as PNG frames comes back as the velocity field, the parameter
guide, a video upload when ffmpeg is installed, and the refusals."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
Image = pytest.importorskip("PIL.Image")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["FLOWTRACE_DATA_DIR"] = str(tmp_path_factory.mktemp("flowtrace"))
    sys.path.insert(0, str(ROOT / "apps" / "flowtrace"))
    from flowtrace.api import app
    return TestClient(app)


def _frames(dx=3, dy=-2, h=128, w=160, n=3):
    rng = np.random.default_rng(0)
    base = (rng.random((h + 40, w + 40)) > 0.97).astype(float)
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, 1.2)
    base = (255 * base / base.max()).astype(np.uint8)
    out = []
    for k in range(n):
        crop = base[20 - k * dy:20 - k * dy + h, 20 - k * dx:20 - k * dx + w]
        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format="PNG")
        out.append((f"frame_{k:03d}.png", buf.getvalue()))
    return out


def test_known_shift_from_png_frames(client):
    assert client.get("/health").json()["status"] == "ok"
    files = [("frames", (n, b, "image/png")) for n, b in _frames()]
    r = client.post("/v1/analyze_sequence", files=files, data={"window_size": "32", "search_margin": "8",
                                                              "output": "json"})
    assert r.status_code == 200, r.text
    body = r.json()
    with np.load(io.BytesIO(base64.b64decode(body["npz_base64"]))) as z:
        assert abs(float(z["u"].mean()) - 3.0) < 0.3 and abs(float(z["v"].mean()) + 2.0) < 0.3
    assert body["summary"]["frame_pairs"] == 2 and body["summary"]["vectors"] > 0
    r = client.post("/v1/analyze_sequence", files=files)
    assert r.status_code == 200 and json.loads(r.headers["x-summary"])["frame_pairs"] == 2


def test_guide_and_refusals(client):
    g = client.get("/v1/parameters_guide", params={"approx_displacement_px": 5, "image_width_px": 1920,
                                                   "image_height_px": 1080}).json()
    assert g["window_size"] == 32
    assert client.post("/v1/analyze_sequence").status_code == 422
    one = [("frames", (n, b, "image/png")) for n, b in _frames(n=1)]
    assert client.post("/v1/analyze_sequence", files=one).status_code == 422
    big = [("frames", (n, b, "image/png")) for n, b in _frames()]
    assert client.post("/v1/analyze_sequence", files=big, data={"window_size": "256"}).status_code == 422


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_upload(client, tmp_path):
    for n, b in _frames(n=4):
        (tmp_path / n).write_bytes(b)
    vid = tmp_path / "flow.avi"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "5", "-i", str(tmp_path / "frame_%03d.png"),
                    "-c:v", "ffv1", "-pix_fmt", "gray", str(vid)], check=True)
    r = client.post("/v1/analyze_sequence", files={"video": ("flow.avi", vid.read_bytes(), "video/x-msvideo")},
                    data={"window_size": "32", "search_margin": "8", "output": "json"})
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["frame_pairs"] == 3
