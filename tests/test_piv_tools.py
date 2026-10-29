"""pinneapple_perception.piv_tools: grid counts match piv_velocity_field, frames decode from images (and a video when
ffmpeg is installed), the parameter guide follows the quarter rule, and the source app's synthetic vortex case
(solid-body rotation, exact particle positions) is recovered."""
from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import pytest

from pinneapple_perception.piv_tools import (
    decode_image_frames,
    decode_video,
    piv_grid_count,
    piv_parameter_guide,
    summarize_piv,
)
from pinneapple_perception.video_piv import piv_velocity_field, piv_velocity_sequence


def _render(xs, ys, h, w, sigma=1.3):
    img = np.zeros((h, w))
    r = int(4 * sigma)
    for x, y in zip(xs, ys, strict=True):
        if not (0 <= x < w and 0 <= y < h):
            continue
        x0, x1, y0, y1 = max(0, int(x) - r), min(w, int(x) + r + 1), max(0, int(y) - r), min(h, int(y) + r + 1)
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        img[y0:y1, x0:x1] += np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0, 1) * 255.0


def _shifted_pair(h=128, w=160, dx=2.3, dy=-1.7, n=900, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
    return _render(xs, ys, h, w), _render(xs + dx, ys + dy, h, w)


def test_grid_count_matches_the_vectors_piv_returns():
    a, b = _shifted_pair()
    for win, margin, step in ((32, 8, None), (16, 6, 8), (24, 10, 12)):
        rows, cols = piv_grid_count(*a.shape, win, margin, step)
        assert piv_velocity_field(a, b, window_size=win, search_margin=margin, step=step)["u"].size == rows * cols


def test_summary_of_a_known_shift():
    a, b = _shifted_pair()
    frames = np.stack([a, b, b])
    seq = piv_velocity_sequence(frames, window_size=32, search_margin=8)
    s = summarize_piv(seq, frames.shape, 32, 8)
    assert s["frame_pairs"] == 2 and s["vectors"] == 2 * s["vectors_per_pair"]
    assert s["withheld_near_border"] >= 0
    assert abs(float(np.mean(seq["u"][: s["vectors_per_pair"]])) - 2.3) < 0.3


def test_image_frames_decode_in_name_order():
    Image = pytest.importorskip("PIL.Image")
    files = []
    for i, val in enumerate((10, 20, 30)):
        buf = io.BytesIO()
        Image.fromarray(np.full((8, 12), val, dtype=np.uint8)).save(buf, format="PNG")
        files.append((f"frame_{i:03d}.png", buf.getvalue()))
    frames = decode_image_frames(reversed(files))
    assert frames.shape == (3, 8, 12) and frames[:, 0, 0].tolist() == [10, 20, 30]
    with pytest.raises(ValueError):
        decode_image_frames(files[:1])
    with pytest.raises(ValueError):
        decode_image_frames([("a.png", files[0][1]), ("b.png", b"not an image")])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_video_decodes_to_frames(tmp_path):
    path = tmp_path / "clip.avi"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5", "-t", "1",
                    "-c:v", "ffv1", str(path)], check=True)
    frames = decode_video(str(path))
    assert frames.shape[1:] == (48, 64) and frames.shape[0] >= 2
    with pytest.raises(ValueError):
        decode_video(str(tmp_path / "missing.mp4"))


def test_parameter_guide_follows_the_quarter_rule():
    g = piv_parameter_guide(5.0, 1920, 1080)
    assert g["window_size"] == 32 and g["search_margin"] == 8 and g["step_50pct_overlap"] == 16
    assert g["vectors_50pct_overlap"]["total"] == g["vectors_50pct_overlap"]["rows"] * g["vectors_50pct_overlap"]["cols"]
    assert not g["warnings"]
    assert piv_parameter_guide(100.0, 160, 120)["warnings"]


def test_solid_body_vortex_is_recovered():
    """The source app's examples/vortex_case.py: 320 x 320 px, 1400 particles, omega 0.035 rad/frame."""
    W = H = 320
    cx, cy, omega = W / 2, H / 2, 0.035
    rng = np.random.default_rng(42)
    xs, ys = rng.uniform(5, W - 5, 1400), rng.uniform(5, H - 5, 1400)
    c, s = np.cos(omega), np.sin(omega)
    xb, yb = cx + (xs - cx) * c - (ys - cy) * s, cy + (xs - cx) * s + (ys - cy) * c
    r = piv_velocity_field(_render(xs, ys, H, W), _render(xb, yb, H, W), window_size=32, search_margin=12, step=32)
    u_true, v_true = -omega * (r["y"] - cy), omega * (r["x"] - cx)
    err = np.hypot(r["u"] - u_true, r["v"] - v_true)
    speed = np.hypot(u_true, v_true)
    rel = 100 * err[speed > 0.5] / speed[speed > 0.5]
    print(f"vortex: {r['u'].size} vectors, mean |error| {err.mean():.4f} px/frame, mean relative {rel.mean():.2f} %, "
          f"max relative {rel.max():.2f} %")
    assert r["u"].size == 9 * 9
    assert err.mean() < 0.15 and rel.mean() < 5.0
