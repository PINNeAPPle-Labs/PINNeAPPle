"""Finishing a figure: the contrast look, the bloom and the vignette, all on the written PNG/JPEG."""
import numpy as np
import pytest
from PIL import Image

from pinneapple_tools.visualization.studio import post as P


def _grey(w=64, h=48, value=0.5):
    return np.full((h, w, 3), value, float)


def test_look_moves_contrast_and_keeps_black_and_white():
    flat = np.array([0.0, 0.25, 0.5, 0.75, 1.0])[:, None, None].repeat(3, axis=2)
    high = P.look(flat, "high")
    assert high[0, 0, 0] == pytest.approx(0.0, abs=1e-6) and high[4, 0, 0] == pytest.approx(1.0, abs=1e-6)
    assert high[1, 0, 0] < flat[1, 0, 0] and high[3, 0, 0] > flat[3, 0, 0], "shadows down, highlights up"
    assert high[2, 0, 0] == pytest.approx(0.5, abs=1e-6), "the mid tone stays where it was"
    assert P.look(flat, "neutral") == pytest.approx(flat)


def test_punchy_adds_saturation_and_leaves_grey_alone():
    colour = np.array([[[0.6, 0.4, 0.4]]])
    out = P.look(colour, "punchy")
    assert out[0, 0, 0] - out[0, 0, 1] > colour[0, 0, 0] - colour[0, 0, 1]
    assert P.look(_grey(), "punchy") == pytest.approx(_grey(), abs=2e-3)


def test_unknown_look_and_post_say_what_the_options_are(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(p)
    with pytest.raises(ValueError, match="punchy"):
        P.apply(str(p), look_name="vintage")
    with pytest.raises(ValueError, match="vignette"):
        P.apply(str(p), post="grain")


def test_vignette_darkens_the_corners_and_leaves_the_middle():
    flat = _grey(value=0.8)
    out = P.vignette(flat, strength=0.6)
    assert out[0, 0, 0] < flat[0, 0, 0] * 0.6, "the corner falls a lot"
    assert out[out.shape[0] // 2, out.shape[1] // 2, 0] == pytest.approx(0.8, abs=1e-6), "the centre is untouched"


def test_bloom_spreads_only_where_the_highlights_are():
    img = _grey(value=0.1)
    img[20:28, 28:36] = 1.0                              # a lamp, not a pixel: a render's highlight has area
    out = P.bloom(img, threshold=0.85, radius=0.06, strength=0.8)
    assert out[24, 32, 0] >= img[24, 32, 0]
    assert out[24, 40, 0] > img[24, 40, 0], "the neighbours pick up light"
    assert out[0, 0, 0] == pytest.approx(0.1, abs=1e-6), "far from the highlight nothing changes"


def test_apply_is_in_place_keeps_alpha_and_leaves_exr_alone(tmp_path):
    p = tmp_path / "shot.png"
    Image.new("RGBA", (32, 32), (128, 128, 128, 255)).save(p)
    before = p.read_bytes()
    P.apply(str(p), look_name="neutral", post=None)
    assert p.read_bytes() == before, "neutral and no post writes nothing"

    P.apply(str(p), look_name="high", post="vignette")
    im = Image.open(p)
    assert im.mode == "RGBA" and im.size == (32, 32)
    arr = np.asarray(im.convert("RGB"), float)
    assert arr[0, 0].mean() < arr[16, 16].mean(), "corner darker than the middle"

    exr = tmp_path / "passes.exr"
    exr.write_bytes(b"not really an exr")
    P.apply(str(exr), look_name="high", post="bloom")
    assert exr.read_bytes() == b"not really an exr", "an EXR is a pass set, not a picture to grade"
