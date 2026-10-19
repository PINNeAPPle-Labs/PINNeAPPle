"""The PINNeAPPle quality seal: tier from the curation record, PNG / SVG badges, and stamps on matplotlib figures,
images, GIFs and videos. The A-F rules and the six seals are in test_lab_grades.py."""
import json
import os

import numpy as np
import pytest

from pinneapple_lab.seal import (
    SealError,
    load_curation,
    pick_corner,
    render_seal,
    seal_for,
    sealed_savefig,
    stamp_all,
    stamp_figure,
    stamp_file,
    stamp_image,
)


def _plot_png(path, colorbar=True, size=(6, 4)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(np.random.default_rng(0).random((40, 60)), cmap="viridis")
    ax.set_title("a field")
    if colorbar:
        fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=60)
    plt.close(fig)


def _fake_lab(tmp_path, tiers):
    """A lab folder with a curation.json and one run (with a figure) per item."""
    root = tmp_path / "lab"
    items = {}
    for i, (key, tier) in enumerate(tiers.items()):
        exp = key.split("/")[0]
        run_id = f"{exp}-{i:04d}"
        figs = root / "runs" / exp / run_id / "figures"
        figs.mkdir(parents=True)
        _plot_png(str(figs / "field.png"))
        items[key] = {"tier": tier, "tier_name": "x", "experiment": exp, "best_run": run_id, "best_score": 90 - i,
                      "title": key, "headline": {"kind": "reference" if tier == "A" else "benchmark",
                                                 "text": f"check of {key}", "check": "c", "missing": ""}}
    (root / "curation.json").write_text(json.dumps({"generated": "2026-10-18 00:00 UTC", "items": items}))
    return str(root)


# ---------------------------------------------------------------------------------------------- the seal
def test_seal_tier_always_from_the_curation(tmp_path):
    root = _fake_lab(tmp_path, {"a/x": "A", "b/y": "B", "c/z": "C"})
    s = seal_for("a/x", root)
    assert s.tier == "A" and s.colour == "#2FBF71" and s.reason == "check of a/x"
    assert seal_for("b/y", root).colour == "#A6D96A"
    cur = load_curation(root)
    cur["items"]["a/x"]["headline"]["label"] = "drag vs Ahmed 1984"   # the display label wins over the raw text
    assert seal_for("a/x", cur).reason == "drag vs Ahmed 1984"
    assert seal_for("c/z", root).tier == "C"                   # every tier is sealed ...
    with pytest.raises(SealError):
        seal_for("c/z", root, tiers="A,B")                     # ... unless the selection leaves it out
    with pytest.raises(KeyError):
        seal_for("nope", root)
    cur = load_curation(root)                                  # re-curation: the same call gives the new tier
    cur["items"]["a/x"]["tier"] = "B"
    (tmp_path / "lab" / "curation.json").write_text(json.dumps({k: v for k, v in cur.items() if k != "_root"}))
    assert seal_for("a/x", root).tier == "B"


def test_seal_png_svg_and_compact(tmp_path):
    s = seal_for("a/x", _fake_lab(tmp_path, {"a/x": "A"}))
    im = render_seal(s, height=96)
    assert im.mode == "RGBA" and im.height == 96 and 2.5 < im.width / im.height < 6
    a = np.asarray(im)
    assert a[..., 3].max() == 255 and a[0, 0, 3] == 0          # rounded corners are transparent
    small = render_seal(s, height=40)                          # compact below 56 px: no small print
    assert small.width / small.height < im.width / im.height
    svg = render_seal(s, fmt="svg")
    assert svg.startswith("<svg") and "#2FBF71" in svg and ">A</text>" in svg and "PINNeAPPle" in svg
    assert "check of a/x" in svg


def test_auto_corner_avoids_busy_regions():
    img = np.full((300, 400, 3), 255, np.uint8)
    img[:, 340:] = (np.linspace(0, 255, 300)[:, None, None] * np.array([1, 0.5, 0.2])).astype(np.uint8)  # a colour bar
    img[260:, :120] = np.random.default_rng(1).integers(0, 255, (40, 120, 3))                          # axis text
    assert pick_corner(img, 90, 30, 6) == "tl"
    assert pick_corner(img, 90, 30, 6, avoid=[(0, 0, 200, 60)]) != "tl"         # an excluded box is never used


def test_stamp_image_figure_and_savefig_hook(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = seal_for("a/x", _fake_lab(tmp_path, {"a/x": "A"}))
    src = str(tmp_path / "plot.png")
    _plot_png(src)
    out = stamp_file(src, s)
    assert out.endswith("plot.sealed.png") and os.path.exists(out)
    before, after = np.asarray(stamp_image(src, s, opacity=0.0)), np.asarray(stamp_image(src, s))
    assert before.shape == after.shape and (before != after).any()
    fig, ax = plt.subplots(figsize=(6, 4))
    cb = fig.colorbar(ax.imshow(np.random.default_rng(0).random((10, 10))), ax=ax)
    stamp_figure(fig, s)
    seal_ax = [a for a in fig.axes if a.get_label() == "pinneapple-seal"]
    assert len(seal_ax) == 1
    r = fig.canvas.get_renderer()
    b1, b2 = seal_ax[0].get_window_extent(r), cb.ax.get_tightbbox(r)
    assert not (b1.x0 < b2.x1 and b2.x0 < b1.x1 and b1.y0 < b2.y1 and b2.y0 < b1.y1)   # off the colour bar
    stamp_figure(fig, s, corner="tl")                          # re-stamping replaces, never stacks
    assert sum(a.get_label() == "pinneapple-seal" for a in fig.axes) == 1
    plt.close(fig)
    fig, _ = plt.subplots()
    with sealed_savefig(s):
        fig.savefig(str(tmp_path / "hooked.png"))
    assert any(a.get_label() == "pinneapple-seal" for a in fig.axes)
    plt.close(fig)


def test_stamp_gif_and_mp4(tmp_path):
    from PIL import Image
    s = seal_for("a/x", _fake_lab(tmp_path, {"a/x": "A"}))
    frames = [Image.fromarray(np.full((120, 200, 3), 30 * k, np.uint8)) for k in range(5)]
    gif = str(tmp_path / "m.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=[100, 100, 200, 100, 100], loop=0)
    out = stamp_file(gif, s)
    g = Image.open(out)
    assert g.n_frames == 5
    pytest.importorskip("imageio_ffmpeg")
    import imageio_ffmpeg
    mp4 = str(tmp_path / "m.mp4")
    w = imageio_ffmpeg.write_frames(mp4, (200, 120), fps=10, macro_block_size=8)
    w.send(None)
    for k in range(10):
        w.send(np.full((120, 200, 3), 20 * k, np.uint8).tobytes())
    w.close()
    out = stamp_file(mp4, s)
    assert os.path.getsize(out) > 0


def test_stamp_all_exports_every_tier_or_a_selection(tmp_path):
    root = _fake_lab(tmp_path, {"a/x": "A", "b/y": "B", "c/z": "C"})
    assert set(stamp_all(root, str(tmp_path / "every"))["items"]) == {"a/x", "b/y", "c/z"}
    man = stamp_all(root, str(tmp_path / "out"), tiers="A,B")
    assert set(man["items"]) == {"a/x", "b/y"} and man["tiers"] == ["A", "B"]
    assert os.path.exists(tmp_path / "out" / "a_x" / "field.png") and os.path.exists(tmp_path / "out" / "a_x" / "seal.svg")
    assert json.load(open(tmp_path / "out" / "manifest.json"))["items"]["b/y"]["tier"] == "B"


def test_cli_seal_and_stamp(tmp_path, capsys):
    from pinneapple_lab.__main__ import main as cli
    root = _fake_lab(tmp_path, {"a/x": "A"})
    assert cli(["--root", root, "seal", "a/x", "-o", str(tmp_path / "s.svg")]) == 0
    src = str(tmp_path / "p.png")
    _plot_png(src)
    assert cli(["--root", root, "stamp", "a/x", src, "-o", str(tmp_path / "p2.png")]) == 0
    assert os.path.exists(tmp_path / "p2.png")


def test_busy_image_and_figure_get_a_strip_instead_of_a_covered_corner(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = seal_for("a/x", _fake_lab(tmp_path, {"a/x": "A"}))
    busy = np.random.default_rng(2).integers(0, 255, (200, 300, 3), dtype=np.uint8)   # no free corner anywhere
    out = stamp_image(busy, s)
    assert out.height > 200 and (np.asarray(out)[:200] == busy).all()                  # the image itself is untouched
    assert stamp_image(busy, s, extend=False).height == 200
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(np.random.default_rng(3).random((30, 40)), aspect="auto")
    h0 = fig.get_size_inches()[1]
    stamp_figure(fig, s)
    assert fig.get_size_inches()[1] > h0 and ax.get_position().y0 > 0                # the plot moved up, uncovered
    plt.close(fig)
