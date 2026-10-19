"""The showcase panel: tiles from the curation (never C or D), crops that keep colour bars whole, MP4 / GIF / PNG."""
import json
import os

import numpy as np
import pytest


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


# ---------------------------------------------------------------------------------------------- the showcase panel
def test_fit_crop_keeps_or_drops_a_colour_bar_whole():
    from pinneapple_lab.showcase import fit_crop
    img = np.full((200, 400, 3), 255, np.uint8)
    img[20:180, 20:300] = 90                                   # the plot
    img[20:180, 330:350] = 30                                  # its colour bar, after a gap
    (x0, y0, x1, y1), _ = fit_crop(img, 1.0)
    assert x1 <= 330 or x1 >= 350                              # never through the bar
    (x0, y0, x1, y1), _ = fit_crop(img, 2.0)
    assert x1 >= 350                                           # wide tile: the bar is kept whole


def test_choose_tiles_and_panel(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from pinneapple_lab.showcase import choose_tiles, panel
    root = _fake_lab(tmp_path, {"a1/x": "A", "a2/x": "A", "a3/x": "A", "b1/y": "B", "c1/z": "C"})
    t = choose_tiles(root, picks=[("c1/z", ""), ("b1/y", ""), ("a2/x", "")])
    assert t["hero"].item == "a2/x"                            # picks first; the C item is never shown
    assert [x.item for x in t["b"]] == ["b1/y"] and all(x.tier == "A" for x in t["grid"])
    assert all(x.evidence == f"check of {x.item}" for x in [t["hero"], *t["grid"], *t["b"]])
    meta = panel(root, lang="pt", out_dir=str(tmp_path / "show"), picks=[], seconds=0.4, fps=5)
    for k in ("mp4", "gif", "png"):
        assert os.path.getsize(meta["files"][k]) > 0
    from PIL import Image
    assert Image.open(meta["files"]["png"]).size == (1080, 1350)
    assert {x["item"] for x in meta["tiles"]} == {"a1/x", "a2/x", "a3/x", "b1/y"}


def test_evidence_lines_labels_fit_and_focus():
    from pinneapple_lab.showcase import _fit_line, _focus, evidence_line
    rec = {"headline": {"text": "drag vs wind tunnel · Ahmed, Ramm & Faltin (1984), 25 deg slant",
                        "label": "drag vs wind tunnel (Ahmed, Ramm & Faltin, 1984)",
                        "label_pt": "arrasto vs túnel de vento (Ahmed, Ramm & Faltin, 1984)",
                        "label_short_pt": "arrasto vs túnel de vento (Ahmed 1984)"}}
    assert evidence_line(rec, "pt") == "arrasto vs túnel de vento (Ahmed, Ramm & Faltin, 1984)"
    assert evidence_line(rec, "pt", short=True) == "arrasto vs túnel de vento (Ahmed 1984)"
    assert evidence_line(rec, "en", short=True) == "drag vs wind tunnel (Ahmed, Ramm & Faltin, 1984)"
    assert evidence_line({"headline": {"text": "raw line"}}, "pt") == "raw line"
    text, font = _fit_line("friction vs Colebrook · loss vs Ito", 13, 120)
    assert "…" not in text and font.getlength(text) <= 120 and text.startswith("friction vs Colebrook")
    a = np.zeros((100, 200, 3), np.uint8)
    assert _focus(a, (0.1, 0.2, 0.6, 0.7)).shape == (50, 100, 3)
    assert _focus(a, [(0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1)]).shape == (50, 200, 3)        # two panels side by side
