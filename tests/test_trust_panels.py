"""Field panels with trust verdicts (pinneapple_tools.visualization.trust_panels)."""
import numpy as np

from pinneapple_tools.visualization.trust_panels import Panel, Row, trust_panels, trust_panels_gif


def _rows(scale=1.0):
    x = np.linspace(0, 1, 12)
    f = scale * np.outer(np.sin(np.pi * x), np.sin(np.pi * x))
    return [Row("regime A", [Panel("reference", f), Panel("model", 1.1 * f, verdict="APPROVED", verdicts=["drift: FAIL 2.0x limit"], note="error 10%"),
                             Panel("residual", f - 0.5, kind="residual")]),
            Row("regime B", [Panel("reference", f), Panel("model", 0.5 * f, verdict="REJECT"),
                             Panel("residual", 3 * f, kind="residual")])]


def test_trust_panels_figure_and_gif(tmp_path):
    fig = trust_panels(_rows(), title="t", footer="f")
    texts = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert {"APPROVED", "REJECT", "error 10%", "drift: FAIL 2.0x limit"} <= texts
    assert len(fig.axes) == 2 * (3 + 2)                       # 3 panels and 2 colour bars per row
    p = trust_panels_gif([_rows(), _rows(2.0)], tmp_path / "a.gif", dpi=40)
    from PIL import Image
    with Image.open(p) as im:
        assert im.n_frames == 2


def test_badge_colour_follows_the_status_word():
    from pinneapple_tools.visualization.trust_panels import STATUS_COLOURS
    fig = trust_panels(_rows())
    boxes = {t.get_text(): t.get_bbox_patch().get_facecolor() for ax in fig.axes for t in ax.texts
             if t.get_bbox_patch() is not None and t.get_text() in ("APPROVED", "REJECT")}
    from matplotlib.colors import to_rgba
    assert boxes["REJECT"] == to_rgba(STATUS_COLOURS["REJECT"]) and boxes["APPROVED"] == to_rgba(STATUS_COLOURS["PASS"])
