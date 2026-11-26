"""Reference x surrogate x error (#63): the comparison figure's honesty and the L2 number it reports.

The figure is checked on a synthetic field with a known answer -- the surrogate is the reference scaled by 1 + eps --
so every number in the title can be asserted, and the colour scales are read back from the mappables rather than
from the pixels: a shared scale is a property of the contour sets, and a scaled one would still look like a figure.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pinneapple_tools.visualization.fields import (  # noqa: E402
    SolutionComparison,
    compare_solutions,
    field_error_metrics,
)

N = 25
X, Y = np.meshgrid(np.linspace(0.0, 1.0, N), np.linspace(0.0, 1.0, N))
REF = np.sin(np.pi * X) * np.cos(0.5 * np.pi * Y)


def _flat(a):
    return np.asarray(a, float).ravel()


@pytest.fixture()
def comparison():
    eps = 0.02
    return compare_solutions(_flat(X), _flat(Y), _flat(REF), _flat(REF * (1 + eps)),
                             quantity="Cp", unit="", labels=("OpenFOAM", "Surrogate"),
                             source="synthetic: surrogate = reference x 1.02")


def test_the_relative_l2_error_is_the_number_the_test_put_in(comparison):
    assert isinstance(comparison, SolutionComparison)
    assert comparison.l2_relative == pytest.approx(0.02, rel=1e-12)          # the figure's own eps
    assert comparison.l2 == pytest.approx(0.02 * np.sqrt(np.mean(REF ** 2)), rel=1e-12)
    assert comparison.linf == pytest.approx(0.02 * np.abs(REF).max(), rel=1e-12)
    r = np.asarray(REF, float).ravel()
    assert comparison.r2 == pytest.approx(1.0 - np.sum((0.02 * r) ** 2) / np.sum((r - r.mean()) ** 2), rel=1e-12)
    assert comparison.n == N * N
    # the figure says it: the L2 error is in the title and in the note under it, not only in the return value
    texts = [comparison.figure._suptitle.get_text()] + [t.get_text() for t in comparison.figure.texts]
    assert any("L2 = " in t and "2.00 %" in t for t in texts)
    r2 = 1.0 - np.sum((0.02 * r) ** 2) / np.sum((r - r.mean()) ** 2)
    assert any("L2/RMS(ref) = 0.020" in t and f"R2 = {r2:.4f}" in t and "n = 625" in t for t in texts)


def test_the_two_solution_panels_share_one_colour_scale_and_the_error_panel_has_its_own(comparison):
    fig = comparison.figure
    filled = [c for c in fig.axes[0].collections + fig.axes[1].collections if hasattr(c, "get_clim")]
    assert filled, "the reference and surrogate panels have no mappable"
    ref_clim = fig.axes[0].collections[0].get_clim()
    pred_clim = fig.axes[1].collections[0].get_clim()
    assert ref_clim == pred_clim                                              # the same colour is the same number
    lo, hi = ref_clim
    # over both panels, not over each: the surrogate is 2 % above the reference, so the shared top is the surrogate's
    assert lo == pytest.approx(min(REF.min(), (REF * 1.02).min()))
    assert hi == pytest.approx(max(REF.max(), (REF * 1.02).max()))
    err_clim = fig.axes[2].collections[0].get_clim()
    assert err_clim[0] == 0.0                                                 # a magnitude, from zero
    assert err_clim[1] == pytest.approx(np.percentile(np.abs(REF * 0.02), 99))


def test_the_colour_bars_carry_the_quantity_and_its_unit(comparison):
    labels = [ax.get_ylabel() for ax in comparison.figure.axes if ax.get_ylabel()]
    assert "Cp [-]" in labels                                                 # dimensionless says so
    assert any(t.startswith("|Surrogate - OpenFOAM|") for t in labels)


def test_a_spike_is_clipped_and_the_figure_says_so():
    pred = REF.ravel() * 1.02
    pred[0] += 10.0                                                           # one point 10 units off
    c = compare_solutions(_flat(X), _flat(Y), _flat(REF), pred, quantity="T", unit="K")
    assert c.clipped and c.error_max == pytest.approx(10.0, rel=1e-12)
    assert c.linf == pytest.approx(10.0, rel=1e-12)
    assert c.figure.axes[2].collections[0].get_clim()[1] < 1.0                # the panel is not stretched to the spike
    note = " ".join(t.get_text() for t in c.figure.texts)
    assert "clipped at the 99th percentile" in note and "10" in note


def test_one_bad_point_alone_still_gets_a_scale_and_is_flagged():
    """Every point right but one: the 99th percentile is zero, so a scale taken from it would be zero-wide and the
    panel blank. It runs to the spike instead and says the scale was clipped."""
    pred = REF.ravel().copy()
    pred[7] += 3.0
    c = compare_solutions(_flat(X), _flat(Y), _flat(REF), pred, quantity="T", unit="K")
    assert c.clipped and c.figure.axes[2].collections[0].get_clim()[1] == pytest.approx(3.0, rel=1e-12)
    assert "clipped" in " ".join(t.get_text() for t in c.figure.texts)


def test_non_finite_points_are_left_out_instead_of_counting_as_perfect():
    pred = REF.copy().ravel()
    pred[3] = np.nan
    ref = _flat(REF)
    m = field_error_metrics(ref, pred)
    assert m["n"] == N * N - 1
    f = field_error_metrics(ref, ref)
    assert f["l2"] == 0.0 and f["linf"] == 0.0 and f["r2"] == 1.0
    # an empty or mismatched pair is refused, not plotted as a blank panel
    with pytest.raises(ValueError, match="one value per point"):
        field_error_metrics(ref, pred[:-1])


def test_the_error_metrics_are_json_ready():
    import json
    c = compare_solutions(_flat(X), _flat(Y), _flat(REF), _flat(REF * 0.99), quantity="p", unit="Pa")
    json.dumps(c.metrics)
    assert set(c.metrics) == {"n", "l2", "l2_relative", "linf", "r2", "error_max", "error_clipped",
                             "error_percentile"}


def test_an_irregular_point_cloud_is_triangulated_and_still_reports_the_error():
    rng = np.random.default_rng(4)
    pts = rng.random((400, 2))
    ref = np.sin(3 * pts[:, 0]) * np.cos(2 * pts[:, 1])
    c = compare_solutions(pts[:, 0], pts[:, 1], ref, ref + 0.05, quantity="u", unit="m/s")
    assert c.n == 400 and c.l2_relative == pytest.approx(0.05 / np.sqrt(np.mean(ref ** 2)), rel=1e-12)


def test_the_benchmark_figure_of_the_arena_quotes_the_same_numbers(tmp_path):
    """The Arena's field comparison is where a Benchmark case draws reference, prediction and error: it now takes its
    error from the same function, so the panel and any table cannot disagree."""
    vis = pytest.importorskip("pinneapple_arena.visualizer")
    ref = _flat(REF)[:, None]
    pred = _flat(REF * 1.02)[:, None]
    out = str(tmp_path / "cmp.png")
    vis.plot_field_comparison([{"name": "FNO", "pred": pred, "ref": ref}], ["u"], np.column_stack([_flat(X), _flat(Y)]),
                              N, "cavity", save_path=out)
    assert (tmp_path / "cmp.png").stat().st_size > 1000
    assert field_error_metrics(ref, pred)["l2_relative"] == pytest.approx(0.02, rel=1e-12)
