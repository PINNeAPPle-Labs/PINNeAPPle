"""pinneapple_perception.change_detection on synthetic arrays (the source app's tests/test_detection.py) and the
Earth Engine credential path of pinneapple_perception.sentinel2 (no credentials needed: it must refuse)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pinneapple_perception import change_detection as det


def _flat_rgb(value: float, shape=(32, 32, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float64)


def test_band_difference_no_change_does_not_exceed_threshold():
    a = _flat_rgb(2000.0)
    b = _flat_rgb(2000.0)
    result = det.band_difference(a, b, band_index=0, band_name="red")
    assert result.mean_abs_diff_raw == 0.0
    assert result.exceeds_threshold is False


def test_band_difference_large_change_exceeds_threshold():
    a = _flat_rgb(2000.0)
    b = _flat_rgb(2000.0 + 5000.0)  # a big, real reflectance jump
    result = det.band_difference(a, b, band_index=0, band_name="red", reflectance_scale=det.DEFAULT_REFLECTANCE_SCALE)
    assert result.mean_abs_diff_reflectance == pytest.approx(0.5, rel=1e-6)
    assert result.exceeds_threshold is True


def test_reflectance_scale_choice_changes_the_verdict_honestly():
    """The real, documented finding: the original notebook's 3000 divisor
    is ~3.3x more sensitive than the physically correct 10000 divisor --
    verify this concretely with a diff that is 'significant' under one
    scale and not the other."""
    a = _flat_rgb(0.0)
    b = _flat_rgb(400.0)  # mean_abs_diff_raw = 400
    correct = det.band_difference(a, b, 0, "red", threshold=0.05, reflectance_scale=10000.0)
    legacy = det.band_difference(a, b, 0, "red", threshold=0.05, reflectance_scale=3000.0)
    assert correct.mean_abs_diff_reflectance == pytest.approx(0.04, rel=1e-6)
    assert correct.exceeds_threshold is False  # 4% < 5% threshold under the real SR scale
    assert legacy.mean_abs_diff_reflectance == pytest.approx(400.0 / 3000.0, rel=1e-6)
    assert legacy.exceeds_threshold is True    # ~13.3% > 5% threshold under the original's conflated scale


def test_compute_ndvi_matches_known_formula_and_range():
    red = np.array([[1000.0, 500.0]])
    nir = np.array([[3000.0, 500.0]])
    ndvi = det.compute_ndvi(red, nir)
    expected = np.array([[(3000 - 1000) / (3000 + 1000 + 1e-6), 0.0]])
    np.testing.assert_allclose(ndvi, expected, atol=1e-6)
    assert (ndvi >= -1.0001).all() and (ndvi <= 1.0001).all()


def test_ndvi_change_detects_real_vegetation_loss_signal():
    red_a, nir_a = np.full((16, 16), 800.0), np.full((16, 16), 4000.0)   # healthy vegetation, NDVI ~0.67
    red_b, nir_b = np.full((16, 16), 2500.0), np.full((16, 16), 2600.0)  # cleared/bare, NDVI ~0.02
    ndvi_a = det.compute_ndvi(red_a, nir_a)
    ndvi_b = det.compute_ndvi(red_b, nir_b)
    result = det.ndvi_change(ndvi_a, ndvi_b, threshold=0.1)
    assert result.mean_abs_diff > 0.5
    assert result.exceeds_threshold is True


def test_analyze_pair_no_change_reports_no_significant_change():
    a = _flat_rgb(2000.0)
    b = _flat_rgb(2000.0)
    pair = det.analyze_pair(a, b, "2024-01-01", "2024-02-01")
    assert pair.any_change is False
    assert "No significant change" in pair.narrative_html
    assert pair.ndvi_result is None  # no NIR supplied -- must not be fabricated


def test_analyze_pair_flags_real_band_and_ndvi_change():
    a = _flat_rgb(1000.0)
    b = a.copy()
    b[:, :, 0] += 6000.0  # big red-band jump
    nir_a = np.full((32, 32), 4000.0)
    nir_b = np.full((32, 32), 2600.0)  # NDVI moves from healthy toward bare
    pair = det.analyze_pair(a, b, "2024-01-01", "2024-06-01", nir_a=nir_a, nir_b=nir_b)
    assert pair.any_change is True
    assert any(r.band == "red" and r.exceeds_threshold for r in pair.band_results)
    assert pair.ndvi_result is not None
    assert pair.ndvi_result.exceeds_threshold is True
    assert "2024-01-01" in pair.narrative_html and "2024-06-01" in pair.narrative_html


def test_analyze_pair_rejects_mismatched_shapes():
    a = _flat_rgb(1000.0, shape=(16, 16, 3))
    b = _flat_rgb(1000.0, shape=(32, 32, 3))
    with pytest.raises(ValueError):
        det.analyze_pair(a, b, "2024-01-01", "2024-02-01")


def test_render_comparison_figure_produces_real_png_bytes():
    a = _flat_rgb(1500.0)
    b = _flat_rgb(1500.0)
    b[:, :, 1] += 2000.0
    png = det.render_comparison_figure(a, b, "2024-01-01", "2024-02-01")
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes
    assert len(png) > 1000


def test_render_comparison_figure_with_ndvi_row():
    a = _flat_rgb(1500.0)
    b = _flat_rgb(1600.0)
    ndvi_a = det.compute_ndvi(a[..., 0], np.full((32, 32), 4000.0))
    ndvi_b = det.compute_ndvi(b[..., 0], np.full((32, 32), 3000.0))
    png = det.render_comparison_figure(a, b, "2024-01-01", "2024-02-01", ndvi_a=ndvi_a, ndvi_b=ndvi_b)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_html_report_embeds_real_base64_images_and_narrative():
    a = _flat_rgb(1000.0)
    b = a.copy()
    b[:, :, 2] += 4000.0
    pair = det.analyze_pair(a, b, "2024-01-01", "2024-03-01")
    png = det.render_comparison_figure(a, b, "2024-01-01", "2024-03-01")
    html = det.render_html_report([pair], [png], lon=-47.8716, lat=-15.6146, buffer_m=5000, has_ndvi=False)
    assert "<!DOCTYPE html>" in html
    assert "2024-01-01" in html and "2024-03-01" in html
    assert "data:image/png;base64," in html
    assert "-47.8716" in html


def test_report_escapes_caller_supplied_dates():
    a = _flat_rgb(1000.0)
    pair = det.analyze_pair(a, a.copy(), "<script>alert(1)</script>", "2024-03-01")
    html = det.render_html_report([pair], [det.render_comparison_figure(a, a, "x", "y")], lon=0.0, lat=0.0,
                                  buffer_m=1.0, has_ndvi=False)
    assert "<script>" not in html and "&lt;script&gt;" in html


@pytest.mark.skipif(bool(os.environ.get("EE_PROJECT_ID")), reason="Earth Engine credentials are configured here")
def test_earth_engine_without_credentials_is_an_explicit_error():
    pytest.importorskip("ee")
    from pinneapple_perception.sentinel2 import EarthEngineUnavailable, ee_initialize

    with pytest.raises(EarthEngineUnavailable) as e:
        ee_initialize()
    assert "EE_PROJECT_ID" in str(e.value)
