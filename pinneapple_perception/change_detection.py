"""Change detection between two multispectral images of the same place: per-band mean absolute reflectance change
and NDVI change against thresholds, a comparison figure and a self-contained HTML report.

Inputs are plain arrays (H, W, 3) in band order red, green, blue (Sentinel-2 B4, B3, B2) and, for NDVI, a real
near-infrared band (H, W) (B8): no NIR is ever synthesised. Sentinel-2 surface reflectance is stored as
DN = 10000 x reflectance, so the default ``reflectance_scale`` is 10000; 3000 is only the display stretch of the
RGB previews. :mod:`pinneapple_perception.sentinel2` fetches such images from Google Earth Engine.

Moved here from the PINNeAPPle-apps ``satellite_change_monitoring`` product (OrbitWatch), which extracted it from
a notebook with two fixes kept here:

1. **The notebook's NDVI was fake.** Its own comment says so:
   "Simulando que banda NIR é R + 400 (ajuste conforme sua fonte real)"
   -- i.e. it approximated the near-infrared band as
   ``nir = red_band + 400``, a made-up placeholder, rather than fetching
   Sentinel-2's real NIR band (B8). `compute_ndvi` here takes a REAL nir
   array as an explicit required argument -- `pinneapple_perception.sentinel2`
   fetches the real B8 band alongside B4/B3/B2 for exactly this reason.
   There is no fallback path in this module that fabricates an NIR band;
   a caller without a real NIR array should not call `compute_ndvi` at
   all (the app layer makes this optional per-request, not a silent
   default).
2. **The notebook conflated a display-stretch constant with a physical
   reflectance-change threshold.** It divided differences by 3000 for
   BOTH the RGB visualization stretch (`img / 3000`, a common contrast
   trick for Sentinel-2's 0-10000 raw digital-number range) AND the
   quantitative band-change threshold check (`mean_diff / 3000 > 0.05`).
   Real Sentinel-2 Surface Reflectance products scale reflectance as
   raw_DN / 10000 (documented by ESA/Copernicus), so a diff normalized by
   3000 is not actually "5% of true reflectance change" -- it's roughly
   3.3x more sensitive than that. This module defaults
   `reflectance_scale=10000.0` (the real, physically correct SR scale)
   for the quantitative threshold, and keeps 3000 available separately,
   only for the RGB preview's own contrast stretch (`rgb_preview_stretch`
   below), where it is purely a display choice with no claim to physical
   meaning. Pass `reflectance_scale=3000.0` to reproduce the original
   notebook's (physically-inconsistent) threshold behavior exactly.
"""
from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass

import numpy as np

DEFAULT_BAND_DIFF_THRESHOLD = 0.05
DEFAULT_NDVI_THRESHOLD = 0.1
DEFAULT_REFLECTANCE_SCALE = 10000.0          # real Sentinel-2 SR scale (raw DN -> reflectance)
LEGACY_VISUALIZATION_NORMALIZER = 3000.0     # original notebook's RGB preview stretch AND (incorrectly) its threshold divisor
BAND_NAMES = ("red", "green", "blue")        # Sentinel-2 B4, B3, B2 respectively


# ---------------------------------------------------------------------------
# Core math -- pure numpy, fully testable with synthetic arrays.
# ---------------------------------------------------------------------------

@dataclass
class BandChangeResult:
    band: str
    mean_abs_diff_raw: float
    mean_abs_diff_reflectance: float
    threshold: float
    exceeds_threshold: bool


def band_difference(
    img_a: np.ndarray, img_b: np.ndarray, band_index: int, band_name: str,
    threshold: float = DEFAULT_BAND_DIFF_THRESHOLD, reflectance_scale: float = DEFAULT_REFLECTANCE_SCALE,
) -> BandChangeResult:
    """Real per-band mean-absolute-difference change check, same
    computation as the original notebook's `detecta_mudanca` (mean of
    |img_b - img_a| for one band, normalized, compared to a threshold),
    but with an explicit, physically-meaningful `reflectance_scale`
    (default the real Sentinel-2 SR scale, 10000) instead of the
    original's conflated 3000 display-stretch constant -- see this
    module's docstring."""
    a = img_a[..., band_index].astype(np.float64)
    b = img_b[..., band_index].astype(np.float64)
    diff = np.abs(b - a)
    mean_raw = float(diff.mean())
    mean_reflectance = mean_raw / reflectance_scale
    return BandChangeResult(
        band=band_name, mean_abs_diff_raw=mean_raw, mean_abs_diff_reflectance=mean_reflectance,
        threshold=threshold, exceeds_threshold=mean_reflectance > threshold,
    )


def compute_ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Real NDVI: (NIR - RED) / (NIR + RED + eps). `red` and `nir` must
    both be REAL band arrays (Sentinel-2 B4 and B8 respectively) -- this
    function has no fallback that fabricates either band. Same formula
    as the original notebook's `calcula_ndvi`, applied to a real NIR
    input instead of the original's `red + 400` placeholder."""
    red = red.astype(np.float64)
    nir = nir.astype(np.float64)
    return (nir - red) / (nir + red + eps)


@dataclass
class NdviChangeResult:
    mean_abs_diff: float
    threshold: float
    exceeds_threshold: bool


def ndvi_change(ndvi_a: np.ndarray, ndvi_b: np.ndarray, threshold: float = DEFAULT_NDVI_THRESHOLD) -> NdviChangeResult:
    mean_abs_diff = float(np.mean(np.abs(ndvi_b - ndvi_a)))
    return NdviChangeResult(mean_abs_diff=mean_abs_diff, threshold=threshold, exceeds_threshold=mean_abs_diff > threshold)


@dataclass
class PairAnalysis:
    date_a: str
    date_b: str
    band_results: list  # list[BandChangeResult]
    ndvi_result: NdviChangeResult | None
    any_change: bool
    narrative_html: str


def analyze_pair(
    rgb_a: np.ndarray, rgb_b: np.ndarray, date_a: str, date_b: str,
    nir_a: np.ndarray | None = None, nir_b: np.ndarray | None = None,
    band_diff_threshold: float = DEFAULT_BAND_DIFF_THRESHOLD,
    ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD,
    reflectance_scale: float = DEFAULT_REFLECTANCE_SCALE,
) -> PairAnalysis:
    """Real per-pair change analysis: three band-difference checks
    (red/green/blue) always run; NDVI is computed and checked only if
    real `nir_a`/`nir_b` arrays are supplied (never fabricated).
    `date_a` should chronologically precede `date_b` -- the caller is
    responsible for ordering (this fixes a real, if cosmetic,
    inconsistency in the original notebook: its narrative text in cell 7
    labeled pairs as "between dates[i+1] and dates[i]" while its HTML
    report in cell 8 labeled the SAME pairs "dates[i] -> dates[i+1]",
    reversed direction for an otherwise-identical diff computation)."""
    if rgb_a.shape != rgb_b.shape:
        raise ValueError(f"rgb_a shape {rgb_a.shape} != rgb_b shape {rgb_b.shape}; both images must be the same size.")

    band_results = [
        band_difference(rgb_a, rgb_b, i, name, threshold=band_diff_threshold, reflectance_scale=reflectance_scale)
        for i, name in enumerate(BAND_NAMES)
    ]

    ndvi_result = None
    if nir_a is not None and nir_b is not None:
        red_a, red_b = rgb_a[..., 0], rgb_b[..., 0]
        ndvi_a = compute_ndvi(red_a, nir_a)
        ndvi_b = compute_ndvi(red_b, nir_b)
        ndvi_result = ndvi_change(ndvi_a, ndvi_b, threshold=ndvi_threshold)

    changes = [r for r in band_results if r.exceeds_threshold]
    any_change = bool(changes) or (ndvi_result is not None and ndvi_result.exceeds_threshold)

    items = []
    for r in changes:
        items.append(f"<li>&Delta; {r.band} band significant: {r.mean_abs_diff_reflectance:.4f} reflectance units (threshold {r.threshold})</li>")
    if ndvi_result is not None and ndvi_result.exceeds_threshold:
        items.append(f"<li>&Delta; NDVI mean: <strong>{ndvi_result.mean_abs_diff:.4f}</strong> (threshold {ndvi_result.threshold}) -- relevant change</li>")
    if not items:
        items.append("<li>No significant change observed in any band or NDVI.</li>")
    # the dates come from the caller: escaped, since the report embeds this fragment as HTML
    narrative_html = (f"Between <strong>{html.escape(str(date_a))}</strong> and "
                      f"<strong>{html.escape(str(date_b))}</strong>:<ul>{''.join(items)}</ul>")

    return PairAnalysis(
        date_a=date_a, date_b=date_b, band_results=band_results, ndvi_result=ndvi_result,
        any_change=any_change, narrative_html=narrative_html,
    )


# ---------------------------------------------------------------------------
# Figure rendering -- same visual layout as the original notebook's cell 8
# 3x3 grid (RGB before/after/diff, per-band diff maps, NDVI before/after/
# diff), returned as PNG bytes rather than saved to a fixed relative path.
# ---------------------------------------------------------------------------

def rgb_preview_stretch(img: np.ndarray, normalizer: float = LEGACY_VISUALIZATION_NORMALIZER) -> np.ndarray:
    """Purely a display contrast stretch (clip to [0,1] after dividing by
    a fixed constant) -- matches the original notebook's `img/3000`
    preview trick. This has NO physical-reflectance meaning (see this
    module's docstring); it is only ever used for the RGB preview panels,
    never for the quantitative threshold checks."""
    return np.clip(img.astype(np.float64) / normalizer, 0.0, 1.0)


def render_comparison_figure(
    rgb_a: np.ndarray, rgb_b: np.ndarray, date_a: str, date_b: str,
    ndvi_a: np.ndarray | None = None, ndvi_b: np.ndarray | None = None,
) -> bytes:
    """Render the same-style comparison grid the original notebook
    produced (RGB pair + diff, per-band diff maps, and -- when real NDVI
    arrays are supplied -- an NDVI before/after/delta row), as real PNG
    bytes (matplotlib `Agg` backend, no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff = np.abs(rgb_b.astype(np.float64) - rgb_a.astype(np.float64))
    has_ndvi = ndvi_a is not None and ndvi_b is not None
    nrows = 3 if has_ndvi else 2
    fig, axs = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    if nrows == 1:
        axs = axs.reshape(1, 3)

    axs[0, 0].imshow(rgb_preview_stretch(rgb_a))
    axs[0, 0].set_title(f"Image {date_a} (RGB)")
    axs[0, 1].imshow(rgb_preview_stretch(rgb_b))
    axs[0, 1].set_title(f"Image {date_b} (RGB)")
    axs[0, 2].imshow(rgb_preview_stretch(diff))
    axs[0, 2].set_title("RGB difference")

    cmaps = ("Reds", "Greens", "Blues")
    labels = ("Red band (B4)", "Green band (B3)", "Blue band (B2)")
    for i, (cmap, label) in enumerate(zip(cmaps, labels, strict=True)):
        axs[1, i].imshow(diff[..., i], cmap=cmap)
        axs[1, i].set_title(f"Δ {label}")

    if has_ndvi:
        axs[2, 0].imshow(ndvi_a, cmap="RdYlGn", vmin=-1, vmax=1)
        axs[2, 0].set_title(f"NDVI - {date_a}")
        axs[2, 1].imshow(ndvi_b, cmap="RdYlGn", vmin=-1, vmax=1)
        axs[2, 1].set_title(f"NDVI - {date_b}")
        im = axs[2, 2].imshow(ndvi_b - ndvi_a, cmap="bwr", vmin=-1, vmax=1)
        axs[2, 2].set_title("Δ NDVI")
        fig.colorbar(im, ax=axs[2, 2], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML report rendering -- same overall shape as the original notebook's
# `relatorio_comparativo_ndvi_rgb.html` (title, pair count, one section per
# consecutive pair with an embedded comparison image and a narrative list),
# but self-contained: images are embedded as base64 data URIs rather than
# written to a separate `comparativo_relatorio/` folder the caller would
# otherwise have to ship alongside the HTML file.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OrbitWatch Change-Detection Report</title>
</head>
<body style="font-family:Arial; max-width:900px; margin:auto">
    <h1>Satellite Change-Detection Report</h1>
    <p><strong>Region:</strong> point ({{ lon }}, {{ lat }}), {{ buffer_m }} m buffer</p>
    <p><strong>Pairs analyzed:</strong> {{ total }}</p>
    <p>Real Sentinel-2 Surface Reflectance imagery, per-band difference thresholding
       {% if has_ndvi %}and real-NIR-band NDVI change detection{% else %}(no NIR band supplied -- NDVI not computed for this report){% endif %}.</p>
    {% for block in blocks %}
        <div style="margin-bottom: 40px;">
            <h3>{{ block.date_a }} &rarr; {{ block.date_b }}</h3>
            <img src="data:image/png;base64,{{ block.image_b64 }}" style="width:100%;"><br>
            {{ block.narrative_html | safe }}
        </div>
    {% endfor %}
    <p style="color:#888; font-size:0.85em;">Generated by OrbitWatch (PINNeAPPle, apps/orbitwatch).</p>
</body>
</html>
"""


def render_html_report(pairs: list, images_png: list, lon: float, lat: float, buffer_m: float, has_ndvi: bool) -> str:
    blocks = []
    for pair, png_bytes in zip(pairs, images_png, strict=True):
        blocks.append({
            "date_a": pair.date_a,
            "date_b": pair.date_b,
            "image_b64": base64.b64encode(png_bytes).decode("ascii"),
            "narrative_html": pair.narrative_html,
        })
    from jinja2 import Environment

    template = Environment(autoescape=True).from_string(_HTML_TEMPLATE)
    return template.render(total=len(blocks), blocks=blocks, lon=lon, lat=lat, buffer_m=buffer_m, has_ndvi=has_ndvi)
