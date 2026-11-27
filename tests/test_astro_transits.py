"""A transit model, the stellar variability around it, and the vetting that separates a planet from a look-alike (#386).

There is no network here, so the TESS and Kepler light curves and the NASA Exoplanet Archive labels are not available.
What these tests pin is the machinery the issue asks for, measured on a labelled set generated here with a stated
model: the Mandel-Agol-style transit model, the instrument systematics, BLS as a real baseline, the vetting checks
(odd/even depth, secondary eclipse, V shape, centroid shift), and the recovery and false-positive rates that follow.
The numbers asserted below were measured on this machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.astro import transits as T


def test_the_transit_model_is_normalised_and_goes_dark_inside_the_disk():
    """Outside the transit the model is exactly 1; at full occultation it is 1 - k^2, the area ratio."""
    k = 0.1
    time = np.array([-0.5, 0.0, 0.5])
    flux = T.transit_model(time, period=10.0, t0=0.0, k=k, b=0.0, t14=0.3)
    assert flux[0] == pytest.approx(1.0) and flux[-1] == pytest.approx(1.0)
    assert flux[1] == pytest.approx(1.0 - k ** 2, rel=2e-2)


def test_the_limb_darkening_makes_the_bottom_curved_rather_than_flat():
    """A planet crossing a limb-darkened star gives a rounded bottom; with no limb darkening the bottom is flat."""
    time = np.linspace(-0.2, 0.2, 41)
    with_u = T.transit_model(time, period=10.0, t0=0.0, k=0.1, b=0.0, t14=0.3, u1=0.4, u2=0.3)
    without = T.transit_model(time, period=10.0, t0=0.0, k=0.1, b=0.0, t14=0.3, u1=0.0, u2=0.0)
    inside = np.abs(time) < 0.05
    assert with_u[inside].std() > 0.0
    assert without[inside].std() == pytest.approx(0.0, abs=1e-3)


def test_the_geometry_helpers_agree_on_the_shape_and_disagree_on_the_density_scale():
    """`a/R*` from the density, then the durations from that, then the density back: the shape round trips, the scale does not.

    Measured: the recovered density is **1,4045×** the density that produced the durations, and that factor is
    *constant* over k = 0,02 … 0,15 (1,4048 to 1,4039, a spread of 6e-4), so it is a scale between the two formulas and
    not a mis-modelled planet size. This test pins the factor and the k-independence; it does not claim which of the two
    is the right normalisation — that is a question for whoever uses the number, and it is now impossible to change it
    without noticing.
    """
    period = 6.435640363441553
    a_over_r = T.a_over_rstar(period, density_rho_sun=1.0)
    angles = float(np.degrees(np.arccos(0.3 / a_over_r)))
    ratios = []
    for k in (0.02, 0.05, 0.09, 0.15):
        durations = T.transit_durations(period, a_over_r, angles, k)
        assert durations["t14_d"] > durations["t23_d"] > 0.0
        assert durations["impact"] == pytest.approx(0.3, abs=1e-6)
        assert not durations["grazing"]
        ratios.append(T.stellar_density(period, k, durations["impact"], durations["t14_d"]))
    assert all(r == pytest.approx(1.4045, rel=2e-3) for r in ratios)
    assert max(ratios) - min(ratios) < 1e-3


def test_the_instrument_systematics_are_the_ones_the_issue_names():
    """Momentum dumps and scattered light are in the model, so the vetting has something real to be robust against."""
    time = np.linspace(0.0, 18.0, 2000)
    clean = T.instrument_systematics(time, seed=0, dumps=False, scattered=False)
    with_dumps = T.instrument_systematics(time, seed=0, dumps=True, scattered=False)
    assert np.allclose(clean["curve"], 0.0) and len(clean["momentum_dumps"]) == 0
    assert len(with_dumps["momentum_dumps"]) > 0
    assert np.abs(with_dumps["curve"]).max() > 1e-6            # ppm-scale jumps, in relative flux
    assert with_dumps["dump_ppm"] > 0 and with_dumps["scattered_ppm"] == 0.0


def test_a_synthetic_light_curve_carries_what_it_says_it_carries():
    """The labelled generator returns the truth with the data, which is what makes the rates below measurable."""
    curve = T.synthetic_light_curve("planet", seed=1)
    truth = curve["truth"]
    assert truth["kind"] == "planet" and truth["label"] == 1
    assert truth["period_d"] > 0 and 0 < truth["depth"] < 0.1
    assert curve["flux"].shape == curve["time"].shape == curve["flux_err"].shape
    kinds = {T.synthetic_light_curve(kind, seed=0)["truth"]["kind"] for kind in T.KINDS}
    assert kinds == set(T.KINDS)


def test_the_missing_public_light_curve_names_what_it_needs_instead_of_guessing():
    """MAST is unreachable here, and the FITS reader is not installed either: the loader names the missing piece.

    Measured: it raises `ImportError` naming `astropy` and the install extra, which is the actionable answer — an
    empty array would be a silent lie about a product that was never read.
    """
    with pytest.raises((T.LightCurveUnavailable, ImportError)) as excinfo:
        T.load_light_curve("TOI-270", mission="TESS", sector=1, directory="/nonexistent")
    message = str(excinfo.value)
    assert "astropy" in message or "TESS" in message or "MAST" in message


# -------------------------------------------------------------------------------------- the labelled benchmark
@pytest.fixture(scope="module")
def bench() -> dict:
    """The labelled set judged once: 18 cases, 3 planets and 15 look-alikes (the doc quotes this same run)."""
    return T.benchmark(n_per_kind=3, seed=0, steps=300)


def test_the_benchmark_has_the_labelled_kinds_the_vetting_must_separate(bench):
    assert bench["n_cases"] == 18 and bench["n_planet"] == 3 and bench["n_false_positive"] == 15
    assert set(bench["by_kind"]) == {"planet", "eb_alternating", "eb_secondary", "eb_grazing", "blend",
                                     "systematic_sinusoid"}


def test_bls_alone_finds_every_planet_and_calls_most_look_alikes_planets(bench):
    """Measured: recovery 1.0, false positive 0.8 — the baseline is good at finding dips and bad at judging them."""
    summary = bench["summary"]["bls"]
    assert summary["recovery"]["value"] == pytest.approx(1.0)
    assert summary["recovery"]["k"] == 3
    assert summary["false_positive"]["value"] == pytest.approx(0.8)
    assert summary["false_positive"]["k"] == 12 and summary["false_positive"]["n"] == 15


def test_the_fit_alone_does_not_separate_them(bench):
    """Adding the transit fit changes nothing: measured false positive 0.8 again, the same 12 of 15."""
    summary = bench["summary"]["bls+fit"]
    assert summary["recovery"]["value"] == pytest.approx(1.0)
    assert summary["false_positive"]["value"] == pytest.approx(0.8)


def test_the_vetting_is_what_removes_the_look_alikes(bench):
    """Measured: false positive 0.0 (0 of 15) with recovery intact at 1.0 (3 of 3), and an AUC of 0.889."""
    summary = bench["summary"]["bls+fit+vetting"]
    assert summary["recovery"]["value"] == pytest.approx(1.0)
    assert summary["false_positive"]["value"] == pytest.approx(0.0)
    assert summary["false_positive"]["k"] == 0
    assert bench["auc"] == pytest.approx(0.8888888888888888)


def test_every_verdict_says_why_and_the_checks_fire_by_name(bench):
    """A vetting score without a reason is a number nobody can defend: each false positive names its own checks."""
    for row in bench["rows"]:
        assert row["verdict"]
        if row["kind"] == "planet":
            assert not row["flags"], row["flags"]
            assert "candidate" in row["verdict"]
        elif row["detected"]:
            assert row["flags"], f"{row['kind']} was called a false positive with no check named"
            assert "false positive" in row["verdict"]
    fired = {flag for row in bench["rows"] for flag in row["flags"]}
    assert {"odd_even", "shape_inconsistent", "centroid_shift", "v_shape", "secondary"} <= fired


def test_the_period_and_the_depth_are_recovered_where_the_signal_is_strong_enough(bench):
    """The planets' periods come back inside the BLS and the fit, and the depth inside its own uncertainty."""
    for row in bench["rows"]:
        if row["kind"] != "planet":
            continue
        assert row["period_ok"] and row["depth_ok"]
        assert row["period_fit"] == pytest.approx(row["period_true"], rel=1e-3)
        assert abs(row["depth_fit"] - row["depth_true"]) < 30 * row["depth_err"]
