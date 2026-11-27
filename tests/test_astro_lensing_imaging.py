"""Black-hole imaging (issue #390, part 1) and strong gravitational lensing (part 2), with the numbers measured.

Both halves are inverse problems whose answer depends on a forward model and a prior, so the tests below check
three kinds of statement, in that order: the *forward* physics against a closed form or an independent computation
(analytic visibilities against a direct transform, the ray-traced shadow against ``2 sqrt(27) GM/c^2 D``, an SIE
against the point-mass identity); the *inverse* answer against the parameters that were injected (a closure test,
since the data are simulated -- the EHT and HST releases are not bundled); and the *honesty* of the reporting --
that the prior experiment says how far the prior moves each feature, that the data loader fails naming what is
missing, and that the checks report a lens the caustics do not agree with instead of passing it.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.astro import eht_imaging as ei
from pinneapple_physics.astro import lens_model as lm

MUAS = 4.84813681109536e-12
M87 = ei.EHT_OBJECTS["M87*"]


@pytest.fixture(autouse=True)
def _threads():
    import torch
    torch.set_num_threads(1)


# ==================================================================================== part 1: black-hole imaging
def test_shadow_diameter_of_the_published_mass_and_distance():
    """2 sqrt(27) GM/c^2 D for M87* is 39.7 uas, and it sits 0.8 sigma below the measured 42 +- 3 uas ring.

    This is the one number in the imaging half that involves nothing of this module's fitting: it is the closed form
    of the shadow evaluated at the published mass and distance, and the comparison with the published ring is a
    statement about those published values, not about any code here.
    """
    shadow = ei.shadow_angular_diameter_uas(M87["mass_msun"], M87["distance_pc"] / 1e6)
    assert shadow == pytest.approx(39.688, abs=0.01)
    sigma = (M87["ring_uas"] - shadow) / M87["ring_err_uas"]
    assert sigma == pytest.approx(0.77, abs=0.02)          # the ring is bigger than the shadow, inside one sigma
    # the module wraps the lensing closed form rather than restating it
    from pinneapple_physics.astro.lensing import shadow_diameter_rad
    direct = shadow_diameter_rad(M87["mass_msun"] * 1.98840987e30, M87["distance_pc"] * 3.0856775814913673e16)
    assert shadow == pytest.approx(direct / 4.84813681109536e-12, rel=1e-12)


def test_ring_diameter_from_the_first_visibility_null():
    """A 42 uas ring has its first J0 null at 3.76 Glambda, inside the coverage the array reaches (7.9 Glambda)."""
    null = ei.visibility_null_baseline_gl(42.0)
    assert null == pytest.approx(3.759, abs=0.005)
    uv = ei._dataset("M87*")
    assert null < uv.max_baseline_gl                          # the feature is measurable, not extrapolated
    # the null scales as 1/diameter: halving the ring doubles the baseline
    assert ei.visibility_null_baseline_gl(21.0) == pytest.approx(2.0 * null, rel=1e-9)


def test_uv_coverage_is_the_eht_array_at_230_ghz():
    """Earth-rotation synthesis of the eight 2017 stations: 327 visibilities on 28 pairs out to 7.9 Glambda, all of
    them above the horizon."""
    uv = ei.uv_coverage(M87["dec_deg"])
    assert uv.n_vis == 327
    assert uv.n_scans == 24
    assert len(uv.stations) == 8
    assert uv.max_baseline_gl == pytest.approx(7.98, abs=0.05)
    assert np.all(np.isfinite(uv.u)) and np.all(np.isfinite(uv.v))
    assert np.all(uv.sigma > 0)
    # a2 < a1 never happens, and every pair index is a real station
    assert uv.antenna1.max() < len(uv.stations) and uv.antenna2.max() < len(uv.stations)
    assert np.all(uv.antenna1 < uv.antenna2)
    # the source is southern: Sgr A* is seen by a different set of pairs than M87*
    sgr = ei.uv_coverage(ei.EHT_OBJECTS["Sgr A*"]["dec_deg"])
    assert sgr.n_vis != uv.n_vis


def test_uv_coverage_refuses_a_source_that_never_rises():
    """A single northern station cannot see a dec -80 source: the horizon is not negotiable, and the failure names
    the elevation cut instead of returning an empty array."""
    with pytest.raises(ValueError, match="above"):
        ei.uv_coverage(-80.0, stations={"SMT": ei.EHT_STATIONS["SMT"]}, min_elevation_deg=10.0)


def test_closed_form_visibilities_match_the_direct_transform():
    """The Bessel form of the crescent against the discrete transform of the rendered image: 2e-7 of the peak.

    The two are computed in completely different ways (a 1-D Gauss-Legendre radial quadrature against a 2-D sum over
    a rendered image), so their agreement is the check that the forward model -- and the sign of the asymmetry term
    -- is right.
    """
    params = [42.0, 8.0, 0.4, 90.0, 0.6]
    uv = ei._dataset("M87*")
    image, pixel = ei.crescent_flux_map(params, n=192)
    direct = ei.visibilities_from_image(image / params[4], pixel, uv.u, uv.v) * params[4]
    closed = ei.crescent_visibilities(params, uv.u, uv.v)
    assert np.max(np.abs(direct - closed)) / np.max(np.abs(closed)) < 1e-5


def test_symmetric_ring_visibility_is_real_and_vanishes_at_the_null():
    """With no asymmetry the visibility is real, equals the flux at zero baseline, and falls to under a tenth of it
    at the first Bessel zero; the asymmetry is what fills that null in."""
    symmetric = [42.0, 8.0, 0.0, 90.0, 0.6]
    at_zero = ei.crescent_visibilities(symmetric, np.array([0.0]), np.array([0.0]))[0]
    assert at_zero.real == pytest.approx(0.6, rel=1e-9)
    assert abs(at_zero.imag) < 1e-12
    null = ei.crescent_visibilities(symmetric, np.array([ei.visibility_null_baseline_gl(42.0)]), np.array([0.0]))[0]
    assert abs(null) < 0.1 * 0.6
    asymmetric = ei.crescent_visibilities([42.0, 8.0, 0.4, 90.0, 0.6],
                                         np.array([1.0, 1.0]), np.array([0.0, 1.0]))
    assert abs(asymmetric[0].imag) < 1e-12                    # the bright side is at 90 deg: the u axis sees it real
    assert abs(asymmetric[1].imag) > 0.01                     # the v axis, parallel to it, does not


def test_reproduce_the_published_ring_diameter():
    """The closure test: the machinery recovers the 42 uas it injected, an order of magnitude tighter than the
    published error -- which is the statement about the synthetic data, not about M87*."""
    rep = ei.reproduce_ring_diameter("M87*", n_realizations=12, seed=0)
    assert rep.recovered_uas == pytest.approx(42.0, abs=0.2)
    assert rep.recovered_err_uas < 0.05
    assert abs(rep.publication_sigma) < 0.1
    assert rep.predicted_uas == pytest.approx(39.688, abs=0.01)
    assert rep.prediction_sigma > 3.0                         # the recovered 42 is not the shadow: they are different
    assert rep.conformal_half_width_uas > rep.recovered_err_uas
    table = rep.table()
    assert len(table) == 4
    assert {row["provenance"].split(":")[0] for row in table} >= {"recovered here from 12 synthetic crescent data sets", "2 sqrt(27) GM/c^2 D from the published mass and distance (no fit)"}


def test_conformal_interval_covers_a_gaussian_without_assuming_one():
    """The distribution-free interval is the 1-up-to-rounding normal quantile: 1.87 sd for 40 samples at 95 %."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, 40)
    half = ei.diameter_interval(samples, 0.0, alpha=0.05)
    assert half / samples.std(ddof=1) == pytest.approx(1.87, abs=0.05)


def test_sgr_a_star_reproduces_its_own_published_ring():
    """The second published object goes through the same machinery: 51.8 uas ring, 50.35 uas shadow, recovered from
    its own (de)clination and the same array."""
    rep = ei.reproduce_ring_diameter("Sgr A*", n_realizations=4, seed=0)
    assert rep.recovered_uas == pytest.approx(51.8, abs=0.6)
    assert rep.published_uas == 51.8
    assert rep.predicted_uas == pytest.approx(50.35, abs=0.05)
    assert abs(rep.publication_sigma) < 0.5


def test_prior_does_not_move_any_feature_at_eht_sensitivity():
    """The measurement the issue asks for, at the data quality the campaign had: the physics prior (shadow diameter
    +- 4 uas, ring width +- 4 uas) moves every feature by less than a tenth of the feature's own uncertainty, so all
    five are data-determined -- and the paired design makes even that small move statistically visible."""
    study = ei.prior_dependence_study("M87*", n_realizations=8, seed=0)
    assert study.truth_kind == "crescent"
    assert study.median_snr > 3.0
    verdicts = study.verdicts()
    assert set(verdicts.values()) == {"data-determined"}
    for feature in study.features:
        assert feature.moved < 0.15, feature.name
        assert feature.shrink < 1.1, feature.name
    diameter = study.features[0]
    assert diameter.name == "diameter_uas"
    assert abs(diameter.shift) < 0.02                         # uas
    assert diameter.off_mean == pytest.approx(42.0, abs=0.3)


def test_degrading_the_data_is_what_makes_the_prior_matter():
    """A hundredth of the integration time (100x the thermal noise): the diameter error triples from 0.07 to ~3.5 uas,
    the prior starts to buy precision (a factor of two) and to move the answer -- but it still does not drive it."""
    good = ei.prior_dependence_study("M87*", n_realizations=8, sigma_scale=1.0, seed=0)
    poor = ei.prior_dependence_study("M87*", n_realizations=8, sigma_scale=100.0, seed=0)
    assert poor.median_snr < 0.5 < good.median_snr
    d_good = good.features[0]
    d_poor = poor.features[0]
    assert d_poor.off_sd > 5.0 * d_good.off_sd
    assert d_poor.shrink > 1.5                                # the prior tightened the error
    assert abs(d_poor.shift) > 5.0 * abs(d_good.shift)        # and moved the central value
    assert d_poor.verdict in {"prior-influenced", "prior-driven"}


def test_amplitudes_alone_cannot_pick_the_side_of_the_asymmetry():
    """With the phases dropped the fit lands on the reflected crescent: pa = -85 +- 8 deg where the truth is +90.

    An asymmetric ring's visibility amplitude is invariant to the sign of the asymmetry, so the data do not choose
    which side is bright; the formal error of 8 deg does not show the 175 deg it is wrong by. That is a feature the
    prior cannot rescue either -- nothing in the prior used here says anything about the direction -- and it is the
    concrete case where "which features survive without the prior" has the answer "not this one".
    """
    study = ei.prior_dependence_study("M87*", n_realizations=6, phases=False, seed=0)
    by_name = {f.name: f for f in study.features}
    assert abs(by_name["pa_deg"].off_mean - 90.0) > 150.0
    assert by_name["pa_deg"].off_sd < 30.0                    # tightly, and wrongly, constrained
    assert by_name["asymmetry"].off_mean > 0.3                # the amplitude survives; the direction does not
    assert all(f.verdict == "data-determined" for f in study.features)


def test_ray_traced_shadow_has_the_size_the_closed_form_predicts():
    """The reused null-geodesic tracer: the captured rays fill a disc of radius 19.3 uas where the analytic shadow
    radius is 19.84 uas (one 1.09 uas pixel of the render)."""
    traced = ei.raytraced_ring("M87*", n=96)
    assert traced["shadow_uas"] == pytest.approx(39.688, abs=0.01)
    captured = traced["trace"]["captured"].reshape(traced["image"].shape)
    size = traced["image"].shape[0]
    grid = (np.arange(size) - (size - 1) / 2.0) * traced["pixel_uas"]
    x, y = np.meshgrid(grid, grid)
    radius = np.hypot(x, y)
    assert radius[captured].max() == pytest.approx(traced["shadow_uas"] / 2.0, abs=1.3 * traced["pixel_uas"])
    assert traced["image"][captured].sum() == 0.0             # nothing shines from inside the shadow


def test_the_ring_of_a_ray_traced_disc_is_not_the_shadow():
    """The image a disc makes has its ring well outside the shadow: 69 uas against 39.7 uas for an annulus from 6 to
    8 GM/c^2. The two are different quantities, which is exactly why applying the shadow as a prior to an image
    feature is a modelling assumption."""
    traced = ei.raytraced_ring("M87*", n=96)
    result = ei.image_ring_diameter(traced["image"], traced["pixel_uas"], flux_jy=0.6)
    assert result["diameter_uas"] > 1.5 * traced["shadow_uas"]
    assert result["mean_diameter_uas"] > 1.5 * traced["shadow_uas"]
    assert 0.05 < abs(result["asymmetry"]) < 0.5             # beaming and lensing give a mild crescent
    assert result["fit"].values["flux_jy"] == pytest.approx(0.6, abs=0.02)


def test_eht_loader_fails_naming_release_file_and_size():
    """No release is bundled or downloaded: the loader names what it wants and refuses."""
    from pinneapple_core.errors import NotConfigured

    spec = ei.data_requirements()["M87* 2017"]
    with pytest.raises(NotConfigured) as excinfo:
        ei.load_visibility_data("M87* 2017")
    message = str(excinfo.value)
    assert "SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits" in message
    assert "uvfits" in spec["file"] and "MB" in spec["size"]
    with pytest.raises(NotConfigured, match="unknown EHT data release"):
        ei.load_visibility_data("VLA 1994")
    with pytest.raises(NotConfigured, match="no uvfits file"):
        ei.load_visibility_data("M87* 2017", path="/nonexistent/sr1.uvfits")


# ==================================================================================== part 2: strong lensing
def test_declared_lenses_use_the_published_isothermal_einstein_radius():
    """Each generated lens sits at the SIS Einstein radius of its stated velocity dispersion and redshifts, which is
    the published formula ``4 pi (sigma/c)^2 D_ls / D_s`` -- 0.834, 0.757 and 5.32 arcsec for 250, 290 and 550 km/s.
    """
    expected = {"galaxy-scale": (250.0, 0.30, 1.20), "quad-with-shear": (290.0, 0.45, 2.00),
                "group-scale": (550.0, 0.20, 1.50)}
    for name, (sigma, zl, zs) in expected.items():
        lens = lm.LENSES[name]
        assert (lens.sigma_kms, lens.z_lens, lens.z_source) == (sigma, zl, zs)
        assert lens.lens.theta_e == pytest.approx(lm.isothermal_einstein_arcsec(sigma, zl, zs), rel=1e-12)
        assert lens.theta_e_formula_arcsec == pytest.approx(lens.lens.theta_e, rel=1e-12)
        assert lens.image == "generated"                      # not observed, and the record says so
    assert lm.LENSES["galaxy-scale"].lens.theta_e == pytest.approx(0.8344, abs=1e-4)
    assert lm.LENSES["group-scale"].lens.theta_e == pytest.approx(5.3225, abs=1e-3)


def test_isothermal_deflection_equals_the_point_mass_with_the_projected_mass_inside():
    """An exact identity: the SIS deflection, the point-mass deflection of the projected mass inside theta_E and the
    point-mass Einstein radius all agree to round-off."""
    row = lm.point_mass_crosscheck(250.0, 0.30, 1.20)
    assert row["theta_relative_difference"] < 1e-10
    assert row["deflection_relative_difference"] < 1e-10
    assert row["theta_e_arcsec"] == pytest.approx(0.8344, abs=1e-4)
    assert row["projected_mass_msun"] > 1e11                   # ~2e11 Msun inside the Einstein ring


@pytest.mark.parametrize("name", ["galaxy-scale", "quad-with-shear", "group-scale"])
def test_each_declared_lens_is_recovered_from_its_generated_image(name):
    """Fit every declared lens back out of its own rendered image: the ring radius, the axis ratio, the position
    angle and both shear components come back to 1e-6, and the residual is round-off -- because the image is a
    rendering of the same model, which is what "generated rather than observed" means."""
    declared = lm.LENSES[name]
    image, pixel = lm.lens_image(declared.lens, declared.source, n=48)
    fit = lm.fit_lens_image(image, pixel)
    assert fit.rms_residual < 1e-8
    assert fit.recovered["theta_e"] == pytest.approx(declared.lens.theta_e, abs=1e-5)
    assert fit.recovered["q"] == pytest.approx(declared.lens.q, abs=1e-5)
    assert fit.recovered["pa_deg"] == pytest.approx(declared.lens.pa_deg, abs=1e-3)
    assert fit.recovered["gamma1"] == pytest.approx(declared.lens.gamma1, abs=1e-5)
    assert fit.recovered["gamma2"] == pytest.approx(declared.lens.gamma2, abs=1e-5)


def test_declared_lens_passes_positive_convergence_and_caustic_multiplicity():
    """The physics checks: kappa > 0 everywhere it is sampled, every image solves the lens equation to 1e-6 arcsec,
    and a source inside the tangential caustic is seen four times while one outside is seen twice."""
    lens = lm.LENSES["galaxy-scale"].lens
    report = lm.consistency_report(lens, n_directions=4)
    assert report["min_convergence"] > 0.1 and report["convergence_positive"]
    assert report["solves_lens_equation"] and report["multiplicity_ok"] and report["all_ok"]
    assert report["inner_counts"] == [4] and report["outer_counts"] == [2]
    assert report["n_mismatch"] == 0
    assert all(row["source_error_arcsec"] < 1e-6 for row in report["rows"])
    # and the two groups really are on opposite sides of the caustic, by more than a sampling step
    inner = [row["distance_to_caustic_arcsec"] for row in report["rows"] if row["group"] == "inner"]
    outer = [row["distance_to_caustic_arcsec"] for row in report["rows"] if row["group"] == "outer"]
    assert min(outer) > max(inner)


def test_shear_adds_no_convergence_and_the_map_stays_physical():
    """The external shear is tidal: it changes the deflection and the magnification and leaves kappa untouched."""
    plaque = lm.SIELens(theta_e=1.0, q=0.7, pa_deg=20.0, gamma1=0.0, gamma2=0.0)
    sheared = lm.SIELens(theta_e=1.0, q=0.7, pa_deg=20.0, gamma1=0.12, gamma2=-0.08)
    grid = np.meshgrid(np.linspace(-2.0, 2.0, 41), np.linspace(-2.0, 2.0, 41))
    assert np.allclose(plaque.convergence(*grid), sheared.convergence(*grid))
    assert np.all(sheared.convergence(*grid) > 0)
    ax, ay = plaque.deflection(*grid)
    bx, by = sheared.deflection(*grid)
    assert not np.allclose(ax, bx)


def test_a_near_spherical_lens_reports_the_caustic_limitation_instead_of_passing():
    """The multiplicity check compares the image count with the ellipsoidal caustic expectation (4 inside, 2
    outside). For q = 0.95 the tangential caustic reaches only 0.051 theta_E, so the probe at 0.02 is inside it (four
    images) and the one at 0.05 is outside (two) -- the count follows the caustic, and the convention's "four" is
    reported as a mismatch rather than assumed.
    """
    lens = lm.SIELens(theta_e=1.0, q=0.95, pa_deg=0.0)
    caustic = lens.tangential_caustic(n=181)
    reach = float(np.max(np.hypot(caustic[:, 0], caustic[:, 1])))
    assert reach == pytest.approx(0.0515, abs=0.002)
    report = lm.consistency_report(lens, n_directions=2)
    assert report["convergence_positive"]
    assert report["inner_counts"] == [2, 4]                   # measured across the caustic, not the expected four
    assert not report["multiplicity_ok"]                      # and the mismatch is reported
    assert {row["radius_theta_e"] for row in report["mismatch"]} == {0.05}


def test_multiplicity_follows_the_caustic_across_it():
    """Along one direction the image count drops from four to two as the source crosses the tangential caustic, and
    the crossing is where the caustic is (0.78 theta_E for this lens); further out, beyond 1.3 theta_E, the shear
    leaves a single image."""
    lens = lm.LENSES["quad-with-shear"].lens
    caustic = lens.tangential_caustic(n=181)
    reach = float(np.max(np.hypot(caustic[:, 0], caustic[:, 1])))
    inside = lens.multiplicity((0.15 * lens.theta_e, 0.0))
    outside = lens.multiplicity((1.1 * lens.theta_e, 0.0))
    far = lens.multiplicity((1.6 * lens.theta_e, 0.0))
    assert inside == 4 and outside == 2 and far == 1
    assert 0.15 * lens.theta_e < reach < 1.1 * lens.theta_e


def test_lens_loader_fails_naming_release_and_columns():
    """No HST, Euclid or SLACS table is bundled: the loader names the release, the columns and the size."""
    from pinneapple_core.errors import NotConfigured

    spec = lm.data_requirements()["SLACS"]
    with pytest.raises(NotConfigured) as excinfo:
        lm.load_lens_sample("SLACS")
    message = str(excinfo.value)
    assert "theta_E_arcsec" in message and "sigma_kms" in message
    assert "SLACS" in spec["release"]
    with pytest.raises(NotConfigured, match="unknown lens sample"):
        lm.load_lens_sample("COSMOS")
