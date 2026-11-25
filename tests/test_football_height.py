"""pinneapple_systems.sports.football: the broadcast-height correction (#911), on simulated flights with a known
distortion. Fast and offline: no tracking data is downloaded."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.sports.football import ball as B
from pinneapple_systems.sports.football import height_study as H

KW = {"dt": 0.05}


def _distorted(n: int = 8, seed: int = 0, noise: float = 0.2):
    """Simulated kicks seen through the known distortion: ``(feed flights, true flights, map)``."""
    kicks = B._synthetic_kicks(n, noise=noise, fps=10.0, seed=seed, g=B.G)
    known = B.HeightMap.from_function(B.known_height_distortion, z_max=35.0)
    fed = [B.Flight(k.match_id, k.t, np.column_stack([k.xyz[:, 0], k.xyz[:, 1], known.to_feed(k.xyz[:, 2])]),
                    k.detected) for k in kicks]
    return fed, kicks, known


# -------------------------------------------------------------------------------------------------- the map itself
def test_identity_map_changes_nothing():
    hm = B.HeightMap.identity()
    z = np.array([0.0, 0.11, 3.0, 17.5])
    assert np.allclose(hm(z), z) and np.allclose(hm.to_feed(z), z)
    assert np.allclose(hm.local_scale(z), 1.0)


def test_each_map_kind_inverts_itself():
    z = np.linspace(0.0, 20.0, 41)
    hm = B.HeightMap("spline", nodes=z, values=B.known_height_distortion(z))
    assert np.allclose(hm.to_feed(hm(z)), z, atol=1e-6)
    s = B.HeightMap("scale", params=np.array([1.62]))
    assert np.allclose(s(z), 1.62 * z) and np.allclose(s.to_feed(s(z)), z)
    c = B.HeightMap("camera", params=np.array([0.62, -0.05]))
    x, y = np.zeros_like(z), np.linspace(-34.0, 34.0, len(z))
    assert np.allclose(c.to_feed(c(z, x, y), x, y), z)
    assert np.all(hm(z) >= z)                        # the known distortion expands the feed's heights


def test_the_ground_stays_on_the_ground():
    """Every kind maps a ball on the ground to a ball on the ground: the feed's zero is the physical zero."""
    z = np.linspace(0.0, 25.0, 26)
    hm = B.HeightMap("spline", nodes=B._strictly_increasing(z / 1.62), values=z)
    assert abs(float(hm(0.0))) < 1e-9


def test_a_degenerate_map_is_refused():
    good = B.HeightMap("spline", nodes=np.linspace(0, 20, 11), values=np.linspace(0, 32, 11))
    assert good.is_sane()
    flat = B.HeightMap("spline", nodes=np.linspace(0, 20, 11), values=np.zeros(11))
    assert not flat.is_sane()
    swapped = B.HeightMap("scale", params=np.array([1e4]))
    assert not swapped.is_sane()


# -------------------------------------------------------------------------------------------- the joint height fit
def test_the_joint_fit_finds_the_scale_of_a_known_distortion():
    """A pure scale error is read off the flights' apparent gravity: 1/0.62 is what the fit must return."""
    fed, _kicks, _known = _distorted(10)
    z, g = B.apparent_gravity(fed, fit_kw=KW)
    assert 5.5 < float(np.median(g)) < 6.7                     # the compressed feed reports a low gravity
    hf = B.fit_height_map(fed, kind="scale", rounds=1, fit_kw=KW)
    assert 1.4 < float(hf.height.params[0]) < 1.9
    corr = [B.fit_flight(f.t, B.correct_flight(f, hf.height).xyz, "drag_free_g", **KW) for f in fed]
    med = float(np.median([c.g for c in corr if c.success]))
    assert 8.5 < med < 11.0                                    # gravity comes back


def test_the_known_answer_recovery_is_run_and_reports_the_checks():
    r = B.height_recovery_test(n=10, kind="scale", rounds=1, noise=0.2, seed=1, workers=1, fit_kw=KW)
    assert r["n_train"] == 5 and r["n_test"] == 5
    assert r["f_rmse_m"] < 4.0
    assert r["height_rmse_after"] < r["height_rmse_before"]
    assert "curve" in r and len(r["curve"]["grid"]) == len(r["curve"]["truth"])


def test_correct_flight_only_touches_the_height():
    fed, _kicks, _known = _distorted(3)
    hm = B.HeightMap("scale", params=np.array([1.62]))
    corr = B.correct_flight(fed[0], hm)
    assert np.allclose(corr.xyz[:, :2], fed[0].xyz[:, :2])
    assert np.allclose(corr.xyz[:, 2], 1.62 * fed[0].xyz[:, 2])
    assert np.allclose(corr.t, fed[0].t)


# ------------------------------------------------------------------------------------------------- identifiability
def test_identifiability_reports_only_what_the_bound_allows():
    rng = np.random.default_rng(3)
    t = np.arange(0, 2.6, 0.05)
    truth = B.simulate([0, 0, 0.11], [18, 0, 11], t, cd=0.3)
    obs = truth + rng.normal(0, 0.1, truth.shape)
    fit = B.fit_flight(t, obs, "drag_magnus", sigma=0.1, loss="linear", **KW)
    r = B.identifiability_of(fit)
    assert r["cd_identified"] is (r["cd_std"] <= B.CR_CD_MAX)
    assert r["cl_identified"] is (r["cl_std"] <= B.CR_CL_MAX)
    assert np.isfinite(r["cd"])                                           # always reported, flagged or not
    empty = B.identifiability_of(None)
    assert not empty["cd_identified"] and not empty["cl_identified"]


def test_a_short_flight_cannot_identify_drag_or_spin():
    """The point of the check: a 1 s flight seen with the tracker's own noise says nothing about ``C_D`` or the spin.

    Both statements are measured, not assumed: the drag of a 1 s 10 Hz flight with 0.2 m of position noise lands at
    a standard deviation above the threshold, and the spin is far above it even on a 2.5 s flight -- the Magnus
    coefficient is the part a broadcast flight almost never pays for.
    """
    rng = np.random.default_rng(0)
    t = np.arange(0, 1.0, 0.1)
    P = B.simulate([0, 0, 0.11], [16, 0, 7], t, cd=0.3) + rng.normal(0, B.CR_SIGMA_FLOOR, (10, 3))
    r = B.identifiability_of(B.fit_flight(t, P, "drag_magnus", loss="linear", **KW))
    assert not r["cd_identified"]
    assert not r["cl_identified"]
    t = np.arange(0, 2.5, 0.1)
    P = B.simulate([0, 0, 0.11], [16, 0, 7], t, cd=0.3, magnus=B.BALL.k(0.15) * np.array([0.0, 1.0, 0.0]))
    P = P + rng.normal(0, B.CR_SIGMA_FLOOR, P.shape)
    long = B.identifiability_of(B.fit_flight(t, P, "drag_magnus", loss="linear", **KW))
    assert long["cd_identified"]                     # a long flight does buy the drag
    assert not long["cl_identified"]                 # and still not the spin


def test_the_bound_is_never_taken_from_an_impossibly_clean_residual():
    """A noiseless synthetic flight must not be called identified on the strength of its own zero residual.

    The Laplace covariance carries the residual scale, which is zero here, so the bound is inflated to the tracker's
    noise floor; ``sigma_effective`` records that the floor, not the residual, set the uncertainty.
    """
    t = np.arange(0, 1.0, 0.1)
    P = B.simulate([0, 0, 0.11], [16, 0, 7], t, cd=0.3)
    fit = B.fit_flight(t, P, "drag_magnus", loss="linear", **KW)
    assert fit.rmse < 1e-6
    assert fit.std("cd") < 1e-6                      # the raw covariance really is degenerate
    r = B.identifiability_of(fit)
    assert r["sigma_effective"] == pytest.approx(B.CR_SIGMA_FLOOR)
    assert r["cd_std"] > 1e-3                        # inflated to the floor, not the raw 1e-6


# ------------------------------------------------------------------------------------------- the landing/apex check
def test_landing_and_apex_runs_on_simulated_flights():
    fed, _kicks, _known = _distorted(6)
    r = H.landing_and_apex(fed, None, frac=0.5)
    assert r["n"] >= 4
    assert r["landing_err_median"] >= 0 and r["apex_err_median"] >= 0
    assert len(r["landing_errors"]) == r["n"]


def test_paired_bootstrap_of_the_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.2, 200)
    b = rng.normal(2.0, 0.2, 200)
    d = H._paired(a, b, seed=0)
    assert d["diff_median"] == pytest.approx(float(np.median(a) - np.median(b)))
    assert d["ci95"][0] < d["diff_median"] < d["ci95"][1]


def test_identifiability_summary_counts_the_fraction():
    fed, _kicks, _known = _distorted(5)
    r = H.identifiability_summary(fed, None, frac=1.0)
    assert r["n"] == 5
    assert 0.0 <= r["cd_fraction"] <= 1.0 and 0.0 <= r["cl_fraction"] <= 1.0


def test_height_band_brackets_the_curve():
    fed, _kicks, _known = _distorted(10)
    hf = B.fit_height_map(fed, kind="spline", rounds=1, fit_kw=KW)
    band = H.height_band(hf, fed, n_boot=20, seed=0)
    assert len(band["grid"]) == len(band["lo"]) == len(band["hi"])
    assert np.all(np.asarray(band["lo"]) <= np.asarray(band["hi"]) + 1e-9)
