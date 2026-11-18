"""The exact 2-D compact range and the paired PO/direct study: the circle series, PO against the exact current on the
same contour, the learned residual and the strategies at a fixed budget of exact solves."""
import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.reflector import paired
from pinneapple_simulation.electromagnetics.reflector import planar2d as P


def test_direct_solve_reproduces_the_circle_series_and_po_does_not():
    v = P.validate_circle(ka=6.0, per_wavelength=12.0, source_distance=6.0)
    assert v["n_segments"] >= 60
    assert v["max_rel_error_mom"] < 1e-2                        # the reference is exact to a few per mille
    assert v["max_rel_error_po"] > 1.0                          # PO is wrong by O(100 %) at the shadow point
    assert v["mean_abs_db_mom"] < 0.1


def test_po_and_the_exact_current_disagree_in_the_quiet_zone():
    rng = P.Range2D()
    po = P.po_metrics(rng)
    dr = P.direct_metrics(rng)
    assert po["n_segments"] == dr["n_segments"]
    for k in paired.METRICS:
        assert np.isfinite(po[k]) and np.isfinite(dr[k])
    assert dr["po_distance"] > 0.05                             # the densities are not the same current
    assert dr["condition"] < 1e6                                # the integral equation is well posed here
    # the ripple is the quantity PO gets closest to; the usable width is where PO moves designs
    assert abs(dr["ripple_pp_dB"] - po["ripple_pp_dB"]) / dr["ripple_pp_dB"] < 0.15
    assert abs(dr["usable_width"] - po["usable_width"]) / dr["usable_width"] > 0.15


def test_design_space_keeps_the_quiet_zone_and_scales_the_aperture():
    base = P.Range2D()
    ds = P.design_space(base, sizes=(0.6, 1.4), qs=(4.0,), profiles=P.EDGE_FAMILY[:2])
    assert len(ds) == 4
    for d, rng in ds:
        assert rng.qz_z == base.qz_z and rng.qz_width == base.qz_width
        assert rng.section.roll == pytest.approx(d.roll_factor * base.roll_reference)
        half = 0.5 * (rng.section.y_high - rng.section.y_low)
        assert half == pytest.approx(0.5 * (base.section.y_high - base.section.y_low) * d.size)
        assert d.tag.startswith(f"a{d.size:g}_q{d.q:g}")


def _synthetic_rows(n=60, seed=0):
    """Rows shaped like the paired checkpoint, from a smooth "truth" plus noise, so the study code can be exercised
    without solving 63 designs."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 5))
    X[:, 0] = rng.choice([0.6, 0.8, 1.0, 1.2, 1.4], n)
    X[:, 1] = rng.choice([1.0, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0], n)
    X[:, 2] = rng.choice([0.6, 1.0, 1.4], n)
    X[:, 3] = rng.choice([0.4, 0.7], n)
    X[:, 4] = rng.choice([0.7, 1.0, 1.4], n)
    usable = 0.05 + 0.3 * X[:, 0] / 1.4 * (1 - 0.4 * X[:, 3])
    amp = 2.0 + 6.0 * X[:, 1] / 23.0
    rows = []
    for i, (s, q, rf, bl, asp) in enumerate(X):
        d = P.Design(float(s), float(q), float(rf), float(bl), float(asp))
        bias = 0.02 * (s - 1.0) + 0.01 * (bl - 0.55)            # a systematic, learnable residual
        rows.append({"tag": d.tag, **d.inputs(), "n_segments": 200,
                     "t_po_s": 0.02, "t_direct_s": 0.06, "mom_s": 0.03, "condition": 100.0, "po_distance": 0.3,
                     "po_usable_width": float(usable[i]), "direct_usable_width": float(usable[i] + bias),
                     "po_amp_pp_dB": float(amp[i]), "direct_amp_pp_dB": float(amp[i])})
    return rows


def test_residual_model_learns_a_smooth_residual():
    rows = _synthetic_rows()
    for r in rows:
        r["residual_usable_width"] = r["direct_usable_width"] - r["po_usable_width"]
    v = paired.leave_out_validation(rows, "usable_width", folds=4, seeds=(0,))
    assert v["random"]["fraction_removed"] > 0.5                # the systematic part is inside the design space
    assert v["random"]["rmse"] < v["po_rmse"]
    assert len(v["rows"]) == len(rows)


def test_the_strategies_spend_the_same_budget_and_never_beat_the_oracle():
    rows = _synthetic_rows()
    res = paired.compare_strategies(rows, n_direct=8, seeds=(0, 1))
    assert res["oracle"] >= res["po_best"]
    for name in ("direct", "po", "multi", "sequential"):
        assert res[f"{name}_mean"] <= res["oracle"] + 1e-12
        assert res[f"{name}_percent_mean"] <= 100.0 + 1e-9
    for run in res["runs"]:
        for name in ("direct", "po", "multi", "sequential"):
            assert run[name]["n"] == 8
    # PO is cheap information, so ranking by it must not be worse than sampling blindly on this space
    assert res["po_mean"] >= res["direct_mean"]


def test_cost_summary_reports_the_measured_ratio():
    rows = _synthetic_rows()
    c = paired.cost_summary(rows)
    assert c["ratio_median"] == pytest.approx(0.06 / 0.02, rel=1e-6)
    assert c["n_designs"] == len(rows)
    assert c["segments"]["median"] == 200


def _factorial_rows(residual=lambda d: 0.02 * (d.size - 1.0)):
    """A full factorial (size x feed) grid at two edge profiles, shaped like the paired checkpoint."""
    rows = []
    for s in (0.6, 0.8, 1.0, 1.2, 1.4):
        for q in (1.0, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0):
            for rf, bl, asp in ((0.6, 0.7, 1.0), (1.4, 0.4, 0.7)):
                d = P.Design(s, q, rf, bl, asp)
                uw = 0.05 + 0.35 * (s - 0.6) / 0.8
                r = residual(d)
                rows.append({"tag": d.tag, **d.inputs(), "n_segments": 200, "t_po_s": 0.02, "t_direct_s": 0.06,
                             "mom_s": 0.03, "condition": 100.0, "po_distance": 0.3,
                             "po_usable_width": uw, "direct_usable_width": uw + r,
                             "po_amp_pp_dB": 3.0, "direct_amp_pp_dB": 3.0,
                             "po_phase_ripple_pp_deg": 5.0, "direct_phase_ripple_pp_deg": 5.0,
                             "residual_usable_width": r})
    return rows


def test_the_objective_holds_the_phase_ripple_only_when_asked():
    row = {"direct_usable_width": 0.40, "direct_amp_pp_dB": 2.0, "direct_phase_ripple_pp_deg": 45.0}
    assert paired.objective(row, "direct") == pytest.approx(0.40)             # amplitude inside, phase not asked
    assert paired.objective(row, "direct", phase_limit=20.0) < 0.40           # the phase term now bites
    flat = {**row, "direct_amp_pp_dB": 6.0, "direct_phase_ripple_pp_deg": 4.0}
    assert paired.objective(flat, "direct", phase_limit=20.0) == pytest.approx(0.40 - 0.1 * 3.0)


def test_coverage_tells_the_levers_apart():
    size_only = _factorial_rows(lambda d: 0.02 * (d.size - 1.0) + 0.003 * (d.q - 5.0) / 10)
    c = paired.coverage_analysis(size_only, "usable_width", folds=3, seeds=(0, 1))
    assert c["removed_all"] > 0.5                                             # the residual is learnable
    assert c["profile_share"] < 0.25                                          # and it lives in size and illumination
    profile_only = _factorial_rows(lambda d: 0.02 * (d.aspect - 1.0) + 0.02 * (d.roll_factor - 1.0))
    c2 = paired.coverage_analysis(profile_only, "usable_width", folds=3, seeds=(0, 1))
    assert c2["profile_share"] > 0.5                                          # this one needs the edge profile


def test_the_surface_is_one_cell_per_size_and_illumination():
    rows = _factorial_rows()
    s = paired.design_surface(rows, "usable_width", profile=(0.6, 0.7, 1.0))
    assert s["sizes"] == [0.6, 0.8, 1.0, 1.2, 1.4] and s["qs"] == [1.0, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0]
    assert np.array(s["direct"]).shape == (5, 7)
    assert np.allclose(np.array(s["difference"]), np.array(s["direct"]) - np.array(s["po"]))
    with pytest.raises(KeyError):
        paired.design_surface(rows, "usable_width", profile=(0.9, 0.9, 0.9))


def test_the_quiet_zone_line_is_centred_on_the_section_and_reaches_the_aperture():
    rng = P.Range2D()
    line = rng.qz_line()
    assert line.shape == (rng.qz_points, 2)
    assert np.allclose(line[:, 1], rng.qz_z)                                  # all of it on the plane z = qz_z
    assert line[:, 0].max() - line[:, 0].min() == pytest.approx(rng.qz_width)
    centre = 0.5 * (rng.section.y_low + rng.section.y_high)
    assert line[:, 0].mean() == pytest.approx(centre)                         # default centre: the section's own
    shifted = P.Range2D(qz_centre_y=centre + 0.1)
    assert shifted.qz_line()[:, 0].mean() == pytest.approx(centre + 0.1)


def test_near_field_map_pairs_the_two_solvers_on_one_grid():
    rng = P.Range2D(per_wavelength=8.0)
    m = P.near_field_map(rng, n_y=40, n_z=32, grid=rng.grid)
    assert m["po_db"].shape == m["direct_db"].shape == m["diff_db"].shape == (40, 32)
    assert len(m["y"]) == 40 and len(m["z"]) == 32
    assert m["qz_z"] == pytest.approx(P.Range2D().qz_z)
    assert len(m["qz_s"]) == len(m["cut_po_db"]) == len(m["cut_direct_db"]) > 8
    pct = m["diff_db_percentiles"]
    assert set(pct) == {"50", "90", "99"}                                    # string keys: it is JSON-ready
    assert pct["50"] < pct["90"] < pct["99"]
    # PO is close to the exact field on the illuminated body and wrong at the rim: the map must show both
    assert np.isfinite(m["po_db"]).all() and np.isfinite(m["direct_db"]).all()
    assert max(m["po_db"].max(), m["direct_db"].max()) == pytest.approx(0.0, abs=1e-9)   # one shared scale
    assert m["diff_db"].min() >= -20.0 and m["diff_db"].max() <= 10.0         # clipped to the requested window
    assert m["n_segments"] == P.po_metrics(rng)["n_segments"]
    assert m["seconds"] > 0
    # a coarser contour is cheaper and still solves the same problem
    coarse = P.near_field_map(P.Range2D(per_wavelength=6.0), n_y=20, n_z=16)
    assert coarse["n_segments"] < m["n_segments"]
