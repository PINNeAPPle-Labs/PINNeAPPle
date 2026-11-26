"""Hurricane pieces that need neither the network nor a data file: HURDAT2 parsing, the continuity vortex tracker,
the wind-pressure relation, the errors against a best track, the horizons and the globe scene."""
import numpy as np
import pytest

from pinneapple_physics.weather.cyclones import (
    AH_REFERENCE,
    BestTrack,
    StormErrors,
    StormTrack,
    atkinson_holliday_wind,
    great_circle_km,
    parse_hurdat2,
    skill_by_lead,
    storm_horizons,
    track_storm,
    verify_storm,
    vortex_centre,
)

HURDAT2 = """\
AL012024,                 TEST,      3,
20240926, 1200,  , TD, 12.0N,  60.0W,  30, 1006,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   60
20240927, 0000,  , HU, 14.0N,  64.0W,  75, -99,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   60
20240927, 1200, L, HU, 16.5S,  70.0E, 100,  950,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   60
AL022024,                  OTHER,      1,
20240928, 0600,  , TS,  8.0N,  20.0W,  40, 1000,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   60
"""

LAT = np.arange(40.0, 4.0, -0.25)                 # a 0.25 degree grid over the basin, descending like ERA5 cropped
LON = np.arange(270.0, 330.0, 0.25)


def gaussian_low(lat, lon, centre, depth=45.0, radius_deg=2.5, base=1012.0):
    """A pressure field with one Gaussian low: what a smoothed hurricane looks like in a reanalysis."""
    d2 = (lat[:, None] - centre[0]) ** 2 + (lon[None, :] - centre[1]) ** 2
    return base - depth * np.exp(-d2 / (2 * radius_deg ** 2))


def test_parse_hurdat2_reads_headers_records_and_hemispheres():
    storms = parse_hurdat2(HURDAT2)
    assert list(storms) == ["AL012024", "AL022024"]
    t = storms["AL012024"]
    assert len(t) == 3 and t.name == "TEST" and t.basin == "North Atlantic"
    assert list(t.times.astype("datetime64[h]").astype(str)) == ["2024-09-26T12", "2024-09-27T00", "2024-09-27T12"]
    assert np.allclose(t.lat, [12.0, 14.0, -16.5]) and np.allclose(t.lon, [-60.0, -64.0, 70.0])
    assert np.allclose(t.wind_kt, [30, 75, 100])
    assert np.isnan(t.pressure_hpa[1]) and t.pressure_hpa[2] == 950      # -99 is the file's missing flag
    assert list(t.status) == ["TD", "HU", "HU"] and list(t.record) == ["", "", "L"]
    assert t.index("2024-09-27T00") == 1 and t.index("2024-09-27T03") is None
    assert len(storms["AL022024"]) == 1


def test_parse_hurdat2_refuses_a_line_that_is_not_a_header():
    with pytest.raises(ValueError, match="not a HURDAT2 header"):
        parse_hurdat2("20240926, 1200,  , TD, 12.0N, 60.0W, 30, 1006\n")


def test_best_track_interpolates_between_records_and_stops_at_the_ends():
    t = parse_hurdat2(HURDAT2)["AL012024"]
    la, lo, wind, pres = t.interpolate(np.array(["2024-09-26T18", "2024-09-27T06", "2024-09-28T00"],
                                                dtype="datetime64[h]"))
    assert np.allclose(la[:2], [13.0, -1.25]) and np.allclose(lo[:2], [-62.0, 3.0])
    # linear in time, in the units the file publishes: averaging 14.0N with the 16.5S record gives -1.25, and -64W
    # with 70E gives 3.0E; the third time is outside the storm's life
    assert np.allclose(wind[:2], [52.5, 87.5])                                          # pressure NaN carried over
    assert np.isnan(pres[:2]).all() and np.isnan(la[2])                                 # outside the storm's life


def test_great_circle_matches_the_globe():
    assert abs(float(great_circle_km(0, 0, 1, 0)) - 111.19) < 0.05
    assert abs(float(great_circle_km(0, 0, 0, 90)) - 0.5 * np.pi * 6371.0) < 1.0     # a quarter of the equator
    assert float(great_circle_km(10, 350, 10, 10)) == pytest.approx(float(great_circle_km(10, -10, 10, 10)), rel=1e-12)


def test_atkinson_holliday_is_the_published_relation():
    assert float(atkinson_holliday_wind(950.0)) == pytest.approx(6.7 * 60 ** 0.644, rel=1e-12)
    assert float(atkinson_holliday_wind(950.0)) == pytest.approx(93.6, abs=0.2)          # a category 2-3 equivalent
    assert float(atkinson_holliday_wind(1010.0)) == 0.0 and float(atkinson_holliday_wind(1020.0)) == 0.0
    assert atkinson_holliday_wind(np.array([1000.0, 960.0])).shape == (2,)
    assert "Atkinson" in AH_REFERENCE


def test_vortex_centre_refines_below_the_grid():
    field = gaussian_low(LAT, LON, (23.37, 283.11))          # between cells on purpose
    guess = (23.0, 283.0)
    c = vortex_centre(field, LAT, LON, guess, radius_km=300.0)
    assert c is not None
    assert abs(c.lat - 23.37) < 0.08 and abs(c.lon - 283.11) < 0.08      # better than a third of a cell
    assert abs(c.pressure_hpa - (1012.0 - 45.0)) < 0.2
    assert vortex_centre(field, LAT, LON, (60.0, 275.0), radius_km=300.0) is None    # the disc is off the grid
    assert vortex_centre(field, LAT, LON, (35.0, 275.0), radius_km=300.0,
                         deepest_hpa=1008.0) is None                                 # only the tail is inside: no storm
    assert vortex_centre(np.full_like(field, 1012.0), LAT, LON, guess, radius_km=300.0, deepest_hpa=1008.0) is None


def test_vortex_centre_ignores_the_nan_mask():
    field = gaussian_low(LAT, LON, (23.0, 283.0))
    field[:, :40] = np.nan                                    # 270-280E masked: 40 columns of the 240
    c = vortex_centre(field, LAT, LON, (23.0, 283.0), radius_km=300.0)
    assert c is not None and abs(c.lat - 23.0) < 0.05


def test_track_storm_follows_a_moving_low_and_reports_the_frame_it_loses_it():
    centres = [(15.0 + 1.5 * k, 280.0 + 1.0 * k) for k in range(8)]
    fields = np.stack([gaussian_low(LAT, LON, c) for c in centres])
    fields[4] = 1012.0                                        # a frame with no storm at all: the grid lost it
    times = np.datetime64("2024-09-26T00", "h") + np.arange(8) * np.timedelta64(6, "h")
    track = track_storm(fields, LAT, LON, times, start=centres[0], radius_km=300.0)
    assert track.found.tolist() == [True, True, True, True, False, True, True, True]
    # frame 5 is re-acquired from the stale guess of frame 3, not from the truth: the storm moved 333 km in the two
    # frames, wider than the 300 km search radius, so the deepest cell the disc reaches is 21.75N 284.5E, not 22.5/285
    assert np.allclose(track.lat[[0, 3, 5, 7]], [15.0, 19.5, 21.75, 25.5], atol=0.08)
    assert np.allclose(track.lon[[0, 3, 5, 7]], [280.0, 283.0, 284.5, 287.0], atol=0.08)
    assert np.isnan(track.lat[4]) and track.pressure_hpa[0] == pytest.approx(967.0, abs=0.2)


def test_verify_storm_measures_distance_pressure_and_equivalent_wind():
    best = BestTrack("AL012024", "TEST", np.array(["2024-09-26T00", "2024-09-26T06"], dtype="datetime64[h]"),
                     np.array([15.0, 16.0]), np.array([-80.0, -80.0]), np.array([80.0, 80.0]),
                     np.array([970.0, 970.0]), np.array(["HU", "HU"]), np.array(["", ""]))
    times = np.array(["2024-09-26T00", "2024-09-26T06"], dtype="datetime64[h]")
    track = StormTrack(times, np.array([15.0, 17.0]), np.array([280.0, 280.0]), np.array([960.0, np.nan]),
                       np.array([True, False]))
    err = verify_storm(track, best, name="model")
    assert err.name == "model" and err.tracked == 1
    assert err.track_km[0] == pytest.approx(0.0, abs=1e-6)
    assert err.pressure_hpa[0] == pytest.approx(-10.0)                       # 10 hPa deeper than official
    assert err.wind_kt[0] == pytest.approx(float(atkinson_holliday_wind(960.0)) - float(atkinson_holliday_wind(970.0)))
    assert np.isnan(err.track_km[1]) and not err.found[1]                    # not found: no error to quote


def test_verify_storm_counts_leads_from_the_initialization_when_it_is_given():
    """A 6-hourly archive starts at T+6: the lead has to be counted from the forecast's own init, not its first field."""
    best = BestTrack("AL012024", "TEST", np.array(["2024-09-26T00", "2024-09-26T12"], dtype="datetime64[h]"),
                     np.array([15.0, 15.0]), np.array([-80.0, -80.0]), np.array([80.0, 80.0]),
                     np.array([970.0, 970.0]), np.array(["HU", "HU"]), np.array(["", ""]))
    times = np.array(["2024-09-26T06", "2024-09-26T12"], dtype="datetime64[h]")      # T+6 and T+12 of the 00 UTC run
    track = StormTrack(times, np.array([15.0, 15.0]), np.array([280.0, 280.0]), np.array([960.0, 960.0]),
                       np.array([True, True]))
    assert verify_storm(track, best).leads_h.tolist() == [0.0, 6.0]                 # no init: the first field is 0
    err = verify_storm(track, best, init="2024-09-26T00")
    assert err.leads_h.tolist() == [6.0, 12.0] and err.init == np.datetime64("2024-09-26T00", "h")


def test_verify_storm_does_not_score_a_lead_past_the_end_of_the_record():
    """A forecast that outlives the storm has no official position to compare with: not a hit, and no NaN median."""
    best = BestTrack("AL012024", "TEST", np.array(["2024-09-26T00"], dtype="datetime64[h]"), np.array([15.0]),
                     np.array([-80.0]), np.array([80.0]), np.array([970.0]), np.array(["HU"]), np.array([""]))
    times = np.array(["2024-09-26T00", "2024-09-26T06"], dtype="datetime64[h]")
    track = StormTrack(times, np.array([15.0, 16.0]), np.array([280.0, 280.0]), np.array([960.0, 955.0]),
                       np.array([True, True]))                     # the vortex was there at both times
    err = verify_storm(track, best)
    assert err.found.tolist() == [True, False] and err.tracked == 1
    assert np.isfinite(err.track_km[0]) and np.isnan(err.track_km[1])
    sk = skill_by_lead([err])
    # the lead with nothing to score is absent from the table, not a NaN folded into a median
    assert np.isfinite(sk["track_km"][0]) and np.isnan(sk["track_km"][1])
    assert sk["found_fraction"].tolist() == [1.0, 0.0] and sk["n"].tolist() == [1, 0]


def test_skill_by_lead_and_horizons():
    leads = np.arange(0, 121, 6.0)
    times = np.datetime64("2024-09-26T00", "h") + leads.astype("timedelta64[h]")
    good = (240.0, 400.0, 1200.0)                                            # km at day 0, 2, 5
    er = []
    for offset in (0.0, 20.0):                                              # two initialisations
        track_km = np.interp(leads, [0, 48, 120], good) + offset
        er.append(StormErrors("m", times[0], times, leads, track_km, np.full_like(leads, 5.0),
                              np.full_like(leads, -8.0), np.ones(len(leads), bool)))
    er[1].found[-4:] = False                                                # this one loses the storm at the end
    sk = skill_by_lead(er)
    assert sk["track_km"][0] == pytest.approx(250.0) and sk["n"][0] == 2
    assert np.allclose(sk["found_fraction"][-1], 0.5) and sk["n"][-1] == 1
    h = storm_horizons(sk, track_km=500.0, wind_kt=20.0, found_fraction=0.75)
    assert h["track_hours"] == pytest.approx(56.1, abs=0.2)     # the median (base + 10 km) reaches 500 km at 56.1 h
    assert h["intensity_hours"] == np.inf                       # 8 kt never reaches 20
    assert h["lost_hours"] == pytest.approx(102.0) and h["thresholds"]["track_km"] == 500.0


def test_the_globe_and_the_skill_figure_are_drawn(tmp_path):
    pytest.importorskip("matplotlib")
    from pinneapple_physics.weather.cyclones import track_globe, track_skill_figure

    best = parse_hurdat2(HURDAT2)["AL012024"]
    fields = np.stack([gaussian_low(LAT, LON, (12.0 + 1.5 * k, 300.0 + 3.0 * k)) for k in range(4)])
    times = np.array(["2024-09-26T12", "2024-09-26T18", "2024-09-27T00", "2024-09-27T06"], dtype="datetime64[h]")
    track = track_storm(fields, LAT, LON, times, start=(12.0, 300.0), radius_km=400.0)
    png = track_globe(tmp_path / "globe.png", best, {"model": track}, pressure_hpa=fields[0], lat=LAT, lon=LON,
                      times=times, centre=(15.0, 300.0), size=120, title="t", note="n", dpi=40)
    gif = track_globe(tmp_path / "globe.gif", best, {"model": track}, pressure_hpa=fields, lat=LAT, lon=LON,
                      times=times, centre=(15.0, 300.0), size=80, title="t", note="n", dpi=30)
    dark = track_globe(tmp_path / "dark.png", best, {"model": track}, centre=(15.0, 300.0), size=80, dpi=30)
    leads = np.arange(0, 121, 6.0)
    skill = {"leads_h": leads, "track_km": np.linspace(1, 900, len(leads)),
             "pressure_hpa": np.linspace(0, 30, len(leads)), "wind_kt": np.linspace(0, 40, len(leads)),
             "found_fraction": np.linspace(1, 0, len(leads)), "n": np.ones(len(leads), int)}
    fig = track_skill_figure(tmp_path / "skill.png", skill, storm_horizons(skill), title="Ian", dpi=40)
    for p in (png, gif, dark, fig):
        assert p.exists() and p.stat().st_size > 1000
