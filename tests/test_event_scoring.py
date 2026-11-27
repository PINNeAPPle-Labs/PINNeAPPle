"""Generic event scoring (#351): the detection of each kind of event, the confusion matrix and lead time of a
deterministic forecast, and the Brier decomposition of an ensemble.

The first tests pin every number by hand -- a frame whose maximum is 5, a forecast that fires one lead early, an
ensemble that is always right. The last one runs the scorer on a real chaotic system (Lorenz-96, the trajectory
model of ``pinneapple_analysis.data_assimilation.models``) against persistence, a forecast whose skill is known to
decay, so that the scores are measured on something the test does not construct to pass.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.events import (
    EVENT_KINDS,
    Detection,
    EnsembleEventScore,
    Event,
    EventScore,
    Tolerances,
    detect,
    ensemble_event_score,
    event_score,
)

# ------------------------------------------------------------------ detection

def test_detect_threshold_reads_the_max_and_the_min():
    frame = np.array([[0.0, 5.0], [1.0, -2.0]])
    up = detect(frame, Event(kind="threshold", threshold=4.0))
    assert up.occurs and up.value == 5.0 and up.position == (0.0, 1.0)
    down = detect(frame, Event(kind="threshold", threshold=-1.0, above=False))
    assert down.occurs and down.value == -2.0 and down.position == (1.0, 1.0)
    assert not detect(frame, Event(kind="threshold", threshold=6.0)).occurs


def test_detect_region_mask_restricts_where_the_event_can_be():
    frame = np.array([[9.0, 0.0], [0.0, 0.0]])
    mask = np.array([[False, False], [True, True]])
    d = detect(frame, Event(kind="threshold", threshold=0.5, region=mask))
    assert not d.occurs                        # the hot corner is outside the region
    assert d.value == 0.0 and d.position == (1.0, 0.0)
    with pytest.raises(ValueError, match="keeps no point"):
        detect(frame, Event(kind="threshold", region=np.zeros((2, 2), bool)))


def test_detect_onset_is_the_change_against_the_reference_frame():
    reference = np.array([1.0, 1.0, 1.0])
    frame = np.array([1.0, 1.0, 2.5])
    ev = Event(kind="onset", threshold=1.0, reference=0)
    d = detect(frame, ev, reference=reference)
    assert d.occurs and d.value == pytest.approx(1.5) and d.position == (2.0,)
    assert not detect(frame, Event(kind="onset", threshold=1.6), reference=reference).occurs
    with pytest.raises(ValueError, match="needs the reference frame"):
        detect(frame, ev)


def test_detect_extremum_is_there_in_every_frame():
    for value in (-3.0, 0.0, 7.0):
        d = detect(np.array([value, value - 1.0]), Event(kind="extremum", threshold=1e9))
        assert isinstance(d, Detection) and d.occurs and d.value == value


def test_a_frame_with_a_non_finite_value_inside_the_region_is_refused():
    with pytest.raises(ValueError, match="non-finite"):
        detect(np.array([1.0, np.nan]), Event(kind="threshold", threshold=0.0))


def test_the_event_kinds_are_the_four_documented_ones():
    assert EVENT_KINDS == ("threshold", "extremum", "onset", "track")
    assert Event(kind="threshold", above=False).direction == "min"
    assert Event(kind="threshold", above=True).direction == "max"
    assert Event(kind="extremum").confusion is False and Event(kind="onset").confusion is True
    with pytest.raises(ValueError, match="unknown event kind"):
        Event(kind="storm")
    with pytest.raises(ValueError, match="direction"):
        Event(kind="threshold", direction="up")
    with pytest.raises(ValueError, match="ends before it starts"):
        Event(kind="threshold", window=(3, 1))


# ------------------------------------------------------------------ deterministic forecast

def _ramp(n_init: int = 2, n_lead: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A scalar truth ramp ``0, 1, 2, ...`` and the perfect forecast of it issued every ``n_lead - 1`` steps."""
    truth = np.arange(20.0)
    inits = np.arange(n_init) * (n_lead - 1)
    fc = np.stack([truth[i:i + n_lead] for i in inits])
    return fc, truth


def test_a_perfect_forecast_scores_every_cell_right():
    fc, truth = _ramp()
    s = event_score(fc, truth, Event(kind="threshold", threshold=3.5), inits=[0, 2])
    assert isinstance(s, EventScore)
    assert (s.hits, s.misses, s.false_alarms, s.correct_negatives) == (1, 0, 0, 5)
    assert s.n_cells == 6 and s.csi == 1.0
    assert s.probability_of_detection == 1.0 and s.false_alarm_ratio == 0.0
    assert s.peak_error == 0.0 and s.relative_peak_error == 0.0
    assert np.isnan(s.position_error)                      # a scalar field has no position to be wrong about
    assert s.detection_lead == 0.0 and s.timing_error == 0.0
    assert s.lead_time_ok == 2                             # every lead, including the last one


def test_an_early_forecast_is_scored_as_early_and_pays_for_it_at_the_lead_it_jumped_the_gun():
    fc, truth = _ramp()
    early = fc + 1.0                                       # the ramp one step ahead of itself
    s = event_score(early, truth, Event(kind="threshold", threshold=3.5), inits=[0, 2])
    assert s.detection_lead == 1.0 and s.timing_error == 1.0
    # at the lead where it fires early the truth has not crossed yet: one false alarm, and the truth's event at
    # the next lead is out of reach of the value tolerance (5 against 4), so it is a miss too
    assert (s.hits, s.misses, s.false_alarms) == (0, 1, 2)
    assert s.lead_time_ok == -1


def test_a_late_forecast_has_a_negative_detection_lead():
    fc, truth = _ramp()
    late = event_score(fc - 1.0, truth, Event(kind="threshold", threshold=2.5), inits=[0, 2])
    assert late.detection_lead == -1.0 and late.timing_error == 1.0


def test_an_event_out_of_tolerance_is_a_false_alarm_and_a_miss():
    truth = np.zeros((6,))
    truth[4] = 10.0
    fc = np.zeros((1, 6))
    fc[0, 4] = 6.0                                         # fires at the right lead, at half the height
    tight = event_score(fc, truth, Event(kind="threshold", threshold=5.0, tolerance=Tolerances(value=1.0)),
                        inits=[0])
    assert (tight.hits, tight.misses, tight.false_alarms) == (0, 1, 1)
    assert tight.csi == 0.0 and tight.lead_time_ok == -1
    assert tight.peak_error == pytest.approx(4.0) and tight.relative_peak_error == pytest.approx(0.4)
    wide = event_score(fc, truth, Event(kind="threshold", threshold=5.0, tolerance=Tolerances(value=5.0)),
                       inits=[0])
    assert (wide.hits, wide.misses, wide.false_alarms) == (1, 0, 0)     # the same call, inside the tolerance

    early = np.zeros((1, 6))
    early[0, 3] = 10.0                                     # right height, one lead early
    on_the_lead = event_score(early, truth, Event(kind="threshold", threshold=5.0), inits=[0])
    assert (on_the_lead.hits, on_the_lead.misses, on_the_lead.false_alarms) == (0, 1, 1)
    window = event_score(early, truth, Event(kind="threshold", threshold=5.0, tolerance=Tolerances(timing=1)),
                         inits=[0])
    assert (window.hits, window.misses, window.false_alarms) == (1, 0, 0)


def test_a_system_that_never_fires_has_no_lead_time():
    fc, truth = _ramp()
    quiet = event_score(fc, truth, Event(kind="threshold", threshold=1e9), inits=[0, 2])
    assert quiet.hits == 0 and quiet.false_alarms == 0
    assert quiet.lead_time_ok == -1
    assert np.isnan(quiet.csi) and np.isnan(quiet.probability_of_detection)


def test_the_lead_time_stops_at_the_first_wrong_lead():
    truth = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    fc = np.array([[1.0, 1.0, 0.0, 1.0]])                  # right, right, missed, then a false alarm
    ev = Event(kind="threshold", threshold=0.5)
    s = event_score(fc, truth, ev, inits=[4])              # cells: truth steps 4..7, leads 0..3
    assert [r.lead for r in s.per_lead] == [0, 1, 2, 3]
    assert [r.hits for r in s.per_lead] == [1, 1, 0, 0]
    assert [r.misses for r in s.per_lead] == [0, 0, 1, 0]
    assert [r.false_alarms for r in s.per_lead] == [0, 0, 0, 1]
    assert s.lead_time_ok == 1                             # lead 2 is wrong, so the run ends there
    assert s.detection_lead == 0.0


def test_the_window_narrows_what_is_scored():
    fc, truth = _ramp(n_init=1, n_lead=10)
    ev = Event(kind="threshold", threshold=3.5, window=(3, 5))
    s = event_score(fc, truth, ev, inits=[0])
    assert [r.lead for r in s.per_lead] == [3, 4, 5]
    assert s.n_cells == 3


def test_position_error_is_reported_in_the_units_of_the_coordinates():
    nx, ny = 5, 4
    xs, ys = np.meshgrid(np.linspace(0, 4, nx), np.linspace(0, 3, ny), indexing="ij")
    coords = np.stack([xs, ys], axis=-1)
    truth = np.zeros((3, nx, ny))
    truth[:, 4, 3] = 10.0
    fc = np.zeros((1, 3, nx, ny))
    fc[0, :, 0, 0] = 10.0
    s = event_score(fc, truth, Event(kind="track"), inits=[0], coords=coords)
    assert s.position_error == pytest.approx(np.hypot(4.0, 3.0))       # 5.0, in the units of coords
    index = event_score(fc, truth, Event(kind="track"), inits=[0])
    assert index.position_error == pytest.approx(np.hypot(4.0, 3.0))   # grid indices when no coords are given


def test_an_extremum_event_scores_the_peak_and_has_no_confusion_matrix():
    truth = np.array([[[0.0, 2.0], [1.0, 8.0]]])
    fc = truth[None] * 0.75
    s = event_score(fc, truth, Event(kind="extremum", tolerance=Tolerances(value=3.0)), inits=[0])
    assert (s.hits, s.misses, s.false_alarms, s.correct_negatives) == (1, 0, 0, 0)
    assert s.peak_error == pytest.approx(2.0) and s.relative_peak_error == pytest.approx(0.25)
    assert s.position_error == 0.0 and s.lead_time_ok == 0
    strict = event_score(fc, truth, Event(kind="extremum"), inits=[0])
    assert strict.hits == 0 and strict.lead_time_ok == -1               # 2.0 is outside a zero tolerance


def test_shapes_and_inputs_are_checked():
    with pytest.raises(ValueError, match="forecasts must be"):
        event_score(np.zeros(4), np.zeros(4), Event())
    with pytest.raises(ValueError, match="does not match frames"):
        event_score(np.zeros((1, 3)), np.zeros((3, 2)), Event())
    with pytest.raises(ValueError, match="inits given"):
        event_score(np.zeros((1, 3)), np.zeros((3,)), Event(), inits=[0, 1])


def test_the_score_serialises():
    fc, truth = _ramp()
    d = event_score(fc, truth, Event(kind="threshold", threshold=3.5), inits=[0, 2]).to_dict()
    assert d["kind"] == "threshold" and d["hits"] == 1 and len(d["per_lead"]) == 3
    assert set(d) >= {"csi", "lead_time_ok", "peak_error", "detection_lead", "timing_error"}


# ------------------------------------------------------------------ ensemble

def test_an_ensemble_that_always_agrees_with_the_truth_has_a_zero_brier_score():
    truth = np.zeros((6,))
    truth[3:] = 1.0
    members = np.repeat(truth[None], 3, axis=0)[None]                  # (n_init=1, n_mem=3, n_lead=6)
    s = ensemble_event_score(members, truth, Event(kind="threshold", threshold=0.5), inits=[0])
    assert isinstance(s, EnsembleEventScore)
    assert s.brier == pytest.approx(0.0) and s.reliability == pytest.approx(0.0)
    assert s.resolution == pytest.approx(s.uncertainty)                 # perfect: 0 = REL - RES + UNC
    assert s.base_rate == pytest.approx(3 / 6)
    assert [b.n for b in s.bins] == [3, 3]
    assert s.bins[0].observed_frequency == 0.0 and s.bins[1].observed_frequency == 1.0


def test_an_ensemble_that_always_says_one_half_scores_a_quarter():
    truth = np.zeros((4,))
    truth[2:] = 1.0
    members = np.zeros((1, 4, 4))
    members[0, :2, :] = 1.0                                             # two of four members always fire -> p = 0.5
    s = ensemble_event_score(members, truth, Event(kind="threshold", threshold=0.5), inits=[0])
    assert s.brier == pytest.approx(0.25)
    assert s.reliability == pytest.approx(0.0) and s.resolution == pytest.approx(0.0)
    assert s.uncertainty == pytest.approx(0.25)
    assert s.brier == pytest.approx(s.reliability - s.resolution + s.uncertainty)


def test_an_ensemble_needs_a_yes_no_event():
    with pytest.raises(ValueError, match="no yes/no event"):
        ensemble_event_score(np.zeros((1, 2, 3)), np.zeros((3,)), Event(kind="extremum"))


# ------------------------------------------------------------------ a real chaotic system

def test_persistence_on_lorenz96_scores_the_way_its_skill_decays():
    """Persistence keeps the initial frame forever. On a chaotic system its events have to be right at short lead
    and wrong at long lead, and the scorer has to show that without being told which is which."""
    torch = pytest.importorskip("torch")
    from pinneapple_analysis.data_assimilation.models import lorenz96_step

    K, dt, n_init, n_lead = 8, 0.01, 40, 30
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(0.0, 0.5, K))
    traj = [np.asarray(x, dtype=float)]
    for _ in range(n_init * n_lead + n_lead):
        x = lorenz96_step(x, dt)
        traj.append(np.asarray(x, dtype=float))
    T = np.asarray(traj)
    inits = np.arange(n_init) * n_lead
    perfect = np.stack([T[i:i + n_lead] for i in inits])
    persistence = np.repeat(T[inits][:, None, :], n_lead, axis=1)

    ev = Event(kind="threshold", threshold=3.0, tolerance=Tolerances(value=0.5, timing=2))
    pers = event_score(persistence, T, ev, inits=inits)
    perf = event_score(perfect, T, ev, inits=inits)

    assert pers.n_cells == n_init * n_lead == 1200
    assert 0.0 < pers.probability_of_detection < 1.0                   # it is neither perfect nor useless
    assert pers.peak_error > perf.peak_error >= 0.0
    assert perf.misses == 0 and perf.false_alarms == 0 and perf.csi == pytest.approx(1.0)
    hits = np.array([r.hits for r in pers.per_lead], dtype=float)
    assert hits[0] >= hits[-1]                                         # skill does not grow with the lead
    assert pers.lead_time_ok < perf.lead_time_ok == n_lead - 1

    # the same field scored as a moving feature: where the maximum is, and how far persistence puts it from the truth
    ridge = event_score(persistence, T, Event(kind="track"), inits=inits)
    peak = event_score(perfect, T, Event(kind="track"), inits=inits)
    assert peak.peak_error == pytest.approx(0.0) and peak.position_error == pytest.approx(0.0)
    assert ridge.position_error >= 0.0 and ridge.peak_error > 0.0
    assert np.isfinite(ridge.detection_lead)

    # and the same event scored as an ensemble: persistence plus a little noise, so the announced probability is
    # neither 0 nor 1 and the Brier score has something to measure
    grng = np.random.default_rng(1)
    members = np.stack([persistence + grng.normal(0.0, 0.25, persistence.shape) for _ in range(20)], axis=1)
    ens = ensemble_event_score(members, T, Event(kind="threshold", threshold=3.0), inits=inits)
    assert ens.n == n_init * n_lead == 1200
    assert 0.0 < ens.brier < ens.uncertainty
    # the decomposition is exact only when a bin holds a single probability; ten bins leave 2 % of the Brier
    # score inside the bins, which is why the identity is checked to a tolerance here and exactly above
    assert ens.brier == pytest.approx(ens.reliability - ens.resolution + ens.uncertainty, abs=0.01)
    assert len(ens.bins) >= 5                                          # the probabilities really do spread

    # the four numbers the module docstring and the product note quote, so a change cannot leave them behind
    assert (pers.hits, pers.misses, pers.false_alarms) == (470, 780, 670)
    assert (perf.hits, perf.misses, perf.false_alarms) == (1165, 0, 0)
    assert pers.csi == pytest.approx(0.245, abs=5e-4) and pers.probability_of_detection == pytest.approx(0.376, abs=5e-4)
    assert pers.lead_time_ok == 0 and perf.lead_time_ok == 29
    assert pers.peak_error == pytest.approx(0.957, abs=5e-4)
    assert ridge.peak_error == pytest.approx(0.954, abs=5e-4)
    assert ridge.position_error == pytest.approx(1.045, abs=5e-4)
    assert ens.brier == pytest.approx(0.0145, abs=5e-5)
    assert ens.reliability == pytest.approx(0.0095, abs=5e-5)
    assert ens.resolution == pytest.approx(0.0232, abs=5e-5)
    assert ens.uncertainty == pytest.approx(0.0283, abs=5e-5)
    assert ens.base_rate == pytest.approx(0.971, abs=5e-4)
