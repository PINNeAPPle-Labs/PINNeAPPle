"""A served model raises a drift alarm, with labels and without (#114).

The scenario is a small surrogate standing in for a deployed one, and a window of live traffic that is either the
same distribution the model was validated on or not. The point of these tests is the *decision*: an alarm with no
consequence is a graph, and what a monitoring layer has to deliver is "retrain" or "do not", with a reason and a
number behind it.

Every threshold here comes from ``calibrate`` on a second held-out split, so the tests also check the thing that
makes the alarms meaningful: an in-distribution window does not ring them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pinneapple_analysis.trust.drift import (
    ENVELOPE,
    ERROR,
    PREDICTION,
    UNCERTAINTY,
    DriftMonitor,
    RetrainDecision,
    ks_distance,
    write_record,
)

SEED = 7


def truth(x: np.ndarray) -> np.ndarray:
    """The physics the surrogate approximates: a sine in the first coordinate, a ramp in the second."""
    return np.sin(2 * np.pi * x[:, :1]) + 0.5 * x[:, 1:2]


def served_model(noise: float = 0.02):
    """A served surrogate: predictions with a small error, and a predictive uncertainty that grows with x2.

    The uncertainty is the model's own claim about itself, which is what the label-free check watches: a model
    served points it is less sure about says so, and that shift is visible before any ground truth arrives.
    """
    rng = np.random.default_rng(SEED)

    def model(x):
        mean = truth(x) + rng.normal(0.0, noise, size=(len(x), 1))
        std = 0.02 + 0.05 * np.abs(x[:, 1])[:, None]
        return mean, std

    return model


def sample(n: int, *, shift: float = 0.0, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)
    x = rng.uniform(0.0, 1.0, size=(n, 2))
    return np.clip(0.5 + (x - 0.5) + shift, low, high) if shift else x


@pytest.fixture(scope="module")
def monitor():
    """Baseline from one held-out split, thresholds from a second one -- the two halves stay separate."""
    model = served_model()
    x_val = sample(2000)
    tuned = DriftMonitor.from_validation(model, x_val, truth(x_val))
    x_hold = sample(4000)
    tuned.calibrate(model, x_hold, truth(x_hold), window=300, windows=100, seed=1)
    return tuned


# ---------------------------------------------------------------- the statistic
def test_the_distance_is_zero_for_the_same_sample_and_one_for_disjoint_support():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.uniform(100, 200, size=500)
    assert ks_distance(a, a) == pytest.approx(0.0)
    assert ks_distance(a, b) == pytest.approx(1.0)


def test_a_variance_squeeze_is_drift_even_when_the_mean_does_not_move():
    """A difference of means would see nothing here; the empirical CDFs differ completely."""
    rng = np.random.default_rng(0)
    wide, narrow = rng.normal(size=2000), rng.normal(scale=1e-3, size=2000)
    assert abs(wide.mean() - narrow.mean()) < 0.1
    assert ks_distance(wide, narrow) > 0.4          # measured: the CDFs differ by at most this much


def test_columns_must_match():
    with pytest.raises(ValueError, match="columns"):
        ks_distance(np.zeros((10, 2)), np.zeros((10, 3)))


# ---------------------------------------------------------------- the alarms
def test_an_in_distribution_window_does_not_rise(monitor):
    """The false-alarm control: if this fails, every alarm below is worth nothing."""
    live = sample(500, shift=0.0)
    report = monitor.check(served_model(), live, labels=truth(live))
    assert not report.has_alarm, (report.alarms, report.statistics)
    assert report.decision.verdict == "STABLE" and not report.decision.retrain


def test_a_shifted_window_raises_the_alarm_without_any_label(monitor):
    """No ground truth in production: the output distribution and the predictive uncertainty are the evidence."""
    live = sample(500, shift=0.15)
    report = monitor.check(served_model(), live)
    assert set(report.alarms) >= {PREDICTION, UNCERTAINTY}, report.statistics
    assert report.decision.verdict == "RETRAIN" and report.decision.retrain
    assert "without any label" in report.decision.reason
    assert UNCERTAINTY in report.decision.triggered


def test_a_worse_model_raises_the_error_alarm_when_the_labels_arrive(monitor):
    """The physics moved and the surrogate did not: the residuals are the alarm, and they are critical."""
    live = sample(500)
    drifted_labels = truth(live) + 0.4          # what actually happened is not what the model learned
    report = monitor.check(served_model(), live, labels=drifted_labels)
    assert ERROR in report.alarms, report.statistics
    assert report.decision.verdict == "RETRAIN" and report.decision.retrain
    assert "validated ground" in report.decision.reason
    assert report.details["rmse_ratio"] > 5.0     # measured, not asserted: the error grew by this factor
    assert report.statistics[ERROR] > report.thresholds.error


def test_inputs_outside_the_fitted_distribution_ring_the_envelope_alarm(monitor):
    live = np.random.default_rng(3).uniform(3.0, 4.0, size=(400, 2))
    report = monitor.check(served_model(), live)
    assert ENVELOPE in report.alarms, report.statistics
    assert report.statistics[ENVELOPE] > 0.5
    assert report.decision.verdict == "RETRAIN"
    assert "envelope" in report.decision.reason


def test_a_model_that_reports_no_uncertainty_says_the_check_did_not_run(monitor):
    """`not run` is not `passed`: a model returning predictions only cannot support that claim."""
    plain = lambda x: truth(x)                                       # noqa: E731  (predictions only)
    report = monitor.check(plain, sample(500))
    unc = next(c for c in report.checks if c.name == UNCERTAINTY)
    assert unc.status == "not_run" and unc.value is None
    assert UNCERTAINTY not in report.alarms


def test_a_window_without_labels_says_so_instead_of_claiming_a_pass(monitor):
    report = monitor.check(served_model(), sample(500))
    err = next(c for c in report.checks if c.name == ERROR)
    assert err.status == "not_run" and "no labels" in err.reason


# ---------------------------------------------------------------- the thresholds
def test_the_thresholds_come_from_a_second_held_out_split(monitor):
    thresholds = monitor.thresholds
    assert thresholds.windows == 100 and thresholds.calibrated_window == 300
    assert thresholds.quantile == pytest.approx(0.99)
    for name in ("input", "prediction", "uncertainty", "error", "envelope"):
        assert 0.0 <= getattr(thresholds, name) < 0.5, (name, getattr(thresholds, name))


def test_calibration_refuses_a_split_too_small_to_hold_ten_windows():
    model = served_model()
    tuned = DriftMonitor.from_validation(model, sample(500), truth(sample(500)))
    with pytest.raises(ValueError, match="fewer than 10 windows"):
        tuned.calibrate(model, sample(400), window=300)


def test_measuring_before_calibrating_is_an_error_not_a_guess():
    model = served_model()
    tuned = DriftMonitor.from_validation(model, sample(500))
    with pytest.raises(RuntimeError, match="no calibrated thresholds"):
        tuned.check(model, sample(100))


# ---------------------------------------------------------------- the record
def test_the_decision_is_recorded_as_a_line_and_can_be_read_back(monitor, tmp_path):
    live = sample(500, shift=0.15)
    report = monitor.check(served_model(), live)
    path = write_record(report, tmp_path / "monitoring" / "drift.jsonl")
    report2 = monitor.check(served_model(), live)
    write_record(report2, path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    record = lines[0]
    assert record["decision"]["verdict"] == "RETRAIN" and record["decision"]["retrain"] is True
    assert "prediction_drift" in record["decision"]["triggered"]
    assert record["thresholds"]["quantile"] == pytest.approx(0.99)
    assert set(record["statistics"]) >= {PREDICTION, UNCERTAINTY}
    assert record["at"]                                            # a timestamp, not a floating report


def test_the_decision_is_a_value_a_pipeline_can_branch_on():
    stable = RetrainDecision(verdict="STABLE", retrain=False, reason="inside the limits")
    assert stable.to_dict() == {"verdict": "STABLE", "retrain": False, "reason": "inside the limits",
                                "triggered": []}


# ---------------------------------------------------------------- the served model
def test_a_torch_model_is_served_as_well_as_a_callable():
    """The dtype is taken from the module's own parameters: a float32 network is not handed float64 points."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    model = lambda x: net(torch.as_tensor(np.asarray(x), dtype=torch.float32)).detach().numpy()  # noqa: E731

    x_val = sample(1500)
    tuned = DriftMonitor.from_validation(model, x_val, truth(x_val))
    x_hold = sample(3000)
    tuned.calibrate(model, x_hold, truth(x_hold), window=250, windows=60, seed=2)
    report = tuned.check(model, sample(400), labels=truth(sample(400)))
    assert 0.0 <= report.statistics[PREDICTION] <= 1.0
    assert report.decision.verdict in ("STABLE", "REVIEW", "RETRAIN")
