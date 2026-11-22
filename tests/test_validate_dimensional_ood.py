"""The two check types #31 adds — dimensional and out-of-distribution — and the "not run" status they need.

The point of the pair is that neither can be faked: a unit check cannot pass a model that does not say what unit
it returns, and a detector cannot pass by flagging everything. Both are reported with their own status, and "not
run" is never a pass — the rule the data gates already follow.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_analysis.validation.core import CheckResult, ValidationReport
from pinneapple_analysis.validation.validator import PhysicsValidator


class TimesTwo(torch.nn.Module):
    def forward(self, x):
        return x[:, :1] * 2.0


class WithUnit(torch.nn.Module):
    """Returns a real :class:`PhysicsTensor` — what the dimensional check needs to have something to compare."""

    def __init__(self, unit: str = "m/s"):
        super().__init__()
        self.unit_name = unit

    def forward(self, x):
        import pinneapple as pp
        return pp.tensor(x[:, :1] * 2.0, name="u", units=self.unit_name)


BOX = {"x": (0.0, 1.0), "y": (0.0, 1.0)}


def validator(model, **kw):
    return PhysicsValidator(model, ["x", "y"], dict(BOX), **kw)


# --------------------------------------------------------------------------- the status a check can be in
def test_a_check_that_could_not_be_asked_is_not_a_pass():
    report = ValidationReport.create("m")
    report.checks.append(CheckResult.not_run("dimensional", "the model returns a plain tensor"))
    assert report.passed is False
    assert report.n_not_run == 1 and report.n_passed == 0 and report.n_failed == 0
    assert report.failures() == [], "a check that did not run is not a failure to fix; it is an unanswered question"
    assert "SKIP" in report.summary() and "1 not run" in report.summary()
    assert report.to_dict()["n_not_run"] == 1


def test_a_report_passes_only_when_a_check_ran_and_passed():
    report = ValidationReport.create("m")
    report.checks.append(CheckResult(name="ok", passed=True, value=0.0, threshold=1.0, description="fine"))
    report.checks.append(CheckResult.not_run("dimensional", "no unit"))
    assert report.passed is True and report.n_not_run == 1


# --------------------------------------------------------------------------- dimensional
def test_the_expected_unit_passes():
    result = validator(WithUnit("m/s")).add_dimensional_check("m/s").validate()
    check = result.checks[0]
    assert check.name == "dimensional" and check.passed is True, result.summary()
    assert check.value == 1.0


def test_the_same_dimensions_scaled_reports_the_factor():
    """km/s where m/s is expected is not a different physics problem: it is every number off by a thousand."""
    result = validator(WithUnit("km/s")).add_dimensional_check("m/s").validate()
    check = result.checks[0]
    assert check.passed is False
    assert check.value == pytest.approx(1000.0)
    assert "scaled by 1000" in check.description


def test_different_dimensions_fail_saying_so():
    result = validator(WithUnit("Pa")).add_dimensional_check("m/s").validate()
    check = result.checks[0]
    assert check.passed is False
    assert "different dimensions" in check.description


def test_a_plain_tensor_is_not_run_with_the_reason():
    """The model says nothing about units, so the check says nothing about units — it does not guess."""
    result = validator(TimesTwo()).add_dimensional_check("m/s").validate()
    check = result.checks[0]
    assert check.passed is None and check.status == "not run"
    assert "plain tensor" in check.description and "PhysicsTensor" in check.description
    assert result.passed is False


def test_a_model_that_refuses_to_run_is_reported_not_crashed():
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("not trained yet")

    result = validator(Broken()).add_dimensional_check("m/s").validate()
    assert result.checks[0].passed is None and "not trained yet" in result.checks[0].description


# --------------------------------------------------------------------------- out of distribution
def test_points_outside_the_domain_are_flagged():
    result = validator(TimesTwo()).add_ood_check().validate()
    check = result.checks[0]
    assert check.name == "ood" and check.passed is True, check.description
    assert check.direction == "higher" and check.value >= check.threshold
    assert "of the points outside the domain were flagged" in check.description


def test_a_detector_that_cannot_tell_is_not_flagged_as_failing_the_model():
    """With an impossible level nothing is flagged, and the check says the detector saw nothing."""
    result = validator(TimesTwo()).add_ood_check(ood_level=1e-30).validate()
    check = result.checks[0]
    assert check.passed is False and check.value == 0.0


def test_flagging_everything_fails_on_the_false_alarm():
    """Recall alone is a detector that answers "out" to every question; the pair of numbers is what makes it real."""
    result = validator(TimesTwo()).add_ood_check(ood_level=1.0 - 1e-12, max_false_alarm=0.05).validate()
    check = result.checks[0]
    assert check.value == pytest.approx(1.0)          # everything outside was flagged ...
    assert check.passed is False                       # ... and so was everything inside
    assert "wrongly flagged" in check.description


def test_a_real_training_sample_is_the_reference_when_given():
    rng = np.random.default_rng(0)
    train = torch.as_tensor(rng.random((2000, 2)), dtype=torch.float32)
    result = validator(TimesTwo()).add_ood_check(training_coords=train).validate()
    check = result.checks[0]
    assert check.passed is True, check.description
    assert "training points" in check.description


def test_ood_needs_bounds_for_every_coordinate():
    v = PhysicsValidator(TimesTwo(), ["x", "y"], {"x": (0.0, 1.0)})
    result = v.add_ood_check().validate()
    assert result.checks[0].passed is None and "bounds" in result.checks[0].description


# --------------------------------------------------------------------------- together with the existing kinds
def test_the_four_kinds_report_side_by_side():
    """What #31 asks for: one class, one report, each check with its own status."""
    def custom(model, coord_names, bounds):
        return 0.0

    result = (validator(WithUnit("m/s"))
              .add_dimensional_check("m/s")
              .add_ood_check()
              .add_custom_check(custom, "custom", 1e-6)
              .validate())
    assert [c.name for c in result.checks] == ["dimensional", "ood", "custom"]
    assert [c.status for c in result.checks] == ["passed", "passed", "passed"]
    assert result.passed is True
