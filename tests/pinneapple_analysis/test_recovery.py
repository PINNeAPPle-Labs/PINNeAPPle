"""A recovered parameter is only recovered if its interval is calibrated (#77).

Two claims, tested separately: the interval contains the truth about as often as the level says it should
(``coverage_study`` over the cooling-curve case), and a parameter the data cannot separate from another is named
instead of being handed a confident-looking number.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.inverse_problems.forward import (
    CallableModel,
    Observations,
    Parameter,
    ParameterSet,
)
from pinneapple_analysis.inverse_problems.presets import get_preset, synthetic
from pinneapple_analysis.inverse_problems.recovery import (
    CoverageReport,
    ParameterInterval,
    coverage_study,
    recover,
)

T = np.linspace(0.0, 10.0, 12)


def _decay(k: float = 0.4, *, scale: float = 1.0) -> CallableModel:
    """``y = exp(-k t)``: one log parameter, mildly nonlinear, and exactly linear in ``ln k`` over the data."""
    def fn(theta, inputs, observable=None):
        return scale * np.exp(-theta["k"] * np.asarray(inputs["t"], float))

    return CallableModel(fn, inputs=["t"], parameters=[Parameter("k", 1.0, 1e-3, 25.0, "1/s", truth=k)])


def _observed(model, params, *, sigma=0.02, seed=0) -> Observations:
    base = Observations({"t": T}, sigma, observable=np.array(["y"] * len(T), dtype=object))
    return synthetic(model, params.all, base, seed)


# ── the interval ───────────────────────────────────────────────────────────
def test_a_recovered_parameter_has_an_interval_that_contains_the_truth():
    params = ParameterSet([Parameter("k", 1.0, 1e-3, 25.0, "1/s", truth=0.4)])
    model = _decay()
    result = recover(model, params, _observed(model, params))
    interval = result.interval("k")
    assert interval.contains(0.4), str(interval)
    assert interval.estimate == pytest.approx(0.4, rel=0.15)
    assert interval.lower < interval.estimate < interval.upper
    assert result.contains(params.truth()) == {"k": True}
    assert result.ok, result.warnings


def test_more_measurements_at_the_same_noise_give_a_narrower_interval():
    """The interval is the data's, not a constant: doubling the points has to halve its width at least once."""
    params = ParameterSet([Parameter("k", 1.0, 1e-3, 25.0, "1/s", truth=0.4)])
    model = _decay()
    few = recover(model, params, _observed(model, params)).interval("k").width
    many = recover(model, params, _observed(model, params).concat(_observed(model, params, seed=1))).interval("k").width
    assert many < few


def test_a_prior_tightens_the_interval():
    """A prior as strong as the data (1 % in z) has to show up in the width; a weak one is allowed to be invisible."""
    from pinneapple_analysis.inverse_problems.estimation import Prior

    params = ParameterSet([Parameter("k", 1.0, 1e-3, 25.0, "1/s", truth=0.4)])
    model = _decay()
    obs = _observed(model, params)
    flat = recover(model, params, obs).interval("k")
    narrow = Prior(mean=np.array([np.log(flat.estimate)]), std=np.array([0.01]))    # mean in z: ln k
    narrowed = recover(model, params, obs, prior=narrow).interval("k")
    assert narrowed.relative_std < 0.8 * flat.relative_std


def test_the_level_changes_the_width_and_a_level_outside_zero_one_is_refused():
    params = ParameterSet([Parameter("k", 1.0, 1e-3, 25.0, "1/s", truth=0.4)])
    model = _decay()
    obs = _observed(model, params)
    assert recover(model, params, obs, level=0.99).interval("k").width > recover(model, params, obs).interval("k").width
    with pytest.raises(ValueError, match="between 0 and 1"):
        recover(model, params, obs, level=1.5)
    with pytest.raises(ValueError, match="'linear' or 'profile'"):
        recover(model, params, obs, interval="bootstrap")


def test_the_profile_interval_is_available_and_refuses_another_level():
    params = ParameterSet([Parameter("k", 1.0, 1e-3, 25.0, "1/s", truth=0.4)])
    model = _decay()
    obs = _observed(model, params)
    result = recover(model, params, obs, interval="profile")
    assert result.interval_kind == "profile"
    assert result.interval("k").contains(0.4)
    with pytest.raises(ValueError, match="95 %"):
        recover(model, params, obs, interval="profile", level=0.9)


# ── what the data cannot recover ───────────────────────────────────────────
def test_a_parameter_the_data_never_sees_is_named_with_an_infinite_interval():
    def fn(theta, inputs, observable=None):
        return np.exp(-theta["k"] * np.asarray(inputs["t"], float))

    model = CallableModel(fn, inputs=["t"], parameters=[Parameter("k", 0.5, 1e-3, 25.0),
                                                        Parameter("ghost", 1.0, 1e-3, 25.0)])
    params = ParameterSet([Parameter("k", 0.4, 1e-3, 25.0, truth=0.4),
                           Parameter("ghost", 2.0, 1e-3, 25.0, truth=2.0)])
    result = recover(model, params, _observed(model, params))
    assert result.unidentifiable == ["ghost"]
    assert not result.ok and "ghost" in result.warnings[0]
    assert not np.isfinite(result.interval("ghost").width), "an unheard parameter gets no confidence interval"
    assert result.interval("k").width < np.inf


def test_two_parameters_that_only_appear_as_a_ratio_are_reported_as_one_combination():
    """The lumped cooling curve depends on h and the heat capacity only through h/C: the data fix the ratio."""
    built = get_preset("cooling_h_C").build()
    params = ParameterSet(built["parameters"])
    result = recover(built["model"], params, built["observations"])
    assert set(result.unidentifiable) == {"h", "C"}
    assert result.dead[0]["fixed_combination"] == "h/C"
    assert "h/C" in result.warnings[0]
    assert result.interval("T0").width < np.inf, "the parameters that are seen keep a finite interval"
    assert result.fit.chi2 < 40, "the ambiguity is real even though the fit is good"


# ── the calibration the issue asks for ─────────────────────────────────────
def _cooling_case(rng: np.random.Generator):
    built = get_preset("cooling_h").build()
    params = ParameterSet(built["parameters"])
    return built["model"], params, synthetic(built["model"], params.all, built["observations"],
                                             int(rng.integers(2**31 - 1)))


def test_the_interval_covers_the_truth_as_often_as_it_claims():
    """The acceptance of #77: a recovered parameter with a calibrated interval, coverage measured over 150 runs."""
    report = coverage_study(_cooling_case, n=150, seed=1)
    assert set(report.covered) == {"h", "T0", "T_air"}
    for name in report.covered:
        low, high = report.ci(name)
        assert low <= 0.95 <= high, f"{name}: coverage {report.rate(name):.0%} with interval [{low:.0%}, {high:.0%}]"
    assert report.ok, report.calibrations


def test_a_measurement_of_the_coverage_from_too_few_runs_is_not_a_measurement():
    """Twenty runs cannot separate 95 % from 85 %; the Wilson interval says so, and the reading stays 'ok'."""
    report = coverage_study(_cooling_case, n=20, seed=2)
    low, high = report.ci("h")
    assert high - low > 0.15


def test_a_model_that_is_missing_a_real_effect_loses_its_coverage():
    """Data with a 2 °C bias the model cannot represent: the fit stays reasonable and the interval stops covering."""
    def case(rng):
        model, params, obs = _cooling_case(rng)
        return model, params, obs.with_values(obs.values + 2.0)

    report = coverage_study(case, n=60, seed=3)
    assert any(v == "under" for v in report.calibrations.values()), report.calibrations


def test_the_study_refuses_a_case_without_a_truth_to_check_against():
    def case(rng):
        model = _decay()
        params = ParameterSet([Parameter("k", 1.0, 1e-3, 25.0)])       # no truth
        base = Observations({"t": T}, 0.02, observable=np.array(["y"] * len(T), dtype=object))
        return model, params, base.with_values(np.exp(-0.4 * T))

    with pytest.raises(ValueError, match="truth"):
        coverage_study(case, n=2)


# ── the report ─────────────────────────────────────────────────────────────
def test_the_report_serialises_and_reads_as_a_table():
    params = ParameterSet([Parameter("k", 1.0, 1e-3, 25.0, "1/s", truth=0.4)])
    model = _decay()
    result = recover(model, params, _observed(model, params))
    d = result.to_dict()
    assert d["interval"] == "linear" and d["ok"] is True and d["unidentifiable"] == []
    assert d["intervals"][0]["name"] == "k"
    text = result.to_markdown()
    assert "| `k` |" in text and "95%" in text
    assert str(result).startswith("k = ")


def test_an_unbounded_interval_is_not_counted_as_a_calibrated_one():
    report = CoverageReport(level=0.95, n=100, covered={"a": 100, "b": 95}, width={"a": np.inf, "b": 1.0})
    assert report.calibrations["a"] == "unbounded" and report.calibrations["b"] == "ok"
    assert not report.ok
    assert "unbounded" in report.to_markdown()


def test_the_wilson_interval_of_a_coverage_is_inside_the_unit_interval():
    report = CoverageReport(level=0.95, n=10, covered={"a": 10}, width={"a": 1.0})
    low, high = report.ci("a")
    assert 0.0 <= low <= high <= 1.0 and high > 0.6
    empty = CoverageReport(level=0.95, n=0, covered={"a": 0}, width={"a": 1.0})
    assert empty.ci("a") == (0.0, 1.0)


def test_an_interval_knows_its_own_bounds():
    interval = ParameterInterval("k", 1.0, 0.8, 1.2, 0.1, 0.95)
    assert interval.width == pytest.approx(0.4)
    assert interval.contains(0.8) and interval.contains(1.2) and not interval.contains(1.3)
    assert interval.to_dict()["relative_std"] == 0.1
