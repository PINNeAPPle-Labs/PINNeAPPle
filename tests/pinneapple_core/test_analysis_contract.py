"""The analysis contract: what a calculation takes, what it gives back, and the sentence it words.

The rules this file holds the code to, one test each:

* a declaration that does not match its function is refused at import time, not at failure time;
* junk in (missing, NaN, infinity, out of range, text where a number belongs, empty array) is a ``Result`` with a
  reason, never an exception — for every analysis the library ships, checked by the same kit;
* an analysis that judges the numbers carries a status and a severity; one that cannot answer says
  ``insufficient_data`` or ``not_applicable``;
* the finding is one sentence, in the language in force, with the numbers in that language's format.
"""
import json
import math

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_core.analysis import (
    Analysis,
    AnalysisError,
    DuplicateAnalysis,
    Finding,
    InputSpec,
    InvalidAnalysis,
    Maturity,
    NotEnoughData,
    OutputSpec,
    Reference,
    Result,
    Severity,
    Status,
    analyses,
    analysis,
    analysis_card,
    check_all,
    check_analysis,
    fmt_number,
    fmt_percent,
    get_analysis,
    junk_calls,
    register_analysis,
    run_analysis,
    use_language,
)
from pinneapple_core.registry import AlreadyRegistered
from pinneapple_physics.closed_form import pipe_flow
from pinneapple_physics.closed_form.fin_array_conduction import (
    fin_array_base_temperature_c,
    fin_array_profile_params,
)
from pinneapple_physics.closed_form.helmholtz_resonator import helmholtz_resonant_frequency_hz

BUILT_IN = ["acoustics.helmholtz_frequency", "pipe.darcy_pressure_drop", "pipe.friction_factor",
            "thermal.fin_array_base_temperature"]


# --------------------------------------------------------------------------- declaring
def test_the_library_ships_the_closed_form_analyses():
    assert analyses() == BUILT_IN, "every closed-form analysis is discoverable without importing anything by hand"
    assert analyses("fluid_mechanics") == ["pipe.darcy_pressure_drop", "pipe.friction_factor"]


def test_a_declaration_is_checked_against_its_function_while_the_module_imports():
    with pytest.raises(InvalidAnalysis, match="is not a parameter"):
        @analysis("coupon.bad_input", inputs={"y": InputSpec()})
        def wrong_name(x):
            return {"z": x}

    with pytest.raises(InvalidAnalysis, match="no default and no InputSpec"):
        @analysis("coupon.undeclared")
        def undeclared(x):
            return {"z": x}

    with pytest.raises(InvalidAnalysis):
        @analysis("coupon.varargs")
        def varargs(*args):
            return {"z": 0}

    with pytest.raises(InvalidAnalysis, match="'strict' is the registry's own argument"):
        @analysis("coupon.reserved", inputs={"strict": InputSpec()})
        def reserved(strict):
            return {"z": strict}


def test_an_array_input_is_checked_for_being_empty_or_holding_nan():
    @analysis("coupon.array", inputs={"series": InputSpec(kind="array")}, outputs={"n": OutputSpec()},
              describe="{n:.0} samples", example={"series": [1.0, 2.0]})
    def counts(series):
        return {"n": float(np.asarray(series, dtype=float).size)}

    try:
        assert check_analysis("coupon.array") == []
        empty = run_analysis("coupon.array", series=[])
        assert empty.status is Status.INSUFFICIENT_DATA and empty.sentence == "series is empty"
        with_nan = run_analysis("coupon.array", series=[1.0, float("nan")])
        assert "non-finite values" in with_nan.sentence
        assert run_analysis("coupon.array", series=np.arange(5.0)).metrics["n"] == 5.0
    finally:
        pp.unregister("analysis", "coupon.array")


def test_the_card_is_everything_a_form_needs_and_it_is_json_ready():
    card = analysis_card("pipe.friction_factor")
    assert card["label"] == "Darcy friction factor"
    assert card["inputs"]["Re"]["label"] == "Reynolds number" and card["inputs"]["Re"]["gt"] == 0.0
    assert card["outputs"]["f"]["unit"] == "-"
    assert card["maturity"] == Maturity.LITERATURE_BASED.value
    assert card["references"] and card["limitations"]
    json.dumps(card)                                     # a form, an app and an assistant all read this
    assert analysis_card("pipe.friction_factor", "pt")["label"] == "fator de atrito de Darcy"


def test_a_duplicate_id_is_refused_and_replace_overwrites():
    def coupon(x):
        return {"y": x}

    first = Analysis(id="coupon.duplicate", function=coupon, inputs={"x": InputSpec()},
                     outputs={"y": OutputSpec()}, example={"x": 1.0})
    pp.register(first, kind="analysis")
    try:
        with pytest.raises(AlreadyRegistered, match="analysis 'coupon.duplicate'"):
            pp.register(first, kind="analysis")
        with pytest.raises(DuplicateAnalysis):
            register_analysis(first)
        second = Analysis(id="coupon.duplicate", function=lambda x: {"y": 2 * x}, inputs={"x": InputSpec()},
                          outputs={"y": OutputSpec()}, example={"x": 1.0})
        pp.register(second, kind="analysis", replace=True)
        assert get_analysis("coupon.duplicate") is second
    finally:
        pp.unregister("analysis", "coupon.duplicate")
    with pytest.raises(AnalysisError):
        get_analysis("coupon.duplicate")


# --------------------------------------------------------------------------- running
def test_a_good_call_returns_the_numbers_and_the_sentence():
    result = run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4)
    assert isinstance(result, Result) and result.ok
    assert result.status is Status.OK and result.severity is Severity.NONE
    assert result.metrics["f"] == pytest.approx(pipe_flow.friction_factor(45_000, 4.5e-4))
    assert "f = 0.0228" in result.sentence and "Re = 45,000" in result.sentence
    assert result.maturity is Maturity.LITERATURE_BASED and result.references
    assert result.to_dict()["status"] == "ok" and "0.0228" in result.to_text()


@pytest.mark.parametrize("name", BUILT_IN)
def test_every_built_in_analysis_passes_the_data_hygiene_kit(name):
    assert check_analysis(name) == []


def test_the_kit_reports_an_analysis_that_raises_instead_of_answering():
    @analysis("coupon.raises", inputs={"x": InputSpec(gt=0)}, outputs={"y": OutputSpec()},
              example={"x": 1.0})
    def always_raises(x):
        raise ZeroDivisionError("the analysis has a bug")

    try:
        problems = check_analysis("coupon.raises")
        assert any("ZeroDivisionError" in p for p in problems), "the example that raises is reported, not inherited"
        result = run_analysis("coupon.raises", x=2.0)
        assert result.status is Status.ERROR and "ZeroDivisionError" in result.sentence

    finally:
        pp.unregister("analysis", "coupon.raises")


def test_the_kit_reports_an_analysis_that_answers_ok_to_a_missing_input():
    @analysis("coupon.tolerates", inputs={"x": InputSpec(required=False)}, outputs={"y": OutputSpec()},
              example={"x": 1.0})
    def tolerates_a_missing_value(x=None):
        return {"y": float(x or 0.0)}                   # a silent zero: the failure the kit exists to catch

    try:
        problems = check_analysis("coupon.tolerates")
        assert any("answered ok with x missing" in p for p in problems)
    finally:
        pp.unregister("analysis", "coupon.tolerates")


def test_a_missing_value_names_the_input_and_never_raises():
    result = run_analysis("pipe.friction_factor", rel_rough=4.5e-4)
    assert result.status is Status.INSUFFICIENT_DATA and not result.ok
    assert "Reynolds number (Re) is missing" in result.sentence
    assert result.problems and result.metrics == {}


@pytest.mark.parametrize("value, expected", [
    (float("nan"), "must be finite"),
    (float("inf"), "must be finite"),
    (-5.0, "must be greater than 0"),
    ("twelve", "must be a number"),
])
def test_a_value_the_model_cannot_use_comes_back_with_a_reason(value, expected):
    result = run_analysis("pipe.friction_factor", Re=value, rel_rough=4.5e-4)
    assert result.status is Status.INSUFFICIENT_DATA
    assert expected in result.sentence and "(Re)" in result.sentence
    assert result.sentence == result.problems[0]


def test_an_input_outside_the_models_range_names_the_limit_and_the_unit():
    result = run_analysis("pipe.darcy_pressure_drop", f=0.02, L=120, D=-0.05, rho=998, U=2.0)
    assert result.status is Status.INSUFFICIENT_DATA
    assert "internal diameter (D) must be greater than 0 m, got -0.05" in result.sentence


def test_an_unknown_input_is_refused_before_the_calculation_runs():
    result = run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4, temperature=20)
    assert result.status is Status.INSUFFICIENT_DATA and "unknown input: temperature" in result.sentence


def test_the_analysis_says_insufficient_data_with_its_own_words():
    @analysis("coupon.short_window", inputs={"series": InputSpec(kind="array")},
              outputs={"mean": OutputSpec(unit="-", label="mean")}, example={"series": [1.0, 2.0, 3.0]})
    def short_window(series):
        if len(series) < 30:
            raise NotEnoughData(f"the window has {len(series)} samples, 30 are needed")
        return {"mean": float(np.mean(series))}

    try:
        result = run_analysis("coupon.short_window", series=np.arange(5.0))
        assert result.status is Status.INSUFFICIENT_DATA
        assert result.sentence == "the window has 5 samples, 30 are needed"
        result = run_analysis("coupon.short_window", series=np.arange(40.0))
        assert result.ok and result.metrics["mean"] == pytest.approx(19.5)
    finally:
        pp.unregister("analysis", "coupon.short_window")


def test_strict_raises_what_a_caller_showing_a_person_never_sees():
    @analysis("coupon.strict", inputs={"x": InputSpec()}, outputs={"y": OutputSpec()}, example={"x": 1.0})
    def broken(x):
        raise RuntimeError("an actual defect")

    try:
        assert run_analysis("coupon.strict", x=1.0).status is Status.ERROR
        with pytest.raises(RuntimeError, match="an actual defect"):
            run_analysis("coupon.strict", x=1.0, strict=True)
    finally:
        pp.unregister("analysis", "coupon.strict")


def test_an_output_the_analysis_did_not_return_is_an_error_not_a_missing_key():
    @analysis("coupon.forgot_the_output", inputs={"x": InputSpec()},
              outputs={"y": OutputSpec(), "z": OutputSpec()}, example={"x": 1.0})
    def forgets(x):
        return {"y": x}

    try:
        result = run_analysis("coupon.forgot_the_output", x=1.0)
        assert result.status is Status.ERROR and "did not return z" in result.sentence
    finally:
        pp.unregister("analysis", "coupon.forgot_the_output")


def test_a_finding_carries_the_status_and_the_severity_of_the_judgement():
    @analysis("coupon.limit", inputs={"value": InputSpec(unit="bar"), "limit": InputSpec(unit="bar", gt=0)},
              outputs={"ratio": OutputSpec(unit="-")},
              describe="the value is {ratio:%} of the limit", example={"value": 10.0, "limit": 16.0})
    def against_a_limit(value, limit):
        ratio = value / limit
        if ratio > 1.0:
            return Finding(metrics={"ratio": ratio}, status=Status.BAD, severity=Severity.HIGH,
                           evidence={"value": value, "limit": limit})
        return Finding(metrics={"ratio": ratio})

    try:
        ok = run_analysis("coupon.limit", value=12.0, limit=16.0)
        assert ok.status is Status.OK and ok.sentence == "the value is 75% of the limit"
        over = run_analysis("coupon.limit", value=20.0, limit=16.0)
        assert over.status is Status.BAD and over.severity is Severity.HIGH
        assert over.sentence == "the value is 125% of the limit" and over.evidence == {"value": 20.0, "limit": 16.0}
    finally:
        pp.unregister("analysis", "coupon.limit")


# --------------------------------------------------------------------------- the sentence
def test_the_sentence_is_written_in_the_language_in_force():
    english = run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4)
    assert english.sentence.startswith("f = 0.0228 at Re = 45,000") and "(turbulent)" in english.sentence
    with use_language("pt"):
        portuguese = run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4)
        assert portuguese.sentence.startswith("f = 0,0228 em Re = 45.000") and "(turbulento)" in portuguese.sentence
        missing = run_analysis("pipe.friction_factor", Re=-5)
        assert "número de Reynolds (Re) precisa ser maior que 0, veio -5" == missing.sentence
        assert missing.language == "pt"
    assert run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4).language == "en"


def test_numbers_are_written_the_way_each_language_writes_them():
    assert fmt_number(1234.5, lang="en") == "1,234.5"
    assert fmt_number(1234.5, lang="pt") == "1.234,5"
    assert fmt_number(0.0, lang="pt") == "0" and fmt_number(float("nan")) == "—" and fmt_number(None) == "—"
    assert fmt_percent(0.734, lang="en") == "73%" and fmt_percent(0.734, lang="pt") == "73%"
    assert fmt_percent(0.0734, 1, "pt") == "7,3%"


def test_a_sentence_naming_a_number_the_result_does_not_carry_prints_a_dash():
    @analysis("coupon.partial", inputs={"x": InputSpec()}, outputs={"y": OutputSpec()},
              describe="y = {y:.2} and z = {z:.2}", example={"x": 1.0})
    def partial(x):
        return {"y": x}

    try:
        assert run_analysis("coupon.partial", x=2.0).sentence == "y = 2 and z = —"
    finally:
        pp.unregister("analysis", "coupon.partial")


# --------------------------------------------------------------------------- the kit
def test_the_kit_builds_the_hostile_calls_from_the_declaration_alone():
    labels = [label for label, _values in junk_calls(get_analysis("pipe.friction_factor"))]
    assert "nothing at all" in labels
    assert "Re NaN" in labels and "Re below its exclusive minimum" in labels
    assert "rel_rough below its minimum" in labels and "rel_rough above its maximum" in labels


def test_a_well_behaved_analysis_passes_and_the_library_ships_no_failing_one():
    assert check_all() == {}, "every analysis the library ships answers junk with a reason"


# --------------------------------------------------------------------------- the physics is the wrapped function
def test_the_built_in_analyses_return_what_the_functions_they_wrap_return():
    friction = run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4)
    assert friction.metrics["f"] == pytest.approx(pipe_flow.friction_factor(45_000, 4.5e-4))

    drop = run_analysis("pipe.darcy_pressure_drop", f=0.0216, L=120.0, D=0.0525, rho=998.0, U=2.0)
    assert drop.metrics["dp"] == pytest.approx(pipe_flow.darcy_pressure_drop(0.0216, 120.0, 0.0525, 998.0, 2.0))

    tuning = run_analysis("acoustics.helmholtz_frequency", port_radius_m=0.025, port_length_m=0.12,
                          cabinet_volume_l=20.0)
    assert tuning.metrics["f"] == pytest.approx(helmholtz_resonant_frequency_hz(0.025, 0.12, 20.0))

    inputs = {"k_material_w_mk": 200.0, "h_conv_w_m2k": 25.0, "n_fins": 20.0, "fin_thickness_m": 0.001,
              "fin_length_m": 0.03, "transverse_extent_m": 0.06, "spacing_extent_m": 0.06,
              "base_thickness_m": 0.005, "power_w": 60.0, "t_ambient_c": 25.0}
    params = fin_array_profile_params(inputs["k_material_w_mk"], inputs["h_conv_w_m2k"], inputs["n_fins"],
                                      inputs["fin_thickness_m"], inputs["fin_length_m"],
                                      inputs["transverse_extent_m"], inputs["spacing_extent_m"],
                                      inputs["base_thickness_m"], inputs["power_w"])
    fins = run_analysis("thermal.fin_array_base_temperature", **inputs)
    assert fins.metrics["t_base_c"] == pytest.approx(fin_array_base_temperature_c(params, 25.0, 60.0))
    assert fins.sentence.startswith("20 fins dissipating 60 W put the base at")


def test_the_references_of_an_analysis_travel_on_its_result():
    result = run_analysis("pipe.friction_factor", Re=45_000)
    assert result.references and all(isinstance(r, Reference) for r in result.references)
    assert result.to_dict()["references"][0]["title"].startswith("White")
    assert math.isfinite(result.metrics["f"])
