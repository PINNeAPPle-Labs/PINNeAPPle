"""pinneapple_llm.draft_preset_kwargs / check_kwargs_in_bounds / draft_part / measure_part, with the LLM call stubbed
(no server needed); the build and the measurement are real."""
from __future__ import annotations

import importlib.util
import math

import pytest

import pinneapple_llm as pl
from pinneapple_llm import draft as draft_mod
from pinneapple_llm.cad_draft import CadRecipeResult
from pinneapple_llm.draft import DraftResult


def _sequence(monkeypatch, module, name, outcomes):
    calls = iter(outcomes)

    def fake(description, **_):
        out = next(calls)
        if isinstance(out, Exception):
            raise out
        return out
    monkeypatch.setattr(module, name, fake)


def test_check_kwargs_in_bounds_coerces_and_refuses():
    assert pl.check_kwargs_in_bounds({"a": "7000", "e": 0.001}, {"a": (6500, 45000)}) == {"a": 7000.0, "e": 0.001}
    for bad in ({"mu": "398600441800000"}, {"e": 1.0}, {"a": "seven"}, {"a": float("nan")}):
        with pytest.raises(ValueError):
            pl.check_kwargs_in_bounds(bad, {"mu": (3.9e5, 4.0e5), "e": (0.0, 0.9), "a": (6500, 45000)})


def test_draft_preset_kwargs_retries_until_a_sane_draft(monkeypatch):
    _sequence(monkeypatch, draft_mod, "draft_problem", [
        ValueError("hallucinated preset"),
        DraftResult("kepler_two_body_orbit", {}, "wrong preset", ""),
        DraftResult("satellite_j2_perturbation", {"e": 1.0}, "parabolic", ""),
        DraftResult("satellite_j2_perturbation", {"a": "7000", "e": "0.001"}, "ok", ""),
    ])
    kwargs, reasoning, attempts = pl.draft_preset_kwargs("a LEO orbit", "satellite_j2_perturbation",
                                                         bounds={"e": (0.0, 0.9), "a": (6500, 45000)})
    assert kwargs == {"a": 7000.0, "e": 0.001} and reasoning == "ok" and attempts == 4


def test_draft_preset_kwargs_gives_up_with_every_reason(monkeypatch):
    _sequence(monkeypatch, draft_mod, "draft_problem", [DraftResult(None, {}, "nothing fits", "")] * 2)
    with pytest.raises(pl.DraftRejected) as e:
        pl.draft_preset_kwargs("x", "satellite_j2_perturbation", max_attempts=2)
    assert len(e.value.attempts) == 2 and "nothing fits" in str(e.value)


def test_draft_part_retries_and_returns_the_attempt(monkeypatch):
    from pinneapple_llm import cad_draft

    good = CadRecipeResult(kind="mesh_recipe", recipe={"name": "box", "params": {"extents": [1, 2, 3]}},
                           reasoning="a box")
    _sequence(monkeypatch, cad_draft, "draft_mesh_recipe",
              [ValueError("unknown builder"), CadRecipeResult(kind="mesh_recipe", recipe=None, reasoning="?"), good])
    result, attempts = pl.draft_part("a box", tier="quick")
    assert result is good and attempts == 3
    with pytest.raises(ValueError):
        pl.draft_part("a box", tier="nonsense")


def _washer():
    return CadRecipeResult(kind="mesh_recipe", reasoning="washer", recipe={
        "name": "cylinder", "params": {"radius": 10, "height": 2},
        "boolean": {"op": "cut", "other": {"name": "cylinder", "params": {"radius": 5, "height": 3}}}})


@pytest.mark.skipif(importlib.util.find_spec("manifold3d") is None, reason="needs manifold3d for boolean cuts")
def test_measure_part_washer_with_a_boolean_engine():
    m = pl.measure_part(_washer())
    exact = math.pi * (10 ** 2 - 5 ** 2) * 2
    assert m["watertight"] and m["boolean_engines"]
    assert abs(m["volume"] - exact) / exact < 0.01           # faceting of the cylinders only
    assert m["extents"] == pytest.approx([20, 20, 2], abs=1e-6)


def test_measure_part_box_is_exact():
    m = pl.measure_part(CadRecipeResult(kind="mesh_recipe", reasoning="", recipe={"name": "box",
                                                                                   "params": {"extents": [40, 20, 5]}}))
    assert m["volume"] == pytest.approx(4000.0) and m["extents"] == pytest.approx([40, 20, 5])
