"""Numerical reproducibility (#59): the seed is applied, the environment recorded, the re-run measured."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from pinneapple_core.reproducibility import (
    ReproducibilityReport,
    compare_results,
    deterministic,
    environment_fingerprint,
    fingerprint_json,
    run_twice,
    seed_everything,
)


def _train_one_step():
    """A tiny deterministic-in-principle training step: same seed, same loss."""
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    x = torch.randn(8, 3)
    loss = (model(x) - torch.ones(8, 1)).pow(2).mean()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.01 * p.grad
    return {"loss": float(loss.detach()), "weights": [p.detach().numpy() for p in model.parameters()]}


def test_seed_everything_reports_what_it_set():
    record = seed_everything(7)
    assert record["seed"] == 7
    assert record["python_random"] is True
    assert record["numpy_random"] is True
    assert record["torch"] is True
    assert random.random() == pytest.approx(random.Random(7).random())


def test_deterministic_turns_the_algorithms_on_and_restores_them():
    before = torch.are_deterministic_algorithms_enabled()
    with deterministic(seed=1):
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is before


def test_deterministic_restores_the_python_and_numpy_generators():
    random.seed(1234)
    np.random.seed(1234)
    expected_python = random.random()
    expected_numpy = float(np.random.random())

    random.seed(1234)
    np.random.seed(1234)
    with deterministic(seed=99):
        random.random()
        np.random.random()
    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.random()) == pytest.approx(expected_numpy)


def test_deterministic_removes_the_cublas_variable_it_introduced():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    with deterministic(seed=0):
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None


def test_run_twice_reports_a_reproduced_training_step():
    report = run_twice(_train_one_step, seed=0)
    assert isinstance(report, ReproducibilityReport)
    assert report.identical is True, report.differences
    assert report.scored is True
    assert "loss" in report.checked


def test_run_twice_catches_a_run_that_uses_no_seed():
    """A generator seeded from OS entropy ignores the seed it was given."""

    def unseeded():
        return {"value": np.random.default_rng().random()}

    report = run_twice(unseeded, seed=0)
    assert report.identical is False
    assert report.differences[0]["field"] == "value"
    assert report.to_markdown().count("divergent") == 1


def test_run_twice_with_a_tolerance_absorbs_float_noise():
    def noisy():
        return {"value": np.array([1.0, 2.0]) + np.random.default_rng().normal(0, 1e-12, 2)}

    strict = run_twice(noisy, seed=0, atol=0.0)
    tolerant = run_twice(noisy, seed=0, atol=1e-6)
    assert strict.identical is False
    assert tolerant.identical is True


def test_run_twice_will_not_call_an_unscorable_result_reproduced():
    def returns_objects():
        return {"label": object()}

    report = run_twice(returns_objects, seed=0)
    assert report.scored is False
    assert report.identical is False
    assert "does not claim" in report.note


def test_run_twice_narrows_to_the_named_fields():
    def two_fields():
        return {"loss": np.array([1.0]), "label": object()}

    report = run_twice(two_fields, seed=0, fields=["loss"])
    assert report.checked == ["loss"]
    assert report.identical is True


def test_compare_results_names_a_shape_change():
    outcome = compare_results(np.zeros(3), np.zeros(4))
    assert outcome["identical"] is False
    assert "shape" in outcome["reason"]


def test_compare_results_handles_nested_mappings_and_tensors():
    a = {"x": torch.zeros(2), "y": [np.ones(2), 3.0]}
    b = {"x": torch.zeros(2), "y": [np.ones(2), 3.0]}
    assert compare_results(a, b)["identical"] is True


def test_compare_results_scales_the_tolerance_with_the_value():
    assert compare_results(np.array([100.0]), np.array([100.5]), rtol=1e-2)["identical"] is True
    assert compare_results(np.array([1.0]), np.array([1.5]), rtol=1e-2)["identical"] is False


def test_fingerprint_records_versions_and_never_guesses():
    fingerprint = environment_fingerprint()
    assert fingerprint["python"].count(".") >= 1
    assert fingerprint["numpy"] == np.__version__
    assert fingerprint["torch"] == torch.__version__
    assert isinstance(fingerprint["gpus"], list)
    assert isinstance(fingerprint["cpu_count"], int)
    assert "deterministic_algorithms" in fingerprint["determinism"]


def test_fingerprint_is_json():
    import json

    parsed = json.loads(fingerprint_json())
    assert parsed["numpy"] == np.__version__


def test_report_serialises_and_renders():
    report = run_twice(_train_one_step, seed=3)
    payload = report.to_dict()
    assert payload["seed"] == 3
    assert payload["identical"] is True
    assert "reproduced" in report.to_markdown()
