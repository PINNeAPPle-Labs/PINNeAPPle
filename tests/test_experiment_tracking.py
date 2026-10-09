"""Experiment tracking adapter (#113): one option sends metrics and artifacts to MLflow or W&B."""
import sys
import types

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_physics.closed_form.burgers import burgers_sine_exact
from pinneapple_physics.tracking import flat_metrics, flat_params, log_result

NU = 0.01 / np.pi


def _exp(**kw):
    exact = lambda X: burgers_sine_exact(X[:, 0], X[:, 1], NU)[:, None]  # noqa: E731
    return pp.Experiment(pp.PhysicalProblem.from_preset("burgers_1d", nu=NU), method="exact",
                         options={"fn": exact}, reference="analytic", n_eval=200, **kw)


class _FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.params, self.metrics, self.tags, self.artifacts = {}, {}, {}, []

    def start_run(self, run_name=None, **kw):
        fake = self

        class _Run:
            info = types.SimpleNamespace(run_id="run-1")

            def __enter__(self):
                fake.run_name = run_name
                return self

            def __exit__(self, *a):
                return False
        return _Run()

    def log_params(self, p): self.params.update(p)
    def log_metrics(self, m): self.metrics.update(m)
    def set_tags(self, t): self.tags.update(t)

    def log_artifacts(self, d, artifact_path=None):
        import os
        self.artifacts += sorted(os.listdir(d))


def test_flat_records():
    r = _exp().run()
    m, p = flat_metrics(r), flat_params(r)
    assert m["relative_l2/u"] < 1e-10 and "wall_time_s" in m
    assert p["method"] == "exact" and p["reference"] == "analytic"


def test_mlflow_tracker_gets_params_metrics_and_the_record(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    r = _exp(tracker="mlflow").run()
    assert r.tracker_run_id == "run-1" and fake.run_name == r.config["name"]
    assert "relative_l2/u" in fake.metrics and fake.params["method"] == "exact"
    assert fake.tags["problem_fingerprint"] == r.problem_fingerprint and "result.json" in fake.artifacts


def test_callable_tracker_and_errors(monkeypatch):
    seen = []
    r = _exp(tracker=lambda res: seen.append(res) or "custom").run()
    assert r.tracker_run_id == "custom" and seen[0] is r
    with pytest.raises(ValueError, match="unknown tracker"):
        log_result(r, "neptune")
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="pip install wandb"):
        log_result(r, "wandb")
