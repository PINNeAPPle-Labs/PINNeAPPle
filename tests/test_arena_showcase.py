"""pinneapple_arena.showcase (ported from the tests of PINNeAPPle-apps poc_model_benchmarking; a few epochs per
model)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
from pinneapple_arena import showcase  # noqa: E402

FAST = [{**a, "epochs": 20} for a in showcase.DEFAULT_ARCHITECTURES]


def test_default_architectures_build_their_own_class():
    from pinneapple_arena.config import ModelConfig, NetworkConfig, TrainingConfig
    from pinneapple_arena.model_factory import build_model

    for arch in showcase.DEFAULT_ARCHITECTURES:
        cfg = ModelConfig(name=arch["display"], type=arch["type"], network=NetworkConfig(**arch["network"]),
                          training=TrainingConfig(epochs=1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = build_model(cfg, in_dim=2, out_dim=1)
        assert type(model).__name__ != "_MLP", arch["type"]


@pytest.fixture(scope="module")
def result():
    return showcase.run_benchmark(architectures=FAST, seed=0, n_col=600, grid_n=24)


def test_metrics_rankings_and_flags(result):
    models = result["models"]
    assert [m["name"] for m in models] == showcase.ARCHITECTURE_ORDER
    for m in models:
        for k in ("rel_l2_error", "max_abs_error", "shock_region_rel_error", "physics_residual"):
            assert np.isfinite(m[k]) and m[k] >= 0
        assert m["shock_region_rel_error"] != m["rel_l2_error"]
    errs = [r["mean_rel_error"] for r in result["accuracy_ranking"]]
    assert errs == sorted(errs)
    assert result["ranking_order_unchanged"]
    pa = result["physics_aware"]
    res = np.array([pa["physics_residuals"][m["name"]] for m in models])
    expected = {m["name"] for m in models if pa["physics_residuals"][m["name"]] > np.median(res) + 2 * res.std()}
    assert set(pa["flagged_high_residual"]) == expected


def test_field_fit_is_the_arena_reference_at_t1(result):
    from pinneapple_arena.problems import get_problem

    fit = result["models"][0]["field_fit"]
    assert fit["t"] == pytest.approx(1.0)
    x = np.asarray(fit["x"])
    ref = get_problem("burgers_1d").analytical(x, np.ones_like(x), **showcase.PROBLEM_PARAMS)["u"]
    assert np.allclose(np.asarray(ref).reshape(-1), fit["u_real"], atol=1e-6)
    assert result["models"][0]["field_fit"]["u_pred"] != result["models"][1]["field_fit"]["u_pred"]


def test_absolute_threshold_flags_in_addition(result):
    r = showcase.run_benchmark(architectures=[{**a, "epochs": 2} for a in FAST[:2]], seed=1, n_col=200, grid_n=12,
                               residual_threshold=0.0)
    assert set(r["physics_aware"]["flagged_high_residual"]) == {"Vanilla PINN", "SIREN"}


def test_seed_reproduces():
    kw = dict(architectures=[{**FAST[0], "epochs": 5}], seed=3, n_col=200, grid_n=12)
    assert showcase.run_benchmark(**kw)["models"][0]["rel_l2_error"] == \
        showcase.run_benchmark(**kw)["models"][0]["rel_l2_error"]
