"""benchmark_suite PhysicsNeMo-Sym backend against nvidia-physicsnemo.sym 2.4.0 (cross-framework benchmark, bug 5).

It imported ``ModulusConfig``, which 2.4.0 does not have, then reported "physicsnemo.sym is not available" with the
package installed, and it never returned a predictor.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinneapple_tools.benchmark_suite.backends.physicsnemo_sym import PhysicsNeMoSymBackend
from pinneapple_tools.benchmark_suite.bundle.loader import BundleData


def _bundle(n=64):
    rng = np.random.default_rng(0)
    col = pd.DataFrame({"x": rng.uniform(0, 2, n), "y": rng.uniform(0, 1, n)})
    rows = []
    for region, x, y in (("inlet", np.zeros(8), np.linspace(0, 1, 8)), ("outlet", np.full(8, 2.0), np.linspace(0, 1, 8)),
                         ("walls", np.linspace(0, 2, 8), np.zeros(8))):
        rows += [{"x": a, "y": b, "region": region} for a, b in zip(x, y, strict=True)]
    return BundleData(root=Path("."), manifest={"nu": 0.05}, conditions={}, points_collocation=col,
                      points_boundary=pd.DataFrame(rows))


def test_missing_package_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "physicsnemo.sym", None)
    with pytest.raises(ImportError, match="nvidia-physicsnemo.sym"):
        PhysicsNeMoSymBackend().train(_bundle(), {"train": {"device": "cpu", "max_steps": 1}})


@pytest.mark.skipif(importlib.util.find_spec("physicsnemo") is None
                    or importlib.util.find_spec("physicsnemo.sym") is None, reason="needs nvidia-physicsnemo.sym")
def test_trains_and_returns_a_predictor(tmp_path):
    out = PhysicsNeMoSymBackend().train(_bundle(), {"train": {"device": "cpu", "max_steps": 3, "lr": 1e-3},
                                                    "model": {"nr_layers": 2, "layer_size": 16},
                                                    "output_dir": str(tmp_path)})
    X = np.array([[0.5, 0.5], [1.0, 0.25], [1.5, 0.75]], dtype=np.float32)
    y = out["predict_fn"](X)
    assert y.shape == (3, 3) and np.all(np.isfinite(y))
    assert out["model"] is not None and out["metrics"]["max_steps"] == 3.0
