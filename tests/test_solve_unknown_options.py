"""pp.solve does not swallow options it does not use (cross-framework benchmark, REPORT.md bug 4)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pinneapple as pp


def test_pinn_rejects_an_unknown_option():
    with pytest.raises(TypeError, match="n_epochs"):
        pp.solve("burgers_1d", "pinn", n_epochs=5)


def test_pinn_warns_on_a_size_the_architecture_cannot_take():
    # elm has a single hidden layer: an explicit n_layers is ignored, and that is said
    with pytest.warns(UserWarning, match="n_layers"):
        pp.solve("burgers_1d", "pinn", architecture="elm", n_layers=3, epochs=1, n_collocation=64)


def test_pinn_honours_the_size_of_vanilla_pinn():
    sol = pp.solve("burgers_1d", "pinn", architecture="vanilla_pinn", hidden_dim=16, n_layers=2, epochs=2,
                   n_collocation=64)
    assert sol.info["n_parameters"] == (2 * 16 + 16) + (16 * 16 + 16) + (16 + 1)


def test_other_methods_warn_on_an_unknown_option():
    with pytest.warns(UserWarning, match="nu"):
        pp.solve("burgers_1d", "analytic", nu=0.01 / np.pi)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pp.solve("burgers_1d", "analytic", seed=0)                    # accepted (and unused) by every method
        pp.solve("burgers_1d", "analytic", preset_kwargs={"nu": 0.01 / np.pi})
