"""ModelRegistry.build honours hidden_dim / n_layers for architectures that name their size differently.

Found by the cross-framework benchmark (benchmarks/vs_frameworks/REPORT.md, bug 4): ``vanilla_pinn`` (``hidden=``)
and ``bench_mlp`` (``width=`` / ``depth=``) dropped both arguments and built a 4 x 128 network.
"""
from __future__ import annotations

import pytest
import torch

import pinneapple_neural.architectures  # noqa: F401  (registers the model zoo)
from pinneapple_neural.architectures.registry import ModelRegistry


def _linears(model):
    return [(m.in_features, m.out_features) for m in model.modules() if isinstance(m, torch.nn.Linear)]


@pytest.mark.parametrize("name", ["vanilla_pinn", "bench_mlp"])
def test_size_kwargs_are_honoured(name):
    model, report = ModelRegistry.build_with_report(name, in_dim=2, out_dim=1, hidden_dim=16, n_layers=2)
    assert not ({"hidden_dim", "n_layers"} & set(report.dropped)), report.dropped
    assert _linears(model) == [(2, 16), (16, 16), (16, 1)]            # two hidden layers of width 16


def test_the_class_own_size_arguments_win():
    model = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1, hidden=[8, 8, 8], hidden_dim=16, n_layers=2)
    assert _linears(model) == [(2, 8), (8, 8), (8, 8), (8, 1)]


def test_defaults_unchanged_without_size_kwargs():
    assert _linears(ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1))[0] == (2, 128)
    assert len(_linears(ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1))) == 5
