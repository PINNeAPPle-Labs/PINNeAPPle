"""Regression test: Arena must not feed graph models their own targets.

Arena used to build MeshGraphNet node features as ``[coords, targets + noise]``.
The same node features are reused for evaluation (``_eval_graph``), so a graph
model could score almost perfectly by reading the answer back from its input,
inflating its place in the ranking. Node features must be inputs only.
"""
from __future__ import annotations

import numpy as np

import pinneapple_arena.arena as arena_mod
from pinneapple_arena.arena import Arena
from pinneapple_arena.config import ArenaConfig, DatasetConfig, ProblemConfig


def _arena(**kw) -> Arena:
    return Arena(ArenaConfig(problem=ProblemConfig(name="poisson_2d", n_mesh_nodes=80, **kw), models=[]),
                 device="cpu")


def test_builtin_problem_node_features_are_coordinates_only():
    a = _arena()
    a._prepare_data()
    d = a._data
    assert d["node_feats"].shape == (80, d["in_dim"])
    np.testing.assert_array_equal(d["node_feats"], d["node_xy"])


def test_builtin_problem_node_features_carry_no_target_information():
    a = _arena()
    a._prepare_data()
    d = a._data
    # No feature column may reproduce a target column (the old code had corr ~ 1).
    for j in range(d["node_targets"].shape[1]):
        for k in range(d["node_feats"].shape[1]):
            corr = np.corrcoef(d["node_feats"][:, k], d["node_targets"][:, j])[0, 1]
            assert abs(corr) < 0.99


def test_dataset_path_node_features_are_inputs_only(monkeypatch):
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 1, (120, 3))
    Y_train = np.stack([np.sin(X_train[:, 0]), X_train[:, 1] ** 2], axis=1)
    X_val, Y_val = X_train[:20], Y_train[:20]

    def fake_loader(*_args, **_kwargs):
        return X_train, Y_train, X_val, Y_val, ["u", "v"]

    monkeypatch.setattr(arena_mod, "load_pinneapple_dataset", fake_loader)
    cfg = ArenaConfig(problem=ProblemConfig(name="poisson_2d"), models=[],
                      dataset=DatasetConfig(dataset_id="fake", input_fields=["x", "y", "z"],
                                            output_fields=["u", "v"]))
    a = Arena(cfg, device="cpu")
    a._prepare_data()
    d = a._data
    assert d["node_feats"].shape[1] == X_train.shape[1]
    assert np.isin(d["node_feats"], X_train).all()
