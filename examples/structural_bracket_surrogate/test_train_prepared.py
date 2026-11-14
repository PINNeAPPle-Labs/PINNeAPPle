"""Prepared dataset training checks, including geometry-group isolation."""
import numpy as np
from train_prepared import group_split, train


def test_group_split_no_leakage():
    groups = np.array(["g1","g1","g2","g2","g3","g3","g4","g4"])
    a, b = group_split(groups, seed=7)
    assert len(a) and len(b)
    assert not set(groups[a]).intersection(set(groups[b]))


def test_training_prepared_features(tmp_path):
    rng = np.random.default_rng(4)
    c, p = 8, 6
    path = tmp_path / "prepared.npz"
    np.savez(path, branch=rng.normal(size=(c,31)).astype("float32"),
             trunk=rng.normal(size=(c,p,8)).astype("float32"),
             stress_pa=np.abs(rng.normal(size=(c,p))).astype("float32")*1e6,
             group=np.repeat(np.arange(4), 2))
    out = tmp_path / "model.pt"
    result = train(str(path), str(out), epochs=2, seed=4)
    assert out.exists()
    assert np.isfinite(result["val_relative_l2"])
