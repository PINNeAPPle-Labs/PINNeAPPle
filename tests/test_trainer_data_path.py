"""The data path of pinneapple_neural.trainer: DataModule batches keep their keys, a preprocessing pipeline fits each
step on the output of the previous one, forward fill leaves no NaN, and load_checkpoint reads the Trainer's own
checkpoints (the bugs examples/trainer/02 and 04 hit)."""
from __future__ import annotations

import torch

from pinneapple_neural.trainer import DataModule, SplitSpec
from pinneapple_neural.trainer.inference import load_checkpoint
from pinneapple_neural.trainer.preprocess import MissingValueStep, NormalizeStep, PreprocessPipeline


def test_datamodule_batches_keep_x_and_y():
    items = [{"x": torch.full((4, 3), float(i)), "y": torch.tensor([float(i)])} for i in range(20)]
    dm = DataModule(dataset=items, split=SplitSpec(method="time", train=0.7, val=0.2, test=0.1, seed=0),
                    batch_size=8)
    b = next(iter(dm.train_dataloader()))
    assert b["x"].shape == (8, 4, 3) and b["y"].shape == (8, 1)


def test_forward_fill_leaves_no_nan_and_keeps_the_values():
    x = torch.tensor([[float("nan")], [1.0], [float("nan")], [3.0]])
    out = MissingValueStep(key="x", strategy="ffill").apply({"x": x})["x"]
    assert out.flatten().tolist() == [1.0, 1.0, 1.0, 3.0]
    assert torch.isnan(x[0, 0])                               # the input is not modified


def test_pipeline_fits_the_normalizer_on_filled_data():
    batches = [{"x": torch.tensor([[[float("nan")], [2.0], [4.0]]])}]
    pipe = PreprocessPipeline(steps=[MissingValueStep(key="x", strategy="ffill"),
                                     NormalizeStep(key="x", dim=(0, 1), store_key="x_scaler")]).fit(batches)
    out = pipe.apply(batches[0])["x"]
    assert not torch.isnan(out).any()
    assert abs(float(out.mean())) < 1e-6


def test_load_checkpoint_reads_trainer_and_model_checkpoint_formats(tmp_path):
    net = torch.nn.Linear(2, 1)
    for key in ("model_state", "model"):
        p = tmp_path / f"{key}.pt"
        torch.save({key: net.state_dict()}, p)
        other = torch.nn.Linear(2, 1)
        load_checkpoint(other, str(p))
        assert torch.equal(other.weight, net.weight)
