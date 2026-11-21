"""Interrupting a run and resuming it gives the same loss curve (#76, roadmap item G3).

"Same curve" is the whole acceptance criterion, so the test builds the curve twice: once straight through, and once
as two halves with a checkpoint in the middle. For the two to agree, the checkpoint has to carry more than the
weights and the optimizer -- the dropout inside the model draws from the torch RNG every epoch, and the scheduler
moves the learning rate every epoch, so the RNG state, the epoch counter, the scheduler and the history have to come
back too. A checkpoint that only had weights would diverge at the first epoch of the second half, and this test says
so instead of passing quietly.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneapple_train.trainer import Trainer, TrainConfig


class DropoutNet(torch.nn.Module):
    """Dropout on purpose: without it the test could pass with a checkpoint that never saved the RNG."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Dropout(0.3),
                                       torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


def _loss_fn(model, y_hat, batch):
    return torch.nn.functional.mse_loss(y_hat, batch["y"])


def _fixture(tmp_path):
    """One loader, and a builder that returns a *fresh* trainer with the same init, so only resuming differs."""
    torch.manual_seed(1234)
    x = torch.randn(64, 4)
    y = torch.randn(64, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    def build(run_name, **cfg_kwargs):
        torch.manual_seed(7)                                     # the same weights in every run
        trainer = Trainer(model=DropoutNet(), loss_fn=_loss_fn,
                          scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5))
        cfg = TrainConfig(device="cpu", log_dir=str(tmp_path / run_name), run_name=run_name, seed=0,
                          save_best=False, **cfg_kwargs)
        return trainer, cfg

    return loader, build


def _curve(result):
    return [h["train_total"] for h in result["history"]]


def test_interrupting_and_resuming_gives_the_same_loss_curve(tmp_path):
    loader, build = _fixture(tmp_path)
    path = str(tmp_path / "last.pt")

    # (1) the curve of a run that is never interrupted
    straight, straight_cfg = build("straight", epochs=4)
    ref = straight.fit(loader, straight_cfg)
    assert len(ref["history"]) == 4

    # (2) the same run, stopped after two epochs, with a resumable checkpoint written at the end of each
    first, first_cfg = build("first", epochs=2, save_last=path)
    part = first.fit(loader, first_cfg)
    assert len(part["history"]) == 2

    # (3) and continued from that checkpoint up to the same total
    rest, rest_cfg = build("rest", epochs=4, resume_from=path)
    cont = rest.fit(loader, rest_cfg)

    assert len(cont["history"]) == 4, "the history comes back with the epochs already done, not starting at zero"
    assert [h["epoch"] for h in cont["history"]] == [0, 1, 2, 3]
    assert _curve(cont) == _curve(ref), f"resumed {_curve(cont)} != straight {_curve(ref)}"
    for (name_a, tensor_a), (name_b, tensor_b) in zip(rest.model.state_dict().items(),
                                                       straight.model.state_dict().items()):
        assert name_a == name_b
        assert torch.allclose(tensor_a, tensor_b, atol=1e-10), f"the weights of {name_a} differ after resuming"


def test_the_scheduler_and_the_rng_are_what_make_the_curves_agree(tmp_path):
    """Dropping either from the checkpoint is enough to break the equality above -- so each is asserted on its own."""
    from pinneapple_neural.trainer.checkpoint import load_checkpoint, restore_rng, rng_state

    loader, build = _fixture(tmp_path)
    path = str(tmp_path / "last.pt")
    trainer, cfg = build("run", epochs=1, save_last=path)
    trainer.fit(loader, cfg)

    saved = load_checkpoint(path)
    assert saved.resume is not None
    for key in ("rng", "epoch", "history", "scaler", "scheduler"):
        assert key in saved.resume, key
    assert {"torch", "numpy", "python"} <= set(saved.resume["rng"])
    assert saved.resume["scheduler"]["last_epoch"] == 1, "the scheduler stepped once and the checkpoint knows it"
    assert saved.resume["epoch"] == 0 and len(saved.resume["history"]) == 1

    state = rng_state()                                          # <- taken *before* the draw it must reproduce
    first_draw = torch.rand(3)
    import numpy as np
    np_draw = np.random.rand(3)
    torch.manual_seed(99)
    torch.rand(10)
    np.random.seed(99)
    restore_rng(state)
    assert torch.equal(torch.rand(3), first_draw), "restoring the state has to reproduce the draw"
    assert np.allclose(np.random.rand(3), np_draw), "and the numpy one too"
