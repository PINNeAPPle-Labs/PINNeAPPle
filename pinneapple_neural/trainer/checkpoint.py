"""Checkpoint save/load for model, optimizer, config, normalizers -- and what resuming needs (#76).

A checkpoint that only holds the weights and the optimizer state is not enough to *continue* a run: the next epoch
would draw different random numbers (dropout, shuffling), the scheduler and the loss balancers would start over, and
the history would begin again at zero. ``Checkpoint.resume`` is that missing part -- the RNG state of torch, numpy
and ``random``, the scaler and scheduler states, the loss-balancer state, the epoch the run stopped at and the history
so far -- so ``Trainer.fit(cfg=TrainConfig(resume_from=...))`` can pick the curve up where it was interrupted.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
import os
import torch


@dataclass
class Checkpoint:
    model_state: dict[str, Any]
    optim_state: dict[str, Any] | None
    cfg: dict[str, Any]
    meta: dict[str, Any]
    normalizers: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None       # everything else a continuation needs; see :func:`rng_state`


def rng_state() -> dict[str, Any]:
    """The RNG state of torch, numpy and ``random`` as plain tensors and numbers, so ``torch.load`` keeps it."""
    import numpy as np

    py = random.getstate()
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": [int(x) for x in py[1]],
        "python_version": int(py[0]),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    np_state = np.random.get_state()
    state["numpy"] = {
        "keys": str(np_state[0]),
        "array": torch.from_numpy(np_state[1].astype("int64")),
        "pos": int(np_state[2]),
        "has_gauss": float(np_state[3]),
        "cached": float(np_state[4]),
    }
    return state


def restore_rng(state: dict[str, Any] | None) -> None:
    """Put torch, numpy and ``random`` back where they were when :func:`rng_state` was taken."""
    if not state:
        return
    import numpy as np

    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("python") is not None:
        random.setstate((state.get("python_version", 3), tuple(int(x) for x in state["python"]), None))
    np_state = state.get("numpy")
    if np_state:
        np.random.set_state((np_state["keys"], np_state["array"].numpy().astype("uint32").copy(),
                             np_state["pos"], np_state["has_gauss"], np_state["cached"]))


def save_checkpoint(path: str, ckpt: Checkpoint) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": ckpt.model_state,
            "optim_state": ckpt.optim_state,
            "cfg": ckpt.cfg,
            "meta": ckpt.meta,
            "normalizers": ckpt.normalizers,
            "resume": ckpt.resume,
        },
        path,
    )

def load_checkpoint(path: str) -> Checkpoint:
    obj = torch.load(path, map_location="cpu")
    return Checkpoint(
        model_state=obj["model_state"],
        optim_state=obj.get("optim_state"),
        cfg=obj.get("cfg", {}),
        meta=obj.get("meta", {}),
        normalizers=obj.get("normalizers"),
        resume=obj.get("resume"),
    )
