"""BaseModel and ModelOutput for Pinneapple model family.

This module defines a minimal *contract* that enables the Arena and Trainer to
run heterogeneous model families in a consistent way.

Key idea
--------
Every model can be executed in two ways:
  1) ``forward(x)``: classic PyTorch signature.
  2) ``forward_batch(batch)``: takes a dict batch (PINN / operator / graph / TS).

The default ``forward_batch`` implementation simply picks ``batch['x']`` (or
``batch['x_col']``) and calls ``forward(x)``.

Models that need different inputs (coords+params, graph objects, sequences,
fields, etc.) should override ``forward_batch``.
"""
from __future__ import annotations
import functools
import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    y: torch.Tensor
    losses: Optional[Dict[str, torch.Tensor]] = None
    extras: Optional[Dict[str, Any]] = None


class BaseModel(nn.Module):
    """
    Unified base class for all Pinneapple models.
    """
    family: str = "generic"
    name: str = "base"

    def __init_subclass__(cls, **kw):
        """Record the arguments of the outermost constructor call, so a checkpoint can rebuild the model (#941)."""
        super().__init_subclass__(**kw)
        init = cls.__dict__.get("__init__")
        if init is None or getattr(init, "_pp_records_args", False):
            return

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            if "_pp_init_args" not in self.__dict__:          # the outermost call: the one the user made
                try:
                    b = inspect.signature(init).bind(self, *args, **kwargs)
                    b.apply_defaults()
                    a = dict(b.arguments)
                    a.pop("self", None)
                    for name, p in inspect.signature(init).parameters.items():
                        if p.kind is inspect.Parameter.VAR_KEYWORD:
                            a.update(a.pop(name, {}) or {})
                        elif p.kind is inspect.Parameter.VAR_POSITIONAL and a.get(name):
                            a = None                          # positional varargs cannot be replayed by name
                            break
                        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
                            a.pop(name, None)
                except TypeError:
                    a = None
                self.__dict__["_pp_init_args"] = a
            init(self, *args, **kwargs)

        __init__._pp_records_args = True
        cls.__init__ = __init__

    def forward(self, *args, **kwargs) -> ModelOutput | torch.Tensor:
        raise NotImplementedError

    def forward_batch(self, batch: Dict[str, Any]) -> ModelOutput | torch.Tensor:
        """Default batch execution.

        The Trainer/Arena can provide a dict batch. This default implementation
        mirrors the Trainer's convention:
          - use ``batch['x']`` if present
          - otherwise use ``batch['x_col']`` (PINN collocation)
        and then call ``forward(x)``.
        """
        x = batch.get("x")
        if x is None:
            x = batch.get("x_col")
        if x is None:
            raise KeyError("forward_batch expects batch to include 'x' or 'x_col'.")
        return self.forward(x)

    # ------------------------------------------------------------------
    # Checkpoint I/O — available to all model families
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model weights + metadata as a pinneapple checkpoint.

        The checkpoint dict contains:
          - ``"state_dict"``: model weights
          - ``"class_name"``: fully qualified class name
          - ``"metadata"``: user-provided dict
          - ``"registry_name"``: the model's key in ``ModelRegistry`` (``None`` if it is not registered)
          - ``"model_kwargs"``: the constructor arguments, so :func:`load_checkpoint` can rebuild the model

        Returns the path where the file was saved.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ckpt = {
            "state_dict": self.state_dict(),
            "class_name": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "metadata": metadata or {},
            # enough to rebuild the model with pinneapple.load_checkpoint(path) (#941)
            "registry_name": _registry_name(type(self)),
            "model_kwargs": self.__dict__.get("_pp_init_args"),
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load_checkpoint(cls, path: str, **init_kwargs) -> "BaseModel":
        """Load a model from a pinneapple checkpoint.

        Usage::
            model = VanillaPINN.load_checkpoint("model.pt", in_dim=2, out_dim=1, hidden=[64,64,64])
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**init_kwargs)
        model.load_state_dict(ckpt["state_dict"])
        return model


def _registry_name(cls) -> Optional[str]:
    try:
        from .registry import ModelRegistry
        return next((k for k, sp in ModelRegistry._REGISTRY.items() if sp.cls is cls), None)
    except Exception:                                         # the registry is optional here
        return None


def load_checkpoint(path: str, *, map_location: str = "cpu", **override_kwargs) -> nn.Module:
    """Rebuild a model from a checkpoint written by :meth:`BaseModel.save_checkpoint` and load its weights.

    The class comes from ``class_name`` and the constructor arguments from ``model_kwargs`` (checkpoints written
    since #941); ``override_kwargs`` replace or supply arguments, which older checkpoints need. The checkpoint's
    ``metadata`` is attached to the model as ``model.checkpoint_metadata``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    mod, _, name = ckpt["class_name"].rpartition(".")
    cls = getattr(importlib.import_module(mod), name)
    kwargs = {**(ckpt.get("model_kwargs") or {}), **override_kwargs}
    if ckpt.get("model_kwargs") is None and not override_kwargs:
        raise ValueError(f"{path} was written without the constructor arguments (before #941): pass them, e.g. "
                         f"load_checkpoint(path, in_dim=2, out_dim=1, ...), or rebuild the {name} and use "
                         "load_state_dict")
    model = cls(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.checkpoint_metadata = ckpt.get("metadata", {})
    return model

