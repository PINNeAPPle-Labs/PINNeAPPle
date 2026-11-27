"""``PhysicsModel``: one interface over the model families — fit, predict, uncertainty and export.

The library has five families of physics AI models, each with its own base class and its own convention for
calling it: PINNs take coordinates and return a ``PINNOutput``, neural operators take a function on a grid (or a
sensor vector plus query points) and return an ``OperatorOutput``, graph networks take a ``GraphBatch``, reduced
order models take a snapshot tensor and are fitted in closed form, and hybrids mix a learned surrogate with an
explicit physics model. A caller that wants "the best model for this problem" had to know which of those it got.

This module is the common contract over all of them. A model of any family, wrapped with
:func:`as_physics_model`, answers the same four calls::

    import pinneapple as pp
    from pinneapple_neural.physics_model import as_physics_model

    m = as_physics_model(pp.build_model("vanilla_pinn", in_dim=2, out_dim=1))
    report = m.fit({"x": coords, "y": values}, epochs=200)     # a FitReport
    prediction = m.predict(coords)                             # a Prediction: .mean, .std
    band = m.uncertainty(coords, method="mc_dropout")           # .std is filled, or it refuses
    m.export("model.pt", fmt="torchscript")                    # a Path that reloads

Honesty rules this module holds itself to
-----------------------------------------
* **A capability that the model does not have is refused, with the measured reason.** ``uncertainty`` on a model
  with no hidden ``nn.Linear`` and no ``nn.Dropout`` is not "0 ± 0": Monte Carlo dropout over such a model returns
  exactly zero spread, which is not an uncertainty, so :meth:`PhysicsModel.uncertainty` raises
  :class:`UncertaintyNotAvailable` naming that. :meth:`PhysicsModel.capabilities` says the same thing in advance.
* **The fit reports which kind of fit it was.** ``method="closed_form"`` for the ROM family (a basis, a linear
  operator, a sparse regression — no gradient steps) and ``method="gradient"`` with the epoch history for the
  families that train with descent.
* **A format that does not export says so.** ``fmt="onnx"`` needs the ``onnx`` package; without it the exporter
  raises, and this module passes the reason through instead of writing a file that looks exported.

What is *not* claimed here: the interface does not make the models interchangeable in accuracy. It makes them
interchangeable in *calling convention*, so a benchmark, a twin or an optimizer can loop over the families and
compare what each one actually delivers (``pinneapple_arena`` does the comparison).
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pinneapple_neural.architectures.graphnn.base import GraphBatch, GraphModelBase
from pinneapple_neural.architectures.neural_operators.base import NeuralOperatorBase
from pinneapple_neural.architectures.pinns.base import PINNBase
from pinneapple_neural.architectures.rom.base import ROMBase

__all__ = [
    "PhysicsModel",
    "Prediction",
    "FitReport",
    "UncertaintyNotAvailable",
    "ExportNotAvailable",
    "NotAPhysicsModel",
    "FAMILIES",
    "as_physics_model",
    "family_of",
]

#: The families the interface covers, in the order the issue names them.
FAMILIES: tuple[str, ...] = ("pinn", "operator", "gnn", "rom", "hybrid")


class NotAPhysicsModel(TypeError):
    """The object handed to :func:`as_physics_model` belongs to no family the interface covers."""


class UncertaintyNotAvailable(RuntimeError):
    """The model cannot produce an honest uncertainty by the requested method, and the message says why."""


class ExportNotAvailable(RuntimeError):
    """The requested export format cannot be produced for this model, and the message says why."""


# ------------------------------------------------------------------------------------------------- records
@dataclass
class Prediction:
    """What every family returns from ``predict``.

    * ``mean`` — the prediction, the same tensor the family's own forward would return (batch first).
    * ``std`` — the spread over the samples the method drew, or ``None`` when the call was a point prediction.
    * ``extras`` — whatever the family attaches (per-family diagnostics), never required by a caller.
    * ``method`` — how ``std`` was obtained (``"mc_dropout"``, ``"ensemble"``), ``None`` for a point prediction.
    """

    mean: torch.Tensor
    std: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    method: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mean, torch.Tensor):
            raise TypeError(f"Prediction.mean must be a tensor, got {type(self.mean).__name__}")
        if self.std is not None and self.std.shape != self.mean.shape:
            raise ValueError(f"std has shape {tuple(self.std.shape)}, mean has {tuple(self.mean.shape)}")

    @property
    def has_uncertainty(self) -> bool:
        return self.std is not None

    def interval(self, level: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian lower and upper bounds at ``level`` — requires ``std`` (raises when there is none)."""
        if self.std is None:
            raise UncertaintyNotAvailable("this prediction is a point estimate: call uncertainty() first")
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + level / 2.0))
        return self.mean - z * self.std, self.mean + z * self.std


@dataclass
class FitReport:
    """What every family returns from ``fit``.

    ``method`` is ``"closed_form"`` (no descent: ROM bases, operator inference, sparse regression) or
    ``"gradient"`` (an optimiser ran, and ``history`` holds one entry per epoch). ``seconds`` and ``n_params`` are
    measured on this machine, not estimated.
    """

    family: str
    method: str
    epochs: int
    seconds: float
    n_params: int
    history: list[dict[str, float]] = field(default_factory=list)
    notes: str = ""

    @property
    def final_loss(self) -> float | None:
        """The last recorded training loss, or ``None`` for a closed-form fit (which records no loss curve)."""
        for entry in reversed(self.history):
            if "train_total" in entry:
                return float(entry["train_total"])
        return None

    def improved(self) -> float | None:
        """Ratio of the first to the last recorded training loss (>1 means the loss fell), or ``None``."""
        seen = [float(e["train_total"]) for e in self.history if "train_total" in e]
        if len(seen) < 2 or seen[-1] <= 0:
            return None
        return seen[0] / seen[-1]

    def summary(self) -> str:
        if self.method == "closed_form":
            return (f"{self.family}: closed-form fit, {self.n_params} parameters, {self.seconds:.3f} s")
        first = self.history[0]["train_total"] if self.history else float("nan")
        last = self.final_loss if self.final_loss is not None else float("nan")
        return (f"{self.family}: {self.epochs} epochs, loss {first:.4g} -> {last:.4g}, "
                f"{self.n_params} parameters, {self.seconds:.3f} s")


# ------------------------------------------------------------------------------------------------- helpers
def _count_params(module: Any) -> int:
    if not isinstance(module, nn.Module):
        return 0
    return int(sum(p.numel() for p in module.parameters()))


def _dropout_capacity(module: nn.Module) -> tuple[int, int]:
    """``(hidden linear layers, dropout modules)`` — the two things MC dropout needs to produce a real spread."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    dropouts = [m for m in module.modules() if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                                                              nn.AlphaDropout, nn.FeatureAlphaDropout))]
    return max(len(linears) - 1, 0), len(dropouts)


def _unwrap(out: Any) -> torch.Tensor:
    """The tensor in a family output (``PINNOutput``/``OperatorOutput``/``GraphOutput``/``ROMOutput``) or itself."""
    if isinstance(out, torch.Tensor):
        return out
    y = getattr(out, "y", None)
    if isinstance(y, torch.Tensor):
        return y
    raise TypeError(f"model output {type(out).__name__} carries no tensor in '.y'")


class _TensorForward(nn.Module):
    """Bind a family model (and any extra inputs) behind a plain ``forward(x) -> tensor``.

    Training through :class:`~pinneapple_neural.trainer.Trainer` and exporting to TorchScript both need a module
    that takes one tensor and returns one tensor; every family in this library returns a dataclass instead. This
    wrapper is the single place that difference is absorbed, and it keeps the constant inputs (a DeepONet's query
    points, for example) so a traced graph carries them.
    """

    def __init__(self, model: nn.Module, **constant: Any):
        super().__init__()
        self.model = model
        self._wrapped = list(constant)
        for k, v in constant.items():
            setattr(self, f"_const_{k}", v.detach().clone() if torch.is_tensor(v) else v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _unwrap(self.model(x, **{k: getattr(self, f"_const_{k}") for k in self._wrapped}))


def _as_batches(data: Any) -> list[dict[str, Any]]:
    """A training payload as a list of dict batches: a mapping, an ``(x, y)`` tuple, or an iterable of those."""
    if isinstance(data, Mapping):
        return [dict(data)]
    if isinstance(data, (tuple, list)) and len(data) == 2 and all(torch.is_tensor(v) for v in data):
        return [{"x": data[0], "y": data[1]}]
    if isinstance(data, (list, tuple)) or hasattr(data, "__iter__"):
        batches = []
        for item in data:
            if isinstance(item, Mapping):
                batches.append(dict(item))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                batches.append({"x": item[0], "y": item[1]})
            else:
                raise TypeError(f"a batch must be a mapping or an (x, y) pair, got {type(item).__name__}")
        if not batches:
            raise ValueError("empty training payload")
        return batches
    raise TypeError(f"cannot read a training payload out of {type(data).__name__}")


def _mse_loss(model: nn.Module, y_hat: torch.Tensor, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The default loss: mean squared error against ``batch["y"]``. The Trainer hands the unwrapped tensor."""
    if "y" not in batch:
        raise KeyError("the default loss needs the target under 'y'; pass loss_fn= for anything else")
    y = batch["y"]
    if y.shape != y_hat.shape:
        raise ValueError(f"prediction has shape {tuple(y_hat.shape)}, target has {tuple(y.shape)}")
    return {"total": torch.mean((y_hat - y) ** 2)}


# ------------------------------------------------------------------------------------------------- interface
class PhysicsModel:
    """The contract. One ``fit``, one ``predict``, one ``uncertainty``, one ``export``, whatever the family.

    Instances are built by :func:`as_physics_model`, which picks the adapter for the model it is given; a family
    that grows a new implementation only needs to subclass the matching adapter. The class is not abstract —
    subclasses set ``family`` and implement the four calls (and ``capabilities`` says which of them are honest for
    this instance).
    """

    #: ``"pinn"``, ``"operator"``, ``"gnn"``, ``"rom"`` or ``"hybrid"``.
    family: str = ""

    def __init__(self, model: Any, *, name: str = "", backend: str = "pytorch"):
        self.model = model
        self.name = name or type(model).__name__
        self.backend = backend

    # -- identity --------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} family={self.family!r} params={self.n_params}>"

    @property
    def n_params(self) -> int:
        """Number of trainable parameters, counted from the model."""
        return _count_params(self.model)

    def describe(self) -> dict[str, Any]:
        """The same fields for every family: what it is, what it can do, and how it is called."""
        return {"name": self.name, "family": self.family, "backend": self.backend, "n_params": self.n_params,
                "class": f"{type(self.model).__module__}.{type(self.model).__name__}",
                "capabilities": self.capabilities()}

    # -- capabilities ----------------------------------------------------------------------------------
    def capabilities(self) -> dict[str, str]:
        """``{"fit"|"predict"|"uncertainty"|"export": "yes" | the reason it is not available}``.

        Callers are expected to read this instead of catching exceptions: an optimizer that loops over families
        should skip the ones that cannot give it an uncertainty rather than discover that at the end of a run.
        """
        raise NotImplementedError

    # -- the four calls --------------------------------------------------------------------------------
    def fit(self, data: Any, **options: Any) -> FitReport:
        """Fit on ``data`` and report how it was fitted. See the adapter for what ``data`` is per family."""
        raise NotImplementedError

    def predict(self, x: Any, **options: Any) -> Prediction:
        """The point prediction for ``x``."""
        raise NotImplementedError

    def uncertainty(self, x: Any, *, method: str = "mc_dropout", n_samples: int = 100, seed: int | None = 0,
                    members: Sequence[Any] | None = None, **options: Any) -> Prediction:
        """``predict`` plus a spread, by ``method``:

        * ``"mc_dropout"`` — Monte Carlo dropout over the model's hidden activations (Gal & Ghahramani, 2016).
          Refused for a model with no hidden ``nn.Linear`` and no ``nn.Dropout``: the samples would be identical.
        * ``"ensemble"`` — the spread across independently fitted ``members`` (models of this family). Works for
          any family, including the ones MC dropout cannot help.
        """
        raise NotImplementedError

    def export(self, path: str | Path, *, fmt: str = "torchscript") -> Path:
        """Write the model to ``path`` and return it. ``fmt``: ``"torchscript"``, ``"onnx"`` or ``"state_dict"``."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- torch base
class _TorchPhysicsModel(PhysicsModel):
    """Shared machinery for the four families that are ``nn.Module``: gradient fit, MC dropout, TorchScript.

    Subclasses say how a batch reaches the model (:meth:`_tensor_module`) and which extra inputs are constant
    (:attr:`_constant`), and nothing else — the four calls are implemented once, here.
    """

    def __init__(self, model: nn.Module, *, name: str = "", constant: Mapping[str, Any] | None = None):
        super().__init__(model, name=name)
        self._constant: dict[str, Any] = dict(constant or {})

    # -- how a call reaches the model ------------------------------------------------------------------
    def _constants_for(self, payload: Any) -> dict[str, Any]:
        """The constant inputs for this call: the bound ones, overridden by any the payload carries by name."""
        const = dict(self._constant)
        if isinstance(payload, Mapping):
            for key in self._constant:
                if payload.get(key) is not None:
                    const[key] = payload[key]
        return const

    def _tensor_module(self, payload: Any = None) -> _TensorForward:
        return _TensorForward(self.model, **self._constants_for(payload))

    def _mc_dropout_reason(self) -> str | None:
        """Why MC dropout cannot be honest for this instance, or ``None`` when it can."""
        hidden, dropouts = _dropout_capacity(self.model)
        if hidden or dropouts:
            return None
        return (f"{self.name} has {hidden} hidden nn.Linear and {dropouts} nn.Dropout layers: Monte Carlo dropout "
                "would return an exactly zero spread, which is not an uncertainty. Pass method='ensemble' with "
                "fitted members of the same family, or use a model with a dropout-able hidden layer.")

    # -- capabilities ----------------------------------------------------------------------------------
    def capabilities(self) -> dict[str, str]:
        reason = self._mc_dropout_reason()
        return {"fit": "yes", "predict": "yes", "uncertainty": "yes" if reason is None else f"no: {reason}",
                "export": "yes"}

    # -- fit -------------------------------------------------------------------------------------------
    def fit(self, data: Any, *, epochs: int = 100, lr: float = 1e-3, loss_fn: Any | None = None,
            val: Any = None, device: str = "cpu", seed: int | None = 0, **config: Any) -> FitReport:
        """Gradient fit through :class:`~pinneapple_neural.trainer.Trainer`.

        ``data`` is a mapping, an ``(x, y)`` pair, or an iterable of those (one per epoch step). Anything the batch
        carries beyond ``x``/``y`` reaches ``loss_fn`` unchanged, so a physics residual, a boundary set or a weight
        can ride along in the batch.
        """
        from pinneapple_neural.trainer import TrainConfig, Trainer

        batches = _as_batches(data)
        val_batches = None if val is None else _as_batches(val)
        net = self._tensor_module(batches[0])
        loss = loss_fn or _mse_loss
        trainer = Trainer(net, loss)
        t0 = time.perf_counter()
        out = trainer.fit(batches, val_batches,
                          TrainConfig(epochs=int(epochs), lr=float(lr), device=device, seed=seed,
                                      save_best=False, **config))
        return FitReport(family=self.family, method="gradient", epochs=len(out.get("history", [])),
                         seconds=time.perf_counter() - t0, n_params=self.n_params,
                         history=list(out.get("history", [])))

    # -- predict ---------------------------------------------------------------------------------------
    def predict(self, x: Any, **options: Any) -> Prediction:
        net = self._tensor_module(x)
        tensor = _primary_tensor(x)
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                y = net(tensor)
        finally:
            net.train(was_training)
        return Prediction(mean=y)

    # -- uncertainty -----------------------------------------------------------------------------------
    def uncertainty(self, x: Any, *, method: str = "mc_dropout", n_samples: int = 100, seed: int | None = 0,
                    members: Sequence[Any] | None = None, **options: Any) -> Prediction:
        tensor = _primary_tensor(x)
        if method == "mc_dropout":
            reason = self._mc_dropout_reason()
            if reason is not None:
                raise UncertaintyNotAvailable(reason)
            from pinneapple_analysis.uncertainty.mc_dropout import MCDropoutConfig, MCDropoutWrapper

            net = self._tensor_module(x)
            wrapper = MCDropoutWrapper(net, MCDropoutConfig(n_samples=int(n_samples), seed=seed))
            result = wrapper.predict_with_uncertainty(tensor)
            return Prediction(mean=result.mean, std=result.std, method="mc_dropout",
                              extras={"n_samples": int(n_samples), "config": dict(result.metadata)})
        if method == "ensemble":
            if not members or len(members) < 2:
                raise UncertaintyNotAvailable(
                    f"method='ensemble' needs at least two fitted members, got {0 if not members else len(members)}")
            models = [m if isinstance(m, PhysicsModel) else as_physics_model(m) for m in members]
            stack = torch.stack([m.predict(x, **options).mean for m in models], dim=0)
            return Prediction(mean=stack.mean(dim=0), std=stack.std(dim=0, unbiased=True), method="ensemble",
                              extras={"members": len(models)})
        raise ValueError(f"unknown method {method!r}: use 'mc_dropout' or 'ensemble'")

    # -- export ----------------------------------------------------------------------------------------
    def export(self, path: str | Path, *, fmt: str = "torchscript", example: torch.Tensor | None = None,
               **options: Any) -> Path:
        """Write the model. ``torchscript`` traces a one-tensor-in/one-tensor-out graph (the family dataclass is
        unwrapped), ``onnx`` needs the ``onnx`` package, ``state_dict`` writes the weights and the class."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "state_dict":
            torch.save({"state_dict": self.model.state_dict(), "class": type(self.model).__qualname__,
                        "module": type(self.model).__module__, "family": self.family,
                        "constant": {k: (v.detach() if torch.is_tensor(v) else v) for k, v in self._constant.items()}},
                       path)
            return path
        net = self._tensor_module(example).eval()
        sample = _primary_tensor(example if example is not None else self._example_input())
        if fmt == "torchscript":
            import warnings

            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)     # jit.trace still works; it is deprecated
                torch.jit.trace(net, sample, check_trace=False).save(str(path))
            return path
        if fmt == "onnx":
            try:
                torch.onnx.export(net, (sample,), str(path), input_names=["input"], output_names=["output"],
                                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=False)
            except Exception as exc:                                   # the exporter's own reason, passed through
                raise ExportNotAvailable(f"{self.name} -> onnx failed: {type(exc).__name__}: {exc}") from exc
            return path
        raise ValueError(f"unknown fmt {fmt!r}: use 'torchscript', 'onnx' or 'state_dict'")

    def _example_input(self) -> torch.Tensor:
        """A representative input for tracing; adapters override it with the shape their family expects."""
        raise ExportNotAvailable(f"{self.name}: pass example=<input tensor> to trace the graph")


def _primary_tensor(x: Any) -> torch.Tensor:
    """The tensor a call forwards: ``x`` itself, or the first tensor of a mapping (``x``/``u``/``x_col``)."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, Mapping):
        for key in ("x", "u", "x_col", "snapshots"):
            if torch.is_tensor(x.get(key)):
                return x[key]
        tensors = [v for v in x.values() if torch.is_tensor(v)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(f"cannot find the input tensor in {type(x).__name__}")


# ------------------------------------------------------------------------------------------------- adapters
class PINNModel(_TorchPhysicsModel):
    """Adapter for :class:`~pinneapple_neural.architectures.pinns.base.PINNBase`: ``forward(*coords) -> PINNOutput``."""

    family = "pinn"

    def _example_input(self) -> torch.Tensor:
        shape = getattr(self.model, "in_dim", None)
        if shape is None:
            raise ExportNotAvailable(f"{self.name}: pass example=<coords tensor> to trace the graph")
        return torch.zeros(2, int(shape))


class OperatorModel(_TorchPhysicsModel):
    """Adapter for :class:`~pinneapple_neural.architectures.neural_operators.base.NeuralOperatorBase`.

    A grid operator (FNO and friends) forwards one tensor; a point operator such as DeepONet also needs its query
    points, which are bound at construction (``as_physics_model(model, coords=...)``) or taken from the payload
    under ``coords`` on each call.
    """

    family = "operator"

    def _constants_for(self, payload: Any) -> dict[str, Any]:
        const = super()._constants_for(payload)
        if "coords" in _signature_params(self.model) and const.get("coords") is None:
            raise ValueError(f"{self.name} takes query points: bind them with as_physics_model(model, coords=...) "
                             "or pass a mapping with 'coords'")
        return const

    def _example_input(self) -> torch.Tensor:
        raise ExportNotAvailable(f"{self.name}: pass example=<input on its grid or sensor vector> to trace the graph")


class GraphModel(_TorchPhysicsModel):
    """Adapter for :class:`~pinneapple_neural.architectures.graphnn.base.GraphModelBase`: ``forward(GraphBatch)``."""

    family = "gnn"

    def _torchify(self, x: Any) -> GraphBatch:
        """A ``GraphBatch``, or a mapping with ``x``/``node_features`` and ``edge_index`` turned into one."""
        if isinstance(x, GraphBatch):
            return x
        if isinstance(x, Mapping):
            node = x.get("x", x.get("node_features"))
            if node is None or "edge_index" not in x:
                raise KeyError("a graph batch needs 'x' (or 'node_features') and 'edge_index'")
            return GraphBatch(x=node, edge_index=x["edge_index"],
                              pos=x.get("pos"), edge_attr=x.get("edge_attr", x.get("edge_features")),
                              mask=x.get("mask"))
        raise TypeError(f"a graph model takes a GraphBatch or a mapping, got {type(x).__name__}")

    def _tensor_module(self, payload: Any = None) -> _TensorForward:  # pragma: no cover - no flat tensor form
        raise ExportNotAvailable(
            f"{self.name}: a graph model takes a GraphBatch (node features plus an edge list), which a traced "
            "one-tensor graph cannot carry; use fmt='state_dict'")

    def predict(self, x: Any, **options: Any) -> Prediction:
        batch = self._torchify(x)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(batch)
        finally:
            self.model.train(was_training)
        return Prediction(mean=_unwrap(out), extras={"node_count": int(batch.x.shape[1])})

    def fit(self, data: Any, *, epochs: int = 100, lr: float = 1e-3, loss_fn: Any | None = None,
            device: str = "cpu", seed: int | None = 0, **config: Any) -> FitReport:
        """Gradient fit over graph batches with a plain Adam loop (a ``GraphBatch`` is not a batch of tensors, so
        the ``Trainer``'s ``model(x)`` contract does not apply). Each item of ``data`` is a mapping carrying the
        graph and the target under ``y``."""
        items = _as_batches(data)
        opt = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        loss = loss_fn or _mse_loss
        history: list[dict[str, float]] = []
        t0 = time.perf_counter()
        for _ in range(int(epochs)):
            self.model.train()
            total, seen = 0.0, 0
            for item in items:
                graph = self._torchify(item)
                opt.zero_grad(set_to_none=True)
                out = self.model(graph, y_true=item.get("y"), return_loss=False)
                y_hat = _unwrap(out)
                losses = loss(self.model, y_hat, item)
                losses["total"].backward()
                opt.step()
                n = int(graph.x.shape[0])
                total += float(losses["total"].item()) * n
                seen += n
            history.append({"train_total": total / max(seen, 1)})
        return FitReport(family=self.family, method="gradient", epochs=len(history),
                         seconds=time.perf_counter() - t0, n_params=self.n_params, history=history)

    def _example_input(self) -> torch.Tensor:  # pragma: no cover - see _tensor_module
        raise ExportNotAvailable(f"{self.name}: use fmt='state_dict' for a graph model")


class ROMModel(_TorchPhysicsModel):
    """Adapter for :class:`~pinneapple_neural.architectures.rom.base.ROMBase`: ``fit(X)`` then ``forward(X)``.

    The ROM family is fitted in closed form — a basis from an SVD, a linear operator by ridge regression, a sparse
    coefficient set — so :meth:`fit` calls the model's own ``fit`` and reports ``method="closed_form"``. Only the
    gradient-fitted members (``neural_rom``, the corrector of ``rom_hybrid``) reach the ``Trainer``, and they do
    so by passing ``epochs=`` explicitly.
    """

    family = "rom"

    def fit(self, data: Any, *, epochs: int = 0, lr: float = 1e-3, loss_fn: Any | None = None,
            device: str = "cpu", seed: int | None = 0, **options: Any) -> FitReport:
        snapshots = data
        if isinstance(data, Mapping):
            snapshots = data.get("snapshots", data.get("x"))
        if not torch.is_tensor(snapshots):
            raise TypeError(f"a ROM is fitted on a snapshot tensor (N, D) or (B, T, D), got {type(snapshots).__name__}")
        t0 = time.perf_counter()
        native = getattr(self.model, "fit", None)
        if native is None:
            raise NotImplementedError(f"{self.name} has no fit(); nothing in this family can be fitted in closed form")
        native(snapshots)
        return FitReport(family=self.family, method="closed_form", epochs=0,
                         seconds=time.perf_counter() - t0, n_params=self.n_params,
                         notes=f"closed form on {tuple(snapshots.shape)} snapshots")

    def predict(self, x: Any, **options: Any) -> Prediction:
        tensor = _primary_tensor(x)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(tensor)
        finally:
            self.model.train(was_training)
        return Prediction(mean=_unwrap(out), extras={"extras": dict(getattr(out, "extras", {}) or {})})

    def _mc_dropout_reason(self) -> str | None:
        return (f"{self.name} is a reduced-order model: a closed-form projection with no stochastic layer, so "
                "Monte Carlo dropout has nothing to sample. Pass method='ensemble' with members fitted on "
                "different samples (or different ranks) of the same snapshots.")


class HybridModel(PhysicsModel):
    """Adapter for the hybrid workflows: a learned surrogate plus an explicit physics post-model.

    Two implementations are covered. :class:`~pinneapple_neural.workflows.hybrid_surrogate_physics.
    HybridSurrogatePhysics` is numpy-only and already reports a Monte Carlo band around the derived quantity, so
    its uncertainty is the band the workflow computed (through the post-model, not around the surrogate's outputs).
    :class:`~pinneapple_neural.architectures.rom.rom_hybrid.ROMHybrid` is a ROM with a neural corrector and is
    handled by :class:`ROMModel`'s closed-form fit plus the corrector's gradient path.
    """

    family = "hybrid"

    def capabilities(self) -> dict[str, str]:
        return {"fit": "yes", "predict": "yes",
                "uncertainty": "yes" if _hybrid_band(self.model) else
                               "no: the surrogate gives a point prediction only (no std, no sampler)",
                "export": "no: the hybrid is a numpy pipeline (surrogate + post-model), not a torch graph; "
                          "state()/from_state() is its persistence"}

    def fit(self, data: Any, **options: Any) -> FitReport:
        """``data`` is ``{"params": ..., "intermediates": {name: values}}``, the payload the workflow's own
        ``fit(params, intermediates)`` takes."""
        if not isinstance(data, Mapping) or "params" not in data or "intermediates" not in data:
            raise TypeError("a hybrid is fitted on {'params': ..., 'intermediates': {name: values}}")
        t0 = time.perf_counter()
        self.model.fit(data["params"], data["intermediates"])
        return FitReport(family=self.family, method="closed_form", epochs=0,
                         seconds=time.perf_counter() - t0, n_params=0,
                         notes="surrogate fitted; the post-model is never trained")

    def predict(self, x: Any, **options: Any) -> Prediction:
        out = self.model.predict(x, n_samples=int(options.get("n_samples", 0)))
        mean = torch.as_tensor(out.derived_at_mean, dtype=torch.float32)
        return Prediction(mean=mean, extras={"intermediates": {k: torch.as_tensor(v)
                                                               for k, v in out.intermediates.items()},
                                             "post_model": dict(out.post_model)})

    def uncertainty(self, x: Any, *, method: str = "mc_dropout", n_samples: int = 1000, seed: int | None = 0,
                    members: Sequence[Any] | None = None, **options: Any) -> Prediction:
        """Only the workflow's own Monte Carlo band is offered: the post-model is nonlinear, so the band is the one
        pushed through it, not a Gaussian around ``predict``. ``mc_dropout`` is not a method a numpy surrogate has."""
        if method != "mc_dropout":
            raise ValueError("the hybrid reports its own Monte Carlo band; use method='mc_dropout'")
        if not _hybrid_band(self.model):
            raise UncertaintyNotAvailable(
                f"{self.name}: the surrogate gives a point prediction (no std, no sampler), so there is no "
                "distribution to push through the post-model")
        out = self.model.predict(x, n_samples=int(n_samples), seed=seed,
                                 interval=float(options.get("interval", 0.95)))
        mean = torch.as_tensor(out.derived_mean, dtype=torch.float32)
        return Prediction(mean=mean, std=torch.as_tensor(out.derived_std, dtype=torch.float32),
                          method="monte_carlo",
                          extras={"n_samples": int(out.n_samples), "interval": float(out.interval),
                                  "post_model": dict(out.post_model)})

    def export(self, path: str | Path, *, fmt: str = "torchscript") -> Path:  # pragma: no cover - refused
        raise ExportNotAvailable(
            "a hybrid is a numpy surrogate plus a post-model, not a torch graph; persist it with state()/"
            "from_state() (the post-model is passed back on reload, which is the point of keeping it explicit)")


def _hybrid_band(model: Any) -> bool:
    """Whether the fitted surrogate carries a predictive distribution the post-model can be pushed through.

    Probed with one zero row of the fitted parameter width (the surrogate's own ``predict`` is what decides), so
    the answer is the surrogate's, not a guess from its class name. Only the probe itself is guarded: a surrogate
    that refuses a zero row says "no band", while a mistake in this module still raises.
    """
    from pinneapple_neural.workflows.hybrid_surrogate_physics import (
        HybridSurrogatePhysics,
        _as_prediction,
    )

    if not isinstance(model, HybridSurrogatePhysics) or not getattr(model, "fitted", False):
        return False
    probe = np.zeros((1, len(model.param_names or [0])), dtype=float)
    try:
        prediction = model.surrogate.predict(probe)
    except (ValueError, TypeError, RuntimeError):                   # the surrogate refuses to be probed: no band
        return False
    return bool(_as_prediction(prediction, 1).has_distribution)


def _signature_params(model: nn.Module) -> tuple[str, ...]:
    import inspect

    try:
        return tuple(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):                                # pragma: no cover - a C-level forward
        return ()


# -------------------------------------------------------------------------------------------------- factory
def family_of(model: Any) -> str:
    """The family a model belongs to, or raise :class:`NotAPhysicsModel` naming what it is."""
    from pinneapple_neural.workflows.hybrid_surrogate_physics import HybridSurrogatePhysics

    if isinstance(model, HybridSurrogatePhysics):
        return "hybrid"
    if isinstance(model, PINNBase):
        return "pinn"
    if isinstance(model, NeuralOperatorBase):
        return "operator"
    if isinstance(model, GraphModelBase):
        return "gnn"
    if isinstance(model, ROMBase):
        return "rom"
    raise NotAPhysicsModel(
        f"{type(model).__module__}.{type(model).__name__} belongs to no family this interface covers "
        f"({', '.join(FAMILIES)}): it must subclass PINNBase, NeuralOperatorBase, GraphModelBase, ROMBase, "
        "or be a HybridSurrogatePhysics")


def as_physics_model(model: Any, *, name: str = "", **constant: Any) -> PhysicsModel:
    """Wrap ``model`` in the adapter for its family.

    ``constant`` binds inputs the family needs on every call but that the interface does not own — a DeepONet's
    query ``coords``, for example::

        op = as_physics_model(deeponet, coords=torch.linspace(0, 1, 64)[:, None])
        op.predict(sensor_values)                      # (B, sensor_dim) -> (B, 64, out_dim)
    """
    family = family_of(model)
    adapter = {"pinn": PINNModel, "operator": OperatorModel, "gnn": GraphModel, "rom": ROMModel,
               "hybrid": HybridModel}[family]
    if adapter is HybridModel:
        return HybridModel(model, name=name)
    return adapter(model, name=name, constant=constant)
