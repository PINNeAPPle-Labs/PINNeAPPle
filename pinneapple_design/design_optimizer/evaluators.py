"""Evaluators that plug models and solvers into a :class:`~.study.DesignProblem`.

* :class:`ZooEvaluator`: a pre-trained model of ``pp.zoo`` (its inputs come from design variables of the same name,
  or from fixed values); designs outside the model's validity range are flagged invalid.
* :class:`TorchModuleEvaluator`: any ``torch.nn.Module`` (MLP surrogate, PINN, neural operator, differentiable
  physics), differentiable, with optional input/output transforms written in torch.
* :class:`CommandEvaluator`: an external solver run as a command: designs go to a CSV file, the command writes a
  CSV of outputs (one row per design, same order). This is the contract of a solver runner.
* :func:`from_model_ref`: an evaluator from a ``ModelRef``-like record (``uri`` ``zoo:<name>``, a TorchScript file,
  or a command), so a model published elsewhere (a surrogate studio, a benchmark arena) can be optimised directly.
"""
from __future__ import annotations

import csv
import os
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .study import VALID, DesignSpace, Evaluator

__all__ = ["ZooEvaluator", "TorchModuleEvaluator", "CommandEvaluator", "from_model_ref"]


class ZooEvaluator(Evaluator):
    """A ``pp.zoo`` model as an evaluator (surrogate fidelity, not differentiable through this generic path)."""
    fidelity = "surrogate"

    def __init__(self, name: str, space: DesignSpace, fixed: Mapping[str, float] | None = None,
                 derived: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None):
        import pinneapple as pp
        self.model = pp.zoo.load(name)
        self.name = f"zoo:{name}"
        self.space, self.fixed = space, dict(fixed or {})
        missing = [k for k in self.model.input_names if k not in space.names and k not in self.fixed]
        if missing:
            raise ValueError(f"zoo model {name!r} needs inputs {missing}: add them as variables or fixed values")
        self.derived = derived
        self.outputs = list(self.model.output_names) if hasattr(self.model, "output_names") else []
        self.description = str(getattr(self.model.card, "description", ""))

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for k in self.model.input_names:
            cols.append(X[:, self.space.names.index(k)] if k in self.space.names else np.full(len(X), self.fixed[k]))
        return np.stack(cols, 1)

    def evaluate(self, X):
        A = self._inputs(X)
        out = {k: np.asarray(v, dtype=float) for k, v in
               self.model.predict(**{k: A[:, i] for i, k in enumerate(self.model.input_names)}).items()}
        out[VALID] = np.asarray(self.model.in_range(A), dtype=bool)
        if self.derived is not None:
            cols = {**{k: X[:, i] for i, k in enumerate(self.space.names)}, **out}
            out.update(self.derived(cols))
        return out


class TorchModuleEvaluator(Evaluator):
    """A torch module as a differentiable evaluator.

    ``input_transform(X_tensor) -> tensor`` feeds the module, ``output_transform(y_tensor, X_tensor) ->
    {name: tensor}`` names its outputs (default: columns of ``y`` named by ``outputs``). Evaluation runs in
    ``float64`` unless the module's parameters are ``float32``, in which case inputs are cast and outputs promoted.
    """
    fidelity = "surrogate"
    relaxable = True

    def __init__(self, module, outputs: Sequence[str], *, input_transform=None, output_transform=None,
                 name: str = "torch-module", fidelity: str = "surrogate", description: str = ""):
        self.module = module.eval() if hasattr(module, "eval") else module
        self.outputs, self.name, self.fidelity, self.description = list(outputs), name, fidelity, description
        self.input_transform, self.output_transform = input_transform, output_transform
        self.differentiable = True

    def torch(self, X):
        import torch
        X = torch.as_tensor(X, dtype=torch.float64)
        z = self.input_transform(X) if self.input_transform is not None else X
        p = next(iter(self.module.parameters()), None) if hasattr(self.module, "parameters") else None
        if p is not None and p.dtype == torch.float32:
            z = z.float()
        y = self.module(z)
        y = (y.y if hasattr(y, "y") else y).double()
        if self.output_transform is not None:
            return dict(self.output_transform(y, X))
        return {k: y[:, j] for j, k in enumerate(self.outputs)}

    def evaluate(self, X):
        import torch
        with torch.no_grad():
            return {k: v.detach().numpy() for k, v in self.torch(X).items()}


class CommandEvaluator(Evaluator):
    """An external solver run once per batch: ``command`` reads a CSV of designs, writes a CSV of outputs.

    ``command`` is a list (or a string split with :func:`shlex.split`) where ``{inputs}`` and ``{outputs}`` are
    replaced by the two file paths; it is run without a shell, in a fresh temporary directory, with ``timeout``
    seconds. Output columns missing from the file or rows that fail to parse become NaN (non-finite outputs make a
    design infeasible). The default fidelity is ``"reference"``: an external solver is usually the ground truth.
    """
    fidelity = "reference"

    def __init__(self, command: str | Sequence[str], space: DesignSpace, outputs: Sequence[str], *,
                 timeout: float = 600.0, name: str = "external solver", fidelity: str = "reference",
                 env: Mapping[str, str] | None = None):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.command:
            raise ValueError("empty command")
        self.space, self.outputs, self.timeout = space, list(outputs), float(timeout)
        self.name, self.fidelity, self.env = name, fidelity, dict(env or {})
        self.description = "external program: " + " ".join(self.command)

    def evaluate(self, X):
        with tempfile.TemporaryDirectory(prefix="pp-eval-") as d:
            fin, fout = os.path.join(d, "inputs.csv"), os.path.join(d, "outputs.csv")
            with open(fin, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self.space.names)
                w.writerows(X.tolist())
            cmd = [c.replace("{inputs}", fin).replace("{outputs}", fout) for c in self.command]
            subprocess.run(cmd, cwd=d, timeout=self.timeout, check=True, capture_output=True,
                           env={**os.environ, **self.env})
            out = {k: np.full(len(X), np.nan) for k in self.outputs}
            if os.path.exists(fout):
                with open(fout, newline="") as f:
                    for i, row in enumerate(csv.DictReader(f)):
                        if i >= len(X):
                            break
                        for k in self.outputs:
                            try:
                                out[k][i] = float(row.get(k, "nan"))
                            except (TypeError, ValueError):
                                pass
            return out


def from_model_ref(ref: Any, space: DesignSpace, *, fixed: Mapping[str, float] | None = None,
                   outputs: Sequence[str] | None = None) -> Evaluator:
    """An evaluator from a ``ModelRef``-like object or dict with a ``uri`` (or ``path``):

    * ``zoo:<name>``: :class:`ZooEvaluator`,
    * a TorchScript file (``.pt``/``.ts``/``.torchscript``): :class:`TorchModuleEvaluator` on the design vector in
      the order of ``space`` (``outputs`` names the columns; default ``ref.outputs``),
    * ``cmd:<command>``: :class:`CommandEvaluator`.
    """
    d = ref.to_dict() if hasattr(ref, "to_dict") else (dict(ref) if isinstance(ref, Mapping) else vars(ref))
    uri = str(d.get("uri") or d.get("path") or "")
    outs = list(outputs or d.get("outputs") or [])
    if uri.startswith("zoo:"):
        return ZooEvaluator(uri[4:], space, fixed=fixed)
    if uri.startswith("cmd:"):
        if not outs:
            raise ValueError("a command model reference needs its output names")
        return CommandEvaluator(uri[4:], space, outs, name=str(d.get("name") or "external solver"))
    if uri.endswith((".pt", ".ts", ".torchscript")):
        import torch
        if not outs:
            raise ValueError("a TorchScript model reference needs its output names")
        mod = torch.jit.load(uri, map_location="cpu")
        return TorchModuleEvaluator(mod, outs, name=str(d.get("name") or os.path.basename(uri)))
    raise ValueError(f"unsupported model reference {uri!r}: use zoo:<name>, cmd:<command> or a TorchScript file")
