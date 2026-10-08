"""PhysicsModule: composable models and solvers, the ``nn.Module`` of physics.

Everything is a torch module with a ``kind``, so a neural network, a numerical solver and a mix of the two
compose with the same few classes::

    import torch.nn as nn
    from pinneapple_core.module import Hybrid, SolverModule

    coarse = SolverModule(coarse_fem_solve)                       # any differentiable function, or a pp.solve method
    model = Hybrid(coarse, nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1)), mode="residual")
    model.fit(x, y, epochs=200)                                  # trains the corrector through the solver

* :class:`PhysicsModule`  base class: ``describe()``, ``num_parameters()``, a small ``fit``.
* :class:`Sequential`     chain components, each feeding the next.
* :class:`SolverModule`   a solver as a component: ``fn`` (autograd flows through it), optional trainable
  parameters, or :meth:`SolverModule.from_method` for any ``pp.solve`` backend (not differentiable).
* :class:`Hybrid`         solver output combined with a neural correction.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

__all__ = ["PhysicsModule", "Sequential", "SolverModule", "Hybrid", "Lambda"]


def _kind_of(m: nn.Module) -> str:
    return getattr(m, "kind", "neural" if any(True for _ in m.parameters()) else "function")


class PhysicsModule(nn.Module):
    """Base class for models and solvers. Subclass it and implement ``forward``."""

    kind: str = "neural"

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def components(self) -> "OrderedDict[str, nn.Module]":
        """Direct sub-modules by name."""
        return OrderedDict(self.named_children())

    def describe(self, _indent: int = 0) -> str:
        """Indented tree of the components with their kind and parameter counts."""
        pad = "  " * _indent
        head = f"{pad}{type(self).__name__} [{_kind_of(self)}] params={self.num_parameters()}"
        lines = [head]
        for name, child in self.named_children():
            if isinstance(child, nn.ParameterDict):
                if len(child):
                    lines.append(f"{pad}  {name}: ParameterDict params={sum(p.numel() for p in child.values() if p.requires_grad)}")
            elif isinstance(child, PhysicsModule):
                lines.append(f"{pad}  {name}:")
                lines.append(child.describe(_indent + 2))
            else:
                n = sum(p.numel() for p in child.parameters() if p.requires_grad)
                lines.append(f"{pad}  {name}: {type(child).__name__} [{_kind_of(child)}] params={n}")
        return "\n".join(lines)

    def fit(self, x, y, *, epochs: int = 200, lr: float = 1e-2, loss_fn: Optional[Callable] = None,
            batch_size: Optional[int] = None, optimizer: Optional[torch.optim.Optimizer] = None,
            seed: int = 0, verbose: bool = False) -> List[float]:
        """Minimal training loop on ``(x, y)`` (tensors, or a tuple of tensors for ``x``). Returns the loss per epoch.
        Gradients reach every trainable parameter, including those of solver components."""
        loss_fn = loss_fn or nn.functional.mse_loss
        xs = tuple(x) if isinstance(x, (tuple, list)) else (x,)
        n = xs[0].shape[0]
        opt = optimizer or torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)
        gen = torch.Generator().manual_seed(seed)
        history: List[float] = []
        for epoch in range(epochs):
            order = torch.randperm(n, generator=gen)
            batches = [order] if batch_size is None else list(order.split(batch_size))
            total = 0.0
            for idx in batches:
                opt.zero_grad()
                loss = loss_fn(self(*[a[idx] for a in xs]), y[idx])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            history.append(total / n)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"epoch {epoch:5d}  loss {history[-1]:.3e}")
        return history


class Lambda(PhysicsModule):
    """Wrap a plain function (no parameters) as a component."""

    kind = "function"

    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _as_module(m: Union[nn.Module, Callable]) -> nn.Module:
    return m if isinstance(m, nn.Module) else Lambda(m)


class Sequential(PhysicsModule):
    """Chain components: the output of each is the input of the next. Accepts modules, solvers and plain
    functions; pass ``(name, module)`` pairs or an ``OrderedDict`` to name them."""

    def __init__(self, *parts: Union[nn.Module, Callable, Tuple[str, Any]]) -> None:
        super().__init__()
        if len(parts) == 1 and isinstance(parts[0], Mapping):
            parts = tuple(parts[0].items())
        if not parts:
            raise ValueError("Sequential needs at least one component")
        for i, part in enumerate(parts):
            name, mod = part if isinstance(part, tuple) else (str(i), part)
            self.add_module(name, _as_module(mod))

    def forward(self, *args, **kwargs):
        mods = list(self.children())
        out = mods[0](*args, **kwargs)
        for m in mods[1:]:
            out = m(out)
        return out

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, i: int) -> nn.Module:
        return list(self.children())[i]


class SolverModule(PhysicsModule):
    """A solver as a component.

    ``fn(*inputs, **params)`` is any torch function, for example a differentiable finite-element solve; autograd
    flows through it to its inputs and to the parameters. ``params`` are tensors passed to ``fn`` by keyword; the
    names in ``trainable`` become ``nn.Parameter`` (identification, calibration), the rest buffers.

    For solvers that are not differentiable (an external code, any ``pp.solve`` method) use
    :meth:`from_method`: the output is a constant for autograd, so only components after it train.
    """

    kind = "solver"

    def __init__(self, fn: Callable, params: Optional[Mapping[str, Any]] = None, trainable: Iterable[str] = (),
                 name: Optional[str] = None) -> None:
        super().__init__()
        self.fn = fn
        self.name_ = name or getattr(fn, "__name__", "solver")
        trainable = set(trainable)
        params = dict(params or {})
        unknown = trainable - set(params)
        if unknown:
            raise ValueError(f"trainable names {sorted(unknown)} are not in params {sorted(params)}")
        self.params = nn.ParameterDict()
        self._constants: List[str] = []
        for k, v in params.items():
            t = torch.as_tensor(v, dtype=torch.get_default_dtype()) if not isinstance(v, Tensor) else v
            if k in trainable:
                self.params[k] = nn.Parameter(t.detach().clone())
            else:
                self.register_buffer(f"const_{k}", t.detach().clone())
                self._constants.append(k)

    def _kwargs(self) -> Dict[str, Tensor]:
        kw: Dict[str, Tensor] = {k: v for k, v in self.params.items()}
        kw.update({k: getattr(self, f"const_{k}") for k in self._constants})
        return kw

    def forward(self, *args, **kwargs):
        return self.fn(*args, **self._kwargs(), **kwargs)

    @classmethod
    def from_method(cls, problem, method: str = "fem", *, dtype=None, **options) -> "SolverModule":
        """Component from a ``pp.solve`` method: solves ``problem`` once and evaluates the solution at the points
        it is called with (``X`` of shape ``(N, n_coords)``, tensor or array). Not differentiable."""
        import numpy as np
        from pinneapple_physics.solving import solve

        sol = solve(problem, method, **options)

        def evaluate(x):
            dt = dtype or (x.dtype if isinstance(x, Tensor) else torch.get_default_dtype())
            xn = x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)
            return torch.as_tensor(sol.predict(xn), dtype=dt)

        mod = cls(evaluate, name=f"{method}:{getattr(problem, 'name', 'problem')}")
        mod.solution = sol
        return mod


class Hybrid(PhysicsModule):
    """A solver plus a neural correction.

    ``u0 = solver(*inputs)``; the corrector sees ``corrector_input(inputs, u0)`` (default: ``u0``) and the output is

    * ``"residual"``       ``u0 + corrector(...)``        learn what the solver misses
    * ``"multiplicative"`` ``u0 * (1 + corrector(...))``  learn a relative error
    * ``"replace"``        ``corrector(...)``             the solver only supplies features

    ``detach_solver=True`` stops gradients into the solver (cheaper when it has nothing to learn).
    ``forward(..., return_parts=True)`` returns ``{"solver", "correction", "output"}`` for diagnostics.
    """

    MODES = ("residual", "multiplicative", "replace")

    def __init__(self, solver: Union[nn.Module, Callable], corrector: Union[nn.Module, Callable], mode: str = "residual",
                 corrector_input: Optional[Callable[[Tuple[Any, ...], Tensor], Tensor]] = None,
                 detach_solver: bool = False) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.solver = _as_module(solver)
        self.corrector = _as_module(corrector)
        self.mode = mode
        self.corrector_input = corrector_input or (lambda args, u0: u0)
        self.detach_solver = detach_solver

    def forward(self, *args, return_parts: bool = False):
        u0 = self.solver(*args)
        if self.detach_solver:
            u0 = u0.detach()
        c = self.corrector(self.corrector_input(args, u0))
        if self.mode == "residual":
            out = u0 + c
        elif self.mode == "multiplicative":
            out = u0 * (1.0 + c)
        else:
            out = c
        return {"solver": u0, "correction": c, "output": out} if return_parts else out
