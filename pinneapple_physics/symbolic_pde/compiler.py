"""Symbolic PDE compiler for PINNeAPPle.

Converts SymPy PDE residual expressions into callable PyTorch autograd functions.

Design note — two SymPy→PyTorch compilation paths exist in this codebase:

* **This module** (``SymbolicPDE``): walks the expression tree recursively using
  ``torch.autograd.grad``.  Gradients flow *through* the evaluation, so you can
  differentiate the residual loss with respect to model weights.  Use this when
  you need higher-order autograd or want the symbolic expression to participate
  directly in the training graph.

* ``pinneapple_pinn.factory.sympy_backend.SympyTorchCompiler``: uses
  ``sympy.lambdify`` to emit a Python lambda; faster to evaluate but the
  lambdified call is opaque to autograd (no gradient tape through the symbolic
  tree).  Use that path when the PDE loss is evaluated at fixed collocation
  points and speed matters more than differentiability through the expression.

``SymbolicPDE`` also supports a pluggable ``grad_method`` (default
``"autograd"``, unchanged from the original behavior above): pass
``grad_method="finite_difference"`` or ``"spectral"`` plus a ``grid``
(``GridAxisSpec``) to evaluate the SAME symbolic residual over grid-shaped
field tensors (e.g. an FNO's output) instead of a continuous model's output
at scattered points — see ``gradient_backends.py`` and
``to_grid_residual_fn``. Note: this module's ``compile_problem``-based
sibling, ``pinneapple_physics.pinn_solver.compiler.compile`` (the actual
engine behind the registered preset catalog in ``pde_environment/presets/``,
string-``kind``-dispatched rather than SymPy-expression-based), is a
different code path and is not affected by this parameter.

Usage example::

    import sympy as sp
    from pinneapple_physics.symbolic_pde import SymbolicPDE

    x, y = sp.symbols("x y")
    u = sp.Function("u")
    pi = sp.pi

    # Poisson: u_xx + u_yy + 2*pi^2*sin(pi*x)*sin(pi*y) = 0
    expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2) + 2 * pi**2 * sp.sin(pi * x) * sp.sin(pi * y)
    pde = SymbolicPDE(expr, coord_syms=[x, y], field_syms=[u])
    residual_fn = pde.to_residual_fn(model)
    # residual_fn(coords_tensor) -> (N,1) tensor
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp
from sympy.core.function import AppliedUndef as _SpAppliedUndef
import torch
import torch.nn as nn

from .gradient_backends import GradMethod, GridAxisSpec, grid_derivative


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _DerivativeOp:
    """Represents a single partial derivative ∂^n u / ∂x_i1 ... ∂x_in."""

    def __init__(self, field_name: str, wrt: List[str]) -> None:
        self.field_name = field_name  # e.g. "u"
        self.wrt = wrt               # e.g. ["x", "x"] for u_xx

    def __repr__(self) -> str:  # pragma: no cover
        return f"D({self.field_name}, {self.wrt})"


def _extract_derivative_ops(expr: sp.Expr, field_syms: List[sp.Function]) -> List[_DerivativeOp]:
    """Walk a SymPy expression tree and collect all Derivative sub-expressions."""
    ops: List[_DerivativeOp] = []
    field_names = {f.__name__ if hasattr(f, "__name__") else str(f): f for f in field_syms}

    for sub in sp.preorder_traversal(expr):
        if not isinstance(sub, sp.Derivative):
            continue
        expr_inner = sub.args[0]
        # expr_inner is e.g. u(x,y); get function name
        if isinstance(expr_inner, _SpAppliedUndef):
            fname = expr_inner.func.__name__
        elif hasattr(expr_inner, "name"):
            fname = expr_inner.name
        else:
            fname = str(expr_inner.func)

        # derivative variables (each entry is (sym, order))
        wrt: List[str] = []
        for sym, order in sub.variable_count:
            wrt.extend([str(sym)] * order)

        ops.append(_DerivativeOp(fname, wrt))

    return ops


def _compute_derivative(
    field_tensor: torch.Tensor,
    coords_tensor: torch.Tensor,
    wrt: List[str],
    coord_names: List[str],
) -> torch.Tensor:
    """Compute an arbitrary-order derivative using autograd.

    Parameters
    ----------
    field_tensor : (N, 1) tensor — the field value.
    coords_tensor : (N, D) tensor with grad enabled.
    wrt : list of coordinate names in differentiation order.
    coord_names : list of coordinate names (columns of coords_tensor).
    """
    current = field_tensor
    for coord_name in wrt:
        idx = coord_names.index(coord_name)
        g = torch.autograd.grad(
            outputs=current,
            inputs=coords_tensor,
            grad_outputs=torch.ones_like(current),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]  # (N, D)
        current = g[:, idx : idx + 1]
    return current


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SymbolicPDE:
    """Symbolic PDE residual — compile to a PyTorch autograd callable.

    Parameters
    ----------
    expr : sympy expression equal to 0 (the PDE residual).
    coord_syms : list of SymPy symbols for independent variables (e.g. [x, y, t]).
    field_syms : list of SymPy *Function* objects for dependent variables (e.g. [u, v]).
    param_syms : optional list of SymPy symbols for (possibly learnable) parameters,
        e.g. an unknown diffusivity in an inverse PDE problem. Their numeric values
        are not fixed at construction time — supply them via ``to_residual_fn(model,
        params={...})`` (or per-call via ``residual_fn(coords, params={...})``) so
        that a ``torch.nn.Parameter`` can be optimized jointly with the model.
    grad_method : how derivatives are computed from the underlying data — see
        ``gradient_backends.py``. One of:

        * ``"autograd"`` (default) — ``torch.autograd.grad`` through a
          continuous model's output at scattered collocation points. This is
          ``SymbolicPDE``'s original (and only, before this parameter existed)
          behavior; the default preserves it unchanged. Use ``to_residual_fn``.
        * ``"finite_difference"`` / ``"spectral"`` — grid-based derivatives
          (e.g. an FNO's structured-grid output), reusing
          ``pinneapple_neural``'s PINO derivative backends. Requires ``grid``.
          Use ``to_grid_residual_fn`` instead of ``to_residual_fn``.

        The SAME symbolic PDE (``expr``) can be reused unmodified across all
        three — only the derivative *computation* changes with the data's shape.
    grid : required when ``grad_method`` is ``"finite_difference"`` or
        ``"spectral"`` — a ``GridAxisSpec`` mapping ``coord_syms`` to the grid
        tensor's axes/extents. Ignored for ``"autograd"``.
    """

    def __init__(
        self,
        expr: sp.Expr,
        coord_syms: List[sp.Symbol],
        field_syms: List[sp.Function],
        param_syms: Optional[List[sp.Symbol]] = None,
        grad_method: GradMethod = "autograd",
        grid: Optional[GridAxisSpec] = None,
    ) -> None:
        # .doit() reduces any unevaluated Derivative (e.g. Derivative(u(x)**2, x,
        # evaluate=False)) via the chain rule into derivatives of bare field
        # applications (e.g. 2*u(x)*Derivative(u(x), x)); it is a no-op for
        # expressions already built via plain chained .diff() calls.
        self.expr = expr.doit()
        self.coord_syms = coord_syms
        self.field_syms = field_syms
        self.param_syms = param_syms or []
        self._compiled_fn: Optional[Callable] = None
        self._coord_names: List[str] = [str(s) for s in coord_syms]
        self._field_names: List[str] = [
            f.__name__ if hasattr(f, "__name__") else str(f) for f in field_syms
        ]
        self._param_names: List[str] = [str(s) for s in self.param_syms]
        self._deriv_ops: List[_DerivativeOp] = _extract_derivative_ops(self.expr, field_syms)

        if grad_method not in ("autograd", "finite_difference", "spectral"):
            raise ValueError(
                f"Unknown grad_method {grad_method!r}; expected 'autograd', "
                f"'finite_difference', or 'spectral'."
            )
        if grad_method != "autograd" and grid is None:
            raise ValueError(
                f"grad_method={grad_method!r} requires a `grid` (GridAxisSpec) "
                f"mapping coordinate names to grid axes/extents — finite-"
                f"difference and spectral derivatives need to know the data's "
                f"grid layout and spacing. Use grad_method='autograd' (the "
                f"default) for pointwise/collocation data instead."
            )
        self.grad_method: GradMethod = grad_method
        self.grid: Optional[GridAxisSpec] = grid

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_residual_fn(
        self,
        model: nn.Module,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Callable[..., torch.Tensor]:
        """Return a callable that computes the PDE residual using autograd.

        Parameters
        ----------
        model : nn.Module — takes (N, D) coords tensor, returns (N, F) fields tensor.
        params : optional dict mapping parameter-symbol name -> scalar tensor,
            used as the default values for any ``param_syms`` appearing in the
            expression (e.g. ``{"nu": nu_param}`` for a learnable diffusivity).
            Pass a ``torch.nn.Parameter`` here to jointly optimize it with the
            model; the same tensor object is read on every call, so updates
            from an optimizer step are picked up automatically.

        Returns
        -------
        residual_fn(coords_tensor, params=None) -> (N, 1) residual tensor.
            ``params`` passed to the returned callable overrides the defaults
            given here for that call only.
        """
        if self.grad_method != "autograd":
            raise ValueError(
                f"to_residual_fn is for grad_method='autograd' (differentiating "
                f"through `model` at scattered points); this SymbolicPDE was "
                f"built with grad_method={self.grad_method!r} — use "
                f"to_grid_residual_fn instead."
            )
        coord_names = self._coord_names
        field_names = self._field_names
        param_names = self._param_names
        expr = self.expr

        def residual_fn(
            coords: torch.Tensor,
            params: Optional[Dict[str, torch.Tensor]] = params,
        ) -> torch.Tensor:
            if param_names:
                missing = [p for p in param_names if not (params and p in params)]
                if missing:
                    raise ValueError(
                        f"Missing values for PDE parameter(s) {missing}; pass them via "
                        f"to_residual_fn(model, params={{...}}) or residual_fn(coords, params={{...}})."
                    )

            coords = coords.clone().requires_grad_(True)
            raw = model(coords)
            if raw.ndim == 1:
                raw = raw[:, None]

            # Build dict: field_name -> (N,1) tensor
            if raw.shape[1] != len(field_names):
                raise ValueError(
                    f"Model output dim {raw.shape[1]} != number of fields {len(field_names)}"
                )
            fields: Dict[str, torch.Tensor] = {
                fname: raw[:, i : i + 1] for i, fname in enumerate(field_names)
            }

            # Build dict: derivative key -> (N,1) tensor
            # Key: (field_name, tuple(wrt))
            deriv_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor] = {}
            for dop in self._deriv_ops:
                key = (dop.field_name, tuple(dop.wrt))
                if key in deriv_cache:
                    continue
                deriv_cache[key] = _compute_derivative(
                    fields[dop.field_name], coords, dop.wrt, coord_names
                )

            # Evaluate the symbolic expression numerically
            return _eval_expr_torch(expr, coords, fields, deriv_cache, coord_names, params)

        return residual_fn

    def to_grid_residual_fn(
        self,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Callable[..., torch.Tensor]:
        """Return a callable that computes the PDE residual on grid-shaped
        field tensors, using this instance's grid-based ``grad_method``
        (``"finite_difference"`` or ``"spectral"``).

        Unlike ``to_residual_fn`` (which differentiates through a `model`
        callable at scattered collocation points via autograd), this expects
        the field values *already evaluated* on a structured grid — the
        natural output shape of a neural operator such as an FNO — and
        computes derivatives via ``gradient_backends.grid_derivative``
        (which itself reuses ``pinneapple_neural``'s PINO derivative
        primitives).

        Parameters
        ----------
        params : optional dict mapping parameter-symbol name -> scalar tensor
            (same contract as ``to_residual_fn``).

        Returns
        -------
        residual_fn(fields, params=None) -> residual tensor, same grid shape
            as the input fields.
            ``fields`` maps field name -> grid-shaped tensor; every field
            must share the layout described by ``self.grid``.
        """
        if self.grad_method == "autograd":
            raise ValueError(
                "to_grid_residual_fn is for grad_method in "
                "('finite_difference', 'spectral'); this SymbolicPDE was "
                "built with grad_method='autograd' — use to_residual_fn(model) "
                "instead."
            )
        coord_names = self._coord_names
        field_names = self._field_names
        param_names = self._param_names
        expr = self.expr
        grid = self.grid
        method = self.grad_method
        assert grid is not None  # enforced in __init__

        def residual_fn(
            fields: Dict[str, torch.Tensor],
            params: Optional[Dict[str, torch.Tensor]] = params,
        ) -> torch.Tensor:
            if param_names:
                missing = [p for p in param_names if not (params and p in params)]
                if missing:
                    raise ValueError(
                        f"Missing values for PDE parameter(s) {missing}; pass them via "
                        f"to_grid_residual_fn(params={{...}}) or residual_fn(fields, params={{...}})."
                    )
            missing_fields = [f for f in field_names if f not in fields]
            if missing_fields:
                raise ValueError(
                    f"Missing field(s) {missing_fields} in `fields`; expected {field_names}"
                )

            deriv_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor] = {}
            for dop in self._deriv_ops:
                key = (dop.field_name, tuple(dop.wrt))
                if key in deriv_cache:
                    continue
                deriv_cache[key] = grid_derivative(
                    fields[dop.field_name], dop.wrt, grid=grid, method=method
                )

            return _eval_expr_torch_grid(
                expr, fields, deriv_cache, coord_names, params, grid, method
            )

        return residual_fn

    def compile(self, backend: str = "torch") -> "SymbolicPDE":
        """Compile the symbolic expression (currently a no-op for 'torch' backend).

        Sets an internal flag; future backends (e.g. JAX, C++) can be added here.
        """
        if backend != "torch":
            raise ValueError(f"Unsupported backend '{backend}'. Only 'torch' is supported.")
        # Mark as compiled; the heavy work happens lazily in to_residual_fn.
        self._compiled_fn = True  # type: ignore[assignment]
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SymbolicPDE(coords={self._coord_names}, fields={self._field_names}, "
            f"expr={self.expr})"
        )


# ---------------------------------------------------------------------------
# Expression evaluator: SymPy -> torch, walking the expression tree
# ---------------------------------------------------------------------------

def _eval_expr_torch(
    expr: sp.Expr,
    coords: torch.Tensor,
    fields: Dict[str, torch.Tensor],
    deriv_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor],
    coord_names: List[str],
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Recursively evaluate a SymPy expression as a PyTorch tensor.

    Supports: Add, Mul, Pow, Number, Symbol (coords or parameters),
    AppliedUndef (fields), Derivative, sin, cos, exp, log, Abs, sign, and
    basic constants.
    """
    N = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    def _const(v: float) -> torch.Tensor:
        return torch.full((N, 1), v, device=device, dtype=dtype)

    def _eval(e: sp.Expr) -> torch.Tensor:
        # ---- constants ----
        if isinstance(e, sp.Number):
            return _const(float(e))

        if e is sp.pi:
            return _const(float(sp.pi))

        if e is sp.E:
            return _const(float(sp.E))

        # ---- coordinate or parameter symbol ----
        if isinstance(e, sp.Symbol):
            s = str(e)
            if s in coord_names:
                idx = coord_names.index(s)
                return coords[:, idx : idx + 1]
            if params is not None and s in params:
                val = params[s]
                if not torch.is_tensor(val):
                    val = torch.as_tensor(val, device=device, dtype=dtype)
                val = val.to(device=device, dtype=dtype).reshape(1, 1)
                return val.expand(N, 1)
            raise ValueError(
                f"Unknown symbol '{s}' — not in coords {coord_names} or "
                f"params {list((params or {}).keys())}"
            )

        # ---- applied function (field value) ----
        if isinstance(e, _SpAppliedUndef):
            fname = e.func.__name__
            if fname not in fields:
                raise ValueError(f"Field '{fname}' not in model outputs {list(fields.keys())}")
            return fields[fname]

        # ---- derivative ----
        if isinstance(e, sp.Derivative):
            inner = e.args[0]
            if isinstance(inner, _SpAppliedUndef):
                fname = inner.func.__name__
            else:
                fname = str(inner.func)
            wrt: List[str] = []
            for sym, order in e.variable_count:
                wrt.extend([str(sym)] * order)
            key = (fname, tuple(wrt))
            if key in deriv_cache:
                return deriv_cache[key]
            # Compute on-the-fly if not pre-cached
            result = _compute_derivative(fields[fname], coords, wrt, coord_names)
            deriv_cache[key] = result
            return result

        # ---- compound expressions ----
        if isinstance(e, sp.Add):
            result = _eval(e.args[0])
            for arg in e.args[1:]:
                result = result + _eval(arg)
            return result

        if isinstance(e, sp.Mul):
            result = _eval(e.args[0])
            for arg in e.args[1:]:
                result = result * _eval(arg)
            return result

        if isinstance(e, sp.Pow):
            base, exp_ = e.args
            b = _eval(base)
            # Constant integer exponents — use torch.pow for graph efficiency
            if isinstance(exp_, sp.Integer):
                n = int(exp_)
                if n == 2:
                    return b * b
                if n == 3:
                    return b * b * b
                return torch.pow(b, n)
            if isinstance(exp_, sp.Number):
                return torch.pow(b, float(exp_))
            return torch.pow(b, _eval(exp_))

        # ---- elementary functions ----
        if isinstance(e, sp.sin):
            return torch.sin(_eval(e.args[0]))

        if isinstance(e, sp.cos):
            return torch.cos(_eval(e.args[0]))

        if isinstance(e, sp.tan):
            return torch.tan(_eval(e.args[0]))

        if isinstance(e, sp.exp):
            return torch.exp(_eval(e.args[0]))

        if isinstance(e, sp.log):
            return torch.log(_eval(e.args[0]))

        if isinstance(e, sp.Abs):
            return torch.abs(_eval(e.args[0]))

        if isinstance(e, sp.sign):
            return torch.sign(_eval(e.args[0]))

        # Note: sp.sqrt is a factory function, not a class (SymPy normalizes
        # sqrt(x) to Pow(x, Rational(1, 2))), so it can't appear here as an
        # isinstance check — it's already handled by the sp.Pow branch above.

        if isinstance(e, sp.tanh):
            return torch.tanh(_eval(e.args[0]))

        if isinstance(e, sp.sinh):
            return torch.sinh(_eval(e.args[0]))

        if isinstance(e, sp.cosh):
            return torch.cosh(_eval(e.args[0]))

        raise NotImplementedError(
            f"Cannot evaluate SymPy expression of type {type(e).__name__}: {e}"
        )

    return _eval(expr)


# ---------------------------------------------------------------------------
# Grid-mode expression evaluator: SymPy -> torch over grid-shaped tensors
# ---------------------------------------------------------------------------
#
# Deliberately a SEPARATE function from _eval_expr_torch above (same tree
# dispatch — Add/Mul/Pow/trig — reimplemented rather than parameterized into
# one shared function) so that the original autograd code path above is not
# touched by adding grid-based grad_method support: every existing caller of
# SymbolicPDE (grad_method="autograd", the default) keeps running through
# _eval_expr_torch completely unmodified.

def _eval_expr_torch_grid(
    expr: sp.Expr,
    fields: Dict[str, torch.Tensor],
    deriv_cache: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor],
    coord_names: List[str],
    params: Optional[Dict[str, torch.Tensor]],
    grid: GridAxisSpec,
    method: str,
) -> torch.Tensor:
    """Recursively evaluate a SymPy expression over grid-shaped tensors.

    Same supported node types as ``_eval_expr_torch`` (Add, Mul, Pow, Number,
    Symbol, AppliedUndef, Derivative, sin, cos, tan, exp, log, Abs, sign,
    tanh, sinh, cosh) — see that function's docstring — but field/derivative
    values are grid-shaped tensors keyed by name (rather than columns of a
    stacked (N, D) collocation tensor), constants are broadcastable 0-dim
    tensors (rather than an explicit (N, 1) fill), and any derivative not
    already in ``deriv_cache`` is computed on the fly via
    ``gradient_backends.grid_derivative`` (finite-difference / spectral)
    instead of ``torch.autograd.grad``.
    """
    ref = next(iter(fields.values()))
    device, dtype = ref.device, ref.dtype

    def _const(v: float) -> torch.Tensor:
        # 0-dim tensors broadcast against any grid shape.
        return torch.tensor(float(v), device=device, dtype=dtype)

    def _eval(e: sp.Expr) -> torch.Tensor:
        # ---- constants ----
        if isinstance(e, sp.Number):
            return _const(float(e))

        if e is sp.pi:
            return _const(float(sp.pi))

        if e is sp.E:
            return _const(float(sp.E))

        # ---- coordinate or parameter symbol ----
        if isinstance(e, sp.Symbol):
            s = str(e)
            if s in coord_names:
                raise NotImplementedError(
                    f"Bare coordinate symbol '{s}' inside a grid-mode PDE "
                    f"expression is not supported yet — only field values and "
                    f"their derivatives may appear directly in expressions "
                    f"evaluated via to_grid_residual_fn (none of the presets "
                    f"this backend has been validated against need it). Add "
                    f"explicit coordinate grid tensors if a preset needs a "
                    f"source term written in terms of a bare coordinate."
                )
            if params is not None and s in params:
                val = params[s]
                if not torch.is_tensor(val):
                    val = torch.as_tensor(val, device=device, dtype=dtype)
                return val.to(device=device, dtype=dtype).reshape(())
            raise ValueError(
                f"Unknown symbol '{s}' — not a field, coordinate, or in "
                f"params {list((params or {}).keys())}"
            )

        # ---- applied function (field value) ----
        if isinstance(e, _SpAppliedUndef):
            fname = e.func.__name__
            if fname not in fields:
                raise ValueError(f"Field '{fname}' not in provided fields {list(fields.keys())}")
            return fields[fname]

        # ---- derivative ----
        if isinstance(e, sp.Derivative):
            inner = e.args[0]
            if isinstance(inner, _SpAppliedUndef):
                fname = inner.func.__name__
            else:
                fname = str(inner.func)
            wrt: List[str] = []
            for sym, order in e.variable_count:
                wrt.extend([str(sym)] * order)
            key = (fname, tuple(wrt))
            if key in deriv_cache:
                return deriv_cache[key]
            result = grid_derivative(fields[fname], wrt, grid=grid, method=method)
            deriv_cache[key] = result
            return result

        # ---- compound expressions ----
        if isinstance(e, sp.Add):
            result = _eval(e.args[0])
            for arg in e.args[1:]:
                result = result + _eval(arg)
            return result

        if isinstance(e, sp.Mul):
            result = _eval(e.args[0])
            for arg in e.args[1:]:
                result = result * _eval(arg)
            return result

        if isinstance(e, sp.Pow):
            base, exp_ = e.args
            b = _eval(base)
            if isinstance(exp_, sp.Integer):
                n = int(exp_)
                if n == 2:
                    return b * b
                if n == 3:
                    return b * b * b
                return torch.pow(b, n)
            if isinstance(exp_, sp.Number):
                return torch.pow(b, float(exp_))
            return torch.pow(b, _eval(exp_))

        # ---- elementary functions ----
        if isinstance(e, sp.sin):
            return torch.sin(_eval(e.args[0]))

        if isinstance(e, sp.cos):
            return torch.cos(_eval(e.args[0]))

        if isinstance(e, sp.tan):
            return torch.tan(_eval(e.args[0]))

        if isinstance(e, sp.exp):
            return torch.exp(_eval(e.args[0]))

        if isinstance(e, sp.log):
            return torch.log(_eval(e.args[0]))

        if isinstance(e, sp.Abs):
            return torch.abs(_eval(e.args[0]))

        if isinstance(e, sp.sign):
            return torch.sign(_eval(e.args[0]))

        if isinstance(e, sp.tanh):
            return torch.tanh(_eval(e.args[0]))

        if isinstance(e, sp.sinh):
            return torch.sinh(_eval(e.args[0]))

        if isinstance(e, sp.cosh):
            return torch.cosh(_eval(e.args[0]))

        raise NotImplementedError(
            f"Cannot evaluate SymPy expression of type {type(e).__name__}: {e}"
        )

    return _eval(expr)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def pde_from_sympy(
    expr: sp.Expr,
    coords: List[sp.Symbol],
    fields: List[sp.Function],
    params: Optional[List[sp.Symbol]] = None,
    grad_method: GradMethod = "autograd",
    grid: Optional[GridAxisSpec] = None,
) -> SymbolicPDE:
    """Create a SymbolicPDE from a SymPy residual expression.

    Parameters
    ----------
    expr : sympy expression equal to 0 (the PDE residual).
    coords : list of SymPy symbols for independent variables.
    fields : list of SymPy Function objects for dependent variables.
    params : optional list of parameter symbols.
    grad_method, grid : forwarded to ``SymbolicPDE`` — see its docstring.

    Returns
    -------
    SymbolicPDE instance.

    Example::

        x, y = sp.symbols("x y")
        u = sp.Function("u")
        expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2)  # Laplace
        pde = pde_from_sympy(expr, [x, y], [u])
    """
    return SymbolicPDE(expr, coords, fields, params, grad_method=grad_method, grid=grid)


def auto_residual(
    model: nn.Module,
    coords_tensor: torch.Tensor,
    derivative_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Compute field values and named derivative quantities via autograd.

    This is a lower-level helper intended for users who prefer to wire
    derivative operators manually rather than through SymPy expressions.

    Parameters
    ----------
    model : nn.Module taking (N, D) -> (N, F) tensors.
    coords_tensor : (N, D) input coordinates tensor (grad will be enabled).
    derivative_fns : dict mapping name -> callable(field, coords) -> tensor.
        Each callable receives the raw model output and the coords tensor
        (with grad enabled) and should return a (N, *) tensor.

    Returns
    -------
    dict with keys:
        "fields" : (N, F) raw model output
        + one key per entry in derivative_fns.

    Example::

        from pinneapple_physics.pinn_solver.compiler.autograd_ops import laplacian

        result = auto_residual(
            model, x_col,
            {"laplacian_u": lambda u, x: laplacian(u, x)}
        )
        res = result["laplacian_u"] - f_source
    """
    coords = coords_tensor.clone().requires_grad_(True)
    raw = model(coords)
    if raw.ndim == 1:
        raw = raw[:, None]

    out: Dict[str, torch.Tensor] = {"fields": raw}
    for name, fn in derivative_fns.items():
        out[name] = fn(raw, coords)
    return out
