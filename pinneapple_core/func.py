"""``pp.func``: function transforms with scientific semantics, and implicit differentiation.

Two things live here.

**Transforms** wrap :mod:`torch.func` (``grad, jacobian, jacrev, jacfwd, hessian, vmap``) and add
``wrt``: the argument to differentiate, by name or position. A model written as
``model(geometry, mu)`` is differentiated with respect to its geometry with
``jacobian(model, wrt="geometry")``; several arguments return a tuple.

**Implicit differentiation** (:func:`implicit_solve`) gives gradients through a solver call
without differentiating the solver's iterations. If ``x*(theta)`` solves
``residual(x, theta) = 0``, then ``dx*/dtheta = -(dF/dx)^-1 dF/dtheta``; the backward pass is one
linear (adjoint) solve, whatever the forward solver did, including an external code.

>>> import torch
>>> from pinneapple_core import func
>>> f = lambda x, mu: (mu * x ** 2).sum()
>>> func.grad(f, wrt="mu")(torch.tensor([1.0, 2.0]), torch.tensor(3.0)).item()
5.0
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch import func as _tf

Wrt = Union[int, str, Sequence[Union[int, str]]]

__all__ = ["grad", "jacobian", "jacrev", "jacfwd", "hessian", "vmap", "implicit_solve", "newton_solve"]


# -- argument selection ------------------------------------------------------
def _argnums(fn: Callable, wrt: Wrt):
    """Positional indices (int, or tuple of ints) of the arguments named by ``wrt``."""
    def one(w) -> int:
        if isinstance(w, bool) or not isinstance(w, (int, str)):
            raise TypeError(f"wrt must be an int, a parameter name or a sequence of them, got {w!r}")
        if isinstance(w, int):
            return w
        try:
            params = [p for p in inspect.signature(fn).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            raise ValueError(f"cannot read the signature of {fn!r}; use an integer position for wrt") from None
        names = [p.name for p in params]
        if w not in names:
            raise ValueError(f"wrt={w!r} is not a positional parameter of {getattr(fn, '__name__', fn)}; have {names}")
        return names.index(w)

    if isinstance(wrt, (list, tuple)):
        return tuple(one(w) for w in wrt)
    return one(wrt)


# -- transforms --------------------------------------------------------------
def grad(fn: Callable, wrt: Wrt = 0, *, has_aux: bool = False) -> Callable:
    """Gradient of a scalar-valued ``fn`` with respect to the argument(s) ``wrt``."""
    return _tf.grad(fn, argnums=_argnums(fn, wrt), has_aux=has_aux)


def jacrev(fn: Callable, wrt: Wrt = 0, *, has_aux: bool = False) -> Callable:
    """Jacobian by reverse mode: cheap when ``fn`` has few outputs."""
    return _tf.jacrev(fn, argnums=_argnums(fn, wrt), has_aux=has_aux)


def jacfwd(fn: Callable, wrt: Wrt = 0, *, has_aux: bool = False) -> Callable:
    """Jacobian by forward mode: cheap when ``fn`` has few inputs."""
    return _tf.jacfwd(fn, argnums=_argnums(fn, wrt), has_aux=has_aux)


def jacobian(fn: Callable, wrt: Wrt = 0, *, mode: str = "auto") -> Callable:
    """Jacobian of ``fn`` with respect to ``wrt``. ``mode`` is ``"rev"``, ``"fwd"`` or
    ``"auto"`` (forward when the argument has fewer entries than the output has)."""
    if mode not in ("auto", "rev", "fwd"):
        raise ValueError("mode must be 'auto', 'rev' or 'fwd'")
    argnums = _argnums(fn, wrt)
    if mode == "rev":
        return _tf.jacrev(fn, argnums=argnums)
    if mode == "fwd":
        return _tf.jacfwd(fn, argnums=argnums)

    def chosen(*args, **kwargs):
        nums = (argnums,) if isinstance(argnums, int) else argnums
        n_in = sum(args[i].numel() for i in nums)
        n_out = sum(t.numel() for t in _tree_tensors(fn(*args, **kwargs)))
        pick = _tf.jacfwd if n_in <= n_out else _tf.jacrev
        return pick(fn, argnums=argnums)(*args, **kwargs)

    return chosen


def hessian(fn: Callable, wrt: Wrt = 0) -> Callable:
    """Hessian of a scalar-valued ``fn`` with respect to ``wrt``."""
    return _tf.hessian(fn, argnums=_argnums(fn, wrt))


def vmap(fn: Callable, in_dims=0, out_dims=0, **kwargs) -> Callable:
    """Vectorise ``fn`` over a batch dimension (``torch.func.vmap``)."""
    return _tf.vmap(fn, in_dims=in_dims, out_dims=out_dims, **kwargs)


def _tree_tensors(out) -> list:
    if isinstance(out, Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for o in out for t in _tree_tensors(o)]
    if isinstance(out, dict):
        return [t for o in out.values() for t in _tree_tensors(o)]
    return []


# -- nonlinear solve ---------------------------------------------------------
def newton_solve(
    residual: Callable[..., Tensor],
    x0: Tensor,
    *params: Tensor,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Tensor:
    """Damped Newton iteration for ``residual(x, *params) = 0`` with a dense Jacobian.

    Not differentiable by design (it runs under ``no_grad``); use :func:`implicit_solve` to
    differentiate through it. Raises ``RuntimeError`` if it does not converge.
    """
    x = x0.detach().clone()
    p = [q.detach() for q in params]
    fn = lambda z: residual(z, *p)
    with torch.no_grad():
        for _ in range(max_iter):
            f = fn(x).reshape(-1)
            if f.norm() <= tol * max(1.0, x.norm().item()):
                return x
            jac = _tf.jacrev(lambda z: fn(z.reshape(x.shape)).reshape(-1))(x.reshape(-1))
            step = torch.linalg.solve(jac, f).reshape(x.shape)
            t, f0 = 1.0, f.norm()
            while t > 1e-4 and fn(x - t * step).reshape(-1).norm() > (1 - 1e-4 * t) * f0:
                t *= 0.5  # backtracking line search
            x = x - t * step
        if fn(x).reshape(-1).norm() <= tol * max(1.0, x.norm().item()):
            return x
    raise RuntimeError(f"newton_solve did not converge in {max_iter} iterations")


def implicit_solve(
    residual: Callable[..., Tensor],
    x0: Tensor,
    *params: Tensor,
    solver: Optional[Callable[..., Tensor]] = None,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Tensor:
    """Solve ``residual(x, *params) = 0`` for ``x`` and differentiate through the solution.

    Gradients with respect to ``params`` come from the implicit function theorem,
    ``dx*/dp = -(dF/dx)^-1 dF/dp``, so the cost of the backward pass is one adjoint linear
    solve and does not depend on how many iterations the forward solver took.

    Args:
        residual: ``residual(x, *params)``, same shape as ``x``, differentiable in ``x`` and ``params``.
        x0: initial guess (also fixes the shape of the solution).
        params: tensors the solution depends on.
        solver: optional forward solver ``solver(residual, x0, *params) -> x*``, for example a
            call to an external code. It is never differentiated. Default: damped Newton.

    Gradients are exact to first order (reverse or forward mode). The Jacobian ``dF/dx`` is dense
    and size ``(x.numel())**2``; second derivatives through the solve ignore its dependence on ``params``.
    """
    forward = solver or (lambda r, x, *p: newton_solve(r, x, *p, tol=tol, max_iter=max_iter))
    with torch.no_grad():
        x_star = forward(residual, x0.detach(), *[p.detach() for p in params]).detach().reshape(x0.shape)
    flat = x_star.reshape(-1)
    # Jacobian in x only, at the converged point, with no graph back to params.
    jac = _tf.jacrev(lambda z: residual(z.reshape(x0.shape), *[p.detach() for p in params]).reshape(-1))(flat)
    # x_star is constant for autograd, so this single step has value x* and the implicit gradient.
    f = residual(x_star, *params).reshape(-1)
    return (flat - torch.linalg.solve(jac, f)).reshape(x0.shape)
