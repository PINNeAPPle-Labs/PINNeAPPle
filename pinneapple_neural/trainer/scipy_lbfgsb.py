"""SciPy's L-BFGS-B on the parameters of a PyTorch model.

This is the optimizer of the original PINN code (Raissi et al., TensorFlow ``ScipyOptimizerInterface`` with
``method="L-BFGS-B"``): full batch, a bounded number of iterations and function evaluations, ``maxcor`` correction
pairs, a More-Thuente-type line search of up to ``maxls`` steps, and the relative-reduction stop ``ftol``. It
differs from :class:`torch.optim.LBFGS` (no ``ftol`` stop on the relative function decrease, a different line
search), so a reproduction of that code should use this one.

The parameters are flattened to one float64 vector for SciPy and copied back into the model (in its own dtype)
before each evaluation; the gradient is returned in float64.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["LBFGSBResult", "minimize_lbfgsb"]


@dataclass
class LBFGSBResult:
    loss: float
    iterations: int
    evaluations: int
    message: str
    seconds: float
    history: list[float] = field(default_factory=list)   # loss at every evaluation


def minimize_lbfgsb(params: Sequence[torch.nn.Parameter], loss_fn: Callable[[], torch.Tensor], *,
                    maxiter: int = 50000, maxfun: int = 50000, maxcor: int = 50, maxls: int = 50,
                    ftol: float = float(np.finfo(float).eps), gtol: float = 1e-5,
                    callback: Callable[[int, float], None] | None = None) -> LBFGSBResult:
    """Minimise ``loss_fn()`` (a scalar tensor built from ``params``) with ``scipy.optimize.minimize`` L-BFGS-B.

    Defaults are those of the original PINN scripts (maxiter = maxfun = 50 000, maxcor = maxls = 50, ftol = machine
    epsilon) and SciPy's ``gtol``. ``callback(iteration, loss)`` runs after every iteration. The model is left at
    the final parameters.
    """
    from scipy.optimize import minimize

    params = [p for p in params if p.requires_grad]
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    history: list[float] = []

    def assign(x: np.ndarray) -> None:
        with torch.no_grad():
            offset = 0
            for p, shape, n in zip(params, shapes, sizes, strict=True):
                p.copy_(torch.from_numpy(x[offset:offset + n]).view(shape).to(p.dtype))
                offset += n

    def fun(x: np.ndarray):
        assign(x)
        for p in params:
            p.grad = None
        loss = loss_fn()
        loss.backward()
        g = np.concatenate([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                            .to(torch.float64).cpu().numpy() for p in params])
        value = float(loss.detach())
        history.append(value)
        return value, g

    x0 = np.concatenate([p.detach().reshape(-1).to(torch.float64).cpu().numpy() for p in params])
    n_iter = [0]

    def cb(xk) -> None:
        n_iter[0] += 1
        if callback is not None:
            callback(n_iter[0], history[-1])

    t0 = time.perf_counter()
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": maxiter, "maxfun": maxfun, "maxcor": maxcor, "maxls": maxls, "ftol": ftol,
                            "gtol": gtol})
    assign(res.x)
    return LBFGSBResult(float(res.fun), int(res.nit), int(res.nfev), str(res.message), time.perf_counter() - t0,
                        history)
