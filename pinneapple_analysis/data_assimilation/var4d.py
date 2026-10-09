"""Strong-constraint 4D-Var through any differentiable PyTorch model: the adjoint comes from autograd.

Cost function for the initial state x0 of an assimilation window with observations y_k at steps k:

    J(x0) = 1/2 (x0 - xb)^T B^-1 (x0 - xb) + 1/2 sum_k (H_k(x_k) - y_k)^T R_k^-1 (H_k(x_k) - y_k),
    x_{k+1} = M(x_k)

(Le Dimet & Talagrand 1986, Tellus 38A:97; Courtier, Thepaut & Hollingsworth 1994, QJRMS 120:1367). The model ``M`` is
any torch function (a numerical solver written in torch, a neural surrogate, or a hybrid), so the gradient of J needs
no hand-written tangent-linear or adjoint code. The minimisation runs in the control variable v = B^{-1/2}(x0 - xb)
(the usual preconditioning, which makes the background term the identity) with L-BFGS.

:func:`gradient_check` compares the autograd gradient with finite differences (the adjoint test of operational
systems); :func:`twin_experiment` runs the classical identical-twin set-up on Lorenz-96.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

__all__ = ["Observation", "Var4D", "Var4DResult", "gradient_check", "twin_experiment"]

Tensor = torch.Tensor


@dataclass
class Observation:
    """Observations ``y`` at model step ``step``; ``H`` maps the state to observation space (default identity) and
    ``R`` is the error covariance (scalar variance, vector of variances, or full matrix)."""
    step: int
    y: Tensor
    H: Callable[[Tensor], Tensor] | None = None
    R: float | Tensor = 1.0

    def misfit(self, x: Tensor) -> Tensor:
        d = (self.H(x) if self.H is not None else x) - self.y
        R = torch.as_tensor(self.R, dtype=d.dtype, device=d.device)
        if R.dim() == 0:
            return 0.5 * (d * d).sum() / R
        if R.dim() == 1:
            return 0.5 * (d * d / R).sum()
        return 0.5 * d @ torch.linalg.solve(R, d)


@dataclass
class Var4DResult:
    x0: Tensor
    trajectory: Tensor
    cost_initial: float
    cost_final: float
    cost_history: list[float] = field(default_factory=list)
    n_iterations: int = 0
    background_term: float = 0.0
    observation_term: float = 0.0


class Var4D:
    """``model(x) -> x_next``; ``B_sqrt``: square root of the background covariance (scalar std, vector of stds, or
    a matrix L with B = L L^T)."""

    def __init__(self, model: Callable[[Tensor], Tensor], B_sqrt: float | Tensor, dtype=torch.float64):
        self.model = model
        self.B_sqrt = B_sqrt
        self.dtype = dtype

    def _L(self, v: Tensor) -> Tensor:
        L = torch.as_tensor(self.B_sqrt, dtype=v.dtype, device=v.device)
        return L * v if L.dim() <= 1 else L @ v

    def run_model(self, x0: Tensor, n_steps: int) -> Tensor:
        xs = [x0]
        for _ in range(n_steps):
            xs.append(self.model(xs[-1]))
        return torch.stack(xs)

    def cost_terms(self, v: Tensor, xb: Tensor, obs: Sequence[Observation], n_steps: int):
        x0 = xb + self._L(v)
        traj = self.run_model(x0, n_steps)
        jb = 0.5 * (v * v).sum()
        jo = sum(o.misfit(traj[o.step]) for o in obs) if obs else torch.zeros((), dtype=v.dtype)
        return jb, jo, traj

    def cost(self, x0: Tensor, xb: Tensor, obs: Sequence[Observation], n_steps: int | None = None) -> Tensor:
        """J at x0 in physical space (handy for gradient checks)."""
        n_steps = n_steps if n_steps is not None else max(o.step for o in obs)
        L = torch.as_tensor(self.B_sqrt, dtype=x0.dtype)
        v = (x0 - xb) / L if L.dim() <= 1 else torch.linalg.solve(L, x0 - xb)
        jb, jo, _ = self.cost_terms(v, xb, obs, n_steps)
        return jb + jo

    def analyse(self, xb: Tensor, obs: Sequence[Observation], n_steps: int | None = None, max_iter: int = 200,
                tol: float = 1e-10) -> Var4DResult:
        xb = xb.detach().to(self.dtype)
        obs = [Observation(o.step, o.y.detach().to(self.dtype), o.H, o.R) for o in obs]
        n_steps = n_steps if n_steps is not None else max(o.step for o in obs)
        v = torch.zeros_like(xb, requires_grad=True)
        opt = torch.optim.LBFGS([v], lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-14,
                                history_size=20, line_search_fn="strong_wolfe")
        history: list[float] = []

        def closure():
            opt.zero_grad()
            jb, jo, _ = self.cost_terms(v, xb, obs, n_steps)
            J = jb + jo
            J.backward()
            history.append(float(J.detach()))
            return J

        with torch.no_grad():
            jb0, jo0, _ = self.cost_terms(v, xb, obs, n_steps)
        opt.step(closure)
        with torch.no_grad():
            jb, jo, traj = self.cost_terms(v, xb, obs, n_steps)
        x0 = (xb + self._L(v)).detach()
        return Var4DResult(x0, traj.detach(), float(jb0 + jo0), float(jb + jo), history, len(history),
                           float(jb), float(jo))


def gradient_check(var: Var4D, x0: Tensor, xb: Tensor, obs: Sequence[Observation], n_steps: int | None = None,
                   eps: float = 1e-6, seed: int = 0) -> dict[str, float]:
    """Directional derivative of J from autograd versus a central finite difference along a random direction."""
    g = torch.Generator().manual_seed(seed)
    x = x0.detach().to(var.dtype).clone().requires_grad_(True)
    J = var.cost(x, xb.to(var.dtype), obs, n_steps)
    (grad,) = torch.autograd.grad(J, x)
    d = torch.randn(x.shape, generator=g, dtype=var.dtype)
    with torch.no_grad():
        fd = (var.cost(x + eps * d, xb, obs, n_steps) - var.cost(x - eps * d, xb, obs, n_steps)) / (2 * eps)
    ad = float((grad * d).sum())
    return {"autograd": ad, "finite_difference": float(fd), "relative_error": abs(ad - float(fd)) / max(abs(ad), 1e-300)}


def twin_experiment(n: int = 40, window_steps: int = 20, obs_every: int = 2, obs_fraction: float = 0.5,
                    obs_std: float = 1.0, background_std: float = 1.0, dt: float = 0.05, F: float = 8.0,
                    seed: int = 0, spinup_steps: int = 1000) -> dict[str, object]:
    """Identical-twin 4D-Var on Lorenz-96: a truth run, noisy observations of every ``1/obs_fraction``-th variable
    every ``obs_every`` steps, a background perturbed by ``background_std``. Returns RMSE of background and analysis
    at the start and end of the window, and of a forecast of one more window from each."""
    from .models import lorenz96_step

    g = torch.Generator().manual_seed(seed)
    model = lambda x: lorenz96_step(x, dt, F)  # noqa: E731
    x = F * torch.ones(n, dtype=torch.float64)
    x[0] += 0.01
    with torch.no_grad():
        for _ in range(spinup_steps):
            x = model(x)
        truth = [x]
        for _ in range(2 * window_steps):
            truth.append(model(truth[-1]))
    truth = torch.stack(truth)
    idx = torch.arange(0, n, max(1, int(round(1 / obs_fraction))))
    H = lambda s: s[idx]  # noqa: E731
    obs = [Observation(k, truth[k, idx] + obs_std * torch.randn(len(idx), generator=g, dtype=torch.float64), H,
                       obs_std ** 2) for k in range(obs_every, window_steps + 1, obs_every)]
    xb = truth[0] + background_std * torch.randn(n, generator=g, dtype=torch.float64)
    var = Var4D(model, background_std)
    res = var.analyse(xb, obs, window_steps)

    def rmse(a, b):
        return float(torch.sqrt(torch.mean((a - b) ** 2)))

    with torch.no_grad():
        fb = var.run_model(xb, 2 * window_steps)
        fa = var.run_model(res.x0, 2 * window_steps)
    return {"result": res, "n_obs": len(obs) * len(idx), "truth": truth, "background_trajectory": fb,
            "analysis_trajectory": fa, "observed_index": idx,
            "rmse_background_t0": rmse(xb, truth[0]), "rmse_analysis_t0": rmse(res.x0, truth[0]),
            "rmse_background_end": rmse(fb[window_steps], truth[window_steps]),
            "rmse_analysis_end": rmse(fa[window_steps], truth[window_steps]),
            "rmse_background_forecast": rmse(fb[-1], truth[-1]), "rmse_analysis_forecast": rmse(fa[-1], truth[-1])}
