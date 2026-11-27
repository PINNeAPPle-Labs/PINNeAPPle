"""One assimilation object over the two analysis kernels: the window problem, not two APIs (#346).

Four-dimensional variational assimilation (:mod:`pinneapple_analysis.data_assimilation.var4d`, gradients through the
rollout by autograd) and the ensemble Kalman analysis (:func:`pinneapple_analysis.state_estimation.kalman.enkf_analysis`)
were delivered as two call signatures: one takes a torch model and a list of ``Observation``, the other takes a
numpy ensemble and an observation vector. What a user has is *one* problem -- a model, a background, a window of
readings -- and what they change is *which analysis* they run on it. :class:`Assimilation` is that problem:

    >>> assimilate = Assimilation(model, background=xb, B=1.0, observations=obs)   # doctest: +SKIP
    >>> variational = assimilate.analyse("4dvar")                                  # doctest: +SKIP
    >>> ensemble = assimilate.analyse("enkf", n_ens=50, inflation=1.05)            # doctest: +SKIP

The method has to be named: the two kernels are different estimators and the object does not pretend otherwise. What
it does share is everything around them -- the model callable (``(..., n) -> (..., n)``, so the same callable serves
one state in 4D-Var and ``n_ens`` states in the ensemble), the background with its ``B``, the observation record
(step, ``y``, ``H``, ``R``, with ``H`` any differentiable callable a network included) and the window length -- so a
neural surrogate enters both paths without being wrapped twice. The report is the same shape for both, and it carries
the RMSE against a truth when one is given, because a filter that is not compared with the truth it is supposed to
recover has not been shown to work (the same rule as :class:`pinneapple_analysis.state_estimation.estimator.StateEstimator`).

What the kernels do *not* share, and what a comparison between them therefore has to keep in mind: 4D-Var analyses
the state at the **start** of the window using every observation in it (a smoother), the ensemble filter analyses the
state at each observation time using the observations up to it (a filter). ``x_start`` of an ``"enkf"`` report is the
background, not an analysis -- the filter has no reading before its first observation time. ``x_end`` is the analysed
state at the end of the window for both, and the honest comparison of two models (or two kernels) is per time.

See :mod:`pinneapple_analysis.data_assimilation.neural` for the twin experiment that measures this with a surrogate
trained in this repository, against the true-model path and against no assimilation.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..state_estimation.kalman import enkf_analysis, gaspari_cohn
from .var4d import Observation, Var4D

__all__ = ["AnalysisReport", "Assimilation", "METHODS", "cyclic_localisation"]

#: the analysis kernels the object runs
METHODS = ("4dvar", "enkf")

Tensor = torch.Tensor

#: options each kernel takes, so a typo is refused instead of silently ignored
_OPTIONS = {"4dvar": ("max_iter", "tol"), "enkf": ("n_ens", "inflation", "localisation", "seed")}


def cyclic_localisation(n_state: int, observed_index: Sequence[int], half_width: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Gaspari-Cohn localisation for a state on a ring: the weights ``loc_xy`` (``n_state``, ``m``) between every
    variable and every observed variable, and ``loc_yy`` (``m``, ``m``) between the observations themselves.

    The distance is the cyclic one, ``min(|i - j|, n - |i - j|)``, which is what a periodic 1-D model such as
    Lorenz-96 needs; ``half_width`` is in grid points and the weight vanishes beyond two half-widths.
    """
    n = int(n_state)
    idx = np.asarray(observed_index, dtype=int)
    i = np.arange(n)[:, None]
    d_xy = np.abs(i - idx[None, :])
    d_xy = np.minimum(d_xy, n - d_xy) / float(half_width)
    d_yy = np.abs(idx[:, None] - idx[None, :])
    d_yy = np.minimum(d_yy, n - d_yy) / float(half_width)
    return gaspari_cohn(d_xy), gaspari_cohn(d_yy)


@dataclass
class AnalysisReport:
    """What an analysis came to: the state at both ends of the window, the trajectory, and the error against the
    truth when it is known.

    ``x_start`` is the analysis at the start of the window for ``"4dvar"`` and the background for ``"enkf"`` (a
    filter has no reading before its first observation time, so what it reports there is the mean of the initial
    draws, the background plus sampling noise); ``x_end`` is the analysis at the end for both.
    """

    method: str
    x_start: Tensor                     # (n,) analysis at step 0 -- the background for the filter
    x_end: Tensor                       # (n,) analysis at step n_steps
    states: Tensor                      # (n_steps + 1, n) the analysed trajectory over the window
    spread: Tensor | None = None        # (n_steps + 1, n) ensemble std, None for 4D-Var
    cost_final: float | None = None     # final cost J, 4D-Var only
    cost_history: list[float] = field(default_factory=list)
    n_observations: int = 0
    rmse: dict[str, float] | None = None  # {"start", "end", "window"} against `truth`, the metric of twin_experiment

    @property
    def n_steps(self) -> int:
        return int(self.states.shape[0]) - 1

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method, "n_steps": self.n_steps,
                               "x_start": self.x_start.tolist(), "x_end": self.x_end.tolist(),
                               "n_observations": self.n_observations}
        if self.rmse is not None:
            out["rmse"] = dict(self.rmse)
        if self.spread is not None:
            out["spread_end"] = float(self.spread[-1].mean())
        if self.cost_final is not None:
            out["cost_final"] = self.cost_final
        return out


def _rmse(a: Tensor, b: Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)))


class Assimilation:
    """A model, a background with its ``B``, and a window of observations: the problem an analysis is run on.

    Parameters
    ----------
    model:
        One step of the forward model, ``x -> M(x)``, differentiable (4D-Var takes the adjoint from autograd) and
        accepting a leading batch dimension (``(..., n) -> (..., n)``), so the ensemble members go through it in one
        call. It may be a numerical solver written in torch, a neural network, or a network inside a solver.
    background:
        ``x_b``, the prior state at step 0.
    B:
        The background-error covariance: a scalar std (``B = B**2 I``), a vector of stds (diagonal ``B``), or a full
        matrix (``B`` itself, taken through its Cholesky factor).
    observations:
        :class:`~pinneapple_analysis.data_assimilation.var4d.Observation` records, ``step`` in ``1..n_steps`` and
        possibly ``0``; ``H`` is any differentiable callable -- an index selection, a physical instrument model or a
        trained network -- and ``R`` a variance, a vector of variances or a covariance matrix.
    n_steps:
        Length of the window, default the last observation step.
    """

    def __init__(self, model: Callable[[Tensor], Tensor], background: Tensor, B: float | Tensor = 1.0,
                 observations: Iterable[Observation] = (), n_steps: int | None = None, dtype=torch.float64):
        self.model = model
        self.dtype = dtype
        self.background = self._tensor(background)
        self.B_sqrt = self._sqrt_of(B, self.background.numel())
        self.observations = [self._cast(o) for o in observations]
        last = max((o.step for o in self.observations), default=0)
        self.n_steps = int(n_steps) if n_steps is not None else int(last)
        if self.n_steps < last:
            raise ValueError(f"n_steps={self.n_steps} is before the last observation at step {last}")

    # -- construction helpers -------------------------------------------------
    def _tensor(self, value: Any) -> Tensor:
        return torch.as_tensor(value, dtype=self.dtype).clone()

    def _sqrt_of(self, B: float | Tensor, n: int) -> Tensor:
        raw = torch.as_tensor(B, dtype=self.dtype)
        if raw.dim() == 0:
            return raw.abs()
        if raw.dim() == 1:
            if raw.numel() != n:
                raise ValueError(f"B has {raw.numel()} stds but the state has {n} variables")
            return raw.abs()
        if raw.shape != (n, n):
            raise ValueError(f"B is {tuple(raw.shape)}, expected ({n}, {n})")
        return torch.linalg.cholesky(raw)

    def _cast(self, o: Observation) -> Observation:
        return Observation(int(o.step), self._tensor(o.y), o.H, self._tensor(o.R))

    # -- the background ensemble ---------------------------------------------
    def sample_background(self, n_ens: int, seed: int = 0) -> Tensor:
        """``n_ens`` draws of the background: ``x_b + L z``, ``L L^T = B``, shape ``(n_ens, n)``."""
        g = torch.Generator().manual_seed(int(seed))
        z = torch.randn((int(n_ens), self.background.numel()), generator=g, dtype=self.dtype)
        L = self.B_sqrt
        return self.background + (z * L if L.dim() <= 1 else z @ L.T)

    def run_model(self, x0: Tensor, n_steps: int | None = None) -> Tensor:
        """The model trajectory ``(n_steps + 1, n)`` from ``x0``, no observations involved."""
        n_steps = self.n_steps if n_steps is None else int(n_steps)
        xs = [x0]
        with torch.no_grad():
            for _ in range(n_steps):
                xs.append(self.model(xs[-1]))
        return torch.stack(xs)

    # -- the one entry point --------------------------------------------------
    def analyse(self, method: str = "4dvar", *, truth: Tensor | None = None, **options: Any) -> AnalysisReport:
        """Run one analysis kernel on this problem.

        ``method="4dvar"`` takes ``max_iter`` and ``tol``; ``method="enkf"`` takes ``n_ens``, ``inflation``,
        ``localisation`` (the ``(loc_xy, loc_yy)`` pair, e.g. from :func:`cyclic_localisation`) and ``seed``.
        ``truth``, when given, fills ``report.rmse``.
        """
        key = str(method).lower()
        if key not in METHODS:
            raise ValueError(f"unknown analysis {method!r}; one of {', '.join(METHODS)}")
        unknown = set(options) - set(_OPTIONS[key])
        if unknown:
            raise TypeError(f"{key} does not take {', '.join(sorted(unknown))}; it takes "
                            f"{', '.join(_OPTIONS[key]) or 'nothing'}")
        report = self._variational(**options) if key == "4dvar" else self._ensemble(**options)
        if truth is not None:
            report.rmse = self.verify(report, truth)
        return report

    def _variational(self, max_iter: int = 200, tol: float = 1e-10) -> AnalysisReport:
        res = Var4D(self.model, self.B_sqrt, dtype=self.dtype).analyse(self.background, self.observations,
                                                                      self.n_steps, max_iter, tol)
        return AnalysisReport(method="4dvar", x_start=res.x0, x_end=res.trajectory[-1], states=res.trajectory,
                              cost_final=res.cost_final, cost_history=list(res.cost_history),
                              n_observations=len(self.observations))

    def _ensemble(self, n_ens: int = 50, inflation: float = 1.05,
                  localisation: tuple[np.ndarray, np.ndarray] | None = None, seed: int = 0) -> AnalysisReport:
        if int(n_ens) < 2:
            raise ValueError("the ensemble analysis needs n_ens >= 2")
        rng = np.random.default_rng(int(seed))
        ens = self.sample_background(n_ens, seed)
        loc_xy, loc_yy = (None, None) if localisation is None else localisation
        by_step: dict[int, Observation] = {}
        for o in self.observations:
            if o.step in by_step:
                raise ValueError(f"two observations at step {o.step}: one Observation carries a taller H and R")
            by_step[o.step] = o

        states, spreads = [ens.mean(dim=0).clone()], [ens.std(dim=0, unbiased=True).clone()]
        if 0 in by_step:
            ens = self._update(ens, by_step[0], rng, inflation, loc_xy, loc_yy)
            states[0], spreads[0] = ens.mean(dim=0), ens.std(dim=0, unbiased=True)
        for step in range(1, self.n_steps + 1):
            with torch.no_grad():
                ens = self.model(ens)
            if step in by_step:
                ens = self._update(ens, by_step[step], rng, inflation, loc_xy, loc_yy)
            states.append(ens.mean(dim=0))
            spreads.append(ens.std(dim=0, unbiased=True))
        return AnalysisReport(method="enkf", x_start=states[0], x_end=states[-1], states=torch.stack(states),
                              spread=torch.stack(spreads), n_observations=len(self.observations))

    def _update(self, ens: Tensor, o: Observation, rng: np.random.Generator, inflation: float,
                loc_xy: np.ndarray | None, loc_yy: np.ndarray | None) -> Tensor:
        """One ensemble Kalman analysis step, through the delivered kernel (perturbed observations, localisation,
        inflation): ``enkf_analysis`` works in numpy, so the ensemble crosses over here and comes back as torch."""
        h = (lambda s: s) if o.H is None else o.H
        with torch.no_grad():
            hx = h(ens)
        r_var = _variances(o.R, hx.shape[-1])
        out = enkf_analysis(ens.numpy(), o.y.numpy(), hx.numpy(), r_var, loc_xy=loc_xy, loc_yy=loc_yy,
                            inflation=float(inflation), rng=rng)
        return torch.as_tensor(out, dtype=self.dtype)

    # -- verification ---------------------------------------------------------
    def verify(self, report: AnalysisReport, truth: Tensor) -> dict[str, float]:
        """RMSE of the report against ``truth`` ``(n_steps + 1, n)`` -- the metric ``twin_experiment`` reports --
        at the start of the window, at its end, and over the whole window."""
        t = torch.as_tensor(truth, dtype=self.dtype)
        return {"start": _rmse(report.x_start, t[0]), "end": _rmse(report.x_end, t[-1]),
                "window": _rmse(report.states, t)}


def _variances(R: float | Tensor, n_obs: int) -> np.ndarray:
    """The diagonal of ``R`` as the variances ``enkf_analysis`` takes."""
    r = torch.as_tensor(R)
    if r.dim() == 0:
        return np.full(int(n_obs), float(r))
    if r.dim() == 1:
        return r.numpy().copy()
    return torch.diag(r).numpy().copy()
