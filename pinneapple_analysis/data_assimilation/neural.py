"""Assimilation with a model trained here: a learned right-hand side inside both analysis kernels (#346).

The variational path was measured on a twin experiment and the ensemble path on linear systems; neither had been run
on a network trained in this repository. This module trains one and puts it inside both -- the forward model of the
assimilation, which is the operator the issue is about.

The surrogate is a periodic one-dimensional convolution that predicts the **tendency** ``dx/dt`` of a state on a ring,
normalised by the attractor's offset and spread, integrated with the same RK4 step the truth uses
(:func:`pinneapple_analysis.data_assimilation.models.rk4`). Only the closure is learned: the network has to
rediscover the stencil ``(x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`` of Lorenz-96 from state pairs alone, which is the
honest version of "a dynamics model written by a network". The periodic convolution carries the ring's translation
symmetry, and that is what makes it trainable on 4 800 samples of a 40-dimensional state: a tanh MLP of 512 units over
the whole state (fifty times the parameters here) only reaches tendency RMSE 2.24 in 4 000 steps, having to rediscover
locality from the data, while this one reaches 1.46 in 1 500.

The model (defaults of :func:`train_tendency_surrogate`: 4 800 one-step pairs of a Lorenz-96 run with 40 variables,
dt = 0.05, F = 8, 1 500 Adam steps, batch 128; seed 0), measured on the 1 200 pairs held back:

=======================================================  =========================
tendency error (held-out, state units per time)          1.46
one-step error (held-out, state units; state std 3.64)   0.071
free-run error over the held-out tail, 20 steps          1.57  (1.39 to 1.77, seeds 0-2)
same, with a fifth of the training                       4.73  (the spread of the attractor)
=======================================================  =========================

The free run is the model started from every tenth held-out state and integrated 20 steps -- the length of the
assimilation window -- and compared with the trajectory it was never shown. A random network of the same shape, the
control in the tests, misses a ten-step rollout by more than 1.5. The 20-step number is the one that decides: the
surrogate's error at the end of the window (1.57) is larger than the observation error (1.0), which is why the two
kernels below behave differently.

The twin experiment (:func:`neural_twin_experiment`) runs the two kernels of
:class:`~pinneapple_analysis.data_assimilation.assimilation.Assimilation` over the same observations and the same
background, once with the true Lorenz-96 step and once with the surrogate, plus the free run of each model as the
"no assimilation" reference. RMSE against the truth at the start of the window, at its end, and after a forecast of one
more window made with the **true** model from each analysed state -- so the forecast row compares the *states*, not how
well each model can run on its own. The window is the one of
:func:`~pinneapple_analysis.data_assimilation.var4d.twin_experiment` (20 steps, half the variables observed every 2
steps, observation error 1.0, background error 1.0), so the numbers are comparable with it. Mean of the three seeds
0, 1 and 2 (the spread is quoted where it is wide), reproduced by

    python -m pinneapple_analysis.data_assimilation.neural --seeds 0,1,2 --train-steps 300,1500 \\
        --json docs/dev/neural_assimilation.json

=================  ======================  ==============  ======  ==========
model              analysis                window start    end     forecast
=================  ======================  ==============  ======  ==========
true model         none (free run)         0.94            3.48    4.14
surrogate          none (free run)         0.94            3.39    4.09
true model         4D-Var                  0.57            0.42    2.47
surrogate          4D-Var                  0.73            0.83    2.71
true model         ensemble Kalman filter  0.95            0.63    3.22
surrogate          ensemble Kalman filter  0.95            0.64    3.21
=================  ======================  ==============  ======  ==========

Read it as three results.

* **The ensemble filter on the surrogate is the ensemble filter on the truth.** Per seed it is 0.84, 1.03 and 1.14
  times the true-model value: once the one-step error (0.07) is well under the observation error (1.0), the filter --
  which re-anchors at every observation -- does not care that the model is a network. Both are 5.3x better than no
  assimilation at the end of the window.
* **4D-Var inherits the model's error.** Per seed 0.88, 1.44 and 4.10 times the true-model 4D-Var, mean 2.1x: the
  analysis fits a whole 20-step trajectory, and by the end of it the surrogate is off by more than the observation
  error. It still beats no assimilation by 2.6-6.1x, and its state at the *start* of the window (0.73) is close to the
  true-model one (0.57) -- the variational problem absorbs the model's bias into the state it recovers.
* **An undertrained network loses most of the gain, and the test says so.** The same architecture with a fifth of the
  training (300 steps: one-step error 0.39, tendency error 7.8) gives 2.30 at the end of the window for the filter and
  2.30 for 4D-Var, against 0.64 and 0.42 with the trained network and 3.39 for no assimilation: 3.6x and 5.5x worse
  than the true-model path, and only 1.5x better than doing nothing. Training the model is the whole difference.

The forecast column spreads widely over seeds (0.94-4.16 for the true-model 4D-Var) because 20 more steps of Lorenz-96
is at the edge of predictability for every method here; it is reported because the issue asks for the forecast gain,
not because it separates the methods.

The numbers are asserted in ``tests/pinneapple_analysis/test_neural_assimilation.py`` (on seed 0, with the bounds that
held on all three seeds; the seed-dependent ratios are quoted above as spreads, not asserted).
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from .assimilation import Assimilation, cyclic_localisation
from .models import lorenz96_step, lorenz96_tendency, rk4
from .var4d import Observation

__all__ = ["PeriodicTendency", "lorenz96_trajectory", "neural_twin_experiment", "surrogate_step",
           "train_tendency_surrogate"]

Tensor = torch.Tensor


class PeriodicTendency(torch.nn.Module):
    """A learned right-hand side for a state on a ring: ``x -> dx/dt`` with circular convolution, so every variable
    is treated the same way and the ring has no edge.

    ``offset`` and ``scale`` normalise the state to zero mean and unit-ish spread (the attractor of Lorenz-96 sits
    around ``F`` with a standard deviation near 3.6), which is all the conditioning the regression needs.
    """

    def __init__(self, width: int = 32, layers: int = 2, kernel: int = 5, offset: float = 8.0, scale: float = 4.0):
        super().__init__()
        self.offset = float(offset)
        self.scale = float(scale)
        pad = kernel // 2
        mods: list[torch.nn.Module] = [
            torch.nn.Conv1d(1, width, kernel, padding=pad, padding_mode="circular"), torch.nn.Tanh()]
        for _ in range(int(layers) - 1):
            mods += [torch.nn.Conv1d(width, width, kernel, padding=pad, padding_mode="circular"), torch.nn.Tanh()]
        mods.append(torch.nn.Conv1d(width, 1, kernel, padding=pad, padding_mode="circular"))
        self.core = torch.nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:
        z = ((x - self.offset) / self.scale).unsqueeze(-2)
        return self.core(z).squeeze(-2)


def surrogate_step(model: PeriodicTendency, dt: float = 0.05):
    """The forward model of the assimilation: the learned tendency integrated by the RK4 step of the truth."""
    return lambda x: rk4(lambda y: model(y), x, dt)


def lorenz96_trajectory(n: int = 40, steps: int = 6000, dt: float = 0.05, F: float = 8.0, seed: int = 0,
                        spinup_steps: int = 1000, perturbation: float = 0.01) -> Tensor:
    """A trajectory of the true Lorenz-96 step, ``(steps, n)``, after ``spinup_steps`` from a state perturbed by
    ``perturbation`` -- the training data of the surrogate and, with another ``seed`` (another orbit of the same
    attractor), the truth of the twin experiment. The two must be different runs: a surrogate trained on the window
    it is then asked to analyse has been shown nothing."""
    x = F * torch.ones(n, dtype=torch.float64)
    x = x + perturbation * torch.randn(n, generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64)
    out = []
    with torch.no_grad():
        for _ in range(int(spinup_steps)):
            x = lorenz96_step(x, dt, F)
        for _ in range(int(steps)):
            out.append(x)
            x = lorenz96_step(x, dt, F)
    return torch.stack(out)


def train_tendency_surrogate(trajectory: Tensor, dt: float = 0.05, *, width: int = 32, layers: int = 2,
                             kernel: int = 5, steps: int = 1500, batch: int = 128, lr: float = 3e-3,
                             seed: int = 0, holdout: float = 0.2, rollout: int = 20,
                             rollout_stride: int = 10) -> tuple[PeriodicTendency, dict[str, float]]:
    """Fit :class:`PeriodicTendency` to the tendency of ``trajectory`` (``(steps, n)``), by Adam on the mean squared
    error, on the first ``1 - holdout`` of it; every metric is measured on the rest, which training never saw.

    ``rollout`` is the length of the free-run error reported: the model is started from every
    ``rollout_stride``-th held-out state and integrated ``rollout`` steps, and the RMSE against the trajectory it
    was never shown is the number that decides whether an assimilation window of that length can work on it.

    Returns the model (float64, ready for both kernels of the assimilation) and
    ``{"tendency_rmse", "step_rmse", "free_run_rmse", "free_run_steps", "n_train", "n_holdout"}``.
    """
    data = torch.as_tensor(trajectory, dtype=torch.float32)
    with torch.no_grad():
        target = lorenz96_tendency(data)
    split = max(2, int(round(data.shape[0] * (1.0 - float(holdout)))))
    train_x, train_t = data[:split], target[:split]
    held_x, held_t = data[split:], target[split:]

    torch.manual_seed(int(seed))
    model = PeriodicTendency(width, layers, kernel)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(steps))
    g = torch.Generator().manual_seed(int(seed) + 1)
    for _ in range(int(steps)):
        idx = torch.randint(0, train_x.shape[0], (int(batch),), generator=g)
        loss = ((model(train_x[idx]) - train_t[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model = model.double()
    with torch.no_grad():
        hx, ht = held_x.double(), held_t.double()
        tendency_rmse = float(torch.sqrt(((model(hx) - ht) ** 2).mean()))
        step_rmse = float(torch.sqrt(((rk4(lambda y: model(y), hx[:-1], dt) - hx[1:]) ** 2).mean()))
        errors = []
        for k0 in range(0, hx.shape[0] - int(rollout), max(1, int(rollout_stride))):
            x = hx[k0].clone()
            for _ in range(int(rollout)):
                x = rk4(lambda y: model(y), x, dt)
            errors.append(torch.mean((x - hx[k0 + int(rollout)]) ** 2))
        free_run = float(torch.sqrt(torch.stack(errors).mean()))
    metrics = {"tendency_rmse": tendency_rmse, "step_rmse": step_rmse, "free_run_rmse": free_run,
               "free_run_steps": int(rollout),
               "n_train": int(split), "n_holdout": int(data.shape[0] - split)}
    return model, metrics


def _rmse(a: Tensor, b: Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)))


def neural_twin_experiment(n: int = 40, window_steps: int = 20, obs_every: int = 2, obs_fraction: float = 0.5,
                           obs_std: float = 1.0, background_std: float = 1.0, dt: float = 0.05, F: float = 8.0,
                           seed: int = 0, spinup_steps: int = 1000, train_steps: int = 1500,
                           n_ens: int = 50, inflation: float = 1.05, localisation_width: float = 4.0,
                           max_iter: int = 200) -> dict[str, Any]:
    """Identical-twin assimilation on Lorenz-96 with a surrogate trained inside the call, against the true-model
    path and against no assimilation.

    The set-up is the one of :func:`pinneapple_analysis.data_assimilation.var4d.twin_experiment` -- a truth run, noisy
    observations of every ``1/obs_fraction``-th variable every ``obs_every`` steps, a background perturbed by
    ``background_std`` -- so the numbers are directly comparable with it. On top of it: a network trained on a
    *disjoint* trajectory (seed ``seed + 101``), the same observations assimilated with the true step and with the
    surrogate, and for each model the free run from the background as the "no assimilation" reference.

    Returns ``{"model", "surrogate", "rmse", "cost", "reports", "truth", ...}``: ``model`` is the trained network,
    ``surrogate`` its held-out metrics, ``rmse`` the ``{case: {"start", "end", "forecast"}}`` table and ``reports``
    the :class:`~pinneapple_analysis.data_assimilation.assimilation.AnalysisReport` of each case. The forecast row
    propagates each analysed state for ``window_steps`` more steps with the **true** model, so it compares the states
    recovered, not the models' own free runs.
    """
    torch.manual_seed(int(seed))
    train_data = lorenz96_trajectory(n=n, dt=dt, F=F, seed=int(seed) + 101, spinup_steps=spinup_steps)
    model, surrogate = train_tendency_surrogate(train_data, dt, steps=train_steps, seed=int(seed))

    g = torch.Generator().manual_seed(int(seed))
    truth = lorenz96_trajectory(n=n, steps=2 * window_steps + 1, dt=dt, F=F, seed=int(seed),
                               spinup_steps=spinup_steps)
    step_true = lambda s: lorenz96_step(s, dt, F)  # noqa: E731

    idx = torch.arange(0, n, max(1, int(round(1 / obs_fraction))))
    H = lambda s: s[..., idx]  # noqa: E731
    obs = [Observation(k, truth[k, idx] + obs_std * torch.randn(len(idx), generator=g, dtype=torch.float64), H,
                       obs_std ** 2) for k in range(obs_every, window_steps + 1, obs_every)]
    xb = truth[0] + background_std * torch.randn(n, generator=g, dtype=torch.float64)

    models = {"true": step_true, "neural": surrogate_step(model, dt)}
    rmse: dict[str, dict[str, float]] = {}
    reports: dict[str, Any] = {}
    cost: dict[str, float] = {}
    with torch.no_grad():
        for name, step in models.items():
            free = [xb]
            for _ in range(2 * window_steps):
                free.append(step(free[-1]))
            free = torch.stack(free)
            rmse[f"none_{name}"] = {"start": _rmse(xb, truth[0]), "end": _rmse(free[window_steps], truth[window_steps]),
                                    "forecast": _rmse(free[-1], truth[-1])}
            problem = Assimilation(step, xb, background_std, obs, window_steps)
            for method, options in (("4dvar", {"max_iter": max_iter}),
                                    ("enkf", {"n_ens": n_ens, "inflation": inflation, "seed": int(seed),
                                              "localisation": cyclic_localisation(n, idx, localisation_width)})):
                report = problem.analyse(method, truth=truth[: window_steps + 1], **options)
                key = f"{method}_{name}"
                reports[key] = report
                with torch.no_grad():
                    tail = [report.x_end]
                    for _ in range(window_steps):
                        tail.append(step_true(tail[-1]))
                rmse[key] = {"start": report.rmse["start"], "end": report.rmse["end"],
                             "forecast": _rmse(tail[-1], truth[-1])}
                if report.cost_final is not None:
                    cost[key] = report.cost_final
    return {"model": model, "surrogate": surrogate, "rmse": rmse, "cost": cost, "reports": reports, "truth": truth,
            "observations": obs, "background": xb, "observed_index": idx}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the twin experiment and print (and optionally write) its numbers.

    Each training budget is a separate surrogate, trained and measured with the same observations, so the table shows
    what the model's own quality does to the analysis.
    """
    ap = argparse.ArgumentParser(description="Neural-model data assimilation on Lorenz-96 (#346)")
    ap.add_argument("--json", default=None, help="write the measured numbers here")
    ap.add_argument("--seeds", default="0", help="comma separated, one twin experiment each")
    ap.add_argument("--train-steps", default="300,1500", help="comma separated training budgets")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    budgets = [int(s) for s in str(args.train_steps).split(",") if s.strip()]
    runs: dict[str, list[dict[str, Any]]] = {}
    for budget in budgets:
        runs[str(budget)] = []
        for seed in seeds:
            out = neural_twin_experiment(seed=seed, train_steps=budget)
            runs[str(budget)].append({"seed": seed, "surrogate": out["surrogate"], "rmse": out["rmse"],
                                      "cost": out["cost"]})
            print(f"budget {budget} seed {seed}: surrogate step rmse {out['surrogate']['step_rmse']:.3f}, "
                  f"4dvar_neural end {out['rmse']['4dvar_neural']['end']:.3f} "
                  f"enkf_neural end {out['rmse']['enkf_neural']['end']:.3f}", flush=True)
    numbers = {"seeds": seeds, "train_steps": budgets, "runs": runs}
    text = json.dumps(numbers, indent=1, sort_keys=True)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return numbers


if __name__ == "__main__":
    main()
