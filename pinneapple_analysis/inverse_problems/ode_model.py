"""An ODE system (:class:`pinneapple_physics.biology.ODESystem`, e.g. read from SBML) as a :class:`ForwardModel`.

Observations are rows ``(t, observable)``; the prediction is the observable (a state or an algebraic variable) at
``t``. The model supplies exact sensitivities from the forward sensitivity equations, so the Fisher information, the
profile likelihood and the fits of :mod:`.identifiability` and :mod:`.estimation` use them instead of finite
differences. Parameter names are the system's parameters or ``<state>_0`` for an unknown initial value.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .forward import ForwardModel, Observations, Parameter

__all__ = ["ODEForwardModel"]


class ODEForwardModel(ForwardModel):
    inputs = ("t",)

    def __init__(self, system, observables: Sequence[str], parameters: Sequence[str], *, method: str = "BDF",
                 rtol: float = 1e-8, atol: float = 1e-10, max_step: float = math.inf,
                 bounds: Mapping[str, tuple[float, float]] | None = None, log: bool = True):
        self.system = system
        self.outputs = list(observables)
        bad = [o for o in self.outputs if o not in system.states and o not in system.assignments]
        if bad:
            raise KeyError(f"observable(s) {bad} are neither states nor assignments of {system.name}")
        self.names = list(parameters)
        self.opts = dict(method=method, rtol=rtol, atol=atol, max_step=max_step)
        self.bounds = dict(bounds or {})
        self.log = log
        self.name = f"ode:{system.name}"
        self.description = f"{system.name}: {len(system.states)} states, observing {', '.join(self.outputs)}"
        self.cost = "moderate" if len(system.states) * (1 + len(self.names)) > 60 else "cheap"

    def _nominal(self, n: str) -> float:
        if n in self.system.parameters:
            return self.system.parameters[n]
        if n.endswith("_0") and n[:-2] in self.system.states:
            return float(self.system.initial_values()[self.system.states.index(n[:-2])])
        raise KeyError(n)

    def default_parameters(self) -> list[Parameter]:
        out = []
        for n in self.names:
            v = self._nominal(n)
            lo, hi = self.bounds.get(n, (v / 10, v * 10) if v > 0 else (v - 1 - abs(v), v + 1 + abs(v)))
            out.append(Parameter(n, v, lo, hi, log=self.log and lo > 0))
        return out

    def _split(self, theta: Mapping[str, float]):
        p = {k: float(v) for k, v in theta.items() if k in self.system.parameters}
        y0 = self.system.initial_values(p)
        for k, v in theta.items():
            if k.endswith("_0") and k[:-2] in self.system.states:
                y0[self.system.states.index(k[:-2])] = float(v)
        return p, y0

    def _grid(self, inputs):
        t = np.asarray(inputs["t"], float)
        tt = np.unique(np.concatenate([[min(0.0, t.min())], t]))
        return t, tt, np.searchsorted(tt, t)

    def predict(self, theta, inputs, observable=None):
        t, tt, idx = self._grid(inputs)
        p, y0 = self._split(theta)
        traj = self.system.simulate(tt, p, y0=y0, outputs=[o for o in self.outputs if o in self.system.assignments],
                                    **self.opts)
        obs = np.asarray(observable if observable is not None else [self.outputs[0]] * len(t), dtype=object)
        out = np.empty(len(t))
        for o in set(obs):
            m = obs == o
            out[m] = traj.values[o][idx[m]]
        return out

    def sensitivities(self, theta, obs: Observations, names: Sequence[str]):
        t, tt, idx = self._grid(obs.inputs)
        p, y0 = self._split(theta)
        traj, S = self.system.sensitivities(tt, list(names), p, y0=y0, **self.opts)
        pred = np.empty(len(t))
        J = np.zeros((len(t), len(names)))
        labels = np.asarray(obs.observable, dtype=object)
        for o in set(labels):
            m = labels == o
            if o in self.system.states:
                pred[m] = traj.values[o][idx[m]]
                J[m] = S[idx[m], self.system.states.index(o), :]
            else:
                val, dval = self.system.assignment_sensitivities(o, traj, S, list(names), p)
                pred[m], J[m] = val[idx[m]], dval[idx[m]]
        return pred, J
