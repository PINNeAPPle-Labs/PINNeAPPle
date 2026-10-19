"""Operator inference with inputs: a discrete-time polynomial reduced model fitted by ridge regression.

``pinneapple_neural.architectures.rom.OperatorInference`` learns autonomous latent dynamics. Operating assets
are driven by inputs (pumps, heaters, valves) and disturbances, so this variant adds them (Peherstorfer & Willcox
2016, "Data-driven operator inference for nonintrusive projection-based model reduction", CMAME 306, sec. 3.3)::

    x_{k+1} = x_k + dt * ( c + A x_k + H (x_k (x) x_k) + B v_k + N (x_k (x) v_k) )

with ``v = [inputs, disturbances]``, all variables scaled to zero mean / unit spread on the training data.
The model is a few matrices: it serialises to JSON, runs in microseconds and is what the digital twin and the
controller can use as a surrogate of an expensive plant model.

Validation is part of the fit: :func:`fit_controlled_opinf` holds out whole trajectories (not random samples)
and reports one-step and full-rollout errors on them, plus the training box of every variable.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


def _features(xs: np.ndarray, vs: np.ndarray, quadratic: bool, bilinear: bool) -> np.ndarray:
    cols = [np.ones((xs.shape[0], 1)), xs, vs]
    if quadratic:
        iu = np.triu_indices(xs.shape[1])
        cols.append((xs[:, :, None] * xs[:, None, :])[:, iu[0], iu[1]])
    if bilinear:
        cols.append((xs[:, :, None] * vs[:, None, :]).reshape(xs.shape[0], -1))
    return np.concatenate(cols, axis=1)


@dataclass
class ControlledOpInf:
    """A fitted model; build it with :func:`fit_controlled_opinf` or :meth:`from_dict`."""

    state_names: list[str]
    input_names: list[str]
    dt: float
    W: np.ndarray                       # (n_features, n_state), acting on scaled variables
    x_mean: np.ndarray
    x_scale: np.ndarray
    v_mean: np.ndarray
    v_scale: np.ndarray
    quadratic: bool = True
    bilinear: bool = True
    bounds: dict[str, list[float]] = field(default_factory=dict)
    validation: dict[str, object] = field(default_factory=dict)

    def step(self, x, v) -> np.ndarray:
        """One step of ``dt`` seconds from state(s) ``x`` (n,) or (N, n) with inputs ``v`` (m,) or (N, m)."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        single = x.ndim == 1
        X = np.atleast_2d(x)
        V = np.broadcast_to(np.atleast_2d(v), (X.shape[0], self.v_mean.size))
        xs = (X - self.x_mean) / self.x_scale
        vs = (V - self.v_mean) / self.v_scale
        dxs = _features(xs, vs, self.quadratic, self.bilinear) @ self.W
        out = X + dxs * self.x_scale
        return out[0] if single else out

    def rollout(self, x0, V) -> np.ndarray:
        """States at every step for an input sequence ``V`` (K, m); returns (K + 1, n)."""
        xs = [np.asarray(x0, dtype=float)]
        for v in np.asarray(V, dtype=float):
            xs.append(self.step(xs[-1], v))
        return np.array(xs)

    def casadi_step(self, x, v):
        """The same step on CasADi symbols (column vectors), for optimisation with IPOPT."""
        import casadi as ca

        n, m = self.x_mean.size, self.v_mean.size
        xs = [(x[i] - self.x_mean[i]) / self.x_scale[i] for i in range(n)]
        vs = [(v[j] - self.v_mean[j]) / self.v_scale[j] for j in range(m)]
        feats = [1.0] + xs + vs
        if self.quadratic:
            feats += [xs[i] * xs[j] for i in range(n) for j in range(i, n)]
        if self.bilinear:
            feats += [xs[i] * vs[j] for i in range(n) for j in range(m)]
        dxs = ca.mtimes(ca.DM(self.W.T), ca.vertcat(*feats))
        return ca.vertcat(*[x[i] + dxs[i] * self.x_scale[i] for i in range(n)])

    def to_dict(self) -> dict:
        return {"kind": "controlled_opinf", "state_names": self.state_names, "input_names": self.input_names,
                "dt": self.dt, "W": self.W.tolist(), "x_mean": self.x_mean.tolist(),
                "x_scale": self.x_scale.tolist(), "v_mean": self.v_mean.tolist(), "v_scale": self.v_scale.tolist(),
                "quadratic": self.quadratic, "bilinear": self.bilinear, "bounds": self.bounds,
                "validation": self.validation}

    @classmethod
    def from_dict(cls, d: dict) -> ControlledOpInf:
        if d.get("kind") != "controlled_opinf":
            raise ValueError(f"not a controlled_opinf model: kind={d.get('kind')!r}")
        arr = {k: np.asarray(d[k], dtype=float) for k in ("W", "x_mean", "x_scale", "v_mean", "v_scale")}
        return cls(state_names=list(d["state_names"]), input_names=list(d["input_names"]), dt=float(d["dt"]),
                   quadratic=bool(d["quadratic"]), bilinear=bool(d["bilinear"]), bounds=d.get("bounds", {}),
                   validation=d.get("validation", {}), **arr)


def _rel_rmse(pred: np.ndarray, true: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / scale


def fit_controlled_opinf(trajectories: Sequence[dict[str, np.ndarray]], state_names: Sequence[str],
                         input_names: Sequence[str], dt: float, *, holdout: int = 1, ridge: float = 1e-6,
                         quadratic: bool = True, bilinear: bool = True) -> ControlledOpInf:
    """Fit on all but the last ``holdout`` trajectories and validate on those.

    Each trajectory is a dict of equally spaced columns (spacing ``dt``) containing ``state_names`` and
    ``input_names``. The returned model's ``validation`` holds, per state, the one-step and rollout RMSE on the
    held-out trajectories (absolute, and relative to the state's spread), and ``bounds`` the training box.
    """
    state_names, input_names = list(state_names), list(input_names)
    if len(trajectories) <= holdout:
        raise ValueError(f"need more than {holdout} trajectories to hold {holdout} out")
    train, test = list(trajectories[:-holdout or None]), list(trajectories[len(trajectories) - holdout:])

    def stack(trajs):
        X0, X1, V = [], [], []
        for tr in trajs:
            X = np.column_stack([np.asarray(tr[n], dtype=float) for n in state_names])
            U = np.column_stack([np.asarray(tr[n], dtype=float) for n in input_names])
            X0.append(X[:-1])
            X1.append(X[1:])
            V.append(U[:-1])
        return np.concatenate(X0), np.concatenate(X1), np.concatenate(V)

    X0, X1, V = stack(train)
    x_mean, x_scale = X0.mean(0), X0.std(0) + 1e-12
    v_mean, v_scale = V.mean(0), V.std(0) + 1e-12
    xs, vs = (X0 - x_mean) / x_scale, (V - v_mean) / v_scale
    Y = (X1 - X0) / x_scale
    F = _features(xs, vs, quadratic, bilinear)
    G = F.T @ F + ridge * len(F) * np.eye(F.shape[1])
    W = np.linalg.solve(G, F.T @ Y)
    allv = np.column_stack([X0, V])
    names = state_names + input_names
    bounds = {n: [float(allv[:, j].min()), float(allv[:, j].max())] for j, n in enumerate(names)}
    model = ControlledOpInf(state_names, input_names, float(dt), W, x_mean, x_scale, v_mean, v_scale,
                            quadratic, bilinear, bounds)

    T0, T1, TV = stack(test)
    one = model.step(T0, TV)
    roll_err, roll_true = [], []
    for tr in test:
        X = np.column_stack([np.asarray(tr[n], dtype=float) for n in state_names])
        U = np.column_stack([np.asarray(tr[n], dtype=float) for n in input_names])
        roll_err.append(model.rollout(X[0], U[:-1]))
        roll_true.append(X)
    R, Rt = np.concatenate(roll_err), np.concatenate(roll_true)
    spread = np.concatenate([T0, T1]).std(0) + 1e-12
    model.validation = {
        "n_train_trajectories": len(train), "n_test_trajectories": len(test),
        "n_train_steps": int(len(X0)), "n_test_steps": int(len(T0)),
        "one_step_rmse": dict(zip(state_names, np.sqrt(np.mean((one - T1) ** 2, 0)).tolist(), strict=False)),
        "rollout_rmse": dict(zip(state_names, np.sqrt(np.mean((R - Rt) ** 2, 0)).tolist(), strict=False)),
        "rollout_rel_rmse": dict(zip(state_names, _rel_rmse(R, Rt, spread).tolist(), strict=False)),
        "rollout_max_abs_error": dict(zip(state_names, np.max(np.abs(R - Rt), 0).tolist(), strict=False)),
        "split": f"whole trajectories held out (last {holdout})",
    }
    return model


def residual_sigma(model: ControlledOpInf, trajectories: Sequence[dict[str, np.ndarray]]) -> dict[str, float]:
    """One-step residual standard deviation per state on ``trajectories`` (use held-out ones): the process noise
    a filter running this model should assume."""
    errs = []
    for tr in trajectories:
        X = np.column_stack([np.asarray(tr[n], dtype=float) for n in model.state_names])
        U = np.column_stack([np.asarray(tr[n], dtype=float) for n in model.input_names])
        errs.append(model.step(X[:-1], U[:-1]) - X[1:])
    e = np.concatenate(errs)
    return dict(zip(model.state_names, e.std(0).tolist(), strict=False))

