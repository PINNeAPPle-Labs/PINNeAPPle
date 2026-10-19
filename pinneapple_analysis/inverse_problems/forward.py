"""Forward models, parameters and observations: the common language of identifiability, calibration and inversion.

One inverse problem is three things:

* a :class:`ForwardModel` that predicts each observation from the parameter values (``predict(theta, inputs,
  observable)``): an analytic formula, a text expression (:class:`ExpressionModel`), any Python callable or external
  solver wrapped by :class:`CallableModel`, or a table of simulated results interpolated over the parameters
  (:class:`TabulatedModel`);
* the unknown :class:`Parameter` list, each with bounds and a scale (log for positive physical constants);
* the :class:`Observations`: one row per measurement, with its inputs (position, time, ...), the observable it
  measures (a temperature, a heat flux, ...), the value (optional before an experiment) and its noise ``sigma``.

The analysis works in scaled coordinates ``z`` (``ln theta`` for log parameters, ``theta / scale`` otherwise), so a
sensitivity ``dy/dz`` of a log parameter is the change of the prediction for a 100 % relative change of the
parameter, and the Fisher information of different units can be compared.

Everything here is NumPy; a model that also defines ``predict_torch`` gets exact autograd Jacobians and gradients,
and a model with ``sensitivities`` (e.g. a discretised PDE with forward/adjoint sensitivities) supplies its own.
"""
from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = ["Parameter", "Observations", "ForwardModel", "CallableModel", "ExpressionModel", "TabulatedModel",
           "ParameterSet", "jacobian", "count_evaluations"]


# --------------------------------------------------------------------------- parameters
@dataclass
class Parameter:
    """One adjustable parameter. ``value`` is the nominal value (analysis point) or the initial guess (estimation).

    ``log`` (default: ``lower > 0``) analyses and estimates the parameter in ``ln`` space, the natural scale of a
    positive physical constant. ``fixed`` keeps it at ``value``. ``truth`` is only for synthetic studies (a known
    answer to check recovery against); it is never used by the estimators."""
    name: str
    value: float
    lower: float = -math.inf
    upper: float = math.inf
    unit: str = ""
    log: bool | None = None
    fixed: bool = False
    description: str = ""
    truth: float | None = None

    def __post_init__(self):
        self.value, self.lower, self.upper = float(self.value), float(self.lower), float(self.upper)
        if self.log is None:
            self.log = self.lower > 0
        if not self.lower <= self.value <= self.upper:
            raise ValueError(f"parameter {self.name}: value {self.value:g} outside [{self.lower:g}, {self.upper:g}]")
        if self.log and not self.lower > 0:
            raise ValueError(f"parameter {self.name}: a log-scaled parameter needs a positive lower bound")

    @property
    def scale(self) -> float:
        """Linear parameters are divided by this (|value|, or 1) so every ``z`` is of order one."""
        return abs(self.value) if self.value != 0 else 1.0

    def to_z(self, v: float) -> float:
        return math.log(v) if self.log else v / self.scale

    def from_z(self, z: float) -> float:
        return math.exp(z) if self.log else z * self.scale

    def z_bounds(self) -> tuple[float, float]:
        lo = self.to_z(self.lower) if math.isfinite(self.lower) and (not self.log or self.lower > 0) else -math.inf
        hi = self.to_z(self.upper) if math.isfinite(self.upper) else math.inf
        return lo, hi

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("lower", "upper"):
            if not math.isfinite(d[k]):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Parameter:
        kw = {k: d[k] for k in ("name", "value", "lower", "upper", "unit", "log", "fixed", "description", "truth")
              if k in d and d[k] is not None}
        kw.setdefault("lower", -math.inf)
        kw.setdefault("upper", math.inf)
        return cls(**kw)


class ParameterSet:
    """The free parameters of a problem and the map between ``theta`` (dict, physical units) and ``z`` (vector)."""

    def __init__(self, parameters: Sequence[Parameter]):
        self.all = list(parameters)
        names = [p.name for p in self.all]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        self.free = [p for p in self.all if not p.fixed]
        if not self.free:
            raise ValueError("at least one parameter must be free (not fixed)")

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.free]

    def __len__(self) -> int:
        return len(self.free)

    def z0(self) -> np.ndarray:
        return np.array([p.to_z(p.value) for p in self.free])

    def theta(self, z: np.ndarray | None = None) -> dict[str, float]:
        """All parameters (fixed ones at their value) as a dict; the free ones from ``z`` when given."""
        out = {p.name: p.value for p in self.all}
        if z is not None:
            for p, zi in zip(self.free, np.asarray(z, float), strict=False):
                out[p.name] = p.from_z(float(zi))
        return out

    def z_of(self, theta: Mapping[str, float]) -> np.ndarray:
        return np.array([p.to_z(float(theta[p.name])) for p in self.free])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        b = np.array([p.z_bounds() for p in self.free])
        return b[:, 0], b[:, 1]

    def clip(self, z: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds()
        return np.clip(z, lo, hi)

    def truth(self) -> dict[str, float] | None:
        if all(p.truth is not None for p in self.free):
            return {p.name: float(p.truth) for p in self.free}
        return None


# --------------------------------------------------------------------------- observations
@dataclass
class Observations:
    """Measurements: ``inputs`` (column name -> array of n), ``observable`` (n labels: which model output each row
    measures), ``values`` (n, NaN or ``None`` before the experiment) and ``sigma`` (n, the noise standard deviation).
    """
    inputs: dict[str, np.ndarray]
    sigma: np.ndarray
    values: np.ndarray | None = None
    observable: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.inputs = {k: np.asarray(v, float).ravel() for k, v in self.inputs.items()}
        n = self.n
        self.sigma = np.broadcast_to(np.asarray(self.sigma, float), (n,)).copy()
        if np.any(~np.isfinite(self.sigma)) or np.any(self.sigma <= 0):
            raise ValueError("every measurement needs a positive noise standard deviation sigma")
        if self.values is not None:
            self.values = np.asarray(self.values, float).ravel()
            if len(self.values) != n:
                raise ValueError("values and inputs have different lengths")
        if self.observable is None:
            self.observable = np.array([""] * n, dtype=object)
        self.observable = np.asarray(self.observable, dtype=object).ravel()
        if len(self.observable) != n:
            raise ValueError("observable labels and inputs have different lengths")

    @property
    def n(self) -> int:
        if self.inputs:
            lens = {len(v) for v in self.inputs.values()}
            if len(lens) != 1:
                raise ValueError("all input columns must have the same length")
            return lens.pop()
        return len(np.atleast_1d(self.values)) if self.values is not None else len(np.atleast_1d(self.sigma))

    @property
    def has_values(self) -> bool:
        return self.values is not None and bool(np.all(np.isfinite(self.values)))

    def subset(self, idx) -> Observations:
        idx = np.asarray(idx)
        return Observations({k: v[idx] for k, v in self.inputs.items()}, self.sigma[idx],
                            None if self.values is None else self.values[idx], self.observable[idx], dict(self.meta))

    def with_values(self, values: np.ndarray) -> Observations:
        return Observations(dict(self.inputs), self.sigma.copy(), np.asarray(values, float), self.observable.copy(),
                            dict(self.meta))

    def concat(self, other: Observations) -> Observations:
        keys = sorted(set(self.inputs) | set(other.inputs))
        cat = {k: np.concatenate([self.inputs.get(k, np.full(self.n, np.nan)),
                                  other.inputs.get(k, np.full(other.n, np.nan))]) for k in keys}
        vals = None
        if self.values is not None or other.values is not None:
            vals = np.concatenate([self.values if self.values is not None else np.full(self.n, np.nan),
                                   other.values if other.values is not None else np.full(other.n, np.nan)])
        return Observations(cat, np.concatenate([self.sigma, other.sigma]), vals,
                            np.concatenate([self.observable, other.observable]), dict(self.meta))

    def split(self, holdout_fraction: float = 0.25, seed: int = 0, by: str | None = None
              ) -> tuple[Observations, Observations]:
        """(fit, held out). ``by="<input>"`` holds out the rows with the largest values of that input (e.g. the last
        times: an extrapolation test); otherwise a random subset of rows, at least one in each part."""
        n = self.n
        m = max(1, int(round(holdout_fraction * n)))
        if n < 2 or not 0 < holdout_fraction < 1:
            raise ValueError("need at least two rows and 0 < holdout_fraction < 1")
        if by is not None:
            order = np.argsort(self.inputs[by], kind="stable")
            hold = order[n - m:]
        else:
            hold = np.random.default_rng(seed).choice(n, size=m, replace=False)
        mask = np.zeros(n, bool)
        mask[hold] = True
        return self.subset(np.where(~mask)[0]), self.subset(np.where(mask)[0])

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": {k: v.tolist() for k, v in self.inputs.items()}, "sigma": self.sigma.tolist(),
                "values": None if self.values is None else [None if not np.isfinite(v) else float(v)
                                                            for v in self.values],
                "observable": [str(o) for o in self.observable]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Observations:
        vals = d.get("values")
        if vals is not None:
            vals = np.array([np.nan if v is None else v for v in vals], float)
        return cls(dict(d["inputs"]), np.asarray(d["sigma"], float), vals, d.get("observable"))

    @classmethod
    def from_frame(cls, df, *, inputs: Sequence[str], value: str | None = None, sigma: Any = 1.0,
                   observable: str | None = None, default_observable: str = "") -> Observations:
        """From a pandas DataFrame (a CSV / Excel / Parquet / sensor export read with
        ``pinneapple_data.data_health.load_table``). ``sigma`` is a number or a column name; rows with a missing
        input or value are dropped (the count is kept in ``meta["dropped_rows"]``)."""
        import pandas as pd

        cols = list(inputs) + ([value] if value else []) + ([sigma] if isinstance(sigma, str) else []) \
            + ([observable] if observable else [])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns not in the table: {missing}; available: {list(df.columns)}")
        def numeric(col):
            if not pd.api.types.is_numeric_dtype(col):        # text column: accept a decimal comma ("94,1")
                col = col.astype(str).str.strip().str.replace(",", ".", regex=False)
            return pd.to_numeric(col, errors="coerce")

        num = df[[c for c in cols if c != observable]].apply(numeric)
        ok = num.notna().all(axis=1).to_numpy()
        sub, d = num[ok], df[ok]
        sig = sub[sigma].to_numpy(float) if isinstance(sigma, str) else float(sigma)
        obs = d[observable].astype(str).to_numpy(dtype=object) if observable \
            else np.array([default_observable] * int(ok.sum()), dtype=object)
        out = cls({c: sub[c].to_numpy(float) for c in inputs}, sig, sub[value].to_numpy(float) if value else None,
                  obs)
        out.meta["dropped_rows"] = int((~ok).sum())
        return out


# --------------------------------------------------------------------------- models
class ForwardModel:
    """Base class. Subclasses set ``name``, ``inputs`` (input column names), ``outputs`` (observable names),
    ``parameters`` (defaults) and implement ``predict``.

    Optional capabilities, detected by the methods that use them:

    * ``predict_torch(theta: dict[str, Tensor], inputs, observable) -> Tensor``: exact autograd Jacobians/gradients;
    * ``sensitivities(theta, obs, names) -> (prediction, dprediction/dtheta)``: the model's own derivatives (forward
      or adjoint sensitivities of a discretised PDE), in physical units;
    * ``adjoint_gradient(theta, obs, names) -> (misfit, dmisfit/dtheta)``: gradient of
      ``0.5 * sum(((pred - values) / sigma)**2)`` by one adjoint solve;
    * ``fields(theta) -> dict``: reconstructed physical fields (``{"name": {"x": [...], "values": [...], "unit"}}``);
    * ``pinn(obs, parameters, **opts) -> dict``: a physics-informed network for this model.

    ``cost``: ``"cheap"`` (milliseconds: MCMC and profile likelihood run by default), ``"moderate"`` (tens of
    milliseconds: profile likelihood, no MCMC) or ``"expensive"`` (neither, unless asked).
    """
    name: str = "model"
    inputs: Sequence[str] = ()
    outputs: Sequence[str] = ("y",)
    cost: str = "cheap"
    description: str = ""

    def default_parameters(self) -> list[Parameter]:
        return []

    def predict(self, theta: Mapping[str, float], inputs: Mapping[str, np.ndarray],
                observable: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def candidate_measurements(self, obs: Observations, n: int = 25) -> Observations:
        """Possible extra measurements for experimental design: for every observable, ``n`` points spread over the
        range of each input column seen in ``obs`` (models with physical domains override this)."""
        cand = None
        sig = float(np.median(obs.sigma))
        for out in self.outputs:
            cols = {}
            for k in self.inputs:
                v = obs.inputs.get(k)
                if v is None or not np.isfinite(v).any():
                    continue
                lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                cols[k] = np.linspace(lo, hi, n) if hi > lo else np.full(n, lo)
            if not cols:
                continue
            o = Observations(cols, sig, observable=np.array([out] * n, dtype=object))
            cand = o if cand is None else cand.concat(o)
        if cand is None:
            raise ValueError("no input columns to build candidate measurements from")
        return cand

    def describe(self) -> dict[str, Any]:
        caps = [c for c in ("predict_torch", "sensitivities", "adjoint_gradient", "fields", "pinn")
                if callable(getattr(self, c, None))]
        return {"name": self.name, "description": self.description, "inputs": list(self.inputs),
                "outputs": list(self.outputs), "cost": self.cost, "capabilities": caps,
                "parameters": [p.to_dict() for p in self.default_parameters()]}


class CallableModel(ForwardModel):
    """Any Python callable or solver interface: ``fn(theta: dict, inputs: dict[str, array]) -> array`` (optionally a
    third argument ``observable``). Use it to wrap an external solver run (a Runner, a ``ModelRef``) that returns the
    predictions at the measurement points."""

    def __init__(self, fn: Callable, *, inputs: Sequence[str], outputs: Sequence[str] = ("y",),
                 parameters: Sequence[Parameter] = (), name: str = "callable", cost: str = "cheap",
                 description: str = ""):
        self.fn, self.inputs, self.outputs = fn, list(inputs), list(outputs)
        self._params, self.name, self.cost, self.description = list(parameters), name, cost, description
        try:
            self._takes_obs = len(inspect.signature(fn).parameters) >= 3
        except (TypeError, ValueError):
            self._takes_obs = False

    def default_parameters(self) -> list[Parameter]:
        return [Parameter(**asdict(p)) for p in self._params]

    def predict(self, theta, inputs, observable=None):
        out = self.fn(dict(theta), inputs, observable) if self._takes_obs else self.fn(dict(theta), inputs)
        n = len(next(iter(inputs.values()))) if inputs else 1
        return np.broadcast_to(np.asarray(out, float).ravel(), (n,)).copy()


class ExpressionModel(ForwardModel):
    """A model written as text, e.g. ``"T_air + (T0 - T_air) * exp(-h * t / tau_c)"``, one expression per observable.

    Parsed safely (``pinneapple_mcp.expressions``: numbers, names, arithmetic and a fixed list of NumPy functions;
    nothing is passed to ``eval``). Names must be parameters or inputs."""

    def __init__(self, expressions: Mapping[str, str] | str, *, inputs: Sequence[str],
                 parameters: Sequence[Parameter], name: str = "expression", description: str = ""):
        from pinneapple_mcp.expressions import compile_expression

        if isinstance(expressions, str):
            expressions = {"y": expressions}
        self.inputs, self._params = list(inputs), list(parameters)
        self.outputs = list(expressions)
        self.expressions = dict(expressions)
        self._names = [p.name for p in self._params] + self.inputs
        self._fns = {k: compile_expression(e, self._names) for k, e in expressions.items()}
        self.name, self.description = name, description or "; ".join(f"{k} = {e}" for k, e in expressions.items())

    def default_parameters(self) -> list[Parameter]:
        return [Parameter(**asdict(p)) for p in self._params]

    def predict(self, theta, inputs, observable=None):
        n = len(next(iter(inputs.values())))
        X = np.column_stack([np.full(n, float(theta[p.name])) for p in self._params]
                            + [np.asarray(inputs[k], float) for k in self.inputs])
        if len(self.outputs) == 1 or observable is None:
            return self._fns[self.outputs[0]](X)
        out = np.full(n, np.nan)
        for k, fn in self._fns.items():
            m = np.asarray(observable) == k
            if m.any():
                out[m] = fn(X[m])
        if np.isnan(out).any():
            bad = sorted({str(o) for o in np.asarray(observable)[np.isnan(out)]})
            raise ValueError(f"no expression for observable(s) {bad}; defined: {self.outputs}")
        return out


class TabulatedModel(ForwardModel):
    """Simulated results as a model: a table with parameter columns, input columns and one output column (e.g. a
    campaign of solver runs over h and the probe position), interpolated over parameters and inputs.

    Interpolation is a thin-plate RBF on coordinates scaled to [0, 1] (exact at the table rows). Queries outside the
    table's parameter range are clipped to it and counted in ``out_of_range`` (a calibration that ends there is
    reported as extrapolating)."""

    def __init__(self, table, *, parameters: Sequence[str], inputs: Sequence[str], output: str,
                 name: str = "simulated results", smoothing: float = 0.0, observable: str = ""):
        from scipy.interpolate import RBFInterpolator

        cols = list(parameters) + list(inputs)
        data = np.column_stack([np.asarray(table[c], float) for c in cols])
        y = np.asarray(table[output], float)
        ok = np.isfinite(data).all(axis=1) & np.isfinite(y)
        data, y = data[ok], y[ok]
        if len(y) < len(cols) + 2:
            raise ValueError(f"the simulated table needs more rows than columns ({len(y)} rows, {len(cols)} columns)")
        self.param_names, self.inputs, self.outputs = list(parameters), list(inputs), [observable or output]
        self.lo, self.hi = data.min(axis=0), data.max(axis=0)
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self._span = span
        self._rbf = RBFInterpolator((data - self.lo) / span, y, kernel="thin_plate_spline", smoothing=smoothing)
        self.n_rows, self.name, self.out_of_range = len(y), name, 0
        self.description = (f"{len(y)} simulated runs interpolated over {', '.join(cols)} "
                            f"(thin-plate RBF) for {output}")

    def default_parameters(self) -> list[Parameter]:
        out = []
        for i, n in enumerate(self.param_names):
            lo, hi = float(self.lo[i]), float(self.hi[i])
            out.append(Parameter(n, (lo + hi) / 2, lo, hi, log=False))
        return out

    def predict(self, theta, inputs, observable=None):
        n = len(next(iter(inputs.values())))
        P = np.column_stack([np.full(n, float(theta[k])) for k in self.param_names]
                            + [np.asarray(inputs[k], float) for k in self.inputs])
        clipped = np.clip(P, self.lo, self.hi)
        if np.any(clipped[:, :len(self.param_names)] != P[:, :len(self.param_names)]):
            self.out_of_range += 1
        return self._rbf((clipped - self.lo) / self._span)


# --------------------------------------------------------------------------- derivatives
def count_evaluations(model: ForwardModel) -> int:
    return int(getattr(model, "_n_eval", 0))


def _predict(model: ForwardModel, theta, obs: Observations) -> np.ndarray:
    model._n_eval = getattr(model, "_n_eval", 0) + 1
    return np.asarray(model.predict(theta, obs.inputs, obs.observable), float)


def jacobian(model: ForwardModel, params: ParameterSet, z: np.ndarray, obs: Observations, *,
             method: str = "auto", rel_step: float = 1e-4) -> tuple[np.ndarray, np.ndarray, str]:
    """Predictions and their Jacobian with respect to the scaled parameters ``z``: ``(pred (n,), J (n, p), how)``.

    ``method``: ``"auto"`` (the model's own sensitivities, else autograd, else central differences), ``"model"``,
    ``"autograd"`` or ``"fd"``. ``how`` says which was used."""
    z = np.asarray(z, float)
    theta = params.theta(z)
    names = params.names
    # d theta / d z: theta for log parameters, scale for linear ones
    dtdz = np.array([theta[p.name] if p.log else p.scale for p in params.free])
    if method in ("auto", "model") and callable(getattr(model, "sensitivities", None)):
        pred, dpdt = model.sensitivities(theta, obs, names)
        model._n_eval = getattr(model, "_n_eval", 0) + 1
        return np.asarray(pred, float), np.asarray(dpdt, float) * dtdz[None, :], "model sensitivities"
    if method in ("auto", "autograd") and callable(getattr(model, "predict_torch", None)):
        import torch

        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)

        def f(zz):
            th = {k: torch.tensor(v, dtype=torch.float64) for k, v in theta.items()}
            for i, p in enumerate(params.free):
                th[p.name] = torch.exp(zz[i]) if p.log else zz[i] * p.scale
            return model.predict_torch(th, obs.inputs, obs.observable)

        J = torch.autograd.functional.jacobian(f, zt).detach().numpy().reshape(obs.n, len(z))
        model._n_eval = getattr(model, "_n_eval", 0) + 1
        with torch.no_grad():
            pred = f(zt).numpy()
        return pred, J, "autograd"
    if method not in ("auto", "fd"):
        raise ValueError(f"model {model.name} has no '{method}' derivatives")
    pred = _predict(model, theta, obs)
    J = np.zeros((obs.n, len(z)))
    lo, hi = params.bounds()
    for i in range(len(z)):
        h = rel_step * max(1.0, abs(z[i]))
        zp, zm = z.copy(), z.copy()
        zp[i] = min(z[i] + h, hi[i])
        zm[i] = max(z[i] - h, lo[i])
        if zp[i] == zm[i]:
            continue
        J[:, i] = (_predict(model, params.theta(zp), obs) - _predict(model, params.theta(zm), obs)) / (zp[i] - zm[i])
    return pred, J, "central differences"
