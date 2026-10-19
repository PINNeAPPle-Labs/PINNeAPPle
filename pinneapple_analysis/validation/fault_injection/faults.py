"""The fault library: controlled defects of a model's output or of a dataset, each with a magnitude.

A model fault turns a correct :class:`~.problems.Case` into what a defective model would return for the same
question, and gives the correct field it should be compared with (a fault that changes the inputs, such as
out-of-range use, changes the correct answer too). A data fault turns a correct :class:`~.problems.Dataset` into
what a defective datapipe would hand to training.

Model faults (#697): ``bias``, ``scale``, ``noise`` (high frequency), ``smoothing`` (spectral bias),
``wrong_boundary``, ``source_sign_flip``, ``out_of_range`` (a model used outside its input range), ``stale_model``
(trained on an older physics configuration, as in the rover zoo v1.0 case).

Data faults: ``coefficient_drift`` (a discretisation coefficient off by a constant factor, modelled on the Darcy2D
datapipe report NVIDIA/physicsnemo#2036), ``duplicates`` and ``low_diversity`` (NVIDIA/physicsnemo#2035),
``leakage`` (test samples copied from training), ``unit_mixup`` (outputs in mm instead of m), ``swapped_channels``.

Each fault has a ladder of magnitudes and a ``documented`` magnitude: the one the detection matrix promises to catch
with at least one check (the acceptance test of #697).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .problems import BenchmarkProblem, BurgersProblem, Case, Dataset

__all__ = ["Fault", "MODEL_FAULTS", "DATA_FAULTS", "model_faults", "data_faults", "get_fault"]


@dataclass(frozen=True)
class Fault:
    """``apply(problem, case_or_dataset, magnitude, rng)``. Model faults return ``(faulty_case, correct_field)``;
    data faults return ``(faulty_dataset, worst relative error of a corrupted sample)`` (0 when the samples are
    right but the dataset is not: duplicates, leakage)."""
    name: str
    target: str                       # "model" | "data"
    apply: Callable
    magnitudes: tuple[float, ...]
    documented: float | dict[str, float]
    unit: str
    description: str

    def documented_for(self, problem: str) -> float:
        """The documented magnitude on ``problem`` (a fault can be harder to see on one problem than another)."""
        d = self.documented
        return float(d.get(problem, max(d.values()))) if isinstance(d, dict) else float(d)


_MEMO: dict[tuple, tuple[Case, np.ndarray]] = {}


def _memo(c: Case, key: tuple, fn: Callable[[], np.ndarray]) -> np.ndarray:
    # datasets share their correct samples: a fault that re-solves a sample does it once per (sample, magnitude)
    k = (id(c), *key)
    hit = _MEMO.get(k)
    if hit is None or hit[0] is not c:
        if len(_MEMO) > 4096:
            _MEMO.clear()
        hit = _MEMO[k] = (c, fn())
    return hit[1]


def _amp(case: Case) -> float:
    return float(np.abs(case.field).max())


# ====================================================================== model faults
def _bias(p, c, m, rng):
    return c.with_field(c.field + m * _amp(c)), c.field


def _scale(p, c, m, rng):
    return c.with_field(c.field * (1 + m)), c.field


def _noise(p, c, m, rng):
    return c.with_field(c.field + m * float(c.field.std()) * rng.standard_normal(c.field.shape)), c.field


def _smoothing(p, c, m, rng):
    from scipy.ndimage import gaussian_filter
    if isinstance(p, BurgersProblem):                  # along x, periodic, every snapshot
        out = gaussian_filter(c.field, sigma=(0.0, m), mode="wrap")
    else:
        out = gaussian_filter(c.field, sigma=m, mode="nearest")
    return c.with_field(out), c.field


def _wrong_boundary(p, c, m, rng):
    co = p.coords(c)
    if isinstance(p, BurgersProblem):                  # a ramp that grows in time: periodicity lost after t = 0
        t, x = np.meshgrid(co["t"], co["x"], indexing="ij")
        ramp = (x - 0.5) * t / co["t"][-1]
        return c.with_field(c.field + m * float(np.abs(c.inputs["u0"]).max()) * ramp), c.field
    x, _ = np.meshgrid(co["x"], co["y"], indexing="ij")
    return c.with_field(c.field + m * _amp(c) * (1.0 - x)), c.field    # u = m max|u| on x = 0 instead of 0


def _source_sign_flip(p, c, m, rng):
    if isinstance(p, BurgersProblem):                  # the advection term flips sign for the last fraction m of time
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d
        nt = c.field.shape[0]
        k = int(round((1 - m) * (nt - 1)))
        out = c.field.copy()
        if k < nt - 1:
            dts = p.options["t_end"] / (nt - 1)
            _, v = burgers_1d(-c.field[k], c.params["nu"], dts * (nt - 1 - k), dt=p.options["dt"], n_save=nt - 1 - k)
            out[k:] = -v[0]
        return c.with_field(out), c.field
    co = p.coords(c)                                   # Darcy: f -> -f on the strip x < m
    x, _ = np.meshgrid(co["x"], co["y"], indexing="ij")
    f = np.where(x < m, -1.0, 1.0) * c.params["f"]
    return c.with_field(p.resolve(c, f=f)), c.field


def _out_of_range(p, c, m, rng):
    if isinstance(p, BurgersProblem):                  # inputs (1 + m) larger; the model scales its answer linearly
        u0 = c.inputs["u0"] * (1 + m)
        faulty = replace(c, inputs={"u0": u0}, field=c.field * (1 + m))
        return faulty, p.resolve(c, u0=u0)
    a = c.inputs["a"]                                  # Darcy: a contrast beyond training; the model saturates
    hi = float(a.max())
    a_ood = np.where(a >= hi, hi * (1 + m), a)
    faulty = replace(c, inputs={"a": a_ood}, field=c.field.copy())
    return faulty, p.resolve(c, a=a_ood)


def _stale_model(p, c, m, rng):
    if isinstance(p, BurgersProblem):                  # trained when the viscosity was (1 + m) nu
        return c.with_field(p.resolve(c, nu=c.params["nu"] * (1 + m))), c.field
    a = c.inputs["a"]                                  # Darcy: trained when the low permeability was (1 + m) low
    lo = float(a.min())
    return c.with_field(p.resolve(c, a=np.where(a <= lo, lo * (1 + m), a))), c.field


MODEL_FAULTS = [
    Fault("bias", "model", _bias, (0.005, 0.01, 0.02, 0.05, 0.1, 0.2), 0.05, "fraction of max |u|",
          "output + m max|u|"),
    Fault("scale", "model", _scale, (0.01, 0.02, 0.05, 0.1, 0.2, 0.5), 0.2, "relative",
          "output x (1 + m)"),
    Fault("noise", "model", _noise, (0.001, 0.003, 0.01, 0.03, 0.1, 0.3), 0.1, "fraction of std(u)",
          "white noise of standard deviation m std(u)"),
    Fault("smoothing", "model", _smoothing, (0.25, 0.5, 1.0, 2.0, 4.0), {"darcy_2d": 2.0, "burgers_1d": 4.0},
          "cells (Gaussian sigma)",
          "Gaussian smoothing over m cells (spectral bias)"),
    Fault("wrong_boundary", "model", _wrong_boundary, (0.01, 0.02, 0.05, 0.1, 0.2, 0.5), 0.1, "fraction of max |u|",
          "Darcy: u = m max|u| on x = 0 (linear in x); Burgers: a ramp of m max|u0| that breaks periodicity"),
    Fault("source_sign_flip", "model", _source_sign_flip, (0.05, 0.1, 0.2, 0.5, 1.0), 0.2, "fraction",
          "Darcy: f -> -f for x < m; Burgers: u u_x -> -u u_x for the last fraction m of the time span"),
    Fault("out_of_range", "model", _out_of_range, (0.1, 0.3, 1.0, 2.0, 4.0), 1.0, "relative excess",
          "inputs (1 + m) beyond the training range (Darcy: high permeability; Burgers: amplitude); the model "
          "saturates (Darcy) or scales linearly (Burgers)"),
    Fault("stale_model", "model", _stale_model, (0.05, 0.1, 0.2, 0.5, 1.0, 2.0), 0.5, "relative",
          "solves an older configuration (Darcy: low permeability x (1 + m); Burgers: viscosity x (1 + m))"),
]


# ====================================================================== data faults
def _pick(n: int, frac: float, rng) -> np.ndarray:
    k = min(n, max(1, int(round(frac * n))))
    return rng.choice(n, size=k, replace=False)


def _coefficient_drift(p, ds, m, rng):
    def bad(c):
        if isinstance(p, BurgersProblem):
            return c.with_field(_memo(c, ("drift", p.name, m), lambda: p.resolve(c, nu=c.params["nu"] * (1 + m))))
        return c.with_field(_memo(c, ("drift", p.name, m), lambda: p.resolve(c, axis_weights=(1.0, 1.0 + m))))
    train, test = [bad(c) for c in ds.train], [bad(c) for c in ds.test]
    err = max(p.true_error(b, c) for b, c in zip([*train, *test], ds.cases, strict=True))
    return Dataset(train, test), err


def _duplicates(p, ds, m, rng, jitter: float = 0.0):
    train = list(ds.train)
    n = len(train)
    for i in _pick(n, m, rng):
        j = int(rng.choice([k for k in range(n) if k != i]))
        src = train[j]
        inputs = {k: v * (1 + jitter * rng.standard_normal(v.shape)) for k, v in src.inputs.items()}
        train[i] = replace(src, inputs=inputs, field=src.field.copy(), meta={**src.meta, "copy_of": j})
    return Dataset(train, list(ds.test)), 0.0


def _low_diversity(p, ds, m, rng):
    return _duplicates(p, ds, m, rng, jitter=1e-5)


def _leakage(p, ds, m, rng):
    test = list(ds.test)
    for i in _pick(len(test), m, rng):
        test[i] = ds.train[int(rng.integers(len(ds.train)))]
    return Dataset(list(ds.train), test), 0.0


def _unit_mixup(p, ds, m, rng):
    cases = ds.cases
    idx = set(int(i) for i in _pick(len(cases), m, rng))
    out = [c.with_field(c.field * 1000.0) if i in idx else c for i, c in enumerate(cases)]
    return Dataset(out[:len(ds.train)], out[len(ds.train):]), 999.0


def _swapped_channels(p, ds, m, rng):
    cases = ds.cases
    idx = set(int(i) for i in _pick(len(cases), m, rng))

    def swap(c):
        if isinstance(p, BurgersProblem):              # snapshots stored in reverse time order
            return c.with_field(c.field[::-1].copy())
        return replace(c, inputs={"a": c.field.copy()}, field=c.inputs["a"].copy())
    out = [swap(c) if i in idx else c for i, c in enumerate(cases)]
    err = max((p.true_error(out[i], cases[i]) for i in idx), default=0.0)
    return Dataset(out[:len(ds.train)], out[len(ds.train):]), err


DATA_FAULTS = [
    Fault("coefficient_drift", "data", _coefficient_drift, (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
          {"darcy_2d": 0.1, "burgers_1d": 0.5}, "relative",
          "every sample solved with a coefficient off by (1 + m): Darcy y-face transmissibility, Burgers viscosity"),
    Fault("duplicates", "data", _duplicates, (0.05, 0.1, 0.25, 0.5), 0.1, "fraction of training samples",
          "training samples replaced by exact copies of others"),
    Fault("low_diversity", "data", _low_diversity, (0.05, 0.1, 0.25, 0.5), 0.1, "fraction of training samples",
          "training samples replaced by near-copies of others (1e-5 relative jitter)"),
    Fault("leakage", "data", _leakage, (0.05, 0.1, 0.25, 0.5), 0.1, "fraction of test samples",
          "test samples copied from the training split"),
    Fault("unit_mixup", "data", _unit_mixup, (0.05, 0.1, 0.25, 1.0), 0.05, "fraction of samples",
          "outputs stored in mm instead of m (x 1000)"),
    Fault("swapped_channels", "data", _swapped_channels, (0.05, 0.1, 0.25, 1.0), 0.05, "fraction of samples",
          "Darcy: input and output channels swapped; Burgers: snapshots in reverse time order"),
]


def model_faults(problem: BenchmarkProblem | None = None) -> list[Fault]:
    return list(MODEL_FAULTS)


def data_faults(problem: BenchmarkProblem | None = None) -> list[Fault]:
    return list(DATA_FAULTS)


def get_fault(name: str) -> Fault:
    for f in (*MODEL_FAULTS, *DATA_FAULTS):
        if f.name == name:
            return f
    raise KeyError(f"unknown fault {name!r}")

