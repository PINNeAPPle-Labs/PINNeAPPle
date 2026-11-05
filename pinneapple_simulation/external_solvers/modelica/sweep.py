"""Parameter sweeps of an FMU: the training and test data for surrogates of a Modelica model.

    from pinneapple_simulation.external_solvers.modelica.sweep import fmu_sweep
    ds = fmu_sweep("LadleCooling.fmu", {"k_lin": (1.5, 4.0), "d_lin": (0.15, 0.35)}, n=64, stop_time=14400,
                   output_interval=60, outputs=["T_melt", "T_shell"], seed=0)
    ds.params      # (n, p) sampled values, columns in ds.names
    ds.time        # (T,) common output grid
    ds.outputs     # {"T_melt": (n, T), ...}
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fmi_runner import FMUSimConfig, simulate_fmu

__all__ = ["FMUDataset", "latin_hypercube", "fmu_sweep"]


@dataclass
class FMUDataset:
    """Dataset of an FMU sweep: parameter names, parameters, time, outputs and bounds, with ``split``, ``save`` and
    ``load``.
    """
    names: list[str]
    params: np.ndarray
    time: np.ndarray
    outputs: dict[str, np.ndarray]
    bounds: dict[str, tuple[float, float]]

    def split(self, n_test: int, seed: int = 0) -> tuple[FMUDataset, FMUDataset]:
        """Random split by run (whole trajectories), so the test runs are unseen parameter combinations."""
        idx = np.random.default_rng(seed).permutation(len(self.params))
        te, tr = idx[:n_test], idx[n_test:]
        sub = lambda i: FMUDataset(self.names, self.params[i], self.time,  # noqa: E731
                                   {k: v[i] for k, v in self.outputs.items()}, self.bounds)
        return sub(tr), sub(te)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, names=np.array(self.names), params=self.params, time=self.time,
                            lo=np.array([self.bounds[n][0] for n in self.names]),
                            hi=np.array([self.bounds[n][1] for n in self.names]),
                            **{f"out_{k}": v for k, v in self.outputs.items()})

    @classmethod
    def load(cls, path: str | Path) -> FMUDataset:
        z = np.load(path)
        names = [str(n) for n in z["names"]]
        return cls(names, z["params"], z["time"], {k[4:]: z[k] for k in z.files if k.startswith("out_")},
                   {n: (float(a), float(b)) for n, a, b in zip(names, z["lo"], z["hi"], strict=True)})


def latin_hypercube(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n points in [0, 1]^d, one per row and column stratum."""
    rng = np.random.default_rng(seed)
    u = (np.argsort(rng.random((d, n)), axis=1).T + rng.random((n, d))) / n
    return u


def fmu_sweep(fmu: str | Path, ranges: dict[str, tuple[float, float]], n: int, stop_time: float,
              output_interval: float, outputs: Sequence[str], seed: int = 0,
              fixed: dict[str, float] | None = None, start_time: float = 0.0,
              step_size: float | None = None) -> FMUDataset:
    """Simulate ``n`` Latin-hypercube samples of ``ranges`` (with ``fixed`` overrides) on one output grid.

    ``step_size``: the communication step, then interpolated onto the ``output_interval`` grid (fmpy uses the
    output interval as the communication step of a Co-Simulation FMU). Co-Simulation FMUs from OpenModelica
    integrate with explicit Euler on that step, so keep it well below the fastest time constant of the model
    (default: ``output_interval``)."""
    names = list(ranges)
    lo = np.array([ranges[k][0] for k in names], float)
    hi = np.array([ranges[k][1] for k in names], float)
    P = lo + latin_hypercube(n, len(names), seed) * (hi - lo)
    grid = np.arange(start_time, stop_time + 0.5 * output_interval, output_interval)
    out = {k: np.zeros((n, len(grid))) for k in outputs}
    for i, row in enumerate(P):
        par = dict(fixed or {})
        par.update({k: float(v) for k, v in zip(names, row, strict=True)})
        r = simulate_fmu(fmu, FMUSimConfig(start_time=start_time, stop_time=stop_time,
                                           output_interval=min(step_size or output_interval, output_interval),
                                           parameters=par,
                                           output_variables=list(outputs)))
        for k in outputs:
            out[k][i] = np.interp(grid, r["time"], r[k])
    return FMUDataset(names, P, grid, out, {k: tuple(ranges[k]) for k in names})
