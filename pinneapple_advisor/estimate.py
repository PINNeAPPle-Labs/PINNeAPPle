"""What one simulation and one training run are likely to cost, with the formula written down next to the number.

The cost model is a formula with published-order coefficients, not a benchmark: work is cells x nonlinear
iterations x seconds per cell-iteration, wall time divides the work by a sub-linear factor of the cores (Amdahl),
and the money is core-hours at a reference cloud rate. It is here so the plan can answer the one question that
decides solver versus surrogate -- "is a sweep cheap or expensive?" -- before the user has run anything.

Three ways to make it stop being a guess, in the order of preference:

1. the user's own measurement: ``measured_solve_seconds`` replaces the formula (``confidence="measured"``);
2. :func:`calibrate` -- one run on the user's machine rescales the model's coefficients for the rest of the study;
3. nothing -- then every number is labelled ``confidence="order of magnitude"`` and the plan says so.

Coefficients (documented so they can be argued with, in SI-ish units a script can inspect):

* ``SECONDS_PER_CELL_ITERATION`` -- 0.2 us, a vectorised finite-volume/finite-difference cell update on one core;
* ``PER_FAMILY_WORK`` -- how much more expensive one iteration of that physics is than that base cell update
  (a coupled Navier-Stokes solve is several times a scalar diffusion sweep; a lattice-Boltzmann step is cheaper);
* ``ITERATIONS`` -- steady linear, multiplied for a nonlinear continuation, a turbulence closure and a time loop;
* ``DEFAULT_CELLS`` -- a default engineering resolution per dimension, used only when the user gives no mesh size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import Estimate

# One cell update on one core, in seconds. 0.2 us: ~5 million cell-updates per second, the order of a vectorised
# numpy finite-difference/finite-volume sweep.
SECONDS_PER_CELL_ITERATION = 2.0e-7

# How much one iteration of this physics costs relative to that base cell update (order-of-magnitude ratios: a
# coupled velocity-pressure solve assembles and inverts more than a scalar update, an explicit lattice step less).
PER_FAMILY_WORK: dict[str, float] = {
    "laplace": 1.0, "poisson": 1.0, "heat_equation": 1.0, "helmholtz": 1.5, "wave": 1.5, "burgers": 2.0,
    "reaction_diffusion": 2.0, "fisher_kpp": 1.5, "darcy": 1.0, "sir": 0.2, "black_scholes": 0.2,
    "inviscid_burgers": 2.0, "linear_elasticity": 3.0, "navier_stokes": 6.0, "custom": 2.0,
}

# Iterations to a steady solution, before the nonlinear and turbulence multipliers.
ITERATIONS = 400
NONLINEAR_FACTOR = 8.0
TURBULENCE_FACTOR = 3.0
TRANSIENT_FACTOR = 60.0        # ~100 time steps of comparable work, with a warm start

# A default engineering resolution, used only when the user has given no mesh size and no measurement.
DEFAULT_CELLS: dict[int, int] = {1: 2_000, 2: 20_000, 3: 500_000}

# A reference on-demand cloud rate, off by default: a local machine costs money, but not per core-hour.
REFERENCE_USD_PER_CORE_HOUR = 0.0
CLOUD_USD_PER_CORE_HOUR = 0.05

# Training: seconds per sample per epoch, and iterations/epochs for a first fit.
SECONDS_PER_SAMPLE_EPOCH: dict[str, float] = {
    "pinn": 0.004, "xpinn": 0.006, "inverse_pinn": 0.004, "pino": 0.008,
    "fno": 0.005, "deeponet": 0.005, "mesh_graph_net": 0.012, "hybrid_ml": 0.005, "none": 0.0,
}
DEFAULT_EPOCHS = 200
PINN_ITERATIONS = 5_000
PINN_SECONDS_PER_ITERATION = 2.0e-3
GPU_SPEEDUP: dict[str, float] = {"pinn": 5.0, "xpinn": 5.0, "inverse_pinn": 5.0, "pino": 8.0, "fno": 10.0,
                                 "deeponet": 10.0, "mesh_graph_net": 6.0, "hybrid_ml": 5.0, "none": 1.0}


@dataclass
class CostModel:
    """The coefficients above, as an object a caller can replace with its own calibrated ones."""
    seconds_per_cell_iteration: float = SECONDS_PER_CELL_ITERATION
    per_family_work: dict[str, float] = field(default_factory=lambda: dict(PER_FAMILY_WORK))
    iterations: int = ITERATIONS
    nonlinear_factor: float = NONLINEAR_FACTOR
    turbulence_factor: float = TURBULENCE_FACTOR
    transient_factor: float = TRANSIENT_FACTOR
    default_cells: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_CELLS))
    usd_per_core_hour: float = REFERENCE_USD_PER_CORE_HOUR

    # ---- the two estimates --------------------------------------------------------------------------------------
    def solve(self, *, family: str = "custom", dim: int = 2, time_dependent: bool = False, nonlinear: bool = False,
              turbulent: bool = False, cells: int | None = None, cores: int = 1, n_evaluations: int = 1,
              measured_solve_seconds: float | None = None, usd_per_core_hour: float | None = None,
              budget_hours: float | None = None) -> Estimate:
        """Time and money for ``n_evaluations`` solves of one case, with the formula in ``basis``."""
        cores = max(int(cores or 1), 1)
        if measured_solve_seconds is not None and measured_solve_seconds > 0:
            per_solve = float(measured_solve_seconds)
            basis = (f"measured: {per_solve:.4g} s per solve given by the user, x {n_evaluations} evaluations "
                     f"(wall time on {cores} core(s); a local run adds no equipment cost)")
            confidence = "measured"
            assumptions = ["the measured solve is representative of the cases the study will run"]
            cells_used = cells
        else:
            cells_used = int(cells) if cells else self.default_cells.get(int(dim), self.default_cells[2])
            iters = self.iterations
            if nonlinear:
                iters *= self.nonlinear_factor
            if turbulent:
                iters *= self.turbulence_factor
            if time_dependent:
                iters *= self.transient_factor
            work = (self.per_family_work.get(family, self.per_family_work["custom"]) * cells_used * iters
                    * self.seconds_per_cell_iteration)
            speedup = cores ** 0.75
            per_solve = work / speedup
            basis = (f"estimated: {self.per_family_work.get(family, self.per_family_work['custom']):g} x "
                     f"{cells_used:,} cells x {iters:,.0f} iterations x {self.seconds_per_cell_iteration:g} s "
                     f"= {work:,.1f} core-s, / {speedup:.2f} (Amdahl, {cores} core(s))")
            confidence = "order of magnitude"
            resolution = "given" if cells else f"default engineering resolution for {dim}-D; pass approx_cells to replace it"
            assumptions = [
                f"mesh of about {cells_used:,} cells ({resolution})",
                f"{iters:,.0f} iterations ({self.iterations} steady linear"
                + (f" x {self.nonlinear_factor:g} nonlinear" if nonlinear else "")
                + (f" x {self.turbulence_factor:g} turbulence closure" if turbulent else "")
                + (f" x {self.transient_factor:g} time loop" if time_dependent else "") + ")",
                "one iteration = 0.2 us per cell (vectorised cell update, one core)",
                f"wall time from {cores} core(s) with a sub-linear (Amdahl) speedup",
            ]
        total = per_solve * max(int(n_evaluations), 1)
        rate = self.usd_per_core_hour if usd_per_core_hour is None else float(usd_per_core_hour)
        cost = (total * cores / 3600.0) * rate if rate else None
        est = Estimate(seconds_per_evaluation=per_solve, n_evaluations=max(int(n_evaluations), 1),
                       total_seconds=total, cost_usd=cost, cells=cells_used, basis=basis, confidence=confidence,
                       assumptions=assumptions)
        if budget_hours is not None and budget_hours > 0:
            est.assumptions.append(f"budget: {budget_hours:g} h -> "
                                   f"{'fits' if total / 3600.0 <= budget_hours else 'DOES NOT FIT'}")
        return est

    def training(self, architecture: str = "fno", *, n_samples: int = 100, epochs: int | None = None,
                 fields_out: int = 1, gpu: bool = False, measured_train_seconds: float | None = None) -> Estimate:
        """Time and money for one training run of ``architecture`` on ``n_samples`` simulations."""
        arch = architecture if architecture in SECONDS_PER_SAMPLE_EPOCH else "fno"
        if measured_train_seconds is not None and measured_train_seconds > 0:
            return Estimate(seconds_per_evaluation=float(measured_train_seconds), n_evaluations=1,
                            total_seconds=float(measured_train_seconds), cells=None,
                            basis="measured: training time given by the user", confidence="measured",
                            assumptions=["the measured run used the same architecture and dataset size"])
        speed = GPU_SPEEDUP.get(arch, 1.0) if gpu else 1.0
        if arch in ("pinn", "xpinn", "inverse_pinn"):
            # physics-only training has no dataset: iterations over collocation points, not epochs
            work = PINN_ITERATIONS * PINN_SECONDS_PER_ITERATION
            basis = (f"estimated: {PINN_ITERATIONS:,} training iterations x {PINN_SECONDS_PER_ITERATION:g} s"
                     f"{' / ' + format(speed, 'g') + ' (GPU)' if gpu else ''}")
            assumptions = ["a PINN trains on collocation points, so the dataset size does not enter this cost",
                           "5 000 iterations is the first fit this library's presets typically need"]
            n_samples_used = 0
        else:
            epochs = int(epochs or DEFAULT_EPOCHS)
            per = SECONDS_PER_SAMPLE_EPOCH.get(arch, 0.005) * max(fields_out, 1)
            work = n_samples * epochs * per
            basis = (f"estimated: {n_samples} samples x {epochs} epochs x {per:g} s per sample-epoch"
                     f"{' / ' + format(speed, 'g') + ' (GPU)' if gpu else ''}")
            assumptions = [f"{epochs} epochs (a first fit; the plan asks for early stopping against a held-out set)",
                           f"one forward+backward per sample per epoch, {per * 1000:.1f} ms at this size"]
            n_samples_used = n_samples
        total = work / speed
        est = Estimate(seconds_per_evaluation=total, n_evaluations=1, total_seconds=total,
                       cost_usd=(total * (4 if gpu else 1) / 3600.0) * self.usd_per_core_hour
                       if self.usd_per_core_hour else None, cells=None, basis=basis,
                       confidence="order of magnitude", assumptions=assumptions)
        est.assumptions.append(f"{n_samples_used or 'no'} samples, {architecture} family")
        return est

    def calibrate(self, measured_seconds: float, *, cells: int, family: str = "custom", dim: int = 2,
                  iterations: int | None = None) -> CostModel:
        """Rescale the per-cell coefficient so the model reproduces one measured solve, keeping the shape of the
        formula (the study's own case becomes the unit of cost)."""
        iters = int(iterations or self.iterations)
        per_iteration = measured_seconds / max(
            cells * iters * max(self.per_family_work.get(family, 1.0), 1e-9), 1e-12)
        return CostModel(seconds_per_cell_iteration=max(per_iteration, 1e-12),
                         per_family_work=dict(self.per_family_work), iterations=self.iterations,
                         nonlinear_factor=self.nonlinear_factor, turbulence_factor=self.turbulence_factor,
                         transient_factor=self.transient_factor, default_cells=dict(self.default_cells),
                         usd_per_core_hour=self.usd_per_core_hour)


DEFAULT = CostModel()


def estimate_solve(**kwargs: Any) -> Estimate:
    """``DEFAULT.solve(...)``: see :meth:`CostModel.solve`."""
    return DEFAULT.solve(**kwargs)


def estimate_training(**kwargs: Any) -> Estimate:
    """``DEFAULT.training(...)``: see :meth:`CostModel.training`."""
    return DEFAULT.training(**kwargs)


__all__ = ["CostModel", "DEFAULT", "estimate_solve", "estimate_training", "SECONDS_PER_CELL_ITERATION",
           "PER_FAMILY_WORK", "DEFAULT_CELLS", "CLOUD_USD_PER_CORE_HOUR"]
