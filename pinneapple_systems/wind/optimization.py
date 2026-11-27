"""Farm-level optimisation on top of the wake model, and the surrogate that accelerates it.

Two things happen here, and they are reported separately because they have very
different epistemic status:

**Curtailment / wake-steering study** (:func:`optimize_curtailment`)
    For one wind speed and direction, find the per-turbine thrust multiplier
    that maximises total farm power. Derating an upwind machine costs its own
    production and helps everything behind it; whether the trade pays is a
    property of the spacing, the thrust curve and the wind speed. The result is
    a **model estimate** built on an assumption -- that reducing :math:`C_T`
    through a multiplier is a faithful stand-in for yaw misalignment or derating
    -- which is stated here and in every output. Wake steering in reality also
    curves the wake sideways, which none of these models represent.

**Surrogate** (:class:`FarmSurrogate`)
    A small network fitted to the analytical farm model. Its value is speed (a
    farm-polar sweep costs one forward pass instead of ~10^4 wake evaluations),
    not fidelity: it is trained on the model it approximates, so it can never be
    more accurate than the model, only faster. Its own error against that model
    is measured on a held-out grid and reported.

Neither is a substitute for measurement. A curtailment recommendation that has
not been tested on the turbines is a hypothesis.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .layout import WindFarmLayout
from .wake import WakeConfig, farm_power

__all__ = [
    "CurtailmentResult",
    "optimize_curtailment",
    "curtailment_study",
    "annual_energy_production",
    "FarmSurrogate",
]


@dataclass(frozen=True)
class CurtailmentResult:
    """Outcome of one curtailment optimisation, including the do-nothing case."""

    wind_speed: float
    direction_deg: float
    baseline_power_kw: float
    optimal_power_kw: float
    thrust_scale: np.ndarray
    n_curtailed: int
    config: WakeConfig

    @property
    def gain_kw(self) -> float:
        return self.optimal_power_kw - self.baseline_power_kw

    @property
    def gain_fraction(self) -> float:
        return self.gain_kw / self.baseline_power_kw if self.baseline_power_kw > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "wind_speed_ms": self.wind_speed,
            "direction_deg": self.direction_deg,
            "baseline_power_kw": self.baseline_power_kw,
            "optimal_power_kw": self.optimal_power_kw,
            "gain_kw": self.gain_kw,
            "gain_fraction": self.gain_fraction,
            "thrust_scale": self.thrust_scale.tolist(),
            "n_curtailed": self.n_curtailed,
            "model": self.config.label,
            "caveat": (
                "Model estimate. Reducing C_T by a multiplier stands in for yaw misalignment or "
                "derating; the wake is not deflected and no measured steering response exists here."
            ),
        }


def optimize_curtailment(
    layout: WindFarmLayout,
    wind_speed: float,
    direction_deg: float,
    power_curves,
    config: WakeConfig | None = None,
    *,
    lower: float = 0.2,
    baseline_threshold: float = 1e-4,
) -> CurtailmentResult:
    """Maximise farm power over per-turbine thrust multipliers in ``[lower, 1]``.

    The optimisation is a bounded L-BFGS-B over ``N`` variables on a smooth
    objective; with 14 turbines and an analytical farm model each evaluation is
    sub-millisecond, so no gradient is needed.

    Only gains above ``baseline_threshold`` (fraction of baseline power) are
    kept: an optimiser run on a flat objective will otherwise report a 0.001 %
    "improvement" that is numerical noise.
    """
    from scipy.optimize import minimize

    cfg = config or WakeConfig()
    n = len(layout)
    baseline = farm_power(layout, wind_speed, direction_deg, power_curves, cfg)

    def objective(scale):
        res = farm_power(layout, wind_speed, direction_deg, power_curves, cfg, thrust_scale=scale)
        return -res.total_power_kw

    best = minimize(
        objective,
        x0=np.ones(n),
        method="L-BFGS-B",
        bounds=[(lower, 1.0)] * n,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    scale = np.clip(best.x, lower, 1.0)
    optimal = farm_power(layout, wind_speed, direction_deg, power_curves, cfg, thrust_scale=scale)
    if optimal.total_power_kw <= baseline.total_power_kw * (1.0 + baseline_threshold):
        scale = np.ones(n)
        optimal = baseline
    return CurtailmentResult(
        wind_speed=float(wind_speed),
        direction_deg=float(direction_deg),
        baseline_power_kw=baseline.total_power_kw,
        optimal_power_kw=optimal.total_power_kw,
        thrust_scale=scale,
        n_curtailed=int((scale < 0.999).sum()),
        config=cfg,
    )


def curtailment_study(
    layout: WindFarmLayout,
    directions: Sequence[float],
    wind_speeds: Sequence[float],
    power_curves,
    config: WakeConfig | None = None,
) -> dict:
    """Run the optimiser over a direction x speed grid and summarise the gain."""
    rows = []
    for d in directions:
        for u in wind_speeds:
            res = optimize_curtailment(layout, float(u), float(d), power_curves, config)
            rows.append({
                "direction_deg": float(d),
                "wind_speed_ms": float(u),
                "baseline_power_kw": res.baseline_power_kw,
                "gain_kw": res.gain_kw,
                "gain_fraction": res.gain_fraction,
                "n_curtailed": res.n_curtailed,
            })
    gains = np.array([r["gain_fraction"] for r in rows])
    return {
        "rows": rows,
        "max_gain_fraction": float(gains.max()) if gains.size else 0.0,
        "mean_gain_fraction": float(gains.mean()) if gains.size else 0.0,
        "cells_with_gain": int((gains > 1e-4).sum()),
        "model": (config or WakeConfig()).label,
    }


def annual_energy_production(
    layout: WindFarmLayout,
    power_curves,
    wind_rose: dict[float, float],
    speed_histogram: dict[float, float],
    config: WakeConfig | None = None,
) -> dict:
    """AEP from a wind rose and a speed histogram, with and without wakes.

    ``wind_rose`` maps direction [deg] -> probability of occurrence;
    ``speed_histogram`` maps speed [m/s] -> probability. The two are assumed
    independent (no direction-dependent speed distribution), which is the usual
    first-order AEP assumption and is stated because it is one.
    """
    cfg = config or WakeConfig()
    rose_total = sum(wind_rose.values())
    hist_total = sum(speed_histogram.values())
    if rose_total <= 0 or hist_total <= 0:
        raise ValueError("wind rose and speed histogram must have positive total probability")
    with_wake = 0.0
    without_wake = 0.0
    for direction, p_dir in wind_rose.items():
        for speed, p_speed in speed_histogram.items():
            weight = (p_dir / rose_total) * (p_speed / hist_total)
            res = farm_power(layout, float(speed), float(direction), power_curves, cfg)
            with_wake += weight * res.total_power_kw
            without_wake += weight * res.total_power_no_wake_kw
    hours = 8766.0
    return {
        "aep_gwh": with_wake * hours / 1e6,
        "aep_no_wake_gwh": without_wake * hours / 1e6,
        "wake_loss_fraction": 1.0 - with_wake / without_wake if without_wake > 0 else 0.0,
        "capacity_factor": with_wake * hours / (layout.installed_capacity_mw * 1000.0 * hours),
        "n_directions": len(wind_rose),
        "n_speeds": len(speed_histogram),
        "model": cfg.label,
    }


class FarmSurrogate:
    """A small MLP mapping (direction, wind speed) to farm efficiency.

    Trained on the analytical model, so it is a *speed* device: it cannot beat
    the model it learned. :meth:`validate` reports its error against that model
    on a grid the fit never saw -- the number that decides whether it may be
    used in place of a sweep.
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        epochs: int = 600,
        learning_rate: float = 5e-3,
        seed: int = 0,
        name: str = "farm-efficiency surrogate (MLP)",
    ) -> None:
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.name = name
        self._net = None
        self.diagnostics: dict = {}

    @staticmethod
    def _features(direction_deg, wind_speed):
        d = np.radians(np.asarray(direction_deg, dtype=float))
        u = np.asarray(wind_speed, dtype=float)
        return np.column_stack([
            np.sin(d), np.cos(d), np.sin(2 * d), np.cos(2 * d), u, u**2 / 20.0, u**3 / 400.0,
        ]).astype(np.float32)

    def fit(self, directions, wind_speeds, efficiencies) -> FarmSurrogate:
        import torch

        torch.manual_seed(self.seed)
        X = torch.tensor(self._features(directions, wind_speeds))
        y = torch.tensor(np.asarray(efficiencies, dtype=np.float32)).view(-1, 1)
        self._net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 1),
            torch.nn.Sigmoid(),  # efficiency is a fraction in (0, 1]
        )
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            loss = loss_fn(self._net(X), y)
            loss.backward()
            optimiser.step()
        self.diagnostics["train_mse"] = float(loss.detach())
        return self

    def predict(self, directions, wind_speeds) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("surrogate is not fitted")
        import torch

        with torch.no_grad():
            out = self._net(torch.tensor(self._features(directions, wind_speeds)))
        return out.numpy().ravel().astype(float)

    def validate(self, directions, wind_speeds, efficiencies) -> dict:
        """Error against the analytical model on an unseen grid."""
        truth = np.asarray(efficiencies, dtype=float)
        pred = self.predict(directions, wind_speeds)
        err = pred - truth
        result = {
            "n": int(truth.size),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs": float(np.max(np.abs(err))),
            "r2": float(1.0 - (err**2).sum() / ((truth - truth.mean()) ** 2).sum())
            if truth.size > 1 else None,
        }
        self.diagnostics["validation"] = result
        return result


def training_grid(
    layout: WindFarmLayout,
    power_curves,
    *,
    n_directions: int = 36,
    speeds: Sequence[float] = (5.0, 7.0, 9.0, 11.0, 13.0),
    config: WakeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Efficiency on a (direction, speed) grid -- the teacher signal for a surrogate."""
    directions = np.linspace(0.0, 360.0, n_directions, endpoint=False)
    dirs, spds, effs = [], [], []
    for d in directions:
        for u in speeds:
            res = farm_power(layout, float(u), float(d), power_curves, config)
            dirs.append(float(d))
            spds.append(float(u))
            effs.append(res.efficiency)
    return np.array(dirs), np.array(spds), np.array(effs)


def fit_surrogate(
    layout: WindFarmLayout,
    power_curves,
    *,
    speeds: Sequence[float] = (5.0, 7.0, 9.0, 11.0, 13.0),
    config: WakeConfig | None = None,
    hidden: int = 64,
    epochs: int = 600,
    seed: int = 0,
) -> FarmSurrogate:
    """Fit on a coarse grid, validate on a finer, phase-shifted one."""
    d_train, s_train, e_train = training_grid(
        layout, power_curves, n_directions=36, speeds=speeds, config=config,
    )
    surrogate = FarmSurrogate(hidden=hidden, epochs=epochs, seed=seed).fit(d_train, s_train, e_train)
    d_val, s_val, e_val = training_grid(
        layout, power_curves, n_directions=54, speeds=tuple(np.asarray(speeds) + 0.7), config=config,
    )
    surrogate.validate(d_val, s_val, e_val)
    return surrogate


# Re-exported for the example: the callable type a farm model accepts.
PowerCurves = Callable[[np.ndarray], np.ndarray]
