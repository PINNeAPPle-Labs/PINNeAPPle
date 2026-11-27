"""Turbine power models: the physics-informed one, and the baselines it is judged against.

Everything here implements the same tiny protocol -- ``fit(X, y)`` and
``predict(X)`` over a feature matrix with named columns -- so a benchmark table
can put all of them side by side without special cases.

The physics-informed model
--------------------------
:class:`PhysicsInformedPowerModel` is a **monotone hinge network**: a piecewise
linear curve whose increments are constrained non-negative by a softplus
parameterisation, saturated below rated power by a smooth cap, and gated to zero
outside the operating window. Five physical statements are therefore satisfied
*by construction*, not as a penalty that training may trade away:

1. power is non-negative;
2. power never decreases with wind speed inside the operating window;
3. power never exceeds the rated (nameplate) value;
4. power is zero below cut-in and above cut-out;
5. the curve starts at zero at zero wind speed.

Two further conditions are enforced as penalties during training:

6. in region 2 the curve must follow the cubic law :math:`P = \\alpha u^3` --
   momentum theory fixes the *shape*, leaving only the scale :math:`\\alpha` free;
7. that scale may not exceed the Betz bound
   :math:`\\alpha \\le \\tfrac12 \\rho_0 A \\tfrac{16}{27} / P_{rated}`.

What that buys and what it costs is measured in the example: the constrained
model has less freedom, so it does not necessarily beat a boosted tree on
in-window RMSE. Its claim is different -- it stays physical where the data is
thin (near cut-in, near cut-out, at rated), and it can be extrapolated to a
density it was not trained at through the IEC density correction.

The conventional baseline
--------------------------
:class:`GradientBoostedPowerModel` is a histogram gradient-boosted regressor
over meteorological and turbine-state features. It is unconstrained: nothing
stops it from predicting 3 MW above rated or a falling curve near cut-in. That
is the point of the comparison.

Both are trained on the same window, on the same target, with the same features
that are available at prediction time -- no lagged power, no future information.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .power_curve import RHO_REF, betz_limit

__all__ = [
    "TurbinePowerModel",
    "PhysicsInformedPowerModel",
    "GradientBoostedPowerModel",
    "ConstantPowerModel",
    "SpeedCurveModel",
    "FrameColumnModel",
    "ALTERNATING_MET_FEATURES",
    "feature_matrix",
]


@runtime_checkable
class TurbinePowerModel(Protocol):
    """What the benchmark needs from a model."""

    name: str
    kind: str
    feature_names: tuple[str, ...]

    def fit(self, X: np.ndarray, y: np.ndarray) -> TurbinePowerModel: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


#: Features a conventional model gets: the density-corrected speed it must
#: compete on, the raw measured speed, the relative density, ambient temperature,
#: the direction as a pair (sin, cos) so 359 deg and 1 deg are adjacent, and the
#: spread between the two nacelle anemometers as a crude turbulence proxy.
ALTERNATING_MET_FEATURES = (
    "u_corrected",
    "u_raw",
    "rho_relative",
    "ambient_temperature",
    "direction_sin",
    "direction_cos",
    "anemometer_spread",
)

#: canonical feature name -> the frame column (or transform) it reads
_FEATURE_SOURCES = {
    "u_corrected": "u_corrected",
    "u_raw": "wind_speed_ms",
    "rho_relative": "rho_relative",
    "ambient_temperature": "ambient_temperature_c",
    "anemometer_spread": "anemometer_spread",
    "wind_speed_std": "wind_speed_std_ms",
    "yaw_error": "yaw_error_deg",
}


def feature_matrix(frame, names: Sequence[str]) -> np.ndarray:
    """Build the ``(n, len(names))`` design matrix a model expects.

    One place that knows how a canonical feature name maps to a SCADA column, so
    an example never hand-assembles columns and no model silently reads a
    different signal than its neighbour in the benchmark table.
    """
    import pandas as pd  # noqa: F401  (documented input type; no runtime dependency added)

    columns = []
    for name in names:
        if name == "direction_sin":
            columns.append(np.sin(np.radians(frame["wind_direction_deg"].to_numpy(dtype=float))))
        elif name == "direction_cos":
            columns.append(np.cos(np.radians(frame["wind_direction_deg"].to_numpy(dtype=float))))
        elif name in _FEATURE_SOURCES:
            columns.append(frame[_FEATURE_SOURCES[name]].to_numpy(dtype=float))
        else:
            raise KeyError(f"unknown feature {name!r}; known: {sorted(_FEATURE_SOURCES) + ['direction_sin', 'direction_cos']}")
    return np.column_stack(columns) if columns else np.zeros((len(frame), 0))


# ---------------------------------------------------------------------------
# Physics-informed monotone hinge model
# ---------------------------------------------------------------------------

class PhysicsInformedPowerModel:
    """Monotone, rate-capped, cut-in/cut-out-gated power curve.

    Parameters
    ----------
    rated_power_kw
        Nameplate. Used both as the cap and as the scale of the target.
    knots
        Wind speeds where the piecewise-linear curve may change slope. The
        curve is linear between them, which is what makes the increments
        interpretable: each one is a local dP/du that training can only push
        up, never below zero.
    region2_max_speed
        Upper end of the cubic region. Defaults to 0.8 x rated speed, the usual
        edge of region 2 before pitch control starts blending in.
    cubic_weight, smoothness_weight, betz_weight
        Weights of the three penalty terms. Recorded on the fitted model so a
        result always carries the trade-off it was trained with.
    """

    kind = "physics-informed"

    def __init__(
        self,
        *,
        rated_power_kw: float,
        rotor_diameter: float,
        hub_height: float = 59.0,
        cut_in: float = 3.0,
        cut_out: float = 25.0,
        rated_speed: float = 13.0,
        n_knots: int = 25,
        max_speed: float = 30.0,
        region2_max_speed: float | None = None,
        cubic_weight: float = 0.05,
        smoothness_weight: float = 1e-3,
        betz_weight: float = 5.0,
        saturation: float = 30.0,
        seed: int = 0,
        epochs: int = 400,
        learning_rate: float = 0.05,
        batch_size: int = 8192,
        name: str = "physics-informed (monotone hinge)",
    ) -> None:
        import torch  # local import: the rest of the package works without torch

        self._torch = torch
        self.name = name
        self.feature_names = ("u_corrected",)
        self.rated_power_kw = float(rated_power_kw)
        self.rotor_area = math.pi * (rotor_diameter / 2.0) ** 2
        self.cut_in = float(cut_in)
        self.cut_out = float(cut_out)
        self.rated_speed = float(rated_speed)
        self.region2_max_speed = float(region2_max_speed if region2_max_speed is not None
                                       else 0.8 * rated_speed)
        self.knots = np.linspace(0.0, max_speed, n_knots, dtype=np.float32)
        self.cubic_weight = float(cubic_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.betz_weight = float(betz_weight)
        self.saturation = float(saturation)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.theta: np.ndarray | None = None
        self.history: list[float] = []
        self.diagnostics: dict = {}

        # Betz bound on the cubic coefficient, in normalised power units.
        self.betz_alpha_max = 0.5 * RHO_REF * self.rotor_area * betz_limit() / (self.rated_power_kw * 1000.0)

    # -- curve construction -------------------------------------------------
    def _curve(self, theta):
        """Normalised power ``f(u)`` from the raw knot parameters (torch)."""
        torch = self._torch
        increments = torch.nn.functional.softplus(theta)
        t = torch.as_tensor(self.knots, dtype=torch.float32)
        u = self._u[:, None]
        hinges = torch.relu(u - t[None, :])
        return (hinges * increments[None, :]).sum(dim=1)

    def _forward(self, theta, u):
        torch = self._torch
        self._u = u
        raw = self._curve(theta)
        # Smooth saturation: g(f) -> 1 as f -> inf, g(f) == f for small f, dg/df > 0.
        saturation = self.saturation
        capped = 1.0 - torch.nn.functional.softplus(saturation * (1.0 - raw)) / saturation
        # Hard operating gate. A smooth sigmoid gate was tried first and rejected:
        # any transition wide enough to differentiate cleanly also leaks tens of
        # kilowatts outside the window and breaks monotonicity *inside* it near
        # cut-out, which would make the two headline physical claims approximate
        # instead of exact. The turbine genuinely stops at cut-in and cut-out, so
        # the gate is a step; samples outside it carry no gradient, and need none.
        inside = ((u >= self.cut_in) & (u <= self.cut_out)).to(u.dtype)
        fraction = torch.clamp(capped, min=0.0, max=1.0) * inside
        return fraction, raw

    def _loss(self, theta, u, target_fraction):
        torch = self._torch
        fraction, _raw = self._forward(theta, u)
        data = torch.mean((fraction - target_fraction) ** 2)

        # Region-2 cubic law, evaluated on a fixed grid so it constrains the curve
        # rather than the noise of the samples.
        grid = torch.linspace(self.cut_in, self.region2_max_speed, 32)
        self._u = grid
        curve, _ = self._forward(theta, grid)
        u3 = grid**3
        alpha = (curve * u3).sum() / (u3 * u3).sum()
        cubic = torch.mean((curve - alpha * u3) ** 2)
        betz = torch.relu(alpha - self.betz_alpha_max) ** 2

        increments = torch.nn.functional.softplus(theta)
        second = increments[2:] - 2.0 * increments[1:-1] + increments[:-2]
        smooth = torch.mean(second**2) / (torch.mean(increments) ** 2 + 1e-9)

        total = data + self.cubic_weight * cubic + self.smoothness_weight * smooth + self.betz_weight * betz
        return total, {
            "data": float(data.detach()),
            "cubic": float(cubic.detach()),
            "alpha_u3": float(alpha.detach()),
            "betz_excess": float(betz.detach()),
            "smooth": float(smooth.detach()),
        }

    # -- fit ----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> PhysicsInformedPowerModel:
        """Train on a *training-only* window; ``X[:, 0]`` is the corrected speed."""
        torch = self._torch
        torch.manual_seed(self.seed)
        u_all = np.asarray(X[:, 0], dtype=np.float32)
        target_all = np.clip(np.asarray(y, dtype=np.float32) / self.rated_power_kw, 0.0, 1.0)
        ok = np.isfinite(u_all) & np.isfinite(target_all)
        u_all, target_all = u_all[ok], target_all[ok]
        if u_all.size < 100:
            raise ValueError("not enough samples to fit the physics-informed curve")

        theta = self._initial_theta()
        optimizer = torch.optim.Adam([theta], lr=self.learning_rate)
        n = u_all.size
        rng = np.random.default_rng(self.seed)
        for _epoch in range(self.epochs):
            if n > self.batch_size:
                batch = rng.choice(n, size=self.batch_size, replace=False)
            else:
                batch = np.arange(n)
            u = torch.as_tensor(u_all[batch], dtype=torch.float32)
            tgt = torch.as_tensor(target_all[batch], dtype=torch.float32)
            self._u = u
            optimizer.zero_grad()
            loss, parts = self._loss(theta, u, tgt)
            loss.backward()
            optimizer.step()
            # Every epoch: the convergence check in VeriPhysics needs a history
            # long enough to compare two windows (it bails out below 20 points).
            self.history.append(float(loss.detach()))
        self.theta = theta.detach().numpy().copy()
        with torch.no_grad():
            u_t = torch.as_tensor(u_all, dtype=torch.float32)
            frac, _ = self._forward(theta, u_t)
            self.diagnostics = {
                **parts,
                "train_mse_fraction": float(torch.mean((frac - torch.as_tensor(target_all)) ** 2)),
                "max_fraction": float(frac.max()),
                "implied_cp_region2": float(parts["alpha_u3"] * self.rated_power_kw * 1000.0
                                            / (0.5 * RHO_REF * self.rotor_area)),
                "betz_limit": betz_limit(),
            }
        return self

    def _initial_theta(self):
        """Start from the reference curve, so the physics prior is the starting point."""
        from .power_curve import ReferencePowerCurve

        torch = self._torch
        ref = ReferencePowerCurve(
            rated_power_kw=self.rated_power_kw,
            rotor_diameter=2.0 * math.sqrt(self.rotor_area / math.pi),
            cut_in=self.cut_in,
            cut_out=self.cut_out,
        )
        knots = self.knots.astype(float)
        frac = np.clip(ref.power(knots) / self.rated_power_kw, 0.0, 1.0)
        increments = np.diff(frac, prepend=0.0) / np.maximum(np.diff(knots, prepend=0.0), 1e-6)
        # softplus^-1(x) = log(expm1(x)); clip away from 0 to keep it finite.
        raw = np.log(np.expm1(np.clip(increments, 1e-4, 50.0)))
        return torch.tensor(raw, dtype=torch.float32, requires_grad=True)

    # -- predict ------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("model is not fitted")
        torch = self._torch
        with torch.no_grad():
            u = torch.as_tensor(np.asarray(X[:, 0], dtype=np.float32))
            frac, _ = self._forward(torch.as_tensor(self.theta), u)
            return (frac.numpy() * self.rated_power_kw).astype(float)

    def curve(self, wind_speeds) -> np.ndarray:
        """Normalised curve evaluated at arbitrary speeds (no features needed)."""
        u = np.asarray(wind_speeds, dtype=float)
        return self.predict(np.column_stack([u]))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "rated_power_kw": self.rated_power_kw,
            "cut_in": self.cut_in,
            "cut_out": self.cut_out,
            "rated_speed": self.rated_speed,
            "region2_max_speed": self.region2_max_speed,
            "knots": self.knots.tolist(),
            "theta": None if self.theta is None else self.theta.tolist(),
            "weights": {
                "cubic": self.cubic_weight,
                "smoothness": self.smoothness_weight,
                "betz": self.betz_weight,
            },
            "diagnostics": self.diagnostics,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Conventional baseline
# ---------------------------------------------------------------------------

class GradientBoostedPowerModel:
    """Histogram gradient-boosted trees over the met/state features.

    Deterministic (fixed seed, no early-stopping on a validation split), so it
    can be refitted and compared without a random component deciding the ranking.
    """

    kind = "conventional-ml"

    def __init__(
        self,
        *,
        feature_names: tuple[str, ...] = ALTERNATING_MET_FEATURES,
        max_iter: int = 400,
        learning_rate: float = 0.06,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 50,
        l2_regularization: float = 1.0,
        seed: int = 0,
        name: str = "gradient-boosted trees",
    ) -> None:
        self.name = name
        self.feature_names = tuple(feature_names)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.seed = int(seed)
        self._model = None
        self.diagnostics: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostedPowerModel:
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.seed,
        ).fit(X[ok], y[ok])
        pred = self._model.predict(X[ok])
        self.diagnostics = {
            "train_rmse_kw": float(np.sqrt(np.mean((pred - y[ok]) ** 2))),
            "n_train": int(ok.sum()),
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float)), dtype=float)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "features": list(self.feature_names),
            "hyperparameters": {
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "max_leaf_nodes": self.max_leaf_nodes,
                "min_samples_leaf": self.min_samples_leaf,
                "l2_regularization": self.l2_regularization,
            },
            "diagnostics": self.diagnostics,
        }


class ConstantPowerModel:
    """Predicts the training-mean power. The floor a benchmark has to beat."""

    kind = "baseline"
    feature_names: tuple[str, ...] = ()

    def __init__(self, name: str = "constant (training mean)") -> None:
        self.name = name
        self.mean = 0.0
        self.diagnostics: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> ConstantPowerModel:
        y = np.asarray(y, dtype=float)
        self.mean = float(np.nanmean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.mean)

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "mean_kw": self.mean}


class SpeedCurveModel:
    """Wrap any ``speeds -> kW`` callable in the benchmark protocol.

    Allowed in a comparison table because the curves it wraps (reference,
    IEC binned, logistic) are real models; it exists so they can be scored
    by the same code as the fitted ones without special cases.
    """

    def __init__(self, curve, *, name: str, kind: str = "empirical") -> None:
        self._curve = curve
        self.name = name
        self.kind = kind
        self.feature_names = ("u_corrected",)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SpeedCurveModel:  # already fitted
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._curve(np.asarray(X, dtype=float)[:, 0]), dtype=float)

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind}


class FrameColumnModel:
    """A *measured reference* that arrives with the data, scored like a model.

    ``reads_frame = True`` tells the benchmark to hand it the whole frame instead
    of a feature matrix. It has no ``fit`` that learns anything: the column is the
    operator's own expected power, included so a fitted curve can be compared
    with what the site already believed. It must never be presented as a model
    this package trained.
    """

    kind = "reference"
    reads_frame = True
    feature_names: tuple[str, ...] = ()

    def __init__(self, column: str = "potential_power_kw",
                 name: str = "operator potential power (reference)") -> None:
        self.column = column
        self.name = name

    def fit(self, *_args, **_kwargs) -> FrameColumnModel:
        return self

    def predict(self, frame) -> np.ndarray:
        return np.asarray(frame[self.column], dtype=float)

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "column": self.column,
                "note": "measured reference present in the dataset, not a fitted model"}


@dataclass
class BenchmarkRow:
    """One model's line in the comparison table."""

    model: str
    kind: str
    metrics: dict
    extra: dict = field(default_factory=dict)
