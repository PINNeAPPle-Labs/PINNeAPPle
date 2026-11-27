"""Analytical wake models and the farm model built on them.

Two classical engineering wake models are implemented, both in the same units and
behind the same interface so a result can always say which one produced it:

* **Jensen (1983), a.k.a. the PARK model** -- a top-hat deficit that grows
  linearly with downstream distance, :math:`\\delta = (1-\\sqrt{1-C_T})(D/(D+2kx))^2`
  inside the cone and zero outside. Fast, robust, and still the default inside
  several commercial AEP tools. It has no notion of the wake recovering in the
  centre faster than at the edge.
* **Gaussian (Bastankhah & Porté-Agel, 2014)** -- a self-similar Gaussian
  profile whose width grows linearly with distance, with the standard
  :math:`\\sigma/D = k^* x/D + \\varepsilon` and the continuity correction that
  ties :math:`\\varepsilon` to :math:`C_T`.

Neither is CFD. Both are *engineering* models: they conserve momentum in a
depth-averaged sense, reproduce the first-order sensitivity to spacing and
thrust, and are wrong in detail (wake meandering, shear, stratification,
atmospheric stability, terrain). Any number produced here is reported as a
model estimate with the model named next to it -- never as a high-fidelity
result. See :mod:`pinneapple_systems.wind.verification` for how the farm result
is checked against SCADA.

Superposition between wakes uses the root-sum-square rule (Katić et al., 1986),
which is the common choice in engineering tools; the linear sum (Lissaman) is
available for comparison because it is the more pessimistic of the two.

References
----------
Jensen, N.O. (1983). *A note on wind generator interaction*. Risø-M-2411.
Katić, I., Højstrup, J., Jensen, N.O. (1986). *A simple model for cluster
    efficiency*. EWEC 1986.
Bastankhah, M., Porté-Agel, F. (2014). *A new analytical model for wind-turbine
    wakes*. Renewable Energy 70, 116-123.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .layout import WindFarmLayout, wind_vector

__all__ = [
    "WakeConfig",
    "jensen_deficit",
    "gaussian_deficit",
    "wake_deficits",
    "FarmPowerResult",
    "farm_power",
    "farm_efficiency",
    "direction_polar",
    "pair_geometry",
    "wake_field",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WakeConfig:
    """Everything the two models need that is not a turbine property.

    ``k`` is the Jensen wake-decay constant (0.075 onshore, 0.05 offshore is the
    usual starting point). ``gaussian_k`` is the :math:`k^*` of Bastankhah &
    Porté-Agel (2014), fitted by them to LES and field data; it is *not* tuned
    here. Both are exposed because a calibrated value would have to be reported.
    """

    kind: str = "gaussian"           # "gaussian" | "jensen"
    k: float = 0.075                 # Jensen decay constant
    gaussian_k: float = 0.0324555    # Bastankhah & Porté-Agel (2014)
    superposition: str = "rss"       # "rss" | "linear"
    max_deficit: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("gaussian", "jensen"):
            raise ValueError(f"unknown wake model {self.kind!r}")
        if self.superposition not in ("rss", "linear"):
            raise ValueError(f"unknown superposition {self.superposition!r}")
        if self.k <= 0 or self.gaussian_k <= 0:
            raise ValueError("wake decay constants must be positive")

    @property
    def label(self) -> str:
        name = "Jensen/PARK" if self.kind == "jensen" else "Gaussian (Bastankhah-Porte-Agel)"
        return f"{name}, {self.superposition.upper()} superposition"


# ---------------------------------------------------------------------------
# Single-wake deficits
# ---------------------------------------------------------------------------

def jensen_deficit(
    downstream: np.ndarray | float,
    crosswind: np.ndarray | float,
    rotor_diameter: float,
    thrust_coefficient: np.ndarray | float,
    k: float = 0.075,
) -> np.ndarray:
    """Fractional velocity deficit of the Jensen top-hat wake.

    Parameters
    ----------
    downstream
        Axial distance behind the rotor [m]. Must be positive.
    crosswind
        Absolute crosswind distance from the wake centre-line [m].
    rotor_diameter
        Upstream rotor diameter [m].
    thrust_coefficient
        :math:`C_T` of the upstream machine at its own inflow speed.
    k
        Wake-decay constant.

    Returns
    -------
    np.ndarray
        Deficit in [0, 1]; zero outside the cone.
    """
    x = np.asarray(downstream, dtype=float)
    r = np.abs(np.asarray(crosswind, dtype=float))
    ct = np.clip(np.asarray(thrust_coefficient, dtype=float), 0.0, 0.999)
    radius = rotor_diameter / 2.0 + k * x
    inside = (x > 0.0) & (r <= radius)
    centre = (1.0 - np.sqrt(1.0 - ct)) * (rotor_diameter / (rotor_diameter + 2.0 * k * x)) ** 2
    return np.where(inside, centre, 0.0)


def gaussian_deficit(
    downstream: np.ndarray | float,
    crosswind: np.ndarray | float,
    rotor_diameter: float,
    thrust_coefficient: np.ndarray | float,
    k: float = 0.0324555,
) -> np.ndarray:
    """Fractional velocity deficit of the Bastankhah & Porté-Agel Gaussian wake.

    The width follows :math:`\\sigma/D = k x/D + \\varepsilon` with
    :math:`\\varepsilon = 0.2\\sqrt{\\beta}` and
    :math:`\\beta = (1+\\sqrt{1-C_T}) / (2\\sqrt{1-C_T})`; the centre-line
    deficit is :math:`1-\\sqrt{1 - C_T/(8(\\sigma/D)^2)}`.

    Inside the near field the radicand can go negative (a physical wake cannot
    remove all the momentum); it is clamped at zero, which makes the deficit 1
    there. That clamp is a limitation of the closed form, not a result.
    """
    x = np.asarray(downstream, dtype=float)
    r = np.abs(np.asarray(crosswind, dtype=float))
    ct = np.clip(np.asarray(thrust_coefficient, dtype=float), 0.0, 0.999)
    root = np.sqrt(np.maximum(1.0 - ct, 1e-9))
    beta = (1.0 + root) / (2.0 * root)
    epsilon = 0.2 * np.sqrt(beta)
    sigma = k * x + epsilon * rotor_diameter
    safe_sigma = np.maximum(sigma, 1e-9)
    radicand = 1.0 - ct / (8.0 * (safe_sigma / rotor_diameter) ** 2)
    centre = 1.0 - np.sqrt(np.maximum(radicand, 0.0))
    deficit = centre * np.exp(-(r**2) / (2.0 * safe_sigma**2))
    return np.where(x > 0.0, np.clip(deficit, 0.0, 1.0), 0.0)


def wake_deficits(
    layout: WindFarmLayout,
    wind_speed: float,
    direction_deg: float,
    config: WakeConfig | None = None,
    thrust_coefficient=None,
) -> np.ndarray:
    """Pairwise deficit matrix ``D[j, i]`` -- how much turbine ``j`` shadows ``i``.

    ``D[j, i]`` is the fractional deficit turbine ``j`` imposes on the inflow of
    turbine ``i`` *before* superposition, and is zero unless ``j`` is upwind of
    ``i`` at this direction. The diagonal is zero. The matrix is not symmetric
    and must not be symmetrised.

    ``thrust_coefficient`` overrides the per-turbine :math:`C_T` curve; pass an
    array of length ``len(layout)`` to model, for instance, a derated machine.
    """
    cfg = config or WakeConfig()
    n = len(layout)
    positions = layout.positions
    ux, uy = wind_vector(direction_deg)
    along = positions[:, 0] * ux + positions[:, 1] * uy          # downstream coordinate
    cross = -positions[:, 0] * uy + positions[:, 1] * ux          # crosswind coordinate

    if thrust_coefficient is None:
        ct = np.asarray([t.thrust_coefficient(wind_speed) for t in layout.turbines], dtype=float)
    else:
        ct = np.broadcast_to(np.asarray(thrust_coefficient, dtype=float), (n,)).astype(float)

    dx = along[None, :] - along[:, None]      # [j, i] downstream distance j -> i
    dy = cross[None, :] - cross[:, None]      # [j, i] crosswind offset
    diameter = layout.rotor_diameters

    if cfg.kind == "jensen":
        d = jensen_deficit(dx, dy, diameter[:, None], ct[:, None], k=cfg.k)
    else:
        d = gaussian_deficit(dx, dy, diameter[:, None], ct[:, None], k=cfg.gaussian_k)
    np.fill_diagonal(d, 0.0)
    return np.where(dx > 0.0, d, 0.0)


# ---------------------------------------------------------------------------
# Farm model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FarmPowerResult:
    """Per-turbine outcome of one farm evaluation, in layout order."""

    free_stream: float
    direction_deg: float
    effective_speeds: np.ndarray       # (N,) local inflow speed [m/s]
    power_kw: np.ndarray               # (N,) electrical power [kW]
    wake_deficit: np.ndarray           # (N,) fractional velocity deficit
    power_no_wake_kw: np.ndarray       # (N,) what each machine would make alone
    config: WakeConfig
    model: str

    @property
    def total_power_kw(self) -> float:
        return float(self.power_kw.sum())

    @property
    def total_power_no_wake_kw(self) -> float:
        return float(self.power_no_wake_kw.sum())

    @property
    def efficiency(self) -> float:
        """Farm power as a fraction of the same machines running wake-free."""
        base = self.total_power_no_wake_kw
        return float(self.total_power_kw / base) if base > 0 else 0.0

    @property
    def wake_loss_fraction(self) -> float:
        return 1.0 - self.efficiency

    def to_dict(self) -> dict:
        return {
            "free_stream_ms": self.free_stream,
            "direction_deg": self.direction_deg,
            "model": self.model,
            "config": {
                "kind": self.config.kind,
                "k": self.config.k,
                "gaussian_k": self.config.gaussian_k,
                "superposition": self.config.superposition,
            },
            "total_power_kw": self.total_power_kw,
            "total_power_no_wake_kw": self.total_power_no_wake_kw,
            "efficiency": self.efficiency,
            "wake_loss_fraction": self.wake_loss_fraction,
            "per_turbine": [
                {
                    "effective_speed_ms": float(u),
                    "wake_deficit": float(d),
                    "power_kw": float(p),
                    "power_no_wake_kw": float(p0),
                }
                for u, d, p, p0 in zip(
                    self.effective_speeds, self.wake_deficit, self.power_kw,
                    self.power_no_wake_kw, strict=True,
                )
            ],
        }


def farm_power(
    layout: WindFarmLayout,
    wind_speed: float,
    direction_deg: float,
    power_curves: Sequence[Callable[[np.ndarray], np.ndarray]] | Callable[[np.ndarray], np.ndarray],
    config: WakeConfig | None = None,
    thrust_scale=None,
) -> FarmPowerResult:
    """Run the farm at one free-stream speed and direction.

    ``power_curves`` is either one callable applied to every turbine or one
    callable per turbine (in layout order); each maps an array of wind speeds to
    electrical power in kW. ``thrust_scale`` multiplies every :math:`C_T` **and**
    that turbine's own power -- the handle used by the curtailment optimiser in
    :mod:`pinneapple_systems.wind.optimization` to model a derated rotor.

    Scaling both is the modelled physics of a derated machine, not an
    implementation detail: holding the power fixed while lowering
    :math:`C_T` would let the optimiser manufacture energy out of nothing (a
    machine that makes full power and no wake), and it inflated the modelled
    curtailment gain from a few percent to 16 % before it was fixed. The linear
    relation between extracted power and :math:`C_T` is itself first order; it
    matches a rotor held at constant induction and is exact only in that limit.
    """
    cfg = config or WakeConfig()
    n = len(layout)
    u_free = float(wind_speed)
    u_eff = np.full(n, u_free)
    deficit = np.zeros(n)

    if callable(power_curves):
        curves = [power_curves] * n
    else:
        curves = list(power_curves)
        if len(curves) != n:
            raise ValueError(f"expected {n} power curves, got {len(curves)}")

    if thrust_scale is None:
        scale = np.ones(n)
    else:
        scale = np.broadcast_to(np.asarray(thrust_scale, dtype=float), (n,)).astype(float)

    ct = np.asarray([t.thrust_coefficient(u_free) for t in layout.turbines], dtype=float) * scale
    matrix = wake_deficits(layout, u_free, direction_deg, cfg, thrust_coefficient=ct)

    # Sweep upwind -> downwind so every upstream machine already has its own
    # (deficit-reduced) inflow when it creates the wake that reaches the next one.
    resolved: set[int] = set()
    for i in layout.upwind_order(direction_deg):
        if resolved:
            upstream = np.fromiter(resolved, dtype=int, count=len(resolved))
            # A wake removes a fraction of the *local* flow at the upstream rotor.
            incoming = matrix[upstream, i] * (u_eff[upstream] / max(u_free, 1e-9))
            if cfg.superposition == "rss":
                deficit[i] = min(math.sqrt(float((incoming**2).sum())), cfg.max_deficit)
            else:
                deficit[i] = min(float(incoming.sum()), cfg.max_deficit)
        u_eff[i] = u_free * (1.0 - deficit[i])
        resolved.add(i)

    power = np.array([float(np.asarray(curves[i](np.array([u]))).ravel()[0]) for i, u in enumerate(u_eff)])
    power = power * scale          # a derated rotor loses power as it sheds thrust
    power_no_wake = np.array([float(np.asarray(curves[i](np.array([u_free]))).ravel()[0]) for i in range(n)])

    return FarmPowerResult(
        free_stream=u_free,
        direction_deg=float(direction_deg),
        effective_speeds=u_eff,
        power_kw=power,
        wake_deficit=deficit,
        power_no_wake_kw=power_no_wake,
        config=cfg,
        model=cfg.label,
    )


def farm_efficiency(
    layout: WindFarmLayout,
    wind_speeds: Sequence[float],
    direction_deg: float,
    power_curves,
    config: WakeConfig | None = None,
) -> dict:
    """Farm efficiency and wake loss for a set of free-stream speeds.

    Returns the per-speed table plus the energy-mean, i.e. the ratio of summed
    energy with wakes to summed energy without — which is *not* the mean of the
    per-speed efficiencies when the speeds carry different power.
    """
    rows = []
    with_wake = 0.0
    without_wake = 0.0
    for u in wind_speeds:
        res = farm_power(layout, float(u), direction_deg, power_curves, config)
        rows.append({
            "wind_speed_ms": float(u),
            "total_power_kw": res.total_power_kw,
            "power_no_wake_kw": res.total_power_no_wake_kw,
            "efficiency": res.efficiency,
        })
        with_wake += res.total_power_kw
        without_wake += res.total_power_no_wake_kw
    return {
        "direction_deg": float(direction_deg),
        "rows": rows,
        "energy_efficiency": with_wake / without_wake if without_wake > 0 else 0.0,
        "mean_efficiency": float(np.mean([r["efficiency"] for r in rows])) if rows else 0.0,
    }


def direction_polar(
    layout: WindFarmLayout,
    wind_speeds: Sequence[float],
    power_curves,
    config: WakeConfig | None = None,
    n_directions: int = 72,
) -> dict:
    """Farm efficiency on a regular direction grid, averaged over the speeds given.

    The speeds are averaged *energy-weighted* (sum of power), so a sector where
    the farm is producing more counts more. This is the curve a wake-loss polar
    plot shows and the curve the SCADA comparison in
    :mod:`pinneapple_systems.wind.verification` is evaluated on.
    """
    directions = np.linspace(0.0, 360.0, n_directions, endpoint=False)
    efficiencies = np.zeros(n_directions)
    for k, d in enumerate(directions):
        efficiencies[k] = farm_efficiency(layout, wind_speeds, d, power_curves, config)["energy_efficiency"]
    return {
        "directions_deg": directions.tolist(),
        "efficiencies": efficiencies.tolist(),
        "wind_speeds_ms": [float(u) for u in wind_speeds],
        "model": (config or WakeConfig()).label,
    }


def pair_geometry(layout: WindFarmLayout) -> list[dict]:
    """Nearest-neighbour distances and bearings, for the layout description."""
    dist, bearing = layout.spacing_matrix()
    n = len(layout)
    np.fill_diagonal(dist, np.inf)
    rows = []
    for i in range(n):
        j = int(np.argmin(dist[i]))
        rows.append({
            "turbine": layout.names[i],
            "nearest": layout.names[j],
            "distance_m": float(dist[i, j]),
            "distance_in_diameters": float(dist[i, j] / layout.rotor_diameters[i]),
            "bearing_deg": float(bearing[i, j]),
        })
    return rows


def wake_field(
    layout: WindFarmLayout,
    points,
    wind_speed: float,
    direction_deg: float,
    config: WakeConfig | None = None,
) -> dict:
    """Velocity deficit at arbitrary plan-view points, for a map or a 3D scene.

    ``points`` is ``(P, 2)`` (or ``(P, 3)``, whose z is ignored) in the layout's
    local frame. Each point is treated as lying at hub height, so the result is a
    *plan-view footprint*: the deficit a turbine at that ground location would
    see, not the deficit at ground level.

    Every wake is seeded with the free-stream speed, so the field is the
    independent superposition of the single wakes rather than a cascade down the
    array. That is the right choice for a picture -- a cascade makes each point
    depend on an ordering -- and it means the field is slightly optimistic
    far downwind compared with :func:`farm_power`, which resolves the cascade
    turbine by turbine. The two agree exactly at the turbines themselves when no
    two of them share a wake path.

    Returns ``{"points", "deficit", "velocity_ratio", "model"}``.
    """
    cfg = config or WakeConfig()
    pts = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    positions = layout.positions
    ux, uy = wind_vector(direction_deg)
    along = pts @ np.array([ux, uy])
    cross = pts @ np.array([-uy, ux])
    turb_along = positions @ np.array([ux, uy])
    turb_cross = positions @ np.array([-uy, ux])

    dx = along[:, None] - turb_along[None, :]        # (P, N) downstream from each turbine
    dy = cross[:, None] - turb_cross[None, :]
    diameter = layout.rotor_diameters[None, :]
    ct = np.asarray([t.thrust_coefficient(wind_speed) for t in layout.turbines], dtype=float)[None, :]

    if cfg.kind == "jensen":
        d = jensen_deficit(dx, dy, diameter, ct, k=cfg.k)
    else:
        d = gaussian_deficit(dx, dy, diameter, ct, k=cfg.gaussian_k)
    if cfg.superposition == "rss":
        deficit = np.sqrt((d**2).sum(axis=1))
    else:
        deficit = d.sum(axis=1)
    deficit = np.clip(deficit, 0.0, cfg.max_deficit)
    return {
        "points": pts,
        "deficit": deficit,
        "velocity_ratio": 1.0 - deficit,
        "model": cfg.label,
        "wind_speed_ms": float(wind_speed),
        "direction_deg": float(direction_deg),
    }
