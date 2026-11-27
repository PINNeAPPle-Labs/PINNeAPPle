"""Mars over centuries: the coupled model, the ensemble, and the horizon of what it can say (#374).

:mod:`.ebm` gives the temperature; :mod:`.co2` gives the frost point and the reservoirs. This module puts them in
the same loop, one time step at a time, for as many centuries as the scenario asks:

    the sun and the greenhouse heat each latitude
        -> the surface temperature rises
            -> where it rises above the CO2 frost point at the current pressure, the cap sublimates,
               which takes latent heat, which holds the cap at the frost point while any ice is left
            -> the gas goes into the atmosphere, which raises the pressure
                -> which raises the frost point (Clausius-Clapeyron) and the greenhouse

That loop is the CO2 thermostat, and it is why warming Mars does not warm its cap: **while there is ice the cap
cannot rise above the frost point**, because the extra energy goes into sublimation instead of kelvin. The model
measures how much from an energy balance in watts, not from a rate parameter, so the only thing that decides how fast
the cap goes is how much energy it is given and how much ice there is.

Two things stop it:

* the **reservoir** -- :data:`pinneapple_physics.terraform.co2.RESERVOIRS`, the published CO2, which runs out;
* the **frost point**, which rises with the pressure the cap itself creates.

The ensemble sweeps the parameters nobody has measured for Mars -- the frost albedo, the poleward diffusivity, the
regolith reservoir and the strength of the CO2 greenhouse -- and the **spread of the global-mean temperature is the
trust horizon**: the century at which the ensemble can no longer say what the planet is doing. That is the honest
answer to "how far ahead can a model say anything about a planet we are changing".

What is not in the model is listed in :data:`~pinneapple_physics.terraform.ebm.ASSUMPTIONS`: seasons, dust, clouds,
water, dynamics beyond zonal diffusion, chemistry, and any engineering at all. The engineering is a flux.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .co2 import RESERVOIRS, FrostCap, frost_point
from .ebm import Climate, Grid, ZonalEBM
from .mars import MARS

__all__ = ["SECONDS_PER_YEAR", "LATENT_HEAT_OF_SUBLIMATION", "TARGET_WARMING_K", "SIGMA_THRESHOLD_K", "Scenario",
           "State", "Trajectory", "TerraformModel", "ParameterSpace", "sample", "Ensemble", "trust_horizon",
           "TERRAFORMING_SCENARIO"]

SECONDS_PER_YEAR = 365.25 * 86_400.0

#: Latent heat of sublimation of CO2, J/kg: the published molar value over the molar mass, ~5.73e5.
LATENT_HEAT_OF_SUBLIMATION = MARS.co2_sublimation_enthalpy / MARS.co2_molar_mass_kg_mol

#: The warming Jakosky & Edwards call necessary for liquid water to be stable: "the ~60 K warming necessary".
TARGET_WARMING_K = 60.0

#: The width of the ensemble's band at which the model stops being able to say where the planet is going. Declared:
#: **2 K**, about a third of the warming the scenario itself delivers (measured, ~8 K in a millennium), and therefore
#: the point past which the ensemble cannot tell the warming the scenario predicts from the uncertainty of the model
#: that predicted it. The measurement is the century at which the spread reaches it -- see :func:`trust_horizon`.
#: A tenth of the 60 K the whole exercise needs (6 K) is also reported, and the spread never reaches it: the model is
#: unsure how much Mars would warm, never about whether the attempt works.
SIGMA_THRESHOLD_K = 2.0


@dataclass(frozen=True)
class Scenario:
    """What is done to the planet, and for how long.

    ``greenhouse_flux_w_m2`` is an engineered greenhouse, applied as a flux. Jakosky & Edwards exclude it from their
    inventory -- "these are short-lived and without a feasible source using current technologies" -- so it is a
    declared *what-if*, not a prediction. ``cap_darkening`` is the method they do describe: raise dust onto the cap
    ("using explosives to raise dust into the atmosphere so that it would deposit on the polar caps, effectively
    decreasing their surface albedo"). Both ramp linearly from zero to full strength over ``ramp_years``.
    """

    years: float = 500.0
    ramp_years: float = 100.0
    greenhouse_flux_w_m2: float = 0.0
    cap_darkening: float = 0.0
    steps_per_year: int = 4

    def strength(self, year: float) -> float:
        """The fraction of the forcing that is on at ``year``: a linear ramp, then held."""
        if self.ramp_years <= 0.0:
            return 1.0
        return float(np.clip(year / self.ramp_years, 0.0, 1.0))

    def to_dict(self) -> dict[str, float]:
        return {"years": float(self.years), "ramp_years": float(self.ramp_years),
                "greenhouse_flux_w_m2": float(self.greenhouse_flux_w_m2),
                "cap_darkening": float(self.cap_darkening), "steps_per_year": float(self.steps_per_year)}


#: The attempt the guide runs: an engineered greenhouse of 3 W m^-2 (a little under a doubling of CO2 on Earth) and
#: 0.06 of dust on the caps, ramped in over a century and held for four more. It is a what-if -- Jakosky & Edwards
#: give no feasible source for the gas -- and the model's answer is what the planet does with it.
TERRAFORMING_SCENARIO = Scenario(years=500.0, ramp_years=150.0, greenhouse_flux_w_m2=3.0, cap_darkening=0.06)


@dataclass
class State:
    """``(members, bands)`` temperature and ice, and the two global scalars: air pressure and regolith CO2."""

    temperature: np.ndarray
    cap_pa: np.ndarray
    atmosphere_pa: np.ndarray
    regolith_pa: np.ndarray

    def copy(self) -> State:
        return State(self.temperature.copy(), self.cap_pa.copy(), self.atmosphere_pa.copy(), self.regolith_pa.copy())


@dataclass
class Trajectory:
    """What one run produced, sampled once per year. Leading axis is time; the second is the ensemble member."""

    years: np.ndarray
    temperature: np.ndarray            # (time, members, bands) K
    pressure_pa: np.ndarray            # (time, members) Pa of CO2
    cap_pa: np.ndarray                 # (time, members) Pa of CO2 held as ice
    cover: np.ndarray                  # (time, members) fraction of the polar area that is frosted
    regolith_pa: np.ndarray            # (time, members) Pa of CO2 still adsorbed
    cap_bands: np.ndarray              # (time, members, bands) Pa of CO2 ice in each band
    latitude_deg: np.ndarray
    weights: np.ndarray
    climate: dict[str, object] = field(default_factory=dict)
    scenario: dict[str, float] = field(default_factory=dict)

    @property
    def mean_temperature(self) -> np.ndarray:
        """Global mean surface temperature (K), ``(time, members)``."""
        return np.sum(self.temperature * self.weights[None, None, :], axis=-1)

    @property
    def warming(self) -> np.ndarray:
        """Warming since year zero (K), ``(time, members)``."""
        return self.mean_temperature - self.mean_temperature[0]

    def series(self, year: float) -> dict[str, np.ndarray]:
        """The state nearest a given year, as ``{name: array over members}``."""
        i = int(np.argmin(np.abs(self.years - year)))
        return {"year": self.years[i], "mean_temperature": self.mean_temperature[i],
                "temperature": self.temperature[i], "pressure_pa": self.pressure_pa[i],
                "cap_pa": self.cap_pa[i], "cover": self.cover[i], "regolith_pa": self.regolith_pa[i]}


class TerraformModel:
    """The CO2 cycle coupled to the energy balance, integrated over centuries.

    The state is the temperature of each latitude band, the CO2 ice in each band, and the regolith's remaining CO2.
    The pressure, the frost point and the albedo are all diagnosed from those three, so the total CO2 is conserved
    exactly and the tests check it to roundoff.

    ``climate`` may carry per-member arrays (shape ``(members, 1)``) in its physical fields: that is how the
    ensemble runs sixty-four different planets in one pass.
    """

    def __init__(self, climate: Climate | None = None, grid: Grid | None = None, *,
                 atmosphere_pa: float | None = None, cap_pa: float | None = None, regolith_pa: float = 0.0,
                 regolith_years: float = 1.0e4):
        self.climate = climate or Climate()
        self.grid = grid or Grid(72)
        self.ebm = ZonalEBM(self.climate, self.grid, MARS)
        self.cap = FrostCap.on_grid(self.grid.latitude_deg, self.grid.weights,
                                    total_pa=RESERVOIRS["polar_cap"].pressure_pa if cap_pa is None else cap_pa,
                                    polar_latitude_deg=self.climate.polar_latitude_deg,
                                    width_k=self.climate.frost_width_k)
        self.atmosphere_pa = float(MARS.co2_partial_pressure_pa if atmosphere_pa is None else atmosphere_pa)
        self.cap_full_pa = float(self.cap.capacity_pa.sum())
        self.regolith_pa = float(regolith_pa)
        self.regolith_years = float(regolith_years)
        self.weights = self.grid.weights
        self.inventory_pa: np.ndarray = np.asarray(self.atmosphere_pa + self.cap_full_pa + self.regolith_pa, float)
        self.forcing_scale: np.ndarray = np.asarray(1.0, float)

    # ------------------------------------------------------------------ derived quantities
    @property
    def total_pa(self) -> np.ndarray:
        """Every molecule of CO2 in the model, as an equivalent global surface pressure -- one number per member."""
        return np.atleast_1d(self.inventory_pa)

    @property
    def members(self) -> int:
        return int(np.size(self.climate.frost_albedo))

    def ice_fraction(self, cap_pa) -> np.ndarray:
        """The ice in each band as a fraction of what that band can hold -- what the albedo reads."""
        return np.clip(np.asarray(cap_pa, float) / np.maximum(self.cap.capacity_pa, 1e-30), 0.0, 1.0)

    def cover(self, cap_pa) -> np.ndarray:
        """Fraction of the polar cap area that is frosted, averaged over the polar bands only."""
        polar = self.weights * (self.cap.capacity_pa > 0.0)
        return np.sum(polar * self.ice_fraction(cap_pa), axis=-1) / max(float(polar.sum()), 1e-30)

    # ------------------------------------------------------------------ the forcing of a scenario
    def extra_flux(self, shape, scenario: Scenario, year: float, frost) -> np.ndarray:
        """The scenario's forcing, W m^-2: an engineered greenhouse plus the sunlight a darkened cap absorbs."""
        strength = scenario.strength(year)
        flux = np.broadcast_to(np.asarray(scenario.greenhouse_flux_w_m2 * strength, float), shape).astype(float)
        flux = flux * np.asarray(self.forcing_scale, float)
        if scenario.cap_darkening and strength > 0.0:
            albedo = np.asarray(self.climate.frost_albedo, float)
            darkened = np.maximum(albedo - scenario.cap_darkening * self.forcing_scale, 0.0)
            flux = flux + (albedo - darkened) * np.asarray(frost, float) * self.ebm.insolation
        return flux

    # ------------------------------------------------------------------ one step
    def step(self, state: State, dt: float, scenario: Scenario, year: float) -> State:
        """Advance one step: the energy balance, then the latent heat that moves CO2 between the cap and the air."""
        atmosphere = state.atmosphere_pa
        frost = self.ice_fraction(state.cap_pa)
        extra = self.extra_flux(state.temperature.shape, scenario, year, frost)

        temperature = self.ebm.advance(state.temperature, dt, atmosphere, frost, extra)
        frost_point_k = np.asarray(frost_point(np.maximum(atmosphere, 1e-6)), float)
        frost_point_k = frost_point_k.reshape((-1, 1)) if frost_point_k.ndim else float(frost_point_k)

        capacity = np.broadcast_to(self.cap.capacity_pa, state.cap_pa.shape)
        heat = self.climate.heat_capacity
        #: mass of CO2 per unit area of band that a column of `pa` of global-equivalent pressure amounts to
        per_pa = self.weights * MARS.gravity_m_s2

        # -- sublimation: whatever is above the frost point goes into latent heat, not into kelvin
        excess = np.maximum(temperature - frost_point_k, 0.0)
        meltable = heat * excess / LATENT_HEAT_OF_SUBLIMATION * per_pa
        melt = np.minimum(meltable, state.cap_pa)
        left_over = heat * excess - melt / per_pa * LATENT_HEAT_OF_SUBLIMATION
        temperature = np.where((state.cap_pa > 0.0) & (excess > 0.0),
                               frost_point_k + left_over / heat, temperature)
        cap = state.cap_pa - melt

        # -- condensation: the same balance the other way, limited by the room in the band and by the air
        deficit = np.maximum(frost_point_k - temperature, 0.0)
        condensable = heat * deficit / LATENT_HEAT_OF_SUBLIMATION * per_pa
        condense = np.minimum(np.minimum(condensable, capacity - cap), np.maximum(atmosphere, 0.0)[:, None])
        asked = np.sum(condense, axis=-1)                                 # the air is shared between the bands
        short = np.maximum(atmosphere, 0.0) < asked
        if np.any(short):
            scale = np.where(short, np.maximum(atmosphere, 0.0) / np.maximum(asked, 1e-30), 1.0)
            condense = condense * scale[:, None]
        short_fall = heat * deficit - condense / per_pa * LATENT_HEAT_OF_SUBLIMATION
        temperature = np.where(deficit > 0.0, frost_point_k - short_fall / heat, temperature)
        cap = cap + condense

        # -- the regolith gives up its CO2 on its own (slow) time scale, straight into the air
        release = np.minimum(state.regolith_pa / self.regolith_years * dt / SECONDS_PER_YEAR, state.regolith_pa)
        regolith = state.regolith_pa - release

        # -- the atmosphere is whatever is left: the total is conserved by construction
        atmosphere = self.total_pa - np.sum(cap, axis=-1) - regolith
        return State(temperature, cap, atmosphere, regolith)

    # ------------------------------------------------------------------ a run
    def initial_state(self, start_k: float | None = None, caps_full: bool = True, members: int = 1,
                      reference: bool = False) -> State:
        """Today's Mars: the polar caps in place and the temperature that goes with them.

        ``start_k`` is a uniform starting temperature for the settle; the returned profile is the equilibrium of the
        energy balance *with the caps present*, which is the only state in which today's Mars is self-consistent.
        Starting instead from a warm uniform planet melts the caps in the first decade and never brings them back --
        the model is bistable, and that hysteresis is a result, not a bug: a dark polar surface that has lost its ice
        does not get it back.
        """
        n = self.grid.bands
        start = MARS.mean_temperature_k if start_k is None else float(start_k)
        cap = (np.ones((members, n)) * self.cap.capacity_pa) if caps_full else np.zeros((members, n))
        frost = self.ice_fraction(cap)
        if reference and members > 1:
            # every member starts from the same observed planet: the spread that follows is the model's, not the
            # initial condition's
            reference_ebm = ZonalEBM(Climate(polar_latitude_deg=self.climate.polar_latitude_deg), self.grid, MARS)
            settled = reference_ebm.settle(self.atmosphere_pa, frost[0], years=30.0, steps_per_year=12, start=start)
            temperature = np.repeat(settled, members, axis=0)
        else:
            temperature = self.ebm.settle(self.atmosphere_pa, frost[0], years=30.0, steps_per_year=12,
                                          members=members, start=start)
        return State(temperature, cap, np.full(members, self.atmosphere_pa), np.full(members, self.regolith_pa))

    def run(self, scenario: Scenario, *, start_k: float | None = None, caps_full: bool = True,
            state: State | None = None) -> Trajectory:
        """Integrate a scenario, sampling once a year."""
        state = (self.initial_state(start_k, caps_full) if state is None else state.copy())
        dt = SECONDS_PER_YEAR / scenario.steps_per_year
        steps = int(round(scenario.years * scenario.steps_per_year))
        sample_every = max(1, int(round(scenario.steps_per_year)))
        record: dict[str, list] = {k: [] for k in ("temperature", "pressure", "cap", "cover", "regolith", "bands")}
        for i in range(steps + 1):
            if i % sample_every == 0:
                record["temperature"].append(state.temperature.copy())
                record["pressure"].append(state.atmosphere_pa.copy())
                record["cap"].append(np.sum(state.cap_pa, axis=-1))
                record["cover"].append(self.cover(state.cap_pa))
                record["regolith"].append(state.regolith_pa.copy())
                record["bands"].append(state.cap_pa.copy())
            if i == steps:
                break
            state = self.step(state, dt, scenario, i / scenario.steps_per_year + 0.5 / scenario.steps_per_year)
        years = np.arange(len(record["temperature"])).astype(float) * sample_every / scenario.steps_per_year
        return Trajectory(
            years=years,
            temperature=np.stack(record["temperature"]),
            pressure_pa=np.stack(record["pressure"]),
            cap_pa=np.stack(record["cap"]),
            cover=np.stack(record["cover"]),
            regolith_pa=np.stack(record["regolith"]),
            cap_bands=np.stack(record["bands"]),
            latitude_deg=self.grid.latitude_deg.copy(),
            weights=self.weights.copy(),
            climate={k: v for k, v in self.climate.to_dict().items()},
            scenario=scenario.to_dict(),
        )


# ---------------------------------------------------------------------------------------------------- the ensemble
@dataclass(frozen=True)
class ParameterSpace:
    """The four things nobody has measured for Mars, with the ranges the ensemble draws from.

    ``frost_albedo`` spans the dusty cap and the clean CO2 snow (the published broadband albedo of clean CO2 snow is
    0.93-0.98 and 0.01 % of dust takes ~60 % off the visible albedo); ``transport`` is the Budyko diffusivity, which
    is not measured for Mars at all; ``regolith_pa`` is how much of Jakosky & Edwards' 40 mbar of adsorbed CO2 a
    scenario manages to reach; ``greenhouse_one_bar_k`` is the CO2 greenhouse at one bar, taken as a range around the
    ~60 K the paper quotes; and ``forcing_scale`` multiplies the scenario itself, from "nothing was done" to "the
    attempt worked better than planned" -- it is the largest uncertainty of the six, because the engineering on which
    the whole question turns does not exist.
    """

    frost_albedo: tuple[float, float] = (0.70, 0.95)
    transport: tuple[float, float] = (0.001, 0.020)
    regolith_pa: tuple[float, float] = (0.0, 4000.0)
    greenhouse_one_bar_k: tuple[float, float] = (40.0, 80.0)
    forcing_scale: tuple[float, float] = (0.0, 1.5)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self.__dict__.items()}


def sample(space: ParameterSpace = ParameterSpace(), members: int = 64, seed: int = 0) -> dict[str, np.ndarray]:
    """Draw ``members`` parameter sets, uniformly in each declared range (a seeded generator: runs reproduce)."""
    rng = np.random.default_rng(seed)
    return {k: rng.uniform(lo, hi, members) for k, (lo, hi) in space.__dict__.items()}


@dataclass
class Ensemble:
    """One trajectory per member, and the spread between them."""

    trajectory: Trajectory
    parameters: dict[str, np.ndarray]

    @property
    def spread(self) -> np.ndarray:
        """Standard deviation of the global-mean temperature across members, per year (K)."""
        return np.std(self.trajectory.mean_temperature, axis=-1)

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.trajectory.mean_temperature, axis=-1)

    @property
    def warming(self) -> np.ndarray:
        return self.mean - self.mean[0]

    def horizon(self, threshold_k: float = SIGMA_THRESHOLD_K) -> float:
        """The first year at which the spread exceeds ``threshold_k``; ``nan`` if it never does."""
        return float(trust_horizon(self.trajectory.years, self.spread, threshold_k))


def trust_horizon(years, spread, threshold_k: float = SIGMA_THRESHOLD_K) -> float:
    """The year at which an ensemble's spread first exceeds a threshold, by linear interpolation.

    This is the number the guide prints: past it, the model's own uncertainty is wider than the question it is being
    asked, whatever the central value says.
    """
    years = np.asarray(years, float)
    spread = np.asarray(spread, float)
    above = np.nonzero(spread >= threshold_k)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return float(years[0])
    y0, y1 = years[i - 1], years[i]
    s0, s1 = spread[i - 1], spread[i]
    if s1 == s0:
        return float(y1)
    return float(y0 + (threshold_k - s0) * (y1 - y0) / (s1 - s0))


def run_ensemble(scenario: Scenario = TERRAFORMING_SCENARIO, space: ParameterSpace = ParameterSpace(),
                 members: int = 64, seed: int = 0, grid: Grid | None = None, regolith_years: float = 1.0e4,
                 **model_kwargs) -> Ensemble:
    """Run ``members`` planets at once: the parameters vary, the scenario does not."""
    grid = grid or Grid(72)
    parameters = sample(space, members, seed)
    climate = Climate(
        frost_albedo=parameters["frost_albedo"][:, None],
        transport=parameters["transport"][:, None],
        greenhouse_one_bar_k=parameters["greenhouse_one_bar_k"][:, None],
    )
    scenario = Scenario(years=scenario.years, ramp_years=scenario.ramp_years,
                        greenhouse_flux_w_m2=scenario.greenhouse_flux_w_m2,
                        cap_darkening=scenario.cap_darkening, steps_per_year=scenario.steps_per_year)
    model = TerraformModel(climate, grid, regolith_pa=0.0, regolith_years=regolith_years, **model_kwargs)
    state = model.initial_state(members=members, reference=True)
    state.regolith_pa = parameters["regolith_pa"].copy()
    model.inventory_pa = model.atmosphere_pa + model.cap_full_pa + state.regolith_pa
    model.forcing_scale = parameters["forcing_scale"][:, None]
    state.atmosphere_pa = model.inventory_pa - np.sum(state.cap_pa, axis=-1) - state.regolith_pa
    trajectory = model.run(scenario, state=state)
    trajectory.climate = climate.to_dict()
    return Ensemble(trajectory, parameters)
