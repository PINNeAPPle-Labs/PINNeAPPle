"""The plant: an exothermic stirred reactor with five cooling tubes, and its reference simulator.

What is modelled
----------------
A continuous stirred tank (volume ``V``) fed with the reactant A, in which ``A <-> B`` runs
reversible and exothermic. An impeller stirs it; five tubes inside the tank carry cooling water.
The three states are the two concentrations and the temperature:

    d c_A / dt = (F/V)(c_Af - c_A) - r
    d c_B / dt = (F/V)(0    - c_B) + r              r = k_f(T) c_A - k_b(T) c_B
    d T   / dt = (F/V)(T_f - T) + (-dH) r / (rho cp) - Q_cool/(rho cp V) - UA_w (T - T_amb)/(rho cp V)

Agitation enters through one physical path and only one: the tank-side film coefficient. The
classical form for a coiled or jacketed agitated vessel is ``Nu = C Re^(2/3) Pr^(1/3)`` with
``Re = rho N D^2 / mu`` (``N`` in revolutions per second, ``D`` the impeller diameter); at zero
agitation the coefficient falls back to natural convection, taken here as a floor of
``h_natural`` because an unstirred organic liquid in a vessel still loses heat. The overall
coefficient is the two films in series, ``1/U = 1/h_tank + 1/h_coolant``, and each tube passes
its duty by effectiveness-NTU against the coolant inlet temperature:

    eps = 1 - exp(-UA_tube / (m_dot cp_c)),    Q_tube = eps m_dot cp_c (T - T_cin)

``eps``-NTU is exact for a tube whose shell side is a single well-mixed temperature, which is
what a well-stirred tank is; it also gives the right limit when a tube is isolated (``UA -> 0``,
``Q -> 0``) without a singularity.

Where the numbers come from
---------------------------
The reaction thermodynamics is **read off the third-party post** this case comes from (see
:data:`AUTHOR`): two operating points, 48 °C at 75 % conversion and 75 °C at 20 %, read as two
equilibrium conversions of one reversible step. Two points determine the two unknowns, so
:func:`thermo_from_pair` returns ``dH = -85.6 kJ/mol`` and ``dS = -257 J/(mol K)`` and the model
carries them. That pin is deliberate and it is the whole debt of this case to the post: because
the thermodynamics is derived from those two numbers, reproducing them is **not** evidence for
them. It is stated here, printed by the example and repeated in the results table.

Everything else — volume, feed, residence time, tube area, coolant flow, activation energy — is
an illustrative design of the same order as a 2-3 m³ industrial reactor, chosen so that the
*agitated* steady state sits on the post's agitated operating point. The unagitated branch is
then a prediction of the model, and it is reported as measured, not as fitted.

What the model cannot say
-------------------------
The post's case is a two-dimensional field with an immersed-boundary impeller; this one is
well mixed. A well-mixed model cannot represent a conversion limited by *mixing*: whatever
segregation, dead zone or bypass the impeller removes, here is removed by assumption. So the
post's unagitated point can be read only as an energy-balance statement, and
:func:`unagitated_reading` measures what the energy balance says it would take to sit at
75 °C with 20 % conversion — which is what makes the comparison in :mod:`~.benchmark` honest.
Temperature is a bulk temperature, not a spatial maximum: there is no hotspot here, only a batch
that is hot.

Reference and trust
-------------------
:func:`steady_state` solves the stationary equations algebraically (a root of the residual in
``T``, with the conversion from the mass balance) and :func:`run_reference` integrates the
transient with a stiff solver at ``rtol = 1e-10``; the two agree to 1e-6 K on the same operating
point, which is two independent routes to the same answer. :meth:`ReferenceRun.energy_balance`
closes the first law over every run.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314_462_618
T0_K = 273.15

#: The post this case comes from: a 2-D Python/Numba CFD model of an exothermic reactor with a
#: three-blade impeller and five cooling tubes. The numbers are the author's and were not
#: verified when this case was written; they are the reference this case measures against.
AUTHOR = {
    "source": "Gabriel Hernandez Rozo, LinkedIn post (2-D Python/Numba CFD, immersed boundary)",
    "geometry": "three-blade impeller, five cooling tubes",
    "agitated": {"conversion": 0.75, "temperature_c": 48.0},
    "unagitated": {"conversion": 0.20, "temperature_c": 75.0},
    "claim": "with agitation conversion reaches about 75 % at about 48 °C; without it the "
             "conversion stalls near 20 % with hotspots near 75 °C",
}

#: Sensor noise, one sigma: an RTD and an in-line analyser of the reactant concentration.
#: Class A RTDs are good to a few tenths of a kelvin, an NIR/ATR analyser to about a per cent of
#: span; the three levels are the demonstration's own choice, not a measured plant.
NOISE_LEVELS = {"tight": (0.2, 10.0), "typical": (1.0, 40.0), "poor": (3.0, 120.0)}


def thermo_from_pair(t_a_c: float, x_a: float, t_b_c: float, x_b: float) -> tuple[float, float]:
    """``(dH, dS)`` of ``A <-> B`` implied by two equilibrium points ``(T, conversion)``.

    For a single reversible step the equilibrium conversion is ``x = K/(1+K)``, so two
    ``(T, x)`` pairs give ``K`` at two temperatures and van't Hoff gives the pair
    ``(dH, dS)``. This is the *only* thing taken from the third-party numbers, and it makes the
    case a vehicle for the twin benchmark rather than a verification of them.
    """
    k_a, k_b = x_a / (1.0 - x_a), x_b / (1.0 - x_b)
    t_a, t_b = t_a_c + T0_K, t_b_c + T0_K
    # ln K = -dH/(R T) + dS/R  ->  two equations, two unknowns
    a = (math.log(k_a) - math.log(k_b)) / (1.0 / t_a - 1.0 / t_b)
    b = math.log(k_a) - a / t_a
    return -a * R_GAS, b * R_GAS


_DH_DEFAULT, _DS_DEFAULT = thermo_from_pair(AUTHOR["agitated"]["temperature_c"],
                                            AUTHOR["agitated"]["conversion"],
                                            AUTHOR["unagitated"]["temperature_c"],
                                            AUTHOR["unagitated"]["conversion"])


@dataclass(frozen=True)
class Reactor:
    """Geometry, kinetics and cooling of the tank (SI; temperatures in the fields that say ``_c``)."""
    volume_m3: float = 3.0
    feed_m3_s: float = 1.2 / 3600.0         # 1.2 m3/h, so the residence time is 2.5 h
    c_feed: float = 2000.0                  # mol/m3 of A in the feed
    t_feed_c: float = 40.0
    rho: float = 1000.0
    cp: float = 3500.0
    delta_h: float = _DH_DEFAULT            # J/mol, from the post's pair
    delta_s: float = _DS_DEFAULT            # J/(mol K), from the post's pair
    ea: float = 60e3                        # J/mol
    k_ref_s: float = 0.01                   # k_f at k_ref_c, 1/s: the reaction is fast, not instant
    k_ref_c: float = 48.0
    n_impeller: float = 5.0                 # rev/s (300 rpm)
    d_impeller: float = 0.4                 # m, three-blade turbine
    blades: int = 3
    #: wetted area [m2] of each tube. They are not identical, and that is deliberate: five real
    #: tubes never are, so the twin's single lumped conductance is an approximation even with
    #: every tube in service, and ``phi`` has a standing bias to carry on top of the failures.
    tube_area_m2: tuple[float, ...] = (0.30, 0.32, 0.34, 0.31, 0.33)          # 1.60 m2 in all
    coolant_kg_s: tuple[float, ...] = (1.05, 0.95, 1.00, 1.10, 0.90)          # 5.00 kg/s in all
    coolant_cp: float = 4180.0
    t_coolant_c: float = 25.0
    h_coolant: float = 1400.0               # W/(m2 K), tube-side film (Dittus-Boelter, 1 kg/s)
    h_natural: float = 100.0                # W/(m2 K), the floor at zero agitation
    nu_prefactor: float = 0.5               # C in Nu = C Re^(2/3) Pr^(1/3); published 0.4-0.9
    conductivity: float = 0.15              # W/(m K), the liquid
    viscosity: float = 5e-3                 # Pa s
    ua_wall_w_k: float = 40.0               # heat lost through the wall to the room
    t_ambient_c: float = 25.0
    t_alarm_c: float = 56.0                 # the operator's high-temperature alarm
    t_trip_c: float = 65.0                  # the interlock the alarm exists to keep away from

    # ------------------------------------------------------------------ properties
    @property
    def tubes(self) -> int:
        return len(self.tube_area_m2)

    @property
    def prandtl(self) -> float:
        return self.viscosity * self.cp / self.conductivity

    def film_coefficient(self, n_rev_s: float) -> float:
        """Tank-side film coefficient [W/(m2 K)] at impeller speed ``n`` (revolutions per second).

        ``Nu = C Re^(2/3) Pr^(1/3)`` with ``Re = rho N D^2 / mu``; the natural-convection floor
        takes over when the impeller stops, which is the mechanism the whole case turns on.
        """
        re = self.rho * max(n_rev_s, 0.0) * self.d_impeller ** 2 / self.viscosity
        forced = (self.nu_prefactor * self.conductivity / self.d_impeller
                  * re ** (2.0 / 3.0) * self.prandtl ** (1.0 / 3.0))
        return max(forced, self.h_natural)

    def overall_coefficient(self, n_rev_s: float) -> float:
        """``U`` [W/(m2 K)] of the two films in series."""
        return 1.0 / (1.0 / self.film_coefficient(n_rev_s) + 1.0 / self.h_coolant)

    def cooling_area(self, tube_scale: tuple[float, ...] | None = None) -> float:
        """Wetted area actually in service [m2]; ``tube_scale[i]`` is 0 for an isolated tube."""
        scale = tube_scale or (1.0,) * self.tubes
        return float(sum(a * max(s, 0.0) for a, s in zip(self.tube_area_m2, scale, strict=True)))

    def duty_coefficient(self, n_rev_s: float,
                         tube_scale: tuple[float, ...] | None = None) -> float:
        """``dQ/dT`` of the tube bundle [W/K] at impeller speed ``n``.

        With ``eps``-NTU the duty is ``sum eps_i m cp_i (T - T_cin)``: the coefficient in front of
        the temperature difference does not depend on the temperature, only on the films and the
        tubes. Computing it once per sample instead of once per integration step is what makes the
        twin cheap, and it is exact, not an approximation.
        """
        scale = tube_scale or (1.0,) * self.tubes
        u = self.overall_coefficient(n_rev_s)
        coefficient = 0.0
        for area, flow, s in zip(self.tube_area_m2, self.coolant_kg_s, scale, strict=True):
            ua = u * area * max(s, 0.0)
            if ua <= 0.0:
                continue
            mc = flow * self.coolant_cp
            coefficient += -math.expm1(-ua / mc) * mc
        return coefficient

    def cooling_duty(self, t_k: float, inputs: Inputs, tube_scale: tuple[float, ...] | None = None) -> float:
        """Heat removed by the tubes [W] (signed: positive out of the tank), tube by tube."""
        return self.duty_coefficient(inputs.n_impeller, tube_scale) * (t_k - inputs.t_coolant_k)

    def effective_phi(self, t_k: float, inputs: Inputs,
                      tube_scale: tuple[float, ...] | None = None) -> float:
        """The duty with the tubes as they are over the duty with all five in service.

        This is the quantity the twin's latent state can identify — not the area fraction, which
        differs from it because the tubes have different areas and flows — so it is what the
        reported ``phi`` is scored against.
        """
        nominal = self.cooling_duty(t_k, inputs)
        if nominal <= 0.0:
            return 1.0
        return self.cooling_duty(t_k, inputs, tube_scale) / nominal

    # ------------------------------------------------------------------ thermodynamics and kinetics
    def equilibrium_constant(self, t_k: float) -> float:
        """``K = exp(-dH/(RT) + dS/R)``, dimensionless for ``A <-> B``."""
        expo = -self.delta_h / (R_GAS * t_k) + self.delta_s / R_GAS
        if expo > 700.0:
            return math.inf
        if expo < -700.0:
            return 0.0
        return math.exp(expo)

    def k_forward(self, t_k: float) -> float:
        """Arrhenius forward rate constant [1/s]."""
        return self.k_ref_s * math.exp(-self.ea / R_GAS * (1.0 / t_k - 1.0 / (self.k_ref_c + T0_K)))

    def rate(self, c_a: float, c_b: float, t_k: float) -> float:
        """Net rate of ``A -> B`` [mol/(m3 s)]: forward minus reverse."""
        return self.k_forward(t_k) * c_a - self.k_forward(t_k) / self.equilibrium_constant(t_k) * c_b

    # ------------------------------------------------------------------ dynamics
    def rhs(self, t_s: float, y: np.ndarray, inputs: Inputs,
            tube_scale: tuple[float, ...] | None = None) -> np.ndarray:
        """Right-hand side of the three-state model."""
        c_a, c_b, t_k = float(y[0]), float(y[1]), float(y[2])
        r = self.rate(c_a, c_b, t_k)
        q = self.cooling_duty(t_k, inputs, tube_scale)
        thermal = self.rho * self.cp * self.volume_m3
        d_t = (inputs.feed_m3_s / self.volume_m3 * (inputs.t_feed_k - t_k)
               + (-self.delta_h) * r / (self.rho * self.cp)
               - q / thermal
               - self.ua_wall_w_k * (t_k - (self.t_ambient_c + T0_K)) / thermal)
        dil = inputs.feed_m3_s / self.volume_m3
        return np.array([dil * (self.c_feed - c_a) - r, dil * (0.0 - c_b) + r, d_t])

    def conversion(self, c_a: float) -> float:
        """Conversion of the reactant in the feed: ``1 - c_A/c_Af``."""
        return 1.0 - c_a / self.c_feed

    # ------------------------------------------------------------------ stationary point
    def steady_residual(self, t_k: float, inputs: Inputs,
                        tube_scale: tuple[float, ...] | None = None) -> float:
        """Energy-balance residual [W] at the stationary concentrations for temperature ``t_k``.

        The mass balance at the stationary state is algebraic: with ``c_B = c_Af - c_A`` (the
        feed carries no B and the reactor conserves the sum of the two), the conversion follows
        ``x = k_f tau / (1 + k_f tau + k_b tau)``. Only the energy balance has to be solved.
        """
        tau = self.volume_m3 / inputs.feed_m3_s
        kf = self.k_forward(t_k)
        kb = kf / self.equilibrium_constant(t_k)
        x = kf * tau / (1.0 + kf * tau + kb * tau)
        generated = (-self.delta_h) * self.c_feed * x * inputs.feed_m3_s
        feed = self.rho * self.cp * inputs.feed_m3_s * (inputs.t_feed_k - t_k)
        return generated + feed - self.cooling_duty(t_k, inputs, tube_scale) \
            - self.ua_wall_w_k * (t_k - (self.t_ambient_c + T0_K))

    def steady_state(self, inputs: Inputs, tube_scale: tuple[float, ...] | None = None,
                     bracket: tuple[float, float] = (250.0, 520.0)) -> dict[str, float]:
        """Stationary state by root-finding, with a scan first because an exothermic tank can have
        more than one (this one has one stable point over the operating box, and that is checked)."""
        lo, hi = bracket
        grid = np.linspace(lo, hi, 600)
        values = np.array([self.steady_residual(t, inputs, tube_scale) for t in grid])
        roots: list[float] = []
        for i in range(len(grid) - 1):
            if values[i] == 0.0:
                roots.append(float(grid[i]))
            elif values[i] * values[i + 1] < 0.0:
                a, b, fa = float(grid[i]), float(grid[i + 1]), float(values[i])
                for _ in range(80):
                    mid = 0.5 * (a + b)
                    fm = self.steady_residual(mid, inputs, tube_scale)
                    if fa * fm <= 0.0:
                        b = mid
                    else:
                        a, fa = mid, fm
                roots.append(0.5 * (a + b))
        if not roots:
            raise ValueError("no stationary state inside the bracket: the tank cannot sit there")
        t_k = max(roots)                    # the hot branch when there is more than one
        tau = self.volume_m3 / inputs.feed_m3_s
        kf = self.k_forward(t_k)
        x = kf * tau / (1.0 + kf * tau + kf / self.equilibrium_constant(t_k) * tau)
        return {"t_k": t_k, "t_c": t_k - T0_K, "conversion": x, "c_a": self.c_feed * (1.0 - x),
                "c_b": self.c_feed * x, "n_roots": float(len(roots)),
                "t_alarm_c": self.t_alarm_c, "t_trip_c": self.t_trip_c,
                "cooling_w": self.cooling_duty(t_k, inputs, tube_scale),
                "ua_w_k": self.overall_coefficient(inputs.n_impeller) * self.cooling_area(tube_scale),
                "h_tank": self.film_coefficient(inputs.n_impeller)}


@dataclass(frozen=True)
class Inputs:
    """The plant's independent variables at one instant (what the operator and the utilities set)."""
    feed_m3_s: float
    t_feed_k: float
    t_coolant_k: float
    n_impeller: float

    @property
    def t_feed_c(self) -> float:
        return self.t_feed_k - T0_K

    @property
    def t_coolant_c(self) -> float:
        return self.t_coolant_k - T0_K


def inputs_of(reactor: Reactor, *, feed_m3_h: float | None = None, t_feed_c: float | None = None,
              t_coolant_c: float | None = None, n_rev_s: float | None = None) -> Inputs:
    """Inputs from the reactor's defaults, with the named ones overridden (human units)."""
    return Inputs(feed_m3_s=(reactor.feed_m3_s if feed_m3_h is None else feed_m3_h / 3600.0),
                  t_feed_k=reactor.t_feed_c + T0_K if t_feed_c is None else t_feed_c + T0_K,
                  t_coolant_k=reactor.t_coolant_c + T0_K if t_coolant_c is None else t_coolant_c + T0_K,
                  n_impeller=reactor.n_impeller if n_rev_s is None else n_rev_s)


#: the box the operator moves inside: the feed and the utilities wander, the impeller does not
NORMAL_BOX = {"feed_m3_h": (0.9, 1.5), "t_feed_c": (36.0, 44.0), "t_coolant_c": (22.0, 28.0),
              "n_rev_s": (4.4, 5.6)}

SENSORS = ("T_R", "c_A")
UNITS = {"T_R": "°C", "c_A": "mol/m3"}


@dataclass
class Faults:
    """What is wrong at one instant: a cooling tube isolated, the impeller stopped, or both."""
    tubes_lost: tuple[int, ...] = ()
    impeller_tripped: bool = False

    def scale(self, reactor: Reactor) -> tuple[float, ...]:
        return tuple(0.0 if i in self.tubes_lost else 1.0 for i in range(reactor.tubes))

    def speed(self, reactor: Reactor, n_rev_s: float) -> float:
        return 0.0 if self.impeller_tripped else n_rev_s


@dataclass
class Scenario:
    """One controlled test case: an operating history, optional faults with their onset, and a
    sensor-noise level. ``kind`` names the perturbation family of the protocol."""
    name: str
    kind: str                       # "baseline" | "noise" | "cooling" | "margin"
    seed: int
    hours: float = 4.0
    dt_s: float = 30.0
    noise: str = "typical"
    onset_s: float = 5400.0         # 1.5 h into the run
    ramp_s: float = 120.0           # a tube does not close instantly
    faults: Faults = field(default_factory=Faults)
    coolant_start_c: float | None = None   # margin family: the coolant ramps from here ...
    coolant_end_c: float | None = None     # ... to here over the run
    margin_k: tuple[float, float] | None = None   # ... or is set from the safety margin directly
    hold: bool = False              # no operator moves: the inputs stay at the middle of the box
    charge_offset_c: float = 0.0    # the tank starts this far from its stationary temperature
    note: str = ""

    @property
    def sigma(self) -> tuple[float, float]:
        return NOISE_LEVELS[self.noise]

    def truth(self) -> dict[str, Any]:
        return {"kind": self.kind, "noise": self.noise, "tubes_lost": list(self.faults.tubes_lost),
                "impeller_tripped": self.faults.impeller_tripped, "onset_s": self.onset_s,
                "coolant_start_c": self.coolant_start_c, "coolant_end_c": self.coolant_end_c}


def operating_history(reactor: Reactor, sc: Scenario) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Slow operator moves inside :data:`NORMAL_BOX` (three low-frequency sines per input, seeded),
    then the scenario's own drive: a coolant ramp for the margin family."""
    rng = np.random.default_rng(sc.seed)
    t = np.arange(0.0, sc.hours * 3600.0 + 1e-9, sc.dt_s)
    span = t[-1]
    u: dict[str, np.ndarray] = {}
    for k, (lo, hi) in NORMAL_BOX.items():
        mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
        if sc.hold:
            u[k] = np.full_like(t, mid)
            continue
        s = np.zeros_like(t)
        for _ in range(3):
            period = rng.uniform(900.0, span / 1.2)
            s += rng.uniform(0.2, 0.45) * np.sin(2 * np.pi * t / period + rng.uniform(0.0, 2 * np.pi))
        u[k] = mid + amp * np.clip(s, -0.95, 0.95)
    if sc.coolant_start_c is not None:
        end = sc.coolant_start_c if sc.coolant_end_c is None else sc.coolant_end_c
        u["t_coolant_c"] = np.linspace(sc.coolant_start_c, end, len(t))
    u["feed_m3_s"] = u.pop("feed_m3_h") / 3600.0        # SI from here on: the box is in m3/h
    return t, u


def coolant_for_margin(reactor: Reactor, inputs: Inputs, margin_k: float) -> float:
    """Coolant inlet temperature [°C] that puts the stationary bulk temperature ``margin_k`` below
    the alarm. Solved on the stationary model by bisection; the relation is monotone (a warmer
    coolant raises the tank) so this is one root, not a search."""
    target_c = reactor.t_alarm_c - margin_k
    lo, hi = -20.0, target_c - 0.2

    def f(tc: float) -> float:
        st = reactor.steady_state(replace(inputs, t_coolant_k=tc + T0_K))
        return st["t_c"] - target_c

    a, b = lo, hi
    if f(a) > 0.0:
        return a
    for _ in range(60):
        mid = 0.5 * (a + b)
        if f(mid) > 0.0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


def readings(reactor: Reactor, y: np.ndarray) -> np.ndarray:
    """True sensor values in sensor units (°C, mol/m3), order :data:`SENSORS`."""
    return np.array([float(y[2]) - T0_K, float(y[0])])


def unagitated_reading(reactor: Reactor, inputs: Inputs) -> dict[str, float]:
    """What the energy balance says the post's *unagitated* point would take.

    With the same feed, the same tubes and the impeller stopped, the stationary point is
    measured. This function also solves, for the same stopped impeller, the two quantities the
    post's unagitated pair implies: the feed temperature that would put the tank at 75 °C and
    the conversion the well-mixed model reaches there, so the difference between the pair and a
    well-mixed energy balance can be stated as a number instead of a shrug.
    """
    stopped = replace(inputs, n_impeller=0.0)
    here = reactor.steady_state(stopped)
    target_k = AUTHOR["unagitated"]["temperature_c"] + T0_K
    lo, hi = inputs.t_feed_k - 80.0, inputs.t_feed_k + 120.0

    def residual(tf: float) -> float:
        return reactor.steady_residual(target_k, replace(stopped, t_feed_k=tf))

    a, b, fa = lo, hi, residual(lo)
    fb = residual(hi)
    if fa * fb > 0.0:
        feed_c = float("nan")
        x_at_target = float("nan")
    else:
        for _ in range(80):
            mid = 0.5 * (a + b)
            fm = residual(mid)
            if fa * fm <= 0.0:
                b = mid
            else:
                a, fa = mid, fm
        feed_c = 0.5 * (a + b) - T0_K
        tau = reactor.volume_m3 / stopped.feed_m3_s
        kf = reactor.k_forward(target_k)
        x_at_target = kf * tau / (1.0 + kf * tau + kf / reactor.equilibrium_constant(target_k) * tau)
    return {"t_c": here["t_c"], "conversion": here["conversion"],
            "t_alarm_c": reactor.t_alarm_c, "t_trip_c": reactor.t_trip_c,
            "feed_for_75c_c": feed_c, "conversion_at_75c": x_at_target,
            "ua_w_k_agitated": reactor.overall_coefficient(inputs.n_impeller) * reactor.cooling_area(),
            "ua_w_k_stopped": here["ua_w_k"]}


# ----------------------------------------------------------------------------- reference run
@dataclass
class ReferenceRun:
    """Everything the true plant did, sampled at the twin's rate, with the noisy sensor readings."""
    t: np.ndarray
    y_true: np.ndarray               # (n, 3) c_A, c_B, T[K]
    inputs: dict[str, np.ndarray]
    readings_true: np.ndarray        # (n, 2) sensor units
    readings: np.ndarray             # (n, 2) with instrument noise
    tube_scale: np.ndarray           # (n, tubes) area fraction actually in service
    scenario: Scenario
    reactor: Reactor
    alarms: np.ndarray               # (n,) truth crossed the alarm
    trips: np.ndarray
    seconds: float

    # ---- convenience views in engineering units
    @property
    def temperature_c(self) -> np.ndarray:
        return self.y_true[:, 2] - T0_K

    @property
    def conversion(self) -> np.ndarray:
        return 1.0 - self.y_true[:, 0] / self.reactor.c_feed

    def cooling_w(self) -> np.ndarray:
        return np.array([self.reactor.cooling_duty(self.y_true[i, 2], self.inputs_at(i), tuple(self.tube_scale[i]))
                         for i in range(len(self.t))])

    def inputs_at(self, i: int) -> Inputs:
        """The inputs in force at sample ``i``, with the impeller trip already applied: the plant's
        own record, which is what the energy balance must be closed on."""
        return Inputs(feed_m3_s=float(self.inputs["feed_m3_s"][i]), t_feed_k=float(self.inputs["t_feed_k"][i]),
                      t_coolant_k=float(self.inputs["t_coolant_k"][i]),
                      n_impeller=float(self.inputs["n_rev_eff"][i]))

    def effective_phi(self, i: int) -> float:
        """Duty with the tubes as they are, over the duty with all five in service, at sample ``i``.

        What an observer of the cooling can see: 1.0 while every tube works, below it once one is
        isolated. The twin's latent state is scored against this, not against the area fraction.
        """
        return self.reactor.effective_phi(float(self.y_true[i, 2]), self.inputs_at(i), tuple(self.tube_scale[i]))

    def energy_balance(self) -> dict[str, float]:
        """The first law over the whole run: generation against sensible change, feed and cooling.

        Integrates ``(-dH) r`` and the three sinks from the recorded state and inputs by the
        trapezoidal rule on the same grid, and returns both sides with their relative residual.
        """
        r = np.array([self.reactor.rate(float(self.y_true[i, 0]), float(self.y_true[i, 1]), float(self.y_true[i, 2]))
                      for i in range(len(self.t))])
        generated = (-self.reactor.delta_h) * self.reactor.volume_m3 * r
        cool = self.cooling_w()
        wall = self.reactor.ua_wall_w_k * (self.y_true[:, 2] - (self.reactor.t_ambient_c + T0_K))
        feed_flow = np.array([self.inputs_at(i).feed_m3_s for i in range(len(self.t))])
        feed_t = np.array([self.inputs_at(i).t_feed_k for i in range(len(self.t))])
        feed_heat = self.reactor.rho * self.reactor.cp * feed_flow * (feed_t - self.y_true[:, 2])
        return {"generated_j": float(np.trapezoid(generated, self.t)),
                "cooling_j": float(np.trapezoid(cool, self.t)),
                "wall_j": float(np.trapezoid(wall, self.t)),
                "feed_j": float(np.trapezoid(feed_heat, self.t)),
                "grid_s": float(np.median(np.diff(self.t)))}


def tube_profile(reactor: Reactor, sc: Scenario) -> list[tuple[float, ...]]:
    """Area fraction in service of each tube at every sample (a ramp, not a step: a valve closes)."""
    n = int(round(sc.hours * 3600.0 / sc.dt_s)) + 1
    out = []
    for i in range(n):
        ti = i * sc.dt_s
        r = float(np.clip((ti - sc.onset_s) / sc.ramp_s, 0.0, 1.0)) if sc.onset_s >= 0 else 1.0
        scale = list(sc.faults.scale(reactor))
        if r > 0.0:
            scale = [1.0 - r * (1.0 - s) for s in scale]
        out.append(tuple(scale))
    return out


def run_reference(reactor: Reactor, sc: Scenario, *, rtol: float = 1e-10,
                  method: str = "LSODA") -> ReferenceRun:
    """Integrate the true plant over the scenario and sample it at the twin's rate.

    The inputs are held over each sample interval, as a control system holds them, and the
    solver is restarted from the previous state, so a fault that ramps in between two samples is
    integrated and not aliased.

    ``rtol = 1e-10`` is the reference, and how much that tolerance is worth is measured rather
    than asserted: the same run at ``rtol = 1e-12`` and the same run with Radau at ``rtol =
    1e-10`` agree with it to better than 1e-7 K (``tests/test_stirred_reactor_twin.py``). LSODA
    is the default because it is the fast one of the two on this problem, not because it is the
    more careful one.
    """
    t0 = _time.perf_counter()
    t_grid, u = operating_history(reactor, sc)
    scales = tube_profile(reactor, sc)
    # the impeller trip has its own ramp, like the tubes: the drive coasts to a stop
    n_eff = u["n_rev_s"].astype(float).copy()
    if sc.onset_s >= 0 and sc.faults.impeller_tripped:
        ramp = np.clip((t_grid - sc.onset_s) / max(sc.ramp_s, 1e-9), 0.0, 1.0)
        n_eff = n_eff * (1.0 - ramp)
    y = np.empty((len(t_grid), 3))
    y[0] = _charge(reactor, Inputs(feed_m3_s=float(u["feed_m3_s"][0]), t_feed_k=float(u["t_feed_c"][0]) + T0_K,
                                  t_coolant_k=float(u["t_coolant_c"][0]) + T0_K,
                                  n_impeller=float(n_eff[0])), offset_c=sc.charge_offset_c)
    for i in range(len(t_grid) - 1):
        held = Inputs(feed_m3_s=float(u["feed_m3_s"][i]), t_feed_k=float(u["t_feed_c"][i]) + T0_K,
                      t_coolant_k=float(u["t_coolant_c"][i]) + T0_K, n_impeller=float(n_eff[i]))
        sol = solve_ivp(lambda tt, yy, held=held, scale=scales[i]: reactor.rhs(tt, yy, held, scale),
                        (t_grid[i], t_grid[i + 1]), y[i], method=method, rtol=rtol, atol=1e-9 * np.ones(3))
        y[i + 1] = sol.y[:, -1]
    readings_true = np.array([readings(reactor, y[i]) for i in range(len(t_grid))])
    sigma_t, sigma_c = sc.sigma
    rng = np.random.default_rng(sc.seed + 7919)
    noisy = readings_true + rng.normal(0.0, 1.0, readings_true.shape) * np.array([sigma_t, sigma_c])
    temperature = readings_true[:, 0]
    return ReferenceRun(
        t=t_grid, y_true=y,
        inputs={"feed_m3_s": u["feed_m3_s"], "t_feed_c": u["t_feed_c"], "t_coolant_c": u["t_coolant_c"],
                "t_feed_k": u["t_feed_c"] + T0_K, "t_coolant_k": u["t_coolant_c"] + T0_K,
                "n_rev_s": u["n_rev_s"], "n_rev_eff": n_eff},
        readings_true=readings_true, readings=noisy, tube_scale=np.array(scales), scenario=sc, reactor=reactor,
        alarms=temperature >= reactor.t_alarm_c, trips=temperature >= reactor.t_trip_c,
        seconds=_time.perf_counter() - t0)


def _charge(reactor: Reactor, inputs: Inputs, offset_c: float = 0.0) -> np.ndarray:
    """Initial state: the tank starts where the stationary model says it sits for these inputs,
    displaced by ``offset_c`` when a case wants to watch it settle (the reference's own check)."""
    st = reactor.steady_state(inputs)
    return np.array([st["c_a"], st["c_b"], st["t_k"] + offset_c])


def with_reactor(reactor: Reactor, **kw) -> Reactor:
    return replace(reactor, **kw)
