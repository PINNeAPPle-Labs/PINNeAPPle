"""A reusable booster: ascent, turnaround, entry and a powered landing, in one trajectory.

What is modelled
----------------
The first stage of a two-stage launcher, flown as a point mass in a plane: a vertical rise
that turns over into a gravity turn up to main engine cutoff, an immediate boostback burn, a
coast to apogee, a ballistic descent with an entry burn, and a landing burn that brings the
stage to rest at the ground. Thrust follows the ambient pressure between its sea-level and
vacuum ratings, drag uses an exponential atmosphere and a constant drag coefficient, and
gravity falls off with altitude.

The question the model answers is the one a recovery engineer actually asks: *how much of the
propellant has to be held back for the way home, and does the stage still close?* The
boostback is not an input — the horizontal velocity it removes is solved by bisection so that
the stage lands where it took off.

The conservation laws it checks
-------------------------------
``mass``      propellant integrated from the mass flow against the tank difference
``momentum``  the acceleration integrated along the path against the endpoint velocity
``energy``    the specific power integrated along the path against the change in kinetic plus
              potential energy

The last two are the ones that catch a bad integrator or a sign error in the drag: a path
integral against a difference of state functions only agrees when the equations being solved
are the ones the model believes it is solving.

Limits, stated rather than discovered
-------------------------------------
Three degrees of freedom, not six: attitude is the thrust direction by assumption, so angle of
attack, base moments, control authority and structural loads are not modelled. The atmosphere
is exponential, not a standard table; drag is a constant coefficient; wind, gusts and the
second stage are absent. A real recovery has to be flown in a 6-DOF simulator, and this
model's answer is a budget, not a clearance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .base import Check, Parameter, ParameterError, Quantity, ReferenceModel, TwinRun, close

__all__ = ["ReusableBooster", "BOOSTER_PARAMETERS", "flight", "atmosphere"]

G0 = 9.80665
R_EARTH = 6_371_000.0
RHO0 = 1.225
H_SCALE = 8_500.0
P0 = 101_325.0


def atmosphere(altitude: float) -> tuple[float, float]:
    """Exponential atmosphere: density and pressure at an altitude (m).

    Two lines instead of a standard table, and the reason is stated where it matters: over
    0 to 70 km an isothermal fit is within a factor of about two on density. That moves the
    drag loss by tens of metres per second — which the model reports as its own output, so the
    size of the approximation can be judged instead of hiding inside a coefficient.
    """
    ratio = math.exp(-max(altitude, 0.0) / H_SCALE)
    return RHO0 * ratio, P0 * ratio


def _ratings(pressure: float, thrust_vac: float, thrust_sl: float,
             isp_vac: float, isp_sl: float) -> tuple[float, float]:
    """Thrust and mass flow at an ambient pressure, linear between the two ratings."""
    f = min(max(pressure / P0, 0.0), 1.0)
    thrust = thrust_vac - (thrust_vac - thrust_sl) * f
    isp = isp_vac - (isp_vac - isp_sl) * f
    return thrust, thrust / (isp * G0)


def _deriv(ph: float, pvx: float, pvh: float, pm: float, thrust: float, mdot: float,
           dirx: float, dirh: float, cd: float, area: float,
           rho0: float, h_scale: float, r_earth: float, g0: float) -> tuple[float, ...]:
    """The equations of motion, as five scalars. Module level on purpose: a closure built once
    per integration step costs more than the arithmetic it wraps."""
    rho = rho0 * math.exp(-(ph if ph > 0.0 else 0.0) / h_scale)
    speed = math.sqrt(pvx * pvx + pvh * pvh)
    g = g0 * (r_earth / (r_earth + (ph if ph > 0.0 else 0.0))) ** 2
    ax = 0.0
    ah = -g
    if speed > 1e-9:
        k = 0.5 * rho * speed * cd * area / pm
        ax -= k * pvx
        ah -= k * pvh
    if mdot > 0.0:
        ax += thrust * dirx / pm
        ah += thrust * dirh / pm
    return pvx, pvh, ax, ah, -mdot


def flight(p: Mapping[str, float], *, boostback_delta_v: float | None = None,
           record: bool = False) -> dict[str, Any]:
    """Fly the whole mission and return the budget, the losses and (optionally) the path.

    ``boostback_delta_v`` is the horizontal velocity the return burn takes out, in m/s. The
    solver in :meth:`ReusableBooster._run` iterates on it: the burn is commanded in velocity
    rather than in propellant, because downrange distance is monotone in the velocity removed
    and badly conditioned in the mass spent.

    The integrator is written in plain floats on purpose. This function is called a few dozen
    times by the solver below, and an RK4 over thousands of steps with NumPy scalars spends all
    its time in call overhead.
    """
    dry = p["dry_mass"]
    capacity = p["propellant_mass"]
    area = math.pi * p["diameter"] ** 2 / 4.0
    cd = p["drag_coefficient"]
    dt = p["time_step"]
    max_t = p["max_flight_time"]
    t_vac, t_sl = p["thrust_vacuum"], p["thrust_sea_level"]
    i_vac, i_sl = p["isp_vacuum"], p["isp_sea_level"]
    meco = p["meco_velocity"]
    boostback_target = p["boostback_delta_v"] if boostback_delta_v is None else boostback_delta_v

    x = h = vx = vh = 0.0
    m = dry + capacity
    prop_used = boostback_used = boostback_dv = 0.0
    entry_used = landing_used = 0.0
    t = 0.0
    energy_gain = momentum_x = momentum_h = thrust_dv = 0.0
    drag_loss = gravity_loss = 0.0
    max_q = max_q_altitude = 0.0
    max_speed = 0.0
    max_specific_energy = 0.0
    path: list[tuple[float, float, float, float, float]] = []

    def step(thrust_fraction: float, dirx: float, dirh: float, dt_local: float) -> float:
        """Advance one RK4 step; returns the propellant burned. Integrals are updated too."""
        nonlocal x, h, vx, vh, m, t, energy_gain, momentum_x, momentum_h, thrust_dv
        nonlocal drag_loss, gravity_loss, max_q, max_q_altitude, max_speed, max_specific_energy

        pressure = atmosphere(h)[1]
        thrust_max, mdot_max = _ratings(pressure, t_vac, t_sl, i_vac, i_sl)
        thrust = thrust_max * thrust_fraction
        mdot = mdot_max * thrust_fraction
        # The tank cannot give what it does not hold: the last step of a burn is a partial one,
        # and thrust has to fall with the flow, not stay where the throttle asked for it.
        available = max(m - dry, 0.0)
        if mdot * dt_local > available:
            mdot = available / dt_local
            isp_here = i_vac - (i_vac - i_sl) * min(max(pressure / P0, 0.0), 1.0)
            thrust = mdot * isp_here * G0

        args = (thrust, mdot, dirx, dirh, cd, area, RHO0, H_SCALE, R_EARTH, G0)
        k1 = _deriv(h, vx, vh, m, *args)
        k2 = _deriv(h + 0.5 * dt_local * k1[1], vx + 0.5 * dt_local * k1[2],
                    vh + 0.5 * dt_local * k1[3], m + 0.5 * dt_local * k1[4], *args)
        k3 = _deriv(h + 0.5 * dt_local * k2[1], vx + 0.5 * dt_local * k2[2],
                    vh + 0.5 * dt_local * k2[3], m + 0.5 * dt_local * k2[4], *args)
        k4 = _deriv(h + dt_local * k3[1], vx + dt_local * k3[2],
                    vh + dt_local * k3[3], m + dt_local * k3[4], *args)

        # Step-averaged acceleration and specific power, from the same stages that advanced the
        # state. Comparing them with the endpoints is a check on the invariant, not a contest
        # between quadrature rules.
        ax_mean = (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
        ah_mean = (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
        power_mean = 0.0
        for k in (k1, k2, k3, k4):
            g_stage = G0 * (R_EARTH / (R_EARTH + max(k[1], 0.0))) ** 2
            power_mean += (k[0] * k[2] + k[1] * k[3] + g_stage * k[1]) / 6.0

        rho_now, _ = atmosphere(h)
        speed_now = math.hypot(vx, vh)
        drag_loss += (0.5 * rho_now * speed_now ** 2 * cd * area / m) * dt_local
        gravity_loss += G0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2 * dt_local
        max_speed = max(max_speed, speed_now)
        potential_now = G0 * R_EARTH ** 2 * (1.0 / R_EARTH - 1.0 / (R_EARTH + max(h, 0.0)))
        max_specific_energy = max(max_specific_energy, 0.5 * speed_now ** 2 + potential_now)
        q = 0.5 * rho_now * speed_now ** 2
        if q > max_q:
            max_q = q
            max_q_altitude = h

        energy_gain += power_mean * dt_local
        thrust_dv += (thrust / m) * dt_local
        momentum_x += ax_mean * dt_local
        momentum_h += ah_mean * dt_local

        x += dt_local * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        h += dt_local * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        vx += dt_local * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
        vh += dt_local * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
        m += dt_local * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6.0
        t += dt_local
        if record:
            path.append((t, x, h, math.hypot(vx, vh), m))
        return max(-dt_local * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6.0, 0.0)

    # -- ascent: vertical, then a gravity turn, to main engine cutoff --------------
    while math.hypot(vx, vh) < meco and m > dry and t < max_t:
        pitch = math.radians(p["max_pitch"]) * min(1.0, h / p["turn_altitude"])
        prop_used += step(1.0, math.sin(pitch), math.cos(pitch), dt)

    # -- boostback: a horizontal burn that removes a commanded velocity -----------
    # It starts at separation, not at apogee: the stage is still climbing, and that is what
    # buys it the time to come back. Horizontal only — the vertical velocity is left to
    # gravity, and downrange responds monotonically to the velocity removed.
    sign = -1.0 if vx >= 0.0 else 1.0
    while boostback_dv < boostback_target and m > dry and t < max_t:
        # The last step is trimmed to the commanded velocity instead of overshooting it: a
        # burn that always ends on a full step quantises downrange to hundreds of metres and
        # no solver can close on the pad through it.
        step_dv = (t_vac / max(m, 1.0)) * dt
        remaining = boostback_target - boostback_dv
        fraction = 1.0 if step_dv <= remaining else min(max(remaining / step_dv, 0.0), 1.0)
        burned = step(fraction, sign, 0.0, dt)
        boostback_dv += (t_vac * fraction / max(m + burned, 1.0)) * dt
        boostback_used += burned
        prop_used += burned
        if fraction < 1.0:
            break

    # -- coast to apogee -----------------------------------------------------------
    while vh > 0.0 and h > 0.0 and t < max_t:
        step(0.0, 0.0, 1.0, dt)

    # -- descent, entry burn and powered landing ----------------------------------
    landing_ignited = False
    while h > p["target_altitude"] and t < max_t:
        speed = math.hypot(vx, vh)
        if (entry_used < p["entry_burn_propellant_fraction"] * capacity
                and 0.0 < h < p["entry_burn_altitude"] and vh < 0.0
                and speed > p["entry_target_speed"]):
            dirx = -vx / speed if speed > 1e-9 else 0.0
            dirh = -vh / speed if speed > 1e-9 else -1.0
            burned = step(1.0, dirx, dirh, dt)
            entry_used += burned
            prop_used += burned
            continue

        g = G0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2
        thrust_max, _ = _ratings(atmosphere(h)[1], t_vac, t_sl, i_vac, i_sl)
        a_max = thrust_max / m + 1.5 * 0.5 * atmosphere(h)[0] * vh * vh * cd * area / m - g
        if not landing_ignited:
            if vh < 0.0 and h < vh * vh / (2.0 * max(a_max, 1e-6)) * p["landing_lead"]:
                landing_ignited = True
            else:
                step(0.0, 0.0, 1.0, dt)
                continue

        # Suicide-burn guidance: the deceleration that nulls the velocity in the height that is
        # left, held by the throttle. The step is cut while the throttle is live, because the
        # law is stiff against a 0.05 s step at accelerations of hundreds of m/s2.
        required = vh * vh / (2.0 * max(h - p["target_altitude"], 0.02)) + g
        throttle = min(max(required / (thrust_max / m), 0.0), 1.0)
        burned = step(throttle, 0.0, 1.0, dt * 0.1)
        landing_used += burned
        prop_used += burned
        if vh >= 0.0 and h < 5.0:            # stopped on the legs: no reason to lift off again
            break

    speed_final = math.hypot(vx, vh)
    potential = G0 * R_EARTH ** 2 * (1.0 / R_EARTH - 1.0 / (R_EARTH + max(h, 0.0)))
    return dict(
        t=t, x=x, h=h, vx=vx, vh=vh, speed=speed_final, mass=m,
        propellant_used=prop_used, boostback_used=boostback_used, boostback_dv=boostback_dv,
        entry_used=entry_used, landing_used=landing_used,
        energy_gain=energy_gain, momentum=(momentum_x, momentum_h), thrust_dv=thrust_dv,
        drag_loss=drag_loss, gravity_loss=gravity_loss,
        max_q=max_q, max_q_altitude=max_q_altitude,
        max_speed=max_speed, max_specific_energy=max_specific_energy,
        final_specific_energy=potential + 0.5 * speed_final ** 2,
        path=path,
    )


BOOSTER_PARAMETERS = {
    "dry_mass": Parameter("dry_mass", "kg", 25_000.0, 2_000.0, 200_000.0, "stage mass without propellant"),
    "propellant_mass": Parameter("propellant_mass", "kg", 400_000.0, 10_000.0, 2_000_000.0,
                                 "propellant the stage carries"),
    "thrust_vacuum": Parameter("thrust_vacuum", "N", 8.0e6, 1.0e5, 5.0e7, "vacuum thrust of the stage"),
    "thrust_sea_level": Parameter("thrust_sea_level", "N", 7.4e6, 1.0e5, 5.0e7, "sea-level thrust of the stage"),
    "isp_vacuum": Parameter("isp_vacuum", "s", 330.0, 200.0, 470.0, "vacuum specific impulse"),
    "isp_sea_level": Parameter("isp_sea_level", "s", 300.0, 180.0, 450.0, "sea-level specific impulse"),
    "diameter": Parameter("diameter", "m", 3.7, 0.5, 10.0, "reference diameter for drag"),
    "drag_coefficient": Parameter("drag_coefficient", "-", 1.0, 0.1, 2.0,
                                  "constant drag coefficient; a base-first stage is a blunt body"),
    "meco_velocity": Parameter("meco_velocity", "m/s", 1900.0, 500.0, 5000.0,
                               "speed at main engine cutoff, the velocity the stage hands to the stack"),
    "max_pitch": Parameter("max_pitch", "deg", 78.0, 10.0, 89.0, "pitch off vertical reached by the turn"),
    "turn_altitude": Parameter("turn_altitude", "m", 60_000.0, 5_000.0, 150_000.0,
                               "altitude over which the gravity turn completes"),
    "boostback_delta_v": Parameter("boostback_delta_v", "m/s", 1000.0, 0.0, 4000.0,
                                   "horizontal velocity the return burn removes; solved in the model"),
    "entry_burn_altitude": Parameter("entry_burn_altitude", "m", 70_000.0, 10_000.0, 90_000.0,
                                     "altitude below which the entry burn is spent"),
    "entry_burn_propellant_fraction": Parameter("entry_burn_propellant_fraction", "-", 0.20, 0.0, 0.30,
                                                "share of the tank reserved for the entry burn"),
    "entry_target_speed": Parameter("entry_target_speed", "m/s", 350.0, 100.0, 2000.0,
                                    "speed the entry burn aims for before handing over to drag"),
    "landing_lead": Parameter("landing_lead", "-", 1.2, 1.0, 4.0,
                              "ignition safety factor on the suicide-burn altitude"),
    "target_altitude": Parameter("target_altitude", "m", 0.0, 0.0, 50.0, "height at touchdown"),
    "max_flight_time": Parameter("max_flight_time", "s", 900.0, 120.0, 3000.0, "guards the loop"),
    "time_step": Parameter("time_step", "s", 0.05, 0.005, 0.5, "RK4 step"),
}


class ReusableBooster(ReferenceModel):
    """A reusable first stage flown from the pad back to the ground."""

    slug = "reusable_booster"
    title = "Reusable booster: ascent, boostback, entry burn and powered landing"
    physics = "point-mass flight, inverse-square gravity, exponential atmosphere, pressure-dependent thrust"
    parameters = BOOSTER_PARAMETERS
    assumptions = (
        "Planar 3-DOF flight: the thrust points along the commanded direction, with no attitude dynamics.",
        "Exponential atmosphere and a constant drag coefficient, both independent of Mach.",
        "Thrust and specific impulse interpolate linearly with ambient pressure between the two ratings.",
        "No wind, no second stage, no propellant slosh, no engine chilldown or residual losses.",
    )
    limits = (
        "Six-degree-of-freedom behaviour (angle of attack, base moments, control authority) is absent.",
        "Structural loads, entry heating and aeroelastic response are not modelled.",
        "The entry burn altitude band is a scheduling choice, not an optimised trajectory.",
        "A real recovery clearance needs a 6-DOF entry simulation and a loads analysis.",
    )
    sources = (
        "Tsiolkovsky's rocket equation, used here as the independent velocity ceiling.",
        "Exponential atmosphere rho = rho0 exp(-h/H) with H = 8.5 km, the standard scale height.",
        "Suicide-burn terminal guidance, a = v^2 / (2 (h - h_target)) + g — the standard descent law.",
    )
    not_verified = (
        "No flight data: the vehicle is a concept with plausible numbers, not a flown stage.",
        "Drag coefficient is constant; the transonic drag rise is missing, which flatters the drag loss.",
        "The atmosphere is exponential rather than a standard table.",
        "Landing propellant is computed with an idealised throttle that has no minimum-throttle limit.",
    )

    def _validate(self, p: Mapping[str, float]) -> None:
        if p["thrust_sea_level"] > p["thrust_vacuum"]:
            raise ParameterError("sea-level thrust cannot exceed vacuum thrust")
        if p["isp_sea_level"] > p["isp_vacuum"]:
            raise ParameterError("sea-level specific impulse cannot exceed the vacuum value")
        if p["entry_burn_propellant_fraction"] >= 0.5:
            raise ParameterError("half the tank reserved for the entry burn leaves nothing for the way home")
        if p["target_altitude"] >= p["entry_burn_altitude"]:
            raise ParameterError("the landing altitude must sit below the entry burn band")

    def _run(self, p: Mapping[str, float]) -> TwinRun:
        capacity = p["propellant_mass"]

        def downrange(boostback: float) -> float:
            return flight(p, boostback_delta_v=boostback)["x"]

        # Solve the boostback velocity so the stage lands where it took off. Downrange is close
        # to linear in the velocity removed, so a secant iteration converges in a few flights;
        # a bisection over the whole bracket would spend twenty-five of them for the same answer.
        # Two starting guesses near the answer: a return burn has to take out the horizontal
        # velocity the stage was given, so 0.85 and 1.0 times the cutoff velocity bracket it.
        dv0 = max(0.85 * p["meco_velocity"], 200.0)
        dv1 = max(p["meco_velocity"], dv0 + 50.0)
        x0, x1 = downrange(dv0), downrange(dv1)
        best_dv, best_x = (dv0, x0) if abs(x0) <= abs(x1) else (dv1, x1)
        for _ in range(12):
            if abs(x1) < 10.0 or abs(x1 - x0) < 1e-9:
                break
            step_dv = x1 * (dv1 - dv0) / (x1 - x0)
            dv0, x0 = dv1, x1
            dv1 = min(max(dv1 - step_dv, 0.0), 4000.0)
            x1 = downrange(dv1)
            if abs(x1) < abs(best_x):
                best_dv, best_x = dv1, x1
        boostback = best_dv

        r = flight(p, boostback_delta_v=boostback, record=True)
        m0 = p["dry_mass"] + capacity
        mf = r["mass"]
        rocket_equation = p["isp_vacuum"] * G0 * math.log(m0 / mf) if 0.0 < mf < m0 else 0.0
        velocity_ideal = r["thrust_dv"]
        apex = max(point[2] for point in r["path"]) if r["path"] else r["h"]

        outputs = {
            "downrange_at_touchdown": Quantity("downrange_at_touchdown", r["x"], "m",
                                               "solved by bisection on the boostback velocity"),
            "touchdown_speed": Quantity("touchdown_speed", r["speed"], "m/s",
                                        "magnitude at the end of the run"),
            "touchdown_altitude": Quantity("touchdown_altitude", r["h"], "m",
                                           "height when the landing loop ended"),
            "flight_time": Quantity("flight_time", r["t"], "s", "pad to touchdown"),
            "propellant_used": Quantity("propellant_used", r["propellant_used"], "kg",
                                        "integrated mass flow"),
            "propellant_margin": Quantity("propellant_margin", capacity - r["propellant_used"], "kg",
                                          "left in the tank; negative means the mission does not close"),
            "boostback_delta_v": Quantity("boostback_delta_v", r["boostback_dv"], "m/s",
                                          "horizontal velocity removed by the return burn"),
            "boostback_propellant": Quantity("boostback_propellant", r["boostback_used"], "kg",
                                             "propellant the return burn spent"),
            "entry_propellant": Quantity("entry_propellant", r["entry_used"], "kg", "entry burn"),
            "landing_propellant": Quantity("landing_propellant", r["landing_used"], "kg", "landing burn"),
            "ideal_delta_v": Quantity("ideal_delta_v", velocity_ideal, "m/s",
                                      "thrust acceleration integrated along the whole flight"),
            "rocket_equation_delta_v": Quantity("rocket_equation_delta_v", rocket_equation, "m/s",
                                                "Isp_vac g0 ln(m0/mf) — the ceiling for the same propellant"),
            "gravity_loss": Quantity("gravity_loss", r["gravity_loss"], "m/s", "integral of g dt"),
            "drag_loss": Quantity("drag_loss", r["drag_loss"], "m/s", "integral of D/m dt"),
            "max_dynamic_pressure": Quantity("max_dynamic_pressure", r["max_q"] / 1e3, "kPa",
                                             "peak of q = rho v^2 / 2"),
            "max_q_altitude": Quantity("max_q_altitude", r["max_q_altitude"] / 1e3, "km",
                                       "altitude of the peak dynamic pressure"),
            "apex_altitude": Quantity("apex_altitude", apex / 1e3, "km", "highest point of the flight"),
            "final_mass": Quantity("final_mass", mf, "kg", "dry mass plus residual propellant"),
            "returned_mass_fraction": Quantity("returned_mass_fraction", mf / m0, "-",
                                               "share of the lift-off mass that comes home"),
        }

        balances = [
            close("mass", r["propellant_used"], m0 - mf, "kg",
                  detail="integrated mass flow against the tank difference"),
            close("momentum", math.hypot(*r["momentum"]), r["speed"], "m/s", tolerance=2e-2,
                  scale=r["max_speed"],
                  detail="acceleration integrated along the path against the endpoint velocity"),
            close("energy", r["energy_gain"], r["final_specific_energy"], "J/kg", tolerance=2e-2,
                  scale=r["max_specific_energy"],
                  detail="integrated specific power against kinetic plus potential energy"),
        ]

        checks = [
            Check("soft_landing", r["speed"] <= 2.0,
                  f"touchdown speed {r['speed']:.2f} m/s has to sit inside a landing-leg budget"),
            Check("returns_to_the_pad", abs(r["x"]) <= 50.0,
                  f"downrange at touchdown {r['x']:.1f} m; the return burn is solved for this"),
            Check("propellant_fits", r["propellant_used"] <= capacity + 1e-6,
                  "the mission cannot burn more propellant than the stage carries"),
            Check("velocity_below_the_rocket_equation", velocity_ideal <= rocket_equation + 1e-6,
                  "no propellant combination beats its own vacuum Tsiolkovsky limit"),
            Check("climbs_before_it_descends", apex > 10_000.0,
                  "the stage has to leave the atmosphere before it can come back through it"),
            Check("dynamic_pressure_bounded", r["max_q"] <= 80_000.0,
                  "peak q above about 80 kPa would exceed the structural envelope of this concept"),
        ]

        run = TwinRun(system=self.slug, title=self.title, parameters=dict(p), outputs=outputs,
                      balances=balances, checks=checks)
        if r["path"]:
            # One sample in ten keeps the series small enough to store and still resolves the
            # burn ends, which is where a trajectory is read.
            sample = r["path"][::10] or r["path"]
            run.series = self._series(
                len(sample),
                time_s=[point[0] for point in sample],
                downrange_km=[point[1] / 1e3 for point in sample],
                altitude_km=[point[2] / 1e3 for point in sample],
                speed_m_s=[point[3] for point in sample],
                mass_kg=[point[4] for point in sample],
            )
        run.not_verified = list(self.not_verified)
        return run
