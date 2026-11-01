"""Depth-keeping of a submerged submarine: hydrostatic instability, the trim tank and the choice of what the
controller commands.

Buoyancy at depth ``z`` (positive down) with a compressible hull in compressible seawater:

    B(z) = rho0 (1 + kw z) * V0 (1 - kh z) * g

The hull compressibility ``kh`` is usually larger than the water compressibility ``kw``. When it is,
``dB/dz < 0``: going deeper loses buoyancy, and the depth equilibrium is statically unstable. The small-signal
model is ``m_eff z'' = -B'(z) z + g u``, with ``u`` the extra mass in the trim tank. It has a real pole at
``+sqrt(-B'(z) / m_eff)``.

Vertical motion uses the net downward force ``(m g - B(z) - Cd |w| w)`` over ``(1 + ca) m``:
* ``m`` is the dry mass plus the trim-tank mass plus a disturbance mass;
* ``ca`` is the added-mass coefficient;
* ``Cd`` is a quadratic drag coefficient in N s^2/m^2.

As in many depth-keeping studies, the momentum of the water pumped in or out is neglected. The trim tank is
driven by a pump with a flow limit, and the tank has a capacity.

:func:`architectures` builds the two command choices:
* **PID → target tank mass:** the pump is a rate-limited servo that moves the tank towards the target.
* **PID → pump flow:** the command is a rate, so the actuator integrates.
:func:`depth_study` simulates both, linearises both closed loops (:func:`pinneapple_systems.control.loop_eigenvalues`)
and explains the difference. With the derivative on the measured vertical speed, the target-mass loop has the
characteristic polynomial ``m s^3 + g Kd s^2 + (k + g Kp) s + g Ki``. Commanding the flow gives
``m s^4 + c s^3 + (k + g Kd) s^2 + g Kp s + g Ki``. With no linear damping at rest (``c = 0``, quadratic drag), the
``s^3`` coefficient is zero, so by the Routh criterion no PID gains can stabilise it.

:data:`ILLUSTRATIVE` holds the parameters of a public illustrative example (not a real boat). With them,
:func:`depth_study` reproduces that example: without control the depth runs away to the 600 m limit, and with the
PID commanding the target mass it holds depth.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy import optimize

from pinneapple_systems.control import (
    PID,
    Actuator,
    Plant,
    compare_architectures,
    loop_eigenvalues,
    place_pid_second_order,
    robustness,
    simulate_loop,
)

__all__ = ["SubmarineParams", "ILLUSTRATIVE", "ILLUSTRATIVE_GAINS", "SubmarineDepth", "architectures", "depth_study"]


@dataclass(frozen=True)
class SubmarineParams:
    """Parameters of the depth-keeping model.

    Attributes
    ----------
    V0 : float [m^3]
        Displaced volume at the surface.
    rho0 : float [kg/m^3]
        Seawater density at the surface.
    kh : float [1/m]
        Hull compressibility: relative volume loss per metre of depth.
    kw : float [1/m]
        Water compressibility: relative density gain per metre of depth.
    g : float [m/s^2]
        Gravity.
    ca : float [-]
        Added-mass coefficient for vertical motion.
    drag : float [N s^2/m^2]
        Quadratic drag coefficient for vertical motion.
    z_ref : float [m]
        Reference depth, where the boat is trimmed neutral with ``trim0`` in the tank.
    trim0 : float [kg]
        Trim-tank mass at neutral buoyancy at ``z_ref``.
    tank : float [kg]
        Trim-tank capacity.
    pump : float [kg/s]
        Largest pump flow, in or out.
    z_limit : float [m]
        Depth at which a run is stopped (a study limit, not a collapse depth).
    """
    V0: float = 3000.0
    rho0: float = 1025.0
    kh: float = 8e-6
    kw: float = 4.6e-6
    g: float = 9.81
    ca: float = 0.5
    drag: float = 2e4
    z_ref: float = 100.0
    trim0: float = 20000.0
    tank: float = 40000.0
    pump: float = 50.0
    z_limit: float = 600.0


ILLUSTRATIVE = SubmarineParams()
"""The parameters of a public illustrative example (not a real boat): 3000 m^3, hull compressibility 8e-6 1/m
against water 4.6e-6 1/m, trim tank 40 t with a 50 kg/s pump, trimmed at 100 m with 20 t in the tank."""

ILLUSTRATIVE_GAINS = {"kp": 500.0, "ki": 5.0, "kd": 27500.0}
"""PID gains of the same example: kg of target tank mass per metre, per metre-second and per (m/s)."""


class SubmarineDepth:
    """The depth-keeping plant: state ``[w, z]`` (vertical speed [m/s], depth [m], positive down), input the
    trim-tank mass [kg], disturbance an extra mass [kg]."""

    def __init__(self, params: SubmarineParams = ILLUSTRATIVE):
        self.p = params
        self.dry = self.buoyancy(params.z_ref) / params.g - params.trim0

    def buoyancy(self, z: float) -> float:
        """Buoyancy at depth ``z`` [m]; returns N."""
        p = self.p
        return p.rho0 * (1 + p.kw * z) * p.V0 * (1 - p.kh * z) * p.g

    def dbuoyancy_dz(self, z: float) -> float:
        """``dB/dz`` at depth ``z`` [m]; returns N/m (negative: unstable)."""
        p = self.p
        return p.rho0 * p.V0 * p.g * (p.kw - p.kh - 2 * p.kw * p.kh * z)

    def rhs(self, x, u, d, t):
        p = self.p
        w, z = float(x[0]), float(x[1])
        m = self.dry + float(u) + float(d)
        a = (m * p.g - self.buoyancy(z) - p.drag * abs(w) * w) / ((1 + p.ca) * m)
        return np.array([a, w])

    def plant(self) -> Plant:
        """The model as a :class:`~pinneapple_systems.control.Plant` (output: depth; rate: vertical speed)."""
        return Plant(rhs=self.rhs, x0=[0.0, self.p.z_ref], output=1, rate=lambda x: float(x[0]),
                     name="submarine depth", state_names=("w [m/s]", "z [m]"))

    def neutral_depth(self, trim: float, extra: float = 0.0) -> float:
        """Depth [m] where the weight with ``trim`` [kg] in the tank and ``extra`` [kg] equals the buoyancy."""
        m = self.dry + trim + extra
        return float(optimize.brentq(lambda z: m * self.p.g - self.buoyancy(z), -1e4, 1e5))

    def linear(self, z: float | None = None) -> dict[str, float]:
        """Small-signal model ``m_eff z'' + c z' + k z = b u`` at depth ``z`` (default ``z_ref``), trimmed neutral.

        Returns
        -------
        m_eff : float [kg]
            Mass with added mass.
        c : float [kg/s]
            Linear damping at rest (0 for quadratic drag).
        k : float [N/m]
            ``dB/dz``: negative means statically unstable.
        b : float [N/kg]
            Force per kg of trim-tank mass (``g``).
        growth_rate : float [1/s]
            ``sqrt(-k / m_eff)`` when unstable, else 0.
        doubling_time : float [s]
            ``ln 2 / growth_rate`` (``inf`` when stable).
        """
        p = self.p
        z = p.z_ref if z is None else float(z)
        m = self.buoyancy(z) / p.g
        m_eff = (1 + p.ca) * m
        k = self.dbuoyancy_dz(z)
        lam = math.sqrt(-k / m_eff) if k < 0 else 0.0
        return {"m_eff": m_eff, "c": 0.0, "k": k, "b": p.g, "growth_rate": lam,
                "doubling_time": math.log(2) / lam if lam > 0 else math.inf}


def architectures(params: SubmarineParams = ILLUSTRATIVE, gains: dict[str, float] | None = None
                  ) -> dict[str, tuple[Any, Actuator]]:
    """The command choices for depth-keeping, with the same PID gains.

    Returns
    -------
    dict
        ``name -> (controller, actuator)``:
        * ``'pid_target_mass'``: the PID commands the target tank mass, clamped to the tank; the pump moves the tank
          towards it at up to ``pump`` kg/s.
        * ``'pid_pump_flow'``: the same PID commands the pump flow directly, clamped to ``+-pump``; the tank mass
          integrates it.
        Both use the derivative on the measured vertical speed.
    """
    g = dict(ILLUSTRATIVE_GAINS if gains is None else gains)
    p = params
    target = PID(g["kp"], g["ki"], g["kd"], setpoint=p.z_ref, output_limits=(0.0, p.tank), bias=p.trim0,
                 derivative="measurement")
    servo = Actuator(lo=0.0, hi=p.tank, rate=p.pump, initial=p.trim0)
    flow = PID(g["kp"], g["ki"], g["kd"], setpoint=p.z_ref, output_limits=(-p.pump, p.pump),
               derivative="measurement")
    pump = Actuator(lo=0.0, hi=p.tank, rate=p.pump, integrating=True, initial=p.trim0)
    # the PID acts on e = r - z: a positive error (too shallow) asks for more mass
    return {"pid_target_mass": (target, servo), "pid_pump_flow": (flow, pump)}


class _OpenLoop:
    def act(self, t, x, y, r, ydot, dt):
        return self.u

    def set_equilibrium(self, y, r, command, t=0.0):
        self.u = command

    def get_state(self):
        return np.array([self.u])

    def set_state(self, s):
        self.u = float(s[0])


def depth_study(params: SubmarineParams = ILLUSTRATIVE, gains: dict[str, float] | None = None,
                disturbance_kg: float = 1500.0, t_disturbance: float = 60.0, t_end: float = 1800.0,
                dt: float = 0.25, method: str = "sequential_euler", tune: bool = True,
                robustness_samples: int = 0, seed: int = 0, tune_wn_factor: float = 8.0,
                tune_zeta: float = 0.8) -> dict[str, Any]:
    """Depth-keeping after a sudden extra mass: open loop, the two command architectures, their closed-loop
    eigenvalues, pole-placement tuning and (optionally) robustness to the hull compressibility and the drag.

    Parameters
    ----------
    params : SubmarineParams
        Model parameters.
    gains : dict, optional
        ``{'kp', 'ki', 'kd'}`` of the PID (default :data:`ILLUSTRATIVE_GAINS`).
    disturbance_kg : float [kg]
        Extra mass that appears at ``t_disturbance``.
    t_disturbance : float [s]
        Time of the disturbance.
    t_end, dt : float [s]
        Duration and step.
    method : str
        Integrator of :func:`pinneapple_systems.control.simulate_loop`. The default is ``'sequential_euler'``,
        the semi-implicit Euler of the illustrative example.
    tune : bool
        Also design gains by pole placement on the linear model and simulate them.
    tune_wn_factor : float [-]
        Natural frequency of the designed dominant poles, as a multiple of the open-loop growth rate.
    tune_zeta : float [-]
        Damping ratio of the designed dominant poles.
    robustness_samples : int
        Monte Carlo samples over ``kh`` (+-30 %) and the drag (/3 to x3); 0 skips it.

    Returns
    -------
    linear : dict
        :meth:`SubmarineDepth.linear` at the reference depth: ``growth_rate`` [1/s], ``doubling_time`` [s],
        ``k`` [N/m], ``m_eff`` [kg].
    open_loop : dict
        ``stopped`` (str or None), ``t_stop`` [s], ``z_max`` [m], ``final_z`` [m].
    architectures : list of dict
        Per architecture (:func:`pinneapple_systems.control.compare_architectures`):
        ``stable_linear``, ``max_real`` [1/s], ``doubling_time`` [s], ``y_min`` / ``y_max`` / ``final_y`` [m],
        ``final_trim`` [kg], ``stopped``, ``t_end`` [s].
    tuned : dict
        With ``tune``: the pole-placement gains (kg/m, kg/(m s), kg s/m), their design poles [1/s] and the same
        rows for the tuned target-mass loop.
    robustness : dict
        With ``robustness_samples``: ``fraction_stable`` [-] and quantiles of the depth excursion [m].
    runs : dict
        Full trajectories per case, for plots.
    """
    sub = SubmarineDepth(params)
    plant = sub.plant()
    p = params

    def dist(t):
        return disturbance_kg if t >= t_disturbance else 0.0

    def stop(x, t):
        if x[1] > p.z_limit:
            return f"depth beyond {p.z_limit:g} m"
        if x[1] < 0:
            return "surfaced"
        return None

    x_eq = [0.0, p.z_ref]
    ol = simulate_loop(plant, _OpenLoop(), Actuator(lo=0.0, hi=p.tank, initial=p.trim0), p.z_ref, t_end, dt,
                       disturbance=dist, method=method, stop=stop)
    arch = architectures(p, gains)
    cmp_ = compare_architectures(plant, arch, p.z_ref, t_end, dt, disturbance=dist, method=method, stop=stop,
                                 x_eq=x_eq, u_eq=p.trim0, d_eq=0.0)
    for row in cmp_["rows"]:
        row["final_trim"] = float(cmp_["runs"][row["name"]]["applied"][-1])
    out: dict[str, Any] = {
        "params": asdict(p), "gains": dict(ILLUSTRATIVE_GAINS if gains is None else gains),
        "linear": sub.linear(),
        "open_loop": {"stopped": ol["stopped"], "t_stop": float(ol["t"][-1]), "z_max": ol["y_max"],
                      "final_z": float(ol["y"][-1])},
        "architectures": cmp_["rows"],
        "runs": {"open_loop": ol, **cmp_["runs"]},
    }
    if tune:
        lin = sub.linear()
        # dominant pair several times faster than the instability, the integral pole four times slower
        wn = tune_wn_factor * lin["growth_rate"] if lin["growth_rate"] > 0 else 0.02
        pp = place_pid_second_order(lin["m_eff"], lin["c"], lin["k"], lin["b"], wn=wn, zeta=tune_zeta, p=wn / 4)
        tg = {"kp": pp["kp"], "ki": pp["ki"], "kd": pp["kd"]}
        arch_t = {"pid_target_mass_tuned": architectures(p, tg)["pid_target_mass"]}
        ct = compare_architectures(plant, arch_t, p.z_ref, t_end, dt, disturbance=dist, method=method, stop=stop,
                                   x_eq=x_eq, u_eq=p.trim0, d_eq=0.0)
        for row in ct["rows"]:
            row["final_trim"] = float(ct["runs"][row["name"]]["applied"][-1])
        out["tuned"] = {"gains": tg, "design": {"wn": wn, "zeta": tune_zeta, "p": wn / 4},
                        "design_poles": pp["poles"], "rows": ct["rows"]}
        out["runs"].update(ct["runs"])
    if robustness_samples:
        g0 = dict(ILLUSTRATIVE_GAINS if gains is None else gains)

        def run(prm):
            q = replace(p, kh=prm["kh"], drag=prm["drag"])
            s2 = SubmarineDepth(q)
            ctrl, act = architectures(q, g0)["pid_target_mass"]
            le = loop_eigenvalues(s2.plant(), ctrl, act, x_eq, q.trim0, q.z_ref, dt, method=method)
            sim = simulate_loop(s2.plant(), ctrl, act, q.z_ref, t_end, dt, disturbance=dist, method=method,
                                stop=stop, u0=q.trim0)
            return {"stable": le["stable"] and sim["stopped"] is None,
                    "excursion_m": max(sim["y_max"] - q.z_ref, q.z_ref - sim["y_min"])}

        rb = robustness(run, {"kh": (0.7 * p.kh, 1.3 * p.kh), "drag": (p.drag / 3, 3 * p.drag)},
                        n=robustness_samples, seed=seed, metric_keys=("excursion_m",))
        out["robustness"] = {"fraction_stable": rb["fraction_stable"], "quantiles": rb["quantiles"],
                             "worst": rb["worst"], "n": robustness_samples}
    return out
