"""Programmable matter: engineering materials beyond conventional design.

A material whose properties are functions of the field it carries turns a linear problem into a non-linear one, and
every claim about it can still be settled exactly -- which is what makes it a good speculative system to start from.
Three laws of the same steady conductor ``d/dx[k dT/dx] = 0`` are solved by finite differences and compared with
their closed forms:

**A conductivity that depends on the temperature**, ``k(T) = k0 (1 + beta T)``. Integrating the flux once gives::

    q = k0 [(T1 - T2) + (beta/2)(T1^2 - T2^2)] / L

and, with ``F(T) = k0 (T + beta T^2/2)``, the profile in closed form::

    F(T(x)) = F(T1) - q x      ->      T(x) = [-1 + sqrt(1 + 2 beta (F(T1) - q x)/k0)] / beta

**A conductivity that depends on the position** (a graded metamaterial), ``k(x) = k0 (1 + gamma x/L)``::

    q = k0 gamma (T1 - T2) / (L ln(1 + gamma)),      T(x) = T1 - (q L / (k0 gamma)) ln(1 + gamma x/L)

**A conductivity that switches** at a transition temperature ``T_s`` (the "structure that changes its properties":
``k_a`` below, ``k_b`` above), with the interface where ``T = T_s`` matched from both sides::

    q = [k_a (T1 - T_s) + k_b (T_s - T2)] / L

The interesting question is not whether these can be solved -- they can, in closed form -- but what a material can
and cannot buy. A conductivity that falls with temperature lowers the sensitivity ``dq/dT1 = k0 (1 + beta T1)/L``,
which is the property a "self-regulating" envelope would want; it reaches zero exactly at ``T1 = -1/beta``, which is
also the temperature at which the conductivity itself vanishes, so the material stops conducting at the moment it
stops responding. The run measures both numbers and states the consequence instead of the marketing.
"""
from __future__ import annotations

import numpy as np

from ..core import SpeculativeExperiment, register
from ..numerics import refinement, steady_conduction_1d
from ..schema import Claim, Hypothesis, Input, Proposal, Scenario

__all__ = ["ProgrammableMatter", "programmable_matter", "variable_k_flux", "graded_k_flux", "switch_k_flux"]


def variable_k_flux(k0: float, beta: float, t_hot: float, t_cold: float, length: float = 1.0) -> float:
    """Exact flux of ``d/dx[k0 (1 + beta T) dT/dx] = 0`` with the two temperatures imposed."""
    return k0 * ((t_hot - t_cold) + 0.5 * beta * (t_hot**2 - t_cold**2)) / length


def variable_k_profile(k0: float, beta: float, t_hot: float, q: float, x: np.ndarray):
    """Exact profile of the same law, from ``F(T) = F(T1) - q x`` with ``F(T) = k0 (T + beta T^2/2)``."""
    f1 = k0 * (t_hot + 0.5 * beta * t_hot**2)
    a = (f1 - q * np.asarray(x, dtype=float)) / k0
    disc = 1.0 + 2.0 * beta * a
    if np.any(disc < 0):
        raise ValueError("the closed form has no real branch over this range: the conductivity changes sign")
    return (np.sqrt(disc) - 1.0) / beta


def graded_k_flux(k0: float, gamma: float, t_hot: float, t_cold: float, length: float = 1.0) -> float:
    """Exact flux of ``d/dx[k0 (1 + gamma x/L) dT/dx] = 0``."""
    return k0 * gamma * (t_hot - t_cold) / (length * np.log1p(gamma))


def switch_k_flux(k_a: float, k_b: float, t_switch: float, t_hot: float, t_cold: float) -> float:
    """Exact flux when ``k`` switches at ``T_s``, with ``T_s`` between the two imposed temperatures."""
    if not t_cold < t_switch < t_hot:
        raise ValueError("the switch temperature must lie between the two imposed temperatures")
    return k_a * (t_hot - t_switch) + k_b * (t_switch - t_cold)


@register
class ProgrammableMatter(SpeculativeExperiment):
    name = "futures.programmable_matter"
    title = "Programmable matter: engineering materials beyond conventional design"
    area = "advanced_materials"
    topic = "conductivities that depend on the field, on the position and on a transition"
    level = "research"
    kind = "theory"
    dims = ("1D",)
    scenario_params = ()
    answers = ("conductivity", "material", "temperature")
    description = ("Three programmable conductivities against their closed forms, the sensitivity a temperature-"
                   "dependent conductivity buys, and the point where the material stops conducting.")
    references = ("Steady conduction with k = k0 (1 + beta T): d/dx[k dT/dx] = 0 integrated once, in this module's "
                  "docstring.",
                  "Conduction in a graded medium: the series-resistance integral q = (T1 - T2)/int dx/k(x).")
    params = {
        "k0": 5.0,              # W/m/K, reference conductivity
        "beta": 2e-3,           # 1/K, the temperature coefficient
        "gamma": 0.8,           # -, the grading strength
        "t_hot": 400.0,         # K
        "t_cold": 300.0,        # K
        "k_a": 2.0,             # W/m/K, below the transition
        "k_b": 8.0,             # W/m/K, above the transition
        "t_switch": 350.0,      # K
        "n": 201,               # nodes of the finite-difference solution
    }
    quick = {"n": 81}

    # ------------------------------------------------------------------ scenario
    def world(self) -> Scenario:
        return Scenario(
            "lab_bench", "A materials bench",
            "A one-dimensional sample between two thermostats, at standard conditions.",
            inputs=(Input("t_hot", 400.0, "K", "design_input", "the hot thermostat of the bench"),
                    Input("t_cold", 300.0, "K", "design_input", "the cold thermostat of the bench"),
                    Input("k0", 5.0, "W/m/K", "hypothesis",
                          "a programmable material that does not exist: its reference conductivity is a "
                          "hypothesis, which is what makes every result here hypothetical at best"),
                    Input("beta", 2e-3, "1/K", "hypothesis", "the programmed temperature coefficient")),
            claims=(Claim("steadiness and one-dimensional conduction across the sample", "d/dx[k dT/dx] = 0",
                          "established", "energy conservation in steady state"),
                    Claim("the conductivity can be made to depend on the temperature, on the position or on a "
                          "transition", "k(T, x)", "hypothetical",
                          "no material with these programmed laws exists at the stated coefficients")))

    # ------------------------------------------------------------------ run
    def _run(self, res, dim: str, p: dict) -> None:
        k0, beta = float(p["k0"]), float(p["beta"])
        t_hot, t_cold, n = float(p["t_hot"]), float(p["t_cold"]), int(p["n"])

        # ---------------------------------------------------------------- 1. k(T)
        sol = steady_conduction_1d(lambda t: k0 * (1.0 + beta * t), t_hot, t_cold, n=n)
        q_exact = variable_k_flux(k0, beta, t_hot, t_cold)
        res.value("flux_variable_k", sol["q"], "W/m^2", "finite differences with k(T) = k0 (1 + beta T)")
        res.compare("flux through a temperature-dependent conductor", sol["q"], q_exact, tol=1e-5,
                    kind="closed_form", unit="W/m^2",
                    source="q = k0 [(T1-T2) + (beta/2)(T1^2 - T2^2)] / L, one integration of d/dx[k dT/dx] = 0")
        t_exact = variable_k_profile(k0, beta, t_hot, q_exact, sol["x"])
        err = float(np.max(np.abs(sol["T"] - t_exact)) / (t_hot - t_cold))
        res.check("the profile matches the implicit closed form", err < 1e-6, value=err, limit=1e-6,
                  detail="max |T_FD - T_closed| relative to the applied difference", source="analytic")
        res.check("the non-linear solver converged", sol["converged"], value=float(sol["iterations"]),
                  limit=2000.0, detail="Picard iterations to the relative tolerance", source="stability")
        spread = float(np.ptp(sol["faces"]) / abs(sol["q"]))
        res.check("the flux is uniform along the sample", spread < 1e-10, value=spread, limit=1e-10,
                  detail="spread of the face fluxes, relative: the discretisation conserves", source="conservation")

        errors = []
        for nn in (n // 4, n // 2, n):
            s = steady_conduction_1d(lambda t: k0 * (1.0 + beta * t), t_hot, t_cold, n=nn)
            errors.append(abs(s["q"] - q_exact))
        orders = [float(np.log(errors[i] / errors[i + 1]) / np.log(2.0)) for i in range(len(errors) - 1)
                  if errors[i] > 0]
        res.check("second-order convergence under refinement", bool(orders) and min(orders) > 1.7,
                  value=float(min(orders)) if orders else None, limit=1.7,
                  detail=f"observed orders {', '.join(f'{o:.2f}' for o in orders)} at n = {n // 4}, {n // 2}, {n}",
                  source="convergence")

        # ---------------------------------------------------------------- 2. k(x), a graded metamaterial
        gamma = float(p["gamma"])
        graded = steady_conduction_1d(lambda t: k0, t_hot, t_cold, n=n, spatial=lambda x: 1.0 + gamma * x)
        q_graded = graded_k_flux(k0, gamma, t_hot, t_cold)
        res.value("flux_graded_k", graded["q"], "W/m^2", "finite differences with k(x) = k0 (1 + gamma x/L)")
        res.compare("flux through a graded conductor", graded["q"], q_graded, tol=1e-5, kind="closed_form",
                    unit="W/m^2",
                    source="q = (T1 - T2)/int_0^L dx/k(x) = k0 gamma (T1-T2)/(L ln(1+gamma))")
        t_graded = t_hot - q_graded * np.log1p(gamma * graded["x"]) / (k0 * gamma)
        err_g = float(np.max(np.abs(graded["T"] - t_graded)) / (t_hot - t_cold))
        res.check("the graded profile matches its closed form", err_g < 1e-5, value=err_g, limit=1e-5,
                  detail="max |T_FD - T_closed| relative to the applied difference", source="analytic")

        # ---------------------------------------------------------------- 3. k switching at a transition
        k_a, k_b, t_s = float(p["k_a"]), float(p["k_b"]), float(p["t_switch"])
        switch = steady_conduction_1d(lambda t: np.where(t < t_s, k_a, k_b), t_hot, t_cold, n=n)
        q_switch = switch_k_flux(k_a, k_b, t_s, t_hot, t_cold)
        res.value("flux_switching_k", switch["q"], "W/m^2", "finite differences with k switching at T_s")
        res.compare("flux through a switching conductor", switch["q"], q_switch, tol=2e-2, kind="closed_form",
                    unit="W/m^2",
                    source="q = [k_a (T1 - T_s) + k_b (T_s - T2)]/L, the interface where T = T_s matched from "
                           "both sides",
                    how="the interface is smeared over one cell by the nodal switch, so this converges at first "
                        "order; the ladder below measures that order instead of assuming it")
        i_switch = int(np.argmin(np.abs(switch["T"] - t_s)))
        res.value("interface_position", float(switch["x"][i_switch]), "m", "where the solution crosses T_s")
        res.check("the switch temperature is inside the sample", 0.0 < switch["x"][i_switch] < 1.0,
                  value=float(switch["x"][i_switch]), limit=1.0,
                  detail="outside it there is no interface and the two-layer formula does not apply",
                  source="limit")

        sizes = [40, 80, 160]
        ladder = refinement(
            lambda nn: abs(steady_conduction_1d(lambda t: np.where(t < t_s, k_a, k_b), t_hot, t_cold, n=nn)["q"]
                           - q_switch) / q_switch, sizes)
        s_orders = ladder["orders"]
        res.check("the switching interface converges at first order",
                  bool(s_orders) and all(0.8 < o < 1.2 for o in s_orders),
                  value=float(np.mean(s_orders)) if s_orders else None, limit=1.0,
                  detail=f"observed orders {', '.join(f'{o:.2f}' for o in s_orders)} at n = {sizes}: the one cell "
                         "around the interface is neither phase exactly", source="convergence")

        # ---------------------------------------------------------------- what a programmable k buys
        delta = 0.5
        q_plus = steady_conduction_1d(lambda t: k0 * (1.0 + beta * t), t_hot + delta, t_cold, n=n)["q"]
        res.value("dq_dt_hot", (q_plus - sol["q"]) / delta, "W/m^2/K",
                  "finite difference of the solved flux over 0.5 K")
        res.compare("sensitivity of the flux to the applied temperature", (q_plus - sol["q"]) / delta,
                    k0 * (1.0 + beta * t_hot), tol=1e-3, kind="closed_form", unit="W/m^2/K",
                    source="differentiating q = k0[(T1-T2) + (beta/2)(T1^2-T2^2)]/L gives dq/dT1 = k0(1+beta T1)/L")
        if beta < 0:
            res.value("temperature_where_k_vanishes", -1.0 / beta, "K", "the root of k0 (1 + beta T) = 0")
        else:
            res.notes.append("beta > 0: the conductivity grows with temperature and never vanishes, so this "
                             "material has no regulation point")

        # a falling conductivity is what a self-regulating envelope would want; the run looks for the point where
        # it stops responding, and finds it where it stops conducting
        beta_drop = -1.0 / (3.0 * t_hot)             # k vanishes at three times the hot temperature
        t_stop_drop = -1.0 / beta_drop
        reduced = 1.0 + beta_drop * t_hot
        res.value("sensitivity_with_a_falling_conductivity", reduced, "-",
                  "k0 (1 + beta T1)/k0 at beta = -1/(3 T1)")
        res.check("a falling conductivity reduces the sensitivity without reaching zero",
                  0.0 < reduced < 1.0, value=float(reduced), limit=0.0,
                  detail=(f"dq/dT1 falls to {reduced:.3f} of the constant-k value at T1, and would reach zero "
                          f"only at {t_stop_drop:.4g} K, where k itself vanishes"),
                  source="sensitivity")

        # at the vanishing point the solver must refuse rather than return a number
        refused = False
        try:
            steady_conduction_1d(lambda t: k0 * (1.0 + beta_drop * t), t_stop_drop + 10.0, t_cold, n=41)
        except ValueError:
            refused = True
        res.check("the solver refuses a conductivity that changes sign", refused, value=float(refused),
                  limit=1.0, detail="beyond T1 = -1/beta the material does not conduct and the run says so",
                  source="limit")

        res.data("profiles", x=sol["x"], T_variable=sol["T"], T_variable_exact=t_exact, T_graded=graded["T"],
                 T_switch=switch["T"])
        res.notes.append("the three laws are one-dimensional and steady: hysteresis, latent heat and the "
                         "mechanics of a real programmable material are outside what was solved")
        res.notes.append("the coefficients k0, beta, gamma are hypotheses, not measurements of any material: the "
                         "closed forms below validate the solver, not the existence of the material")
        res.notes.append("no technological feasibility is claimed: nothing here says such a material can be made")


# ---------------------------------------------------------------------------------------------- the proposal
def programmable_matter(beta: float = -1.0 / 1200.0) -> Proposal:
    """A material whose conductivity falls with temperature, and what that buys."""
    exp = ProgrammableMatter()
    return Proposal(
        title="A programmable thermal envelope",
        question="Can a material whose conductivity depends on the field regulate the heat that crosses it?",
        speculation=("A metamaterial whose conductivity is a designed function of the temperature, the position or "
                     "a transition, so that the flux through a wall becomes almost independent of the load."),
        scenario=exp.world(),
        hypotheses=(Hypothesis("a conductivity that falls with temperature lowers dq/dT1",
                               phenomenon="sensitivity of the flux",
                               falsified_by="a measured dq/dT1 that does not follow k0 (1 + beta T1)/L",
                               plausibility="established"),
                    Hypothesis("the regulation cannot reach zero while the material conducts",
                               phenomenon="vanishing conductivity",
                               falsified_by="a material that regulates at a temperature where k is still positive",
                               plausibility="hypothetical")),
        equations=("d/dx[k(T, x) dT/dx] = 0",
                   "q = k0 [(T1-T2) + (beta/2)(T1^2 - T2^2)]/L",
                   "dq/dT1 = k0 (1 + beta T1)/L"),
        phenomena=("non-linear conduction", "graded conductivity", "transition"),
        experiment=ProgrammableMatter.name)
