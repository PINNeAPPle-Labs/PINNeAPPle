"""Verification of the underground-storage simulator: four analytic references and two observed orders.

Every case is a number, not a claim, and each compares the simulator with something that was derived without it:

* :func:`theis_line_source` -- the drawdown around a well at constant rate in an infinite homogeneous reservoir,
  against the exponential-integral solution of Theis (1935), in the slightly compressible limit. The reservoir
  is a logarithmically spaced radial grid large enough that the pressure disturbance has not reached the outer
  boundary by the last time reported (the code returns that ratio, ``r_investigated / r_out``, so the reader
  can check the infinite-acting assumption instead of trusting it);
* :func:`pseudo_steady_state` -- in a closed reservoir the average pressure falls linearly once the whole
  volume drains, at ``q / (V phi ct)``; both the linearity and the slope are measured;
* :func:`material_balance` -- ``p/z`` against cumulative production is a straight line whose slope gives the
  gas in place (the oldest plot in reservoir engineering); the gas in place it implies is compared with the
  one the simulator integrates;
* :func:`mass_conservation` -- the accumulated mass-balance error of the cases above, which is the solver's
  own tolerance because the well terms enter the residual implicitly;
* :func:`time_convergence` and :func:`grid_convergence` -- the observed order of accuracy from a
  three-resolution refinement, using :func:`pinneapple_analysis.verification.convergence.mesh_independence_study`
  (Roache 1994 / Celik et al. 2008), on the error against the line-source solution.

    from pinneapple_simulation.ugs import verification
    report = verification.run()          # dict, ~15 s
    report = verification.run(quick=True)   # the same cases at a size a test can afford, ~2 s

The output of a run is a dict of dicts; :func:`summary` prints it as the paragraph a report quotes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import exp1

from pinneapple_analysis.verification.convergence import mesh_independence_study

from .grid import RadialGrid
from .pvt import Gas, Rock, SlightlyCompressible
from .simulator import SECONDS_PER_DAY, SimConfig, UGSSimulator
from .wells import Control, Period, Schedule, Well

__all__ = ["THEIS", "PSS", "MATERIAL_BALANCE", "theis_line_source", "pseudo_steady_state", "material_balance",
           "mass_conservation", "time_convergence", "grid_convergence", "run", "summary"]

# the line-source case: a 100 mD, 10 m thick, 20 % porosity radial model with a rock compressibility, so that
# the slightly compressible limit applies (Bg = 1, all the storage in the rock)
THEIS = {
    "permeability": 1.0e-13,       # m2
    "height": 10.0,                # m
    "porosity": 0.2,
    "cr": 1.0e-8,                  # 1/Pa
    "mu": 2.0e-5,                  # Pa s
    "p0": 1.0e7,                   # Pa
    "rate": 0.01,                  # m3/s (reservoir = surface here), withdrawn
    "r_w": 0.1,                    # m
    "r_out": 1.0e4,                # m
    "r_min": 1.0,                  # the norm starts here: the line source is singular at r = 0
}

PSS = {"permeability": 1.0e-13, "height": 10.0, "porosity": 0.2, "cr": 1.0e-8, "mu": 2.0e-5, "p0": 1.0e7,
       "rate": 0.01, "r_w": 0.1, "r_out": 1000.0}

MATERIAL_BALANCE = {"permeability": 1.0e-13, "height": 10.0, "porosity": 0.2, "cr": 1.0e-10, "T": 320.0,
                    "p0": 3.0e7, "rate": 0.65, "r_w": 0.1, "r_out": 100.0, "sg": 0.65}


def _theis_solve(n_r: int, dt_days: float, days: float, case: dict[str, float]):
    g = RadialGrid(n_r, r_w=case["r_w"], r_out=case["r_out"], height=case["height"],
                   permeability=case["permeability"], porosity=case["porosity"])
    sim = UGSSimulator(g, SlightlyCompressible(viscosity=case["mu"]), Rock(cr=case["cr"], p_ref=case["p0"]),
                       [Well("W", 0, r_w=case["r_w"])], p0=case["p0"])
    schedule = Schedule((Period(days, {"W": Control("rate", -case["rate"])}),))
    return g, sim, sim.run(schedule, SimConfig(dt_days=dt_days))


def theis_drawdown(radius, t_seconds: float, case: dict[str, float] | None = None):
    """Exact Theis drawdown [Pa] at ``radius`` and time: ``q mu / (4 pi k h) E1(r^2 / (4 eta t))``."""
    case = {**THEIS, **(case or {})}
    eta = case["permeability"] / (case["porosity"] * case["mu"] * case["cr"])
    return (case["rate"] * case["mu"] / (4.0 * np.pi * case["permeability"] * case["height"])
            * exp1(np.asarray(radius, dtype=float) ** 2 / (4.0 * eta * t_seconds)))


def theis_line_source(n_r: int = 60, dt_days: float = 0.25, days: float = 2.0,
                      case: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare the simulated drawdown with the line-source solution at the last time step."""
    case = {**THEIS, **(case or {})}
    g, sim, res = _theis_solve(n_r, dt_days, days, case)
    if not res.converged:
        return {"ok": False, "reason": res.reason}
    t = days * SECONDS_PER_DAY
    exact = theis_drawdown(g.centre, t, case)
    erred = np.abs(res.pressure[-1] - (case["p0"] - exact))
    mask = g.centre >= case["r_min"]
    scale = float(np.max(np.abs(exact[mask])))
    eta = case["permeability"] / (case["porosity"] * case["mu"] * case["cr"])
    radius = float(np.sqrt(4.0 * eta * t))
    return {
        "ok": True,
        "cells": n_r,
        "dt_days": dt_days,
        "days": days,
        "max_abs_error": float(np.max(erred[mask])),
        "relative_error": float(np.max(erred[mask]) / scale),
        "compared_drawdown": scale,
        "max_drawdown_at_the_well": float(np.max(exact)),
        "radius_of_influence": radius,
        "r_out": case["r_out"],
        "infinite_acting": bool(radius < 0.1 * case["r_out"]),
        "mass_balance_error": res.mass_balance_error,
    }


def pseudo_steady_state(n_r: int = 40, dt_days: float = 0.5, days: float = 15.0,
                        case: dict[str, float] | None = None) -> dict[str, Any]:
    """The closed reservoir drains linearly once it is in pseudo-steady state.

    Measures the coefficient of determination of the average pressure against time over the last 40 % of the
    run, and the fitted slope against the material balance slope ``-q / (V phi ct)``.
    """
    case = {**PSS, **(case or {})}
    g = RadialGrid(n_r, r_w=case["r_w"], r_out=case["r_out"], height=case["height"],
                   permeability=case["permeability"], porosity=case["porosity"])
    sim = UGSSimulator(g, SlightlyCompressible(viscosity=case["mu"]), Rock(cr=case["cr"], p_ref=case["p0"]),
                       [Well("W", 0, r_w=case["r_w"])], p0=case["p0"])
    schedule = Schedule((Period(days, {"W": Control("rate", -case["rate"])}),))
    res = sim.run(schedule, SimConfig(dt_days=dt_days))
    if not res.converged:
        return {"ok": False, "reason": res.reason}
    t = res.times_days * SECONDS_PER_DAY
    pbar = res.mean_pressure
    start = int(0.6 * res.n_steps)
    slope, intercept = np.polyfit(t[start:] - t[start], pbar[start:], 1)
    fitted = slope * (t[start:] - t[start]) + intercept
    ss_res = float(np.sum((pbar[start:] - fitted) ** 2))
    ss_tot = float(np.sum((pbar[start:] - pbar[start:].mean()) ** 2))
    analytic = -case["rate"] / (g.pore_volume() * case["cr"])
    return {
        "ok": True,
        "days": days,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "slope_fitted": float(slope),
        "slope_analytic": float(analytic),
        "slope_relative_error": float(abs(slope - analytic) / abs(analytic)),
        "mass_balance_error": res.mass_balance_error,
    }


def material_balance(n_r: int = 40, dt_days: float = 2.0, days: float = 120.0,
                     case: dict[str, float] | None = None) -> dict[str, Any]:
    """The ``p/z`` line of a closed gas reservoir, and the gas in place it implies.

    ``p/z`` is plotted against cumulative production; a straight line of slope ``-(p_i/z_i)/G`` is the
    material balance, so the gas in place it gives is compared with the one the simulator integrates.
    """
    case = {**MATERIAL_BALANCE, **(case or {})}
    g = RadialGrid(n_r, r_w=case["r_w"], r_out=case["r_out"], height=case["height"],
                   permeability=case["permeability"], porosity=case["porosity"])
    fluid = Gas(T=case["T"], sg=case["sg"])
    sim = UGSSimulator(g, fluid, Rock(cr=case["cr"], p_ref=case["p0"]),
                       [Well("W", 0, r_w=case["r_w"], bhp_min=1.0e6)], p0=case["p0"])
    schedule = Schedule((Period(days, {"W": Control("rate", -case["rate"])}),))
    res = sim.run(schedule, SimConfig(dt_days=dt_days))
    if not res.converged:
        return {"ok": False, "reason": res.reason}
    pz = res.mean_pressure / fluid.z(res.mean_pressure)
    gp = -res.well_cumulative["W"]
    slope, intercept = np.polyfit(gp, pz, 1)
    fitted = slope * gp + intercept
    ss_res = float(np.sum((pz - fitted) ** 2))
    ss_tot = float(np.sum((pz - pz.mean()) ** 2))
    pz_initial = float(case["p0"] / fluid.z(case["p0"]))
    if slope >= 0.0:
        return {"ok": False, "reason": f"p/z did not fall with production (slope {slope:.3e})"}
    gas_in_place_fit = -pz_initial / slope
    gip = float(res.gas_in_place[0])
    return {
        "ok": True,
        "days": days,
        "fraction_produced": float(gp[-1] / gip),
        "r2": 1.0 - ss_res / ss_tot,
        "intercept_relative_error": float(abs(intercept - pz_initial) / pz_initial),
        "gas_in_place_fit": float(gas_in_place_fit),
        "gas_in_place_simulated": gip,
        "gas_in_place_relative_error": float(abs(gas_in_place_fit - gip) / gip),
        "mass_balance_error": res.mass_balance_error,
    }


def mass_conservation(cases: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The worst mass-balance error of the cases, measured as (GIP_end - GIP_0 - moved) / GIP_0."""
    cases = cases if cases is not None else run(quick=True)
    errors = {name: float(case["mass_balance_error"]) for name, case in cases.items()
              if isinstance(case, dict) and "mass_balance_error" in case}
    if not errors:
        return {"ok": False, "reason": "no case with a mass balance to read"}
    worst = max(errors, key=lambda k: abs(errors[k]))
    return {"ok": abs(errors[worst]) < 1e-8, "worst_case": worst, "errors": errors,
            "worst_abs": float(abs(errors[worst]))}


def time_convergence(n_r: int = 200, steps_per_day=(2, 4, 8), days: float = 2.0,
                     case: dict[str, float] | None = None) -> dict[str, Any]:
    """Observed order of the time integration, from the error against the line source at three step sizes."""
    case = {**THEIS, **(case or {})}

    def solve(steps: float) -> float:
        dt_days = 1.0 / steps
        g, _sim, res = _theis_solve(n_r, dt_days, days, case)
        if not res.converged:
            raise RuntimeError(f"the line-source case did not converge at {dt_days:g} d: {res.reason}")
        mask = g.centre >= case["r_min"]
        exact = theis_drawdown(g.centre[mask], days * SECONDS_PER_DAY, case)
        return float(np.max(np.abs(res.pressure[-1][mask] - (case["p0"] - exact))))

    study = mesh_independence_study(solve, list(steps_per_day), lambda e: e)
    return {
        "ok": True,
        "cells": n_r,
        "steps_per_day": list(steps_per_day),
        "errors": [float(v) for v in study.values],
        "observed_order": float(study.convergence.observed_order),
        "nominal_order": 1.0,
        "asymptotic": bool(study.convergence.is_asymptotic),
        "extrapolated_error": float(study.convergence.extrapolated_value),
    }


def grid_convergence(cells=(20, 40, 80), dt_days: float = 1.0e-3, days: float = 0.5,
                     case: dict[str, float] | None = None) -> dict[str, Any]:
    """Observed order of the spatial discretisation, from the error against the line source at three grids.

    ``dt_days`` is small enough that the first-order time error stays below the spatial one at the finest grid,
    which the returned ``time_error`` (the error at the finest grid with a coarser step) lets the reader check.
    """
    case = {**THEIS, **(case or {})}

    def solve(n_r: float) -> float:
        g, _sim, res = _theis_solve(int(n_r), dt_days, days, case)
        if not res.converged:
            raise RuntimeError(f"the line-source case did not converge on {n_r} cells: {res.reason}")
        mask = g.centre >= case["r_min"]
        exact = theis_drawdown(g.centre[mask], days * SECONDS_PER_DAY, case)
        return float(np.max(np.abs(res.pressure[-1][mask] - (case["p0"] - exact))))

    study = mesh_independence_study(solve, list(cells), lambda e: e)
    # the time error is a property of the step, not of the grid: measure it on the coarsest grid, where it is
    # cheap, so that the reader can see it sits below the spatial error of the finest one
    coarse = float(cells[0])
    time_error = abs(_time_only_error(coarse, dt_days * 2.0, days, case)
                     - _time_only_error(coarse, dt_days, days, case))
    return {
        "ok": True,
        "cells": list(cells),
        "dt_days": dt_days,
        "days": days,
        "errors": [float(v) for v in study.values],
        "observed_order": float(study.convergence.observed_order),
        "nominal_order": 2.0,
        "asymptotic": bool(study.convergence.is_asymptotic),
        "extrapolated_error": float(study.convergence.extrapolated_value),
        "finest_error": float(study.values[-1]),
        "time_error_on_the_coarsest_grid": float(time_error),
    }


def _time_only_error(n_r: int, dt_days: float, days: float, case: dict[str, float]) -> float:
    g, _sim, res = _theis_solve(n_r, dt_days, days, case)
    mask = g.centre >= case["r_min"]
    exact = theis_drawdown(g.centre[mask], days * SECONDS_PER_DAY, case)
    return float(np.max(np.abs(res.pressure[-1][mask] - (case["p0"] - exact))))


def run(quick: bool = False) -> dict[str, Any]:
    """Run every case. ``quick`` shrinks the resolutions to what a unit test can afford."""
    if quick:
        report = {
            "theis": theis_line_source(n_r=40, dt_days=0.25, days=1.0),
            "pseudo_steady_state": pseudo_steady_state(n_r=30, dt_days=0.5, days=15.0),
            "material_balance": material_balance(n_r=30, dt_days=2.0, days=120.0),
            "time_convergence": time_convergence(n_r=100, steps_per_day=(2, 4, 8), days=1.0),
            "grid_convergence": grid_convergence(cells=(10, 20, 40), dt_days=5.0e-3, days=0.5),
        }
    else:
        report = {
            "theis": theis_line_source(),
            "pseudo_steady_state": pseudo_steady_state(),
            "material_balance": material_balance(),
            "time_convergence": time_convergence(),
            "grid_convergence": grid_convergence(),
        }
    report["mass_conservation"] = mass_conservation(report)
    return report


def summary(report: dict[str, Any]) -> str:
    """One paragraph with the numbers the report quotes, and the tolerances they are held to."""
    theis = report["theis"]
    pss = report["pseudo_steady_state"]
    mb = report["material_balance"]
    tc = report["time_convergence"]
    gc = report["grid_convergence"]
    mc = report["mass_conservation"]
    return (
        f"line source: {theis['relative_error']:.2e} of the drawdown at {theis['days']:g} d "
        f"(radius of influence {theis['radius_of_influence']:.0f} m, outer boundary {theis['r_out']:.0f} m); "
        f"pseudo-steady state: R2 {pss['r2']:.10f}, slope {pss['slope_relative_error']:.1e} from q/(V phi ct); "
        f"material balance: p/z R2 {mb['r2']:.10f}, gas in place {mb['gas_in_place_relative_error']:.1e} off; "
        f"observed orders {tc['observed_order']:.2f} in time and {gc['observed_order']:.2f} in space; "
        f"worst mass-balance error {mc['worst_abs']:.1e}"
    )
