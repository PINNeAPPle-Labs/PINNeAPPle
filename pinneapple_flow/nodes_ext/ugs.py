"""Underground gas storage as a flow step: one storage season on the reference simulator.

The node wraps :mod:`pinneapple_simulation.ugs` (the simulator of #1083) the way the other ``nodes_ext`` modules
wrap a library study: the parameters are the season an operator would type in -- the injection and withdrawal
bottom-hole pressures, the durations and how many wells are shut -- and the node reports what the season
delivered, the peak day first, with the measured mass-balance error beside it as the check that the numbers on
the panel are a solution and not a guess.
"""
from __future__ import annotations

import numpy as np

from ..registry import Param, Port, node


def _num(v):
    """A metric a report can print: a float, or the word the run used."""
    if v is None:
        return "n/a"
    if isinstance(v, (bool, str)):
        return v
    return float(v)


@node("ugs.storage_season", "Underground gas storage: a season, its peak day and its balance",
      category="Subsurface",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("injection_bhp", "float", 2.0e7, minimum=1.0e6, help="injection bottom-hole pressure (Pa)"),
              Param("withdrawal_bhp", "float", 7.0e6, minimum=1.0e6, help="withdrawal bottom-hole pressure (Pa)"),
              Param("days_injection", "float", 12.0, minimum=1.0, help="length of the injection period (d)"),
              Param("days_withdrawal", "float", 10.0, minimum=1.0, help="length of the withdrawal period (d)"),
              Param("wells_shut", "int", 2, minimum=0, maximum=9, help="wells left closed for the season"),
              Param("seed", "int", 0, help="seed of the permeability field")],
      calls="pinneapple_simulation.ugs.synthetic_model + UGSSimulator.run + Schedule + Control")
def storage_season(ctx, inputs, params):
    """Run one season on the synthetic nine-well model: average pressure, gas in place, the peak daily deliverability,
    the rate and bottom-hole pressure of every well on that day, and the mass-balance error the implicit solve
    measured. The figure shows the pressure field at the end of the withdrawal and the deliverability per day."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_simulation.ugs import Control, Period, Schedule, SimConfig, synthetic_model

    model = synthetic_model(seed=int(params["seed"]))
    names = [w.name for w in model.wells]
    shut = set(names[: max(0, min(int(params["wells_shut"]), len(names)))])
    injection = {n: Control("bhp", float(params["injection_bhp"])) for n in names if n not in shut}
    withdrawal = {n: Control("bhp", float(params["withdrawal_bhp"])) for n in names if n not in shut}
    days_injection = float(params["days_injection"])
    schedule = Schedule((Period(days_injection, injection, "injection"),
                         Period(2.0, {}, "shut-in"),
                         Period(float(params["days_withdrawal"]), withdrawal, "withdrawal")))
    result = model.simulator().run(schedule, SimConfig(dt_days=1.0))

    for name, value in (("converged", result.converged), ("mass_balance_error", _num(result.mass_balance_error)),
                        ("mean_pressure_start_MPa", result.mean_pressure[0] / 1e6),
                        ("mean_pressure_end_MPa", result.mean_pressure[-1] / 1e6),
                        ("gas_in_place_start_Mm3", result.gas_in_place[0] / 1e6),
                        ("gas_in_place_end_Mm3", result.gas_in_place[-1] / 1e6),
                        ("peak_deliverability_m3_per_day", float(result.deliverability().max())),
                        ("wells_open", len(names) - len(shut))):
        ctx.metric(name, value)
    if not result.converged:
        ctx.summary(f"the season did not converge: {result.reason}")
        return {"metrics": ctx.report["metrics"]}

    peak = int(np.argmax(result.deliverability()))
    ctx.table("wells on the peak day", ["well", "rate (m3/d)", "BHP (MPa)", "cumulative (Mm3)"],
              [[n, round(float(-result.well_rate[n][peak]), 1) + 0.0,
                round(float(result.well_bhp[n][peak]) / 1e6, 3),
                round(float(-result.well_cumulative[n][-1]) / 1e6, 4) + 0.0] for n in names])
    ctx.summary(f"season of {schedule.days:g} d: average pressure {result.mean_pressure[0] / 1e6:.2f} -> "
                f"{result.mean_pressure[-1] / 1e6:.2f} MPa, peak deliverability "
                f"{result.deliverability().max():,.0f} m3/d on day {result.times_days[peak]:g}, "
                f"gas in place {result.gas_in_place[0] / 1e6:.1f} -> {result.gas_in_place[-1] / 1e6:.1f} Mm3, "
                f"mass balance error {result.mass_balance_error:.1e}")

    shape = model.grid.shape
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    field = result.pressure[-1].reshape(shape[:2]) / 1e6
    image = ax[0].imshow(field.T, origin="lower", cmap="viridis", aspect="equal")
    fig.colorbar(image, ax=ax[0], label="pressure (MPa)")
    ax[0].set(title=f"pressure after the withdrawal ({schedule.days:g} d)", xlabel="x cell", ylabel="y cell")
    ax[1].plot(result.times_days, result.deliverability(), "C3", lw=2)
    ax[1].axvline(days_injection, color="0.6", ls=":", label="injection ends")
    ax[1].axvline(days_injection + 2.0, color="0.4", ls="--", label="withdrawal starts")
    ax[1].set(xlabel="time (d)", ylabel="deliverability (m3/day)", title="daily deliverability")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    ctx.figure(fig, "ugs_season")
    return {"metrics": ctx.report["metrics"]}
