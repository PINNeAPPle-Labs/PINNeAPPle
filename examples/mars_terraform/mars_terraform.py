"""Mars over centuries: the caps retreat, the pressure climbs, the confidence bar runs out, and the verdict.

Four measurements, all from :mod:`pinneapple_physics.terraform`:

* **the scenario** -- an engineered greenhouse of 3 W m^-2 and 0.06 of dust on the caps, ramped in over 150 years
  and held for 350 more: the polar cap retreats from the whole cap area to a quarter of it, the CO2 pressure climbs
  from 6.1 to 12.8 mbar, and the planet warms by 6 K;
* **the ensemble** -- sixty-four planets with the frost albedo, the poleward diffusivity, the regolith reservoir, the
  CO2 greenhouse and the strength of the engineering all drawn from their declared ranges: the spread on the global
  mean reaches the declared 2 K after :func:`pinneapple_physics.terraform.trust_horizon` years;
* **the verdict** -- every published CO2 inventory, put into the air at once and left to settle: 20 mbar (Jakosky &
  Edwards' representative maximum) warms Mars by 3.8 K, and the only reservoir that reaches the melting point is the
  ~1 bar of deep carbonates their figure 3 lists as inaccessible;
* **the picture** -- the globe, with the caps shrinking and a bar that says how much of the model's answer is still
  a measurement.

Run::

    python examples/mars_terraform/mars_terraform.py           # everything, about a minute with the GIF
    python examples/mars_terraform/mars_terraform.py --quick   # no GIF, 32 ensemble members
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pinneapple_physics.terraform import co2, ebm, globe
from pinneapple_physics.terraform import terraforming as tf

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"

#: The published inventories, in the order the guide reports them.
INVENTORY_ORDER = ("today", "mobilizable", "accessible_top", "everything", "deep_carbonates")


def verdict_table(years: float = 1500.0) -> list[dict[str, object]]:
    """Every reservoir in the air at once, settled with no engineering at all: what CO2 does by itself."""
    rows = []
    for name in INVENTORY_ORDER:
        total = co2.inventory(name)
        model = tf.TerraformModel(ebm.Climate(), ebm.Grid(72), atmosphere_pa=total, cap_pa=0.0, regolith_pa=0.0)
        run = model.run(tf.Scenario(years=years, ramp_years=1e9))       # ramp_years = inf: no engineering
        rows.append({
            "inventory": name,
            "inventory_mbar": total / 100.0,
            "pressure_mbar": float(run.pressure_pa[-1, 0]) / 100.0,
            "mean_temperature_k": float(run.mean_temperature[-1, 0]),
            "greenhouse_k": float(ebm.greenhouse_warming(total)),
        })
    base = rows[0]["mean_temperature_k"]
    for row in rows:
        row["warming_k"] = row["mean_temperature_k"] - base
    return rows


def scenario_track(model: tf.TerraformModel, scenario: tf.Scenario) -> dict[str, list[float]]:
    """The central run, sampled once a year, as the lists a results file wants."""
    run = model.run(scenario)
    return {"years": run.years.tolist(),
            "mean_temperature_k": run.mean_temperature[:, 0].tolist(),
            "pressure_mbar": (run.pressure_pa[:, 0] / 100.0).tolist(),
            "cap_mbar": (run.cap_pa[:, 0] / 100.0).tolist(),
            "cap_cover": run.cover[:, 0].tolist(),
            "cap_edge_deg": globe.cap_edge_latitude(run.cover[:, 0]).tolist()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="no GIF, 32 ensemble members, 24 frames")
    parser.add_argument("--members", type=int, default=None)
    args = parser.parse_args(argv)
    members = args.members or (32 if args.quick else 64)
    started = time.time()
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    scenario = tf.TERRAFORMING_SCENARIO
    model = tf.TerraformModel()
    track = scenario_track(model, scenario)
    print(f"scenario: pressure {track['pressure_mbar'][0]:.2f} -> {track['pressure_mbar'][-1]:.2f} mbar, "
          f"mean {track['mean_temperature_k'][0]:.1f} -> {track['mean_temperature_k'][-1]:.1f} K, "
          f"cap {track['cap_mbar'][0]:.2f} -> {track['cap_mbar'][-1]:.2f} mbar")

    ensemble = tf.run_ensemble(scenario=scenario, members=members, seed=0)
    horizon = ensemble.horizon()
    spread = ensemble.spread
    print(f"ensemble of {members}: spread {spread[0]:.2f} -> {spread[-1]:.2f} K, trust horizon {horizon:.0f} years")

    rows = verdict_table()
    for row in rows:
        print(f"  {row['inventory']:16s} {row['inventory_mbar']:9.2f} mbar -> {row['mean_temperature_k']:6.1f} K "
              f"(+{row['warming_k']:4.1f} K)")

    run = model.run(scenario)
    figure = globe.century_figure(str(FIGURES / "mars_centuries.png"), run.years,
                                  ensemble.trajectory.mean_temperature, run.pressure_pa[:, 0], run.cover[:, 0],
                                  spread, threshold=tf.SIGMA_THRESHOLD_K, horizon=horizon)
    print(f"wrote {figure}")

    gif = None
    if not args.quick:
        gif = globe.mars_gif(str(FIGURES / "mars_globe.gif"), run.years, run.temperature[:, 0], run.cover[:, 0],
                             run.pressure_pa[:, 0], spread, run.latitude_deg,
                             threshold=tf.SIGMA_THRESHOLD_K, frames=48)
        print(f"wrote {gif}")

    payload = {
        "scenario": scenario.to_dict(),
        "track": track,
        "ensemble": {
            "members": members,
            "seed": 0,
            "space": tf.ParameterSpace().to_dict(),
            "spread_k": spread.tolist(),
            "warming_k": ensemble.warming.tolist(),
            "horizon_years": horizon,
            "threshold_k": float(tf.SIGMA_THRESHOLD_K),
            "final_pressure_mbar": (ensemble.trajectory.pressure_pa[-1] / 100.0).tolist(),
            "final_temperature_k": ensemble.trajectory.mean_temperature[-1].tolist(),
        },
        "verdict": rows,
        "model": {
            "bands": model.grid.bands,
            "climate": model.climate.to_dict(),
            "inventory_mbar": float(model.total_pa[0]) / 100.0,
            "cap_mbar": model.cap_full_pa / 100.0,
            "frost_point_6mbar_k": float(co2.frost_point(636.0)),
            "greenhouse_6mbar_k": float(ebm.greenhouse_warming(636.0)),
            "greenhouse_20mbar_k": float(ebm.greenhouse_warming(2000.0)),
            "greenhouse_1bar_k": float(ebm.greenhouse_warming(1e5)),
            "assumptions": list(ebm.ASSUMPTIONS),
        },
        "seconds": round(time.time() - started, 1),
    }
    out = RESULTS / "mars_terraform.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} in {payload['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
