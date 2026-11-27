"""Planetary defence, measured: the momentum DART delivered, what a nudge does to the 2029 Apophis flyby, and how
far ahead the miss can be guaranteed (#381).

    python examples/planetary_defence/apophis_deflection.py             # the full study, ~15 min on 4 cores
    python examples/planetary_defence/apophis_deflection.py --quick     # the DART anchor and one deflection

Three acts, each with the number it produces:

1. **The impulse, held to the mission that flew it.** DART shortened Dimorphos' orbital period around Didymos by
   32 minutes with 570 kg at 6.1 km/s. The momentum model is asked for the enhancement factor ``beta`` that explains
   it; it answers 3.65 against the published 3.61. Every ``delta_v`` below uses that ``beta``.
2. **The nudge.** Apophis passes 38 356 km from the Earth on 2029-04-13 (7.41 km/s relative). A kinetic impactor
   fired five years earlier changes that miss: the table is the miss distance against the impactor's mass and against
   the warning time, from the propagator this repository ships (Sun, planets, Moon, the Sun's relativistic term).
3. **What can be guaranteed.** ``beta`` and the Yarkovsky ``A2`` are not known exactly, so the answer is an ensemble.
   The fraction of it that clears the Earth is the success probability; the warning time at which the ensemble spread
   grows past the Earth's radius is the trust horizon -- after it, "we will miss" is not a guarantee.

Figures: ``figures/miss_vs_mass.png``, ``figures/trust_horizon.png``. Numbers: ``apophis_deflection.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
APOPHIS_MASS_KG = 5.4e10                   # 340 m at 2600 kg/m3
EARTH_RADIUS_KM = 6371.0


def _out(name: str) -> str:
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path) or OUT, exist_ok=True)
    return path


def dart_anchor() -> dict:
    """The momentum enhancement the published DART period change implies -- the calibration of everything below."""
    from pinneapple_physics.astro.defence import DART, dart_beta, delta_v, dimorphos_period_hours, \
        period_change_fraction

    beta = dart_beta()
    dv = delta_v(DART["impactor_kg"], DART["speed_km_s"], DART["target_kg"], beta)
    implied = period_change_fraction(-dv, 0.174) * DART["period_before_s"] / 60.0
    return {"beta_from_period_change": beta, "beta_published": 3.61,
            "period_before_hours": DART["period_before_s"] / 3600.0,
            "period_before_hours_from_orbital_speed": dimorphos_period_hours(),
            "period_change_minutes_implied": implied, "period_change_minutes_published": -32.0,
            "delta_v_mm_s": dv * 1000.0}


def deflection_study(masses, warnings_years, quick: bool = False, target_kg: float = APOPHIS_MASS_KG) -> dict:
    """Miss distance at the 2029 encounter against impactor mass and warning time.

    ``target_kg`` is the mass that absorbs the momentum: Apophis (5.4e10 kg) by default, or a smaller body on the
    same orbit -- the what-if that shows the regime a real mission works in.
    """
    from pinneapple_physics.astro.defence import (APOPHIS_2029_JD, BETA_DART, deflect, delta_v, miss_distance)
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    base = miss_distance(body, APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30, step_days=0.5)
    rows = []
    for years in warnings_years:
        jd_impact = APOPHIS_2029_JD - years * 365.25
        for m in masses:
            t0 = time.time()
            dv = delta_v(m, 6.1, target_kg, BETA_DART)
            ca = miss_distance(deflect(body, jd_impact, dv), APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30,
                               step_days=0.5)
            rows.append({"impactor_kg": m, "warning_years": years, "delta_v_mm_s": dv * 1000.0,
                         "miss_km": ca["dist_km"], "jd_tdb": ca["jd_tdb"], "seconds": time.time() - t0})
            print(f"  {m:8.0f} kg, {years:2.0f} yr ahead: dv = {dv * 1000:.4f} mm/s -> miss {ca['dist_km']:,.0f} km"
                  f"  ({rows[-1]['seconds']:.0f} s)", flush=True)
            if quick:
                return {"baseline": base, "rows": rows}
    return {"baseline": base, "rows": rows}


def uncertainty_study(masses_kg=(1000.0,), warnings_years=(5.0,), n: int = 8, seed: int = 0,
                      beta_sigma: float = 0.35, a2_sigma: float = 3e-15) -> dict:
    """The same deflection over an ensemble of ``beta`` and Yarkovsky ``A2``: success fraction and spread."""
    from pinneapple_physics.astro.defence import (APOPHIS_2029_JD, BETA_DART, deflect, delta_v, miss_distance,
                                                  sample_ensemble, success_probability, trust_horizon)
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    draws = sample_ensemble(n=n, a2_au_d2=-2.9e-14, a2_sigma=a2_sigma, beta=BETA_DART, beta_sigma=beta_sigma,
                            seed=seed)
    out = {"draws": draws, "rows": []}
    for years in warnings_years:
        jd_impact = APOPHIS_2029_JD - years * 365.25
        misses = []
        for d in draws:
            dv = delta_v(masses_kg[0], 6.1, APOPHIS_MASS_KG, d["beta"])
            ca = miss_distance(deflect(body, jd_impact, dv), APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30,
                               step_days=0.5)
            misses.append(ca["dist_km"])
        row = {"warning_years": years, "misses_km": misses, "spread_km": float(np.std(misses)),
               "median_km": float(np.median(misses)),
               "success": success_probability(misses, earth_radius_km=EARTH_RADIUS_KM)}
        out["rows"].append(row)
        print(f"  ensemble at {years:2.0f} yr: median {row['median_km']:,.0f} km, spread {row['spread_km']:,.0f} km,"
              f" success {row['success']:.2f}", flush=True)
    if len(out["rows"]) > 1:
        out["trust_horizon_years"] = trust_horizon(
            warning_days=[r["warning_years"] * 365.25 for r in out["rows"]],
            spread_km=[r["spread_km"] for r in out["rows"]], earth_radius_km=EARTH_RADIUS_KM) / 365.25
    return out


def figures(result: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    rows = result["deflection"]["rows"]
    base = result["deflection"]["baseline"]["dist_km"]
    if len({r["warning_years"] for r in rows}) > 1 or len(rows) > 1:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for years in sorted({r["warning_years"] for r in rows}):
            sel = [r for r in rows if r["warning_years"] == years]
            ax.loglog([r["impactor_kg"] for r in sel], [r["miss_km"] / base for r in sel], "o-",
                      label=f"{years:.0f} years of warning")
        ax.axhline(1.0, color="0.5", ls=":", lw=1, label="no deflection")
        ax.axhline(EARTH_RADIUS_KM / base, color="tab:red", ls="--", lw=1, label="Earth's radius")
        ax.set_xlabel("impactor mass (kg)")
        ax.set_ylabel("miss distance / undeflected miss")
        ax.set_title("Apophis, 2029-04-13: what a kinetic impactor does to the miss")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        path = _out("figures/miss_vs_mass.png")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        made.append(path)

    ens = result.get("uncertainty")
    if ens and len(ens["rows"]) > 1:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        years = [r["warning_years"] for r in ens["rows"]]
        ax.plot(years, [r["spread_km"] for r in ens["rows"]], "o-", label="ensemble spread (1 sigma)")
        ax.axhline(EARTH_RADIUS_KM, color="tab:red", ls="--", lw=1, label="Earth's radius")
        if "trust_horizon_years" in ens:
            ax.axvline(ens["trust_horizon_years"], color="0.4", ls=":", lw=1,
                       label=f"trust horizon: {ens['trust_horizon_years']:.1f} yr")
        ax.set_xlabel("warning time (years)")
        ax.set_ylabel("spread of the miss distance (km)")
        ax.set_title("How far ahead the miss can be guaranteed")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        path = _out("figures/trust_horizon.png")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        made.append(path)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="the DART anchor and a single deflection")
    ap.add_argument("--out", default=_out("apophis_deflection.json"))
    args = ap.parse_args(argv)

    t0 = time.time()
    anchor = dart_anchor()
    print(f"DART: beta = {anchor['beta_from_period_change']:.2f} (published {anchor['beta_published']}), "
          f"period change {anchor['period_change_minutes_implied']:.1f} min "
          f"(published {anchor['period_change_minutes_published']:.0f})", flush=True)

    masses = (100.0, 1000.0) if args.quick else (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
    years = (5.0,) if args.quick else (1.0, 5.0, 10.0)
    print("deflection study:", flush=True)
    deflection = deflection_study(masses, years, quick=args.quick)
    print(f"undeflected miss: {deflection['baseline']['dist_km']:,.0f} km at JD {deflection['baseline']['jd_tdb']:.2f}",
          flush=True)

    small = None
    if not args.quick:
        print("what-if: a 50 m body (1.3e8 kg) on the same orbit:", flush=True)
        small = deflection_study((10.0, 100.0, 1000.0), (5.0,), target_kg=1.3e8)

    uncertainty = None
    if not args.quick:
        print("uncertainty ensemble:", flush=True)
        uncertainty = uncertainty_study(masses_kg=(1000.0,), warnings_years=(1.0, 5.0, 10.0), n=6)

    result = {"issue": 381, "body": "99942 Apophis", "target": "Earth", "encounter_jd_tdb": deflection["baseline"]
              ["jd_tdb"], "apophis_mass_kg": APOPHIS_MASS_KG, "earth_radius_km": EARTH_RADIUS_KM,
              "dart": anchor, "deflection": deflection, "what_if_50m": small, "uncertainty": uncertainty,
              "seconds": time.time() - t0}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    for path in figures(result):
        print(f"figure: {path}")
    print(f"wrote {args.out} in {result['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
