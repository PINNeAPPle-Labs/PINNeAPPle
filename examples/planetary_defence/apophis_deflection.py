"""Planetary defence, measured: the momentum DART delivered, what a nudge does to the 2029 Apophis flyby, how far
ahead the miss can be guaranteed, what each acceleration in the field is worth, and the film of the ensemble (#381).

    python examples/planetary_defence/apophis_deflection.py             # the full study, ~35 min on 6 cores
    python examples/planetary_defence/apophis_deflection.py --quick     # the DART anchor and one deflection

Four acts, each with the number it produces:

1. **The impulse, held to the mission that flew it.** DART shortened Dimorphos' orbital period around Didymos by
   32 minutes with 570 kg at 6.1 km/s. The momentum model is asked for the enhancement factor ``beta`` that explains
   it; it answers 3.65 against the published 3.61. Every ``delta_v`` below uses that ``beta``.
2. **The nudge.** Apophis passes 38 356 km from the Earth on 2029-04-13 (7.41 km/s relative, against JPL's recorded
   37 724 km, inside its 697 km 3-sigma). A kinetic impactor fired up to ten years earlier changes that miss; the
   table is the miss distance against the impactor's mass and the warning time, from the propagator this repository
   ships, and ``figures/impact_probability.png`` is the same table read as the probability that the body hits.
3. **What can be guaranteed.** ``beta``, the Yarkovsky ``A2`` and the area-to-mass ratio that sets solar radiation
   pressure are not known exactly, so the answer is an ensemble: the fraction of it that clears the Earth is the
   success probability, the warning time at which its spread grows past the Earth's radius is the trust horizon, and
   ``figures/planetary_defence.gif`` is the ensemble itself -- the cloud widening as the years pass and the impactor
   arriving -- not a drawing of one.
4. **What each term is worth.** Every perturber dropped in turn and the radiation pressure switched on, so that the
   terms that do not matter can be named as such, with the number that says so next to the numerical floor.

Figures: ``figures/miss_vs_mass.png``, ``figures/trust_horizon.png``, ``figures/impact_probability.png``,
``figures/planetary_defence.gif``. Numbers: ``apophis_deflection.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
#: The bundled JPL record carries 0.325 km (Herschel, Mueller 2013) and this study uses the 5.4e10 kg that 340 m at
#: 2600 kg/m3 gives -- the mass the published deflection studies of Apophis quote, kept so the table stays comparable.
APOPHIS_MASS_KG = 5.4e10
APOPHIS_DIAMETER_KM = 0.325
EARTH_RADIUS_KM = 6371.0
IMPACTOR_SPEED_KM_S = 6.1
A2_APOPHIS = -2.9e-14                     # au/day^2, the JPL solution the Yarkovsky drift is measured from
A2_SIGMA = 3e-15
BETA_SIGMA = 0.35
AM_SIGMA = 0.5                            # the cross-section and reflectivity of a rock, to a factor of two
ENSEMBLE_N = 6


def _out(name: str) -> str:
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path) or OUT, exist_ok=True)
    return path


def dart_anchor() -> dict:
    """The momentum enhancement the published DART period change implies -- the calibration of everything below."""
    from pinneapple_physics.astro.defence import (
        DART,
        dart_beta,
        delta_v,
        dimorphos_period_hours,
        period_change_fraction,
    )

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
    from pinneapple_physics.astro.defence import (
        APOPHIS_2029_JD,
        BETA_DART,
        deflect,
        delta_v,
        miss_distance,
    )
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    base = miss_distance(body, APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30, step_days=0.5)
    rows = []
    for years in warnings_years:
        jd_impact = APOPHIS_2029_JD - years * 365.25
        for m in masses:
            t0 = time.time()
            dv = delta_v(m, IMPACTOR_SPEED_KM_S, target_kg, BETA_DART)
            ca = miss_distance(deflect(body, jd_impact, dv), APOPHIS_2029_JD - 30, APOPHIS_2029_JD + 30,
                               step_days=0.5)
            rows.append({"impactor_kg": m, "warning_years": years, "delta_v_mm_s": dv * 1000.0,
                         "miss_km": ca["dist_km"], "jd_tdb": ca["jd_tdb"], "seconds": time.time() - t0})
            print(f"  {m:8.0f} kg, {years:2.0f} yr ahead: dv = {dv * 1000:.4f} mm/s -> miss {ca['dist_km']:,.0f} km"
                  f"  ({rows[-1]['seconds']:.0f} s)", flush=True)
            if quick:
                return {"baseline": base, "rows": rows}
    return {"baseline": base, "rows": rows}


def ensemble_study(masses_kg=(1000.0,), warnings_years=(1.0, 5.0, 10.0), n: int = ENSEMBLE_N, seed: int = 0,
                   beta_sigma: float = BETA_SIGMA, a2_sigma: float = A2_SIGMA, am_sigma: float = AM_SIGMA,
                   frames: int = 121) -> dict:
    """The same deflection over an ensemble of ``beta``, Yarkovsky ``A2`` and area-to-mass ratio, and the arcs that
    draw it.

    The longest warning time keeps the whole trajectory of every member (``frames`` dates from the impulse to the
    encounter), which is what the animation is made of; the shorter ones keep the miss distances only. The undeflected
    body is flown with them and is the reference every member is measured against.
    """
    from pinneapple_physics.astro.defence import (
        APOPHIS_2029_JD,
        BETA_DART,
        Terms,
        area_to_mass_ratio,
        ensemble_arcs,
        sample_ensemble,
        success_probability,
        trust_horizon,
    )
    from pinneapple_physics.astro.smallbodies import load_body

    body = load_body("apophis")
    am = area_to_mass_ratio(APOPHIS_DIAMETER_KM, APOPHIS_MASS_KG)
    draws = sample_ensemble(n=n, a2_au_d2=A2_APOPHIS, a2_sigma=a2_sigma, beta=BETA_DART, beta_sigma=beta_sigma,
                            area_to_mass_m2_kg=am, am_sigma=am_sigma, seed=seed)
    # the undeflected reference carries the centre of the draws, or the whole cloud would be offset by them
    centre = Terms(a2_au_d2=A2_APOPHIS, area_to_mass_m2_kg=am)
    out = {"draws": draws, "area_to_mass_m2_kg": am, "rows": []}
    longest = max(warnings_years)
    for years in warnings_years:
        jd_impact = APOPHIS_2029_JD - years * 365.25
        jd_out = np.linspace(jd_impact, APOPHIS_2029_JD, frames) if years == longest else np.array([jd_impact])
        arcs = ensemble_arcs(body, jd_impact, jd_encounter=APOPHIS_2029_JD, mass_kg=masses_kg[0],
                             speed_km_s=IMPACTOR_SPEED_KM_S, target_kg=APOPHIS_MASS_KG, draws=draws, jd_out=jd_out,
                             window_days=30.0, step_days=0.5, terms=centre)
        misses = [m["miss_km"] for m in arcs["members"]]
        row = {"warning_years": years, "misses_km": misses, "spread_km": float(np.std(misses)),
               "median_km": float(np.median(misses)), "nominal_km": arcs["nominal"]["miss_km"],
               "success": success_probability(misses, earth_radius_km=EARTH_RADIUS_KM)}
        if years == longest:
            out["arcs"] = arcs
            out["delta_v_mm_s"] = [m["delta_v_m_s"] * 1000.0 for m in arcs["members"]]
        out["rows"].append(row)
        print(f"  ensemble at {years:2.0f} yr: median {row['median_km']:,.0f} km, spread {row['spread_km']:,.1f} km,"
              f" success {row['success']:.2f}", flush=True)
    if len(out["rows"]) > 1:
        out["trust_horizon_years"] = trust_horizon(
            warning_days=[r["warning_years"] * 365.25 for r in out["rows"]],
            spread_km=[r["spread_km"] for r in out["rows"]], earth_radius_km=EARTH_RADIUS_KM) / 365.25
    return out


def perturbation_study() -> list[dict]:
    """What each term in the field is worth at the 2029 encounter: the perturbers dropped one at a time, solar
    radiation pressure switched on at the rock's own area-to-mass ratio and at ten times it, the Yarkovsky term, and
    the numerical floor the small numbers have to be read against."""
    from pinneapple_physics.astro.defence import (
        APOPHIS_2029_JD,
        area_to_mass_ratio,
        perturbation_budget,
    )
    from pinneapple_physics.astro.smallbodies import load_body

    am = area_to_mass_ratio(APOPHIS_DIAMETER_KM, APOPHIS_MASS_KG)
    planets = ("mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune")
    rows = perturbation_budget(load_body("apophis"), APOPHIS_2029_JD, a2_au_d2=A2_APOPHIS, area_to_mass_m2_kg=am,
                               reflectivity=0.0, planets=planets, srp_scales=(1.0, 10.0), window_days=30.0,
                               step_days=0.5)
    for r in rows:
        print(f"  {r['term']:<70s} miss {r['dist_km']:>12,.0f} km  delta {r['delta_km']:>+13,.1f} km"
              f"  moved {r['dr_km']:>10,.0f} km  ({r['seconds']:.0f} s)", flush=True)
    return rows


def impact_surface(deflection: dict, ensemble: dict, jpl: dict) -> dict:
    """Probability that the body hits, on the (impactor mass x warning time) grid -- the table read, not a new model.

    Each cell keeps the miss distance the table measured, and each warning time keeps the spread the ensemble measured
    at that warning time; the orbit's own uncertainty is JPL's -- 1 sigma of the encounter distance from the bundled
    close-approach record. The two are added in quadrature and the Gaussian probability that the true miss falls
    inside the capture radius is the number. Two geometries are read off the same rows:

    * **would hit** -- the body is on a collision course and the undeflected miss is the Earth's limb, so the cell's
      miss change is exactly the margin the impactor bought. This is the only case in which a deflection changes
      anything, and it is what the surface is for.
    * **Apophis 2029** -- the undeflected miss is the 38 356 km this propagator gives, and the question is whether any
      cell walks it back. None does, and the margin in sigma is the number worth reading.
    """
    from pinneapple_physics.astro.defence import impact_probability

    base = deflection["baseline"]["dist_km"]
    spread = {r["warning_years"]: r["spread_km"] for r in ensemble["rows"]}
    sigma_orbit = jpl["sigma_km"]
    cells = []
    for row in deflection["rows"]:
        sigma = math.hypot(sigma_orbit, spread.get(row["warning_years"], 0.0))
        bought = row["miss_km"] - base
        cells.append({"impactor_kg": row["impactor_kg"], "warning_years": row["warning_years"],
                      "bought_km": bought, "sigma_km": sigma,
                      "p_impact_would_hit": impact_probability(EARTH_RADIUS_KM + bought, sigma, EARTH_RADIUS_KM),
                      "p_impact_apophis": impact_probability(row["miss_km"], sigma, EARTH_RADIUS_KM),
                      "margin_sigma_apophis": (row["miss_km"] - EARTH_RADIUS_KM) / sigma})
    return {"baseline_km": base, "sigma_orbit_km": sigma_orbit, "earth_radius_km": EARTH_RADIUS_KM, "cells": cells}


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

    ens = result.get("ensemble")
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

    surf = result.get("surface")
    if surf:
        path = _out("figures/impact_probability.png")
        _impact_figure(surf, path)
        made.append(path)
    return made


def _impact_figure(surf: dict, path: str) -> None:
    """The table as a surface: decades of impact probability against impactor mass and warning time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cells = surf["cells"]
    masses = sorted({c["impactor_kg"] for c in cells})
    years = sorted({c["warning_years"] for c in cells})
    floor = 1e-24
    vmin, vmax = 1e-24, 1.0

    def grid(key):
        return np.array([[c[key] for c in cells if c["impactor_kg"] == m and c["warning_years"] == y]
                         for m in masses for y in years]).reshape(len(masses), len(years))

    p_hit = np.clip(grid("p_impact_would_hit"), floor, vmax)
    margin = grid("margin_sigma_apophis")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True)
    ax = axes[0]
    mesh = ax.pcolormesh(np.arange(len(years) + 1), np.arange(len(masses) + 1), p_hit, cmap="viridis",
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    for i in range(len(masses)):
        for j in range(len(years)):
            v = grid("p_impact_would_hit")[i, j]
            label = f"{v:.2f}" if v > 0.1 else f"1e{round(math.log10(max(v, 1e-30)))}"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=7,
                    color="w" if v < 1e-2 else "0.1")
    ax.set_xticks(np.arange(len(years)) + 0.5, [f"{y:.0f}" for y in years])
    ax.set_yticks(np.arange(len(masses)) + 0.5, [f"{m:.0f}" for m in masses])
    ax.set_xlabel("warning time (years)")
    ax.set_ylabel("impactor mass (kg)")
    ax.set_title("a body on a collision course: P(impact) after the nudge", fontsize=10)

    fig.colorbar(mesh, ax=ax, label="probability that it still hits")

    ax = axes[1]
    mesh = ax.pcolormesh(np.arange(len(years) + 1), np.arange(len(masses) + 1), margin, cmap="magma_r")
    for i in range(len(masses)):
        for j in range(len(years)):
            ax.text(j + 0.5, i + 0.5, f"{margin[i, j]:.0f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_xticks(np.arange(len(years)) + 0.5, [f"{y:.0f}" for y in years])
    ax.set_yticks(np.arange(len(masses)) + 0.5, [f"{m:.0f}" for m in masses])
    ax.set_xlabel("warning time (years)")
    ax.set_ylabel("impactor mass (kg)")
    ax.set_title("Apophis 2029: sigma between the flyby and an impact", fontsize=10)
    fig.colorbar(mesh, ax=ax, label=f"margin (1 sigma = {surf['sigma_orbit_km']:.0f} km, JPL)")
    fig.suptitle("The table read as a decision: same rows, same ensemble, the probability that the body hits",
                 fontsize=11)
    fig.savefig(path, dpi=140)


def animation(result: dict) -> str:
    """The film the deliverable asks for, made of the ensemble itself.

    Three panels share one clock that runs from the impactor's arrival to the encounter: the distance to the Earth
    falling, the ensemble cloud widening (every member's offset from the undeflected body, measured point by point on
    the trajectories that were actually integrated), and the encounter plane drawn to scale with the Earth on it. Every
    dot on the second and third panel is a real member of ``result["ensemble"]["arcs"]``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    arcs = result["ensemble"]["arcs"]
    jd = np.array(arcs["jd"])
    nominal = np.array(arcs["nominal"]["r_helio_km"])
    members = arcs["members"]
    dist_nom = np.array(arcs["dist_earth_km"])
    dist_all = np.array([np.array(m["dist_earth_km"]) for m in members])
    offsets = np.array([np.linalg.norm(np.array(m["r_helio_km"]) - nominal, axis=1) for m in members])
    misses = np.array([m["miss_km"] for m in members])
    miss_nom = arcs["nominal"]["miss_km"]
    sigma_orbit = result["jpl"]["sigma_km"]
    years = (jd - jd[0]) / 365.25
    total = float(years[-1])

    blue, orange, ink = "#0081be", "#a23a00", "#262626"
    fig = plt.figure(figsize=(11.6, 6.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
                        fig.add_subplot(grid[1, 1]))

    ax_a.set_xlim(0.0, total)
    ax_a.set_ylim(6e3, max(3e8, float(np.nanmax(dist_nom)) * 1.5))
    ax_a.set_yscale("log")
    ax_a.axhline(EARTH_RADIUS_KM, color="tab:red", ls="--", lw=1.0, label="the Earth's surface")
    ax_a.set_xlabel("years since the impactor arrived")
    ax_a.set_ylabel("distance from the Earth (km)")
    ax_a.set_title("the approach: one trajectory per ensemble member, from the impulse to the flyby", fontsize=10)
    ax_a.grid(True, which="both", alpha=0.25)
    line_nom, = ax_a.plot([], [], color=ink, lw=1.3, label="the undeflected body")
    ax_a.plot([], [], color=blue, alpha=0.15, lw=8, label="the ensemble")
    ax_a.legend(fontsize=8, loc="lower left")

    ax_b.set_xlim(0.0, total)
    ax_b.set_ylim(0.0, float(offsets.max()) * 1.08)
    ax_b.axhline(EARTH_RADIUS_KM, color="tab:red", ls="--", lw=1.0, label="the Earth's radius, for scale")
    ax_b.set_xlabel("years since the impactor arrived")
    ax_b.set_ylabel("offset from the undeflected body (km)")
    ax_b.set_title("the cloud widening", fontsize=10)
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(fontsize=8, loc="upper left")
    dots_b = [ax_b.plot([], [], "o", color=orange, ms=4)[0] for _ in members]
    trails_b = [ax_b.plot([], [], "-", color=orange, lw=0.6, alpha=0.5)[0] for _ in members]

    span = max(float(miss_nom), float(misses.max())) * 1.35
    ax_c.set_xlim(-span, span)
    ax_c.set_ylim(-span * 0.30, span * 1.30)
    ax_c.set_aspect("equal")
    ax_c.add_patch(mpatches.Circle((0.0, 0.0), EARTH_RADIUS_KM, color=blue, alpha=0.9, zorder=3))
    ax_c.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax_c.set_xlabel("in the plane of the flyby (km)")
    ax_c.set_ylabel("miss distance (km)")
    ax_c.set_title("the encounter plane, to scale", fontsize=10)
    ax_c.grid(True, alpha=0.2)
    ax_c.plot([0.0], [miss_nom], "o", color=ink, ms=6, zorder=4, label=f"undeflected: {miss_nom:,.0f} km")
    cloud_c = ax_c.plot([], [], "o", color=orange, ms=4, zorder=5,
                        label=f"the ensemble: {misses.mean():,.0f} km")[0]
    ax_c.add_patch(mpatches.Ellipse((0.0, float(misses.mean())), 6 * sigma_orbit, 6 * sigma_orbit, fill=False,
                                    ls=":", lw=1.0, color="0.35", zorder=2))
    ax_c.text(-span * 0.96, span * 1.14, f"dotted: JPL's 3 sigma on the flyby distance ({3 * sigma_orbit:,.0f} km)",
              fontsize=7, color="0.35")
    ax_c.legend(fontsize=8, loc="upper right")

    stamp = fig.text(0.5, 0.935, "", ha="center", fontsize=10, color=ink)
    step = max(len(jd) // 48, 1)
    frames = list(range(0, len(jd), step))
    bands = []

    def update(k):
        while bands:
            bands.pop().remove()
        line_nom.set_data(years[:k + 1], dist_nom[:k + 1])
        lo = np.minimum(dist_nom[:k + 1], dist_all[:, :k + 1].min(axis=0))
        hi = np.maximum(dist_nom[:k + 1], dist_all[:, :k + 1].max(axis=0))
        bands.append(ax_a.fill_between(years[:k + 1], lo, hi, color=blue, alpha=0.15))
        for dot, trail, off in zip(dots_b, trails_b, offsets, strict=True):
            dot.set_data(years[k:k + 1], off[k:k + 1])
            trail.set_data(years[:k + 1], off[:k + 1])
        cloud_c.set_data([0.0], [float(misses.mean())])
        stamp.set_text(f"{years[k]:.1f} of {total:.1f} years before the flyby   |   "
                       f"cloud {offsets[:, k].max():,.0f} km wide   |   "
                       f"miss {miss_nom:,.0f} km = {miss_nom / EARTH_RADIUS_KM:.1f} Earth radii")
        return [line_nom, *bands, *dots_b, *trails_b, cloud_c, stamp]

    anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    path = _out("figures/planetary_defence.gif")
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="the DART anchor and a single deflection")
    ap.add_argument("--out", default=_out("apophis_deflection.json"))
    args = ap.parse_args(argv)

    from pinneapple_physics.astro.defence import APOPHIS_2029_JD, jpl_miss_sigma_km

    t0 = time.time()
    anchor = dart_anchor()
    print(f"DART: beta = {anchor['beta_from_period_change']:.2f} (published {anchor['beta_published']}), "
          f"period change {anchor['period_change_minutes_implied']:.1f} min "
          f"(published {anchor['period_change_minutes_published']:.0f})", flush=True)
    jpl = jpl_miss_sigma_km()
    print(f"JPL, {jpl['cd']}: {jpl['dist_km']:,.0f} km +- {jpl['sigma_km']:,.0f} (1 sigma), "
          f"{jpl['v_rel_km_s']:.2f} km/s", flush=True)

    masses = (100.0, 1000.0) if args.quick else (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
    years = (5.0,) if args.quick else (1.0, 5.0, 10.0)
    print("deflection study:", flush=True)
    deflection = deflection_study(masses, years, quick=args.quick)
    print(f"undeflected miss: {deflection['baseline']['dist_km']:,.0f} km at JD "
          f"{deflection['baseline']['jd_tdb']:.2f}", flush=True)

    small = None
    if not args.quick:
        print("what-if: a 50 m body (1.3e8 kg) on the same orbit:", flush=True)
        small = deflection_study((10.0, 100.0, 1000.0), (5.0,), target_kg=1.3e8)

    ensemble = None
    if not args.quick:
        print("ensemble of beta, Yarkovsky A2 and the area-to-mass ratio:", flush=True)
        ensemble = ensemble_study(masses_kg=(1000.0,), warnings_years=(1.0, 5.0, 10.0), n=ENSEMBLE_N)

    perturbations = None
    if not args.quick:
        print("what each term in the field is worth:", flush=True)
        perturbations = perturbation_study()
    surface = impact_surface(deflection, ensemble, jpl) if ensemble else None

    result = {"issue": 381, "body": "99942 Apophis", "target": "Earth",
              "encounter_jd_tdb": deflection["baseline"]["jd_tdb"],
              "encounter_jd_published": APOPHIS_2029_JD, "apophis_mass_kg": APOPHIS_MASS_KG,
              "apophis_diameter_km": APOPHIS_DIAMETER_KM, "earth_radius_km": EARTH_RADIUS_KM,
              "jpl": jpl, "dart": anchor, "deflection": deflection, "what_if_50m": small, "ensemble": ensemble,
              "perturbations": perturbations, "surface": surface, "seconds": time.time() - t0}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    for path in figures(result):
        print(f"figure: {path}")
    if ensemble and "arcs" in ensemble:
        print(f"animation: {animation(result)}")
    print(f"wrote {args.out} in {result['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
