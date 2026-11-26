"""A double-wishbone corner designed and sized in one loop: CAD, finite elements and multibody, with structural and
elastokinematic objectives.

The problem is the one a chassis engineer has: place the hardpoints and choose the arm's tube so the corner is light,
carries the loads, gains camber in bump, does not steer when it moves, and does not steer when it is pushed sideways —
and the last two fight each other, because the bushes that soften the ride are the ones that let the wheel steer.

What runs, in order:

1. the closed-loop multibody solve through wheel travel — camber, toe, caster, KPI, scrub, half track, motion ratio,
   wheel rate and roll centre (the *kinematic* objectives);
2. the load cases and the force each arm carries (the lower arm carries bending, the upper one is a two-force member);
3. the sizing at each candidate: tube stress against yield, Euler buckling, and the mass;
4. the elastokinematic compliance: six bush modes through the same solver, the elastic equilibrium under a lateral
   force, a braking force and an aligning torque (the *elastokinematic* objectives);
5. the pattern search over five variables, which returns the lightest corner that passes everything;
6. the finite-element check of the winning arm (8-node hexahedra, incompatible modes) against beam theory, and the 3-D
   field of von Mises on the deformed leg.

Run: ``python examples/suspension_elastokinematics/01_corner_design.py [--budget N] [--smoke] [--out DIR]``
Outputs in ``examples/suspension_elastokinematics/_out/``: the four figures (committed, they are what the manifest
shows), the 3-D field, the CAD and the JSON with every number (git-ignored, they are rebuilt by the run).
"""
from __future__ import annotations

import argparse
import json
import os

import pinneapple_lab
from pinneapple_design.suspension import (
    CONSTRAINTS,
    corner_load_cases,
    front_corner,
    kinematic_sweep,
    suspension_design,
    suspension_study,
    wheel_compliance,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    ap.add_argument("--budget", type=int, default=28, help="rounds of the design search")
    ap.add_argument("--smoke", action="store_true", help="a short search, for CI")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spec = front_corner()
    budget = 16 if args.smoke else args.budget
    with pinneapple_lab.experiment(
            "examples/suspension_elastokinematics",
            title="Suspension design and sizing: CAD, finite elements and multibody in one loop",
            description="A double-wishbone corner whose hardpoints, tube and bushings are searched for the lightest "
                        "design that meets a static margin, a camber-gain band, a bump-steer limit and a steer "
                        "compliance, re-checked with a 3-D finite-element model of the arm.",
            domain="vehicle dynamics", methods=["closed-loop multibody kinematics", "two-force load resolution",
                                                "beam stress and Euler buckling", "3-D hexahedral elasticity",
                                                "bushing compliance modes", "bounded pattern search"],
            modules=["pinneapple_design.suspension"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/suspension_elastokinematics.md",
                   "script": "examples/suspension_elastokinematics/01_corner_design.py"}) as ex:

        sweep = kinematic_sweep(spec)
        summary = sweep.summary()
        print("the nominal corner, wheels in bump and rebound")
        for k in ("static_camber_deg", "static_caster_deg", "static_kpi_deg", "static_scrub_radius_mm",
                  "camber_gain_deg_per_50mm", "bump_steer_deg_per_50mm", "half_track_change_mm_per_50mm",
                  "motion_ratio", "wheel_rate_n_per_mm", "roll_centre_height_mm"):
            print(f"  {k:<34s} {summary[k]:+9.3f}")
        print(f"  closure residual over the sweep    {summary['kinematic_closure_residual_m']:.1e} m")

        print("\nthe load cases at the contact patch [N]")
        for name, force in corner_load_cases(spec).items():
            print(f"  {name:<24s} Fx {force[0]:+7.0f}  Fy {force[1]:+7.0f}  Fz {force[2]:+7.0f}")

        comp = wheel_compliance(spec)
        print("\nthe bushes at their nominal rate: what the wheel does under load")
        print(f"  toe under 1 kN lateral    {comp['toe_deg_per_kn']:+.3f} deg")
        print(f"  camber under 1 kN lateral {comp['camber_deg_per_kn']:+.3f} deg")
        print(f"  toe under 1 kN braking    {comp['toe_deg_per_kn_braking']:+.3f} deg")

        result = suspension_design(spec, steps=budget)
        print(f"\nthe search: {result.evaluations} candidates")
        print(f"  baseline  {result.baseline['mass_kg']:.2f} kg  "
              f"feasible: {result.baseline['feasible']}  "
              f"violations: { {k: round(v, 2) for k, v in result.baseline['violations'].items() if v > 0} }")
        print(f"  optimum   {result.best['mass_kg']:.2f} kg  feasible: {result.best['feasible']}")
        for k, v in result.design.as_dict().items():
            print(f"    {k:<22s} {v:8.2f}")
        print("  constraints at the optimum")
        for name, rule in CONSTRAINTS.items():
            print(f"    {name:<28s} {result.best['measured'][name]:+8.3f}   ({rule['what']})")

        out = suspension_study(corner_mass_kg=spec.corner_mass_kg, bump_g=3.0, budget=budget, out_dir=args.out)
        print("\nthe study's own checks")
        for c in out["checks"]:
            value = c["value"]
            shown = f"{value:9.3f}" if isinstance(value, (int, float)) else str(value)
            print(f"  {c['status']:<4s} {c['code']:<34s} {shown}   {c['detail']}")
        print(f"\n{out['verdict']}")
        print("files:", ", ".join(os.path.basename(f) for f in out["files"]))

        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump({"kinematics": summary, "compliance": comp,
                       "design": result.design.as_dict(),
                       "metrics": out["metrics"], "checks": out["checks"], "limits": out["limits"]}, fh, indent=1)

        # the checks an experiment records: each one a reference the model is compared with, not a smoke test
        ex.check("the closed loop closes over the travel sweep", summary["kinematic_closure_residual_m"] < 1e-8,
                 detail=f"|residual| {summary['kinematic_closure_residual_m']:.1e} m over +/-60 mm", kind="physics")
        ex.check("the finite elements reproduce the beam",
                 abs(out["metrics"]["fem_to_beam_ratio"] - 1.0) <= 0.15,
                 detail=f"peak {out['metrics']['fem_peak_mpa']:.0f} MPa, ratio "
                        f"{out['metrics']['fem_to_beam_ratio']:.3f}", kind="numerics")
        ex.check("the design meets every constraint", all(c["status"] == "pass" for c in out["checks"]),
                 detail="; ".join(c["code"] for c in out["checks"] if c["status"] != "pass") or "all of them",
                 kind="physics")
        ex.log_metric("arm_and_bushing_mass", out["metrics"]["mass_kg"], "kg", "the winning design, both A-arms")
        ex.log_metric("mass_reduction", out["metrics"]["mass_reduction_pct"], "%", "against the nominal corner")
        ex.log_metric("outer_fibre_stress", out["metrics"]["sigma_mpa"], "MPa", "worst load case, closed form")
        ex.log_metric("fem_peak_stress", out["metrics"]["fem_peak_mpa"], "MPa", "3-D hexahedra on the winning arm")
        ex.log_metric("bump_steer", out["metrics"]["bump_steer_deg_per_50mm"], "deg/50mm", "the kinematic objective")
        ex.log_metric("toe_compliance", out["metrics"]["toe_compliance_deg_per_kn"], "deg/kN",
                      "the elastokinematic objective")


if __name__ == "__main__":
    main()
