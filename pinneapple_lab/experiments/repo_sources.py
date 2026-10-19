"""More committed results for ``repo_results``: electric machines and electromagnetics, the airfoil hand-off package,
the airliner CFD, the lunar rover, the near-random forecasting study, the design-from-intent demos, topology
optimisation, the free-surface trough, the benchmark arena, the SPM motor and the weather model.

Each source re-reads the files a repository script committed and turns their numbers into lab checks. A check's
``kind`` says what it is evidence of (``RunContext.check``):

* ``reference`` only for an independent result: a published experiment or simulation by another group, an exact
  solution of the same problem, or another code that does not share the model's inputs (XFOIL against OpenFOAM);
* ``verification`` for a second code or method that shares the model's geometry, materials or loads (GetDP on the
  same Prius polygons, a reduced model against the full solver of the same pipeline), and for solver checks on
  canonical problems: they answer "is the solver right", never "is the result right", so they never lift a tier;
* ``physics`` for a law the result was not fitted to (energy, mass, force balance, symmetry, a physical bound).

Tolerances that the producing script did not state are declared here, in the check's detail, before reading the
result into a tier; nothing is widened to get a pass. Where a result has no independent reference, the source says
so in ``LIMITATIONS`` and the curation keeps it below tier A.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EM = "examples/electrodynamics/results/"


def _p(rel: str) -> str:
    return os.path.join(_REPO, rel)


def _json(rel: str):
    with open(_p(rel)) as f:
        return json.load(f)


def _figs(ctx, files, prefix: str = "") -> None:
    for f in files:
        if os.path.exists(_p(f)):
            ctx.figure_file(_p(f), prefix + os.path.basename(f))


# ---------------------------------------------------------------------------------------------- electric machines
_MACHINE_CASES = {   # machines.verification case -> (check name, kind)
    "Magnetised ring, sector mesher + P1 FEM": ("fem_vs_magnetised_ring_closed_form", "verification"),
    "Machine sector geometry (Prius slot and magnet areas)": ("geometry_slot_and_magnet_areas_exact", "sanity"),
    "Prius IPM: Maxwell stress vs co-energy torque": ("maxwell_stress_vs_coenergy_torque", "physics"),
    "Prius IPM: dq power balance and frozen permeability": ("dq_power_balance", "physics"),
    "Prius IPM: mesh convergence of the peak torque": ("mesh_convergence_peak_torque", "sanity"),
}


def _prius_pmsm(ctx):
    r = _json(_EM + "10_pmsm_prius.json")
    g = _json(_EM + "14_prius_getdp_crosscheck.json")
    sk = _json(_EM + "15_prius_skew.json")
    tol = _json(_EM + "17_prius_tolerances.json")
    ctx.input("fea", {"machine": "2004 Toyota Prius interior-PM motor (pyleecan Toyota_Prius geometry)",
                      "n_gap": r["n_gap"], "elements": r["elements"], "magnets": "20 degC, Br 1.24 T",
                      "load": "250 A peak, 12 rotor angles per point"})
    tv = r["torque_vs_angle"]
    phi, T, Y = (np.asarray(tv[k], float) for k in ("phi0_deg", "torque_Nm", "yang_2013_Nm"))
    c = tv["comparison"]
    ctx.metric("peak_torque_Nm", float(T.max()))
    ctx.metric("peak_torque_yang_2013_Nm", float(Y.max()))
    ctx.metric("torque_rms_vs_yang_2013_Nm", c["rms_error_Nm"])
    ctx.metric("torque_max_abs_vs_yang_2013_Nm", c["max_abs_error_Nm"])
    ctx.check("peak_torque_vs_yang_2013", value=float(T.max()), reference=float(Y.max()), rtol=0.05, kind="reference",
              label="torque vs published FEA (Yang et al. 2013)",
              detail="published FEA of the same motor (Yang, Krishnamurthy & Brown, IEEE IEMDC 2013): another group "
                     "and code, not a measurement; tolerance 5 % (machines.verification); largest deviation "
                     f"{c['max_abs_error_Nm']:+.1f} N m at {c['max_abs_error_at_deg']:.0f} deg, rms {c['rms_error_Nm']:.1f} N m")
    ctx.check("zero_mean_torque_at_180deg_symmetry", value=abs(float(T[-1])) / float(T.max()), max=1e-3, kind="physics",
              detail="pure negative d-axis current on a rotor symmetric about its d-axis gives no mean torque")
    for case in r["verification"]:
        if case["case"] not in _MACHINE_CASES:
            continue                                 # the Yang case is recomputed above from the curve itself
        name, kind = _MACHINE_CASES[case["case"]]
        ctx.check(name, bool(case["passed"]), value=case["error"], kind=kind,
                  detail=f"{case['case']}, pass/fail and tolerance of machines.verification; {case.get('detail', '')}")
    # second code: gmsh + GetDP. Own mesh, nonlinear solver and torque integral; shared polygons, B-H data, currents
    dif = g["torque_difference_getdp_minus_pinneapple"]
    emf = g["back_emf_1000rpm"]
    ctx.metric("getdp_torque_max_abs_difference_Nm", dif["max_abs_Nm"])
    ctx.check("torque_vs_getdp_second_code", value=dif["max_abs_Nm"] / float(T.max()), max=0.01, kind="verification",
              detail="gmsh 4.12 + GetDP 3.2, 13 current angles: its own mesh, Newton solver and Arkkio torque, but the "
                     "same geometry, materials and currents, so partly independent (it cannot catch an input error); "
                     "tolerance 1 % of the peak")
    ctx.check("back_emf_vs_getdp_second_code",
              value=emf["getdp_fundamental_peak_V"], reference=emf["pinneapple_fundamental_peak_V"], rtol=0.01,
              kind="verification", detail="fundamental at 1000 rpm, 60 positions; same sharing as the torque")
    cg = g["cogging"]
    spread = abs(cg["pinneapple_pkpk_Nm"] - cg["getdp_pkpk_Nm"]) / cg["pinneapple_pkpk_Nm"]
    ctx.metric("cogging_pkpk_Nm", cg["pinneapple_pkpk_Nm"])
    ctx.metric("cogging_pkpk_getdp_Nm", cg["getdp_pkpk_Nm"])
    ctx.metric("cogging_code_spread_rel", spread)
    for row in g.get("cogging_convergence", []):
        ctx.metric(f"cogging_pkpk_n{row['n_gap']}_Nm", row["pinneapple_pkpk_Nm"])
        ctx.metric(f"cogging_pkpk_getdp_n{row['n_gap']}_Nm", row["getdp_pkpk_Nm"])
    lin = g["cogging_diagnosis"]["linear_iron"]
    ctx.check("cogging_linear_iron_vs_getdp", value=lin["getdp_pkpk_Nm"], reference=lin["pinneapple_pkpk_Nm"],
              rtol=0.02, kind="verification",
              detail="with linear iron the two codes agree on cogging; with saturated bridges they converge 25 % apart "
                     "(a declared uncertainty, not settled without a measurement or a third code)")
    # step skew: the multi-slice back-EMF follows the closed-form skew factor; skew lowers the cogging
    be = sk["back_emf"]
    ctx.check("skew_back_emf_vs_closed_form_factor", value=be["max_deviation_from_factor_rel_fundamental"], max=1e-6,
              kind="verification", detail="every harmonic = straight harmonic x |sin(K h pi s / lam) / (K sin(h pi s / "
                                          "lam))|; checks the multi-slice composition, not the 2-D field")
    sw = {(s["steps"], round(s["skew_slot_pitches"], 3)): s for s in sk["sweep"]}
    ctx.check("half_pitch_skew_lowers_cogging", passed=sw[(2, 0.5)]["cogging_pkpk_Nm"] < sw[(2, 0.0)]["cogging_pkpk_Nm"],
              kind="physics", detail=f"2 steps over half a slot pitch: {sw[(2, 0.0)]['cogging_pkpk_Nm']:.3f} -> "
                                     f"{sw[(2, 0.5)]['cogging_pkpk_Nm']:.3f} N m")
    # manufacturing tolerances (JCGM 101 Monte Carlo through the FEA)
    o = tol["outputs"]
    ctx.metric("torque_std_Nm", o["torque_Nm"]["std"])
    ctx.metric("ripple_std", o["ripple"]["std"])
    ctx.metric("yield_within_spec", tol["p_within_spec_joint"])
    ctx.metric("yield_wilson95_low", tol["p_within_spec_joint_wilson95"][0])
    ds = ctx.dataset("torque_vs_current_angle", description="Prius IPM mean torque at 250 A against current angle: "
                     "this FEA, GetDP and the published curve of Yang et al. (2013)", units={"phi0": "deg", "T": "N m"})
    for row in g["torque_vs_angle"]:
        ds.add(phi0_deg=row["phi0_deg"], torque_pinneapple=row["pinneapple"], torque_getdp=row["getdp"],
               torque_yang_2013=row["yang_2013"])
    _figs(ctx, [_EM + f for f in ("10_pmsm_field.png", "15_prius_skew_3d.png", "10_pmsm_curves.png",
                                  "14_prius_getdp_crosscheck.png", "14_prius_cogging_convergence.png",
                                  "10_pmsm_mtpa.png", "15_prius_skew.png", "17_prius_tolerances.png")], "prius_")


def _prius_end_effects(ctx):
    e = _json(_EM + "16_prius_end_effects.json")
    lk = sorted(e["magnet_end_leakage_3d"], key=lambda r: r["end_air"]["prisms"])
    lk = [r for r in lk if r["end_air_m"] == lk[0]["end_air_m"]]
    ratios = [r["ratio"] for r in lk]
    for r in lk:
        ctx.metric(f"end_leakage_ratio_{r['end_air']['prisms']}_prisms", r["ratio"])
    ctx.check("end_leakage_mesh_convergence", value=abs(ratios[-1] - ratios[-2]) / ratios[-1], max=0.005,
              kind="sanity", detail=f"3-D/2-D magnet flux ratio on {len(ratios)} meshes: "
                                    + ", ".join(f"{x:.4f}" for x in ratios) + "; tolerance 0.5 % on the last step")
    ctx.check("end_leakage_ratio_below_one", passed=all(x < 1 for x in ratios), kind="physics",
              detail="flux that leaks round the magnet ends can only lower the 2-D magnet flux")
    ew = e["end_winding"]
    a, b = ew["150"]["air"]["L_end_dq"], ew["300"]["air"]["L_end_dq"]
    ctx.metric("L_end_air_uH", 1e6 * b)
    ctx.metric("L_end_iron_face_uH", 1e6 * ew["300"]["iron_face"]["L_end_dq"])
    ctx.check("end_winding_discretisation_convergence", value=abs(a - b) / b, max=1e-3, kind="sanity",
              detail="Neumann integral with 150 and 300 pieces per metre")
    env = e["envelope_Nm"]
    ctx.check("end_winding_reactance_lowers_envelope",
              passed=all(x <= y + 1e-9 for x, y in zip(env["air"], env["2d"], strict=True)), kind="physics",
              detail="the end-winding reactance takes voltage, so the torque envelope cannot rise")
    cc = e["corrected_2d"]
    ctx.metric("torque_2d_Nm", cc["2d"]["torque_Nm"])
    ctx.metric("torque_end_corrected_Nm", cc["end_leakage"]["torque_Nm"])
    _figs(ctx, [_EM + "16_prius_end_effects.png"], "prius_")


def _prius_thermal_demag(ctx):
    from pinneapple_simulation.electromagnetics.machines import verification as mv
    t = _json(_EM + "13_prius_thermal.json")
    with ctx.stage("closed-form verification (run now)"):
        res = mv.run(["Thermal network: first-order transient", "Thermal network: cylindrical conduction",
                      "Demagnetisation field in a magnetised ring"])
    for c in res:
        ctx.check(c.name.lower().replace(":", "").replace(" ", "_"), bool(c.passed), value=c.error, kind="verification",
                  detail=f"{c.reference}; tolerance {c.tolerance:g}, observed orders "
                         + ", ".join(f"{o:.2f}" for o in c.orders))
    sw = sorted(t["magnet_temperature_sweep"], key=lambda r: r["magnet_temperature_C"])
    tq = [r["torque_Nm"] for r in sw]
    emf = [r["emf_phase_fundamental_peak_V_1000rpm"] for r in sw]
    ctx.check("torque_and_emf_fall_with_magnet_temperature",
              passed=bool(np.all(np.diff(tq) < 0) and np.all(np.diff(emf) < 0)), kind="physics",
              detail=f"20 -> {sw[-1]['magnet_temperature_C']:.0f} degC: {tq[0]:.1f} -> {tq[-1]:.1f} N m")
    lv = t["demagnetisation"]["mesh_refinement"]
    fr = [lvl["cases"]["i_d = -250 A"]["H"]["demagnetised_fraction"] for lvl in lv]
    ctx.check("demag_fraction_mesh_convergence", value=abs(fr[-1] - fr[-2]), max=0.01, kind="sanity",
              detail="grade H at i_d = -250 A, demagnetised area fraction on the two finest of "
                     f"{len(fr)} magnet meshes; tolerance 1 point (the worst-element ratio does not converge: corners)")
    for k, v in t["peak_temperatures_C"].items():
        ctx.metric(f"wltc_peak_{k}_C", v)
    ctx.metric("time_to_180C_winding_at_peak_load_s", t["peak_load"]["time_to_180C_winding_s"])
    _figs(ctx, [_EM + f for f in ("13_prius_demag.png", "13_prius_thermal.png", "13_prius_demag_mesh.png")], "prius_")


def _prius_drive(ctx):
    d = _json(_EM + "11_prius_drive.json")
    ctx.input("assumptions", d["assumptions"])
    ctx.input("vehicle", d["vehicle"])
    names = {"Drive: torque envelope vs closed form (linear dq)": "envelope_vs_closed_form_linear_dq",
             "Drive: least-loss currents vs MTPA closed form": "least_loss_currents_vs_mtpa_closed_form",
             "Drive cycle: WLTC class 3b energy balance": "wltc_energy_balance_lossless_drive",
             "Prius flux map: interpolation vs direct FEA": "flux_map_interpolation_vs_direct_fea"}
    for c in d["verification"]:
        kind = "physics" if "energy balance" in c["case"] else "verification"
        ctx.check(names.get(c["case"], c["case"]), bool(c["passed"]), value=c["error"], kind=kind,
                  detail=f"{c['case']} (machines.verification); {c.get('detail', '')}")
    w = d["wltc_class3b"]
    bal = w["motoring_J"] - w["regenerated_J"]
    ctx.check("battery_energy_balance", value=w["battery_J"], reference=bal, rtol=1e-9, kind="physics",
              detail="battery energy = motoring - regenerated over the WLTC class 3b cycle")
    ctx.check("efficiency_below_one", passed=0 < d["efficiency_map"]["peak_efficiency"] < 1, kind="physics")
    ctx.metric("peak_efficiency", d["efficiency_map"]["peak_efficiency"])
    ctx.metric("wltc_battery_Wh_per_km", w["battery_Wh_per_km"])
    _figs(ctx, [_EM + "11_prius_drive.png"], "prius_")


def _em_verification(ctx):
    rows = _json(_EM + "em_verification.json")
    for r in rows:
        ref = r["reference"].lower()
        kind = "physics" if ("conservation" in ref or "solution of maxwell" in ref or "residual" in ref) else "reference"
        name = "".join(ch if ch.isalnum() else "_" for ch in r["case"].lower()).strip("_")
        while "__" in name:
            name = name.replace("__", "_")
        ctx.check(name, bool(r["passed"]), value=r["error"], kind=kind,
                  detail=f"{r['reference']}; pass/fail and tolerance of pp.em.verification"
                         + (f"; observed orders {', '.join(f'{o:.2f}' for o in r['orders'])} (expected "
                            f"{r['expected_order']:g})" if r.get("orders") and r.get("expected_order") else ""))
        ctx.metric(f"err_{name}"[:60], r["error"])
    g = _json(_EM + "fdtd_gallery.json")
    ctx.check("fdtd_fields_pass_maxwell_checks", passed=g["maxwell_checks"]["decision"] == "APPROVED", kind="physics",
              detail="Gauss, Faraday, Ampere-Maxwell and Poynting balance on the FDTD gallery fields")
    ctx.metric("fdtd_cylinder_ring_error_vs_series", g["cylinder"]["ring_error_vs_series"])
    ctx.metric("fdtd_dipole_power_rel_error",
               abs(g["dipole"]["radiated_power_W"] / g["dipole"]["radiated_power_closed_form_W"] - 1))
    _figs(ctx, [_EM + f for f in ("fdtd_cylinder_scattering.gif", "fdtd_cylinder_field_map.png",
                                  "fdtd_dipole_pattern.png")])


def _motor_spm(ctx):
    d = _json("examples/motor_magnetics/results/spm_summary.json")
    ctx.input("machine", d["machine"])
    pairs = d["cogging"]["maxwell_vs_coenergy_Nm"] + d["torque_vs_gamma"]["maxwell_vs_coenergy_Nm"]
    on_load = d["torque_vs_gamma"]["maxwell_vs_coenergy_Nm"]
    dev = max(abs(m - c) / abs(c) for _, m, c in on_load)
    ctx.check("maxwell_stress_vs_coenergy_torque", value=dev, max=0.02, kind="physics",
              detail="on-load torque at 3 current angles, two torque definitions of the same field; tolerance 2 % as "
                     "for the Prius")
    ctx.metric("cogging_maxwell_vs_coenergy_max_abs_Nm", max(abs(m - c) for _, m, c in pairs[:3]))
    ctx.metric("cogging_pkpk_Nm", d["cogging"]["peak_to_peak_Nm"])
    ctx.metric("peak_torque_Nm", max(d["torque_vs_gamma"]["torque_Nm"]))
    ctx.metric("emf_rms_V_1500rpm", d["back_emf"]["emf_rms_V"])
    _figs(ctx, ["examples/motor_magnetics/results/spm_torque_cogging_emf.png",
                "examples/motor_magnetics/results/spm_flux_density.png"])


# ---------------------------------------------------------------------------------------------- aero
_AF = "examples/handoff_cases/airfoil_ld/results/"


def _airfoil_handoff(ctx):
    b = _json(_AF + "baseline/baseline_summary.json")
    ctx.input("case", b["case"])
    xt = b["xfoil_at_alpha_4"]["naca2412_exact_tripped"]
    of = b["openfoam"]
    gc = b["grid_convergence"]
    ctx.check("cd_vs_xfoil_tripped", value=gc["Cd"]["extrapolated"], reference=xt["Cd"], rtol=0.05, kind="reference",
              label="Cd vs XFOIL (tripped)",
              detail="NACA 2412, Re 6e6, alpha 4 deg: grid-extrapolated OpenFOAM Cd against XFOIL 6.99 (Drela 1989; an "
                     "independent panel + boundary-layer code), transition fixed at the leading edge; tolerance 5 %")
    ctx.check("cl_vs_xfoil_tripped", value=of["medium"]["Cl"], reference=xt["Cl"], rtol=0.05, kind="reference",
              detail="same case, lift on the medium grid; tolerance 5 %")
    lc = b["lift_curve"]
    ctx.check("lift_slope_vs_thin_airfoil_theory", value=lc["openfoam_campaign_slope_per_deg"],
              reference=lc["thin_airfoil_theory_slope_per_deg"], rtol=0.05, kind="reference",
              detail="thin-airfoil theory (Abbott & von Doenhoff 1959), fit over -2..6 deg; tolerance 5 %")
    ctx.check("zero_lift_angle_vs_thin_airfoil_theory", value=lc["openfoam_campaign_alpha_L0_deg"],
              reference=lc["thin_airfoil_theory_alpha_L0_deg"], atol=0.25, kind="reference",
              detail="zero-lift angle from the mean line; tolerance 0.25 deg")
    ctx.check("grid_convergence_cd_gci", value=gc["Cd"]["GCI_fine_pct"], max=5.0, kind="sanity",
              detail=f"Celik et al. (2008) GCI on 3 grids, observed order {gc['Cd']['observed_order']:.2f}; tolerance 5 %")
    ctx.check("pressure_and_friction_drag_split", passed=all(of[k]["Cd"] > 0 and of[k]["Cl"] > 0 for k in of),
              kind="sanity")
    rr = _json(_AF + "campaign/sample_fields/rerun_check.json")
    ctx.check("campaign_rerun_reproduces_stored", value=max(abs(v["diff_Cl"]) for v in rr.values()), max=1e-3,
              kind="verification", detail=f"{len(rr)} campaign cases re-run from scratch")
    o = _json(_AF + "optimisation/optimised.json")["designs"]
    errs = {k: abs(v["surrogate"]["L_over_D"] / v["openfoam"]["L_over_D"] - 1) for k, v in o.items()}
    ctx.check("surrogate_ld_on_unseen_optimised_shapes", value=max(errs.values()), max=0.02, kind="generalization",
              detail=f"surrogate L/D against OpenFOAM on the {len(o)} optimiser picks (not in training); tolerance 2 %")
    ctx.metric("Cd_extrapolated", gc["Cd"]["extrapolated"])
    ctx.metric("Cd_xfoil_tripped", xt["Cd"])
    ctx.metric("Cl_medium", of["medium"]["Cl"])
    ctx.metric("optimum_LD_gain_fine_pct", b["optimum_grid_study"]["gain_over_baseline_pct"]["fine"])
    ctx.metric("surrogate_ld_max_rel_error_unseen", max(errs.values()))
    _figs(ctx, [_AF + f for f in ("baseline/polar_vs_xfoil.png", "baseline/cp_alpha4.png", "baseline/convergence.png",
                                  "campaign/campaign_overview.png", "optimisation/optimised_shapes.png")], "airfoil_")


def _airliner(ctx):
    v = _json("apps/aero_optimizer/model/verification3d.json")
    d = next(x for x in v["designs"] if x["label"] == "A320 class")
    ctx.input("case", {"note": v["note"], "cells": d["cells"], "alpha": d["alpha"], "speed": d["speed"], "x": d["x"]})
    for k in ("CL", "CD", "CD_pressure", "CD_friction", "cells"):
        ctx.metric(k, d[k])
    ctx.check("drag_balance_pressure_plus_friction", value=d["CD_pressure"] + d["CD_friction"], reference=d["CD"],
              rtol=1e-6, kind="physics", label="force balance (pressure + friction)",
              detail="total drag = integrated pressure + skin-friction drag")
    pl = sum(p["CL"] for p in d["patches"].values())
    pd = sum(p["CD"] for p in d["patches"].values())
    ctx.check("force_balance_over_patches", passed=abs(pl - d["CL"]) <= 1e-3 * abs(d["CL"]) and
              abs(pd - d["CD"]) <= 1e-3 * abs(d["CD"]), kind="physics",
              detail=f"fuselage + wing + tails = aircraft: CL {pl:.4f} vs {d['CL']:.4f}, CD {pd:.5f} vs {d['CD']:.5f}")
    ctx.check("lift_vs_vortex_lattice", value=d["CL"], reference=d["model"]["CL"], atol=0.1, kind="verification",
              detail="the design tool's own vortex lattice (wing + tail, no fuselage): same pipeline, not independent; "
                     "tolerance 0.1 as in tests/test_aero_optimizer_3d.py")
    rd = "apps/aero_optimizer/model/renders/a320_class_"
    _figs(ctx, [rd + f for f in ("lines.jpg", "cp.jpg", "cf.jpg", "flight.jpg")], "airliner_render_")
    _figs(ctx, ["docs/assets/studio/airliner-cp.jpg"])


# ---------------------------------------------------------------------------------------------- vehicles, forecasting
def _lunar_rover(ctx):
    d = _json("docs/assets/lunar_rover/results.json")
    ctx.input("vehicle", {k: d["vehicle"][k] for k in ("name", "mass_kg", "track_m", "wheelbase_m")})
    ds = ctx.dataset("single_wheel_validation", description="Wong-Reece wheel with the cited soils against published "
                     "single-wheel tests (third-party digitisations)", units={"rmse_N": "N", "rmse_m": "m"})
    for k, v in d["validation"].items():
        tag = k.replace(" ", "_")
        ctx.metric(f"{tag}_drawbar_rmse_over_load", v["drawbar_pull_N"]["rmse"] / v["load_N"])
        ctx.metric(f"{tag}_sinkage_rmse_mm", 1e3 * v["sinkage_m"]["rmse"])
        ds.add(dataset=k, source=v["source"], load_N=v["load_N"], drawbar_rmse_N=v["drawbar_pull_N"]["rmse"],
               sinkage_rmse_m=v["sinkage_m"]["rmse"])
    cal = d["slip_sinkage_calibration"]
    better = {k: v["held_out_after"]["sinkage_m"]["rmse"] < v["held_out_before"]["sinkage_m"]["rmse"]
              for k, v in cal.items()}
    ctx.check("held_out_sinkage_improves_after_calibration", passed=all(better.values()), kind="generalization",
              detail="slip-dependent exponent fitted on every other published point; error on the other points only, "
                     f"{len(cal)} series")
    gs = sorted(d["grade_sweep"], key=lambda r: r["grade_deg"])
    slip = [r["slip_mean"] for r in gs]
    sink = [r["sinkage_mm"] for r in gs]
    ctx.check("slip_and_sinkage_rise_with_grade", passed=bool(np.all(np.diff(slip) > 0) and np.all(np.diff(sink) > 0)),
              kind="physics", detail=f"0..{gs[-1]['grade_deg']:.0f} deg: slip {slip[0]:.3f} -> {slip[-1]:.3f}")
    te = d["surrogate"]["test"]
    ctx.check("wheel_surrogate_held_out_r2", value=min(te[k]["r2"] for k in te), min=0.999, kind="generalization",
              detail=f"{d['surrogate']['n_test']} held-out states against the Wong-Reece model it learned")
    ctx.metric("crater_max_sinkage_mm", d["crater"]["max_sinkage_mm"])
    ctx.metric("crater_energy_J", d["crater"]["energy_J"])
    _figs(ctx, ["docs/assets/lunar_rover/" + f for f in ("viper_beauty.png",
                                                         "validation_single_wheel.png", "grade_sweep.png",
                                                         "surrogate.png", "crater_series.png")], "rover_")


_NR = "examples/near_random_forecasting/"


def _near_random(ctx):
    cases = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(_p(_NR + "results/cases/*.json")))}
    lz = cases["lorenz63_noise0.01"]
    lam = lz["truth"]["lyapunov"]
    ctx.check("lorenz63_generator_lyapunov_vs_sprott_2003", value=lam, reference=0.906, rtol=0.05, kind="verification",
              detail="largest Lyapunov exponent of the synthetic truth (Benettin) against the published 0.906 (J. C. "
                     "Sprott, Chaos and Time-Series Analysis, 2003); tolerance 5 %. It checks the data generator, not "
                     "the forecasts, which have no independent reference")

    def wins(case):
        z = case["zoo"]
        hs = z["protocol"]["horizons"]
        out = {}
        for n, m in z["models"].items():
            if m["family"] == "baseline":
                continue
            out[n] = [hs[i] for i, x in enumerate(m.get("dm_vs_best_baseline") or [])
                      if x and x["p_holm"] < 0.05 and x["mean_loss_difference"] < 0]
        return out

    chaotic = [k for k, c in cases.items() if c["truth"].get("kind") == "chaotic"]
    beats = {k: max(len(v) for v in wins(cases[k]).values()) for k in chaotic}
    ctx.check("beats_best_baseline_on_untouched_test", passed=all(v > 0 for v in beats.values()),
              kind="generalization", detail="every chaotic series: at least one model beats the best baseline (chosen "
                                            "on validation) on the untouched test block, one-sided Diebold-Mariano, "
                                            "Holm-corrected p < 0.05")
    pers = [m["dm_vs_persistence"][0] for m in cases["lorenz63_noise0.01"]["zoo"]["models"].values()
            if m["family"] != "baseline" and m.get("dm_vs_persistence") and m["dm_vs_persistence"][0]]
    ctx.check("beats_persistence_lorenz63", passed=any(x["p_holm"] < 0.05 and x["mean_loss_difference"] < 0
                                                       for x in pers), kind="baseline")
    noise = [k for k in ("white_noise", "random_walk") if k in cases]
    ctx.check("no_false_skill_on_pure_noise", passed=all(not any(wins(cases[k]).values()) for k in noise),
              kind="sanity", detail="negative controls: no model may beat the baseline on white noise or a random walk")
    st_ok = all(c["predictability"]["surrogate_tests"]["npe_iaaft"]["reject_at_0_05"] ==
                (c["truth"].get("kind") == "chaotic") for k, c in cases.items()
                if c["truth"].get("kind") in ("chaotic", "stochastic", "noise"))
    ctx.check("iaaft_test_flags_nonlinearity_only_where_present", passed=st_ok, kind="sanity",
              detail="IAAFT surrogate test of permutation entropy rejects on chaotic truths only (synthetic cases)")
    for k, c in cases.items():
        w = wins(c)
        ctx.metric(f"{k}_horizons_beating_best_baseline", max((len(v) for v in w.values()), default=0))
        r = c["predictability"].get("rosenstein", {}).get("lyapunov")
        if r is not None:
            ctx.metric(f"{k}_rosenstein_lyapunov", r)
    ctx.metric("test_origins_lorenz63", lz["zoo"]["protocol"]["n_test_origins"])
    _figs(ctx, [_NR + "figures/" + f for f in ("fan_charts.png", "skill_vs_horizon.png", "skill_vs_persistence.png",
                                               "surrogate_tests.png", "pit.png", "overfitting.png")], "forecast_")


# ---------------------------------------------------------------------------------------------- design and CFD apps
def _design_from_intent(ctx):
    d = _json("examples/design_from_intent/_out/summary.json")
    for case, v in d.items():
        if case == "out_of_scope":
            ctx.check("out_of_scope_goal_refused", passed=v["reply"]["status"] == "out_of_scope", kind="sanity")
            continue
        ctx.metric(f"{case}_probability", v["probability"])
        for c in v["verification"]["comparison"]:
            key = f"{case}_{c['output']}"
            ctx.metric(f"{key}_full_vs_reduced_rel", c["rel_err"])
            if c["kind"] == "at_least":
                ctx.check(f"{key}_full_solver_at_least_bound", passed=c["full"] >= c["closed_form"], kind="physics",
                          detail=f"{v['verification']['label']}: {c['full']:.3f} >= closed-form bound "
                                 f"{c['closed_form']:.3f} (v_top^2 = g R)")
            else:
                ctx.check(f"{key}_full_solver_vs_reduced", value=c["rel_err"], max=0.05, kind="verification",
                          detail=f"{v['verification']['label']} against the pipeline's reduced model (same design, "
                                 "same loads: not independent); tolerance 5 %")
        if case == "glider":
            se = next(c for c in v["verification"]["comparison"] if c["output"] == "span_eff")
            ctx.check("glider_span_efficiency_at_most_elliptic", passed=se["full"] <= 1.0, kind="physics",
                      detail="Munk: no planar wing beats the elliptic loading (e = 1)")
    img = "docs/guides/img/design_from_intent/"
    _figs(ctx, sorted(os.path.relpath(f, _REPO) for f in glob.glob(_p(img + "*.png")) + glob.glob(_p(img + "*.gif"))))


def _topology(ctx):
    r = "examples/topology_optimization/results/"
    m = _json(r + "mbb_summary.json")
    h = _json(r + "heat_stokes_summary.json")
    rr = _json(r + "reluctance_rotor_summary.json")
    ctx.check("adjoint_gradient_vs_finite_differences", value=m["gradient_check_max_rel_err"], max=1e-4,
              kind="verification", detail="MBB compliance sensitivities; tolerance 1e-4")
    ctx.check("rotor_adjoint_gradient_vs_finite_differences", value=rr["gradient_check_max_rel_err"], max=1e-4,
              kind="verification")
    oc, mma = m["MBB, OC, filter"]["compliance"], m["MBB, MMA, filter"]["compliance"]
    ctx.check("oc_and_mma_reach_the_same_compliance", value=abs(oc - mma) / mma, max=0.01, kind="verification",
              detail="two optimisers, same problem; tolerance 1 %")
    ctx.check("optimised_conductor_beats_uniform", passed=h["thermal"]["optimized"] < h["thermal"]["uniform"],
              kind="baseline", detail="thermal compliance of the optimised layout against the uniform material")
    pipes = [k for k in h if k.startswith("double_pipe")]
    ctx.check("optimised_channels_beat_uniform", passed=all(h[k]["optimized"] < h[k]["uniform"] for k in pipes),
              kind="baseline", detail="Stokes dissipation, " + ", ".join(pipes))
    for k, v in m.items():
        if isinstance(v, dict) and "compliance" in v:
            ctx.metric(k.replace(", ", "_").replace(" ", "_") + "_compliance", v["compliance"])
    ctx.metric("rotor_torque_linear_Nm", rr["optimized_mean_torque_linear_Nm"])
    ctx.metric("rotor_torque_nonlinear_Nm", rr["optimized_mean_torque_nonlinear_Nm"])
    _figs(ctx, [r + f for f in ("mbb_and_cantilever.png", "heat_and_stokes.png", "reluctance_rotor.png")], "topopt_")


def _free_surface(ctx):
    r = "examples/handoff_cases/free_surface_trough/results/"
    f5, f10 = _json(r + "mesh_5mm/summary.json"), _json(r + "mesh_10mm/summary.json")
    ctx.input("parameters", f5["parameters"])
    for tag, s in (("5mm", f5), ("10mm", f10)):
        ctx.check(f"mass_balance_{tag}", value=s["mass_balance"]["imbalance_max_abs_pct_of_inflow_after_1s"], max=1.0,
                  kind="physics", detail="liquid in - out - stored, % of the inflow, after 1 s; tolerance 1 %")
    sc = f5["solver_checks"]
    ctx.check("solver_volume_matches_fields", value=sc["liquid_volume_solver_vs_fields_max_rel_diff"], max=1e-6,
              kind="sanity")
    ctx.check("courant_below_one", value=sc["courant_max_after_0p1s"], max=1.0, kind="sanity")
    x5, x10 = f5["impingement"]["x_floor_pressure_peak_m"], f10["impingement"]["x_floor_pressure_peak_m"]
    ctx.check("impingement_point_mesh_sensitivity", value=abs(x5 - x10) / x5, max=0.05, kind="sanity",
              detail="floor pressure peak, 5 mm against 10 mm cells; tolerance 5 %")
    ctx.metric("x_pressure_peak_m", x5)
    ctx.metric("x_ballistic_jet_m", f5["impingement"]["x_ballistic_at_mean_free_surface_m"])
    ctx.metric("floor_tau_mean_abs_max_Pa", f5["impingement"]["floor_tau_mean_abs_max_Pa"])
    _figs(ctx, [r + "mesh_5mm/" + f for f in ("free_surface_snapshots.png", "floor_shear_and_k.png", "checks.png",
                                              "k_vertical_lines.png")], "trough_")


def _arena(ctx):
    a = _json("apps/benchmark_arena/results/burgers_fields.json")
    ctx.input("problem", {"source": a["problem"].get("meta", {}).get("source"), "protocol": a["protocol"]})
    base = a["baselines"][0]
    best = a["rankings"]["interpolation"][0]
    ctx.check("best_method_beats_fdm_baseline", value=best["error_mean"], max=base["error_mean"], kind="baseline",
              detail=f"{best['method']} relative L2 against the exact Cole-Hopf solution, interpolation regime, vs "
                     f"{base['id']}")
    for c in a["audit"]["checks"]:
        if c["status"] in ("pass", "fail"):
            ctx.check(f"dataset_audit_{c['id']}".replace(" ", "_").replace(":", "")[:60], c["status"] == "pass",
                      kind="sanity", detail=c["summary"])
    for reg, rows in a["rankings"].items():
        for row in rows:
            ctx.metric(f"{reg}_{row['method']}_rel_l2", row["error_mean"])
    _figs(ctx, ["docs/assets/apps/d22-arena-bench09.jpg"])


def _weather(ctx):
    b = _json("examples/weather_forecasting/data/baselines_2020.json")
    ctx.input("evaluation", {"leads_h": b["_meta"]["leads_h"], "inits": len(b["_meta"]["inits"]),
                             "baselines": sorted(k for k in b if not k.startswith("_")),
                             "checkpoint": "examples/weather_forecasting/checkpoints/phase1_step15000_fp16.pt"})
    ctx.log("the model's own scores are not in a committed file (run_use_cases.py writes horizon.json, not "
            "committed): no check can be recorded until they are")
    _figs(ctx, ["docs/assets/weather/" + f for f in ("skill_z500_preview.png", "pnw_heat_dome_2021_preview_day3.png")],
          "weather_")


SOURCES = {
    "prius_pmsm": "2004 Toyota Prius interior-PM motor, nonlinear 2-D FEA: torque against current angle vs the "
                  "published curve of Yang et al. (2013), second code GetDP, cogging, step skew, tolerances",
    "prius_end_effects": "Prius end effects: end-winding inductance (Neumann integral) and 3-D magnet end leakage "
                         "(GetDP), with the corrected 2-D torque and drive envelope",
    "prius_thermal_demag": "Prius magnet temperature, WLTC thermal network and demagnetisation by grade",
    "prius_drive": "Prius drive: flux map, MTPA and flux weakening, efficiency map, WLTC class 3b energy",
    "em_verification": "Electromagnetics verification suite (32 cases: FEM, Biot-Savart, Mie, Fresnel, dipoles, "
                       "FDTD) against closed forms and published values, with the FDTD gallery",
    "motor_spm": "Surface-PM motor (24 slots, 4 poles): cogging, torque against current angle, back-EMF, losses",
    "airfoil_handoff": "NACA 2412 hand-off package: OpenFOAM on three grids against XFOIL and thin-airfoil theory, "
                       "a 436-run campaign and optimised shapes checked in OpenFOAM",
    "airliner_cfd": "A320-class airliner, OpenFOAM half model (0.4 M cells): pressure, friction and streamlines at "
                    "approach",
    "lunar_rover": "NASA VIPER-class lunar rover: Wong-Reece wheels against published single-wheel tests, grades, "
                   "cross-slope and a crater crossing",
    "near_random_forecasting": "Forecasting near-random series without fooling yourself: 14 series, predictability "
                               "tests, a model zoo, purged walk-forward and Diebold-Mariano tests on an untouched test",
    "design_from_intent": "Plain-language goals to verified designs (loop, bracket, heat sink, glider), each checked "
                          "by a full solver",
    "topology_optimization": "Topology optimisation: MBB beam and cantilever, heat conduction and Stokes channels, a "
                             "reluctance rotor",
    "free_surface_trough": "Free-surface jet into a trough (interFoam, 5 and 10 mm cells): mass balance, impingement, "
                           "floor shear",
    "benchmark_arena": "Benchmark arena: equal-budget methods on viscous Burgers (exact Cole-Hopf data) in five regimes",
    "weather_era5": "ERA5 64 x 32 global weather model (phase 1): heat-dome forecast and z500 skill previews",
}

SCRIPTS = {
    "prius_pmsm": ["examples/electrodynamics/10_pmsm_prius.py", "examples/electrodynamics/14_prius_getdp_crosscheck.py",
                   "examples/electrodynamics/15_prius_skew.py", "examples/electrodynamics/17_prius_tolerances.py",
                   "pinneapple_simulation/electromagnetics/machines/verification.py"],
    "prius_end_effects": ["examples/electrodynamics/16_prius_end_effects.py"],
    "prius_thermal_demag": ["examples/electrodynamics/13_prius_thermal.py"],
    "prius_drive": ["examples/electrodynamics/11_prius_drive.py"],
    "em_verification": ["examples/electrodynamics/07_em_verification.py", "examples/electrodynamics/08_fdtd_gallery.py",
                        "pinneapple_simulation/electromagnetics/verification.py"],
    "motor_spm": ["examples/motor_magnetics/01_spm_cogging_and_torque.py"],
    "airfoil_handoff": ["examples/handoff_cases/airfoil_ld/simulation/summarise_baseline.py",
                        "examples/handoff_cases/airfoil_ld/simulation/xfoil_reference.py"],
    "airliner_cfd": ["apps/aero_optimizer/tools/verify3d_airliner.py"],
    "lunar_rover": ["examples/use_cases/terramechanics/lunar_rover_3d.py"],
    "near_random_forecasting": ["examples/near_random_forecasting/run_study.py"],
    "design_from_intent": ["examples/design_from_intent/design_from_intent.py"],
    "topology_optimization": ["examples/topology_optimization/01_mbb_beam.py",
                              "examples/topology_optimization/02_heat_and_stokes_flow.py",
                              "examples/topology_optimization/03_reluctance_rotor.py"],
    "free_surface_trough": ["examples/handoff_cases/free_surface_trough/simulation/trough_case.py"],
    "benchmark_arena": ["apps/benchmark_arena/run_public_benchmarks.py"],
    "weather_era5": ["examples/weather_forecasting/run_use_cases.py"],
}

LIMITATIONS = {
    "prius_pmsm": ["the reference is a published FEA (Yang et al. 2013), not a measurement; its model differs (centre "
                   "post, magnet temperature, steel data): +25 N m at 90 deg, rms 13 N m",
                   "cogging converges to values 25 % apart in this code and GetDP (saturated 1.5 mm bridges); no "
                   "measured Prius cogging was available",
                   "GetDP shares the geometry, materials and currents: it verifies the solver, not the model inputs",
                   "2-D: magnet end leakage lowers the torque by 2.6 % (see prius_end_effects)"],
    "prius_end_effects": ["no independent reference: no measured end-winding inductance or end leakage of the Prius "
                          "was available; the filament inductance is verified against closed forms in the tests only",
                          "the iron end face is bounded by the air and perfect-mirror cases, not modelled"],
    "prius_thermal_demag": ["thermal network data are illustrative defaults, not measured on a Prius: no independent "
                            "reference for the temperatures",
                            "magnet grades are candidates: the 2004 Prius grade is not public; the worst-element "
                            "field does not converge at sharp corners"],
    "prius_drive": ["vehicle parameters are illustrative (not a measured Prius); no measured efficiency map compared"],
    "em_verification": ["canonical problems: verifies the solvers, not any device model"],
    "motor_spm": ["illustrative machine: dimensions, materials and loss data are not a real motor; no published "
                  "reference (TEAM benchmarks not done)",
                  "docs/core_concepts/motor_and_topopt_audit.md quotes 16.80 vs 16.82 N m; the committed JSON gives "
                  "16.70 vs 16.71 N m"],
    "airfoil_handoff": ["2-D steady RANS (k-omega SST) at Re 6e6; XFOIL is an independent code but not an experiment; "
                        "the free-transition XFOIL drag is 80 % lower (laminar runs), the comparison uses tripped "
                        "transition"],
    "airliner_cfd": ["no mesh study (one 0.4 M-cell half model, wall functions)",
                     "no published reference for an A320-class aircraft at this condition was found: the only "
                     "comparison is the design tool's own vortex lattice",
                     "no nacelles or sharklets, tail at zero incidence, approach condition only"],
    "lunar_rover": ["agreement with the published single-wheel tests has no stated acceptance criterion: RMSE / load "
                    "0.05-0.24 for drawbar pull with the cited soils; the points are third-party digitisations",
                    "grades are model results, compared only qualitatively with NASA figures; no Apollo LRV data"],
    "near_random_forecasting": ["one realisation and one seed per series; tuning budget 4 configurations",
                                "most truths are synthetic from the same code; real data: USD/EUR, sunspots"],
    "design_from_intent": ["multi-fidelity cross-checks inside one pipeline: no published data"],
    "topology_optimization": ["no published reference in the committed files (Svanberg 1987, Borrvall-Petersson 2003 "
                              "agreement is asserted in the tests only)"],
    "free_surface_trough": ["no published data; the ballistic jet estimate is an indicator, not a reference"],
    "benchmark_arena": ["the dataset audit flags a collapse of the effective sample count (one pattern)"],
    "weather_era5": ["phase 1 only; the model's skill against persistence and climatology is quoted in the README but "
                     "no committed file holds it, so the lab records no check: commit horizon.json from "
                     "run_use_cases.py to curate it"],
}

# the short lines seals and showcase tiles print for each item's headline check (see Experiment.check_labels)
CHECK_LABELS = {
    "drag_vs_wind_tunnel": {"en": "drag vs wind tunnel (Ahmed, Ramm & Faltin, 1984)",
                            "pt": "arrasto vs t\u00fanel de vento (Ahmed, Ramm & Faltin, 1984)",
                            "short": "drag vs wind tunnel (Ahmed 1984)",
                            "short_pt": "arrasto vs t\u00fanel de vento (Ahmed 1984)"},
    "peak_torque_vs_yang_2013": {"en": "torque vs published FEA (Yang et al. 2013)",
                                 "pt": "torque vs FEA publicado (Yang et al. 2013)"},
    "coaxial_capacitor_p1_fem": {"en": "32 cases vs closed forms and published values",
                                 "pt": "32 casos vs formas fechadas e valores publicados"},
    "cd_vs_xfoil_tripped": {"en": "drag vs XFOIL \u00b7 lift slope vs thin-airfoil theory",
                            "pt": "arrasto vs XFOIL \u00b7 inclina\u00e7\u00e3o vs teoria do aerof\u00f3lio fino",
                            "also": ["lift_slope_vs_thin_airfoil_theory"]},
    "drag_balance_pressure_plus_friction": {"en": "force balance (pressure + friction)",
                                            "pt": "balan\u00e7o de for\u00e7as (press\u00e3o + atrito)"},
    "held_out_sinkage_improves_after_calibration": {
        "en": "held-out sinkage improves (published wheel tests)",
        "pt": "afundamento em testes retidos melhora (ensaios publicados)"},
    "beats_best_baseline_on_untouched_test": {"en": "beats the best baseline on an untouched test (DM, Holm)",
                                              "pt": "supera o melhor baseline em teste intocado (DM, Holm)"},
    "maxwell_stress_vs_coenergy_torque": {"en": "Maxwell-stress torque = co-energy torque",
                                          "pt": "torque de Maxwell = torque de co-energia"},
    "wltc_energy_balance_lossless_drive": {"en": "WLTC energy balance", "pt": "balan\u00e7o de energia WLTC"},
    "end_leakage_ratio_below_one": {"en": "3-D end leakage below the 2-D flux",
                                    "pt": "fuga 3-D nas pontas abaixo do fluxo 2-D"},
    "torque_and_emf_fall_with_magnet_temperature": {"en": "torque and EMF fall with magnet temperature",
                                                    "pt": "torque e FEM caem com a temperatura do \u00edm\u00e3"},
    "optimised_conductor_beats_uniform": {"en": "optimised layout beats uniform material",
                                          "pt": "layout otimizado supera material uniforme"},
    "mass_balance_5mm": {"en": "mass balance within 1 %", "pt": "balan\u00e7o de massa dentro de 1 %"},
    "loop_v_top_full_solver_at_least_bound": {"en": "loop speed \u2265 closed-form bound (Newton)",
                                              "pt": "velocidade no loop \u2265 limite anal\u00edtico (Newton)"},
}

RUN = {"prius_pmsm": _prius_pmsm, "prius_end_effects": _prius_end_effects, "prius_thermal_demag": _prius_thermal_demag,
       "prius_drive": _prius_drive, "em_verification": _em_verification, "motor_spm": _motor_spm,
       "airfoil_handoff": _airfoil_handoff, "airliner_cfd": _airliner, "lunar_rover": _lunar_rover,
       "near_random_forecasting": _near_random, "design_from_intent": _design_from_intent,
       "topology_optimization": _topology, "free_surface_trough": _free_surface, "benchmark_arena": _arena,
       "weather_era5": _weather}
