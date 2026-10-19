"""Fill the tables of docs/guides/flight_dynamics.md (between <!-- table:NAME --> markers) and results/tables.md
from the committed results JSON, so every number in the guide comes from a run.

    python examples/flight_dynamics/06_tables.py
"""
import json
import math
import os
import re

from pinneapple_simulation.flight.figures import LABELS, table_markdown

HERE = os.path.dirname(os.path.abspath(__file__))
GUIDE = os.path.join(HERE, "..", "..", "docs", "guides", "flight_dynamics.md")
R = {k: json.load(open(os.path.join(HERE, "results", f"{k}.json")))
     for k in ("verification", "stability", "takeoff", "turbulence", "same_gust") if os.path.exists(os.path.join(HERE, "results", f"{k}.json"))}
T = {}

if "verification" in R:
    v = R["verification"]
    T["psd"] = table_markdown(v["turbulence"]["cases"], [("generator", "generator"), ("model", "form"),
                              ("component", "comp."), ("sigma_target", "σ target [m/s]"), ("sigma", "σ generated [m/s]"),
                              ("psd_ratio_min", "PSD ratio min"), ("psd_ratio_max", "PSD ratio max")],
                              {"sigma_target": 3, "sigma": 3})
    rows = [{**r, "aircraft": LABELS[r["aircraft"]]} for r in v["trim"]]
    T["trim"] = table_markdown(rows, [("aircraft", "aircraft"), ("V", "V [m/s]"), ("altitude", "h [m]"),
                               ("alpha_deg", "α [deg]"), ("elevator_deg", "δe [deg]"), ("throttle", "throttle"),
                               ("residual", "trim residual [m/s²]"), ("dV_max", "10 s: max |ΔV| [m/s]"),
                               ("dh_max", "max |Δh| [m]"), ("dtheta_max_deg", "max |Δθ| [deg]")],
                               {"residual": 1, "dV_max": 4, "dh_max": 4, "dtheta_max_deg": 4, "V": 0, "altitude": 0})
    c = v["conservation"]
    T["conservation"] = table_markdown([{"dt": str(k), **c[k]} for k in c], [("dt", "Δt [s]"),
                                       ("energy_rel_error", "relative energy error (20 s)"),
                                       ("momentum_rel_error", "relative angular-momentum error"),
                                       ("quat_norm_error", "abs(norm(q) - 1)")], {"dt": 3})
    T["lattice"] = table_markdown([{**r, "aircraft": LABELS[r["aircraft"]]} for r in v["lattice"]],
                                  [("aircraft", "aircraft"), ("alpha_deg", "α [deg]"), ("CL_loop", "CL lattice in the loop"),
                                   ("CL_vlm", "CL steady lattice"), ("rel_diff", "relative difference")],
                                  {"CL_loop": 4, "CL_vlm": 4, "rel_diff": 4, "alpha_deg": 0})
if "stability" in R:
    rows = []
    for n, s in R["stability"].items():
        d, m = s["derivatives"], s["modes"]["oscillatory"]
        rows.append({"aircraft": LABELS[n], "mass": s["mass"], "S": s["wing_area"], "b": s["span"],
                     "ws": s["wing_loading"], "CLa": d["CL_alpha"], "SM": d["static_margin"], "Cma": d["Cm_alpha"],
                     "Cmq": d["Cm_q"], "Clb": d["Cl_beta"], "Cnb": d["Cn_beta"], "Clp": d["Cl_p"], "Cnr": d["Cn_r"],
                     "Cmde": d["Cm_de"], "Clda": d["Cl_da"], "Cndr": d["Cn_dr"],
                     "modes": ", ".join(f"{o['wn']:.2f} / {o['zeta']:.2f}" for o in m),
                     "Ixx": s["inertia_body"][0][0], "Iyy": s["inertia_body"][1][1], "Izz": s["inertia_body"][2][2]})
    T["aircraft"] = table_markdown(rows, [("aircraft", "aircraft"), ("mass", "mass [kg]"), ("S", "S [m²]"),
                                   ("b", "span [m]"), ("ws", "W/S [Pa]"), ("Ixx", "Ixx [kg m²]"), ("Iyy", "Iyy"),
                                   ("Izz", "Izz")], {"mass": 0, "S": 1, "b": 2, "ws": 0, "Ixx": 0, "Iyy": 0, "Izz": 0})
    T["derivatives"] = table_markdown(rows, [("aircraft", "aircraft"), ("CLa", "CLα"), ("Cma", "Cmα"), ("SM", "static margin"),
                                      ("Cmq", "Cmq"), ("Clb", "Clβ"), ("Cnb", "Cnβ"), ("Clp", "Clp"), ("Cnr", "Cnr"),
                                      ("Cmde", "Cmδe"), ("Clda", "Clδa"), ("Cndr", "Cnδr"),
                                      ("modes", "oscillatory modes ωn [rad/s] / ζ")],
                                      {"Clb": 3, "Cnb": 3, "Cnr": 3, "Clda": 3, "Cndr": 3})
if "takeoff" in R:
    t = R["takeoff"]
    rows = []
    for n, m in t["nominal"].items():
        cf = t["closed_form"][n]
        rows.append({"aircraft": LABELS[n], "VR": m["V_R_kt"], "VLOF": m["V_LOF_kt"], "theta": m["theta_LOF_deg"],
                     "alpha": m["alpha_LOF_deg"], "SG": m["ground_roll_m"], "S35": m["distance_35ft_m"],
                     "V35": m["V_35ft_kt"], "grad": m.get("climb_gradient_pct", float("nan")),
                     "cf": cf["raymer_1p1Vs"]["total_35ft_m"], "cf2": cf["raymer_with_sim_VLOF_and_rotation_time"]["total_35ft_m"],
                     "tail": m["tail_clearance_min_m"], "CLmax": m["CL_max_reached"],
                     "d1": 100 * (m["distance_35ft_m"] / cf["raymer_1p1Vs"]["total_35ft_m"] - 1),
                     "d2": 100 * (m["distance_35ft_m"] / cf["raymer_with_sim_VLOF_and_rotation_time"]["total_35ft_m"] - 1)})
    T["takeoff"] = table_markdown(rows, [("aircraft", "aircraft"), ("VR", "V_R [kt]"), ("VLOF", "V_LOF [kt]"),
                                  ("theta", "θ at lift-off [deg]"), ("alpha", "α at lift-off [deg]"),
                                  ("SG", "ground roll [m]"), ("S35", "to 35 ft [m]"), ("V35", "V at 35 ft [kt]"),
                                  ("grad", "climb gradient [%]"), ("cf", "Raymer, V_LOF = 1.1 V_s [m]"),
                                  ("d1", "sim vs Raymer [%]"),
                                  ("cf2", "Raymer, simulated V_LOF and rotation time [m]"), ("d2", "sim vs Raymer [%]"),
                                  ("tail", "min tail clearance [m]"), ("CLmax", "max CL")],
                                  {"VR": 1, "VLOF": 1, "theta": 1, "alpha": 1, "SG": 0, "S35": 0, "V35": 1, "grad": 1,
                                   "cf": 0, "cf2": 0, "d1": 1, "d2": 1})
    rows = []
    for n, m in t.get("crosswind", {}).items():
        g = t["gust_at_rotation"][n]
        rows.append({"aircraft": LABELS[n], "dev": m["max_lateral_deviation_ground_m"], "bank": m["max_bank_deg"],
                     "rud": m["max_rudder_deg"], "ail": m["max_aileron_deg"], "hdg": m["heading_at_LOF_deg"],
                     "S35c": m.get("distance_35ft_m", float("nan")), "tip": m["wingtip_clearance_min_m"],
                     "S35g": g.get("distance_35ft_m", float("nan")), "dS": g["delta_distance_35ft_m"],
                     "thg": g["theta_LOF_deg"], "VLg": g["V_LOF_kt"], "tailg": g["tail_clearance_min_m"]})
    if rows:
        T["crosswind"] = table_markdown(rows, [("aircraft", "aircraft"), ("dev", "max offset on the ground [m]"),
                                        ("bank", "max bank [deg]"), ("rud", "max rudder [deg]"),
                                        ("ail", "max aileron [deg]"), ("hdg", "heading at lift-off [deg]"),
                                        ("tip", "min wing-tip clearance [m]"), ("S35c", "to 35 ft [m]")],
                                        {"S35c": 0})
        T["gust"] = table_markdown(rows, [("aircraft", "aircraft"), ("S35g", "to 35 ft [m]"),
                                   ("dS", "change vs nominal [m]"), ("VLg", "V_LOF [kt]"), ("thg", "θ at lift-off [deg]"),
                                   ("tailg", "min tail clearance [m]")], {"S35g": 0, "dS": 0, "VLg": 1, "thg": 1})
    p = t["a320_vs_published"]
    T["a320"] = (f"| quantity | value |\n|---|---|\n| simulated all-engines distance to 35 ft, {p['sim_mass_kg']:.0f} kg, "
                 f"ISA sea level | {p['sim_all_engines_35ft_m']:.0f} m |\n| 1.15 x the simulated all-engines distance "
                 f"(the all-engines part of the certified rule) | {p['sim_all_engines_35ft_m']:.0f} m x 1.15 = "
                 f"{1.15 * p['sim_all_engines_35ft_m']:.0f} m |\n| published | {p['published']} |")
if "turbulence" in R:
    u = R["turbulence"]
    # validity: the linear lattice has no stall; alpha at the clean CLmax of the catalog (an assumption) from the trim
    from pinneapple_simulation.flight import catalog
    CLMAX = {n: catalog.build(n).clmax_clean for n in catalog.names()}

    def alpha_stall(m):
        return m["trim"]["alpha_deg"] + math.degrees((CLMAX[m["aircraft"]] - m["trim"]["CL"]) / m["cl_alpha"])

    def status(rows_):
        dep = sum(r.get("departed", r["phi_peak_deg"] > 90 or r["alpha_max_deg"] > alpha_stall(r) + 10)
                  for r in rows_)
        stall = sum(r.get("time_above_stall_frac", float(r["alpha_max_deg"] > alpha_stall(r))) > 0 for r in rows_)
        return "departed" if dep else ("beyond stall" if stall else "valid")
    rows = []
    for s in u["summary"]:
        runs = [r for r in u["runs"] if r["aircraft"] == s["aircraft"] and r["controller"] == s["controller"]]
        s["status"] = status(runs)
        s["alpha_stall"] = alpha_stall(runs[0])
        cab = s["cabin"]
        worst = max(cab, key=lambda k: cab[k]["dnz_rms"])
        rows.append({"aircraft": LABELS[s["aircraft"]], "ctl": s["controller"].replace("_", " "), "V": s["V"],
                     "ws": s["wing_loading"], "A": s["gust_sensitivity"], "qs": s["quasi_steady_dnz_rms"],
                     "rms": s["dnz_cg_rms"], "spr": s["dnz_cg_rms_spread"], "max": s["dnz_cg_max"], "min": s["dnz_cg_min"],
                     "phi": s["phi_rms_deg"], "phip": s["phi_peak_deg"], "th": s["theta_rms_deg"], "ny": s["ny_cg_rms"],
                     "cab": f"{worst.replace('_', ' ')}: {cab[worst]['dnz_rms']:.3f} / {cab[worst]['dnz_peak']:.2f}",
                     "amax": s["alpha_max_deg"], "ast": s["alpha_stall"], "st": s["status"]})
        if s["status"] == "departed":                 # a loss of control: the numbers are not a gust response
            for k in ("rms", "spr", "max", "min", "phi", "phip", "th", "ny", "cab", "amax"):
                rows[-1][k] = "–"
    T["turbulence"] = table_markdown(rows, [("aircraft", "aircraft"), ("ctl", "controls"), ("V", "V [m/s]"),
                                     ("ws", "W/S [Pa]"), ("A", "A = ρVCLα/2(W/S) [g per m/s]"), ("qs", "A·σ_w [g]"),
                                     ("rms", "RMS Δn_z CG [g]"), ("spr", "spread over seeds"), ("max", "max Δn_z"),
                                     ("min", "min Δn_z"), ("phi", "RMS bank [deg]"), ("phip", "peak bank [deg]"),
                                     ("th", "RMS pitch [deg]"), ("ny", "RMS n_y CG [g]"),
                                     ("cab", "worst cabin station: RMS / peak Δn_z [g]"), ("amax", "max α [deg]"),
                                     ("ast", "α at clean CLmax [deg]"), ("st", "model validity")],
                                     {"V": 0, "ws": 0, "A": 4, "qs": 3, "rms": 3, "spr": 3, "phi": 1, "phip": 1, "th": 1,
                                      "ny": 3, "amax": 1, "ast": 1})
    st = {(s["aircraft"]): s for s in u["summary"] if s["controller"] == "stick_fixed"}
    rows = [{"aircraft": LABELS[m["aircraft"]], "line_phi": m["phi_rms_deg"], "line_p": m["p_rms_deg_s"],
             "f3_phi": st[m["aircraft"]]["phi_rms_deg"], "line_rms": m["dnz_cg_rms"], "f3_rms": st[m["aircraft"]]["dnz_cg_rms"]}
            for m in u["line_field"] if st[m["aircraft"]].get("status") != "departed"]
    rows += [{"aircraft": LABELS[n] + " (departed stick fixed in severe turbulence: not shown)"}
             for n, x in st.items() if x.get("status") == "departed"]
    T["spanwise"] = table_markdown(rows, [("aircraft", "aircraft"), ("f3_phi", "RMS bank, 3-D field [deg]"),
                                   ("line_phi", "RMS bank, frozen line [deg]"), ("f3_rms", "RMS Δn_z, 3-D field [g]"),
                                   ("line_rms", "RMS Δn_z, frozen line [g]")], {"f3_phi": 1, "line_phi": 1, "f3_rms": 3,
                                                                                 "line_rms": 3})
    rows = [{"aircraft": LABELS[m["aircraft"]], "sev": m["severity"], "sw": m["spec"]["sigma_w"], "rms": m["dnz_cg_rms"],
             "max": m["dnz_cg_max"], "phi": m["phi_rms_deg"], "amax": m["alpha_max_deg"], "st": status([m])}
            for m in u["intensity"]]
    rows += [{"aircraft": LABELS[s["aircraft"]], "sev": "severe (mean of seeds)", "sw": u["spec"]["sigma_w"],
              "rms": s["dnz_cg_rms"], "max": s["dnz_cg_max"], "phi": s["phi_rms_deg"], "amax": s["alpha_max_deg"],
              "st": s["status"]} for s in u["summary"] if s["controller"] == "autopilot"]
    for r in rows:
        if r["st"] == "departed":
            r.update(rms="–", max="–", phi="–", amax="–")
    T["intensity"] = table_markdown(sorted(rows, key=lambda r: (r["aircraft"], r["sw"])),
                                    [("aircraft", "aircraft"), ("sev", "severity"), ("sw", "σ [m/s]"),
                                     ("rms", "RMS Δn_z [g]"), ("max", "max Δn_z [g]"), ("phi", "RMS bank [deg]"),
                                     ("amax", "max α [deg]"), ("st", "model validity")],
                                    {"sw": 2, "rms": 3, "phi": 1, "amax": 1})
    T["pratt"] = table_markdown([{**p, "aircraft": LABELS[p["aircraft"]]} for p in u["pratt"]],
                                [("aircraft", "aircraft"), ("V", "V [m/s]"), ("mu", "mass ratio μ"), ("K_g", "K_g"),
                                 ("dn_unalleviated", "Δn, no alleviation"), ("dn_pratt", "Δn Pratt"),
                                 ("dn_sim", "Δn simulated"), ("ratio_sim_to_pratt", "sim / Pratt")],
                                {"V": 0, "mu": 1, "K_g": 3, "dn_unalleviated": 3, "dn_pratt": 3, "dn_sim": 3,
                                 "ratio_sim_to_pratt": 3})

if "same_gust" in R:
    rows = []
    for n, r in R["same_gust"]["aircraft"].items():
        cab = r["cabin_dnz_max"]
        worst = max(cab, key=cab.get)
        rows.append({"aircraft": LABELS[n], "V": r["V"], "ws": r["wing_loading"], "tg": r["gust_time_s"],
                     "dn": r["dnz_cg_max"], "cab": f"{worst.replace('_', ' ')}: {cab[worst]:.3f}",
                     "qs": r["quasi_steady_dn"], "pr": r["pratt_dn_for_this_gust"], "th": r["theta_max_deg"],
                     "dh": r["dh_max_m"]})
    T["same_gust"] = table_markdown(rows, [("aircraft", "aircraft"), ("V", "V [m/s]"), ("ws", "W/S [Pa]"),
                                    ("tg", "gust duration [s]"), ("dn", "peak Δn_z CG [g]"),
                                    ("cab", "peak cabin Δn_z [g]"), ("qs", "quasi-steady A·U [g]"),
                                    ("pr", "Pratt (for H = 12.5 MAC) [g]"), ("th", "max Δθ [deg]"),
                                    ("dh", "max Δh [m]")], {"V": 0, "ws": 0, "dn": 3, "qs": 3, "pr": 3, "th": 2,
                                                             "dh": 1})

if "turbulence" in R:                       # turbulence figures, departed runs marked (status from above)
    import pickle

    from pinneapple_simulation.flight.figures import (
        encounter_bars_figure,
        encounter_traces_figure,
        gust_sensitivity_figure,
    )
    ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "flight_dynamics")
    summ = R["turbulence"]["summary"]
    encounter_bars_figure([s for s in summ if s["controller"] == "autopilot"], os.path.join(ASSETS, "turbulence_rms.png"))
    gust_sensitivity_figure(summ, R["turbulence"]["pratt"], os.path.join(ASSETS, "gust_sensitivity.png"))
    traces, notes = {}, {}
    for n in [s["aircraft"] for s in summ if s["controller"] == "autopilot"]:
        for case, note in ((f"{n}_turbulence", "severe"), (f"{n}_turbulence_moderate", "moderate")):
            f = os.path.join(HERE, "_out", case, "run.pkl")
            dep = next(s["status"] for s in summ if s["aircraft"] == n and s["controller"] == "autopilot") == "departed"
            if os.path.exists(f) and (note == "moderate") == dep:
                traces[n] = pickle.load(open(f, "rb"))["hist"]
                notes[n] = f"{note}, autopilot"
    if traces:
        encounter_traces_figure(traces, os.path.join(ASSETS, "turbulence_traces.png"), notes=notes)

with open(os.path.join(HERE, "results", "tables.md"), "w") as f:
    for k, v in T.items():
        f.write(f"## {k}\n\n{v}\n\n")
if os.path.exists(GUIDE):
    g = open(GUIDE).read()
    for k, v in T.items():
        g = re.sub(rf"(<!-- table:{k} -->\n).*?(\n<!-- /table:{k} -->)", lambda m, v=v: m.group(1) + v + m.group(2), g,
                   flags=re.S)
    open(GUIDE, "w").write(g)
print("tables:", ", ".join(T))
