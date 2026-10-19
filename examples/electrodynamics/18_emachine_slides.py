"""Slides of the e-machine study (2004 Toyota Prius traction motor) from the committed results of examples 10 to 17.

Every number and every chart is read from ``examples/electrodynamics/results/*.json`` (the title slide's field plot is
the PNG of example 10). Headline numbers refer to one set of reference values (R1 to R4 on the FEA slide); where a
slide uses another mesh, sampling or operating point, the number carries that definition. Each slide states its
limits. Writes ``results/18_emachine_slides.pptx`` and, when LibreOffice is installed, a PDF next to it.

    pip install python-pptx
    python examples/electrodynamics/18_emachine_slides.py
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

RES = Path(__file__).parent / "results"
FONT = 15


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


# ------------------------------------------------------------------------------------------------ numbers
def facts() -> dict:
    p, x, d, e, h = (load(n) for n in ("10_pmsm_prius.json", "14_prius_getdp_crosscheck.json", "11_prius_drive.json",
                                       "16_prius_end_effects.json", "13_prius_thermal.json"))
    s, t, g = load("15_prius_skew.json"), load("17_prius_tolerances.json"), load("12_prius_rotor_design.json")
    mt = [r for r in p["mtpa"] if r["I_peak_A"] == 250.0][0]
    ta = p["torque_vs_angle"]
    T, Y = np.array(ta["torque_Nm"]), np.array(ta["yang_2013_Nm"])
    ref = {"T": mt["torque_Nm"], "gamma": mt["gamma_deg"], "Ld": mt["L_d_mH"], "Lq": mt["L_q_mH"],
           "cog": p["no_load"]["cogging_peak_to_peak_Nm"], "emf": p["no_load"]["emf_phase_fundamental_peak_V_at_1000rpm"],
           "thd": p["no_load"]["emf_thd_phase"], "n_gap": p["n_gap"], "elements": p["elements"],
           "T_sweep": float(T.max()), "T_pub": float(Y.max()), "rms_pub": float(np.sqrt(np.mean((T - Y) ** 2))),
           "cases": [(v["case"], v["passed"]) for v in p["verification"]], "phi": ta["phi0_deg"], "T_phi": T,
           "Y_phi": Y}
    sw = {(r["steps"], round(r["skew_mech_deg"], 3)): r for r in s["sweep"]}
    c = e["corrected_2d"]
    return {
        "ref": ref,
        "getdp": {"rows": x["torque_vs_angle"], "dT": x["torque_difference_getdp_minus_pinneapple"]["max_abs_Nm"],
                  "emf": (x["back_emf_1000rpm"]["pinneapple_fundamental_peak_V"],
                          x["back_emf_1000rpm"]["getdp_fundamental_peak_V"]),
                  "conv": x["cogging_convergence"], "lin": x["cogging_diagnosis"]["linear_iron"]},
        "drive": {"env": d["envelope"], "map": d["efficiency_map"], "wltc": d["wltc_class3b"],
                  "flux_map_gap_nodes": 480},
        "ends": {"rpm": e["envelope_rpm"], "env": e["envelope_Nm"], "L_air": 1e6 * e["end_winding"]["300"]["air"]["L_end_dq"],
                 "L_iron": 1e6 * e["end_winding"]["300"]["iron_face"]["L_end_dq"],
                 "levels": [r for r in e["magnet_end_leakage_3d"] if r["end_air_m"] == 0.04],
                 "T2d": c["2d"]["torque_Nm"], "Tend": c["end_leakage"]["torque_Nm"],
                 "E2d": c["2d"]["emf_fundamental_peak_V_1000rpm"], "Eend": c["end_leakage"]["emf_fundamental_peak_V_1000rpm"]},
        "thermal": {"sweep": h["magnet_temperature_sweep"], "peak": h["peak_temperatures_C"],
                    "t180": h["peak_load"]["time_to_180C_winding_s"], "T_peak": h["peak_load"]["torque_Nm"],
                    "rpm_peak": h["peak_load"]["rpm"], "i_sc": h["demagnetisation"]["short_circuit_peak_estimate_A"], "Tmag": h["demagnetisation"]["magnet_temperature_C"],
                    "demag": h["demagnetisation"]["cases"], "mesh": h["demagnetisation"]["mesh_refinement"]},
        "skew": {"sweep": s["sweep"], "s0": sw[(2, 0.0)], "k2": sw[(2, 3.75)], "k3": sw[(3, 5.625)],
                 "emf": (s["back_emf"]["fundamental_ratio"], s["back_emf"]["factor_fundamental"])},
        "tol": t,
        "design": g,
    }


# ------------------------------------------------------------------------------------------------ charts
def chart(draw, width=12.3, height=4.6, ncols=1):
    """A slide-sized figure drawn by ``draw(axes)`` at slide font size, returned as PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "legend.fontsize": FONT - 2,
                         "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "lines.linewidth": 2.2,
                         "lines.markersize": 7, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axs = plt.subplots(1, ncols, figsize=(width, height))
        draw(np.atleast_1d(axs))
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
    buf.seek(0)
    return buf


def ch_fea(f):
    r = f["ref"]

    def draw(ax):
        a = ax[0]
        a.plot(r["phi"], r["T_phi"], "o-", label="PINNeAPPle FEA")
        a.plot(r["phi"], r["Y_phi"], "s--", label="Yang et al. 2013")
        a.set_xlabel("current angle Phi0 (deg)")
        a.set_ylabel("mean torque at 250 A (N m)")
        a.legend(loc="lower center")
    return chart(draw, width=7.6)


def ch_getdp(f):
    g = f["getdp"]

    def draw(ax):
        p = [r["phi0_deg"] for r in g["rows"]]
        ax[0].plot(p, [r["pinneapple"] for r in g["rows"]], "o-", label="PINNeAPPle")
        ax[0].plot(p, [r["getdp"] for r in g["rows"]], "x--", label="gmsh + GetDP")
        ax[0].set_xlabel("current angle Phi0 (deg)")
        ax[0].set_ylabel("torque at 250 A (N m)")
        ax[0].legend()
        n = [c["n_gap"] for c in g["conv"]]
        ax[1].plot(n, [c["pinneapple_pkpk_Nm"] for c in g["conv"]], "o-", label="PINNeAPPle")
        ax[1].plot(n, [c["getdp_pkpk_Nm"] for c in g["conv"]], "x--", label="gmsh + GetDP")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xticks(n, [str(v) for v in n])
        ax[1].set_xlabel("gap nodes")
        ax[1].set_ylabel("cogging pk-pk (N m)")
        ax[1].set_ylim(0, None)
        ax[1].legend()
    return chart(draw, ncols=2)


def ch_drive(f):
    dr = f["drive"]

    def draw(ax):
        m = dr["map"]
        rpm = np.array(m["rpm"])
        T = np.array(m["torque_Nm"])
        eff = np.array(m["efficiency"], float)
        eff[eff < 0] = np.nan
        S = np.broadcast_to(rpm[:, None], T.shape)
        cs = ax[0].contourf(S, T, 100 * eff, levels=[70, 80, 85, 88, 90, 92, 93, 94, 95, 96, 97], cmap="viridis",
                            extend="min")
        ax[0].plot(dr["env"]["rpm"], dr["env"]["torque_Nm"], "k-", lw=2.5)
        ax[0].set_xlabel("speed (rpm)")
        ax[0].set_ylabel("torque (N m)")
        cb = ax[0].figure.colorbar(cs, ax=ax[0])
        cb.set_label("efficiency (%)")
    return chart(draw, width=9.0)


def ch_ends(f):
    en = f["ends"]

    def draw(ax):
        for k, lab in (("2d", "2-D model"), ("air", "+ end windings (air)"), ("iron_face", "+ end windings (iron face)")):
            ax[0].plot(en["rpm"], en["env"][k], "o-", label=lab)
        ax[0].set_xlabel("speed (rpm)")
        ax[0].set_ylabel("max torque, 250 A, 500 V (N m)")
        ax[0].legend()
        lv = en["levels"]
        ax[1].plot([r["end_air"]["prisms"] / 1e3 for r in lv], [r["ratio"] for r in lv], "o-")
        ax[1].set_xlabel("3-D mesh (thousand prisms)")
        ax[1].set_ylabel("magnet flux, 3-D / 2-D")
    return chart(draw, ncols=2)


def ch_thermal(f):
    th = f["thermal"]

    def draw(ax):
        sw = th["sweep"]
        Tm = [r["magnet_temperature_C"] for r in sw]
        for key, lab, mk in (("torque_Nm", "torque at 250 A (R1)", "o-"),
                             ("emf_phase_fundamental_peak_V_1000rpm", "back-EMF (R3)", "s-"),
                             ("Br_T", "remanence", "^--")):
            ax[0].plot(Tm, [100 * (r[key] / sw[0][key] - 1) for r in sw], mk, label=lab)
        ax[0].set_xlabel("magnet temperature (degC)")
        ax[0].set_ylabel("change from 20 degC (%)")
        ax[0].legend()
        names = ["teeth", "winding", "magnets", "end_winding", "yoke"]
        ax[1].barh(names[::-1], [th["peak"][n] for n in names[::-1]], color="C3")
        ax[1].set_xlabel("peak over 4 x WLTC 3b (degC)")
        ax[1].set_xlim(60, None)
    return chart(draw, ncols=2)


def ch_demag(f):
    th = f["thermal"]

    def draw(ax):
        grades = ["N", "M", "H", "SH", "UH"]
        labs = list(th["demag"])
        k = np.arange(len(grades))
        for j, lab in enumerate(labs):
            v = [100 * th["demag"][lab][g]["demagnetised_fraction"] for g in grades]
            short = "-250 A on the d-axis" if lab.startswith("i_d") else f"short-circuit peak (-{th['i_sc']:.0f} A)"
            ax[0].bar(k + (j - 0.5) * 0.38, v, 0.38, label=short)
        ax[0].set_xticks(k, grades)
        ax[0].set_xlabel("NdFeB grade (minimum Hcj)")
        ax[0].set_ylabel("demagnetised magnet area (%)")
        ax[0].legend()
        lv = th["mesh"]
        h = [1e3 * x["magnet_element_size_m"] for x in lv]
        full = [x for x in lv[0]["cases"] if x.startswith("i_d")][0]
        for g, mk in (("H", "o-"), ("SH", "s-")):
            ax[1].plot(h, [100 * x["cases"][full][g]["demagnetised_fraction"] for x in lv], mk, label=f"{g}, -250 A")
        ax[1].invert_xaxis()
        ax[1].set_xlabel("magnet element size (mm)")
        ax[1].set_ylabel("demagnetised area (%)")
        ax[1].set_ylim(0, None)
        ax[1].legend()
    return chart(draw, ncols=2)


def ch_skew(f):
    sk = f["skew"]

    def draw(ax):
        for K, mk in ((2, "o-"), (3, "s-")):
            rows = [r for r in sk["sweep"] if r["steps"] == K]
            a = [r["skew_mech_deg"] for r in rows]
            ax[0].plot(a, [r["cogging_pkpk_Nm"] for r in rows], mk, label=f"{K} steps")
            ax[1].plot(a, [100 * r["ripple_pkpk"] for r in rows], mk, label=f"{K} steps")
        ax[0].set_ylabel("cogging pk-pk (N m)")
        ax[1].set_ylabel("ripple at 250 A (%)")
        for a in ax:
            a.set_xlabel("total skew (mech. deg)")
            a.set_ylim(0, None)
            a.legend()
    return chart(draw, ncols=2)


def ch_tol(f):
    t = f["tol"]

    def draw(ax):
        Y = np.array(t["samples_outputs"])
        ax[0].hist(Y[:, 0], bins=20, color="C0")
        ax[0].axvline(t["nominal_outputs"]["torque_Nm"], color="k", lw=2)
        ax[0].set_xlabel("torque at 250 A (N m), 480 gap nodes")
        ax[0].set_ylabel("samples")
        names = ["remanence", "thickness", "width", "air gap"]
        k = np.arange(4)
        for j, (o, c) in enumerate((("torque_Nm", "C0"), ("ripple", "C1"))):
            ax[1].bar(k + (j - 0.5) * 0.38, [t["outputs"][o]["src"][v] for v in t["variables"]], 0.38, color=c,
                      label="torque" if j == 0 else "ripple")
        ax[1].axhline(0, color="k", lw=1)
        ax[1].set_xticks(k, names)
        ax[1].set_ylabel("sensitivity (SRC)")
        ax[1].legend()
    return chart(draw, ncols=2)


def ch_design(f):
    g = f["design"]

    def draw(ax):
        for r in g["runs"]:
            c = r["fea_only"]["hypervolume_curve"]
            line, = ax[0].plot(np.arange(1, len(c) + 1), c, label=f"NSGA-II on FEA, seed {r['seed']}")
            ax[0].plot([r["hybrid"]["fea_calls"]], [r["hybrid"]["hypervolume"]], "o", ms=11, color=line.get_color())
            fr = r["hybrid"]["front"]
            ax[1].plot([x["magnet_mass"] for x in fr], [x["torque"] for x in fr], "o", label=f"seed {r['seed']}")
        ax[0].set_xlabel("designs evaluated")
        ax[0].set_ylabel("hypervolume")
        ax[0].set_xlim(0, 1000)
        ax[0].legend(loc="lower right")
        b = g["baseline"]
        ax[1].plot([b["magnet_mass"]], [b["torque"]], "k*", ms=18, label="Prius rotor")
        ax[1].set_xlabel("magnet mass (kg)")
        ax[1].set_ylabel("torque at 250 A (N m)")
        ax[1].legend()
    return chart(draw, ncols=2)


# ------------------------------------------------------------------------------------------------ deck
def deck(f: dict):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Emu, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 16:9, 13.33 x 7.5 in
    W, H = prs.slide_width, prs.slide_height
    IN = 914400
    M = int(0.45 * IN)

    def text(s, x, y, w, h, lines, size=18, bold_first=False, italic=False, color=None):
        tf = s.shapes.add_textbox(x, y, w, h).text_frame
        tf.word_wrap = True
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = ln
            p.font.size = Pt(size)
            p.font.italic = italic
            p.font.bold = bold_first and i == 0
            if color:
                p.font.color.rgb = RGBColor(*color)
            p.space_after = Pt(5)
        return tf

    def slide(title, lines, png=None, limits=None, box=None):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        text(s, M, int(0.25 * IN), W - 2 * M, int(0.7 * IN), [title], size=28, bold_first=True)
        y = int(1.0 * IN)
        text_w = W - 2 * M if box is None else int(W * 0.62) - M
        per_line = max(20, int(text_w / IN * 8.0))  # characters per line at 18 pt
        n_lines = sum(-(-len(ln) // per_line) for ln in lines)
        text_h = int(0.33 * IN) * n_lines + int(0.08 * IN) * len(lines)
        text(s, M, y, text_w, text_h, lines, size=18)
        y += text_h + int(0.15 * IN)
        if png is not None:
            bottom = H - int(0.65 * IN)
            pic = s.shapes.add_picture(png, M, y, height=bottom - y)
            if pic.width > W - 2 * M - (int(W * 0.36) if box else 0):
                maxw = W - 2 * M - (int(W * 0.36) if box else 0)
                scale = maxw / pic.width
                pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
            if box is None:
                pic.left = int((W - pic.width) / 2)
        if box is not None:
            bx = int(W * 0.62)
            sh = s.shapes.add_shape(1, bx, int(1.0 * IN), W - bx - M, H - int(1.8 * IN))
            sh.fill.solid()
            sh.fill.fore_color.rgb = RGBColor(0xEE, 0xF3, 0xF8)
            sh.line.color.rgb = RGBColor(0x5B, 0x7D, 0xA8)
            tf = sh.text_frame
            tf.word_wrap = True
            for i, ln in enumerate(box):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = ln
                p.font.size = Pt(15 if i else 17)
                p.font.bold = i == 0
                p.font.color.rgb = RGBColor(0x1F, 0x2D, 0x3D)
                p.space_after = Pt(6)
        if limits:
            text(s, M, H - int(0.6 * IN), W - 2 * M, int(0.5 * IN), ["Limits: " + limits], size=13, italic=True,
                 color=(0x55, 0x55, 0x55))
        return s

    r, gd, dr, en, th, sk, tl, ds = (f[k] for k in ("ref", "getdp", "drive", "ends", "thermal", "skew", "tol", "design"))
    runs = ds["runs"]
    sav = [100 * x["fea_call_saving"] for x in runs]
    k2, s0 = sk["k2"], sk["s0"]
    wl = dr["wltc"]

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, M, int(1.2 * IN), int(5.6 * IN), int(1.6 * IN),
         ["2004 Toyota Prius traction motor", "Verified FEA to drive, thermal, robustness and design"], size=30,
         bold_first=True)
    text(s, M, int(3.3 * IN), int(5.6 * IN), int(2.6 * IN),
         ["Interior-PM, 48 slots, 8 poles, V magnets", "Open-source chain in PINNeAPPle",
          "Every number from the committed results of examples 10 to 17"], size=18)
    s.shapes.add_picture(str(RES / "10_pmsm_field.png"), int(6.3 * IN), int(0.4 * IN), height=H - int(0.8 * IN))

    # 2 executive summary
    slide("Executive summary",
          ["Done: 2-D nonlinear FEA of the Prius motor, checked against closed forms, a second FEA code (GetDP) and "
           "a published torque curve; drive, end effects, thermal, demagnetisation, skew, tolerances, design.",
           f"Torque at 250 A (R1, MTPA): {r['T']:.1f} N m. Peak of the angle sweep {r['T_sweep']:.1f} vs published "
           f"{r['T_pub']:.0f} N m ({100 * (r['T_sweep'] / r['T_pub'] - 1):+.1f} %); second code within {gd['dT']:.1f} N m.",
           f"WLTC 3b, illustrative car: {wl['battery_Wh_per_km']:.1f} Wh/km battery-side (motor losses only).",
           f"End effects: torque -{100 * (1 - en['Tend'] / en['T2d']):.1f} % (magnet end leakage), high-speed envelope "
           f"-{100 * (1 - en['env']['air'][2] / en['env']['2d'][2]):.0f} % at 3000 rpm (end windings).",
           f"Two-step skew: ripple {100 * s0['ripple_pkpk']:.1f} -> {100 * k2['ripple_pkpk']:.1f} % for "
           f"{100 * (k2['torque_mean_Nm'] / s0['torque_mean_Nm'] - 1):.1f} % torque. FEA + GP + NSGA-II: "
           f"{min(sav):.0f}-{max(sav):.0f} % fewer FEA calls at equal quality.",
           "Not verified: no measurement of this motor; cogging uncertain to ~25 % (two codes); thermal data "
           "illustrative; magnet grade unknown.",
           "Test-bench data that would close the gaps: torque vs current angle and back-EMF; cogging; winding and "
           "magnet temperatures over a cycle; standstill inductances; loss map."],
          limits="a model study from public data (pyleecan geometry, M400-50A steel).")

    # 3 FEA + reference values
    slide("2-D FEA, verified",
          [f"Torque vs current angle at 250 A against Yang et al. 2013: rms difference {r['rms_pub']:.1f} N m.",
           f"Closed-form verification cases: {sum(c[1] for c in r['cases'])} of {len(r['cases'])} pass."],
          png=ch_fea(f),
          box=["Reference values (R)",
               f"Mesh: {r['n_gap']} gap nodes, {r['elements']} elements; magnets 20 degC",
               f"R1  torque at 250 A, MTPA ({r['gamma']:.1f} deg), 12 rotor angles: {r['T']:.1f} N m",
               f"R2  cogging pk-pk, 24 angles per slot pitch: {r['cog']:.2f} N m",
               f"R3  back-EMF fundamental, 1000 rpm: {r['emf']:.1f} V peak, THD {100 * r['thd']:.1f} %",
               f"R4  L_d / L_q at R1: {r['Ld']:.3f} / {r['Lq']:.3f} mH",
               "Other slides give their mesh or definition where it differs."],
          limits="2-D; P1 elements; the published model and the pyleecan geometry differ (centre post, steel data).")

    # 4 second code
    c1920 = gd["conv"][-1]
    slide("Second code: gmsh + GetDP",
          [f"Torque vs angle: within {gd['dT']:.2f} N m (960 nodes, as R1); EMF {gd['emf'][0]:.2f} vs "
           f"{gd['emf'][1]:.2f} V (R3).",
           f"Cogging, 12 angles per slot pitch: {c1920['pinneapple_pkpk_Nm']:.2f} vs {c1920['getdp_pkpk_Nm']:.2f} N m "
           f"at {c1920['n_gap']} nodes; linear iron agrees ({gd['lin']['pinneapple_pkpk_Nm']:.2f} vs "
           f"{gd['lin']['getdp_pkpk_Nm']:.2f}): the gap is in the saturated bridges."],
          png=ch_getdp(f), limits="cogging uncertain by ~25 % between the codes; FEMM, pyleecan, Elmer not run.")

    # 5 drive
    env = dr["env"]
    slide("Drive: MTPA, flux weakening, efficiency",
          [f"250 A, 500 V DC (SVPWM), FEA flux map ({dr['flux_map_gap_nodes']} gap nodes, interpolated): "
           f"{env['torque_Nm'][0]:.1f} N m at low speed, {max(env['power_kW']):.1f} kW peak.",
           f"Peak efficiency {100 * dr['map']['peak_efficiency']:.1f} % at {dr['map']['peak_at']['rpm']:.0f} rpm; "
           f"WLTC 3b {wl['battery_Wh_per_km']:.1f} Wh/km (illustrative car)."],
          png=ch_drive(f), limits="steady state; no inverter, mechanical or battery losses; the car is not a Prius.")

    # 6 end effects
    lv = en["levels"]
    slide("End effects",
          [f"End windings: L_end {en['L_air']:.0f} uH (air) / {en['L_iron']:.0f} uH (iron face); envelope at 3000 rpm "
           f"{en['env']['2d'][2]:.1f} -> {en['env']['air'][2]:.1f} N m.",
           f"Magnet end leakage (3-D GetDP, 3 meshes): flux x{lv[-1]['ratio']:.3f}; R1 {en['T2d']:.1f} -> "
           f"{en['Tend']:.1f} N m, R3 {en['E2d']:.1f} -> {en['Eend']:.1f} V."],
          png=ch_ends(f), limits="corrections, not a coupled 3-D solution at load; no measured end-winding inductance.")

    # 7 thermal
    sw = th["sweep"]
    slide("Magnet temperature and thermal network",
          [f"Magnets {sw[0]['magnet_temperature_C']:.0f} -> {sw[-1]['magnet_temperature_C']:.0f} degC: R1 "
           f"{sw[0]['torque_Nm']:.1f} -> {sw[-1]['torque_Nm']:.1f} N m, R3 "
           f"{sw[0]['emf_phase_fundamental_peak_V_1000rpm']:.1f} -> {sw[-1]['emf_phase_fundamental_peak_V_1000rpm']:.1f} V.",
           f"Peak load ({th['T_peak']:.0f} N m, {th['rpm_peak']:.0f} rpm) after 4 cycles: winding at 180 degC in {th['t180']:.0f} s; magnets settle "
           f"at {th['Tmag']:.0f} degC."],
          png=ch_thermal(f), limits="seven-node network, illustrative thermal data (not fitted to a Prius).")

    # 8 demagnetisation
    full = [k for k in th["demag"] if k.startswith("i_d")][0]
    slide("Demagnetisation at the hottest magnet temperature",
          [f"Magnets at {th['Tmag']:.0f} degC, reverse field per element against Hcj(T) of each grade letter.",
           f"Grade H loses {100 * th['demag'][full]['H']['demagnetised_fraction']:.0f} % of the magnet area at -250 A, "
           f"SH {100 * th['demag'][full]['SH']['demagnetised_fraction']:.1f} %; area fractions are mesh-converged."],
          png=ch_demag(f),
          limits="Hcj coefficient assumed (-0.55 %/K); static knee test; the Prius grade was not found in public "
                 "sources: the bars compare candidate grades.")

    # 9 skew
    slide("Step skew",
          [f"Two steps over 3.75 deg: cogging {s0['cogging_pkpk_Nm']:.2f} -> {k2['cogging_pkpk_Nm']:.2f} N m "
           f"(12 angles per slot pitch; R2 uses 24), ripple {100 * s0['ripple_pkpk']:.1f} -> "
           f"{100 * k2['ripple_pkpk']:.1f} %, R1 {s0['torque_mean_Nm']:.1f} -> {k2['torque_mean_Nm']:.1f} N m.",
           f"Back-EMF fundamental x{sk['emf'][0]:.4f} = closed-form skew factor {sk['emf'][1]:.4f}."],
          png=ch_skew(f), limits="multi-slice: no inter-segment axial flux, no end effects.")

    # 10 tolerances
    T = tl["outputs"]["torque_Nm"]
    slide("Manufacturing tolerances",
          [f"{tl['valid_samples']} Monte-Carlo FEA samples (480 gap nodes; nominal {tl['nominal_outputs']['torque_Nm']:.1f}"
           f" N m on that mesh, R1 at 960 nodes): torque sigma {T['std']:.2f} N m ({100 * T['std'] / T['mean']:.2f} %).",
           "Remanence drives the torque spread, the air gap the ripple spread."],
          png=ch_tol(f), limits="the tolerance sigmas are assumptions, not measured spreads.")

    # 11 design
    reach = [x["fea_only"]["calls_to_reach_hybrid"] for x in runs]
    hyb = [x["hybrid"]["fea_calls"] for x in runs]
    slide("Rotor design: FEA + Gaussian process + NSGA-II",
          [f"Equal quality: {min(hyb)}-{max(hyb)} FEA calls (hybrid, dots) vs {min(reach)}-{max(reach)} (NSGA-II on "
           f"the FEA): {min(sav):.0f}-{max(sav):.0f} % fewer.",
           "Front designs beating the Prius rotor sit at the 1.0 mm bridge bound (Prius 1.5 mm); 288 gap nodes."],
          png=ch_design(f),
          limits="no bridge stress, demagnetisation or speed constraints: the method and its saving, not a better rotor.")

    # 12 verified / not / next
    slide("What is verified, what is not, what would close it",
          ["Verified: closed forms (FEM, filament inductance, thermal network, skew factor, drive optimiser); a second "
           "FEA code (torque, EMF, inductances); a published torque curve.",
           "Not verified: any measurement of this motor; cogging to better than ~25 %; thermal parameters; magnet "
           "grade; FEMM, pyleecan and Elmer runs (input files are written).",
           "Test bench: locked-rotor torque vs current angle; no-load back-EMF and cogging; thermocouples in winding "
           "and end winding over a WLTC; standstill L_d, L_q; loss map.",
           "Exchange: pyleecan in/out, FEMM Lua, Elmer, gmsh/GetDP, MATLAB, DXF, VTK, CSV/Parquet/HDF5 contracts."],
          limits="see the Limits section of docs/guides/electromagnetics.md.")
    return prs


def main() -> Path:
    prs = deck(facts())
    out = RES / "18_emachine_slides.pptx"
    prs.save(out)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(RES), str(out)],
                       check=False, capture_output=True, timeout=300)
    print(out, len(prs.slides), "slides")
    return out


if __name__ == "__main__":
    main()
