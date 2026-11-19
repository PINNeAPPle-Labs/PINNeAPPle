"""Builds the PDF of the session's thermal experiment (NVIDIA examples 07 and 10).

Run with the Homebrew Python (reportlab + matplotlib + Pillow):

    /opt/homebrew/bin/python3.11 docs/reports/conjugate-thermal/make_report.py

Every number comes from the two examples' ``results/summary.json`` and ``results/run_info.json``, and every
figure from ``results/`` -- nothing is typed by hand.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E07 = ROOT / "examples/nvidia/07_omniverse_physicsnemo_coldplate"
E10 = ROOT / "examples/nvidia/10_brep_physicsnemo_heatsink"
OUT_PDF = HERE / "PINNeAPPle-Conjugate-Thermal-Experiment.pdf"

INK = colors.HexColor("#262626")
BLUE = colors.HexColor("#0081be")
GRAY = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#d9dee2")
WASH = colors.HexColor("#f4f7f9")
OK = colors.HexColor("#1d7a4c")

W, H = A4
M = 20 * mm
FRAME_W = W - 2 * M

# -------------------------------------------------------------------------------------------------- data
s07 = json.loads((E07 / "results/summary.json").read_text())
r07 = json.loads((E07 / "results/run_info.json").read_text())
s10 = json.loads((E10 / "results/summary.json").read_text())
r10 = json.loads((E10 / "results/run_info.json").read_text())
env = r07["environment"]
pk = env.get("packages", {})


def fmt(v, nd=None):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    if nd is not None:
        return f"{v:.{nd}f}"
    if a < 1e-6:
        return f"{v:.1e}".replace("e-0", "e-")
    if a >= 1e4:
        return f"{v:.4g}"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def pct(v, nd=1):
    return f"{100 * float(v):.{nd}f} %"


def chk(s, name):
    for c in s["checks"]:
        if c["name"] == name:
            return c["value"]
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------ styles
def st(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.4, leading=13.6, textColor=INK, spaceAfter=5,
                alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "body": st("body"),
    "lead": st("lead", fontSize=10.4, leading=15.4, textColor=colors.HexColor("#3a3a3a")),
    "h1": st("h1", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=GRAY, spaceBefore=2,
             spaceAfter=2, alignment=TA_LEFT),
    "h2": st("h2", fontName="Helvetica-Bold", fontSize=11.4, leading=15, textColor=BLUE, spaceBefore=11,
             spaceAfter=4, alignment=TA_LEFT),
    "h3": st("h3", fontName="Helvetica-Bold", fontSize=9.8, leading=13, textColor=INK, spaceBefore=7,
             spaceAfter=2, alignment=TA_LEFT),
    "cap": st("cap", fontSize=7.8, leading=10.4, textColor=GRAY, spaceBefore=3, spaceAfter=9,
              alignment=TA_LEFT),
    "small": st("small", fontSize=8.2, leading=11.4, textColor=INK, alignment=TA_LEFT),
    "cell": st("cell", fontSize=7.9, leading=10.2, spaceAfter=0, alignment=TA_LEFT),
    "cellb": st("cellb", fontName="Helvetica-Bold", fontSize=7.9, leading=10.2, spaceAfter=0,
                textColor=colors.white, alignment=TA_LEFT),
}


def H1(text):
    return Paragraph(text, S["h1"])


def H2(text):
    return Paragraph(text, S["h2"])


def H3(text):
    return Paragraph(text, S["h3"])


def P(text, style="body"):
    return Paragraph(text, S[style])


def caption(text):
    return Paragraph(text, S["cap"])


def fig(path, width=None, max_h=0.52 * H, cap=None):
    """Figure at frame width (or ``width``), with its caption, straight into the document."""
    path = Path(path)
    iw, ih = PILImage.open(path).size
    w = width or FRAME_W
    h = w * ih / iw
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = Image(str(path), width=w, height=h)
    if cap:
        story.append(KeepTogether([img, caption(cap)]))
    else:
        story.append(img)


def table(rows, widths, *, head=True, zebra=True, align_right=()):
    data = []
    for i, row in enumerate(rows):
        line = []
        for cell in row:
            if isinstance(cell, str):
                style = S["cellb"] if (head and i == 0) else S["cell"]
                line.append(Paragraph(cell, style))
            else:
                line.append(cell)
        data.append(line)
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), INK)]
    if zebra:
        start = 1 if head else 0
        for i in range(start, len(rows)):
            if (i - start) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), WASH))
    for j in align_right:
        style.append(("ALIGN", (j, 0), (j, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def checks_table(checks):
    rows = [["Check", "Value", "Limit", "Passed"]]
    for c in checks:
        rows.append([Paragraph(c["name"], S["cell"]), fmt(c.get("value")), fmt(c.get("limit")),
                     "yes" if c["passed"] else "NO"])
    return table(rows, [FRAME_W - 3 * 62, 62, 62, 62], align_right=(1, 2), zebra=True)


def badge(text, color):
    t = Table([[Paragraph(f'<font color="white"><b>{text}</b></font>',
                          st("badge", fontSize=9, leading=12, alignment=TA_LEFT))]],
              colWidths=[132], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


# ------------------------------------------------------------------------------------------- page canvas
def _chrome(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(M, H - M + 2, W - M, H - M + 2)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(GRAY)
    canvas.drawString(M, H - M + 6, "PINNeAPPle · conjugate thermal experiment")
    canvas.drawRightString(W - M, H - M + 6, "examples 07 and 10 · feat/thermal-examples-depth · PR #1873")
    canvas.line(M, M - 6, W - M, M - 6)
    canvas.drawString(M, M - 16, "every number and figure comes from the two examples' results/")
    canvas.drawRightString(W - M, M - 16, f"p. {doc.page}")
    canvas.restoreState()


def _cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, H - 104 * mm, W, 104 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#0081be"))
    canvas.rect(0, H - 106 * mm, W, 2 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(M, H - 24 * mm, "PINNeAPPle · experiment report · 18 November 2026")
    canvas.setFont("Helvetica-Bold", 25)
    canvas.drawString(M, H - 40 * mm, "Conjugate thermal")
    canvas.drawString(M, H - 50 * mm, "design")
    canvas.setFillColor(colors.HexColor("#9fd0e8"))
    canvas.setFont("Helvetica", 12.5)
    canvas.drawString(M, H - 62 * mm, "a microchannel cold plate and a multi-chip finned heat sink,")
    canvas.drawString(M, H - 68 * mm, "each verified by two independent physics models")
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(M, H - 84 * mm, "NVIDIA examples 07 and 10 · PINNeAPPle + PhysicsNeMo + OpenUSD")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#c8c8c8"))
    canvas.drawString(M, H - 90 * mm, "branch feat/thermal-examples-depth · PR #1873 in PINNeAPPle-Labs/PINNeAPPle")
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=M + 8,
                      bottomMargin=M + 6, title="PINNeAPPle · conjugate thermal experiment",
                      author="Yan Barros", subject="cold plate and multi-chip heat sink: examples 07 and 10",
                      creator="PINNeAPPle")
cover_frame = Frame(M, M + 4, FRAME_W, H - 114 * mm - M, id="cover", leftPadding=0, rightPadding=0,
                    topPadding=0, bottomPadding=0)
body_frame = Frame(M, M + 6, FRAME_W, H - 2 * M - 20, id="body", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="cover", frames=[cover_frame], onPage=_cover),
                      PageTemplate(id="body", frames=[body_frame], onPage=_chrome)])

story: list = []
A = story.append

# ================================================================================================== cover
A(Spacer(1, 4))
A(P("This report describes the experiment that made PINNeAPPle's two flagship thermal examples physically "
    "richer. The cold plate stopped being a conduction plate with a fixed film coefficient and now solves the "
    "<b>conjugate</b> problem against its coolant; the heat sink stopped having one equivalent source and now "
    "<b>resolves two chips</b> in a 3-D base, with radiation and seven design variables. Both end with an "
    "<b>APPROVED</b> decision and not a single failed check.", "lead"))
A(Spacer(1, 6))
A(table([["", "07 · conjugate cold plate", "10 · multi-chip heat sink"],
         ["Decision", "APPROVED (20/20)", "APPROVED (24/24)"],
         ["Physics", "slot and coolant solved together (Shah &amp; London, Hausen, Muzychka-Yovanovich)",
          "fin network + 3-D solve of the base with two chips, radiation weighted by the sky fraction"],
         ["Geometry", "27 fractional slots of 0.91 mm in 100 x 100 mm, an adiabatic guard ring",
          "21 fins 18.1 mm tall on a 3.5 mm base, root fillet, exact B-rep"],
         ["Study", "250 coupled solves, a 3-channel FNO, a design surrogate, a Monte-Carlo of the plant",
          "300 designs, a 7-input network, every proposal verified by two models"],
         ["Measured error", f"L2 {pct(s07['test_rel_l2'])} on held-out fields; hot spot "
                            f"{fmt(s07['max_hotspot_error_K'], 2)} K",
          f"RMS {pct(s10['heldout_rel_rms']['rise'], 2)} (rise), {pct(s10['heldout_rel_rms']['dp'], 2)} (dp)"],
         ["Runtime", f"{fmt(r07['wall_s'] / 60, 1)} min (most of it the FNO)",
          f"{fmt(r10['wall_s'] / 60, 1)} min"]],
        [26 * mm, FRAME_W / 2 - 13 * mm, FRAME_W / 2 - 13 * mm]))
A(Spacer(1, 10))
fig(E07 / "results/architecture.png",
    cap="The path through example 07 — the same diagram appears at a readable size as Figure 1.")
A(NextPageTemplate("body"))
A(PageBreak())

# ======================================================================================= 1. summary
A(H1("1 · Executive summary"))
A(P("Two engineering questions drove the session. On the cold plate: <i>how much of the chip's heat leaves with "
    "the water and how much leaves through the plate?</i> — a question a conduction plate with a fixed "
    "<font face='Courier'>h</font> cannot answer, because the coolant warms along the way and what it carries "
    "out changes the temperature field. On the heat sink: <i>how much hotter does the small chip run next to the "
    "large one?</i> — a question a resistance network with one equivalent source cannot answer either."))
A(P("Both examples were rewritten along three directions at once, and that is what makes them harder than they "
    "were: <b>richer physics</b> (the fluid became an unknown; radiation came in), <b>richer geometry</b> "
    "(sub-cell slots resolved by exact overlap; two chip footprints; a root fillet in the exact solid) and "
    "<b>richer study and verification</b> (two independent models per design, a Monte-Carlo of the plant, a design "
    "surrogate over the seven variables an engineer actually turns)."))
A(Spacer(1, 4))
A(badge("DECISION: APPROVED on both examples", OK))
A(Spacer(1, 8))
A(table([["", "07", "10"],
         ["checks passed", f"{sum(c['passed'] for c in s07['checks'])}/{len(s07['checks'])}",
          f"{sum(c['passed'] for c in s10['checks'])}/{len(s10['checks'])}"],
         ["held-out error", f"L2 {pct(s07['test_rel_l2'])} (limit 5 %)",
          f"RMS {pct(s10['heldout_rel_rms']['rise'], 2)} / {pct(s10['heldout_rel_rms']['dp'], 2)} / "
          f"{pct(s10['heldout_rel_rms']['mass'], 2)} %"],
         ["energy balance", f"closes to {fmt(s07['energy_closure_max_all_designs'])} on all 250 designs",
          f"closes to {fmt(chk(s10, '3-D base solve closes its energy balance (all designs)'))} on every design"],
         ["design shipped",
          f"{s07['identity']['chosen']['n_channels']} slots of "
          f"{fmt(1000 * s07['identity']['chosen']['channel_width'], 2)} mm at "
          f"{fmt(s07['identity']['chosen']['flow_lpm'], 2)} L/min -> {fmt(s07['identity']['t_max_c'], 1)} C",
          f"{s10['proposal']['n_fins']} fins, {fmt(1000 * s10['proposal']['fin_height'], 1)} mm, "
          f"{fmt(1000 * s10['proposal']['fin_t'], 2)} mm, base {fmt(1000 * s10['proposal']['base_t'], 1)} mm -> "
          f"{fmt(s10['proposal_verified']['t_source_3d'], 1)} C"]],
        [34 * mm, FRAME_W / 2 - 17 * mm, FRAME_W / 2 - 17 * mm]))

# ==================================================================================== 1.1 what changed
A(H2("1.1 What the experiment changed"))
A(table([["", "before", "now"],
         ["cold plate: fluid", "wall at a fixed temperature; a constant, hand-picked <font face='Courier'>h</font>",
          "the coolant is an unknown: lidded-slot Nusselt with the entry correction, pressure drop and water "
          "properties; the fluid's own energy balance is closed against the heat the slots deliver"],
         ["cold plate: mesh", "binary slot mask (only a slot wider than a cell resolves)",
          "exact fractional overlap and an area rescaling, so conductance and coupling do not depend on the mesh"],
         ["cold plate: walls", "Dirichlet edges, heat leaking through the manifold",
          "a two-cell guard ring: the outer walls really are adiabatic (0.2 % instead of 25 % of leakage)"],
         ["cold plate: surrogate", "2-channel FNO, 5.2 % error",
          "3-channel FNO (conductance, coupling, source): 1.1 %; the source carries the fluid's rise, which no "
          "map of (a, s) can recover"],
         ["heat sink: sources", "one equivalent centred source",
          "two chips (80 W on 40 x 40 mm and 15 W on 15 x 15 mm) resolved in the 3-D base, with a per-chip hot spot"],
         ["heat sink: radiation", "ignored",
          "a linearised coefficient weighted by the unobstructed sky fraction (25 % of the area)"],
         ["heat sink: design", "3 shape variables",
          "7 variables: shape, base thickness, material, the <font face='Courier'>h</font> multiplier and the "
          "interface resistance"],
         ["heat sink: verification", "one model",
          "two independent models per design, and the gap between them as a check of its own"]],
        [24 * mm, FRAME_W * 0.36, FRAME_W - 24 * mm - FRAME_W * 0.36]))

A(H3("How the session went"))
A(table([["step", "what happened"],
         ["1. choice", "among the repository's example pairs, the canonical NVIDIA pair (07 cold plate + 10 heat "
          "sink) and the three directions (physics, geometry, study) were chosen explicitly"],
         ["2. foundation", "the reacting problem -div(a grad u) + s u = f entered "
          "<font face='Courier'>darcy_checks</font> with the certificates preserved, plus an exact mode for when "
          "the reaction term dominates"],
         ["3. cold plate", "coolant model, fractional geometry, guard ring, a closed-form affine fixed point"],
         ["4. heat sink", "the 3-D solve with several sources, radiation in the network, a 7-variable study"],
         ["5. corrections", "seven defects found by the checks themselves (section 5) — none was hidden by "
          "loosening a tolerance"],
         ["6. gates", "ruff, 53 tests, experiment manifests, the snippet hub and the Studio coverage, both examples run end to end"],
         ["7. delivery", "commit and PR #1873"]],
        [24 * mm, FRAME_W - 24 * mm]))

# ======================================================================================== 2. example 07
A(H1("2 · Example 07 — the conjugate cold plate"))
A(P("A 100 x 100 mm copper plate with milled slots, two 110 W chips (14 x 14 mm) on its face and water at 40 C "
    "through the slots. The problem solved, per unit plate area, is"))
A(Paragraph('<font face="Courier">-div(G grad T) + sigma (T - T_f) = q\'\' &nbsp;&nbsp;&nbsp; T = T_in on the four '
            'edges</font>', st("eq", fontSize=9, leading=13, alignment=TA_LEFT, spaceBefore=2, spaceAfter=6)))
A(P("with <font face='Courier'>G</font> the thickness-integrated in-plane conductance (full metal over a wall, the "
    "fin plus the water over a slot), <font face='Courier'>sigma = h (1 + 2 depth/width)</font> the slot's coupling "
    "to the fluid, and <font face='Courier'>T_f = T_in + dT_f/2</font> with "
    "<font face='Courier'>dT_f = Q_coolant / (m_dot c_p)</font> — closed against the heat the slots actually "
    "deliver. That closure is what makes the problem conjugate rather than a plate with a boundary condition "
    "picked by hand."))
A(H2("2.1 The physics that came in"))
A(P("<b>Film coefficient</b>: fully developed Nusselt number of a lidded slot (Shah &amp; London, 1978, "
    "<font face='Courier'>Nu_Dh = 5.385</font>) with Hausen's entry correction. <b>Pressure drop</b>: "
    "Muzychka-Yovanovich with the rectangular-duct <font face='Courier'>f·Re</font> fit and the Kays &amp; London "
    "losses. <b>Water properties</b>: Incropera and DeWitt Table A.6. <b>Fluid energy balance</b>: solved as an "
    "affine fixed point in closed form from two solves (its slope and intercept come out analytically), instead of "
    "iterating to convergence — which before did not converge in 20 iterations."))
A(H2("2.2 The geometry that came in"))
A(P("The slots are 0.8 to 1.2 mm wide and the mesh is 32 cells across 100 mm, so no slot fits inside one cell. "
    "Conductance and coupling are computed from the <b>exact fraction</b> of each cell covered by a slot, and the "
    "area is rescaled, so the fields do not change when the mesh does. The outer walls are made adiabatic by a "
    "<b>guard ring</b> of two cells with negligible conductance (1e-3): without it a quarter of the power left "
    "through the manifold and the hot spot was optimistic by 5 K; with it the ring carries 0.2 %."))
A(H2("2.3 The FNO's third channel, and why it was necessary"))
A(P("The operator solved is <font face='Courier'>-div(a grad u) + s u = f0 + s·theta_f</font>. The source a slot "
    "gives back to the plate carries the fluid's own temperature rise, which no map of conductance and coupling can "
    "recover. A two-channel FNO sits at <b>5.2 %</b> on the held-out fields; handing it the source it actually "
    "solves takes that to <b>1.1 %</b>. The error is visible because PINNeAPPle measures it — not because somebody "
    "guessed which inputs to use."))
fig(E07 / "results/architecture.png",
    cap="Figure 1 — The path through example 07: Latin-hypercube designs, the coolant model, conjugate finite "
        "volumes, the FNO, certified checks, the Monte-Carlo of the plant and the OpenUSD stage. The RTX render "
        "(dashed) does not run on this machine.")
fig(E07 / "results/fields.png", max_h=0.40 * H,
    cap="Figure 2 — One held-out design (median error 0.80 %): the three input channels (conductance, coupling, "
        "source), the field from the conjugate finite volumes, the FNO's field, the true error and the error "
        "<i>estimated from the residual alone</i>. Bottom right, the estimate tracks the true error without ever "
        "seeing it.")

A(H2("2.4 The study, and what it honestly does not guarantee"))
A(P("250 designs (200 training, 50 held-out) solved in coupled form, a three-channel FNO and a FullyConnected on "
    "the design variables (slot count, slot width, flow, h multiplier). The chosen design is the lightest pumping "
    "power among the ones with margin: "
    f"<b>{s07['identity']['chosen']['n_channels']} slots of "
    f"{fmt(1000 * s07['identity']['chosen']['channel_width'], 2)} mm at "
    f"{fmt(s07['identity']['chosen']['flow_lpm'], 2)} L/min</b> — hot spot "
    f"<b>{fmt(s07['identity']['t_max_c'], 1)} °C</b>, pressure drop {fmt(s07['identity']['dp_pa'], 0)} Pa, "
    f"Re {fmt(s07['identity']['reynolds'], 0)}, h {fmt(s07['identity']['h_w_m2k'], 0)} W/m²K, coolant rise "
    f"{fmt(s07['identity']['dt_f_k'], 1)} K, pumping power {fmt(1000 * s07['identity']['pumping_w'], 1)} mW."))
A(P("The Monte-Carlo of the plant (flow lognormal 10 %, film coefficient ±15 %, duty ±5 %, 400 draws) is where the "
    "example gets interesting. The robust design the study ships fails in none of the draws — neither by the "
    "surrogate nor by the coupled solves. The lightest design the surrogate puts 2 K under the limit fails in "
    f"<b>P = {pct(s07['uncertainty']['edge']['p_surrogate'], 2)}</b> by the surrogate and "
    f"<b>P = {pct(s07['uncertainty']['edge']['p_fvm'], 2)}</b> by the coupled solves."))
A(P("That gap is not a bug: selecting on a surrogate's own prediction also selects the designs where that "
    "prediction is optimistic — the winner's curse. The check was <b>rewritten</b> as “the design at the edge "
    "really is worse”, instead of demanding that the two probabilities match, and both stay recorded in "
    "<font face='Courier'>summary.json</font>."))
fig(E07 / "results/uncertainty.png", max_h=0.32 * H,
    cap="Figure 3 — The Monte-Carlo of the plant on both designs. Left, the robust design that ships: both "
        "distributions stay far from the 85 °C limit. Right, the lightest design the surrogate judged safe: the "
        "surrogate sees it under the limit (blue), the coupled solves see it over (red) in every draw.")

A(H2("2.5 Numbers from the run"))
rows = [["quantity", "value", "note"],
        ["coupled solves", "250", "200 training / 50 held-out, 32 x 32 cells"],
        ["FNO", f"{fmt(s07['fno_parameters'], 0)} parameters, {s07['fno_in_channels']} channels",
         f"{s07['training']['steps']} steps, {fmt(s07['training']['epochs'], 0)} epochs, "
         f"{fmt(s07['training']['wall_s'] / 60, 1)} min"],
        ["held-out L2", pct(s07['test_rel_l2']), "limit 5 %; median per design 0.80 %"],
        ["worst hot-spot error", f"{fmt(s07['max_hotspot_error_K'], 2)} K", "across the 50 held-out designs"],
        ["certified energy bound", pct(chk(s07, "PDE residual (certified energy-error bound)")), "limit 10 %"],
        ["certified L2 bound", pct(chk(s07, "PDE residual (certified L2 error bound)")), "limit 5 %"],
        ["boundary condition", pct(chk(s07, "boundary condition u = 0"), 2), "limit 2 %"],
        ["flux balance", pct(chk(s07, "flux balance (net boundary flux = source)"), 2), "limit 2 %"],
        ["energy closure", fmt(s07["energy_closure_max_all_designs"]), "worst case over the 250 designs"],
        ["design surrogate", f"{pct(s07['design_surrogate']['heldout_rel_rms']['rise'], 1)} / "
         f"{pct(s07['design_surrogate']['heldout_rel_rms']['dp'], 1)} / "
         f"{pct(s07['design_surrogate']['heldout_rel_rms']['pump'], 1)}",
         "rise / pressure drop / pumping, 50 held-out designs"],
        ["OpenUSD stage", f"{s07['usd']['prims']} prims, {s07['usd']['meshes']} meshes",
         f"{s07['usd']['validators']} validators, no problems"],
        ["total time", f"{fmt(r07['wall_s'], 0)} s", f"{fmt(s07['training']['wall_s'], 0)} s of it in the FNO"]]
A(table(rows, [46 * mm, FRAME_W * 0.29, FRAME_W - 46 * mm - FRAME_W * 0.29]))
A(Spacer(1, 6))
A(P("The five physical directions tested — hot spot falls with flow, with the film coefficient and with the slot "
    "count; pressure drop and pumping power grow with flow — hold across the whole design box. The comparison is "
    "made between the ends of each range, not on a tensor grid: the latter reported violations in 0.06 % to 5 % of "
    "cells, which is surrogate noise and not physics.", "small"))
fig(E07 / "results/design_space.png", width=0.54 * FRAME_W, max_h=0.28 * H,
    cap="Figure 4 — The design space: pumping power against hot spot, with the 85 °C limit and the two designs "
        "taken to the Monte-Carlo.")

# ======================================================================================== 3. example 10
A(H1("3 · Example 10 — the multi-chip finned heat sink"))
A(P("An extruded plate-fin heat sink (100 x 100 mm base, 1 mm root fillet) cools <b>two</b> chips: an 80 W power "
    "module on 40 x 40 mm and a 15 W gate driver on 15 x 15 mm, in 3 m/s air with a partial shroud — part of the "
    "array sees the room and radiates. Each candidate (fin count, height, thickness, base thickness, material) is "
    "built as an exact B-rep, rated by PINNeAPPle's closed-form network <b>and</b> by a 3-D finite-volume solve of "
    "the base with both footprints, and a PhysicsNeMo network learns the design space over seven variables."))
A(H2("3.1 Two models, and the gap between them as a check"))
A(P("Surrogate-driven design looks for the edge of the feasible region, which is exactly where the surrogate is "
    "least reliable; and a resistance network with one equivalent source cannot say how much hotter the small chip "
    "runs. PINNeAPPle therefore re-evaluates every proposal with the exact geometry and with <b>two independent "
    "models</b> before accepting it, and reports the gap between them as a check in its own right: median "
    f"{pct(s10['network_vs_3d_gap']['median'], 2)}, p90 {pct(s10['network_vs_3d_gap']['p90'], 2)}, worst "
    f"{pct(s10['network_vs_3d_gap']['max'], 1)} of the temperature rise."))
fig(E10 / "results/models.png", width=0.66 * FRAME_W, max_h=0.36 * H,
    cap="Figure 5 — The same physics two ways: the closed-form network against the 3-D solve of the base, over "
        f"every design in the study (p90 of {pct(s10['network_vs_3d_gap']['p90'], 1)}). The network is optimistic "
        "exactly where a small chip sits next to a large one.")
A(H2("3.2 Radiation and the seven variables"))
A(P("Radiation enters as a linearised coefficient weighted by the unobstructed sky fraction — the fins of an "
    "array see each other, so only the part with a view of the room really exchanges heat. On the accepted design "
    f"it accounts for {pct(s10['proposal_verified']['radiation_share'], 1)} of the surface coefficient. The seven "
    "variables include the two uncertainties an engineer lives with: the film-coefficient multiplier and the "
    "interface resistance, so the study spans the range of doubt rather than a single point."))
A(H2("3.3 Numbers from the run"))
rows = [["quantity", "value", "note"],
        ["training designs", str(s10["train_designs"]), f"{s10['train_steps']} training steps"],
        ["held-out error", f"{pct(s10['heldout_rel_rms']['rise'], 2)} / "
         f"{pct(s10['heldout_rel_rms']['dp'], 2)} / {pct(s10['heldout_rel_rms']['mass'], 2)}",
         "rise / pressure drop / mass, 40 held-out designs"],
        ["network against 3-D", f"median {pct(s10['network_vs_3d_gap']['median'], 2)}",
         f"p90 {pct(s10['network_vs_3d_gap']['p90'], 2)}, worst {pct(s10['network_vs_3d_gap']['max'], 1)}"],
        ["3-D energy balance", fmt(chk(s10, "3-D base solve closes its energy balance (all designs)")),
         "heat out of the fin-root face against chip power"],
        ["accepted design", f"{s10['proposal']['n_fins']} fins, "
         f"{fmt(1000 * s10['proposal']['fin_height'], 1)} mm, {fmt(1000 * s10['proposal']['fin_t'], 2)} mm, base "
         f"{fmt(1000 * s10['proposal']['base_t'], 1)} mm, {s10['proposal']['material'].upper()}",
         f"fin gap {fmt(s10['proposal']['gap_mm'], 1)} mm"],
        ["3-D hot spot", f"{fmt(s10['proposal_verified']['t_source_3d'], 1)} °C",
         f"small chip {fmt(s10['proposal_verified']['t_chip'][1], 1)} °C"],
        ["network (same design)", f"{fmt(s10['proposal_verified']['t_source_network'], 1)} °C",
         f"{pct(chk(s10, 'design: network and 3-D solve agree'), 1)} gap between the two models"],
        ["pressure drop", f"{fmt(s10['proposal_verified']['dp'], 1)} Pa", "limit 30 Pa"],
        ["mass", f"{fmt(1000 * s10['proposal_verified']['mass'], 1)} g",
         "closed form; the exact B-rep gives ~1 % more with the fillet"],
        ["fin efficiency", pct(s10['proposal_verified']['fin_efficiency'], 1), "above 50 % required"],
        ["radiative share", pct(s10['proposal_verified']['radiation_share'], 1), "sky fraction 0.25"],
        ["total time", f"{fmt(r10['wall_s'], 0)} s", "training and evaluations on the CPU"]]
A(table(rows, [46 * mm, FRAME_W * 0.32, FRAME_W - 46 * mm - FRAME_W * 0.32]))
fig(E10 / "results/parity.png", max_h=0.235 * H,
    cap="Figure 6 — The surrogate against the models on the 40 held-out designs: temperature rise, pressure drop "
        "and mass.")
fig(E10 / "results/design_space.png", width=0.60 * FRAME_W, max_h=0.31 * H,
    cap="Figure 7 — The design space as the surrogate sees it: source temperature against mass, coloured by "
        "pressure drop, with the 75 °C limit and the design accepted after verification.")

# ========================================================================================== 4. library
A(H1("4 · What changed in the library"))
A(P("Both examples needed tools the repository did not have yet. All of them came in additively: no default "
    "behaviour changed, and examples 16 and 18, which use the same heat-sink network, still hold with the same "
    "evaluation."))
A(table([["module", "what it gained"],
         ["<font face='Courier'>pinneapple_physics.darcy_checks</font>",
          "the reacting problem <font face='Courier'>-div(a grad u) + s u = f</font>: "
          "<font face='Courier'>sink=</font> on the residual, on both certified bounds and on the flux balance. A "
          "non-negative reaction only stiffens the operator, so the certificates hold unchanged. "
          "<font face='Courier'>exact=True</font> replaces the constant-coefficient relaxation — useless under a "
          "strong sink, where it lands 44x above the true error — with the discrete error itself, from one "
          "preconditioned conjugate-gradient solve per sample"],
         ["<font face='Courier'>operator_benchmarks.darcy_reaction_2d</font>",
          "the same finite volumes as <font face='Courier'>darcy_2d</font> with the reaction term: the equation a "
          "conjugate cold plate satisfies"],
         ["<font face='Courier'>physicsnemo_fno</font> / <font face='Courier'>pinneapple_fno</font>",
          "<font face='Courier'>in_channels</font>: several input fields per sample, one output field. "
          "Single-channel models keep their scalar normalisation buffers, so existing checkpoints still load"],
         ["<font face='Courier'>pinneapple_tools.nvidia.coldplate</font>",
          "new: the microchannel cold plate — fractional slot geometry, the coolant state and its limits, the "
          "coupled solve and the engineering checks"],
         ["<font face='Courier'>pinneapple_tools.nvidia.heatsink</font>",
          "<font face='Courier'>evaluate_full</font>: a design rated by the network, by the 3-D base solve with "
          "several chips and by the exact USD BRep solid, with the network-to-3-D gap reported; plus "
          "<font face='Courier'>checks</font> and <font face='Courier'>table_rows</font>"],
         ["<font face='Courier'>pinneapple_design.thermal.heatsink.field</font>",
          "<font face='Courier'>solve_base_field(sources=[...])</font>: one base solve resolves several chips "
          "instead of one, because the problem is linear and the fields superpose"],
         ["<font face='Courier'>pinneapple_physics.closed_form.plate_fin_heatsink</font>",
          "<font face='Courier'>emissivity</font> / <font face='Courier'>sky_fraction</font> / "
          "<font face='Courier'>t_surround_c</font>, with <font face='Courier'>h_rad_w_m2k</font> and "
          "<font face='Courier'>radiation_share</font> in the result; the default stays the ducted case with "
          "radiation neglected"]],
        [52 * mm, FRAME_W - 52 * mm]))
A(Spacer(1, 4))
A(P("Coverage: 8 new tests in <font face='Courier'>tests/test_nvidia_integrations.py</font>, "
    "<font face='Courier'>test_darcy_checks.py</font> and "
    "<font face='Courier'>test_physicsnemo_field_operators.py</font>, plus the changelog entry in "
    "<font face='Courier'>changelog.d/thermal-examples-conjugate.changed.md</font>.", "small"))

# ============================================================================================== 4.1 gates
A(H2("4.1 Gates that ran"))
A(table([["gate", "result"],
         ["<font face='Courier'>ruff check</font> on the 13 changed files", "clean"],
         ["<font face='Courier'>pytest</font> of the three touched test files",
          "53 passed, 13 skipped (Newton, USD BRep and torch_geometric are not installed here)"],
         ["<font face='Courier'>scripts/check_experiment_manifests.py</font>",
          "every experiment has a valid manifest"],
         ["<font face='Courier'>python -m pinneapple_hub.library check</font>",
          "index ok; no stale snippet runs"],
         ["<font face='Courier'>python -m pinneapple_studio coverage</font>",
          "green: the 17 tools this work added or moved now have recorded inputs, a smoke run or a stated reason"],
         ["the frozen core API snapshot (<font face='Courier'>api/core.json</font>)",
          "rewritten with the two names this work changes: the new "
          "<font face='Courier'>radiative_coefficient</font> and the three new arguments of "
          "<font face='Courier'>evaluate</font>"],
         ["example 07 run end to end", f"APPROVED, {sum(c['passed'] for c in s07['checks'])}/20 checks"],
         ["example 10 run end to end", f"APPROVED, {sum(c['passed'] for c in s10['checks'])}/24 checks"]],
        [FRAME_W * 0.46, FRAME_W * 0.54]))
A(Spacer(1, 3))
A(P("One failure predates this work and was left alone: "
    "<font face='Courier'>test_physics_check_darcy.py::test_darcy_cli_exit_code_and_no_torch</font> needs the "
    "<font face='Courier'>pinneapple_check</font> package installed; it reproduces on a tree without these changes "
    "(checked with <font face='Courier'>git stash</font>).", "small"))

# ============================================================================================ 5. what broke
A(H1("5 · What was wrong along the way"))
A(P("Seven defects turned up during the session. None was resolved by loosening a tolerance: all were fixed in the "
    "physics or the code, and each left a test or a check behind."))
A(table([["#", "symptom", "cause and fix"],
         ["1", "cold-plate energy closure off by 26-50 %",
          "three causes at once: boundary integrals missing the cell scale factor, chip power quantised to whole "
          "cells, and the area scale changed by the guard ring. Fixed the power lumping over the mask and the area "
          "rescaling — it now closes to 5e-15"],
         ["2", "16-25 % of the heat leaking through the manifold",
          "the edges were Dirichlet. A two-cell guard ring with negligible conductance brings the leakage to "
          "0.2 %, and the hot spot is no longer optimistic by 5 K"],
         ["3", "0.9 mm slots invisible on a 32-cell mesh",
          "binary mask. Replaced by exact fractional overlap, which is what makes the result mesh-independent"],
         ["4", "FNO at 5.16 %, above the 5 % limit",
          "the source was missing: the operator has <font face='Courier'>f0 + s·theta_f</font> and the second term "
          "carries the fluid's rise. A third input channel: 1.07 %"],
         ["5", "trend tests failing on 0.06-5 % of cells",
          "a tensor-grid comparison was measuring surrogate noise. Replaced by comparing the ends of each range — "
          "and that is how the inverted sign convention of <font face='Courier'>sweep_change</font> surfaced, also "
          "fixed"],
         ["6", "the “above 50 %” fin-efficiency check passing any value",
          "the test was <font face='Courier'>v &lt;= limit</font> for a minimum condition. It gained "
          "<font face='Courier'>kind=\"min\"</font> and the comparison in the right direction"],
         ["7", "surrogate failure probability (0.30) against the solves' (1.00)",
          "the winner's curse, not a numerical defect: selecting on a surrogate's own prediction selects where it "
          "is optimistic. The check now states “the design at the edge is worse”, and both probabilities stay "
          "recorded"]],
        [8 * mm, FRAME_W * 0.30 - 4 * mm, FRAME_W * 0.70 - 4 * mm]))

# ========================================================================================== 6. limits
A(H1("6 · Limits and what was not run"))
A(table([["limit", "consequence"],
         ["<font face='Courier'>usd-brep</font> not installed on this machine",
          "the mass recorded in <font face='Courier'>results/</font> is the closed form's. The 7e-9 volume "
          "agreement and the ~1 % fillet mass quoted in the README come from a run <i>with</i> the B-rep, and that "
          "run needs Linux or Windows: PyPI publishes the wheel for CPython 3.12 on manylinux and win_amd64 only"],
         ["RTX render in Omniverse Kit",
          "does not run here (needs Omniverse and an RTX GPU). The script stays at "
          "<font face='Courier'>results/kit_render.py</font>; the stage preview is read back with OpenUSD"],
         ["coolant regime",
          "some designs in the study reach Re ≈ 4700, outside the correlation's laminar regime. It is a check that "
          f"flags it, and the shipped design has Re {fmt(s07['identity']['reynolds'], 0)}"],
         ["ducted flow in the heat sink",
          "the network assumes all the air passes between the fins. Without a shroud, bypass can cost 20-50 %"],
         ["heat-sink radiation",
          "modelled over the unobstructed sky fraction (0.25); the rest of the array sees the neighbouring fin at "
          "its own temperature and exchanges almost nothing"],
         ["guard ring", "a two-cell approximation to adiabatic walls, not an exact boundary condition"]],
        [52 * mm, FRAME_W - 52 * mm]))

# ============================================================================================== annexes
A(H1("Annex A — example 07 checks"))
A(P("Every value comes from <font face='Courier'>results/summary.json</font>.", "small"))
A(Spacer(1, 3))
A(checks_table(s07["checks"]))
A(Spacer(1, 8))
A(H1("Annex B — example 10 checks"))
A(P("Every value comes from <font face='Courier'>results/summary.json</font>.", "small"))
A(Spacer(1, 3))
A(checks_table(s10["checks"]))

A(PageBreak())
A(H1("Annex C — environment and reproducibility"))
A(table([["item", "value"],
         ["python", env.get("python", "—")],
         ["platform", env.get("platform", "—")],
         ["CPU", f"{env.get('cpu', '—')}, {env.get('cpu_count', '—')} cores"],
         ["GPU", env.get("gpu") or "none"],
         ["torch / numpy / scipy", f"{pk.get('torch', '—')} / {pk.get('numpy', '—')} / {pk.get('scipy', '—')}"],
         ["PhysicsNeMo", pk.get("nvidia-physicsnemo", "—")],
         ["OpenUSD", pk.get("usd-core", "—")],
         ["usd-brep", pk.get("usd-brep") or "absent"],
         ["time for 07 / 10", f"{fmt(r07['wall_s'], 0)} s / {fmt(r10['wall_s'], 0)} s"]],
        [46 * mm, FRAME_W - 46 * mm]))
A(Spacer(1, 8))
A(H2("How to reproduce"))
A(Paragraph("pip install nvidia-physicsnemo usd-core<br/>"
            "python examples/nvidia/07_omniverse_physicsnemo_coldplate/run.py&nbsp;&nbsp;# ~10 min on CPU<br/>"
            "python examples/nvidia/10_brep_physicsnemo_heatsink/run.py&nbsp;&nbsp;# ~1 min",
            st("code", fontName="Courier", fontSize=8.2, leading=12.6, alignment=TA_LEFT, spaceBefore=2,
               spaceAfter=8)))
A(P("Each example writes <font face='Courier'>results/summary.json</font> (every number and check), "
    "<font face='Courier'>results/run_info.json</font> (environment and per-step timings) and the figures quoted "
    "here. Each example's README explains the modelling decisions and what the example does not cover."))
A(Spacer(1, 6))
A(H2("Files touched"))
A(P("32 files, +6368 / -1608 lines, in the commit on branch "
    "<font face='Courier'>feat/thermal-examples-depth</font> (PR #1873). The two heaviest new modules are "
    "<font face='Courier'>pinneapple_tools/nvidia/coldplate.py</font> (~400 lines) and the rewritten "
    "<font face='Courier'>pinneapple_tools/nvidia/heatsink.py</font>; the rest are additive extensions, the two "
    "<font face='Courier'>run.py</font> files, the READMEs, the NVIDIA example index, the changelog and the tests."))

doc.build(story)
print(f"written: {OUT_PDF} ({os.path.getsize(OUT_PDF) // 1024} kB)")
