"""Build the fictitious input documents used by the demo and the tests (needs reportlab), one folder per format.

Each set describes one made-up item of the same fictitious gas plant the way different parties would (process
datasheet, contractor's mechanical datasheet, owner's specification, calculations). One conflict between documents
and one missing required item are planted in every set:

  asme_u-dr-1  separator V-101       conflict: shell corrosion allowance 6 mm (owner) vs 3 mm (mechanical DS)
                                     gap: cyclic service
  psv          relief valve PSV-101  conflict: set pressure 15 barg (relief load summary) vs 14.5 barg (sizing calc)
                                     gap: valve type (conventional / balanced bellows / pilot)
  shell_tube   gas cooler E-101      conflict: tube-side design pressure 15 barg (process DS) vs 16 barg (mechanical DS)
                                     gap: TEMA class
  tank         diesel tank T-201     conflict: shell corrosion allowance 3 mm (tank spec) vs 2 mm (mechanical DS)
                                     gap: maximum emptying rate
  pump         condensate pump P-101 conflict: NPSH available 4.2 m (process DS) vs 3.8 m (hydraulic calc)
                                     gap: viscosity

The PSV set's relief load summary is a scanned copy (image only, slightly rotated, noisy, JPEG), so the example
exercises the OCR path.

Run: python make_samples.py  (writes the PDFs into the folders next to this file; needs reportlab, pypdfium2, Pillow)
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
ST = getSampleStyleSheet()
BANNER = "SAMPLE DOCUMENT - FICTITIOUS DATA FOR DEMONSTRATION ONLY"


def table(rows, widths=(220, 260)):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
    return t


def build(name, title, blocks):
    os.makedirs(os.path.dirname(os.path.join(HERE, name)), exist_ok=True)
    doc = SimpleDocTemplate(os.path.join(HERE, name), pagesize=A4, title=title)
    story = [Paragraph(BANNER, ST["Italic"]), Paragraph(title, ST["Title"])]
    for b in blocks:
        story.append(b if not isinstance(b, str) else Paragraph(b, ST["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)


build("asme_u-dr-1/process_datasheet_V-101.pdf", "Process Datasheet - Separator V-101", [
    "<b>Project:</b> Demo Gas Plant Expansion (fictitious). Document PDS-V-101 Rev B.",
    table([["Item / Tag No.", "V-101"], ["Service", "HP Gas/Condensate Separator"],
           ["Fluid", "Natural gas, hydrocarbon condensate, produced water"], ["Lethal service", "No"],
           ["Corrosive service", "Yes (wet CO2)"], ["Specific gravity", "0.72"],
           ["Normal liquid level", "900 mm from bottom tangent"]]),
    "<b>Operating conditions</b>",
    table([["Case 1 minimum operating pressure", "8 barg"], ["Case 1 maximum operating pressure", "12 barg"],
           ["Case 1 minimum operating temperature", "5 °C"], ["Case 1 maximum operating temperature", "85 °C"],
           ["Case 2 minimum operating pressure", "0 barg"], ["Case 2 maximum operating pressure", "3 barg"],
           ["Case 2 minimum operating temperature", "-20 °C"], ["Case 2 maximum operating temperature", "40 °C"]]),
    "<b>Design conditions</b>",
    table([["Internal design pressure", "15 barg"], ["Design temperature", "120 °C"],
           ["External design pressure", "Full vacuum"], ["Temperature at external design pressure", "120 °C"],
           ["MDMT", "-29 °C"], ["MDMT coincident pressure", "15 barg"], ["MDMT due to", "Ambient temperature"]]),
    "Overpressure protection: pressure relief valve PSV-101 set at design pressure.",
    "Note: depressurisation case 2 is the start-up after blowdown.",
])

build("asme_u-dr-1/mechanical_datasheet_V-101.pdf", "Mechanical Datasheet - Separator V-101", [
    "Contractor document MDS-V-101 Rev 1. Design code: ASME BPVC Section VIII, Division 1, 2025 edition.",
    table([["Equipment No.", "V-101"], ["Orientation", "Vertical"], ["Inside diameter", "1800 mm"],
           ["Length, tangent-to-tangent", "5400 mm"], ["Vessel support", "Skirt"],
           ["Shell corrosion allowance internal", "3 mm"], ["Heads corrosion allowance internal", "3 mm"],
           ["Nozzles corrosion allowance internal", "3 mm"], ["Shell joint efficiency", "1.0"],
           ["Heads joint efficiency", "1.0"], ["PWHT", "Per code"], ["Insulated", "Yes"],
           ["Insulation by", "Others"], ["External insulation", "Mineral wool, 50 mm"],
           ["Insulation density", "128 kg/m3"], ["MAWP basis", "Calculated by manufacturer"]]),
    "<b>Materials</b>",
    table([["Shell", "SA-516 Gr. 70N"], ["2:1 ellipsoidal heads", "SA-516 Gr. 70N"], ["Nozzles", "SA-106 Gr. B"],
           ["Flanges", "SA-105N"], ["Bolting", "SA-193 B7 / SA-194 2H"], ["Reinforcing pads", "SA-516 Gr. 70N"],
           ["Internals", "SS 316L"], ["Attachments", "SA-516 Gr. 70"]]),
    "<b>Nozzle schedule</b>",
    table([["Mark", "Service", "Qty", "Size", "Flange type", "Class"],
           ["N1", "Gas/liquid inlet", "1", "8 in", "WNRF", "300"], ["N2", "Gas outlet", "1", "6 in", "WNRF", "300"],
           ["N3", "Liquid outlet", "1", "3 in", "WNRF", "300"], ["N4", "PSV connection", "1", "4 in", "WNRF", "300"],
           ["N5", "Level instrument", "2", "2 in", "WNRF", "300"], ["M1", "Manway", "1", "24 in", "WNRF", "300"]],
          widths=(50, 150, 40, 60, 90, 60)),
    "<b>Welded joints</b>",
    table([["Category A joint type", "Type 1"], ["Category A NDE", "Full RT per UW-51"],
           ["Category B head-to-shell joint type", "Type 1"], ["Category B head-to-shell NDE", "Full RT per UW-51"],
           ["Category D joint type", "Full penetration"], ["Category D NDE", "MT after PWHT"]]),
])

build("asme_u-dr-1/owner_specification_V-101.pdf", "Owner Project Specification - Pressure Vessels (extract)", [
    "Specification SPEC-PV-001 Rev 3, issued by the owner for all pressure vessels of the project.",
    table([["Owner", "Demo Energy Ltd. (fictitious)"], ["Operator", "Demo Operations Co."],
           ["Country of installation", "Brazil"], ["State/Province of installation", "Rio de Janeiro"],
           ["City of installation", "Macaé"], ["National Board registration required", "Yes"],
           ["Canadian registration required", "No"], ["Design life", "25 years"]]),
    "<b>Corrosion</b>",
    "Shell corrosion allowance internal: 6 mm minimum for wet CO2 service.",
    "<b>Loadings</b>",
    table([["Wind loading code", "ASCE 7-22"], ["Basic wind speed", "42 m/s"], ["Risk category", "II"],
           ["Exposure category", "C"], ["Topographic factor", "1.0"], ["Site elevation", "15 m"],
           ["Seismic loading code", "ASCE 7-22"], ["Site class", "D"]]),
    "<b>Fabrication and painting</b>",
    table([["Coating specification", "SPEC-PT-004 system C"],
           ["Coating permitted prior to pressure test", "No"], ["Fireproofing", "No"]]),
    "General notes: all welding procedures to be qualified per ASME IX. Hydrotest water chloride below 50 ppm.",
])

def sides(rows):
    """Item | Unit | Shell side | Tube side table."""
    t = Table([["Item", "Unit", "Shell side", "Tube side"]] + rows, colWidths=(190, 70, 110, 110))
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    return t


# ---------------------------------------------------------------- PSV-101
build("psv/_relief_load_summary_PSV-101_clean.pdf", "Relief Load Summary - PSV-101", [
    "Process document RLS-101 Rev A. Relief study for separator V-101 (fictitious).",
    table([["Tag No.", "PSV-101"], ["Protected equipment", "V-101 HP Gas/Condensate Separator"],
           ["P&ID", "PID-100-002"], ["Governing relief case", "External fire"], ["Fluid", "Hydrocarbon vapour"],
           ["Fluid state", "Vapour"], ["Required relieving capacity", "18,500 kg/h"], ["Molecular weight", "19.6"],
           ["Ratio of specific heats (Cp/Cv)", "1.27"], ["Compressibility factor Z", "0.92"],
           ["Operating temperature", "85 °C"], ["Relieving temperature", "160 °C"],
           ["Operating pressure", "12 barg"], ["Set pressure", "15 barg"], ["Allowable overpressure", "21 % (fire case)"],
           ["Superimposed back pressure", "0.5 barg"], ["Built-up back pressure", "0.7 barg"]]),
    "Other cases checked: blocked outlet (8,200 kg/h) and gas blow-by (not credible, control valve sized below "
    "the separator design pressure).",
])
build("psv/psv_sizing_PSV-101.pdf", "PSV Sizing Calculation - PSV-101", [
    "Calculation CAL-PSV-101 Rev 0, per API 520 Part I, critical vapour flow, Kd 0.975, Kb 1.0, Kc 1.0.",
    table([["Tag No.", "PSV-101"], ["Set pressure", "14.5 barg"], ["Relieving temperature", "160 °C"],
           ["Required relieving capacity", "18500 kg/h"], ["Coefficient of discharge Kd", "0.975"],
           ["Calculated orifice area", "2,650 mm2"], ["Selected orifice", "P (4P6)"],
           ["Selected orifice area", "4,116 mm2"], ["Rated capacity", "28,700 kg/h"]]),
    "Inlet pressure loss below 3 % of set pressure; built-up back pressure below 10 % of set pressure.",
])
build("psv/valve_specification_PSV-101.pdf", "Pressure Relief Valve Specification (extract)", [
    "Owner specification SPEC-PRV-002 Rev 1, applicable to PSV-101.",
    table([["Design code", "ASME BPVC VIII-1 / API 526"], ["Inlet", "4 in CL300 RF"], ["Outlet", "6 in CL150 RF"],
           ["Flange facing", "RF, 125-250 AARH"], ["Body and bonnet material", "SA-216 WCB"],
           ["Nozzle and disc material", "SS 316"], ["Spring material", "Chrome alloy steel"],
           ["Lifting lever", "Yes, packed"], ["Test gag", "No"], ["Quantity", "1"]]),
])

# ---------------------------------------------------------------- E-101
build("shell_tube/process_datasheet_E-101.pdf", "Process Datasheet - Gas Cooler E-101", [
    "Process document PDS-E-101 Rev B (fictitious).",
    table([["Item No.", "E-101"], ["Service", "HP gas cooler"], ["TEMA type", "AES"], ["Orientation", "Horizontal"],
           ["Heat duty", "1.45 MW"], ["Shells", "1 (no series/parallel)"]]),
    sides([["Fluid name", "-", "Cooling water", "Natural gas"], ["Total flow", "kg/h", "125,000", "46,000"],
           ["Temperature in", "°C", "30", "85"], ["Temperature out", "°C", "40", "45"],
           ["Operating pressure, inlet", "barg", "4", "12"], ["Allowable pressure drop", "bar", "0.7", "0.5"],
           ["Fouling resistance", "m2K/W", "0.00035", "0.00018"], ["Design pressure", "barg", "7", "15"],
           ["Design temperature", "°C", "65", "120"]]),
])
build("shell_tube/mechanical_datasheet_E-101.pdf", "Mechanical Datasheet - Gas Cooler E-101", [
    "Contractor document MDS-E-101 Rev 1. Design code: ASME BPVC Section VIII, Division 1.",
    table([["Item No.", "E-101"], ["Size", "800 x 6096 mm"], ["Effective surface per shell", "142 m2"]]),
    sides([["Design pressure", "barg", "7", "16"], ["Design temperature", "°C", "65", "120"],
           ["MDMT", "°C", "0", "-10"], ["Test pressure", "barg", "9.1", "20.8"], ["Number of passes", "-", "1", "2"],
           ["Corrosion allowance", "mm", "3", "3"], ["Inlet connection", "-", "8 in CL150 RF", "6 in CL300 RF"],
           ["Outlet connection", "-", "8 in CL150 RF", "6 in CL300 RF"]]),
    "<b>Construction</b>",
    table([["Tube OD", "19.05 mm"], ["Tube thickness", "2.11 mm (14 BWG)"], ["Tube length", "6096 mm"],
           ["Tube pitch", "23.81 mm"], ["Tube layout angle", "30°"], ["Number of tubes", "412"],
           ["Tube type", "Plain, seamless"], ["Tube material", "SA-179"], ["Shell ID", "800 mm"],
           ["Shell material", "SA-516 Gr. 70"], ["Channel material", "SA-516 Gr. 70"],
           ["Tubesheet material", "SA-266 Gr. 2"], ["Baffle type", "Single segmental"], ["Baffle cut", "25 %"],
           ["Baffle spacing", "450 mm"], ["Gaskets", "Spiral wound SS316/graphite"], ["Empty weight", "7,800 kg"]]),
])

# ---------------------------------------------------------------- T-201
build("tank/process_datasheet_T-201.pdf", "Process Datasheet - Diesel Storage Tank T-201", [
    "Process document PDS-T-201 Rev A (fictitious).",
    table([["Tag No.", "T-201"], ["Service", "Diesel storage"], ["Product stored", "Diesel oil"],
           ["Specific gravity", "0.85"], ["Vapour pressure", "0.01 bar(a)"], ["Operating temperature", "35 °C"],
           ["Maximum filling rate", "600 m3/h"], ["Nominal capacity", "10,000 m3"],
           ["Net working capacity", "8,800 m3"]]),
])
build("tank/mechanical_datasheet_T-201.pdf", "Mechanical Datasheet - Diesel Storage Tank T-201", [
    "Contractor document MDS-T-201 Rev 0.",
    table([["Tag No.", "T-201"], ["Nominal diameter", "30 m"], ["Shell height", "15 m"],
           ["Maximum design liquid level", "14.2 m"], ["Design pressure", "20 mbarg"], ["Design vacuum", "2.5 mbarg"],
           ["Maximum design temperature", "60 °C"], ["Design metal temperature", "5 °C"],
           ["Shell corrosion allowance", "2 mm"], ["Bottom corrosion allowance", "3 mm"],
           ["Roof corrosion allowance", "1.5 mm"], ["Roof type", "Fixed cone, self-supporting"],
           ["Bottom type", "Cone up, slope 1:100"], ["Annular plates", "Yes"], ["Shell material", "ASTM A516 Gr. 70"],
           ["Bottom material", "ASTM A36"], ["Roof material", "ASTM A36"], ["Foundation", "Concrete ringwall"],
           ["Insulation", "No"], ["Heating coil", "No"], ["Hydrotest medium", "Fresh water"]]),
    "<b>Nozzle schedule</b>",
    table([["Mark", "Service", "Qty", "Size", "Rating", "Facing"],
           ["N1", "Inlet", "1", "12 in", "CL150", "RF"], ["N2", "Outlet", "1", "10 in", "CL150", "RF"],
           ["N3", "Drain", "1", "4 in", "CL150", "RF"], ["N4", "Vent", "2", "8 in", "CL150", "FF"],
           ["M1", "Shell manhole", "2", "24 in", "CL150", "FF"]], widths=(50, 150, 40, 60, 90, 60)),
])
build("tank/tank_specification.pdf", "Owner Specification - Atmospheric Storage Tanks (extract)", [
    "Specification SPEC-TK-003 Rev 2, issued by the owner for all atmospheric tanks of the project.",
    table([["Design code", "API 650, 13th edition"], ["Site", "Macaé, Brazil"],
           ["Shell corrosion allowance", "3 mm"], ["Joint efficiency", "0.85"], ["Venting basis", "API 2000"],
           ["Coating specification", "SPEC-PT-004 system T2"]]),
    "<b>Site data</b>",
    table([["Design wind speed", "38 m/s"], ["Seismic design basis", "ASCE 7, site class D, SUG I"],
           ["Snow load", "Not applicable"], ["Roof live load", "1.0 kPa"]]),
])

# ---------------------------------------------------------------- P-101
build("pump/process_datasheet_P-101.pdf", "Process Datasheet - Condensate Pumps P-101 A/B", [
    "Process document PDS-P-101 Rev B (fictitious). Pumps condensate from separator V-101.",
    table([["Item No.", "P-101 A/B"], ["Service", "Condensate transfer"], ["Pumps required", "2 (1 operating + 1 spare)"],
           ["Liquid", "Hydrocarbon condensate"], ["Pumping temperature, normal", "40 °C"],
           ["Pumping temperature, maximum", "85 °C"], ["Specific gravity", "0.72"],
           ["Vapour pressure", "11.5 bar(a)"], ["Hazardous liquid", "Yes (flammable)"], ["Rated flow", "55 m3/h"],
           ["Normal flow", "46 m3/h"], ["Minimum flow", "15 m3/h"], ["Suction pressure, rated", "12.3 barg"],
           ["Discharge pressure", "28 barg"], ["Differential head", "225 m"], ["NPSH available", "4.2 m"]]),
])
build("pump/hydraulic_calculation_P-101.pdf", "Hydraulic Calculation - P-101 Suction and Discharge", [
    "Calculation CAL-HYD-101 Rev 0.",
    "Suction line 6 in, 18 m equivalent length; minimum liquid level in V-101 at 900 mm above the bottom tangent.",
    table([["NPSH available", "3.8 m"], ["Differential head", "225 m"], ["Differential pressure", "15.7 bar"]]),
])
build("pump/pump_specification.pdf", "Owner Specification - Centrifugal Pumps (extract)", [
    "Specification SPEC-PU-001 Rev 4.",
    table([["Design code", "API 610, 12th edition"], ["Pump type", "OH2"], ["Material class", "S-6"],
           ["Mechanical seal", "API 682 Category 2, Arrangement 2"], ["Seal flush plan", "Plan 11/52"],
           ["Driver", "Electric motor"], ["Power supply", "4160 V / 3 ph / 60 Hz"],
           ["Hazardous area classification", "Zone 2, IIA T3"], ["Ambient temperature range", "15 to 40 °C"],
           ["Location", "Outdoor, under roof"]]),
])
def scanned(src, dst, angle=0.7, dpi=200, seed=0):
    """An image-only copy of ``src`` that looks scanned: rotated, speckled, blurred, JPEG-compressed."""
    import io
    import random

    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter
    rnd, pages = random.Random(seed), []
    for page in pdfium.PdfDocument(os.path.join(HERE, src)):
        im = page.render(scale=dpi / 72).to_pil().convert("L")
        im = im.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
        px = im.load()
        w, h = im.size
        for _ in range(w * h // 400):
            px[rnd.randrange(w), rnd.randrange(h)] = rnd.choice((0, 180, 255))
        im = im.filter(ImageFilter.GaussianBlur(0.6))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=60)
        pages.append(Image.open(buf))
    pages[0].save(os.path.join(HERE, dst), "PDF", resolution=dpi, save_all=True, append_images=pages[1:])
    os.remove(os.path.join(HERE, src))


scanned("psv/_relief_load_summary_PSV-101_clean.pdf", "psv/relief_load_summary_PSV-101_scanned.pdf")
print("wrote", sorted(os.path.relpath(os.path.join(d, f), HERE) for d, _, fs in os.walk(HERE) for f in fs
                      if f.endswith(".pdf")))
