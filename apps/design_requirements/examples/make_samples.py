"""Build the three fictitious input documents used by the demo and the tests (needs reportlab).

They describe one made-up separator, V-101, the way three different parties would: a process datasheet, a
mechanical datasheet from the engineering contractor, and the owner's project specification. Two things are
planted on purpose: the owner's specification asks for a larger shell corrosion allowance than the mechanical
datasheet (a conflict to resolve), and nobody states whether the vessel is in cyclic service (a gap).
Run: python make_samples.py  (writes the three PDFs next to this file)
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
    doc = SimpleDocTemplate(os.path.join(HERE, name), pagesize=A4, title=title)
    story = [Paragraph(BANNER, ST["Italic"]), Paragraph(title, ST["Title"])]
    for b in blocks:
        story.append(b if not isinstance(b, str) else Paragraph(b, ST["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)


build("process_datasheet_V-101.pdf", "Process Datasheet - Separator V-101", [
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

build("mechanical_datasheet_V-101.pdf", "Mechanical Datasheet - Separator V-101", [
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

build("owner_specification_V-101.pdf", "Owner Project Specification - Pressure Vessels (extract)", [
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
print("wrote", [f for f in os.listdir(HERE) if f.endswith(".pdf")])
