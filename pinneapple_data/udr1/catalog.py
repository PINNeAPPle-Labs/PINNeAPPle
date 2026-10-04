"""What ASME Form U-DR-1 asks for, and where each answer goes in the fillable PDF.

Each ``Field`` is one answer on the form (User's Design Requirements for Single-Chamber Pressure Vessels, ASME BPVC
Section VIII, Division 1, Nonmandatory Appendix KK). ``pdf`` names the AcroForm field it is written to; for check
boxes and radio groups ``states`` maps an answer to the ``(field, export state)`` pair that represents it. The PDF
field names come from the 07/25 revision of the form; many are generic (``Group10``, ``Text7``), so the mapping was
made by overlaying each widget's rectangle on the rendered page (see ``tests/test_udr1.py``).

``patterns`` are the phrasings used in process datasheets, mechanical datasheets and client specifications for the
same item (case-insensitive regular expressions, matched at a label position). ``quantity`` names the physical
quantity for unit-aware comparison between documents. ``required`` marks the items without which a manufacturer
cannot start the design; missing ones are reported as data gaps.

The blank form itself is ASME's copyrighted document and is not shipped here: users upload their copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["Field", "FIELDS", "FIELD_BY_KEY", "SECTIONS", "NOZZLE_COLUMNS", "NOZZLE_PDF_ROWS", "fields_in"]


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    section: str
    kind: str = "text"                       # text | quantity | number | choice | bool | multi
    pdf: Optional[str] = None                # text field name
    states: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # answer -> (pdf field, export state)
    patterns: Tuple[str, ...] = ()
    quantity: Optional[str] = None           # pressure | temperature | length | speed | density | time
    required: bool = False
    source_hint: str = ""                    # which document usually holds it


_SIDE = {"int": r"(?:internal|int\.?)", "ext": r"(?:external|ext\.?)"}
YES_NO = lambda group: {"yes": (group, "/Choice1"), "no": (group, "/Choice2")}  # noqa: E731


def _f(*args, **kw) -> Field:
    return Field(*args, **kw)


SECTIONS = ["General", "Operating conditions", "Design conditions", "Corrosion and fatigue", "Loadings",
            "Heat treatment, insulation and support", "Materials", "Welded joints", "Body flanges", "Certification"]

FIELDS: List[Field] = [
    # ---------------------------------------------------------------- general
    _f("owner", "Owner", "General", pdf="Owner", patterns=(r"owner", r"end[- ]user", r"client"), required=True,
       source_hint="client specification"),
    _f("operator", "Operator", "General", pdf="Operator", patterns=(r"operator", r"operating company")),
    _f("country", "Country of installation", "General", pdf="Country of Installation",
       patterns=(r"country(?: of installation)?",), required=True, source_hint="client specification"),
    _f("state_province", "State/Province of installation", "General", pdf="StateProvince of Installation",
       patterns=(r"state(?:/province)?(?: of installation)?", r"province")),
    _f("city", "City of installation", "General", pdf="City of Installation",
       patterns=(r"city(?: of installation)?", r"site location", r"plant location")),
    _f("service", "Service", "General", pdf="Service", patterns=(r"service", r"fluid service"), required=True,
       source_hint="process datasheet"),
    _f("liquid_level", "Liquid level", "General", kind="quantity", quantity="length", pdf="1",
       patterns=(r"(?:normal |design )?liquid level", r"\bnll\b", r"\bhll\b")),
    _f("specific_gravity", "Specific gravity", "General", kind="number", pdf="2",
       patterns=(r"specific gravity", r"\bs\.?g\.?\b", r"relative density"), source_hint="process datasheet"),
    _f("item_no", "Item No.", "General", pdf="Item No", patterns=(r"item no\.?", r"item number", r"tag(?: no\.?| number)?",
                                                                    r"equipment (?:no\.?|number|tag)"), required=True),
    _f("diameter", "Diameter", "General", kind="quantity", quantity="length", pdf="Diameter in",
       patterns=(r"inside diameter", r"internal diameter", r"\bi\.?d\.?\b", r"diameter"), required=True,
       source_hint="mechanical datasheet"),
    _f("length_tt", "Length, tangent-to-tangent", "General", kind="quantity", quantity="length",
       pdf="Length  T angentto T angent",
       patterns=(r"length,? tangent[- ]to[- ]tangent", r"t/t length", r"length t/t", r"tan(?:gent)?[- ]tan(?:gent)?(?: length)?",
                 r"length \(t/t\)"), required=True, source_hint="mechanical datasheet"),
    _f("orientation", "Type (orientation)", "General", kind="choice",
       states={"vertical": ("roup6", "/Choice1"), "horizontal": ("roup6", "/Choice2"), "sphere": ("roup6", "/Choice3")},
       patterns=(r"orientation", r"vessel type", r"position"), required=True),
    _f("nb_registration", "National Board registration required", "General", kind="bool", states=YES_NO("Group10"),
       patterns=(r"national board(?: registration)?(?: required)?", r"\bnb registration"), source_hint="client specification"),
    _f("crn_registration", "Canadian registration required", "General", kind="bool", states=YES_NO("Group11"),
       patterns=(r"canadian registration(?: required)?", r"\bcrn\b(?: registration)?(?: required)?")),
    _f("lethal", "Special service: lethal (L)", "General", kind="bool", states={"yes": ("Lethal L", "/On")},
       patterns=(r"lethal service", r"lethal"), source_hint="process datasheet"),
    _f("direct_firing", "Special service: direct firing (DF)", "General", kind="bool",
       states={"yes": ("Check Box2", "/Yes")}, patterns=(r"direct(?:ly)? fir(?:ing|ed)",)),
    _f("unfired_steam_boiler", "Special service: unfired steam boiler (UB)", "General", kind="bool",
       states={"yes": ("Unfired Steam Boiler UB", "/On")}, patterns=(r"unfired steam boiler",)),
    _f("overpressure", "Overpressure protection", "General", kind="multi",
       states={"valve": ("V alve", "/On"), "rupture disk": ("Rupture Disk", "/On"), "other": ("Other", "/On"),
               "system design": ("System Design", "/On")},
       patterns=(r"overpressure protection", r"pressure relief(?: device)?", r"relief device"), required=True),
    # ---------------------------------------------------------------- operating
    *[_f(f"op{c}_{k}", f"Operating case {c}: {lab}", "Operating conditions", kind="quantity", quantity=q,
         pdf=f"{pdfname}Case {c}",
         patterns=tuple(p.format(c=c) for p in pats), required=(c == 1), source_hint="process datasheet")
      for c in (1, 2)
      for k, lab, q, pdfname, pats in (
          ("pmin", "minimum pressure", "pressure", "Minimum Pressure",
           (r"(?:operating )?case {c}[ ,:-]*min(?:imum)? (?:operating )?pressure", r"min(?:imum)? operating pressure,? case {c}")),
          ("pmax", "maximum pressure", "pressure", "Maximum Pressure",
           (r"(?:operating )?case {c}[ ,:-]*max(?:imum)? (?:operating )?pressure", r"max(?:imum)? operating pressure,? case {c}")),
          ("tmin", "minimum temperature", "temperature", "Minimum  T emperature",
           (r"(?:operating )?case {c}[ ,:-]*min(?:imum)? (?:operating )?temperature", r"min(?:imum)? operating temperature,? case {c}")),
          ("tmax", "maximum temperature", "temperature", "Maximum T emperature",
           (r"(?:operating )?case {c}[ ,:-]*max(?:imum)? (?:operating )?temperature", r"max(?:imum)? operating temperature,? case {c}")),
      )],
    # ---------------------------------------------------------------- design
    _f("design_p_int", "Internal design pressure", "Design conditions", kind="quantity", quantity="pressure",
       pdf="PressureInternal Design Pressure", patterns=(r"internal design pressure", r"design pressure,? internal",
                                                          r"design pressure(?! ,?ext)"), required=True,
       source_hint="process datasheet"),
    _f("design_t_int", "Temperature at internal design pressure", "Design conditions", kind="quantity",
       quantity="temperature", pdf="T emperatureInternal Design Pressure",
       patterns=(r"(?:max(?:imum)? )?design temperature(?:,? internal)?", r"temperature at internal design pressure"),
       required=True, source_hint="process datasheet"),
    _f("design_p_ext", "External design pressure", "Design conditions", kind="quantity", quantity="pressure",
       pdf="PressureExternal Design Pressure", patterns=(r"external design pressure", r"design pressure,? external",
                                                          r"design vacuum", r"vacuum design")),
    _f("design_t_ext", "Temperature at external design pressure", "Design conditions", kind="quantity",
       quantity="temperature", pdf="T emperatureExternal Design Pressure",
       patterns=(r"temperature at external design pressure", r"external design temperature")),
    _f("mawp_int", "MAWP internal", "Design conditions", kind="quantity", quantity="pressure", pdf="M A WP Internal",
       patterns=(r"mawp(?:,? internal)?", r"maximum allowable working pressure(?:,? internal)?")),
    _f("mawp_int_basis", "MAWP internal basis", "Design conditions", kind="choice",
       states={"same as design pressure": ("group66", "/Choice1"), "calculated by manufacturer": ("group66", "/Choice2")},
       patterns=(r"mawp(?:,? internal)? basis", r"mawp(?:,? internal)?")),
    _f("mawp_ext", "MAWP external", "Design conditions", kind="quantity", quantity="pressure", pdf="M A WP External",
       patterns=(r"mawp,? external", r"maximum allowable (?:working )?(?:external|vacuum) pressure")),
    _f("mawp_ext_basis", "MAWP external basis", "Design conditions", kind="choice",
       states={"same as design pressure": ("group67", "/Choice1"), "calculated by manufacturer": ("group67", "/Choice2")},
       patterns=(r"mawp,? external basis", r"mawp,? external")),
    *[_f(f"mdmt{c}_temp", f"MDMT case {c}: temperature", "Design conditions", kind="quantity", quantity="temperature",
         pdf=f"Same as Design PressureMinimum Design Metal  T emperature MDMT  Case {c}",
         patterns=((rf"mdmt(?:,? case {c})?", r"minimum design metal temperature") if c == 1
                   else (rf"mdmt,? case {c}", rf"minimum design metal temperature,? case {c}")),
         required=(c == 1), source_hint="process datasheet") for c in (1, 2)],
    *[_f(f"mdmt{c}_pressure", f"MDMT case {c}: coincident pressure", "Design conditions", kind="quantity",
         quantity="pressure", pdf="Deg" if c == 1 else "Deg_2",
         patterns=((rf"mdmt coincident pressure(?:,? case {c})?", r"coincident pressure at mdmt") if c == 1
                   else (rf"mdmt coincident pressure,? case {c}",))) for c in (1, 2)],
    *[_f(f"mdmt{c}_due_to", f"MDMT case {c}: due to", "Design conditions", kind="choice",
         states={"process": (g, "/Choice1"), "other": (g, "/Choice2"), "ambient temperature": (g, "/Choice3")},
         patterns=((rf"mdmt(?:,? case {c})? (?:due to|governed by|basis)",) if c == 1
                   else (rf"mdmt,? case {c} (?:due to|governed by|basis)",)))
      for c, g in ((1, "Group29"), (2, "Group30x"))],
    # ---------------------------------------------------------------- corrosion, cyclic
    *[_f(f"ca_{comp}_{side}", f"Corrosion allowance: {name}, {'internal' if side == 'int' else 'external'}",
         "Corrosion and fatigue", kind="quantity", quantity="length", pdf=f"{'IntRow1' if side == 'int' else 'ExtRow1'}{suffix}",
         patterns=(rf"{pat}[ ,]*(?:corrosion allowance|c\.?a\.?)[ ,]*{_SIDE[side]}",
                   rf"corrosion allowance[ ,:-]*{pat}[ ,]*{_SIDE[side]}"),
         required=(comp in ("shell", "heads") and side == "int"), source_hint="client specification")
      for comp, name, pat, suffix in (("shell", "shell", "shell", ""), ("heads", "heads", "heads?", "_2"),
                                      ("nozzles", "nozzles", "nozzles?", "_3"), ("jacket", "jacket", "jacket", "_4"),
                                      ("coil", "coil", "coil", "_5"), ("supports", "supports", "supports?", "_6"))
      for side in ("int", "ext")],
    _f("ca_internals", "Corrosion allowance: internals", "Corrosion and fatigue", kind="quantity", quantity="length",
       pdf="InternalsRow2", patterns=(r"internals[ ,]*(?:corrosion allowance|c\.?a\.?)", r"corrosion allowance[ ,:-]*internals")),
    _f("corrosive_service", "Corrosive service", "Corrosion and fatigue", kind="bool", states=YES_NO("Group2"),
       patterns=(r"corrosive service",), source_hint="process datasheet"),
    _f("cyclic_service", "Cyclic service", "Corrosion and fatigue", kind="bool", states=YES_NO("Group4"),
       patterns=(r"cyclic service", r"cyclic operation"), required=True, source_hint="process datasheet"),
    _f("cycles", "Number of cycles", "Corrosion and fatigue", kind="number", pdf="undefined_4",
       patterns=(r"number of (?:pressure )?cycles", r"cycles(?! per)")),
    _f("cycles_per", "Cycles per (period)", "Corrosion and fatigue", pdf="Cycles per", patterns=(r"cycles per",)),
    _f("design_life", "Design life (years)", "Corrosion and fatigue", kind="number", pdf="Design Life",
       patterns=(r"design life", r"service life"), source_hint="client specification"),
    _f("fatigue_analysis", "Fatigue analysis required", "Corrosion and fatigue", kind="bool", states=YES_NO("Group3"),
       patterns=(r"fatigue analysis(?: required)?",)),
    # ---------------------------------------------------------------- loadings
    _f("wind_code", "Wind loading code", "Loadings", kind="choice",
       states={"ubc": ("group69", "/Choice1"), "other": ("group69", "/Choice2"), "asce 7": ("group69", "/Choice3"),
               "ibc": ("group69", "/Choice4"), "none": ("group69", "/Choice5")},
       patterns=(r"wind (?:load(?:ing)? )?(?:code|standard|design code|loading)",), required=True,
       source_hint="client specification"),
    _f("wind_speed", "Wind speed", "Loadings", kind="quantity", quantity="speed", pdf="Wind Speed",
       patterns=(r"(?:basic |design )?wind speed",)),
    _f("wind_classification", "Wind classification category", "Loadings", pdf="Classification Category",
       patterns=(r"(?:risk|occupancy|wind) (?:classification )?category", r"classification category")),
    _f("wind_exposure", "Wind exposure category", "Loadings", pdf="Exposure Category", patterns=(r"exposure category",)),
    _f("wind_topographic", "Topographic factor", "Loadings", pdf="T opographic Factor",
       patterns=(r"topographic factor", r"\bk\s?zt\b")),
    _f("elevation", "Elevation", "Loadings", kind="quantity", quantity="length", pdf="Elevation",
       patterns=(r"(?:site |ground )?elevation",)),
    _f("seismic_code", "Seismic loading code", "Loadings", kind="choice",
       states={"ubc": ("group70", "/Choice1"), "other": ("group70", "/Choice2"), "asce 7": ("group70", "/Choice3"),
               "ibc": ("group70", "/Choice4"), "none": ("group70", "/Choice5")},
       patterns=(r"seismic (?:load(?:ing)? )?(?:code|standard|design code|loading)",), required=True,
       source_hint="client specification"),
    _f("soil_profile", "Soil profile classification", "Loadings", pdf="Soil Profile Classification",
       patterns=(r"soil (?:profile|site) class(?:ification)?", r"site class")),
    _f("temp_gradients", "Other loadings: temperature gradients", "Loadings", kind="bool",
       states={"yes": ("Temp Gradients", "/On")}, patterns=(r"temperature gradients?",)),
    _f("deflagration", "Other loadings: deflagration", "Loadings", kind="bool", states={"yes": ("Deflagration", "/On")},
       patterns=(r"deflagration",)),
    _f("diff_thermal_exp", "Other loadings: differential thermal expansion", "Loadings", kind="bool",
       states={"yes": ("Diff Thermal Exp", "/On")}, patterns=(r"differential thermal expansion",)),
    # ---------------------------------------------------------------- PWHT, insulation, support
    _f("pwht", "PWHT", "Heat treatment, insulation and support", kind="choice",
       states={"per code": ("Check Box1", "/Yes"), "process required": ("Process Required", "/On")},
       patterns=(r"pwht", r"post[- ]weld heat treatment"), required=True),
    _f("insulated", "Insulated", "Heat treatment, insulation and support", kind="bool", states=YES_NO("Group30"),
       patterns=(r"insulat(?:ed|ion)(?: required)?",)),
    _f("insulation_by", "Insulation by", "Heat treatment, insulation and support", kind="choice",
       states={"manufacturer": ("group68", "/Choice1"), "others": ("group68", "/Choice2")},
       patterns=(r"insulation (?:by|supplied by|furnished by)",)),
    _f("insulation_external", "Insulation, external (type and thickness)", "Heat treatment, insulation and support",
       pdf="Text7", patterns=(r"external insulation", r"insulation,? external", r"insulation type")),
    _f("insulation_internal", "Insulation, internal (type and thickness)", "Heat treatment, insulation and support",
       pdf="Internal", patterns=(r"internal insulation", r"insulation,? internal", r"refractory lining")),
    _f("insulation_density_ext", "Insulation density, external", "Heat treatment, insulation and support",
       kind="quantity", quantity="density", pdf="Density 1", patterns=(r"(?:external )?insulation density",)),
    _f("insulation_density_int", "Insulation density, internal", "Heat treatment, insulation and support",
       kind="quantity", quantity="density", pdf="Density 2", patterns=(r"internal insulation density",)),
    _f("coating_spec", "Coating specification", "Heat treatment, insulation and support", pdf="Coating Specification",
       patterns=(r"coating spec(?:ification)?", r"painting spec(?:ification)?", r"paint(?:ing)? system")),
    _f("coating_before_test", "Coating permitted prior to pressure test", "Heat treatment, insulation and support",
       kind="bool", states=YES_NO("Group1"), patterns=(r"(?:coating|painting) (?:permitted |allowed )?(?:prior to|before) (?:pressure|hydro(?:static)?) test",)),
    _f("support", "Vessel support", "Heat treatment, insulation and support", kind="choice",
       states={"legs": ("group72", "/Choice1"), "skirt": ("group72", "/Choice2"), "lugs": ("group72", "/Choice3"),
               "saddles": ("group72", "/Choice4")},
       patterns=(r"(?:vessel )?support(?: type)?",), required=True, source_hint="mechanical datasheet"),
    _f("fireproofing", "Fireproofing", "Heat treatment, insulation and support", kind="bool", states=YES_NO("Group20"),
       patterns=(r"fire ?proofing(?: required)?",)),
    _f("fireproofing_type", "Fireproofing type", "Heat treatment, insulation and support", pdf="T ype",
       patterns=(r"fire ?proofing type",)),
    _f("fireproofing_rating", "Fireproofing rating (hr)", "Heat treatment, insulation and support", pdf="Rating hr",
       patterns=(r"fire ?proofing rating", r"fire rating")),
    # ---------------------------------------------------------------- materials
    *[_f(f"mat_{k}", f"Material: {lab}", "Materials", pdf=f"Specification{pdfname}",
         patterns=tuple(rf"{p}(?: material)?(?: spec(?:ification)?)?" for p in pats) + tuple(rf"material[ ,:-]*{p}" for p in pats),
         required=k in ("shell",), source_hint="mechanical datasheet")
      for k, lab, pdfname, pats in (
          ("shell", "shell", "Shell", (r"shell",)),
          ("ellipsoidal_head", "ellipsoidal head", "Ellipsoidal Head", (r"(?:2:1 )?(?:semi-?)?ellipsoidal heads?",)),
          ("hemispherical_head", "hemispherical head", "Hemispherical Head", (r"hemispherical heads?",)),
          ("torispherical_head", "torispherical head", "T orispherical Head", (r"torispherical heads?", r"f&d heads?")),
          ("toriconical_head", "toriconical head", "T oriconical Head", (r"toriconical heads?",)),
          ("conical_head", "conical head", "Conical Head", (r"conical heads?", r"cones?")),
          ("nozzles", "nozzles", "Nozzles", (r"nozzles?(?: necks?)?",)),
          ("flanges", "flanges", "Flanges", (r"flanges?",)),
          ("stiffener_rings", "stiffener rings", "Stiffener Rings", (r"stiffener rings?", r"stiffening rings?")),
          ("bolts", "pressure-retaining bolts", "Pressure Retaining Bolts", (r"(?:pressure[- ]retaining )?bolt(?:s|ing)", r"studs?")),
          ("attachments", "attachments", "Attachments", (r"(?:external |internal )?attachments",)),
          ("internals", "internals", "Internals", (r"internals",)),
          ("reinforcing_pads", "reinforcing pads", "Reinforcing Pads", (r"reinforc(?:ing|ement) pads?", r"re-?pads?")),
      )],
    _f("mat_other_component", "Material: other component (name)", "Materials", pdf="Other_6"),
    _f("mat_other", "Material: other component (specification)", "Materials", pdf="SpecificationOther"),
    # ---------------------------------------------------------------- welded joints (back page)
    _f("design_basis", "Design basis", "Welded joints", pdf="DESIGN BASIS", patterns=(r"design basis", r"design code"),
       source_hint="client specification"),
    _f("joint_eff_shell", "Joint efficiency E, shell and cone", "Welded joints", kind="number", pdf="JOINT EFFICIENCY  E",
       patterns=(r"(?:shell(?: and cone)?) joint efficiency", r"joint efficiency,? shell", r"joint efficiency(?! ,?head)"),
       required=True),
    _f("joint_eff_head", "Joint efficiency E, heads", "Welded joints", kind="number", pdf="undefined_14",
       patterns=(r"heads? joint efficiency", r"joint efficiency,? heads?")),
    *[_f(f"joint_{k}_type", f"Joint type (UW-12), {lab}", "Welded joints",
         pdf=f"TYPE OF JOINT Use  T ypes as Described in U W 12Category A.{i}",
         patterns=(rf"{p}[ ,:-]*(?:joint )?type", rf"joint type[ ,:-]*{p}"), required=k in ("cat_a", "cat_b_head"))
      for i, (k, lab, p) in enumerate((
          ("cat_a", "Category A", r"category a"), ("cat_b_head", "Category B head-to-shell", r"category b,? head[- ]to[- ]shell"),
          ("cat_b_other", "Category B other", r"category b,? other"), ("cat_c_body", "Category C body flanges", r"category c,? body flanges?"),
          ("cat_c_nozzle", "Category C nozzle flanges", r"category c,? nozzle flanges?"), ("cat_d", "Category D", r"category d"),
          ("cat_f", "Category F", r"category f")))],
    *[_f(f"joint_{k}_nde", f"NDE with comments, {lab}", "Welded joints", pdf=f"NDE WITH COMMENTSCategory A.{i}",
         patterns=(rf"{p}[ ,:-]*(?:nde|examination|radiography)", rf"(?:nde|examination)[ ,:-]*{p}"),
         required=k in ("cat_a", "cat_b_head"), source_hint="client specification")
      for i, (k, lab, p) in enumerate((
          ("cat_a", "Category A", r"category a"), ("cat_b_head", "Category B head-to-shell", r"category b,? head[- ]to[- ]shell"),
          ("cat_b_other", "Category B other", r"category b,? other"), ("cat_c_body", "Category C body flanges", r"category c,? body flanges?"),
          ("cat_c_nozzle", "Category C nozzle flanges", r"category c,? nozzle flanges?"), ("cat_d", "Category D", r"category d"),
          ("cat_f", "Category F", r"category f")))],
    # ---------------------------------------------------------------- body flanges
    *[_f(f"bodyflange{r}_{k}", f"Body flange {r}: {lab}", "Body flanges", pdf=pdfname.format(r=r))
      for r in (1, 2)
      for k, lab, pdfname in (("description", "description", "DescriptionRow{r}_3"), ("type", "type", "T ypeRow{r}"),
                              ("facing", "facing / surface finish", "FacingSurface FinishRow{r}"),
                              ("gasket", "gasket style", "Gasket StyleRow{r}"),
                              ("assembly", "joint assembly (PCC-1)", "Joint Assembly See ASME PCC1Row{r}"))],
    # ---------------------------------------------------------------- notes and certification
    _f("general_notes", "General notes", "Certification", pdf="GENERAL NOTESRow1", patterns=(r"general notes?",)),
    _f("date", "Date", "Certification", pdf="Date"),
    _f("user", "User", "Certification", pdf="User", patterns=(r"user(?:'s)? (?:name|representative)",)),
    _f("registration_id", "Registration identification", "Certification", pdf="Registration Identification"),
]

FIELD_BY_KEY: Dict[str, Field] = {f.key: f for f in FIELDS}

# Nozzle schedule: 12 rows (6 in the left half of the table, 6 in the right half).
NOZZLE_COLUMNS = ("description", "number", "size", "flange_type", "class")
NOZZLE_PDF_ROWS: List[Dict[str, str]] = [
    {"description": f"DescriptionRow{r}{s}", "number": f"Number RequiredRow{r}{s}", "size": f"SizeRow{r}{s}",
     "flange_type": f"Flange T ypeRow{r}{s}", "class": f"ClassRow{r}{s}"}
    for s in ("", "_2") for r in range(1, 7)
]


def fields_in(section: str) -> List[Field]:
    return [f for f in FIELDS if f.section == section]
