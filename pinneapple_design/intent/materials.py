"""Materials with cited properties for the design templates.

Every number carries the document it was read from. Only the properties listed are known: a template that needs a
property a material does not have must not offer that material (``Material.has``). Values are typical / minimum
datasheet values at room temperature, not design allowables of a particular code; the design report says so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ASM_6061 = "ASM Aerospace Specification Metals, 'Aluminum 6061-T6; 6061-T651' datasheet (asm.matweb.com)"
ASM_6063 = "ASM Aerospace Specification Metals, 'Aluminum 6063-T5' datasheet (asm.matweb.com)"
ASM_304 = "ASM Aerospace Specification Metals, 'AISI Type 304 Stainless Steel' datasheet, annealed (asm.matweb.com)"
MATWEB_1018 = "MatWeb, 'AISI 1018 Steel, cold drawn' (matweb.com)"
NATUREWORKS_4043D = ("NatureWorks, 'Ingeo Biopolymer 4043D Technical Data Sheet' (moulded / film specimens; "
                     "printed parts are weaker and anisotropic)")
ZAMAK3 = "International Zinc Association / Eastern Alloys, 'Zamak 3' die-casting alloy properties (as cast)"
WOOD_HANDBOOK = ("USDA Forest Products Laboratory, 'Wood Handbook' FPL-GTR-190 (2010), ch. 5, imported woods: "
                 "balsa at 12 % moisture content (specific gravity on oven-dry weight)")
HEATSINK_TABLE = "pinneapple_physics.closed_form.plate_fin_heatsink.MATERIALS (the heat-sink model's own table)"


@dataclass(frozen=True)
class Prop:
    value: float
    unit: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class Material:
    key: str
    label: str
    category: str                   # metal | polymer | wood
    props: dict[str, Prop] = field(default_factory=dict)
    processes: tuple[str, ...] = ()
    notes: str = ""

    def has(self, *names: str) -> bool:
        return all(n in self.props for n in names)

    def __getitem__(self, name: str) -> float:
        if name not in self.props:
            raise KeyError(f"{self.label}: no cited value for '{name}'")
        return self.props[name].value

    def cite(self, name: str) -> str:
        return self.props[name].source

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "category": self.category, "processes": list(self.processes),
                "notes": self.notes, "properties": {k: {"value": p.value, "unit": p.unit, "source": p.source,
                                                        "note": p.note} for k, p in self.props.items()}}


def _m(key, label, category, processes=(), notes="", **props) -> Material:
    return Material(key, label, category, {k: Prop(*v) for k, v in props.items()}, tuple(processes), notes)


MATERIALS: dict[str, Material] = {m.key: m for m in [
    _m("al6061_t6", "Aluminium 6061-T6", "metal", ("machining", "sheet bending", "extrusion"),
       density=(2700.0, "kg/m^3", ASM_6061), youngs_modulus=(68.9e9, "Pa", ASM_6061),
       poisson=(0.33, "1", ASM_6061), yield_strength=(276e6, "Pa", ASM_6061),
       tensile_strength=(310e6, "Pa", ASM_6061), conductivity=(167.0, "W/(m*K)", ASM_6061)),
    _m("al6063_t5", "Aluminium 6063-T5", "metal", ("extrusion",),
       density=(2700.0, "kg/m^3", ASM_6063), youngs_modulus=(68.9e9, "Pa", ASM_6063),
       poisson=(0.33, "1", ASM_6063), yield_strength=(145e6, "Pa", ASM_6063),
       tensile_strength=(186e6, "Pa", ASM_6063), conductivity=(209.0, "W/(m*K)", ASM_6063)),
    _m("steel_1018", "Steel AISI 1018, cold drawn", "metal", ("machining", "welding", "sheet bending"),
       density=(7870.0, "kg/m^3", MATWEB_1018), youngs_modulus=(205e9, "Pa", MATWEB_1018),
       poisson=(0.29, "1", MATWEB_1018), yield_strength=(370e6, "Pa", MATWEB_1018),
       tensile_strength=(440e6, "Pa", MATWEB_1018), conductivity=(51.9, "W/(m*K)", MATWEB_1018)),
    _m("ss304", "Stainless steel AISI 304, annealed", "metal", ("machining", "welding", "sheet bending"),
       density=(8000.0, "kg/m^3", ASM_304), youngs_modulus=(193e9, "Pa", ASM_304),
       poisson=(0.29, "1", ASM_304), yield_strength=(215e6, "Pa", ASM_304),
       tensile_strength=(505e6, "Pa", ASM_304), conductivity=(16.2, "W/(m*K)", ASM_304)),
    _m("pla_4043d", "PLA (Ingeo 4043D)", "polymer", ("3-D printing (FDM)",),
       notes="printed parts: take the strength as an upper bound and test a coupon",
       density=(1240.0, "kg/m^3", NATUREWORKS_4043D), youngs_modulus=(3.6e9, "Pa", NATUREWORKS_4043D),
       yield_strength=(60e6, "Pa", NATUREWORKS_4043D)),
    _m("zamak3", "Zinc alloy Zamak 3 (die cast)", "metal", ("die casting",),
       density=(6600.0, "kg/m^3", ZAMAK3), yield_strength=(221e6, "Pa", ZAMAK3),
       tensile_strength=(283e6, "Pa", ZAMAK3)),
    _m("balsa", "Balsa wood", "wood", ("cutting", "sanding", "gluing"),
       notes="density varies by a factor of about 2 between sheets: weigh yours",
       density=(160.0, "kg/m^3", WOOD_HANDBOOK, "specific gravity 0.16"),
       youngs_modulus=(3.71e9, "Pa", WOOD_HANDBOOK, "along the grain"),
       bending_strength=(21.6e6, "Pa", WOOD_HANDBOOK, "modulus of rupture")),
]}


def heat_sink_material(key: str) -> Material:
    """The heat-sink model's own material (conductivity and density exactly as its physics uses them)."""
    from pinneapple_physics.closed_form.plate_fin_heatsink import MATERIALS as HS

    d = HS[key]
    return _m(key, d.get("label", key), "metal", ("extrusion",) if key.startswith("al") else ("machining",),
              density=(float(d["rho"]), "kg/m^3", HEATSINK_TABLE),
              conductivity=(float(d["k"]), "W/(m*K)", HEATSINK_TABLE))


def get(key: str) -> Material:
    if key not in MATERIALS:
        raise KeyError(f"unknown material {key!r}; known: {sorted(MATERIALS)}")
    return MATERIALS[key]


def candidates(*needs: str, category: str | None = None) -> list[Material]:
    """Materials with a cited value for every property in ``needs``."""
    return [m for m in MATERIALS.values() if m.has(*needs) and (category is None or m.category == category)]


def find(text: str) -> list[str]:
    """Material keys named in free text ("aluminium", "steel", "PLA", "balsa", ...)."""
    t = text.lower()
    words = {"al6061_t6": ("6061",), "al6063_t5": ("6063",), "steel_1018": ("1018", "mild steel"),
             "ss304": ("stainless", "304"), "pla_4043d": ("pla",), "zamak3": ("zamak", "die-cast", "diecast"),
             "balsa": ("balsa",)}
    hits = [k for k, ws in words.items() if any(re.search(rf"\b{re.escape(w)}\b", t) for w in ws)]
    if not hits and ("alumin" in t):
        hits = ["al6061_t6"]
    if not hits and "steel" in t:
        hits = ["steel_1018"]
    return hits
