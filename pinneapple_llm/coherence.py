"""Does the chosen preset solve the problem the user described?

:class:`~pinneapple_llm.guardrail.PhysicsGuardrail` re-checks the physics of the preset a model was trained on; it
cannot tell whether that preset was the right one. An LLM that maps "heat conduction in a rod" to the Burgers preset
produces a model that passes every guardrail check and answers a different question. :func:`check_preset_coherence`
compares the description with the preset before anything is trained:

* **physics**: the classes of physics the description names (heat, diffusion, waves, fluid flow, structures,
  electrostatics, electromagnetism, orbits, ...; English and Portuguese words) against the classes the preset's
  ``pde.kind`` belongs to. A preset whose classes share none with the description's is incoherent;
* **regime**: a description that says *steady* (or *transient*) against a preset with (or without) a time axis;
* **dimension**: an explicit 1-D / 2-D / 3-D against the preset's number of spatial coordinates.

The verdict is ``"coherent"``, ``"incoherent"`` (with the reasons) or ``"unknown"`` when the description names no
physics the check recognises: that is not a confirmation, and callers should not report the result as verified.

    v = check_preset_coherence("heat conduction in a steel rod", pp.get_preset("burgers_1d"))
    v.status, v.reasons          # "incoherent", ["the description is about heat ...; the preset ... is fluid / shock"]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# class -> (words in a description, patterns of the presets' pde.kind)
PHYSICS_CLASSES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "heat": (("heat", "thermal", "temperature", "conduction", "conducting", "cooling", "cools", "heating", "furnace",
              "insulat", "calor", "térmic", "termic", "temperatura", "condução", "conducao", "resfri", "aquec",
              "forno", "isolamento"),
             ("heat", "poisson", "laplace", "thermoelasticity", "phonon")),
    "diffusion": (("diffusion", "diffuses", "diffusing", "concentration", "species", "drug", "dopant", "difusão",
                   "difusao", "concentração", "concentracao", "fármaco", "farmaco"),
                  ("diffusion", "heat", "poisson", "laplace", "darcy")),
    "waves": (("wave", "vibrat", "string", "acoustic", "sound", "ultrasound", "oscillating membrane", "drum",
               "onda", "vibra", "corda", "acústic", "acustic", "som", "ultrassom"),
              ("wave", "helmholtz", "maxwell")),
    "fluid": (("flow", "fluid", "navier", "stokes", "airflow", "air flow", "aerodynam", "cfd", "reynolds", "drag",
               "lift", "pipe", "channel", "turbulen", "viscous", "shock", "burgers", "inviscid", "compressible",
               "nozzle", "jet", "escoamento", "fluido", "aerodin", "tubo", "canal", "viscos", "choque", "bocal"),
              ("navier", "euler", "burgers", "shallow_water", "stommel", "darcy", "meanline")),
    "electrostatics": (("electrostatic", "electric potential", "charge", "capacitor", "voltage", "dielectric",
                        "permittivity", "eletrost", "potencial elétrico", "potencial eletrico", "carga elétrica",
                        "tensão elétrica", "dielétric", "dieletric"),
                       ("electrostatic", "poisson", "laplace")),
    "electromagnetism": (("electromagnet", "maxwell", "magnetic", "eddy", "skin effect", "antenna", "scattering",
                          "inductance", "waveguide", "magnét", "magnet", "eletromagn", "antena", "espalhamento"),
                         ("maxwell", "magnetostatic", "eddy_current", "helmholtz", "electrostatic")),
    "structures": (("stress", "strain", "elastic", "beam", "deform", "displacement", "fracture", "crack", "fatigue",
                    "torsion", "bending", "cantilever", "von mises", "tensão mecânica", "deformação", "deformacao",
                    "viga", "elástic", "elastic", "fratura", "trinca", "fadiga", "torção", "torcao", "flexão"),
                   ("elasticity", "fracture", "thermoelasticity", "terramechanics")),
    "orbits": (("orbit", "satellite", "kepler", "spacecraft", "debris", "three-body", "lagrange point", "attitude",
                "órbita", "orbita", "satélite", "satelite", "espacial", "detrito"),
               ("kepler", "satellite", "cr3bp", "space_debris", "spacecraft", "schwarzschild")),
    "epidemics": (("epidemic", "infect", "outbreak", "sir model", "susceptible", "epidemia", "infec", "surto"),
                  ("sir",)),
    "finance": (("option", "pricing", "black-scholes", "black scholes", "heston", "volatility", "stock",
                 "opção", "opcao", "precificação", "precificacao", "volatilidade", "ação", "ações"),
                ("black_scholes", "heston")),
    "pharmacokinetics": (("pharmacokinetic", "plasma", "compartment", "dose", "farmacocin", "dosagem"),
                         ("pk_",)),
    "quantum": (("quantum", "schrodinger", "schrödinger", "wavefunction", "wave function", "quântic", "quantic"),
                ("schrodinger", "quantum")),
    "astrophysics": (("star", "stellar", "accretion", "polytrope", "dark matter", "galaxy", "estrela", "galáxia",
                      "acreção"),
                     ("lane_emden", "shakura", "poisson", "euler_compressible", "schwarzschild")),
}

STEADY_WORDS = ("steady", "steady-state", "stationary", "equilibrium", "time-independent", "estacionári",
                "estacionari", "regime permanente", "equilíbrio")
TRANSIENT_WORDS = ("transient", "over time", "time-dependent", "evolves", "evolution", "decays", "cools down",
                   "heats up", "transiente", "ao longo do tempo", "evolui", "evolução", "decai")
DIMENSION_PATTERNS = {1: (r"\b1\s*-?\s*d\b", r"one[- ]dimensional", r"unidimensional"),
                      2: (r"\b2\s*-?\s*d\b", r"two[- ]dimensional", r"bidimensional"),
                      3: (r"\b3\s*-?\s*d\b", r"three[- ]dimensional", r"tridimensional")}
_TIME_COORDS = {"t", "time", "tau"}


def _hit(text: str, word: str) -> bool:
    """``word`` at the start of a word (a stem: "vibrat" matches "vibrating"); words of four letters or fewer must
    match a whole word ("star" is not "start", "som" is not "some")."""
    tail = r"(?![a-z0-9à-ü])" if len(word) <= 4 else ""
    return re.search(r"(?<![a-z0-9à-ü])" + re.escape(word) + tail, text) is not None


def description_classes(text: str) -> dict[str, list[str]]:
    """The physics classes a description names, each with the words that matched."""
    t = text.lower()
    out = {}
    for cls, (words, _) in PHYSICS_CLASSES.items():
        hits = [w for w in words if _hit(t, w)]
        if hits:
            out[cls] = hits
    return out


def preset_classes(kind: str) -> list[str]:
    """The physics classes a preset's ``pde.kind`` belongs to."""
    k = str(kind).lower()
    return [cls for cls, (_, patterns) in PHYSICS_CLASSES.items() if any(p in k for p in patterns)]


@dataclass
class CoherenceVerdict:
    status: str                                  # "coherent" | "incoherent" | "unknown"
    reasons: list[str] = field(default_factory=list)
    description_classes: dict[str, list[str]] = field(default_factory=dict)
    preset_classes: list[str] = field(default_factory=list)
    preset: str = ""
    preset_kind: str = ""

    @property
    def coherent(self) -> bool:
        return self.status == "coherent"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons), "description_classes": self.description_classes,
                "preset_classes": self.preset_classes, "preset": self.preset, "preset_kind": self.preset_kind}


def check_preset_coherence(description: str, spec: Any) -> CoherenceVerdict:
    """Compare a problem description with a preset's ``ProblemSpec`` (see the module text)."""
    kind = str(getattr(getattr(spec, "pde", None), "kind", "") or "")
    name = str(getattr(spec, "name", "") or getattr(spec, "problem_id", "") or kind)
    coords = [str(c) for c in (getattr(spec, "coords", ()) or ())]
    spatial = [c for c in coords if c.lower() not in _TIME_COORDS]
    has_time = len(spatial) < len(coords)
    t = description.lower()
    d_cls = description_classes(description)
    p_cls = preset_classes(kind)
    reasons: list[str] = []

    if d_cls and not set(d_cls) & set(p_cls):
        said = ", ".join(f"{c} ({', '.join(w)})" for c, w in d_cls.items())
        reasons.append(f"the description is about {said}; the preset {name!r} solves {kind!r}"
                       f" ({', '.join(p_cls) or 'no recognised class'})")
    steady = any(_hit(t, w) for w in STEADY_WORDS)
    transient = any(_hit(t, w) for w in TRANSIENT_WORDS)
    if steady and not transient and has_time and spatial:
        reasons.append(f"the description is steady, the preset {name!r} evolves in time (coordinates {coords})")
    if transient and not steady and not has_time:
        reasons.append(f"the description is time-dependent, the preset {name!r} is steady (coordinates {coords})")
    said_dims = [d for d, pats in DIMENSION_PATTERNS.items() if any(re.search(p, t) for p in pats)]
    if len(said_dims) == 1 and spatial and said_dims[0] != len(spatial):
        reasons.append(f"the description is {said_dims[0]}-D, the preset {name!r} has {len(spatial)} spatial "
                       f"coordinate(s) {spatial}")

    status = "incoherent" if reasons else ("coherent" if d_cls else "unknown")
    if status == "unknown":
        reasons.append("the description names no physics this check recognises: the choice of preset is not "
                       "confirmed")
    return CoherenceVerdict(status, reasons, d_cls, p_cls, name, kind)


__all__ = ["PHYSICS_CLASSES", "CoherenceVerdict", "check_preset_coherence", "description_classes", "preset_classes"]
