"""The preset catalogue: what a preset takes, what it gives back, and the smallest program that uses it.

The preset registry (``pinneapple_physics.pde_environment.presets``) stores a factory per name and
nothing else: no description, no tags, no audience. That is fine for the solvers and useless for a
person choosing what to run. This module adds the missing half, derived from the live
:class:`~pinneapple_physics.pde_environment.spec.ProblemSpec` where the spec can answer and curated
by hand where it cannot:

* :func:`card` — one preset as a :class:`Card`: coordinates, fields, conditions, the method that
  runs today, and a snippet that executes as printed.
* :func:`search` — free text, English or Portuguese, to presets, with the terms that matched.
* :func:`snippet` — the smallest correct program for one preset, choosing a method that exists.

Everything here runs offline and without training. ``search`` is keyword scoring, not a model, and
it says so: every hit carries the terms that matched so the user can judge it.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Card", "card", "describe", "format_card", "snippet", "search", "catalogue", "domains"]


# ---------------------------------------------------------------------------
# Curated half: one line per preset, and the tags a person would search with.
# Keys are preset names; the value is (domain, description, tags).
# ---------------------------------------------------------------------------
_CURATED: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    # --- academic PDEs -----------------------------------------------------
    "burgers_1d": ("fluid", "1-D Burgers equation: a decaying shock, the smallest test of a PINN.", ("shock", "advection", "viscous", "nonlinear")),
    "poisson_2d": ("academic", "2-D Poisson with a manufactured solution: the first elliptic solve.", ("elliptic", "manufactured", "laplace")),
    "laplace_2d": ("academic", "2-D Laplace on the unit square with Dirichlet sides.", ("elliptic", "harmonic", "dirichlet")),
    "reaction_diffusion_2d": ("academic", "2-D reaction-diffusion: pattern formation and stiff source terms.", ("patterns", "stiff", "chemistry")),
    "wave_ultrasound_3d": ("acoustics", "3-D acoustic wave propagation, the ultrasound imaging case.", ("wave", "transient", "imaging")),
    "helmholtz_acoustics_3d": ("acoustics", "3-D Helmholtz: steady acoustics at one frequency.", ("frequency", "sound", "steady-state")),
    "darcy_pressure_only_3d": ("fluid", "3-D Darcy flow: pressure only, the classic porous-medium surrogate.", ("porous", "permeability", "groundwater")),
    "black_scholes_1d": ("finance", "Black-Scholes option pricing as a PDE in price and time.", ("option", "pricing", "finance")),
    "heston_pde_2d": ("finance", "Heston stochastic-volatility PDE in price and variance.", ("option", "volatility", "finance")),
    "sir_epidemic": ("life sciences", "SIR epidemic compartment model over time and space.", ("epidemic", "infection", "population")),
    "opinion_dynamics_2d": ("social", "Opinion dynamics: consensus and polarisation on a 2-D domain.", ("agent", "consensus", "social")),
    "pk_two_compartment": ("life sciences", "Two-compartment pharmacokinetics: drug concentration over time.", ("pharmacokinetics", "drug", "compartment")),
    "drug_diffusion_tissue": ("life sciences", "Drug diffusion into tissue from a source.", ("diffusion", "drug", "tissue")),
    "lane_emden_polytrope": ("astrophysics", "Lane-Emden polytrope: the density profile of a star.", ("star", "polytrope", "structure")),
    "crystal_phonon": ("physics", "Phonon transport in a crystal lattice.", ("phonon", "lattice", "heat")),
    # --- solid mechanics ---------------------------------------------------
    "linear_elasticity_3d": ("structural", "3-D linear elasticity with Dirichlet supports.", ("stress", "strain", "displacement")),
    "linear_elasticity_3d_industry": ("structural", "3-D elasticity sized as an industrial bracket.", ("stress", "bracket", "industrial")),
    "plane_stress_2d": ("structural", "2-D plane stress, the thin-plate idealisation.", ("stress", "thin", "plate")),
    "plane_strain_2d": ("structural", "2-D plane strain, the long-body idealisation.", ("strain", "long", "section")),
    "axisymmetric_linear_elasticity_2d": ("structural", "Axisymmetric elasticity: a solid of revolution in (r, z).", ("axisymmetric", "revolution", "stress")),
    "von_mises_2d": ("structural", "2-D field driven to a target von Mises stress.", ("yield", "stress", "criterion")),
    "thermoelasticity_2d": ("structural", "Coupled thermal expansion and stress in 2-D.", ("thermal stress", "expansion", "coupled")),
    "material_fracture_2d": ("structural", "2-D crack field: stress concentration around a fracture.", ("crack", "fracture", "concentration")),
    "thick_walled_cylinder_lame": ("structural", "Lame solution for a pressurised thick-walled cylinder.", ("pressure vessel", "cylinder", "hoop stress")),
    "rotary_coupling_torsion": ("structural", "Torsion in a rotary coupling.", ("torsion", "shaft", "torque")),
    "threaded_coupling_tc50_box": ("structural", "Threaded coupling TC50, box side: contact stress in the thread.", ("thread", "coupling", "contact")),
    "threaded_coupling_tc50_pin": ("structural", "Threaded coupling TC50, pin side.", ("thread", "coupling", "contact")),
    "threaded_coupling_tc50_rotating": ("structural", "Threaded coupling TC50 under rotation, with the centrifugal load.", ("thread", "rotating", "coupling")),
    "bekker_wong_surrogate_2d": ("terramechanics", "Bekker-Wong soil pressure-sinkage surrogate for a wheel.", ("soil", "wheel", "sinkage")),
    "aircraft_wing_structural": ("aerospace", "Wing box under lift load: structural sizing of the spar.", ("wing", "spar", "lift")),
    "rocket_structural": ("aerospace", "Rocket case under internal pressure and axial load.", ("pressure vessel", "rocket", "case")),
    "car_suspension_fatigue": ("automotive", "Suspension arm fatigue under a road load history.", ("fatigue", "suspension", "cyclic")),
    # --- heat transfer -----------------------------------------------------
    "steady_heat_conduction_3d": ("thermal", "Steady 3-D conduction in a solid body.", ("conduction", "steady", "temperature")),
    "transient_heat_3d": ("thermal", "Transient 3-D conduction: the body heating up over time.", ("conduction", "transient", "temperature")),
    "cpu_heatsink_thermal": ("thermal", "CPU heat sink with a heat load and convective cooling: the canonical thermal case.", ("heatsink", "electronics", "convection")),
    "pcb_thermal": ("thermal", "Power PCB junction and board temperature.", ("electronics", "pcb", "junction")),
    "car_brake_thermal": ("automotive", "Brake disc heating during a stop.", ("brake", "disc", "friction")),
    "industrial_furnace_thermal": ("thermal", "Industrial furnace wall temperatures and heat losses.", ("furnace", "refractory", "heat loss")),
    "furnace_combustion_zone": ("thermal", "Combustion zone of a furnace: temperature field from the flame.", ("combustion", "flame", "furnace")),
    "refractory_lining": ("thermal", "Refractory lining of a high-temperature vessel.", ("refractory", "lining", "insulation")),
    "datacenter_server_thermal": ("thermal", "Server-level thermal field inside a rack.", ("datacenter", "server", "electronics")),
    # --- CFD ---------------------------------------------------------------
    "ns_incompressible_2d": ("fluid", "2-D incompressible Navier-Stokes with inlet, walls and outlet.", ("navier-stokes", "flow", "incompressible")),
    "channel_flow_3d": ("fluid", "3-D channel flow with a parabolic inlet profile.", ("channel", "duct", "flow")),
    "pipe_flow_3d": ("fluid", "3-D pipe flow: the case to start from for a pipe or a duct.", ("pipe", "tube", "pressure drop")),
    "lid_driven_cavity_3d": ("fluid", "3-D lid-driven cavity, the standard recirculation benchmark.", ("cavity", "recirculation", "benchmark")),
    "aircraft_wing_aerodynamics": ("aerospace", "External flow over a wing section: lift and drag from the field.", ("airfoil", "wing", "lift", "drag")),
    "car_external_aero": ("automotive", "External aerodynamics of a car body.", ("car", "drag", "external flow")),
    "rocket_nozzle_cfd": ("aerospace", "Compressible flow through a rocket nozzle.", ("nozzle", "compressible", "thrust")),
    "fan_cooler_cfd": ("thermal", "Fan cooler airflow: the forced-convection case.", ("fan", "cooling", "forced convection")),
    "datacenter_airflow_2d": ("fluid", "2-D airflow through a data-centre aisle.", ("datacenter", "airflow", "cooling")),
    "datacenter_cfd_3d": ("fluid", "3-D data-centre room airflow and hot spots.", ("datacenter", "airflow", "3d")),
    "climate_atmosphere_2d": ("climate", "2-D atmosphere slice: a climate dynamics case.", ("atmosphere", "climate", "weather")),
    "climate_ocean_gyre": ("climate", "Ocean gyre circulation on a 2-D basin.", ("ocean", "gyre", "circulation")),
    "axial_compressor_cascade_2d": ("turbomachinery", "2-D compressor cascade: blade-to-blade flow.", ("compressor", "cascade", "blade")),
    "axial_compressor_meanline": ("turbomachinery", "Compressor meanline design point.", ("compressor", "meanline", "design")),
    "axial_compressor_stage_3d": ("turbomachinery", "3-D compressor stage flow.", ("compressor", "stage", "3d")),
    # --- electromagnetics --------------------------------------------------
    "em_cylinder_scattering": ("electromagnetics", "Scattering of a plane wave by a cylinder.", ("scattering", "wave", "cylinder")),
    "em_skin_effect": ("electromagnetics", "Skin effect: current crowding at high frequency.", ("skin effect", "current", "frequency")),
    "em_tm_cavity": ("electromagnetics", "TM mode of a resonant cavity, by the differential formulation.", ("cavity", "resonance", "mode")),
    "em_tm_cavity_integral": ("electromagnetics", "The same TM cavity by the integral formulation.", ("cavity", "integral", "comparison")),
    "em_two_dielectric_capacitor": ("electromagnetics", "Capacitor with two dielectrics: field at the interface.", ("capacitor", "dielectric", "electrostatics")),
    "em_two_dielectric_capacitor_differential": ("electromagnetics", "The two-dielectric capacitor by the differential formulation.", ("capacitor", "dielectric", "comparison")),
    "em_wire_az": ("electromagnetics", "Field of a wire in the azimuthal plane.", ("wire", "field", "magnetostatics")),
    # --- space and astrophysics --------------------------------------------
    "kepler_two_body_orbit": ("astrodynamics", "Kepler two-body orbit: position and velocity from the elements.", ("orbit", "kepler", "two-body")),
    "cr3bp_planar_synodic": ("astrodynamics", "Planar circular restricted three-body problem, synodic frame.", ("three-body", "libration", "trajectory")),
    "satellite_j2_perturbation": ("astrodynamics", "J2 perturbation of a satellite orbit.", ("orbit", "perturbation", "earth")),
    "space_debris_cw_relative_motion": ("astrodynamics", "Clohessy-Wiltshire relative motion: rendezvous and debris.", ("rendezvous", "relative", "debris")),
    "spacecraft_attitude_euler_rotation": ("astrodynamics", "Spacecraft attitude from Euler rotation dynamics.", ("attitude", "rotation", "control")),
    "nfw_dark_matter_potential": ("astrophysics", "NFW dark-matter halo potential.", ("halo", "dark matter", "potential")),
    "schwarzschild_light_bending_weak_field": ("astrophysics", "Weak-field light bending around a Schwarzschild mass.", ("gravity", "lensing", "relativity")),
    "shakura_sunyaev_accretion_disk": ("astrophysics", "Shakura-Sunyaev accretion disc profile.", ("accretion", "disc", "temperature")),
    "sod_shock_tube_astro": ("astrophysics", "Sod shock tube: the compressible Riemann benchmark.", ("shock", "riemann", "compressible")),
}

# Domain synonyms so a description written by a person, in either language, reaches the right models.
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "heat": ("calor", "temperatura", "aquecimento", "resfriamento", "termico", "termica", "conducao", "conveccao"),
    "flow": ("escoamento", "fluido", "fluidos", "vazao", "cfd", "aerodinamica", "hidrodinamica"),
    "pressure": ("pressao", "queda de pressao"),
    "pipe": ("tubo", "tubulacao", "duto", "cano", "encanamento"),
    "stress": ("tensao", "tensoes", "esforco", "esforcos"),
    "strain": ("deformacao", "deslocamento"),
    "fatigue": ("fadiga", "ciclico", "ciclica"),
    "crack": ("trinca", "fratura", "rachadura"),
    "vibration": ("vibracao", "oscilacao"),
    "acoustics": ("acustica", "som", "ruido", "sonoro"),
    "electromagnetics": ("eletromagnetico", "eletromagnetismo", "campo eletrico", "campo magnetico", "antena", "rf"),
    "orbit": ("orbita", "satelite", "espacial"),
    "epidemic": ("epidemia", "doenca", "contagio"),
    "finance": ("financas", "opcao", "derivativo", "mercado"),
    "soil": ("solo", "terreno", "pneu"),
    "fan": ("ventilador", "ventoinha"),
    "furnace": ("forno", "fornalha", "caldeira"),
    "brake": ("freio", "disco de freio"),
    "wing": ("asa", "aerofolio", "perfil aerodinamico"),
    "compressor": ("compressor", "turbina", "turbomaquina"),
    "nozzle": ("bocal", "bico", "foguete"),
    "cooling": ("refrigeracao", "arrefecimento", "dissipador", "cooler", "aleta", "aletas", "fin", "fins"),
    "heatsink": ("aleta", "aletas", "dissipador", "fin", "fins", "heat sink"),
    "plate": ("placa", "chapa", "painel", "panel"),
}

# Domain words: when one appears in the query, every preset of that domain gets a boost. This is
# what stops "flow past a cylinder" from landing on a thick-walled cylinder because of one noun.
_DOMAIN_WORDS: Dict[str, Tuple[str, ...]] = {
    "fluid": ("flow", "escoamento", "fluido", "cfd", "navier", "turbulence", "turbulencia", "wake",
              "esteira", "incompressible", "airflow", "aerodinamica"),
    "thermal": ("heat", "calor", "temperature", "temperatura", "thermal", "termico", "cooling",
                "resfriamento", "conduction", "conducao", "convection", "conveccao", "heatsink",
                "cooler", "aleta", "dissipador"),
    "structural": ("stress", "tensao", "strain", "deformacao", "elasticity", "elasticidade", "crack",
                   "trinca", "fracture", "fratura", "fatigue", "fadiga", "bracket", "suporte",
                   "beam", "viga", "torsion", "torcao", "lamina"),
    "acoustics": ("acoustic", "acustica", "sound", "som", "noise", "ruido", "ultrasound", "ultrassom"),
    "electromagnetics": ("electric", "eletrico", "magnetic", "magnetico", "electromagnetic",
                         "eletromagnetico", "antenna", "antena", "capacitor", "dielectric", "dieletrico",
                         "scattering", "espalhamento", "skin"),
    "astrodynamics": ("orbit", "orbita", "satellite", "satelite", "spacecraft", "trajectory",
                      "trajetoria", "rendezvous", "attitude"),
    "astrophysics": ("star", "estrela", "galaxy", "galaxia", "halo", "dark", "accretion", "acrescao",
                     "schwarzschild", "polytrope"),
    "finance": ("option", "opcao", "finance", "financeiro", "volatility", "volatilidade", "pricing"),
    "life sciences": ("epidemic", "epidemia", "drug", "droga", "farmaco", "tissue", "tecido",
                      "pharmacokinetics", "infection", "contagio"),
    "turbomachinery": ("compressor", "turbine", "turbina", "blade", "pa", "cascade", "cascata"),
    "aerospace": ("wing", "asa", "aircraft", "aeronave", "rocket", "foguete", "nozzle", "bocal",
                  "flight", "voo"),
    "automotive": ("car", "carro", "vehicle", "veiculo", "brake", "freio", "suspension", "suspensao",
                   "tire", "pneu"),
    "climate": ("climate", "clima", "ocean", "oceano", "atmosphere", "atmosfera", "weather", "tempo"),
    "academic": ("poisson", "laplace", "burgers", "manufactured", "benchmark"),
    "physics": ("phonon", "foton", "lattice", "reticulo", "crystal", "cristal"),
}

_STOPWORDS = frozenset(
    """a o as os de da do das dos em no na nos nas um uma uns umas para por com sem que e ou the of in on at to \
    for with without and or a an is are be been this that my our we i it its how can do does what which where""".split()
)


def _norm(text: str) -> str:
    """Lowercase, strip accents, and collapse anything that is not a letter or digit to a space."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in _norm(text).split() if t and t not in _STOPWORDS and len(t) > 2]


def _expand(tokens: Sequence[str]) -> Dict[str, float]:
    """Token → weight, with the aliases added: a Portuguese word scores the English tag it maps to."""
    weights: Dict[str, float] = {}
    for t in tokens:
        weights[t] = max(weights.get(t, 0.0), 1.0)
    for english, aliases in _ALIASES.items():
        for alias in aliases:
            alias_tokens = _tokens(alias)
            if any(a in weights for a in alias_tokens):
                weights[english] = max(weights.get(english, 0.0), 0.9)
    return weights


@dataclass
class Card:
    """One preset as a person needs it: what it is, what goes in, what comes out, how to run it."""

    name: str
    domain: str
    description: str
    tags: Tuple[str, ...]
    coordinates: Tuple[str, ...] = ()
    fields: Tuple[str, ...] = ()
    spatial_dim: int = 0
    transient: bool = False
    domain_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    conditions: Tuple[str, ...] = ()
    solver: str = ""
    method: str = "pinn"
    snippet: str = ""
    related: Tuple[str, ...] = ()
    available: bool = True
    error: str = ""
    requires: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "domain": self.domain, "description": self.description,
            "tags": list(self.tags), "coordinates": list(self.coordinates), "fields": list(self.fields),
            "spatial_dim": self.spatial_dim, "transient": self.transient,
            "domain_bounds": {k: list(v) for k, v in self.domain_bounds.items()},
            "conditions": list(self.conditions), "solver": self.solver, "method": self.method,
            "snippet": self.snippet, "related": list(self.related), "available": self.available,
            "error": self.error, "requires": self.requires,
        }

    def format(self) -> str:
        """The card as text, ready to print."""
        return format_card(self)

    def __str__(self) -> str:  # ``print(pp.card("burgers_1d"))`` shows the card, not a dataclass repr
        return self.format()


def _spec(name: str):
    from pinneapple_physics.pde_environment import get_preset

    return get_preset(name)


def best_method(name: str) -> str:
    """The cheapest method that actually runs for this preset today: analytic, else the reference
    solver, else a trained network. Chosen from what the library registers and what this machine
    has, not from a table."""
    try:
        from pinneapple_physics.solving import analytic_for

        if analytic_for(name):
            return "analytic"
    except Exception:
        pass
    try:
        solver = str(_spec(name).solver_spec.get("name", ""))
    except Exception:
        solver = ""
    return "reference" if solver and solver_available(solver) else "pinn"


#: What an external reference solver needs before it can run. A solver missing from this table is
#: assumed to be pure Python inside the library.
_SOLVER_REQUIREMENTS: Dict[str, Tuple[Tuple[str, ...], str, str]] = {
    # solver name: (import probes, what to install, the sentence a user reads)
    "fenics": (("dolfin", "dolfinx"), 'pip install "pinneapple[fenics]"',
               "FEniCS is not installed"),
    "openfoam": ((), "install OpenFOAM and source its env",
                 "OpenFOAM is not on PATH"),
    "turbodesigner": (("turbodesigner",), "pip install turbodesigner",
                      "turbodesigner is not installed"),
}


def solver_available(solver: str) -> bool:
    """Can this machine run the named reference solver right now?"""
    if not solver:
        return False
    req = _SOLVER_REQUIREMENTS.get(str(solver).lower())
    if req is None:
        return True
    probes, _install, _why = req
    if not probes:
        import shutil

        return shutil.which(str(solver)) is not None
    import importlib.util

    for mod in probes:
        try:
            if importlib.util.find_spec(mod) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def solver_requirement(solver: str) -> str:
    """The install command for a reference solver this machine does not have, or ``""``."""
    req = _SOLVER_REQUIREMENTS.get(str(solver).lower())
    if req is None or solver_available(solver):
        return ""
    _probes, install, why = req
    return f"{install} ({why})"


def snippet(name: str, method: Optional[str] = None, *, train: bool = False) -> str:
    """The smallest correct program for one preset. Runs as printed: no placeholders to fill in."""
    method = method or best_method(name)
    coords: Tuple[str, ...] = ()
    bounds: Dict[str, Tuple[float, float]] = {}
    try:
        spec = _spec(name)
        coords = tuple(getattr(spec, "coords", ()) or ())
        bounds = {k: tuple(v) for k, v in (getattr(spec, "domain_bounds", {}) or {}).items()}
    except Exception:
        pass
    point = ", ".join(f"{(bounds[c][0] + bounds[c][1]) / 2.0:.3g}" if c in bounds else "0.5" for c in coords) or "0.5"
    labels = ", ".join(coords) if coords else "coordinates"
    lines = ["import numpy as np", "import pinneapple as pp", ""]
    if train or method == "pinn":
        lines.append(f"sol = pp.solve({name!r}, method='pinn', epochs=2000)   # trains; minutes on a laptop")
    else:
        # The classical method is the cheap one, but it can refuse a problem its solver does not
        # cover (a field it does not compute) and it can need an extra. The printed program falls
        # back to a trained network instead of failing, and says which one ran.
        lines += [
            "try:",
            f"    sol = pp.solve({name!r}, method={method!r})   # the classical answer: seconds",
            "except pp.MethodNotAvailable:   # this install has no solver for every field of this problem",
            f"    sol = pp.solve({name!r}, method='pinn', epochs=2000)   # train instead: minutes",
        ]
    lines += [
        "print(sol.method, sol.metadata())   # which method actually ran",
        "",
        "# ask the solution for values anywhere in the domain",
        f"X = np.array([[{point}]])   # one point ({labels}), in this order",
        "print(sol.predict(X))          # one column per field",
    ]
    return "\n".join(lines)


def card(name: str, *, with_related: bool = True) -> Card:
    """Describe one preset. Raises ``KeyError`` for an unknown name, with the near misses."""
    domain, description, tags = _CURATED.get(name, ("", "", ()))
    try:
        spec = _spec(name)
    except Exception as e:  # an unregistered name: the registry already lists what exists
        raise KeyError(str(e)) from None
    coords = tuple(getattr(spec, "coords", ()) or ())
    fields = tuple(getattr(spec, "fields", ()) or ())
    transient = "t" in coords
    spatial = len([c for c in coords if c != "t"])
    conditions = tuple(getattr(c, "name", "") for c in getattr(spec, "conditions", ()) or ())
    solver = str((getattr(spec, "solver_spec", {}) or {}).get("name", ""))
    kind = str(getattr(getattr(spec, "pde", None), "kind", "") or "")
    if not description:
        description = f"{kind.replace('_', ' ')} in {spatial}-D" + (" + time" if transient else "")
    return Card(
        name=name,
        domain=domain or (kind.split("_")[0] if kind else "general"),
        description=description,
        tags=tags,
        coordinates=coords,
        fields=fields,
        spatial_dim=spatial,
        transient=transient,
        domain_bounds={k: tuple(v) for k, v in (getattr(spec, "domain_bounds", {}) or {}).items()},
        conditions=conditions,
        solver=solver,
        method=best_method(name),
        snippet=snippet(name),
        related=_related(name, tags, domain) if with_related else (),
        requires=solver_requirement(solver),
    )


def _related(name: str, tags: Sequence[str], domain: str, k: int = 3) -> Tuple[str, ...]:
    scored = []
    for other, (o_domain, _d, o_tags) in _CURATED.items():
        if other == name:
            continue
        overlap = len(set(tags) & set(o_tags))
        if overlap or o_domain == domain:
            scored.append((overlap + (1 if o_domain == domain else 0), other))
    scored.sort(reverse=True)
    return tuple(n for _, n in scored[:k])


def describe(name: str) -> Dict[str, Any]:
    """``card(name)`` as a JSON-friendly dict."""
    return card(name).as_dict()


def catalogue() -> List[Card]:
    """Every registered preset as a card, sorted by domain then name."""
    from pinneapple_physics.pde_environment import list_presets

    out = []
    for n in list_presets():
        try:
            out.append(card(n, with_related=False))
        except Exception:
            continue
    return sorted(out, key=lambda c: (c.domain, c.name))


def domains() -> Dict[str, List[str]]:
    """Preset names grouped by domain, for a menu that fits on one screen."""
    out: Dict[str, List[str]] = {}
    for c in catalogue():
        out.setdefault(c.domain or "general", []).append(c.name)
    return dict(sorted(out.items()))


def format_card(c: Card, width: int = 78) -> str:
    """The card as text, printed or returned. No colour and no box drawing: it survives a pipe."""
    bar = "-" * width
    out = [bar, f"  {c.name}  [{c.domain}]", bar, f"  {c.description}", ""]
    dims = f"{c.spatial_dim}-D" + (" + time" if c.transient else "")
    out.append(f"  takes    : {', '.join(c.coordinates)}   ({dims})")
    out.append(f"  gives    : {', '.join(c.fields)}")
    if c.domain_bounds:
        bounds = "; ".join(f"{k} in [{lo:g}, {hi:g}]" for k, (lo, hi) in c.domain_bounds.items())
        out.append(f"  domain   : {bounds}")
    if c.conditions:
        out.append(f"  applies  : {', '.join(c.conditions)}")
    out.append(f"  method   : {c.method}" + (f"   (solver: {c.solver})" if c.solver else ""))
    if c.requires:
        out.append(f"  needs    : {c.requires}")
        out.append("             without it the snippet below trains a network instead")
    if c.tags:
        out.append(f"  tags     : {', '.join(c.tags)}")
    if c.related:
        out.append(f"  similar  : {', '.join(c.related)}")
    out += ["", "  run it:", ""]
    out += ["    " + line for line in c.snippet.splitlines()]
    out.append(bar)
    return "\n".join(out)


def _idf() -> Dict[str, float]:
    """Inverse document frequency over the curated catalogue: a rare word says more than a common one."""
    df: Dict[str, int] = {}
    for name, (domain, description, tags) in _CURATED.items():
        seen = set(_tokens(" ".join([name.replace("_", " "), description, " ".join(tags), domain])))
        for t in seen:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(_CURATED))
    return {t: 1.0 + math.log(n / c) for t, c in df.items()}


_IDF_CACHE: Optional[Dict[str, float]] = None


def _idf_cached() -> Dict[str, float]:
    global _IDF_CACHE
    if _IDF_CACHE is None:
        _IDF_CACHE = _idf()
    return _IDF_CACHE


def search(text: str, top: int = 5, *, min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Rank presets against a free-text description, English or Portuguese.

    Returns dicts with ``name``, ``score`` (0-1, relative to the best hit), ``matched`` (the terms
    that hit, so the ranking can be judged) and the card fields a caller needs to explain it.

    Two things keep it honest: a term counts for more when few presets use it (``crack`` beats
    ``plane``), and a domain named in the query boosts every preset of that domain (``flow past a
    cylinder`` is a fluid problem, whatever else the sentence contains).
    """
    query_tokens = _tokens(text)
    weights = _expand(query_tokens)
    if not weights:
        return []
    idf = _idf_cached()
    fluid_domains = {d for d, words in _DOMAIN_WORDS.items() if set(query_tokens) & set(words)}
    raw: List[Tuple[float, str, List[str]]] = []
    query_bigrams = [f"{a} {b}" for a, b in zip(query_tokens, query_tokens[1:], strict=False)]
    for name, (domain, description, tags) in _CURATED.items():
        name_tokens = set(_tokens(name.replace("_", " ")))
        tag_tokens = set(_tokens(" ".join(tags)))
        desc_tokens = set(_tokens(description))
        domain_tokens = set(_tokens(domain))
        # the whole corpus, for the phrases the words alone miss ("heat sink" vs "heatsink")
        corpus = _norm(" ".join([name.replace("_", " "), description, " ".join(tags), domain]))
        score, matched = 0.0, []
        for token, w in weights.items():
            rare = idf.get(token, 1.0)
            if token in name_tokens:
                score += 3.0 * w * rare
                matched.append(token)
            if token in tag_tokens:
                score += 3.0 * w * rare
                matched.append(token)
            if token in domain_tokens:
                score += 2.0 * w * rare
                matched.append(token)
            if token in desc_tokens:
                score += 1.0 * w * rare
                matched.append(token)
        for phrase in query_bigrams:
            if phrase in corpus:
                score += 2.5 * idf.get(phrase.split()[0], 1.0)
                matched.append(phrase)
        # A word that names a domain decides what the query is about: a match outside the domains the
        # query named counts half, a match inside them counts half again more.
        if fluid_domains:
            score *= 1.5 if domain in fluid_domains else 0.5
        if score > 0:
            raw.append((score, name, sorted(set(matched))))
    if not raw:
        return []
    raw.sort(key=lambda r: (-r[0], r[1]))          # ties go alphabetically, not by luck
    best = raw[0][0]
    out = []
    for score, name, matched in raw[:top]:
        rel = score / best
        if rel < min_score:
            continue
        domain, description, tags = _CURATED[name]
        out.append({
            "name": name, "score": round(rel, 3), "raw_score": round(score, 2),
            "matched": matched, "domain": domain, "description": description,
            "tags": list(tags), "method": best_method(name),
        })
    return out
