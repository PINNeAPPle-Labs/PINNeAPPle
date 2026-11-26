"""The analyses PINNeAPPle ships: the closed-form models, declared with what they take and how they word it.

Every function here wraps a real model of :mod:`pinneapple_physics.closed_form` — no physics is added in this
module, only the declaration the registry needs: the unit and the range of validity of each input, the outputs and
one sentence in English and Portuguese that says what was found, with the numbers in the reader's format. The
wrapped function stays the source of the number; this module is the form, the report line and the assistant's view
of it, written once.

Each function returns its declared outputs *plus* the inputs its sentence names, because a finding that quotes a
number must carry it: the sentence is built from the result, never from what the caller says it passed.

    >>> from pinneapple_core.analysis import run_analysis
    >>> result = run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4)
    >>> round(result.metrics["f"], 4)
    0.0216

The registry loads this module on its first use (``pinneapple_core.analysis.registry.BUILTIN_MODULES``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pinneapple_core.analysis import (
    InputSpec,
    Maturity,
    OutputSpec,
    Reference,
    analysis,
    say,
    text_of,
    tr,
)

from .fin_array_conduction import fin_array_base_temperature_c, fin_array_profile_params
from .helmholtz_resonator import helmholtz_resonant_frequency_hz
from .pipe_flow import RE_LAMINAR_MAX, RE_TURBULENT_MIN
from .pipe_flow import darcy_pressure_drop as _darcy_pressure_drop
from .pipe_flow import friction_factor as _friction_factor

__all__ = ["pipe_friction_factor", "pipe_pressure_drop", "acoustics_helmholtz_frequency",
           "thermal_fin_array_base_temperature"]

_FRICTION_SENTENCE = tr("f = {f:.4} em Re = {Re:.0} com ε/D = {rel_rough:.5} ({regime})",
                        "f = {f:.4} at Re = {Re:.0} with ε/D = {rel_rough:.5} ({regime})")


def _friction_sentence(metrics: Mapping[str, Any], lang: str | None = None) -> str:
    """The finding, with the regime read in the language the rest of the sentence is in."""
    return say(_FRICTION_SENTENCE, lang, f=metrics.get("f"), Re=metrics.get("Re"),
               rel_rough=metrics.get("rel_rough"), regime=text_of(metrics.get("regime"), lang))


@analysis(
    "pipe.friction_factor",
    label=tr("fator de atrito de Darcy", "Darcy friction factor"),
    category="fluid_mechanics",
    description=tr("Fator de atrito de Darcy em tubo circular: laminar abaixo de Re = 2300, Colebrook acima de 4000.",
                   "Darcy friction factor in a circular pipe: laminar below Re = 2300, Colebrook above 4000."),
    inputs={
        "Re": InputSpec(gt=0.0, label=tr("número de Reynolds", "Reynolds number")),
        "rel_rough": InputSpec(min=0.0, max=0.05, required=False,
                               label=tr("rugosidade relativa ε/D", "relative roughness ε/D")),
    },
    outputs={
        "f": OutputSpec(unit="-", label=tr("fator de atrito", "friction factor")),
        "regime": OutputSpec(unit="", label=tr("regime do escoamento", "flow regime")),
    },
    describe=_friction_sentence,
    maturity=Maturity.LITERATURE_BASED,
    references=[Reference("White, Fluid Mechanics, eq. 6.48 (Colebrook, 1939)")],
    limitations=[tr("na faixa 2300 < Re < 4000 a correlação turbulenta é usada sem aviso",
                    "in the band 2300 < Re < 4000 the turbulent correlation is used without a warning")],
    example={"Re": 45_000.0, "rel_rough": 4.5e-4},
)
def pipe_friction_factor(Re: float, rel_rough: float = 0.0) -> dict[str, Any]:
    """Darcy friction factor and the regime it was computed in (laminar, transitional or turbulent)."""
    if Re < RE_LAMINAR_MAX:
        regime = tr("laminar", "laminar")
    elif Re < RE_TURBULENT_MIN:
        regime = tr("transição", "transitional")
    else:
        regime = tr("turbulento", "turbulent")
    return {"f": float(_friction_factor(Re, rel_rough)), "regime": regime, "Re": float(Re),
            "rel_rough": float(rel_rough)}


@analysis(
    "pipe.darcy_pressure_drop",
    label=tr("queda de pressão de Darcy–Weisbach", "Darcy–Weisbach pressure drop"),
    category="fluid_mechanics",
    description=tr("Queda de pressão de Darcy–Weisbach em um trecho de tubo de diâmetro constante.",
                   "Darcy–Weisbach pressure drop over a length of constant-diameter pipe."),
    inputs={
        "f": InputSpec(gt=0.0, label=tr("fator de atrito", "friction factor")),
        "L": InputSpec(unit="m", gt=0.0, label=tr("comprimento", "length")),
        "D": InputSpec(unit="m", gt=0.0, label=tr("diâmetro interno", "internal diameter")),
        "rho": InputSpec(unit="kg/m³", gt=0.0, label=tr("massa específica", "density")),
        "U": InputSpec(unit="m/s", min=0.0, label=tr("velocidade média", "mean velocity")),
    },
    outputs={"dp": OutputSpec(unit="Pa", label=tr("queda de pressão", "pressure drop"))},
    describe=tr("dp = {dp:.4} Pa em {L:.1} m de tubo de {D:.4} m a {U:.2} m/s",
                "dp = {dp:.4} Pa over {L:.1} m of {D:.4} m pipe at {U:.2} m/s"),
    maturity=Maturity.LITERATURE_BASED,
    references=[Reference("White, Fluid Mechanics, eq. 6.10 (Darcy–Weisbach)")],
    limitations=[tr("regime permanente, propriedades constantes, tubo reto e horizontal",
                    "steady flow, constant properties, straight and horizontal pipe")],
    example={"f": 0.0216, "L": 120.0, "D": 0.0525, "rho": 998.0, "U": 2.0},
)
def pipe_pressure_drop(f: float, L: float, D: float, rho: float, U: float) -> dict[str, float]:
    """Pressure drop over ``L`` metres of pipe, in pascals."""
    return {"dp": float(_darcy_pressure_drop(f, L, D, rho, U)),
            "f": float(f), "L": float(L), "D": float(D), "rho": float(rho), "U": float(U)}


@analysis(
    "acoustics.helmholtz_frequency",
    label=tr("frequência de ressonância de Helmholtz", "Helmholtz resonant frequency"),
    category="acoustics",
    description=tr("Frequência de ressonância de um volume com um duto (porta de bass-reflex, cavidade com gargalo).",
                   "Resonant frequency of a volume with a port (bass-reflex port, cavity with a neck)."),
    inputs={
        "port_radius_m": InputSpec(unit="m", gt=0.0, label=tr("raio do duto", "port radius")),
        "port_length_m": InputSpec(unit="m", gt=0.0, label=tr("comprimento do duto", "port length")),
        "cabinet_volume_l": InputSpec(unit="L", gt=0.0, label=tr("volume interno", "internal volume")),
        "end_correction_factor": InputSpec(min=0.0, required=False,
                                           label=tr("fator de correção de extremidade", "end correction factor")),
    },
    outputs={"f": OutputSpec(unit="Hz", label=tr("frequência de ressonância", "resonant frequency"))},
    describe=tr("um duto de r = {port_radius_m:.3} m e L = {port_length_m:.3} m sintoniza {cabinet_volume_l:.1} L em {f:.1} Hz",
                "a port of r = {port_radius_m:.3} m and L = {port_length_m:.3} m tunes {cabinet_volume_l:.1} L to {f:.1} Hz"),
    maturity=Maturity.LITERATURE_BASED,
    references=[Reference("Helmholtz resonator equation with the flanged end correction (1.7 r)")],
    limitations=[tr("som em ar a 343 m/s e um volume rígido com paredes muito maiores que o comprimento de onda",
                    "sound in air at 343 m/s in a rigid volume much larger than the wavelength")],
    example={"port_radius_m": 0.025, "port_length_m": 0.12, "cabinet_volume_l": 20.0},
)
def acoustics_helmholtz_frequency(port_radius_m: float, port_length_m: float, cabinet_volume_l: float,
                                  end_correction_factor: float = 1.7) -> dict[str, float]:
    """Resonant frequency of the port and volume, in hertz."""
    return {"f": float(helmholtz_resonant_frequency_hz(port_radius_m, port_length_m, cabinet_volume_l,
                                                       end_correction_factor)),
            "port_radius_m": float(port_radius_m), "port_length_m": float(port_length_m),
            "cabinet_volume_l": float(cabinet_volume_l)}


@analysis(
    "thermal.fin_array_base_temperature",
    label=tr("temperatura na base de uma matriz de aletas", "base temperature of a fin array"),
    category="heat_transfer",
    description=tr("Temperatura na base de uma matriz de aletas retangulares em convecção, com a correção de ponta "
                   "adiabática (Incropera & DeWitt).",
                   "Base temperature of an array of straight rectangular fins in convection, with the adiabatic-tip "
                   "correction (Incropera & DeWitt)."),
    inputs={
        "k_material_w_mk": InputSpec(unit="W/(m·K)", gt=0.0,
                                     label=tr("condutividade da aleta", "fin conductivity")),
        "h_conv_w_m2k": InputSpec(unit="W/(m²·K)", gt=0.0,
                                  label=tr("coeficiente de convecção", "convection coefficient")),
        "n_fins": InputSpec(unit="-", min=1.0, label=tr("número de aletas", "number of fins")),
        "fin_thickness_m": InputSpec(unit="m", gt=0.0, label=tr("espessura da aleta", "fin thickness")),
        "fin_length_m": InputSpec(unit="m", gt=0.0, label=tr("altura da aleta", "fin length")),
        "transverse_extent_m": InputSpec(unit="m", gt=0.0, label=tr("extensão transversal", "transverse extent")),
        "spacing_extent_m": InputSpec(unit="m", gt=0.0, label=tr("extensão em que as aletas se distribuem",
                                                                "extent the fins are spread along")),
        "base_thickness_m": InputSpec(unit="m", gt=0.0, label=tr("espessura da base", "base thickness")),
        "power_w": InputSpec(unit="W", min=0.0, label=tr("potência dissipada", "dissipated power")),
        "t_ambient_c": InputSpec(unit="°C", label=tr("temperatura do ar", "ambient temperature")),
    },
    outputs={
        "t_base_c": OutputSpec(unit="°C", label=tr("temperatura na base", "base temperature")),
        "r_conv_k_w": OutputSpec(unit="K/W", label=tr("resistência convectiva", "convective resistance")),
        "r_base_k_w": OutputSpec(unit="K/W", label=tr("resistência de espalhamento na base",
                                                      "base spreading resistance")),
    },
    describe=tr("{n_fins:.0} aletas dissipando {power_w:.1} W levam a base a {t_base_c:.1} °C com o ar a {t_ambient_c:.1} °C",
                "{n_fins:.0} fins dissipating {power_w:.1} W put the base at {t_base_c:.1} °C with the air at {t_ambient_c:.1} °C"),
    maturity=Maturity.LITERATURE_BASED,
    references=[Reference("Incropera & DeWitt, Fundamentals of Heat and Mass Transfer (fin efficiency, adiabatic tip)")],
    limitations=[tr("aletas retangulares de seção constante, radiação desprezada, convecção com h uniforme",
                    "straight rectangular fins, radiation neglected, uniform convection coefficient")],
    example={"k_material_w_mk": 200.0, "h_conv_w_m2k": 25.0, "n_fins": 20.0, "fin_thickness_m": 0.001,
             "fin_length_m": 0.03, "transverse_extent_m": 0.06, "spacing_extent_m": 0.06, "base_thickness_m": 0.005,
             "power_w": 60.0, "t_ambient_c": 25.0},
)
def thermal_fin_array_base_temperature(k_material_w_mk: float, h_conv_w_m2k: float, n_fins: float,
                                       fin_thickness_m: float, fin_length_m: float, transverse_extent_m: float,
                                       spacing_extent_m: float, base_thickness_m: float, power_w: float,
                                       t_ambient_c: float) -> dict[str, float]:
    """Base temperature and the two resistances it is built from (convection and base spreading)."""
    params = fin_array_profile_params(k_material_w_mk, h_conv_w_m2k, n_fins, fin_thickness_m, fin_length_m,
                                      transverse_extent_m, spacing_extent_m, base_thickness_m, power_w)
    return {"t_base_c": float(fin_array_base_temperature_c(params, t_ambient_c, power_w)),
            "r_conv_k_w": float(params["r_conv"]),
            "r_base_k_w": float(params["r_base"]),
            "n_fins": float(n_fins), "power_w": float(power_w), "t_ambient_c": float(t_ambient_c)}
