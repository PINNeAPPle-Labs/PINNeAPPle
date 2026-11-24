"""Three studies written as data, with the analytic evaluator each one runs in the tests.

They are the shape a user gets from the Studio wizard: a kind, a base, a tool with its input mapping and outputs with
units, objectives, limits (one of them relative to the baseline) and a budget. None of them needs a solver to run, so
they are also the studies the library's own tests measure.

The evaluators are **models**, and every number they return is a measurement of that model, not of a real wing, a real
heatsink or a real airfoil. Each model is a closed-form engineering approximation with its validity limit stated, and
each refuses the designs where its own approximation does not hold (a Reynolds number past the laminar limit, a Mach
number past the Prandtl-Glauert singularity) instead of returning a number anyway. A real study replaces the
evaluator, not the spec: the same ``Evaluator`` section names ``openfoam.rans_airfoil`` and the adapter the app builds
from the catalog tool.

* :func:`wing_study` / :func:`wing_vlm` — a transport wing on finite-wing theory (Helmbold's lift slope, a span
  efficiency that peaks near a 0.4-0.5 taper, a structural mass that grows with span). Two objectives (drag and mass)
  and two limits, one absolute (lift) and one relative to the baseline (mass).
* :func:`heatsink_study` / :func:`heatsink_model` — a forced-convection fin array on the parallel-plate correlations
  (Nu = 4.86, straight-fin efficiency, Darcy friction of two parallel plates). Objectives: thermal resistance and mass;
  limits: pressure drop and thermal resistance relative to the baseline; the model refuses laminar-invalid channels.
* :func:`polar_study` / :func:`polar_model` — an airfoil polar over Mach and angle of attack, a **surrogate** study:
  it measures, fits and cross-validates, and does not optimise.
* :func:`beam_study` / :func:`beam_fdm` — a cantilever box beam whose section is designed against **a numerical
  solver** (the 4th-order finite-difference Euler-Bernoulli solve of ``pinneapple_simulation.numerical_solvers``,
  not a closed-form formula): the reference here is solve of the BVP, so a study that verifies on it verifies on a
  solver. It is the case #79 asks for: mass against deflection, a stress cap and a deflection cap.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .spec import (
    Base,
    Budget,
    DesignStudy,
    Evaluator,
    Limit,
    Model,
    Objective,
    Quantity,
    Variable,
    Verifier,
)

__all__ = ["wing_study", "wing_vlm", "heatsink_study", "heatsink_model", "polar_study", "polar_model", "beam_study",
           "beam_fdm", "EXAMPLES", "example"]


# =========================================================================== the wing
def wing_vlm(params: Mapping[str, float]) -> dict[str, float]:
    """Finite-wing approximation of a wing at a fixed angle of attack. Every number below is in SI unless stated.

    * lift slope, Helmbold's form: ``a = 2 pi AR / (AR + 2)`` [1/rad], which recovers the 2-D value 2 pi for AR -> inf
      and is the standard finite-span correction for a straight tapered wing;
    * span efficiency ``e(taper) = 0.95 - 0.6 (taper - 0.45)^2`` [-]: it peaks at a taper of 0.45, the elliptic-load
      region, and falls off either side (a project decision, not a measurement, and the only place the taper enters);
    * ``cl = a (alpha - alpha0)`` [-], ``cd = cd0 + cl^2 / (pi e AR)`` [-] with ``cd0 = 0.008`` [-];
    * span ``b = sqrt(AR S)`` [m], mass ``m = 12 kg/m^2 * S * (b / 10 m)^{1/2}`` [kg]: a structure that grows with
      span, which is what makes aspect ratio a trade-off at all.

    Conditions (fixed by the study): ``alpha_deg`` [deg], ``alpha0_deg`` [deg], ``cd0`` [-], ``area_m2`` [m^2].
    """
    ar = float(params["aspect_ratio"])
    taper = float(params["taper_ratio"])
    alpha = math.radians(float(params["alpha_deg"]) - float(params.get("alpha0_deg", -2.0)))
    s = float(params["area_m2"])
    e = 0.95 - 0.6 * (taper - 0.45) ** 2
    cl = (2 * math.pi * ar / (ar + 2.0)) * alpha
    cd = float(params.get("cd0", 0.008)) + cl ** 2 / (math.pi * e * ar)
    b = math.sqrt(ar * s)
    mass = 12.0 * s * math.sqrt(b / 10.0)
    return {"cl": cl, "cd": cd, "mass_kg": mass}


def wing_study() -> DesignStudy:
    """Two objectives (drag and mass) and two limits: a lift floor, and a mass no more than 5 % above the baseline."""
    return DesignStudy.create(
        kind="optimize", title="Wing planform: lower drag without a heavier wing",
        statement="Find a planform with the least drag and the least structural mass, keeping the lift coefficient "
                  "at 5 deg above 0.55 and the mass within 5 % of the reference design.",
        base=Base(kind="geometry", ref="pp.geometries:wing_planform",
                  variables=[Variable("aspect_ratio", 4.0, 10.0, unit="1", description="span^2 / area [-]"),
                             Variable("taper_ratio", 0.30, 0.80, unit="1",
                                      description="tip chord / root chord [-]")],
                  fixed={"area_m2": 30.0}, baseline={"aspect_ratio": 7.0, "taper_ratio": 0.50},
                  description="straight tapered wing, 30 m^2 of reference area"),
        evaluator=Evaluator(tool="vlm.wing_planform", fidelity="model",
                            inputs={"aspect_ratio": "aspect_ratio", "taper_ratio": "taper_ratio"},
                            conditions={"alpha_deg": 5.0, "alpha0_deg": -2.0, "cd0": 0.008, "area_m2": 30.0},
                            outputs=[Quantity("cl", "1", label="lift coefficient"),
                                     Quantity("cd", "1", label="drag coefficient"),
                                     Quantity("mass_kg", "kg", label="structural mass")],
                            description="finite-wing lift slope with span efficiency and a span-driven mass"),
        objectives=[Objective("cd", "min", label="drag coefficient"),
                    Objective("mass_kg", "min", label="structural mass")],
        constraints=[Limit("cl", lower=0.55, label="lift coefficient"),
                     Limit("mass_kg", upper_frac_of_baseline=1.05, label="mass")],
        budget=Budget(max_evaluations=20, n_doe=10, n_verify=3, surrogate_evaluations=600, seed=3),
        verifier=Verifier(surrogate_target=0.10, min_verified=1, folds=5),
        model=Model(families=("gp", "quadratic")),
        metadata={"example": "wing", "model": "finite-wing approximation, not a solver"})


# =========================================================================== the heatsink
_AIR = {"k_w_mk": 0.0263, "nu_m2_s": 1.57e-5, "rho_kg_m3": 1.16}      # dry air at 300 K, 1 atm
_NU_PARALLEL = 4.86                                                  # fully developed parallel plates, T wall
_RE_LAMINAR = 2300.0                                                 # the limit of the correlations used below
_ALUMINIUM = {"k_w_mk": 205.0, "rho_kg_m3": 2700.0}


def heatsink_model(params: Mapping[str, float]) -> dict[str, Any]:
    """Forced-convection fin array, ducted, laminar parallel plates.

    ``W`` [m] duct width, ``L`` [m] duct length, ``u`` [m/s] air speed, ``t_base`` [m] base thickness — the study's
    conditions. Variables: fin height ``H`` [mm], fin thickness ``t`` [mm] and fin pitch ``p`` [mm] (the centre-to-
    centre distance: the number of fins is ``W / p``, which is the variable a designer actually turns, and it keeps
    the response smooth).

    * channel gap ``g = p - t`` [m]; hydraulic diameter ``Dh = 2 g`` [m] (parallel plates);
    * Reynolds ``Re = u Dh / nu`` [-]; **the model refuses** ``Re > 2300``, where Nu = 4.86 and f = 96/Re (both
      laminar) stop applying. It says so in words instead of returning a number;
    * ``h = Nu k_air / Dh`` [W/m^2/K]; straight-fin efficiency ``eta = tanh(mH)/(mH)`` with ``m = sqrt(2h/(k_Al t))``
      [1/m];
    * wetted area ``A = 2 N H L + (N - 1) g L`` [m^2]; ``R = 1 / (h (eta A_fin + A_base))`` [K/W];
    * mass ``= rho_Al (N t H L + W L t_base)`` [kg];
    * pressure drop ``dp = (96/Re) (L/Dh) rho u^2 / 2`` [Pa] (Darcy friction of two parallel plates).
    """
    w, ell, u = float(params["width_m"]), float(params["length_m"]), float(params["speed_m_s"])
    t_base = float(params["base_thickness_m"])
    h_fin = float(params["fin_height_mm"]) / 1000.0
    t_fin = float(params["fin_thickness_mm"]) / 1000.0
    pitch = float(params["fin_pitch_mm"]) / 1000.0
    n = w / pitch
    if n < 1.0:
        return {"ok": False, "message": f"a pitch of {pitch * 1000:.2f} mm leaves {n:.2f} fin(s) in "
                                        f"{w * 1000:.0f} mm"}
    gap = pitch - t_fin
    if gap <= 0:
        return {"ok": False, "message": f"a fin of {t_fin * 1000:.2f} mm does not fit in a pitch of "
                                        f"{pitch * 1000:.2f} mm"}
    dh = 2.0 * gap
    re = u * dh / _AIR["nu_m2_s"]
    if re > _RE_LAMINAR:
        return {"ok": False, "message": f"Re = {re:.0f} in the {gap * 1000:.2f} mm channel is above the laminar "
                                        f"limit {_RE_LAMINAR:.0f} of these correlations"}
    h = _NU_PARALLEL * _AIR["k_w_mk"] / dh
    m = math.sqrt(2.0 * h / (_ALUMINIUM["k_w_mk"] * t_fin))
    eta = math.tanh(m * h_fin) / (m * h_fin)
    a_fin = 2.0 * n * h_fin * ell
    a_base = (n - 1.0) * gap * ell
    r_th = 1.0 / (h * (eta * a_fin + a_base))
    mass = _ALUMINIUM["rho_kg_m3"] * (n * t_fin * h_fin * ell + w * ell * t_base)
    dp = (96.0 / re) * (ell / dh) * 0.5 * _AIR["rho_kg_m3"] * u ** 2
    return {"r_th_k_w": r_th, "mass_kg": mass, "pressure_drop_pa": dp}


def heatsink_study() -> DesignStudy:
    """Thermal resistance against mass, with the pressure drop cap and the baseline's resistance as the limits."""
    return DesignStudy.create(
        kind="optimize", title="Forced-convection heatsink: thermal resistance against mass",
        statement="Find the fin array with the lowest thermal resistance and the lowest mass, without exceeding "
                  "10 Pa of pressure drop and without losing thermal resistance against the reference design.",
        base=Base(kind="geometry", ref="pp.geometries:plate_fin_heatsink",
                  variables=[Variable("fin_height_mm", 10.0, 40.0, unit="mm", description="fin height"),
                             Variable("fin_thickness_mm", 0.8, 3.0, unit="mm", description="fin thickness"),
                             Variable("fin_pitch_mm", 4.5, 8.0, unit="mm",
                                      description="centre-to-centre fin spacing")],
                  fixed={"width_m": 0.10, "length_m": 0.10, "base_thickness_m": 0.004},
                  baseline={"fin_height_mm": 25.0, "fin_thickness_mm": 1.5, "fin_pitch_mm": 6.67},
                  description="100 mm x 100 mm ducted aluminium fin array on a 4 mm base"),
        evaluator=Evaluator(tool="convection.fin_array_laminar", fidelity="model",
                            inputs={"fin_height_mm": "fin_height_mm", "fin_thickness_mm": "fin_thickness_mm",
                                    "fin_pitch_mm": "fin_pitch_mm"},
                            conditions={"width_m": 0.10, "length_m": 0.10, "speed_m_s": 1.5,
                                        "base_thickness_m": 0.004},
                            outputs=[Quantity("r_th_k_w", "K/W", label="thermal resistance"),
                                     Quantity("mass_kg", "kg", label="aluminium mass"),
                                     Quantity("pressure_drop_pa", "Pa", label="air-side pressure drop")],
                            description="laminar parallel-plate correlations with straight-fin efficiency"),
        objectives=[Objective("r_th_k_w", "min", label="thermal resistance"),
                    Objective("mass_kg", "min", label="mass")],
        constraints=[Limit("pressure_drop_pa", upper=10.0, label="pressure drop"),
                     Limit("r_th_k_w", upper_frac_of_baseline=1.0, label="thermal resistance")],
        budget=Budget(max_evaluations=38, n_doe=32, n_verify=3, surrogate_evaluations=600, seed=5),
        verifier=Verifier(surrogate_target=0.10, min_verified=1, folds=5),
        model=Model(families=("gp", "quadratic")),
        metadata={"example": "heatsink", "model": "laminar parallel-plate correlations, not a CFD solve"})


# =========================================================================== the polar (surrogate kind)
def polar_model(params: Mapping[str, float]) -> dict[str, float]:
    """An airfoil polar from thin-airfoil theory with the Prandtl-Glauert compressibility factor.

    ``cl = 2 pi alpha_rad / sqrt(1 - M^2)`` [-], ``cd = cd0 + cl^2 / (pi e AR) + 0.002 exp(20 (M - 0.78))`` [-].

    Conditions: ``cd0`` [-], ``e`` [-], ``ar`` [-]. The study's range stops at ``M = 0.8`` on purpose: the factor is
    singular at 1 and already wrong well before it, so anything above 0.8 is refused with that reason.
    """
    mach = float(params["mach"])
    if mach >= 0.8:
        return {"ok": False, "message": f"Mach {mach:.2f} is past the validity of Prandtl-Glauert (the study's range "
                                        "stops at 0.80)"}
    alpha = math.radians(float(params["alpha_deg"]))
    cl = 2 * math.pi * alpha / math.sqrt(1.0 - mach ** 2)
    cd = (float(params.get("cd0", 0.0065)) + cl ** 2 / (math.pi * float(params.get("e", 0.85)) *
                                                        float(params.get("ar", 8.0)))
          + 0.002 * math.exp(20.0 * (mach - 0.78)))
    return {"cl": cl, "cd": cd}


def polar_study() -> DesignStudy:
    """A surrogate study: measure a polar over Mach and angle of attack, fit it, measure the fit out of sample."""
    return DesignStudy.create(
        kind="surrogate", title="Airfoil polar: a fast model of cl and cd over Mach and angle of attack",
        statement="Build a surrogate of the airfoil polar over Mach 0.30-0.80 and 0 to 8 degrees, accurate enough "
                  "to choose an operating point with.",
        base=Base(kind="geometry", ref="pp.geometries:naca2412",
                  variables=[Variable("mach", 0.30, 0.79, unit="1", description="free-stream Mach number"),
                             Variable("alpha_deg", 0.0, 8.0, unit="deg", description="angle of attack")],
                  baseline={"mach": 0.60, "alpha_deg": 2.0},
                  description="NACA 2412 section, thin-airfoil polar with a compressibility factor"),
        evaluator=Evaluator(tool="aero.thin_airfoil_polar", fidelity="model",
                            inputs={"mach": "mach", "alpha_deg": "alpha_deg"},
                            conditions={"cd0": 0.0065, "e": 0.85, "ar": 8.0},
                            outputs=[Quantity("cl", "1", label="lift coefficient"),
                                     Quantity("cd", "1", label="drag coefficient")],
                            description="thin-airfoil lift with Prandtl-Glauert and a parabolic drag polar"),
        objectives=[Objective("cl", "max", label="lift coefficient"), Objective("cd", "min", label="drag coefficient")],
        constraints=[],
        budget=Budget(max_evaluations=34, n_doe=30, n_verify=0, surrogate_evaluations=200, seed=7),
        verifier=Verifier(surrogate_target=0.10, min_verified=0, folds=4),
        model=Model(families=("gp", "quadratic")),
        metadata={"example": "polar", "model": "thin-airfoil polar, not a RANS solve"})


# =========================================================================== the beam, on a numerical solver
#: Aluminium 6082-T6, and the load the beam carries: a 2 m cantilever under 2 kN/m, which is the condition the
#: study never varies.
_BEAM = {"length_m": 2.0, "load_n_per_m": 2000.0, "E_gpa": 69.0, "rho_kg_m3": 2700.0, "nx": 200}


def beam_fdm(params: Mapping[str, float]) -> dict[str, Any]:
    """A cantilever box beam solved, not approximated: the 4th-order finite-difference Euler-Bernoulli BVP.

    This is the reference the study verifies against — ``pinneapple_simulation.numerical_solvers.beam_bvp_fdm``
    solves ``E I w'''' = q`` on a 200-point grid for a uniform load (sparse direct solve), and the section
    (``SectionProps.rectangular``) is the one its stresses are computed with, so ``I``, the extreme-fibre distance
    and the deflection describe the same beam.

    Variables: section width ``b_mm`` [mm] and height ``h_mm`` [mm]. Outputs: ``mass_kg`` [kg] (2700 kg/m^3 over
    the volume), ``deflection_mm`` [mm] (the largest |w|) and ``sigma_vm_mpa`` [MPa] (the von Mises stress at the
    extreme fibre, largest over the span).
    """
    from pinneapple_simulation.numerical_solvers.beam_bvp_fdm import (
        BC_CANTILEVER,
        SectionProps,
        solve_euler_bernoulli_beam_1d,
    )

    b = float(params["b_mm"]) / 1000.0
    h = float(params["h_mm"]) / 1000.0
    ell = float(params.get("length_m", _BEAM["length_m"]))
    if b <= 0 or h <= 0:
        return {"ok": False, "message": f"a section of {b * 1000:.1f} x {h * 1000:.1f} mm has no area"}
    section = SectionProps.rectangular(b, h)
    res = solve_euler_bernoulli_beam_1d(
        L_m=ell, E_Pa=float(params.get("E_gpa", _BEAM["E_gpa"])) * 1e9, I_m4=section.I_m4,
        q_N_per_m=float(params.get("load_n_per_m", _BEAM["load_n_per_m"])), nx=int(params.get("nx", _BEAM["nx"])),
        bc_type=BC_CANTILEVER, section=section,
        yield_Pa=float(params.get("yield_mpa", 0.0)) * 1e6 if params.get("yield_mpa") else 0.0)
    return {"mass_kg": float(params.get("rho_kg_m3", _BEAM["rho_kg_m3"])) * section.A_m2 * ell,
            "deflection_mm": float(abs(res["deflection_m"]).max()) * 1000.0,
            "sigma_vm_mpa": float(res["sigma_vm_Pa"].max()) / 1e6}


def beam_study() -> DesignStudy:
    """Mass against deflection, a stress cap at 150 MPa and a deflection cap at L/80, on a solver."""
    return DesignStudy.create(
        kind="optimize", title="Cantilever box beam: mass against deflection",
        statement="Find the smallest section that keeps the tip deflection under 30 mm and the von Mises stress "
                  "under 150 MPa, for a 2 m aluminium cantilever under 2 kN/m.",
        base=Base(kind="geometry", ref="pp.geometries:cantilever_box",
                  variables=[Variable("b_mm", 30.0, 80.0, unit="mm", description="section width"),
                             Variable("h_mm", 50.0, 120.0, unit="mm", description="section height")],
                  fixed={"length_m": _BEAM["length_m"]},
                  baseline={"b_mm": 50.0, "h_mm": 80.0},
                  description="2 m aluminium cantilever, rectangular box section, uniform load"),
        evaluator=Evaluator(tool="beam.euler_bernoulli_fdm", fidelity="reference",
                            inputs={"b_mm": "b_mm", "h_mm": "h_mm"},
                            conditions={"length_m": _BEAM["length_m"], "load_n_per_m": _BEAM["load_n_per_m"],
                                        "E_gpa": _BEAM["E_gpa"], "rho_kg_m3": _BEAM["rho_kg_m3"], "nx": 200},
                            outputs=[Quantity("mass_kg", "kg", label="mass"),
                                     Quantity("deflection_mm", "mm", label="tip deflection"),
                                     Quantity("sigma_vm_mpa", "MPa", label="von Mises stress")],
                            settings={"solver": "pinneapple_simulation.numerical_solvers.beam_bvp_fdm",
                                      "stencil": "4th-order finite differences, sparse direct solve"},
                            description="Euler-Bernoulli BVP solved by FDM, not a closed-form formula"),
        objectives=[Objective("mass_kg", "min", label="mass"),
                    Objective("deflection_mm", "min", label="tip deflection")],
        constraints=[Limit("sigma_vm_mpa", upper=150.0, label="von Mises stress"),
                     Limit("deflection_mm", upper=30.0, label="tip deflection")],
        budget=Budget(max_evaluations=30, n_doe=20, n_verify=2, surrogate_evaluations=400, seed=0),
        verifier=Verifier(surrogate_target=0.10, min_verified=1, folds=5),
        model=Model(families=("gp", "quadratic")),
        metadata={"example": "beam", "model": "the FDM solver in pinneapple_simulation is the reference"})


#: The examples by name: ``(spec, evaluator)``, ready for :func:`run_study`.
EXAMPLES = {"wing": (wing_study, wing_vlm), "heatsink": (heatsink_study, heatsink_model),
            "polar": (polar_study, polar_model), "beam": (beam_study, beam_fdm)}


def example(name: str) -> tuple[DesignStudy, Any]:
    """``(spec, evaluator)`` of one example, by name (``wing``, ``heatsink``, ``polar``, ``beam``)."""
    try:
        make = EXAMPLES[name][0]
    except KeyError:
        raise ValueError(f"unknown example {name!r}; known: {sorted(EXAMPLES)}") from None
    return make(), EXAMPLES[name][1]
