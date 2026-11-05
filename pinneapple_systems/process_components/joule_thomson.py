"""Joule–Thomson cooling of natural gas across a throttling (globe) valve, verified (#1463).

A gas throttled through a valve keeps its stagnation enthalpy (adiabatic, no shaft work):
``h1 + v1²/2 = h2 + v2²/2``. Its temperature then changes by the Joule–Thomson effect,
``mu_JT = (dT/dp)_h``, which is **exactly zero for an ideal gas** (its enthalpy depends on T only). A simulation
run with an ideal-gas equation of state therefore shows no cooling at all, whatever its mesh, and that is the most
common reason a valve CFD "does not show the Joule–Thomson effect".

:func:`valve_joule_thomson_study` computes, for a stated inlet state and outlet pressure:

* the outlet temperature from four equations of state: GERG-2008 (the natural-gas reference behind ISO 20765-2,
  through :mod:`pinneapple_systems.process_components.real_gas_eos`, CoolProp HEOS), Peng–Robinson, SRK and the
  ideal gas;
* the isenthalpic path T(p), and mu_JT along it. The temperature drop is computed twice, by the flash and by the
  integral of mu_JT, and the two must agree;
* the valve's mass flow from its Cv (IEC 60534-2-1, :mod:`.control_valve`) with the real-gas density and isentropic
  exponent, whether it is choked, the pipe velocities and the kinetic-energy correction of the outlet temperature;
* the jet through the trim: its local static temperature falls with its velocity, down to the value at Mach 1
  along the isentrope from the inlet. A temperature probe inside the jet reads that, not the throttled outlet
  temperature;
* a map of the temperature drop against the outlet pressure and the inlet temperature;
* optionally, a check of the outlet temperature a CFD run gives (``cfd_T2_C``) against the GERG-2008 value, with a
  diagnosis.

The study reports checks (pass / fail / inconclusive) and a VeriPhysics decision. Every number is computed in the
run. The default composition is an illustrative natural gas, not a measured one: give the gas's analysis.

Needs CoolProp (``pip install CoolProp``; the ``potential`` extra).
"""
from __future__ import annotations

import math
import os
import re
import time
import unicodedata
from typing import Any

import numpy as np

__all__ = ["DEFAULT_GAS", "DEFAULT_CASE", "parse_valve_request", "throttle", "isenthalpic_path",
           "valve_joule_thomson_study"]

# an illustrative natural gas (mole fractions); give the gas's own analysis for a real case
DEFAULT_GAS = {"Methane": 0.90, "Ethane": 0.05, "Propane": 0.02, "Nitrogen": 0.02, "CarbonDioxide": 0.01}
# P in bar (absolute), T in degC, pipe diameter in mm; x_T is typical of a single-port globe valve with a contoured
# plug, flow to open (IEC 60534-2-1:2011, Table 2): give the valve's datasheet value for a real case
DEFAULT_CASE = {"P1_bar": 70.0, "T1_C": 25.0, "P2_bar": 20.0, "Cv": 20.0, "pipe_d_mm": 100.0, "xT": 0.72}
EOS = {"GERG-2008": "HEOS", "Peng-Robinson": "PR", "SRK": "SRK", "ideal gas": None}
_ALIASES = {"Methane": ("methane", "metano", "ch4"), "Ethane": ("ethane", "etano", "c2h6"),
            "Propane": ("propane", "propano", "c3h8"), "n-Butane": ("n-butane", "butane", "butano"),
            "IsoButane": ("isobutane", "isobutano"), "Nitrogen": ("nitrogen", "nitrogenio", "n2"),
            "CarbonDioxide": ("carbon dioxide", "dioxido de carbono", "co2"),
            "HydrogenSulfide": ("hydrogen sulfide", "sulfeto de hidrogenio", "h2s")}
BG, FG, MUTED, GRID = "#0b0f17", "#e8edf5", "#8a96a8", "#1c2433"
AMBER, TEAL, RED, VIOLET, BLUE = "#ffb547", "#2ee6c5", "#ff5d6c", "#9d8cff", "#5aa9ff"
_EOS_COL = {"GERG-2008": TEAL, "Peng-Robinson": AMBER, "SRK": VIOLET, "ideal gas": RED}


# ------------------------------------------------------------------------------------------------- the request
def _fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


_NUM = r"([-−]?\d+(?:[.,]\d+)?)"


def _num(s: str) -> float:
    return float(s.replace(",", ".").replace("−", "-"))


def _pressure_bar(v: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith("kpa"):
        return v / 100.0
    if u.startswith("mpa"):
        return v * 10.0
    if u.startswith("psi"):
        b = v * 0.0689476
        return b + 1.01325 if u.endswith("g") else b
    return v + 1.01325 if u.endswith("g") else v                        # bar / bara / barg


def _temperature_C(v: float, unit: str) -> float:
    u = unit.lower().replace("°", "").replace("º", "")
    if u.startswith("k"):
        return v - 273.15
    if u.startswith("f"):
        return (v - 32.0) * 5.0 / 9.0
    return v


def parse_valve_request(text: str) -> dict[str, Any]:
    """Read a valve case from a plain-text request (PT or EN).

    Read: inlet and outlet pressures ("inlet pressure 70 bar", "pressão de entrada 70 bar", "from 70 bar to 20
    bar"; bar, barg, kPa, MPa, psi), the inlet temperature ("25 °C", K or °F), the valve Cv, the pipe diameter
    ("DN 100", "100 mm", "4 in"), the gas composition ("methane 90 %, ethane 5 %, ...", normalised to 1), and an
    outlet temperature a CFD run gave ("CFD outlet temperature 24.6 °C"). A value not found is taken from
    :data:`DEFAULT_CASE` / :data:`DEFAULT_GAS` and listed in ``defaulted``.
    """
    t = _fold(text or "")
    case: dict[str, Any] = {}
    found = []
    P = r"\s*(bara|barg|bar|kpa|mpa|psig|psia|psi)"
    pats = {
        "P1_bar": rf"(?:\bp1\b|inlet pressure|upstream pressure|pressao (?:de )?(?:entrada|montante)|inlet)[^\d\-−]{{0,24}}?{_NUM}{P}",
        "P2_bar": rf"(?:\bp2\b|outlet pressure|downstream pressure|pressao (?:de )?(?:saida|jusante)|outlet)[^\d\-−]{{0,24}}?{_NUM}{P}",
    }
    for k, rx in pats.items():
        m = re.search(rx, t)
        if m:
            case[k] = _pressure_bar(_num(m.group(1)), m.group(2))
            found.append(k)
    if "P1_bar" not in case or "P2_bar" not in case:
        m = re.search(rf"(?:from|de)\s*{_NUM}{P}\s*(?:to|para|a|ate)\s*{_NUM}{P}", t)
        if m:
            case.setdefault("P1_bar", _pressure_bar(_num(m.group(1)), m.group(2)))
            case.setdefault("P2_bar", _pressure_bar(_num(m.group(3)), m.group(4)))
            found += [k for k in ("P1_bar", "P2_bar") if k not in found]
    m = re.search(rf"(?:cfd|simulation|simulacao)[^\d\-−]{{0,40}}?temperat\w*[^\d\-−]{{0,16}}?{_NUM}\s*(°\s*c|ºc|c\b|k\b|°\s*f)",
                  t)
    if m:
        case["cfd_T2_C"] = _temperature_C(_num(m.group(1)), m.group(2))
        found.append("cfd_T2_C")
        t_wo = t[:m.start()] + t[m.end():]
    else:
        t_wo = t
    m = re.search(rf"(?:\bt1\b|inlet temperature|upstream temperature|temperatura (?:de )?(?:entrada|montante)|"
                  rf"temperature|temperatura)[^\d\-−]{{0,24}}?{_NUM}\s*(°\s*c|ºc|c\b|k\b|°\s*f)", t_wo)
    if m:
        case["T1_C"] = _temperature_C(_num(m.group(1)), m.group(2))
        found.append("T1_C")
    m = re.search(rf"\bcv\b[^\d]{{0,12}}?{_NUM}", t)
    if m:
        case["Cv"] = _num(m.group(1))
        found.append("Cv")
    m = re.search(rf"\bdn\s*{_NUM}", t) or re.search(
        rf"(?:pipe diameter|diameter|diametro|pipe)[^\d]{{0,16}}?{_NUM}\s*(mm|in|\"|polegadas?)", t)
    if m:
        v = _num(m.group(1))
        unit = m.group(2) if m.lastindex and m.lastindex >= 2 else "mm"
        case["pipe_d_mm"] = v * 25.4 if unit and unit[0] in ('i', '"', 'p') else v
        found.append("pipe_d_mm")
    gas: dict[str, float] = {}
    for name, keys in _ALIASES.items():
        for a in keys:
            m = re.search(rf"\b{re.escape(a)}\b[^\d%]{{0,8}}?{_NUM}\s*%", t)
            if m:
                gas[name] = _num(m.group(1)) / 100.0
                break
    if gas:
        s = sum(gas.values())
        gas = {k: v / s for k, v in gas.items()}
        found.append("composition")
    defaulted = [k for k in ("P1_bar", "T1_C", "P2_bar", "Cv", "pipe_d_mm") if k not in case]
    for k in defaulted:
        case[k] = DEFAULT_CASE[k]
    case.setdefault("xT", DEFAULT_CASE["xT"])
    if not gas:
        defaulted.append("composition")
    return {"case": case, "gas": gas or dict(DEFAULT_GAS), "found": found, "defaulted": defaulted}


# ------------------------------------------------------------------------------------------------- thermodynamics
def _state(backend: str, gas: dict[str, float]):
    import CoolProp.CoolProp as CP
    from CoolProp.CoolProp import AbstractState

    a = AbstractState(backend, "&".join(gas))
    a.set_mole_fractions(list(gas.values()))
    a.specify_phase(CP.iphase_gas)
    return a


def _PT(a, P: float, T: float) -> dict[str, float]:
    import CoolProp.CoolProp as CP
    a.update(CP.PT_INPUTS, P, T)
    return {"P": P, "T": a.T(), "h": a.hmass(), "s": a.smass(), "rho": a.rhomass(), "cp": a.cpmass(),
            "k": a.cpmass() / a.cvmass(), "Z": a.compressibility_factor(), "c": a.speed_sound(),
            "mu_JT": a.first_partial_deriv(CP.iT, CP.iP, CP.iHmass)}


def _flash(a, pair, v1: float, v2: float) -> None:
    """``a.update(pair, v1, v2)`` with the gas phase imposed; when that flash fails (it does for some pure fluids),
    again with CoolProp's own phase determination, then the gas phase is imposed back."""
    import CoolProp.CoolProp as CP
    try:
        a.update(pair, v1, v2)
    except ValueError:
        a.unspecify_phase()
        try:
            a.update(pair, v1, v2)
        finally:
            a.specify_phase(CP.iphase_gas)


def _Ph(a, P: float, h: float) -> dict[str, float]:
    import CoolProp.CoolProp as CP
    _flash(a, CP.HmassP_INPUTS, h, P)
    return _PT(a, P, a.T())


def _Ps(a, P: float, s: float) -> dict[str, float]:
    import CoolProp.CoolProp as CP
    _flash(a, CP.PSmass_INPUTS, P, s)
    return _PT(a, P, a.T())


def throttle(gas: dict[str, float], P1_Pa: float, T1_K: float, P2_Pa: float, eos: str = "GERG-2008",
             v1_m_s: float = 0.0, area_m2: float | None = None, mass_flow_kg_s: float | None = None
             ) -> dict[str, Any]:
    """Outlet state of an adiabatic throttling from (P1, T1) to P2 with the named equation of state.

    Without a flow (``mass_flow_kg_s`` None) the enthalpy is conserved (h2 = h1). With a mass flow and the pipe
    ``area_m2`` the stagnation enthalpy is (h2 + v2²/2 = h1 + v1²/2, v = m / (rho A)), solved by fixed-point
    iteration on the outlet density. The ideal gas uses the GERG-2008 ideal-gas heat capacity at the inlet (its
    enthalpy depends on T only, so mu_JT = 0)."""
    if eos == "ideal gas":
        a = _state("HEOS", gas)
        s1 = _PT(a, P1_Pa, T1_K)
        cp0, M = a.cp0mass(), a.molar_mass()
        R = 8.314462618 / M
        rho1 = P1_Pa / (R * T1_K)
        T2, v1, v2 = T1_K, 0.0, 0.0
        if mass_flow_kg_s is not None and area_m2:
            v1 = mass_flow_kg_s / (rho1 * area_m2)
            for _ in range(30):
                v2 = mass_flow_kg_s / (P2_Pa / (R * T2) * area_m2)
                T2 = T1_K + (v1 ** 2 - v2 ** 2) / (2 * cp0)
        return {"eos": eos, "T2_K": T2, "dT_K": T2 - T1_K, "v1_m_s": v1, "v2_m_s": v2, "rho1": rho1,
                "rho2": P2_Pa / (R * T2), "mu_JT_inlet_K_per_bar": 0.0, "Z1": 1.0, "k1": cp0 / (cp0 - R),
                "h_residual_J_kg": 0.0}
    a = _state(EOS[eos], gas)
    s1 = _PT(a, P1_Pa, T1_K)
    v1, v2 = 0.0, 0.0
    s2 = _Ph(a, P2_Pa, s1["h"])
    if mass_flow_kg_s is not None and area_m2:
        v1 = mass_flow_kg_s / (s1["rho"] * area_m2)
        for _ in range(30):
            v2 = mass_flow_kg_s / (s2["rho"] * area_m2)
            s2 = _Ph(a, P2_Pa, s1["h"] + 0.5 * v1 ** 2 - 0.5 * v2 ** 2)
    res = (s2["h"] + 0.5 * v2 ** 2) - (s1["h"] + 0.5 * v1 ** 2)
    return {"eos": eos, "T2_K": s2["T"], "dT_K": s2["T"] - T1_K, "v1_m_s": v1, "v2_m_s": v2, "rho1": s1["rho"],
            "rho2": s2["rho"], "mu_JT_inlet_K_per_bar": s1["mu_JT"] * 1e5, "Z1": s1["Z"], "k1": s1["k"],
            "h_residual_J_kg": res}


def isenthalpic_path(gas: dict[str, float], P1_Pa: float, T1_K: float, P2_Pa: float, n: int = 41,
                     eos: str = "GERG-2008") -> dict[str, np.ndarray]:
    """T(p) at the inlet enthalpy from P1 to P2, and mu_JT along it [K/Pa]."""
    a = _state(EOS[eos], gas)
    h1 = _PT(a, P1_Pa, T1_K)["h"]
    P = np.linspace(P1_Pa, P2_Pa, n)
    T, mu = np.zeros(n), np.zeros(n)
    for i, p in enumerate(P):
        st = _Ph(a, p, h1)
        T[i], mu[i] = st["T"], st["mu_JT"]
    return {"P": P, "T": T, "mu_JT": mu}


# ------------------------------------------------------------------------------------------------- the study
def valve_joule_thomson_study(request: str = "", out_dir: str = "valve_joule_thomson", P1_bar: float | None = None,
                              T1_C: float | None = None, P2_bar: float | None = None, Cv: float | None = None,
                              pipe_d_mm: float | None = None, xT: float | None = None,
                              cfd_T2_C: float | None = None, composition: dict[str, float] | None = None,
                              width_px: int = 1280) -> dict[str, Any]:
    """Joule–Thomson temperature drop of natural gas across a globe valve, from a request in words or from values.

    Parameters
    ----------
    request : str
        The case in words (PT/EN, :func:`parse_valve_request`); explicit arguments below override what it reads.
    out_dir : str
        Folder for the figures.
    P1_bar, P2_bar : float [bar]
        Inlet and outlet pressures (absolute).
    T1_C : float [degC]
        Inlet temperature.
    Cv : float [US gpm/psi^0.5]
        Valve flow coefficient at the operating opening.
    pipe_d_mm : float [mm]
        Pipe inner diameter, upstream and downstream (for the velocities and the kinetic correction).
    xT : float [-]
        Pressure-differential ratio factor of the valve (datasheet; the default is typical of a single-port globe
        valve, IEC 60534-2-1 Table 2).
    cfd_T2_C : float [degC], optional
        The mixed-out outlet temperature a CFD run gave, checked against the GERG-2008 value.
    composition : dict, optional
        Mole fractions by CoolProp name (Methane, Ethane, ...); default :data:`DEFAULT_GAS` (illustrative).
    width_px : int [px]
        Width of the figures.

    Returns
    -------
    dict
        ``case`` (what was used, and what was read or defaulted), ``outputs``, ``by_eos``, ``checks``,
        ``decision`` (VeriPhysics), ``files``, ``assumptions``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_veriphysics.policy import decide, load_policy

    from .control_valve import compressible_mass_flow

    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    parsed = parse_valve_request(request)
    case = dict(parsed["case"])
    for k, v in (("P1_bar", P1_bar), ("T1_C", T1_C), ("P2_bar", P2_bar), ("Cv", Cv), ("pipe_d_mm", pipe_d_mm),
                 ("xT", xT), ("cfd_T2_C", cfd_T2_C)):
        if v is not None:
            case[k] = float(v)
            if k in parsed["defaulted"]:
                parsed["defaulted"].remove(k)
    gas = dict(composition) if composition else parsed["gas"]
    if composition and "composition" in parsed["defaulted"]:
        parsed["defaulted"].remove("composition")
    P1, P2, T1 = case["P1_bar"] * 1e5, case["P2_bar"] * 1e5, case["T1_C"] + 273.15
    if not P2 < P1:
        raise ValueError(f"the outlet pressure ({case['P2_bar']:g} bar) must be below the inlet pressure "
                         f"({case['P1_bar']:g} bar)")
    A = math.pi * (case["pipe_d_mm"] / 1000.0) ** 2 / 4.0

    # 1. the valve: mass flow from Cv with the real-gas inlet state --------------------------------------------
    a = _state("HEOS", gas)
    s1 = _PT(a, P1, T1)
    flow = compressible_mass_flow(case["Cv"], P1, P2, s1["rho"], s1["k"], case["xT"])
    w = flow.mass_flow_kg_s

    # 2. outlet temperature by equation of state (stagnation enthalpy with the pipe velocities) ---------------
    by_eos = {name: throttle(gas, P1, T1, P2, name, area_m2=A, mass_flow_kg_s=w) for name in EOS}
    static = {name: throttle(gas, P1, T1, P2, name) for name in EOS}          # h2 = h1, no kinetic term
    g = by_eos["GERG-2008"]

    # 3. isenthalpic path and the mu_JT integral ---------------------------------------------------------------
    path = isenthalpic_path(gas, P1, T1, P2)
    dT_integral = float(np.trapezoid(path["mu_JT"], path["P"])) if hasattr(np, "trapezoid") else float(
        np.trapz(path["mu_JT"], path["P"]))
    dT_flash = static["GERG-2008"]["dT_K"]

    # 4. the jet: the gas accelerates through the trim before it mixes out downstream. Its local static temperature
    # falls with the velocity (T0 - v^2/2cp in the ideal limit); along the isentrope from the inlet it is coldest
    # where the jet reaches Mach 1. A probe inside the jet reads down to that, not the throttled outlet temperature
    h0 = s1["h"] + 0.5 * g["v1_m_s"] ** 2

    def mach(p):
        st = _Ps(a, p, s1["s"])
        return math.sqrt(max(2.0 * (h0 - st["h"]), 0.0)) / st["c"], st

    lo, hi = 0.05 * P1, P1
    for _ in range(60):                                     # bisection for M = 1 (M falls as p rises)
        mid = 0.5 * (lo + hi)
        if mach(mid)[0] > 1.0:
            lo = mid
        else:
            hi = mid
    P_jet = hi
    mach_jet, s_jet = mach(P_jet)

    # 5. map of the drop: outlet pressure x inlet temperature (GERG-2008, h2 = h1) ----------------------------
    P2_grid = np.linspace(0.95 * P1, max(0.1 * P1, 1.5e5), 24)
    T1_grid = np.array([T1 - 20.0, T1, T1 + 20.0])
    dT_map = np.array([[throttle(gas, P1, t1, p2)["dT_K"] for p2 in P2_grid] for t1 in T1_grid])

    # 6. checks ---------------------------------------------------------------------------------------------
    pol = load_policy(None)
    checks: list[dict[str, Any]] = []

    def chk(code, status, detail, value=None):
        checks.append({"code": code, "status": status, "detail": detail, "value": value})

    hres = abs(g["h_residual_J_kg"])
    chk("thermo.energy_conservation", "pass" if hres <= 1.0 else "fail",
        f"stagnation enthalpy of the GERG-2008 outlet state minus the inlet's: {hres:.3g} J/kg (allowed 1 J/kg; "
        f"inlet enthalpy scale {abs(s1['h']):.3g} J/kg)", hres)
    rel = abs(dT_integral - dT_flash) / max(abs(dT_flash), 1e-9)
    chk("thermo.jt_integral_vs_flash", "pass" if rel <= 0.01 else "fail",
        f"temperature drop by the isenthalpic flash {dT_flash:.3f} K and by the integral of mu_JT along the path "
        f"{dT_integral:.3f} K (41 points): {100 * rel:.2f} % apart (allowed 1 %)", rel)
    ig = static["ideal gas"]["dT_K"]
    chk("thermo.ideal_gas_limit", "pass" if abs(ig) < 1e-9 else "fail",
        f"ideal gas, h2 = h1: drop {ig:.2e} K (mu_JT = 0 exactly; an ideal-gas CFD cannot show the effect)", ig)
    try:
        import CoolProp.CoolProp as CP
        from CoolProp.CoolProp import AbstractState

        ph = AbstractState("HEOS", "&".join(gas))
        ph.set_mole_fractions(list(gas.values()))
        ph.update(CP.PT_INPUTS, P2, g["T2_K"])
        phase = ph.phase()
        single = phase in (CP.iphase_gas, CP.iphase_supercritical_gas, CP.iphase_supercritical)
        chk("thermo.single_phase_outlet", "pass" if single else "fail",
            f"GERG-2008 phase at the outlet state ({case['P2_bar']:g} bar, {g['T2_K'] - 273.15:.1f} °C): "
            + ("gas" if single else f"not a single gas phase (CoolProp phase {phase}): liquid may drop out, the "
                                    "single-phase results do not hold"), float(phase))
    except Exception as e:  # noqa: BLE001 - mixture phase detection can fail; say so rather than guess
        chk("thermo.single_phase_outlet", "inconclusive",
            f"the mixture's phase at the outlet could not be computed ({type(e).__name__}): check the dew point "
            "against the outlet temperature separately")
    chk("valve.choked", "pass" if not flow.choked else "fail",
        f"pressure-drop ratio x = {flow.x:.3f}, choked above Fk·xT = {s1['k'] / 1.4 * case['xT']:.3f}: "
        + ("not choked" if not flow.choked else "CHOKED: the flow no longer rises with the drop; the mass flow "
                                                "and velocities are at the choked limit"), flow.x)
    spread = max(abs(by_eos[n]["dT_K"] - g["dT_K"]) for n in ("Peng-Robinson", "SRK"))
    chk("thermo.cubic_eos_spread", "pass" if spread <= 0.15 * abs(g["dT_K"]) else "fail",
        f"Peng–Robinson and SRK differ from GERG-2008 by up to {spread:.2f} K on a {abs(g['dT_K']):.2f} K drop "
        "(allowed 15 % of it; a cubic EOS is often what a CFD code offers)", spread)
    if "cfd_T2_C" in case:
        cfd_dT = case["cfd_T2_C"] - case["T1_C"]
        err = case["cfd_T2_C"] - (g["T2_K"] - 273.15)
        tol = max(1.0, 0.1 * abs(g["dT_K"]))
        if abs(cfd_dT) < 0.2 * abs(g["dT_K"]):
            why = ("the CFD shows almost no cooling: the expected sign of an ideal-gas equation of state (mu_JT = 0)"
                   " or of a disabled energy equation; switch the CFD to a real-gas EOS (Peng–Robinson at least)")
        elif cfd_dT < g["dT_K"] - tol:
            why = (f"the CFD is colder than throttling allows: check that the probe is far downstream, where the "
                   f"jet has mixed out (inside the jet the static temperature can fall to "
                   f"{s_jet['T'] - 273.15:.1f} °C because the gas is fast there), and that it reports a "
                   "mass-weighted static temperature")
        else:
            why = "within tolerance of the real-gas value"
        chk("cfd.outlet_temperature", "pass" if abs(err) <= tol else "fail",
            f"CFD outlet {case['cfd_T2_C']:.2f} °C against the GERG-2008 isenthalpic {g['T2_K'] - 273.15:.2f} °C: "
            f"{err:+.2f} K (allowed ±{tol:.2f} K). {why}", err)
    decision, reasons = decide(checks, pol)

    # 7. figures --------------------------------------------------------------------------------------------
    W = width_px / 100
    plt.rcParams.update({"axes.facecolor": BG, "figure.facecolor": BG, "text.color": FG, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    files = []
    fig, ax = plt.subplots(figsize=(W * 0.55, W * 0.36))
    hs = np.linspace(-25, 25, 6)
    for dT0 in hs:                                            # isenthalps through (P1, T1 + dT0), GERG-2008
        try:
            pth = isenthalpic_path(gas, P1, T1 + dT0, P2_grid[-1], n=25)
            ax.plot(pth["P"] / 1e5, pth["T"] - 273.15, color=GRID, lw=1.2)
        except Exception:  # noqa: BLE001 - an isenthalp leaving the gas region is simply not drawn
            pass
    for name, col in _EOS_COL.items():
        if name == "ideal gas":
            ax.plot([case["P1_bar"], case["P2_bar"]], [case["T1_C"]] * 2, color=col, lw=2, ls="--", label=name)
            continue
        pth = path if name == "GERG-2008" else isenthalpic_path(gas, P1, T1, P2, n=25, eos=name)
        ax.plot(pth["P"] / 1e5, pth["T"] - 273.15, color=col, lw=2.6 if name == "GERG-2008" else 1.8, label=name)
    ax.scatter([case["P1_bar"]], [case["T1_C"]], color=FG, zorder=5, s=40)
    ax.scatter([P_jet / 1e5], [s_jet["T"] - 273.15], color=BLUE, zorder=5, s=50, marker="v",
               label="coldest point inside the jet (local, Mach 1)")
    if "cfd_T2_C" in case:
        ax.scatter([case["P2_bar"]], [case["cfd_T2_C"]], color=RED, marker="X", s=120, zorder=6, label="your CFD")
    ax.invert_xaxis()
    ax.set_xlabel("pressure (bar)")
    ax.set_ylabel("temperature (°C)")
    ax.set_title("Throttling path: temperature against pressure (grey: isenthalps, GERG-2008)", color=FG)
    ax.legend(frameon=False, labelcolor=FG, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throttling_path.png"), dpi=100)
    plt.close(fig)
    files.append("throttling_path.png")

    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.33))
    names = list(EOS)
    vals = [static[n]["dT_K"] for n in names]                  # the Joule-Thomson effect alone (h2 = h1)
    bars = axs[0].bar(names, vals, color=[_EOS_COL[n] for n in names], zorder=3)
    lo_y = min(vals) * 1.25
    for b, v in zip(bars, vals, strict=True):
        if abs(v) > 0.05 * abs(lo_y):
            axs[0].text(b.get_x() + b.get_width() / 2, v / 2, f"{v:.2f} K", ha="center", va="center", color=BG,
                        fontsize=12, weight="bold", zorder=4)
        else:
            axs[0].text(b.get_x() + b.get_width() / 2, 0.04 * abs(lo_y), f"{v:.2f} K\n(μ_JT = 0)", ha="center",
                        va="bottom", color=FG, fontsize=11, zorder=4)
    axs[0].set_ylim(lo_y, 0.22 * abs(lo_y))
    axs[0].axhline(0, color=MUTED, lw=1, zorder=2)
    axs[0].set_ylabel("outlet − inlet temperature (K)")
    axs[0].set_title("Joule–Thomson change by equation of state (h2 = h1)", color=FG)
    axs[0].text(0.01, 0.02, f"kinetic correction at the pipe velocities (GERG-2008): "
                            f"{g['dT_K'] - static['GERG-2008']['dT_K']:+.2f} K", transform=axs[0].transAxes,
                color=MUTED, fontsize=10)
    axs[1].plot(path["P"] / 1e5, path["mu_JT"] * 1e5, color=TEAL, lw=2.4)
    axs[1].invert_xaxis()
    axs[1].set_xlabel("pressure along the path (bar)")
    axs[1].set_ylabel("μ_JT (K/bar)")
    axs[1].set_title("Joule–Thomson coefficient along the path (GERG-2008)", color=FG)
    axs[0].grid(color=GRID, lw=0.6, axis="y", zorder=0)
    axs[1].grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eos_comparison.png"), dpi=100)
    plt.close(fig)
    files.append("eos_comparison.png")

    fig, ax = plt.subplots(figsize=(W * 0.55, W * 0.33))
    for t1, row, col in zip(T1_grid, dT_map, (BLUE, TEAL, AMBER), strict=True):
        ax.plot(P2_grid / 1e5, row, color=col, lw=2.2, label=f"inlet {t1 - 273.15:.0f} °C")
    ax.axvline(case["P2_bar"], color=MUTED, ls=":", lw=1.2)
    ax.set_xlabel("outlet pressure (bar)")
    ax.set_ylabel("temperature change (K)")
    ax.set_title(f"Temperature drop from {case['P1_bar']:g} bar (GERG-2008, h2 = h1)", color=FG)
    ax.legend(frameon=False, labelcolor=FG)
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temperature_drop_map.png"), dpi=100)
    plt.close(fig)
    files.append("temperature_drop_map.png")

    fig, ax = plt.subplots(figsize=(W * 0.5, W * 0.3))
    ax.set_axis_off()
    col = {"APPROVED": TEAL, "REVIEW": AMBER, "REJECT": RED}[decision]
    ax.text(0.0, 0.92, "VeriPhysics grade", color=MUTED, fontsize=15, transform=ax.transAxes)
    ax.text(0.0, 0.70, decision, color=col, fontsize=34, weight="bold", transform=ax.transAxes)
    for k, c in enumerate(checks):
        mark, cc = {"pass": ("✔", TEAL), "fail": ("✘", RED)}.get(c["status"], ("?", AMBER))
        ax.text(0.0, 0.52 - 0.075 * k, f"{mark}  {c['code']}", color=cc, fontsize=12, transform=ax.transAxes,
                family="monospace")
    fig.savefig(os.path.join(out_dir, "veriphysics_grade.png"), dpi=100, facecolor=BG)
    plt.close(fig)
    files.append("veriphysics_grade.png")

    outputs = {
        "T2_C": g["T2_K"] - 273.15, "dT_K": g["dT_K"], "dT_isenthalpic_K": dT_flash, "dT_mu_integral_K": dT_integral,
        "mu_JT_inlet_K_per_bar": g["mu_JT_inlet_K_per_bar"], "Z_inlet": g["Z1"], "k_inlet": g["k1"],
        "mass_flow_kg_s": w, "choked": bool(flow.choked), "pressure_drop_ratio": flow.x,
        "v_inlet_m_s": g["v1_m_s"], "v_outlet_m_s": g["v2_m_s"],
        "T_jet_coldest_C": s_jet["T"] - 273.15, "P_jet_sonic_bar": P_jet / 1e5, "mach_jet": mach_jet,
        "kinetic_correction_K": g["dT_K"] - dT_flash,
        **{f"dT_isenthalpic_K[{n}]": static[n]["dT_K"] for n in EOS},
        **{f"dT_K[{n}]": by_eos[n]["dT_K"] for n in EOS},
        "seconds": time.perf_counter() - t0,
    }
    if "cfd_T2_C" in case:
        outputs["cfd_error_K"] = case["cfd_T2_C"] - (g["T2_K"] - 273.15)
    return {"case": {"values": case, "composition": gas, "found": parsed["found"],
                     "defaulted": parsed["defaulted"]},
            "outputs": outputs, "by_eos": by_eos, "checks": checks,
            "decision": {"verdict": decision, "reasons": reasons, "policy": pol.name}, "files": files,
            "assumptions": [
                "adiabatic valve with no shaft work: stagnation enthalpy conserved from the inlet to the mixed-out "
                "outlet; the same pipe diameter upstream and downstream",
                "single gas phase (CoolProp phase forced to gas in the flashes; the outlet phase is checked "
                "separately)",
                "valve flow by IEC 60534-2-1 with the given Cv, F_L and x_T (no piping geometry factor)",
                "coldest point inside the jet: the static state at Mach 1 along the isentrope from the inlet, a "
                "bound on what a probe inside the jet can read, not a resolved flow field",
                "composition: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in gas.items())
                + (" (illustrative default)" if "composition" in parsed["defaulted"] else "")]}
