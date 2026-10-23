"""Structures: bending strength, deflection, lateral-torsional buckling and pedestrian-induced resonance of steel
girders; flexural buckling of a column; fatigue (Basquin); wind uplift of a roof covering.

Girders are doubly symmetric I-sections (``h``, ``b``, ``t_f``, ``t_w``), simply supported over ``span``, sharing the
deck load equally (``n_girders``). Section properties are computed from the plates (thin-walled formulas for ``J`` and
``I_w``). Steel constants from EN 1993-1-1, 3.2.6; partial factors EN 1990 Table A1.2(B) (1.35 / 1.5) and
EN 1993-1-1 6.1 (gamma_M0 = gamma_M1 = 1.0); footbridge live load from EN 1991-2, 5.3.2.1.
"""
from __future__ import annotations

import math

import numpy as np

from ..chain import tr
from ..library import (
    Countermeasure,
    FailureMode,
    Need,
    register_countermeasure,
    register_mode,
    validity,
)
from ..props import SRC, STEEL, G
from ._util import fmt, q, self_check, size_to_pass, verdict

EN1990 = "EN 1990:2002, Table A1.2(B): gamma_G = 1.35, gamma_Q = 1.5; Annex A2, A2.4.3.2: vertical acceleration " \
         "of footbridge decks <= 0.7 m/s^2 (recommended)"
EN1991_2 = "EN 1991-2:2003, 5.3.2.1: footbridge crowd load q_fk = 2.0 + 120/(L + 30) kN/m^2, 2.5 <= q_fk <= 5.0"
AASHTO_PED = ("AASHTO LRFD Guide Specifications for the Design of Pedestrian Bridges (2009), sec. 5: deflection "
              "under pedestrian live load <= span/360")
TIMOSHENKO = "Timoshenko & Gere, Theory of Elastic Stability, 2nd ed., 1961, ch. 6 (lateral buckling of beams)"
BACHMANN = ("Bachmann & Ammann (1987), Vibrations in Structures Induced by Man and Machines, IABSE: walking pace "
            "1.6-2.4 Hz, body weight 700 N, dynamic load factors 0.4 / 0.1 / 0.1 for harmonics 1-3")
DEN_HARTOG = "Den Hartog, Mechanical Vibrations, 4th ed., 1956, sec. 3.3 (optimum vibration absorber)"
FDM_BEAM = "pinneapple_simulation.numerical_solvers.beam_bvp_fdm.solve_euler_bernoulli_beam_1d (4th-order FD)"


def i_section(h: float, b: float, tf: float, tw: float) -> dict:
    hw = h - 2 * tf
    A = 2 * b * tf + hw * tw
    Iy = (b * h ** 3 - (b - tw) * hw ** 3) / 12
    Iz = 2 * tf * b ** 3 / 12 + hw * tw ** 3 / 12
    J = (2 * b * tf ** 3 + (h - tf) * tw ** 3) / 3
    Iw = Iz * (h - tf) ** 2 / 4
    return {"A": A, "Iy": Iy, "Iz": Iz, "J": J, "Iw": Iw, "Wy": 2 * Iy / h, "hw": hw}


def _sec(p):
    return i_section(p["girder_h"], p["girder_b"], p["girder_tf"], p["girder_tw"])


def crowd_load(span: float) -> float:
    """EN 1991-2 5.3.2.1 footbridge uniformly distributed load, Pa."""
    return 1e3 * min(5.0, max(2.5, 2.0 + 120.0 / (span + 30.0)))


def girder_loads(p: dict) -> dict:
    s = _sec(p)
    n = int(p.get("n_girders", 2))
    w = float(p["deck_width"])
    steel = s["A"] * STEEL["rho"] * G
    gk = steel + float(p["deck_dead_load"]) * w / n
    qfk = float(p.get("live_load", crowd_load(p["span"])))
    qk = qfk * w / n
    L = float(p["span"])
    wed = 1.35 * gk + 1.5 * qk
    return {"gk": gk, "qk": qk, "qfk": qfk, "wed": wed, "MEd": wed * L ** 2 / 8, "sec": s, "n": n,
            "mass_per_m": (n * steel + float(p["deck_dead_load"]) * w) / G}


def _fy(p):
    return float(p.get("fy", 355e6))


def _bending(p, limits):
    ld = girder_loads(p)
    MRd = ld["sec"]["Wy"] * _fy(p)
    ok = ld["MEd"] <= MRd
    mech = tr(f"Momento de cálculo por viga {fmt(ld['MEd'] / 1e3)} kN·m (1,35·{fmt(ld['gk'] / 1e3)} + "
              f"1,5·{fmt(ld['qk'] / 1e3)} kN/m em {fmt(p['span'])} m) contra o momento resistente elástico "
              f"{fmt(MRd / 1e3)} kN·m: " + ("a seção resiste." if ok else "a seção escoa."),
              f"Design moment per girder {fmt(ld['MEd'] / 1e3)} kN m (1.35 x {fmt(ld['gk'] / 1e3)} + "
              f"1.5 x {fmt(ld['qk'] / 1e3)} kN/m over {fmt(p['span'])} m) against the elastic moment resistance "
              f"{fmt(MRd / 1e3)} kN m: " + ("the section resists." if ok else "the section yields."))
    return verdict(demand=ld["MEd"] / 1e3, capacity=MRd / 1e3, unit="kN m",
                   demand_label=tr("momento de cálculo", "design moment"),
                   capacity_label=tr("momento resistente elástico", "elastic moment resistance"),
                   mechanism=mech, method="simply supported girder, M = w L^2/8, M_Rd = W_el f_y / gamma_M0",
                   quantities={"M_Ed": q(ld["MEd"], "N m", "design moment per girder"),
                               "M_Rd": q(MRd, "N m", "elastic moment resistance"),
                               "q_fk": q(ld["qfk"], "Pa", "footbridge crowd load (EN 1991-2)"),
                               "W_el": q(ld["sec"]["Wy"], "m^3", "elastic section modulus")},
                   assumptions=[tr("seção classe 3 ou melhor (resistência elástica)", "class 3 section or better "
                                   "(elastic resistance)")],
                   sources=[EN1990, EN1991_2, SRC["ec3"]])


register_mode(FailureMode(
    "bending_strength", "structures",
    tr("resistência à flexão", "bending strength"),
    tr("momento de cálculo contra o momento resistente da seção", "design moment against the section resistance"),
    _bending,
    needs=(Need("span", "m", "span"), Need("deck_width", "m", "deck width"),
           Need("deck_dead_load", "Pa", "deck dead load"), Need("girder_h", "m", "girder depth"),
           Need("girder_b", "m", "flange width"), Need("girder_tf", "m", "flange thickness"),
           Need("girder_tw", "m", "web thickness")),
    sources=(EN1990, EN1991_2, SRC["ec3"]), countermeasures=("deepen_girders",),
    uncertain={"deck_dead_load": 0.1, "fy": 0.05}))


def _deflection(p, limits):
    ld = girder_loads(p)
    L, E, Iy = float(p["span"]), STEEL["E"], ld["sec"]["Iy"]
    d = 5 * ld["qk"] * L ** 4 / (384 * E * Iy)
    ratio = float(limits.get("deflection_ratio", p.get("deflection_ratio", 360.0)))
    lim = L / ratio
    checks = []
    try:
        from pinneapple_simulation.numerical_solvers.beam_bvp_fdm import (
            solve_euler_bernoulli_beam_1d,
        )
        fd = solve_euler_bernoulli_beam_1d(L_m=L, E_Pa=E, I_m4=Iy, q_N_per_m=ld["qk"], nx=401)
        dfd = float(np.max(np.abs(fd["deflection_m"])))
        rel = abs(dfd - d) / d
        checks.append(self_check("solver vs closed form: FD beam deflection", rel < 0.01, rel, 0.01, FDM_BEAM))
        reac = abs(fd["shear_N"][0]) + abs(fd["shear_N"][-1])
        eq = abs(reac - ld["qk"] * L) / (ld["qk"] * L)
        checks.append(self_check("equilibrium: FD support reactions = applied load", eq < 0.01, eq, 0.01, FDM_BEAM))
    except ImportError as exc:                       # the solver module needs torch
        checks.append(self_check("solver vs closed form: FD beam deflection", True, None, None, FDM_BEAM,
                                 f"not run: {exc}"))
    ok = d <= lim
    mech = tr(f"Sob a multidão ({fmt(ld['qfk'] / 1e3)} kN/m²) cada viga flecha {fmt(d * 1e3)} mm, "
              + (f"mais que L/{fmt(ratio)} = {fmt(lim * 1e3)} mm: a passarela resiste, mas balança e assusta quem passa"
                 " — a rigidez (E·I), não a resistência, governa." if not ok else
                 f"dentro de L/{fmt(ratio)} = {fmt(lim * 1e3)} mm."),
              f"Under the crowd ({fmt(ld['qfk'] / 1e3)} kN/m^2) each girder deflects {fmt(d * 1e3)} mm, "
              + (f"more than L/{fmt(ratio)} = {fmt(lim * 1e3)} mm: the bridge is strong enough but sags visibly — "
                 "stiffness (E I), not strength, governs." if not ok else
                 f"within L/{fmt(ratio)} = {fmt(lim * 1e3)} mm."))
    return verdict(demand=d * 1e3, capacity=lim * 1e3, unit="mm",
                   demand_label=tr("flecha sob carga de pedestres", "deflection under pedestrian load"),
                   capacity_label=tr(f"limite L/{fmt(ratio)}", f"limit L/{fmt(ratio)}"),
                   mechanism=mech, method="Euler-Bernoulli 5 q L^4 / (384 E I), checked by a finite-difference "
                                         "solve of E I w'''' = q",
                   quantities={"deflection": q(d, "m", "mid-span deflection (live load)"),
                               "I_y": q(Iy, "m^4", "second moment of area per girder"),
                               "q_k": q(ld["qk"], "N/m", "live load per girder")},
                   sources=[AASHTO_PED, EN1991_2, SRC["ec3"], FDM_BEAM], self_checks=checks)


register_mode(FailureMode(
    "deflection", "structures", tr("flecha excessiva", "excessive deflection"),
    tr("flecha sob carga de uso acima do limite de serviço", "service deflection above the limit"), _deflection,
    needs=(Need("span", "m", "span"), Need("girder_h", "m", "girder depth")),
    sources=(AASHTO_PED, EN1991_2), countermeasures=("deepen_girders",),
    uncertain={"live_load": 0.1},
    known_answers=lambda: [{
        "params": {"span": 10.0, "deck_width": 2.0, "deck_dead_load": 0.0, "n_girders": 2, "live_load": 1000.0,
                   "girder_h": 0.3, "girder_b": 0.15, "girder_tf": 0.01, "girder_tw": 0.007},
        "quantity": "deflection",
        "expected": 5 * 1000.0 * 10 ** 4 / (384 * 210e9 * i_section(0.3, 0.15, 0.01, 0.007)["Iy"]), "rtol": 1e-12,
        "source": "5 q L^4/(384 E I) (Timoshenko, Strength of Materials)"}]))


def chi(lam: float, alpha: float) -> float:
    """EN 1993-1-1 6.3.1.2 / 6.3.2.2 reduction factor for slenderness ``lam`` and imperfection factor ``alpha``."""
    phi = 0.5 * (1 + alpha * (lam - 0.2) + lam ** 2)
    return min(1.0, 1.0 / (phi + math.sqrt(max(phi ** 2 - lam ** 2, 0.0))))


def m_cr(sec: dict, L_cr: float, C1: float = 1.0) -> float:
    """Elastic critical moment of a doubly symmetric I-beam, load at the shear centre (Timoshenko & Gere)."""
    E, Gs = STEEL["E"], STEEL["G"]
    return C1 * math.pi ** 2 * E * sec["Iz"] / L_cr ** 2 * math.sqrt(
        sec["Iw"] / sec["Iz"] + L_cr ** 2 * Gs * sec["J"] / (math.pi ** 2 * E * sec["Iz"]))


def _ltb_alpha(p):
    h_b = p["girder_h"] / p["girder_b"]
    if p.get("section_type", "welded") == "rolled":
        return (0.21, "a") if h_b <= 2 else (0.34, "b")
    return (0.49, "c") if h_b <= 2 else (0.76, "d")


def _ltb(p, limits):
    ld = girder_loads(p)
    s = ld["sec"]
    n_br = int(p.get("lateral_braces", 0))
    L_cr = float(p["span"]) / (n_br + 1)
    C1 = float(p.get("ltb_C1", 1.0))
    Mcr = m_cr(s, L_cr, C1)
    lam = math.sqrt(s["Wy"] * _fy(p) / Mcr)
    alpha, curve = _ltb_alpha(p)
    x = chi(lam, alpha)
    Mb = x * s["Wy"] * _fy(p)
    ok = ld["MEd"] <= Mb
    mech = tr(f"A mesa comprimida, travada só a cada {fmt(L_cr)} m, tende a sair do plano e torcer: momento crítico "
              f"elástico {fmt(Mcr / 1e3)} kN·m, esbeltez λ_LT = {fmt(lam)}, fator χ_LT = {fmt(x)} (curva {curve}); "
              f"resistência à flambagem lateral {fmt(Mb / 1e3)} kN·m contra {fmt(ld['MEd'] / 1e3)} kN·m de cálculo"
              + (" — a viga alta e estreita que resolveu a flecha flamba de lado antes de escoar." if not ok else "."),
              f"The compression flange, restrained only every {fmt(L_cr)} m, tends to move sideways and twist: elastic "
              f"critical moment {fmt(Mcr / 1e3)} kN m, slenderness {fmt(lam)}, reduction chi_LT = {fmt(x)} (curve "
              f"{curve}); buckling resistance {fmt(Mb / 1e3)} kN m against {fmt(ld['MEd'] / 1e3)} kN m design"
              + (" — the deep, narrow girder that cured the deflection buckles sideways before it yields."
                 if not ok else "."))
    return verdict(demand=ld["MEd"] / 1e3, capacity=Mb / 1e3, unit="kN m",
                   demand_label=tr("momento de cálculo", "design moment"),
                   capacity_label=tr("resistência à flambagem lateral", "lateral-torsional buckling resistance"),
                   mechanism=mech, method="EN 1993-1-1 6.3.2.2 general case with the elastic critical moment of "
                                         "Timoshenko & Gere (C1 given, load at the shear centre)",
                   quantities={"M_cr": q(Mcr, "N m", "elastic critical moment"),
                               "lambda_LT": q(lam, "1", "non-dimensional slenderness"),
                               "chi_LT": q(x, "1", "reduction factor"), "L_cr": q(L_cr, "m", "unbraced length"),
                               "M_b_Rd": q(Mb, "N m", "buckling resistance moment")},
                   assumptions=[tr("o tabuleiro não trava a mesa comprimida", "the deck does not restrain the "
                                   "compression flange"),
                                tr(f"C1 = {fmt(C1)} (momento uniforme: conservador para carga distribuída)",
                                   f"C1 = {fmt(C1)} (uniform moment: conservative for a distributed load)")],
                   sources=[SRC["ec3"], TIMOSHENKO])


register_mode(FailureMode(
    "lateral_torsional_buckling", "structures", tr("flambagem lateral com torção", "lateral-torsional buckling"),
    tr("mesa comprimida sem travamento lateral flamba antes de escoar",
       "unrestrained compression flange buckles before the section yields"), _ltb,
    needs=(Need("span", "m", "span"), Need("girder_b", "m", "flange width")),
    sources=(SRC["ec3"], TIMOSHENKO), countermeasures=("lateral_bracing", "wider_flanges"),
    uncertain={"fy": 0.05, "deck_dead_load": 0.1},
    known_answers=lambda: [{
        "params": {"span": 6.0, "girder_h": 0.3, "girder_b": 0.15, "girder_tf": 0.0107, "girder_tw": 0.0071,
                   "deck_width": 2.0, "deck_dead_load": 0.0},
        "quantity": "M_cr",
        "expected": math.pi ** 2 * 210e9 * i_section(0.3, 0.15, 0.0107, 0.0071)["Iz"] / 36 * math.sqrt(
            i_section(0.3, 0.15, 0.0107, 0.0071)["Iw"] / i_section(0.3, 0.15, 0.0107, 0.0071)["Iz"]
            + 36 * 81e9 * i_section(0.3, 0.15, 0.0107, 0.0071)["J"]
            / (math.pi ** 2 * 210e9 * i_section(0.3, 0.15, 0.0107, 0.0071)["Iz"])),
        "rtol": 1e-12, "source": "M_cr formula, Timoshenko & Gere (1961), ch. 6, with C1 = 1"}]))


def frequency(p: dict) -> dict:
    ld = girder_loads(p)
    m = ld["mass_per_m"]
    EI = ld["n"] * STEEL["E"] * ld["sec"]["Iy"]
    L = float(p["span"])
    f1 = math.pi / (2 * L ** 2) * math.sqrt(EI / m)
    return {"f1": f1, "m": m, "EI": EI, "M": m * L / 2}


def _amplification(r, xi):
    return r ** 2 / math.sqrt((1 - r ** 2) ** 2 + (2 * xi * r) ** 2)


def pedestrian_response(p: dict) -> dict:
    fr = frequency(p)
    f1, M = fr["f1"], fr["M"]
    xi = float(p.get("damping_ratio", 0.005))
    Gw = float(p.get("pedestrian_weight", 700.0))
    dlf = (0.4, 0.1, 0.1)
    mu = float(p.get("tmd_mass_ratio", 0.0)) if "tmd" in p.get("_elements", ()) else 0.0
    best = (0.0, 0, 0.0)
    for k, a in enumerate(dlf, start=1):
        fs = np.linspace(1.6 * k, 2.4 * k, 801)
        if 1.6 * k <= f1 <= 2.4 * k:
            fs = np.append(fs, f1)
        amp = np.array([_amplification(f / f1, xi) for f in fs])
        if mu > 0:
            amp = np.minimum(amp, math.sqrt(1 + 2 / mu))
        i = int(np.argmax(amp))
        acc = a * Gw / M * float(amp[i])
        if acc > best[0]:
            best = (acc, k, float(fs[i]))
    return {**fr, "acc": best[0], "harmonic": best[1], "f_pace": best[2], "xi": xi, "mu": mu}


def _rayleigh_check(p, fr):
    try:
        from pinneapple_simulation.numerical_solvers.beam_bvp_fdm import (
            solve_euler_bernoulli_beam_1d,
        )
    except ImportError as exc:
        return self_check("solver vs closed form: Rayleigh frequency from FD deflection", True, None, None,
                          FDM_BEAM, f"not run: {exc}")
    sol = solve_euler_bernoulli_beam_1d(L_m=p["span"], E_Pa=fr["EI"], I_m4=1.0, q_N_per_m=fr["m"] * G, nx=401)
    w = np.abs(sol["deflection_m"])
    f_r = math.sqrt(G * np.trapezoid(w, sol["z"]) / np.trapezoid(w ** 2, sol["z"])) / (2 * math.pi)
    rel = abs(f_r - fr["f1"]) / fr["f1"]
    return self_check("solver vs closed form: Rayleigh frequency from FD deflection", rel < 0.005, rel, 0.005,
                      FDM_BEAM + " + Rayleigh's method (static deflection shape)")


def _resonance(p, limits):
    r = pedestrian_response(p)
    lim = float(limits.get("acc_max", p.get("acc_max", 0.7)))
    ok = r["acc"] <= lim
    tmd = r["mu"] > 0
    band = f"{fmt(1.6 * r['harmonic'])}-{fmt(2.4 * r['harmonic'])} Hz"
    if not ok:
        mech = tr(f"Frequência natural vertical {fmt(r['f1'])} Hz dentro da faixa do {r['harmonic']}º harmônico do "
                  f"passo ({band}): um pedestre a {fmt(r['f_pace'])} Hz entra em ressonância; com amortecimento de "
                  f"{fmt(100 * r['xi'])} % e massa modal {fmt(r['M'] / 1e3)} t a aceleração chega a "
                  f"{fmt(r['acc'])} m/s², acima de {fmt(lim)} m/s² de conforto.",
                  f"Vertical natural frequency {fmt(r['f1'])} Hz inside the band of walking harmonic {r['harmonic']} "
                  f"({band}): a pedestrian at {fmt(r['f_pace'])} Hz drives it at resonance; with {fmt(100 * r['xi'])} % "
                  f"damping and {fmt(r['M'] / 1e3)} t modal mass the acceleration reaches {fmt(r['acc'])} m/s^2, "
                  f"above the {fmt(lim)} m/s^2 comfort limit.")
    else:
        mech = tr(f"f1 = {fmt(r['f1'])} Hz; pior caso {fmt(r['acc'])} m/s² ({r['harmonic']}º harmônico a "
                  f"{fmt(r['f_pace'])} Hz)" + (f", limitado pelo amortecedor sintonizado (μ = {fmt(r['mu'])})"
                                               if tmd else "") + f", abaixo de {fmt(lim)} m/s².",
                  f"f1 = {fmt(r['f1'])} Hz; worst case {fmt(r['acc'])} m/s^2 (harmonic {r['harmonic']} at "
                  f"{fmt(r['f_pace'])} Hz)" + (f", capped by the tuned mass damper (mu = {fmt(r['mu'])})" if tmd else "")
                  + f", below {fmt(lim)} m/s^2.")
    return verdict(demand=r["acc"], capacity=lim, unit="m/s^2",
                   demand_label=tr("aceleração vertical de pico", "peak vertical acceleration"),
                   capacity_label=tr("limite de conforto", "comfort limit"), mechanism=mech,
                   method="first vertical mode of a simply supported beam, steady resonant response to one walking "
                          "pedestrian (harmonics 1-3); tuned mass damper as Den Hartog's optimum amplification",
                   quantities={"f1": q(r["f1"], "Hz", "first vertical natural frequency"),
                               "modal_mass": q(r["M"], "kg", "modal mass m L/2"),
                               "acceleration": q(r["acc"], "m/s^2", "peak vertical acceleration"),
                               "pace": q(r["f_pace"], "Hz", "governing pacing frequency")},
                   assumptions=[tr("um pedestre em regime permanente no meio do vão (limite superior)",
                                   "one pedestrian in steady state at mid-span (upper bound)"),
                                tr(f"amortecimento estrutural {fmt(100 * r['xi'])} % (premissa)",
                                   f"structural damping {fmt(100 * r['xi'])} % (premise)")],
                   sources=[BACHMANN, EN1990, DEN_HARTOG], self_checks=[_rayleigh_check(p, r)],
                   validity=[validity("first mode below 5 Hz (EN 1990 A2.4.3.2 asks for the check)", r["f1"], None,
                                      1e9)],
                   series={"f1": r["f1"], "bands": [[1.6 * k, 2.4 * k] for k in (1, 2, 3)]})


register_mode(FailureMode(
    "pedestrian_resonance", "structural dynamics", tr("ressonância com o passo de pedestres",
                                                      "resonance with pedestrian footfall"),
    tr("frequência natural na faixa do passo humano: aceleração acima do conforto",
       "natural frequency in the walking band: acceleration above comfort"), _resonance,
    needs=(Need("span", "m", "span"), Need("deck_dead_load", "Pa", "deck dead load")),
    sources=(BACHMANN, EN1990, DEN_HARTOG), countermeasures=("tuned_mass_damper", "deepen_girders"),
    uncertain={"damping_ratio": 0.4, "deck_dead_load": 0.1},
    known_answers=lambda: [{
        "params": {"span": 20.0, "deck_width": 2.0, "deck_dead_load": 1000.0, "n_girders": 2,
                   "girder_h": 0.6, "girder_b": 0.2, "girder_tf": 0.015, "girder_tw": 0.008},
        "quantity": "f1",
        "expected": math.pi / (2 * 400) * math.sqrt(2 * 210e9 * i_section(0.6, 0.2, 0.015, 0.008)["Iy"] / (
            (2 * i_section(0.6, 0.2, 0.015, 0.008)["A"] * 7850 * G + 2000.0) / G)),
        "rtol": 1e-12, "source": "f1 = pi/(2 L^2) sqrt(EI/m) (simply supported beam)"}]))


# ---------------------------------------------------------------------------------------------- countermeasures
def _eps(p):
    return math.sqrt(235e6 / _fy(p))


def _deepen(p, v):
    tw0, tf = p["girder_tw"], p["girder_tf"]

    def make(h):
        tw = max(tw0, math.ceil((h - 2 * tf) / (124 * _eps(p)) * 1e3) / 1e3)   # web class 3 (EN 1993-1-1 Table 5.2)
        return {"girder_h": round(h, 3), "girder_tw": tw}
    h0 = p["girder_h"]
    ch, _ = size_to_pass(v.mode, p, [h0 + 0.05 * k for k in range(1, 61)], make)
    return ch, (), ()


def _steel_cost(b, a):
    n = int(b.get("n_girders", 2))
    dA = i_section(a["girder_h"], a["girder_b"], a["girder_tf"], a["girder_tw"])["A"] - \
        i_section(b["girder_h"], b["girder_b"], b["girder_tf"], b["girder_tw"])["A"]
    return n * dA * STEEL["rho"] * b["span"], "kg", "added girder steel"


register_countermeasure(Countermeasure(
    "deepen_girders", "girder_depth", tr("vigas mais altas", "deeper girders"),
    tr("I cresce com o cubo da altura: menos flecha e frequência natural mais alta (f ∝ √(EI/m))",
       "I grows with the cube of the depth: less deflection and a higher natural frequency (f ~ sqrt(EI/m))"),
    ("deflection", "bending_strength", "pedestrian_resonance"), _deepen, _steel_cost, unit_cost=1.0,
    unit_cost_note="relative index per kg of girder steel (premise)", repeatable=True))


def _bracing(p, v):
    ch, _ = size_to_pass(v.mode, p, range(1, 15), lambda n: {"lateral_braces": n})
    return ch, (), ()


def _bracing_cost(b, a):
    n_new = a["lateral_braces"] - b.get("lateral_braces", 0)
    s = b["deck_width"] / max(int(b.get("n_girders", 2)) - 1, 1)
    length = 2 * math.hypot(s, a["girder_h"]) * (int(b.get("n_girders", 2)) - 1)
    return n_new * length * float(b.get("brace_mass_per_m", 3.77)), "kg", "cross-frame steel"


register_countermeasure(Countermeasure(
    "lateral_bracing", "lateral_bracing", tr("contraventamento lateral (quadros transversais)",
                                             "lateral bracing (cross-frames)"),
    tr("trava a mesa comprimida: o comprimento destravado cai e o momento crítico sobe com 1/L²",
       "restrains the compression flange: the unbraced length falls and the critical moment rises as 1/L^2"),
    ("lateral_torsional_buckling",), _bracing, _bracing_cost, unit_cost=1.5,
    unit_cost_note="relative index per kg of fabricated bracing steel (premise; girder steel = 1)", repeatable=True))


def _wider(p, v):
    tf0, b0 = p["girder_tf"], p["girder_b"]

    def make(b):
        tf = max(tf0, math.ceil((b / 2 - p["girder_tw"] / 2) / (14 * _eps(p)) * 1e3) / 1e3)   # outstand class 3
        return {"girder_b": round(b, 3), "girder_tf": tf}
    ch, _ = size_to_pass(v.mode, p, [b0 + 0.025 * k for k in range(1, 41)], make)
    return ch, (), ()


register_countermeasure(Countermeasure(
    "wider_flanges", "flange_width", tr("mesas mais largas", "wider flanges"),
    tr("I_z e a rigidez à torção crescem: a viga resiste a sair do plano",
       "I_z and the torsional stiffness grow: the girder resists moving out of plane"),
    ("lateral_torsional_buckling",), _wider, _steel_cost, unit_cost=1.0,
    unit_cost_note="relative index per kg of girder steel (premise)", repeatable=True))


def _tmd(p, v):
    mu_min = float(p.get("tmd_min_mass_ratio", 0.01))          # practical minimum: robust to detuning (premise)
    ch, _ = size_to_pass("pedestrian_resonance", p, [mu_min + 0.0025 * k for k in range(0, 41)],
                         lambda mu: {"tmd_mass_ratio": round(mu, 4)}, add=("tmd",))
    return ch, (), ()


def _tmd_cost(b, a):
    return a["tmd_mass_ratio"] * frequency(a)["M"], "kg", "tuned mass"


register_countermeasure(Countermeasure(
    "tuned_mass_damper", "tmd", tr("amortecedor de massa sintonizada", "tuned mass damper"),
    tr("uma massa pequena sintonizada no 1º modo absorve a energia na ressonância: amplificação limitada a "
       "√(1 + 2/μ) (Den Hartog)",
       "a small mass tuned to the first mode absorbs the energy at resonance: amplification capped at "
       "sqrt(1 + 2/mu) (Den Hartog)"),
    ("pedestrian_resonance",), _tmd, _tmd_cost, unit_cost=3.0,
    unit_cost_note="relative index per kg of damper mass (premise; girder steel = 1)"))


# ------------------------------------------------------------------------------------------------ column buckling
def _column(p, limits):
    A, Ic, L, fy = float(p["column_A"]), float(p["column_I"]), float(p["column_L_cr"]), _fy(p)
    Ncr = math.pi ** 2 * STEEL["E"] * Ic / L ** 2
    lam = math.sqrt(A * fy / Ncr)
    alpha = {"a0": 0.13, "a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}[p.get("column_curve", "b")]
    x = chi(lam, alpha)
    Nb = x * A * fy
    NEd = float(p["column_N_Ed"])
    return verdict(demand=NEd / 1e3, capacity=Nb / 1e3, unit="kN",
                   demand_label=tr("compressão de cálculo", "design compression"),
                   capacity_label=tr("resistência à flambagem", "buckling resistance"),
                   mechanism=tr(f"Carga crítica de Euler {fmt(Ncr / 1e3)} kN, esbeltez {fmt(lam)}, χ = {fmt(x)}: "
                                f"resistência {fmt(Nb / 1e3)} kN contra {fmt(NEd / 1e3)} kN.",
                                f"Euler load {fmt(Ncr / 1e3)} kN, slenderness {fmt(lam)}, chi = {fmt(x)}: resistance "
                                f"{fmt(Nb / 1e3)} kN against {fmt(NEd / 1e3)} kN."),
                   method="EN 1993-1-1 6.3.1 flexural buckling (Euler critical load, buckling curve)",
                   quantities={"N_cr": q(Ncr, "N", "Euler critical load"), "lambda": q(lam, "1", "slenderness"),
                               "chi": q(x, "1", "reduction factor")},
                   sources=[SRC["ec3"]])


register_mode(FailureMode(
    "flexural_buckling", "structures", tr("flambagem de coluna", "column buckling"),
    tr("compressão acima da resistência à flambagem por flexão", "compression above the flexural buckling "
       "resistance"), _column,
    needs=(Need("column_A", "m^2", "area"), Need("column_I", "m^4", "minor-axis I"),
           Need("column_L_cr", "m", "buckling length"), Need("column_N_Ed", "N", "design compression")),
    sources=(SRC["ec3"],), countermeasures=("column_midheight_brace",),
    known_answers=lambda: [{"params": {"column_A": 1e-3, "column_I": 1e-6, "column_L_cr": 3.0,
                                       "column_N_Ed": 1e5, "fy": 235e6},
                            "quantity": "N_cr", "expected": math.pi ** 2 * 210e9 * 1e-6 / 9.0, "rtol": 1e-12,
                            "source": "Euler: pi^2 E I / L^2"}]))

register_countermeasure(Countermeasure(
    "column_midheight_brace", "column_brace", tr("travar a coluna a meia altura", "brace the column at mid-height"),
    tr("o comprimento de flambagem cai à metade e a carga crítica quadruplica",
       "the buckling length halves and the critical load quadruples"),
    ("flexural_buckling",), lambda p, v: ({"column_L_cr": p["column_L_cr"] / 2}, (), ()),
    lambda b, a: (1.0, "brace", "one brace"), unit_cost=0.5, unit_cost_note="relative index per brace (premise)"))


# ------------------------------------------------------------------------------------------------ fatigue
def _fatigue(p, limits):
    from pinneapple_physics.closed_form.cantilever_fatigue import basquin_fatigue_life_cycles
    sa = float(p["stress_amplitude"])
    life = basquin_fatigue_life_cycles(sa, float(p["fatigue_strength_coeff"]), float(p["fatigue_strength_exp"]))
    need = float(p["design_cycles"])
    return verdict(demand=need, capacity=life, unit="cycles", scale=need,
                   demand_label=tr("ciclos de projeto", "design cycles"),
                   capacity_label=tr("vida em fadiga (Basquin)", "fatigue life (Basquin)"),
                   mechanism=tr(f"Amplitude de tensão {fmt(sa / 1e6)} MPa: vida de Basquin {fmt(life)} ciclos contra "
                                f"{fmt(need)} de projeto.",
                                f"Stress amplitude {fmt(sa / 1e6)} MPa: Basquin life {fmt(life)} cycles against "
                                f"{fmt(need)} design cycles."),
                   method="Basquin S-N law (pinneapple_physics.closed_form.cantilever_fatigue)",
                   quantities={"life": q(life, "cycles", "Basquin life")},
                   sources=["Basquin (1910); pinneapple_physics.closed_form.cantilever_fatigue"])


register_mode(FailureMode(
    "fatigue", "structures", tr("fadiga", "fatigue"),
    tr("vida em fadiga menor que os ciclos de projeto", "fatigue life below the design cycles"), _fatigue,
    needs=(Need("stress_amplitude", "Pa", "stress amplitude"), Need("fatigue_strength_coeff", "Pa", "sigma_f'"),
           Need("fatigue_strength_exp", "1", "Basquin exponent b"), Need("design_cycles", "1", "design cycles")),
    countermeasures=("thicker_section",),
    known_answers=lambda: [{"params": {"stress_amplitude": 200e6, "fatigue_strength_coeff": 900e6,
                                       "fatigue_strength_exp": -0.1, "design_cycles": 1e6},
                            "quantity": "life", "expected": 0.5 * (200 / 900) ** (1 / -0.1), "rtol": 1e-12,
                            "source": "Basquin: N = 1/2 (sigma_a / sigma_f')^(1/b) reversals -> cycles"}]))


def _thicker(p, v):
    ch, _ = size_to_pass("fatigue", p, [1.05 ** k for k in range(1, 40)],
                         lambda k: {"stress_amplitude": p["stress_amplitude"] / k ** 2, "section_scale":
                                    p.get("section_scale", 1.0) * k})
    return ch, (), ()


register_countermeasure(Countermeasure(
    "thicker_section", "section", tr("seção mais espessa", "thicker section"),
    tr("tensão de flexão cai com o quadrado da espessura", "bending stress falls with the square of the thickness"),
    ("fatigue",), _thicker, lambda b, a: (a.get("section_scale", 1.0) - b.get("section_scale", 1.0), "1",
                                          "relative extra section mass"), unit_cost=1.0, repeatable=True))


# ------------------------------------------------------------------------------------------------ wind uplift
def _uplift(p, limits):
    from ..props import air
    rho = air(20.0)["rho"]
    V = float(p["wind_speed_design"])
    qp = 0.5 * rho * V ** 2
    suction = abs(float(p["cp_uplift"])) * qp
    w = float(p["covering_mass"]) * G * math.cos(math.radians(float(p.get("slope_deg", 0.0))))
    fix = float(p.get("fixings_per_m2", 0.0)) * float(p.get("fixing_capacity", 0.0))
    cap = w + fix
    return verdict(demand=suction, capacity=cap, unit="Pa",
                   demand_label=tr("sucção do vento", "wind suction"),
                   capacity_label=tr("peso próprio + fixações", "self weight + fixings"),
                   mechanism=tr(f"Vento de {fmt(V)} m/s: pressão dinâmica {fmt(qp)} Pa, sucção {fmt(suction)} Pa "
                                f"contra {fmt(w)} Pa de peso e {fmt(fix)} Pa de fixações" +
                                (": as peças levantam." if suction > cap else "."),
                                f"{fmt(V)} m/s wind: dynamic pressure {fmt(qp)} Pa, suction {fmt(suction)} Pa against "
                                f"{fmt(w)} Pa of weight and {fmt(fix)} Pa of fixings" +
                                (": the pieces lift." if suction > cap else ".")),
                   method="quasi-static wind suction |Cp| 1/2 rho V^2 against weight and fixings",
                   quantities={"q_dyn": q(qp, "Pa", "dynamic pressure"), "suction": q(suction, "Pa", "suction")},
                   assumptions=[tr("Cp de sucção dado (premissa do caso)", "given suction Cp (case premise)")],
                   sources=["Bernoulli dynamic pressure; Cp from the case"])


register_mode(FailureMode(
    "wind_uplift", "wind", tr("arrancamento por sucção do vento", "wind-suction uplift"),
    tr("sucção do vento maior que peso + fixações", "wind suction above weight + fixings"), _uplift,
    needs=(Need("wind_speed_design", "m/s", "design wind speed"), Need("cp_uplift", "1", "suction coefficient"),
           Need("covering_mass", "kg/m^2", "covering mass")),
    countermeasures=("tile_clips",), uncertain={"wind_speed_design": 0.1, "cp_uplift": 0.2},
    known_answers=lambda: [{"params": {"wind_speed_design": 30.0, "cp_uplift": -1.0, "covering_mass": 40.0},
                            "quantity": "q_dyn", "expected": 0.5 * 1.2045 * 900.0, "rtol": 2e-3,
                            "source": "1/2 rho V^2 with rho = 1.204 kg/m^3 at 20 °C"}]))

register_countermeasure(Countermeasure(
    "tile_clips", "clips", tr("grampos/fixação de cada peça", "a clip on every piece"),
    tr("cada peça passa a ter uma fixação mecânica somada ao peso", "every piece gets a mechanical fixing on top of "
       "its weight"),
    ("wind_uplift",), lambda p, v: ({"fixings_per_m2": float(p.get("pieces_per_m2", 15.0)),
                                     "fixing_capacity": float(p.get("clip_capacity", 100.0))}, (), ()),
    lambda b, a: (a["fixings_per_m2"] - b.get("fixings_per_m2", 0.0), "clips/m2", "clips per m2"),
    unit_cost=0.02, unit_cost_note="relative index per clip (premise)"))
