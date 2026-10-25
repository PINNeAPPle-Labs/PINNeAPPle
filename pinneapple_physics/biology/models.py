"""Biological models as :class:`ODESystem` builders: the ultradian glucose-insulin model, nonlinear PK/PD and a
minimal PBPK.

* :func:`ultradian_glucose_insulin`: Sturis et al., Am J Physiol 260(5):E801-E809 (1991), in the form and with the
  nominal values of Daneker, Zhang, Karniadakis and Lu, arXiv:2202.01723 (2022), Eqs (1)-(2) and Table 1, the model
  of the SBINN papers. States are amounts (Ip, Ii in mU, G in mg; G/(10 Vg) is mg/dl); meals enter through
  ``IG(t) = sum_j m_j k exp(k (t_j - t)) H(t - t_j)``.
* :func:`pk_one_compartment_mm`: one-compartment pharmacokinetics with first-order absorption and Michaelis-Menten
  (saturable) elimination ``dA/dt = ka A_gut - Vmax C / (Km + C)``, ``C = A / V``, with an effect compartment
  ``dCe/dt = ke0 (C - Ce)`` and the sigmoid Emax pharmacodynamic model ``E = E0 + Emax Ce^h / (EC50^h + Ce^h)``
  (Holford and Sheiner, Clin Pharmacokinet 6:429-453, 1981). Doses: an oral bolus at t = 0 and optional repeats.
* :func:`minimal_pbpk`: a flow-limited (perfusion-rate-limited) PBPK model with venous and arterial blood, lung,
  liver (well-stirred hepatic clearance ``CLint fu``), kidney (renal clearance) and a lumped rest of body; every
  tissue ``V_T dC_T/dt = Q_T (C_in - C_T / (Kp_T / B:P))`` (Rowland and Tozer, Clinical Pharmacokinetics and
  Pharmacodynamics, 4th ed., ch. 6 and appendix). The default physiology is a placeholder of the right order for a
  70 kg adult and must be replaced by values from a physiology source for any real use; the defaults are not a
  dataset.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import sympy as sp

from .ode_system import TIME, ODESystem

__all__ = ["ultradian_glucose_insulin", "ULTRADIAN_NOMINAL", "ULTRADIAN_SEARCH_RANGE", "pk_one_compartment_mm",
           "minimal_pbpk"]

# Daneker et al. 2022, Table 1 (nominal values); k is 1/120 min^-1 in the authors' data generator
# (github.com/lu-group/sbinn, sbinn/data_generation.py), printed as 0.0083 in the table.
ULTRADIAN_NOMINAL = {"Vp": 3.0, "Vi": 11.0, "Vg": 10.0, "E": 0.2, "tp": 6.0, "ti": 100.0, "td": 12.0,
                     "k": 1 / 120, "Rm": 209.0, "a1": 6.6, "C1": 300.0, "C2": 144.0, "C3": 100.0, "C4": 80.0,
                     "C5": 26.0, "Ub": 72.0, "U0": 4.0, "Um": 90.0, "Rg": 180.0, "alpha": 7.5, "beta": 1.772}
# Table 1 search ranges: the first seven from Sturis et al., the rest (0.2 p*, 1.8 p*)
ULTRADIAN_SEARCH_RANGE = {"E": (0.1, 0.3), "tp": (4.0, 8.0), "ti": (60.0, 140.0), "td": (25 / 3, 50 / 3),
                          **{p: (0.2 * ULTRADIAN_NOMINAL[p], 1.8 * ULTRADIAN_NOMINAL[p])
                             for p in ("k", "Rm", "a1", "C1", "C2", "C4", "C5", "Ub", "U0", "Um", "Rg", "alpha",
                                       "beta")}}


def _s(n: str) -> sp.Symbol:
    return sp.Symbol(n, real=True)


def ultradian_glucose_insulin(meals: Sequence[tuple[float, float]] = ((300, 60), (650, 40), (1100, 50)),
                              initial: Sequence[float] = (12.0, 4.0, 110.0, 0.0, 0.0, 0.0),
                              **parameters: float) -> ODESystem:
    """The six-state ultradian model. ``meals`` are (time in min, carbohydrate in g); ``initial`` is
    (Ip uU/ml, Ii uU/ml, G mg/dl, h1, h2, h3) as in the paper, converted to amounts with Vp, Vi, Vg.
    The algebraic variable ``G_conc`` is the glucose concentration in mg/dl."""
    p = {**ULTRADIAN_NOMINAL, **parameters}
    Ip, Ii, G, h1, h2, h3 = (_s(n) for n in ("Ip", "Ii", "G", "h1", "h2", "h3"))
    S = {n: _s(n) for n in p}
    f1 = S["Rm"] / (1 + sp.exp(-G / (S["Vg"] * S["C1"]) + S["a1"]))
    f2 = S["Ub"] * (1 - sp.exp(-G / (S["Vg"] * S["C2"])))
    kappa = (1 / S["Vi"] + 1 / (S["E"] * S["ti"])) / S["C4"]
    f3 = (S["U0"] + S["Um"] / (1 + (kappa * Ii) ** (-S["beta"]))) / (S["Vg"] * S["C3"])
    f4 = S["Rg"] / (1 + sp.exp(S["alpha"] * (h3 / (S["Vp"] * S["C5"]) - 1)))
    IG = sp.Integer(0)
    for tj, mj in meals:
        IG += sp.Float(mj * 1000.0) * S["k"] * sp.exp(S["k"] * (sp.Float(tj) - TIME)) * sp.Heaviside(TIME - tj, 0)
    ex = S["E"] * (Ip / S["Vp"] - Ii / S["Vi"])
    rhs = {"Ip": f1 - ex - Ip / S["tp"], "Ii": ex - Ii / S["ti"], "G": f4 + IG - f2 - f3 * G,
           "h1": (Ip - h1) / S["td"], "h2": (h1 - h2) / S["td"], "h3": (h2 - h3) / S["td"]}
    ip0, ii0, g0, a, b, c = initial
    init = {"Ip": ip0 * S["Vp"], "Ii": ii0 * S["Vi"], "G": g0 * 10 * S["Vg"], "h1": a, "h2": b, "h3": c}
    return ODESystem("ultradian_glucose_insulin", list(rhs), rhs, init, p,
                     {"IG": IG, "G_conc": G / (10 * S["Vg"])},
                     units={"Ip": "mU", "Ii": "mU", "G": "mg", "G_conc": "mg/dl", "IG": "mg/min", "t": "min"},
                     meta={"meals": [tuple(m) for m in meals],
                           "source": "Sturis et al. 1991; Daneker et al. 2022 (arXiv:2202.01723) Eqs 1-2, Table 1"})


def pk_one_compartment_mm(*, dose: float = 100.0, doses: Sequence[tuple[float, float]] = (), ka: float = 1.0,
                          V: float = 50.0, Vmax: float = 10.0, Km: float = 2.0, ke0: float = 0.5, E0: float = 0.0,
                          Emax: float = 100.0, EC50: float = 1.0, hill: float = 1.0, F: float = 1.0) -> ODESystem:
    """Oral dose ``dose`` (mg) at t = 0 into the gut, extra oral ``doses`` as (time h, mg) given as first-order
    absorption inputs ``F m ka exp(-ka (t - t_j)) H(t - t_j)``; time in h, C in mg/L. Algebraic variables:
    ``C`` (plasma), ``E`` (effect), ``elim`` (elimination rate, mg/h)."""
    A_gut, A, Ce = _s("A_gut"), _s("A"), _s("Ce")
    S = {n: _s(n) for n in ("ka", "V", "Vmax", "Km", "ke0", "E0", "Emax", "EC50", "hill", "F")}
    C = _s("C")
    extra = sp.Integer(0)
    for tj, m in doses:
        extra += S["F"] * sp.Float(m) * S["ka"] * sp.exp(-S["ka"] * (TIME - tj)) * sp.Heaviside(TIME - tj, 0)
    rhs = {"A_gut": -S["ka"] * A_gut, "A": S["F"] * S["ka"] * A_gut + extra - _s("elim"),
           "Ce": S["ke0"] * (C - Ce)}
    assign = {"C": A / S["V"], "elim": S["Vmax"] * C / (S["Km"] + C),
              "E": S["E0"] + S["Emax"] * Ce ** S["hill"] / (S["EC50"] ** S["hill"] + Ce ** S["hill"])}
    params = dict(ka=ka, V=V, Vmax=Vmax, Km=Km, ke0=ke0, E0=E0, Emax=Emax, EC50=EC50, hill=hill, F=F)
    return ODESystem("pk_one_compartment_mm", list(rhs), rhs, {"A_gut": float(dose), "A": 0.0, "Ce": 0.0}, params,
                     assign, units={"A": "mg", "C": "mg/L", "t": "h", "elim": "mg/h"},
                     meta={"source": "Michaelis-Menten elimination; sigmoid Emax with effect compartment "
                                     "(Holford and Sheiner 1981)"})


_PBPK_DEFAULT = {   # placeholder physiology (L, L/h) of the right order for a 70 kg adult; replace for real use
    "Q_lung": 312.0, "Q_liver": 90.0, "Q_kidney": 72.0, "V_art": 1.7, "V_ven": 3.4, "V_lung": 0.5,
    "V_liver": 1.8, "V_kidney": 0.3, "V_rest": 60.0,
}


def minimal_pbpk(*, dose: float = 100.0, Kp: Mapping[str, float] | None = None, BP: float = 1.0, fu: float = 0.1,
                 CLint: float = 100.0, CLr: float = 1.0, physiology: Mapping[str, float] | None = None) -> ODESystem:
    """IV bolus ``dose`` (mg) into venous blood. Blood and tissue concentrations in mg/L, time in h. ``Kp``
    tissue:plasma partition coefficients (lung, liver, kidney, rest). The liver eliminates the unbound concentration
    ``fu C_liver / Kp_liver`` with intrinsic clearance ``CLint`` (which at steady state is the well-stirred hepatic
    plasma clearance ``CL_ws = Q fu CLint / (Q + fu CLint / BP)``); the kidney filters ``CLr`` times the arterial
    plasma concentration. ``A_elim`` accumulates the eliminated amount, so ``sum(V C) + A_elim = dose`` is a mass
    balance check, and ``dose / (CL_ws + CLr)`` is the area under the arterial plasma curve."""
    phys = {**_PBPK_DEFAULT, **(physiology or {})}
    kp = {"lung": 1.0, "liver": 5.0, "kidney": 3.0, "rest": 2.0, **(Kp or {})}
    params = {**phys, **{f"Kp_{k}": v for k, v in kp.items()}, "BP": BP, "fu": fu, "CLint": CLint, "CLr": CLr}
    S = {n: _s(n) for n in params}
    Cv, Ca, Clu, Cli, Ck, Cr = (_s(n) for n in ("C_ven", "C_art", "C_lung", "C_liver", "C_kidney", "C_rest"))
    Q_rest = S["Q_lung"] - S["Q_liver"] - S["Q_kidney"]
    out = {t: c * S["BP"] / S[f"Kp_{t}"] for t, c in (("lung", Clu), ("liver", Cli), ("kidney", Ck), ("rest", Cr))}
    hep = S["fu"] * S["CLint"] * Cli / S["Kp_liver"]
    ren = S["CLr"] * Ca / S["BP"]
    rhs = {
        "C_ven": (S["Q_liver"] * out["liver"] + S["Q_kidney"] * out["kidney"] + Q_rest * out["rest"]
                  - S["Q_lung"] * Cv) / S["V_ven"],
        "C_art": S["Q_lung"] * (out["lung"] - Ca) / S["V_art"],
        "C_lung": S["Q_lung"] * (Cv - out["lung"]) / S["V_lung"],
        "C_liver": (S["Q_liver"] * (Ca - out["liver"]) - hep) / S["V_liver"],
        "C_kidney": (S["Q_kidney"] * (Ca - out["kidney"]) - ren) / S["V_kidney"],
        "C_rest": Q_rest * (Ca - out["rest"]) / S["V_rest"],
        "A_elim": hep + ren,
    }
    assign = {"C_plasma_art": Ca / S["BP"], "C_plasma": Cv / S["BP"],
              "CL_ws": S["Q_liver"] * S["fu"] * S["CLint"] / (S["Q_liver"] + S["fu"] * S["CLint"] / S["BP"])}
    init = {k: 0.0 for k in rhs}
    init["C_ven"] = float(dose) / phys["V_ven"]
    return ODESystem("minimal_pbpk", list(rhs), rhs, init, params, assign,
                     units={"t": "h", "C_ven": "mg/L", "C_plasma": "mg/L", "A_elim": "mg"},
                     meta={"dose": float(dose), "source": "flow-limited PBPK, well-stirred liver (Rowland and Tozer)",
                           "physiology": "placeholder defaults; replace from a physiology source",
                           "volumes": {"C_ven": "V_ven", "C_art": "V_art", "C_lung": "V_lung",
                                       "C_liver": "V_liver", "C_kidney": "V_kidney", "C_rest": "V_rest"}})
