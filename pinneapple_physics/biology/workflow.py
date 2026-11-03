"""The standard bio workflow (#1205): a well-mixed biological model (SBML or a library model) simulated with a
stiffness-aware solver, checked, analysed for sensitivity, identified from data and used for a dosing or
intervention design that is re-verified under parameter uncertainty.

1. **Model**: an SBML file or shipped BioModels entry (:func:`.sbml.read_sbml`), or a library model: SIR / SEIR,
   Hodgkin-Huxley, FitzHugh-Nagumo, two-compartment PK with Emax PD, Michaelis-Menten (:mod:`.standard_models`).
2. **Checks of the model**: mass conservation (the declared balance, or every linear invariant of the right-hand
   side found from its null space), positivity of the amounts, gating variables in [0, 1], and the SBML round trip
   (export, re-import, same trajectory; against the stored libRoadRunner reference for the shipped BioModels).
3. **Simulation**: the stiffness ratio of the Jacobian along the trajectory picks the solver; BDF and RK45 are both
   run when the problem is not too stiff for RK45 and their quantities of interest must agree.
4. **Closed-form verification**: the SIR / SEIR final-size relation; PK ``AUC(inf) = F D / CL``; the Hodgkin-Huxley
   all-or-none threshold and its Hopf onset against Rinzel and Miller (1980); the FitzHugh-Nagumo Hopf current in
   closed form; the Michaelis-Menten substrate against the Schnell-Mendoza Lambert-W solution of the quasi-steady
   state.
5. **Sensitivity**: normalised local sensitivities (central differences in log parameters) and Sobol first-order and
   total indices (:func:`pinneapple_analysis.sensitivity.sobol.sobol_indices`) over a stated range.
6. **Identification**: measurements (given, or synthetic at the nominal parameters with stated noise) fitted through
   :func:`pinneapple_analysis.inverse_problems.workflow.inverse_workflow`: identifiability first, intervals, the
   held-out rows, residual whiteness and a synthetic recovery. Optionally an SBINN
   (:class:`pinneapple_analysis.inverse_problems.sbinn.SBINN`) on the same data as a second estimate.
7. **Design under constraints**: an epidemic intervention (the smallest contact reduction that keeps the peak
   prevalence under a capacity), a dosing regimen (interval and dose: steady-state trough above the efficacy level and
   peak below the toxicity level, the smallest daily dose) or a stimulus amplitude (fires the cell); each chosen so
   the constraint holds with the target probability over the parameter uncertainty.
8. **Re-verification**: the design simulated on a fresh, independent sample of the uncertain parameters.
"""
from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ode_system import ODESystem, Trajectory

__all__ = ["bio_workflow", "build_model", "LIBRARY_MODELS", "hh_hopf_current", "sir_final_size", "fhn_hopf_current",
           "wilks_order"]

LIBRARY_MODELS = ("sir", "seir", "hodgkin_huxley", "fitzhugh_nagumo", "pk_two_compartment", "michaelis_menten")
# Rinzel and Miller, Math Biosci 49:27-59 (1980): the resting state of the HH membrane loses stability (subcritical
# Hopf) at a sustained current of 9.78 uA/cm^2.
HH_HOPF_PUBLISHED = 9.78
FAST = {"rtol": 1e-6, "atol": 1e-8}          # sampling runs (Sobol, Monte Carlo, design)
TIGHT = {"rtol": 1e-9, "atol": 1e-11}        # verification runs


# --------------------------------------------------------------------------- closed forms
def sir_final_size(N: float, S0: float, R_init: float, R0: float) -> float:
    """``S_inf`` from ``ln(S0 / S_inf) = R0 (N - R(0) - S_inf) / N`` (the root below ``N / R0``)."""
    from scipy.optimize import brentq
    g = lambda s: math.log(S0 / s) - R0 * (N - R_init - s) / N      # noqa: E731
    hi = min(S0, N / R0) if R0 > 1 else S0
    if R0 <= 1 or g(hi * (1 - 1e-12)) > 0:
        return S0
    return float(brentq(g, 1e-300 * N + 1e-12, hi * (1 - 1e-12), xtol=1e-12 * N, rtol=1e-14))


def fhn_hopf_current(a: float, b: float, eps: float) -> float:
    """Sustained current at which the FitzHugh-Nagumo rest point loses stability: trace ``1 - v^2 - eps b = 0`` at
    ``v = -sqrt(1 - eps b)``, then ``I = v^3/3 - v + (v + a)/b``."""
    v = -math.sqrt(1 - eps * b)
    return v ** 3 / 3 - v + (v + a) / b


def _equilibrium(system: ODESystem, p: Mapping[str, float], guess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import fsolve
    f, J = system._numeric()
    pv = [p[k] for k in system.parameters]
    n = len(system.states)
    F = lambda y: np.asarray(f(1e9, y, pv), float).reshape(n)          # noqa: E731  (after every pulse)
    Jf = lambda y: np.asarray(J(1e9, y, pv), float).reshape(n, n)      # noqa: E731
    with warnings.catch_warnings():           # slow progress near the Hopf point; the eigenvalues decide
        warnings.simplefilter("ignore", RuntimeWarning)
        y = fsolve(F, guess, fprime=Jf, xtol=1e-13)
    return y, np.linalg.eigvals(Jf(y))


def hh_hopf_current(system: ODESystem, lo: float = 0.0, hi: float = 20.0) -> float:
    """Sustained ``I_ext`` where the largest real part of the rest point's eigenvalues crosses zero (bisection)."""
    guess = system.initial_values()
    p = dict(system.parameters)

    def lam(I):
        y, ev = _equilibrium(system, {**p, "I_ext": I, "amp": 0.0}, guess)
        return float(np.max(ev.real)), y
    a, _ = lam(lo)
    b, _ = lam(hi)
    if a >= 0 or b <= 0:
        return math.nan
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        v, y = lam(mid)
        guess = y
        lo, hi = (mid, hi) if v < 0 else (lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- models
@dataclass
class _Spec:
    t_end: float
    observe: str
    identify: tuple[str, ...]
    vary: tuple[str, ...]
    uncertain: dict[str, float]                 # parameter -> coefficient of variation
    qoi: Callable[[Trajectory, ODESystem, Mapping[str, float]], dict[str, float]]
    primary: str
    kind: str
    design: dict[str, Any] = field(default_factory=dict)


def _epi_qoi(tr, s, p):
    N = p["N"]
    i = int(np.argmax(tr["I"]))
    return {"peak_prevalence": float(tr["I"][i] / N), "t_peak": float(tr.t[i]), "attack_rate": float(tr["R"][-1] / N)}


def _hh_qoi(tr, s, p):
    v = tr["V"] if "V" in tr.values else tr["v"]
    return {"V_peak": float(v.max()), "fires": float(v.max() > (0.0 if "V" in tr.values else 1.0))}


def _pk_qoi(tr, s, p):
    C = tr["A_c"] / p["V1"]
    i = int(np.argmax(C))
    E = p["E0"] + p["Emax"] * tr["Ce"] ** p["hill"] / (p["EC50"] ** p["hill"] + tr["Ce"] ** p["hill"])
    return {"AUC": float(tr["AUC"][-1]), "Cmax": float(C[i]), "Tmax": float(tr.t[i]), "E_peak": float(E.max())}


def _mm_qoi(tr, s, p):
    S0 = s.meta["S0"]
    St = tr["S"] + tr["C"]
    k = int(np.argmax(St <= 0.5 * S0)) if np.any(St <= 0.5 * S0) else -1
    t_half = float(np.interp(0.5 * S0, St[k::-1], tr.t[k::-1])) if k > 0 else math.nan
    return {"t_half": t_half, "P_end": float(tr["P"][-1])}


SPECS: dict[str, _Spec] = {
    "sir": _Spec(300.0, "I", ("beta", "gamma"), ("beta", "gamma"), {"beta": 0.15, "gamma": 0.10}, _epi_qoi,
                 "peak_prevalence", "epidemic", {"capacity": 0.05, "t_on": 20.0, "t_off": None, "u_max": 0.9,
                                                 "horizon": 1500.0}),
    "seir": _Spec(400.0, "I", ("beta", "gamma"), ("beta", "sigma", "gamma"),
                  {"beta": 0.15, "sigma": 0.1, "gamma": 0.10}, _epi_qoi, "peak_prevalence", "epidemic",
                  {"capacity": 0.05, "t_on": 30.0, "t_off": None, "u_max": 0.9, "horizon": 2000.0}),
    "hodgkin_huxley": _Spec(30.0, "V", ("gNa", "gK"), ("gNa", "gK", "gL"), {"gNa": 0.1, "gK": 0.1, "gL": 0.1},
                            _hh_qoi, "V_peak", "excitable", {"amp_max": 50.0}),
    "fitzhugh_nagumo": _Spec(60.0, "v", ("a", "eps"), ("a", "b", "eps"), {"a": 0.1, "b": 0.05, "eps": 0.1},
                             _hh_qoi, "V_peak", "excitable", {"amp_max": 5.0}),
    "pk_two_compartment": _Spec(240.0, "A_c", ("CL", "V1"), ("ka", "CL", "V1", "Q"),
                                {"ka": 0.3, "CL": 0.3, "V1": 0.2}, _pk_qoi, "AUC", "pk",
                                {"c_min": 1.0, "c_max": 10.0, "intervals": [4.0, 6.0, 8.0, 12.0, 24.0],
                                 "days": 7.0}),
    "michaelis_menten": _Spec(600.0, "P", ("k1", "k2"), ("k1", "km1", "k2"), {}, _mm_qoi, "t_half", "enzyme"),
}


def build_model(model: str = "sir", sbml: str | None = None, **params: Any) -> tuple[ODESystem, _Spec]:
    """The system and its workflow defaults. ``model="sbml"`` reads ``sbml`` (a path, SBML text, or a shipped
    BioModels id such as ``"BIOMD0000000012"`` or ``"12"``)."""
    from . import standard_models as sm
    if model == "sbml":
        from .sbml import biomodel, list_biomodels, read_sbml
        if not sbml:
            raise ValueError("model='sbml' needs sbml= (a path, SBML text or a BioModels id); shipped: "
                             f"{list_biomodels()}")
        src = str(sbml)
        s = biomodel(src) if (src.isdigit() or src.startswith("BIOMD")) and "<" not in src else read_sbml(src)
        if params:
            s = s.with_parameters(**{k: float(v) for k, v in params.items()})
        s.meta.setdefault("nonnegative", list(s.states))
        obs = s.states[0]

        def q(tr, _s, _p, obs=obs):
            return {f"mean_{obs}": float(np.mean(tr[obs])), f"max_{obs}": float(np.max(tr[obs]))}
        comps = set((s.meta.get("compartments") or {}).values())
        vary = tuple(k for k, v in s.parameters.items() if v > 0 and k not in comps)[:3]
        return s, _Spec(100.0, obs, (), vary, {}, q, f"mean_{obs}", "sbml")
    builders = {"sir": sm.sir, "seir": sm.seir, "hodgkin_huxley": sm.hodgkin_huxley,
                "fitzhugh_nagumo": sm.fitzhugh_nagumo, "pk_two_compartment": sm.pk_two_compartment,
                "michaelis_menten": sm.michaelis_menten}
    if model not in builders:
        raise ValueError(f"unknown model {model!r}; library models: {list(builders)} or 'sbml'")
    return builders[model](**params), SPECS[model]


# --------------------------------------------------------------------------- helpers
def _lognormal(rng, nominal: Mapping[str, float], cv: Mapping[str, float], n: int) -> list[dict[str, float]]:
    out = []
    for _ in range(n):
        d = {}
        for k, c in cv.items():
            sg = math.sqrt(math.log(1 + c * c))
            d[k] = float(nominal[k] * math.exp(sg * rng.standard_normal()))
        out.append(d)
    return out


def _linear_invariants(system: ODESystem, T: float, rng) -> np.ndarray:
    """Rows ``w`` with ``w . f(x, p, t) = 0`` at random states and times: conserved linear combinations."""
    f, _ = system._numeric()
    pv = list(system.parameters.values())
    n = len(system.states)
    y0 = system.initial_values()
    scale = np.where(np.abs(y0) > 0, np.abs(y0), max(1.0, float(np.abs(y0).max())))
    F = []
    for _ in range(4 * n + 4):
        y = scale * rng.uniform(0.05, 1.5, n)
        F.append(np.asarray(f(float(rng.uniform(0, T)), y, pv), float).reshape(n) / scale.max())
    F = np.asarray(F)
    if not np.all(np.isfinite(F)):
        return np.zeros((0, n))
    _, sv, Vt = np.linalg.svd(F)
    sv = np.concatenate([sv, np.zeros(n - sv.size)])
    tol = 1e-9 * max(sv.max(), 1e-300)
    return Vt[sv <= tol] if Vt.shape[0] == n else np.zeros((0, n))


def _stiffness(system: ODESystem, tr: Trajectory, p: Mapping[str, float]) -> dict[str, float]:
    _, J = system._numeric()
    pv = [p[k] for k in system.parameters]
    n = len(system.states)
    lmax, ratios = 0.0, []
    for i in np.linspace(0, tr.t.size - 1, 12).astype(int):
        y = np.array([tr[s][i] for s in system.states])
        Jm = np.asarray(J(float(tr.t[i]), y, pv), float).reshape(n, n)
        if not np.all(np.isfinite(Jm)):
            continue
        re = np.abs(np.linalg.eigvals(Jm).real)
        re = re[re > 1e-12 * max(re.max(), 1e-300)]
        if re.size:
            lmax = max(lmax, float(re.max()))
            ratios.append(float(re.max() / re.min()))
    T = float(tr.t[-1] - tr.t[0])
    return {"max_rate_x_horizon": lmax * T, "stiffness_ratio": max(ratios) if ratios else 1.0}


class _Model:
    """Simulations of one system with parameter overrides, counting the runs."""

    def __init__(self, system: ODESystem, spec: _Spec, t_eval: np.ndarray, method: str):
        self.s, self.spec, self.t, self.method, self.runs = system, spec, t_eval, method, 0

    def run(self, p: Mapping[str, float] | None = None, *, tight: bool = False, t=None, system=None) -> Trajectory:
        s = system or self.s
        self.runs += 1
        tol = TIGHT if tight else FAST
        with warnings.catch_warnings():       # RK45 ignores the analytic Jacobian the system passes
            warnings.simplefilter("ignore", UserWarning)
            tr = s.simulate(self.t if t is None else t, {**s.parameters, **(p or {})}, method=self.method,
                            outputs=[], **tol)
        if not tr.success:
            raise RuntimeError(f"{s.name}: integration failed ({tr.message}) at {dict(p or {})}")
        return tr

    def qoi(self, p=None, *, tight=False, system=None) -> dict[str, float]:
        s = system or self.s
        pp = {**s.parameters, **(p or {})}
        return self.spec.qoi(self.run(p, tight=tight, system=system), s, pp)


# --------------------------------------------------------------------------- the workflow
def bio_workflow(*, model: str = "sir", sbml: str | None = None, model_params: Mapping[str, Any] | None = None,
                 t_end: float | None = None, n_points: int = 401, sensitivity_range: float = 0.2,
                 n_sobol: int = 32, identify: Sequence[str] | None = None, measurements: Mapping[str, Any] | None
                 = None, n_measurements: int = 24, noise: float = 0.03, design: Mapping[str, Any] | None = None,
                 uncertainty: Mapping[str, float] | None = None, target_probability: float = 0.9,
                 n_design: int = 32, n_reverify: int = 100, sbinn: Mapping[str, Any] | None = None,
                 conservation_tolerance: float = 1e-6, closed_form_tolerance: float = 1e-3,
                 seed: int = 0) -> dict[str, Any]:
    """Run the standard bio workflow; see the module docstring. ``measurements``:
    ``{"t": [...], "values": [...], "sigma": float}`` of the observable (default: synthetic at the nominal
    parameters with ``noise`` times the observable's range). ``uncertainty`` maps parameters to coefficients of
    variation (lognormal, median at the nominal value). ``design`` overrides the model's design constraints."""
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    system, spec = build_model(model, sbml, **dict(model_params or {}))
    T = float(t_end or spec.t_end)
    t = np.linspace(0.0, T, int(n_points))
    p0 = dict(system.parameters)
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        status = "skip" if ok == "skip" else ("inconclusive" if ok is None else ("pass" if ok else "fail"))
        checks.append({"code": code, "status": status, "detail": detail,
                       "value": value if not isinstance(value, np.generic) else value.item()})

    # ---- 3. simulation and solver choice
    system._numeric()                     # compile once, so the timings below are of the integration only
    t0 = time.perf_counter()
    tr = system.simulate(t, method="BDF", **TIGHT)
    bdf_s = time.perf_counter() - t0
    if not tr.success:
        raise RuntimeError(f"{system.name}: BDF integration failed: {tr.message}")
    stiff = _stiffness(system, tr, p0)
    solver: dict[str, Any] = {"BDF": {"rhs_evaluations": tr.n_rhs, "seconds": bdf_s}, **stiff}
    qoi_bdf = spec.qoi(tr, system, p0)
    if stiff["max_rate_x_horizon"] < 1e5:
        t0 = time.perf_counter()
        with warnings.catch_warnings():       # the analytic Jacobian is passed but unused by RK45
            warnings.simplefilter("ignore", UserWarning)
            tr_rk = system.simulate(t, method="RK45", **TIGHT)
        solver["RK45"] = {"rhs_evaluations": tr_rk.n_rhs, "seconds": time.perf_counter() - t0}
        if tr_rk.success:
            q_rk = spec.qoi(tr_rk, system, p0)
            dev = max(abs(q_rk[k] - v) / max(abs(v), 1e-12) for k, v in qoi_bdf.items() if np.isfinite(v))
            chk("solver.agree", dev < 1e-4, f"BDF and RK45 quantities of interest agree to {dev:.2g} (< 1e-4)", dev)
        method = "BDF" if (not tr_rk.success or solver["RK45"]["seconds"] > bdf_s) else "RK45"
    else:
        method = "BDF"
        chk("solver.agree", "skip", f"max |Re lambda| x horizon = {stiff['max_rate_x_horizon']:.3g}: too stiff to "
                                    "run RK45 as a cross-check")
    solver["chosen"] = method
    solver["why"] = (f"stiffness ratio {stiff['stiffness_ratio']:.3g}, fastest rate x horizon "
                     f"{stiff['max_rate_x_horizon']:.3g}: " + (
                         f"{method} was the faster here ({solver['BDF']['seconds'] * 1e3:.3g} ms, "
                         f"{solver['BDF']['rhs_evaluations']} evaluations BDF; {solver['RK45']['seconds'] * 1e3:.3g} "
                         f"ms, {solver['RK45']['rhs_evaluations']} RK45)" if "RK45" in solver else
                         "BDF (implicit) because the problem is too stiff for an explicit method"))
    M = _Model(system, spec, t, method)

    # ---- 2. model checks
    mb = system.meta.get("mass_balance")
    if mb:
        traj = system.simulate(t, method="BDF", **TIGHT)
        vals = dict(traj.values)
        need = [k for k in [*mb["weights"], mb["total"]] if k not in vals and k in system.assignments]
        vals.update(system.evaluate(need, t, vals))
        total = np.asarray(vals[mb["total"]] if mb["total"] in vals else p0[mb["total"]], float)
        bal = sum(w * np.asarray(vals[k]) for k, w in mb["weights"].items())
        err = float(np.max(np.abs(bal - total)) / max(float(np.max(np.abs(total))), 1e-300))
        chk("model.conservation", err < conservation_tolerance,
            f"{' + '.join(mb['weights'])} = {mb['total']} to {err:.2g} relative (< {conservation_tolerance:g})", err)
    else:
        W = _linear_invariants(system, T, rng)
        if W.shape[0]:
            Y = tr.matrix(system.states)
            inv = Y @ W.T
            err = float(np.max(np.abs(inv - inv[0])) / max(float(np.max(np.abs(Y))), 1e-300))
            names = ["+".join(f"{w:.3g}*{s}" for w, s in zip(row / row[np.argmax(np.abs(row))], system.states,
                                                                strict=True) if abs(w) > 1e-8) for row in W]
            shown = "; ".join(n if len(n) <= 80 else n[:77] + "..." for n in names[:3])
            chk("model.conservation", err < conservation_tolerance,
                f"{W.shape[0]} linear invariant(s) of the right-hand side ({shown}) hold to {err:.2g} "
                f"of the largest amount (< {conservation_tolerance:g})", err)
        else:
            chk("model.conservation", "skip", "the right-hand side has no conserved linear combination of states "
                                              "(open system): nothing to conserve")
    nonneg = [s for s in system.meta.get("nonnegative", []) if s in tr.values]
    if nonneg:
        worst = min(float(np.min(tr[s]) / max(float(np.max(np.abs(tr[s]))), 1e-300)) for s in nonneg)
        chk("model.positivity", worst >= -1e-7, f"min over {nonneg} of x / max|x| = {worst:.3g} (>= -1e-7)", worst)
    else:
        chk("model.positivity", "skip", "no state is an amount or a concentration")
    bounded = system.meta.get("bounded") or {}
    if bounded:
        out = {k: (float(tr[k].min()), float(tr[k].max())) for k in bounded}
        ok = all(lo - 1e-9 <= out[k][0] and out[k][1] <= hi + 1e-9 for k, (lo, hi) in bounded.items())
        chk("model.bounds", ok, "gating variables " + ", ".join(f"{k} in [{a:.3g}, {b:.3g}]" for k, (a, b) in
                                                                 out.items()) + " (within [0, 1])")
    # SBML round trip
    try:
        from .sbml import BIOMODELS_DIR, read_sbml
        s2 = read_sbml(system.to_sbml())
        tr2 = s2.simulate(t, method="BDF", **TIGHT, outputs=[])
        err = max(float(np.linalg.norm(tr2[s] - tr[s]) / max(np.linalg.norm(tr[s]), 1e-300)) for s in system.states)
        chk("sbml.round_trip", err < 1e-6, f"exported to SBML L3V2 and re-imported: trajectories agree to {err:.2g}",
            err)
        if model == "sbml":
            import json
            ref = json.loads((BIOMODELS_DIR / "roadrunner_reference.json").read_text())
            mid = system.name
            if mid in ref["models"]:
                tref = np.asarray(ref["t"])
                trr = system.simulate(tref, rtol=1e-11, atol=1e-14, outputs=[])
                e = max(float(np.linalg.norm(trr[k] - np.asarray(v)) / max(np.linalg.norm(v), 1e-30))
                        for k, v in ref["models"][mid].items())
                chk("sbml.reference", e < 1e-6, f"{mid} against the stored libRoadRunner trajectories: {e:.2g} "
                                                "(< 1e-6)", e)
    except Exception as exc:  # noqa: BLE001 - an unsupported construct is reported, not hidden
        chk("sbml.round_trip", False, f"SBML export or re-import failed: {type(exc).__name__}: {exc}")

    # ---- 4. closed-form verification
    closed: dict[str, Any] = {}
    if spec.kind == "epidemic":
        iv = system.meta.get("intervention", {})
        if p0.get("u", 0.0) > 0 and iv.get("t_off", 0) > iv.get("t_on", 0):
            chk("verify.final_size", "skip", "an intervention is active: the final-size relation holds without one")
        else:
            N = p0["N"]
            s_inf = sir_final_size(N, float(tr["S"][0]), float(tr["R"][0]), p0["beta"] / p0["gamma"])
            over = float(tr["I"][-1] / N) < 1e-6 and float(tr.values.get("E", np.zeros(1))[-1] / N) < 1e-6
            rel = abs(float(tr["S"][-1]) - s_inf) / N
            closed = {"S_inf_closed_form": s_inf, "S_inf_simulated": float(tr["S"][-1]), "error_over_N": rel}
            chk("verify.final_size", (rel < closed_form_tolerance) if over else None,
                f"ln(S0/S_inf) = R0 (N - R(0) - S_inf)/N gives S_inf/N = {s_inf / N:.6f}; simulated "
                f"{float(tr['S'][-1]) / N:.6f} (difference {rel:.2g} of N)"
                + ("" if over else "; the epidemic is not over at t_end: lengthen it"), rel)
    elif spec.kind == "pk":
        p = p0
        n_d = len(system.meta.get("dose_times", [0]))
        auc_cf = p["F"] * p["dose"] * n_d / p["CL"]
        vals = dict(tr.values)
        vals.update(system.evaluate(["gut_pending"], t, vals))
        left = float(vals["A_gut"][-1] + vals["A_c"][-1] + vals["A_p"][-1] + vals["gut_pending"][-1])
        auc_inf = float(tr["AUC"][-1]) + left / p["CL"]        # linear PK: what is left is eliminated at CL
        rel = abs(auc_inf - auc_cf) / auc_cf
        closed = {"AUC_closed_form": auc_cf, "AUC_simulated": auc_inf, "tail_fraction": left / max(n_d * p["F"] *
                                                                                                   p["dose"], 1e-300)}
        chk("verify.auc", rel < closed_form_tolerance,
            f"AUC(0-inf) = F n D / CL = {auc_cf:.6g} mg h/L; simulated {auc_inf:.6g} (tail {closed['tail_fraction']:.2g}"
            f" of the dose added as amount left / CL): {rel:.2g} relative", rel)
    elif spec.kind == "excitable":
        hh = model == "hodgkin_huxley"
        thr = _threshold(M, spec.design.get("amp_max", 50.0), tight=True)
        below = M.qoi({"amp": 0.98 * thr}, tight=True)["V_peak"]
        above = M.qoi({"amp": 1.02 * thr}, tight=True)["V_peak"]
        lo_lim, hi_lim = (-40.0, 0.0) if hh else (0.0, 1.0)
        closed["pulse_threshold"] = thr
        closed["V_peak_below_above"] = [below, above]
        chk("verify.all_or_none", below < lo_lim and above > hi_lim,
            f"pulse threshold {thr:.4g} ({'uA/cm^2' if hh else 'dimensionless'}): peak {below:.3g} at 0.98x and "
            f"{above:.3g} at 1.02x (all or none)", thr)
        if hh:
            Ih = hh_hopf_current(system)
            rel = abs(Ih - HH_HOPF_PUBLISHED) / HH_HOPF_PUBLISHED
            closed["hopf_current"] = Ih
            chk("verify.hopf", rel < 0.01, f"rest point loses stability at I_ext = {Ih:.4g} uA/cm^2 (eigenvalues of "
                                           f"the Jacobian); Rinzel and Miller 1980: {HH_HOPF_PUBLISHED} ({rel:.2%})", rel)
        else:
            Ih = hh_hopf_current(system, 0.0, 1.0)      # below the upper Hopf point
            cf = fhn_hopf_current(p0["a"], p0["b"], p0["eps"])
            rel = abs(Ih - cf) / abs(cf)
            closed["hopf_current"] = Ih
            chk("verify.hopf", rel < 1e-6, f"Hopf current from the eigenvalues {Ih:.8g}; closed form "
                                           f"v^3/3 - v + (v+a)/b at v = -sqrt(1 - eps b): {cf:.8g} ({rel:.2g})", rel)
    elif spec.kind == "enzyme":
        from scipy.special import lambertw
        p = p0
        E0, S0 = system.meta["E0"], system.meta["S0"]
        Km, Vmax = (p["km1"] + p["k2"]) / p["k1"], p["k2"] * E0
        eps = E0 / (S0 + Km)
        St = tr["S"] + tr["C"]
        Sq = Km * np.real(lambertw((S0 / Km) * np.exp((S0 - Vmax * t) / Km)))
        err = float(np.max(np.abs(St - Sq)) / S0)
        closed = {"Km": Km, "Vmax": Vmax, "epsilon": eps, "max_error_over_S0": err}
        chk("verify.qssa", err < 5 * eps, f"S + C against the Lambert-W solution of dS/dt = -Vmax S/(Km + S): "
                                          f"{err:.3g} of S0 (< 5 eps, eps = E0/(S0 + Km) = {eps:.3g})", err)

    # ---- 5. sensitivity
    vary = [k for k in spec.vary if k in p0]
    local: dict[str, dict[str, float]] = {}
    base = M.qoi()
    for k in vary:
        up, dn = M.qoi({k: p0[k] * 1.01}), M.qoi({k: p0[k] / 1.01})
        local[k] = {q: float((math.log(abs(up[q]) + 1e-300) - math.log(abs(dn[q]) + 1e-300)) / (2 * math.log(1.01)))
                    if base[q] != 0 and np.isfinite(base[q]) else 0.0 for q in base}
    sobol: dict[str, Any] = {}
    if vary and n_sobol > 0:
        from pinneapple_analysis.sensitivity.sobol import sobol_indices
        space = {k: {"dist": "uniform", "low": p0[k] * (1 - sensitivity_range), "high": p0[k] * (1 + sensitivity_range)}
                 for k in vary}
        qnames = [q for q in base if q != "fires"]

        def fn(X):
            return np.array([[M.qoi(dict(zip(vary, row, strict=True)))[q] for q in qnames] for row in X])
        si = sobol_indices(fn, space, n=n_sobol, n_boot=100, seed=seed, outputs=qnames)
        j = qnames.index(spec.primary) if spec.primary in qnames else 0
        sobol = {"output": qnames[j], "names": vary, "S1": si.S1[j].tolist(), "ST": si.ST[j].tolist(),
                 "S1_ci": si.S1_ci[:, j].tolist(), "ST_ci": si.ST_ci[:, j].tolist(), "n_evaluations": si.n_evals,
                 "range": sensitivity_range}
        bad = [vary[i] for i in range(len(vary)) if si.ST_ci[1, j, i] < si.S1_ci[0, j, i]]
        chk("sensitivity.sobol_consistent", not bad,
            f"total >= first-order index for every parameter of {qnames[j]} within the bootstrap intervals "
            f"({si.n_evals} runs)" if not bad else f"total below first order for {bad}: raise n_sobol")
        rank_l = max(vary, key=lambda k: abs(local[k][qnames[j]]))
        rank_s = vary[int(np.argmax(si.ST[j]))]
        sobol["most_influential"] = {"local": rank_l, "sobol_total": rank_s}

    # ---- 6. identification
    ident: dict[str, Any] = {"run": False}
    names = list(identify) if identify is not None else [k for k in spec.identify if k in p0]
    if names:
        obs = spec.observe
        synthetic = measurements is None
        if synthetic:
            tm = np.linspace(T / n_measurements, T, int(n_measurements))
            y_true = np.interp(tm, t, tr[obs] if obs in tr.values else system.evaluate([obs], t, tr.values)[obs])
            sig = noise * float(np.ptp(tr[obs] if obs in tr.values else y_true))
            y = y_true + sig * rng.standard_normal(tm.size)
            meas = {"inputs": {"t": tm.tolist()}, "values": y.tolist(), "sigma": sig}
        else:
            tm = np.asarray(measurements["t"], float)
            meas = {"inputs": {"t": tm.tolist()}, "values": list(map(float, measurements["values"])),
                    "sigma": measurements.get("sigma", 1.0)}
        cache: dict[tuple, Any] = {}
        grid = np.unique(np.concatenate([[0.0], tm]))

        def tool(t, **theta):
            key = tuple(sorted(theta.items()))
            if key not in cache:
                cache.clear()
                trj = M.run(theta, t=grid)
                cache[key] = trj[obs] if obs in trj.values else system.evaluate([obs], grid, trj.values,
                                                                                  {**p0, **theta})[obs]
            return float(np.interp(t, grid, cache[key]))
        start = 1.3
        params = {k: {"lower": p0[k] / 3, "upper": p0[k] * 3, "initial": p0[k] * start} for k in names}
        from pinneapple_analysis.inverse_problems.workflow import inverse_workflow
        r = inverse_workflow(tool=tool, measurements=meas, output="y", parameters=params, bootstrap=0,
                             n_recovery=6, seed=seed)
        ident = {"run": True, "synthetic": synthetic, "observable": obs, "n_measurements": int(tm.size),
                 "sigma": meas["sigma"], "truth": {k: p0[k] for k in names} if synthetic else None,
                 "start": {k: p0[k] * start for k in names}, **{k: v for k, v in r.items() if k != "checks"}}
        for c in r["checks"]:
            checks.append(dict(c))
        if synthetic and r.get("estimated"):
            inside = {k: bool(r["intervals"][k] and r["intervals"][k][0] <= p0[k] <= r["intervals"][k][1])
                      for k in names}
            ident["truth_inside"] = inside
            chk("identify.truth_recovered", all(inside.values()),
                "the 95 % intervals hold the parameters that generated the data: "
                + ", ".join(f"{k} {p0[k]:.4g} in [{r['intervals'][k][0]:.4g}, {r['intervals'][k][1]:.4g}]"
                            for k in names))
        if sbinn:
            ident["sbinn"] = _run_sbinn(system, names, params, obs, meas, sbinn, chk)
        else:
            ident["sbinn"] = {"run": False, "why": "not run: pass sbinn={'epochs': ...} for a second estimate by a "
                                                   "systems-biology informed network"}
    else:
        chk("identify.structural", "skip", "no parameter to identify was named (identify=[...])")

    # ---- 7-8. design and re-verification
    unc = {k: float(v) for k, v in (uncertainty if uncertainty is not None else spec.uncertain).items() if k in p0}
    dz = {**spec.design, **dict(design or {})}
    des = _design(model, spec, system, M, p0, unc, dz, target_probability, n_design, n_reverify, seed, chk)

    failed = [c["code"] for c in checks if c["status"] == "fail"]
    verdict = "pass" if not failed else "fail"
    summary = (f"{system.name}: {len(system.states)} states, solver {method}; "
               + "; ".join(f"{k} {v:.4g}" for k, v in base.items())
               + (f"; design: {des.get('summary')}" if des.get("summary") else "")
               + f"; {sum(c['status'] == 'pass' for c in checks)} checks pass, {len(failed)} fail"
               + (f" ({', '.join(failed)})" if failed else ""))
    return {"model": system.describe(), "kind": spec.kind, "t_end": T, "solver": solver, "qoi": base,
            "closed_form": closed, "local_sensitivity": local, "sobol": sobol, "identification": ident,
            "design": des, "checks": checks, "failed": failed, "verdict": verdict, "summary": summary,
            "runs": M.runs, "seconds": time.perf_counter() - t_start,
            "trajectory": {"t": t.tolist(), **{s: tr[s].tolist() for s in system.states}}}


def _threshold(M: _Model, amp_max: float, *, tight: bool = False, p: Mapping[str, float] | None = None,
               iters: int = 30) -> float:
    """Smallest pulse amplitude that fires (bisection on the all-or-none response)."""
    p = dict(p or {})
    fires = lambda a: M.qoi({**p, "amp": a}, tight=tight)["fires"] > 0.5      # noqa: E731
    lo, hi = 0.0, float(amp_max)
    if not fires(hi):
        return math.inf
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fires(mid) else (mid, hi)
    return hi


def _run_sbinn(system, names, params, obs, meas, cfg, chk) -> dict[str, Any]:
    from pinneapple_analysis.inverse_problems.sbinn import SBINN, SBINNConfig
    if obs not in system.states:
        return {"run": False, "why": f"SBINN observes states; {obs} is an algebraic variable"}
    tm = np.asarray(meas["inputs"]["t"], float)
    y = np.asarray(meas["values"], float)
    c = SBINNConfig(hidden=tuple(cfg.get("hidden", (32, 32))), n_features=int(cfg.get("n_features", 0)),
                    epochs_data=int(cfg.get("epochs_data", 1000)), epochs=int(cfg.get("epochs", 5000)),
                    lr=float(cfg.get("lr", 3e-3)), record_every=500, seed=int(cfg.get("seed", 0)),
                    time_budget=float(cfg.get("time_budget", 120.0)))
    t0 = time.perf_counter()
    sb = SBINN(system, names, {k: (v["lower"], v["upper"]) for k, v in params.items()}, {obs: (tm, y)}, config=c)
    res = sb.fit()
    fc = res.forward_check(system, tm, observed=[obs], data={obs: y})
    out = {"run": True, "parameters": res.parameters, "epochs": res.epochs_run, "stopped_by_budget":
           res.stopped_by_budget, "seconds": time.perf_counter() - t0, "ode_vs_data": fc["ode_vs_data"].get(obs)}
    rel_noise = float(meas["sigma"]) / max(float(np.sqrt(np.mean(y ** 2))), 1e-300)
    ok = fc["ode_vs_data"].get(obs, math.inf) < 3 * rel_noise
    chk("identify.sbinn_forward", ok, f"the ODE integrated with the SBINN parameters "
                                      f"{', '.join(f'{k}={v:.4g}' for k, v in res.parameters.items())} misfits the "
                                      f"data by {fc['ode_vs_data'].get(obs, math.nan):.3g} relative (< 3x the noise "
                                      f"{rel_noise:.3g})")
    return out


def wilks_order(n: int, target: float, confidence: float = 0.95) -> int:
    """Smallest ``k`` such that the ``k``-th smallest of ``n`` independent draws bounds the ``target`` quantile with
    ``confidence`` (non-parametric tolerance limit, Wilks 1941): ``P(Beta(k, n + 1 - k) >= target) >= confidence``.
    Returns ``n + 1`` when ``n`` draws are too few."""
    from scipy.stats import beta
    for k in range(1, n + 1):
        if beta.sf(target, k, n + 1 - k) >= confidence:
            return k
    return n + 1


def _design(model, spec, system, M, p0, unc, dz, target, n_design, n_reverify, seed, chk) -> dict[str, Any]:
    """Each design picks, per draw of the uncertain parameters, the least effort that meets the constraint, then the
    Wilks order statistic of those efforts: the constraint holds with ``target`` probability at 95 % confidence."""
    if spec.kind not in ("epidemic", "pk", "excitable"):
        chk("design.feasible", "skip", f"no design objective is defined for a {spec.kind} model")
        return {"run": False}
    rng_d = np.random.default_rng(seed + 101)
    rng_v = np.random.default_rng(seed + 202)          # fresh, independent sample for the re-verification
    samples = _lognormal(rng_d, p0, unc, n_design) if unc else [{}]
    fresh = _lognormal(rng_v, p0, unc, n_reverify) if unc else [{}]
    k = wilks_order(len(samples), target) if unc else 1
    if k > len(samples):
        need = int(math.ceil(math.log(0.05) / math.log(target)))
        chk("design.feasible", False, f"{len(samples)} design draws cannot bound the {target:.0%} quantile at 95 % "
                                      f"confidence: n_design must be at least {need}")
        return {"run": True, "feasible": False}
    floor = target - 2 * math.sqrt(target * (1 - target) / max(len(fresh), 1))
    how = (f"the {k}-th smallest of {len(samples)} per-draw requirements (Wilks: >= {target:.0%} with 95 % "
           f"confidence)" if unc else "nominal parameters (no uncertainty given)")
    common = {"n_design": len(samples), "n_reverify": len(fresh), "wilks_k": k, "uncertainty_cv": unc,
              "target_probability": target}

    def kth(values):
        v = np.sort(np.asarray(values, float))
        return float(v[k - 1])

    if spec.kind == "epidemic":
        from . import standard_models as sm
        cap, t_on, u_max = float(dz["capacity"]), float(dz["t_on"]), float(dz["u_max"])
        horizon = float(dz.get("horizon") or 5 * M.t[-1])
        t_off = float(dz["t_off"]) if dz.get("t_off") is not None else horizon       # sustained by default
        build = sm.sir if model == "sir" else sm.seir
        kw = {k_: p0[k_] for k_ in ("beta", "gamma", "N", "sigma") if k_ in p0}
        s_iv = build(**kw, I0=float(system.initial_values()[system.states.index("I")]), t_on=t_on, t_off=t_off)
        Md = _Model(s_iv, spec, np.linspace(0.0, horizon, 1501), M.method)
        late = []

        def peak(u, smp):
            trj = Md.run({**smp, "u": u})
            i = int(np.argmax(trj["I"]))
            if i == trj.t.size - 1 and trj["I"][-1] > trj["I"][0]:
                late.append(u)                    # still rising at the horizon: the peak is not resolved
            return float(trj["I"][i] / p0["N"])

        def need(smp):                            # least sustained contact reduction meeting the capacity
            if peak(0.0, smp) <= cap:
                return 0.0
            if peak(u_max, smp) > cap:
                return math.inf
            lo, hi = 0.0, u_max
            while hi - lo > 2e-3:
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if peak(mid, smp) <= cap else (mid, hi)
            return hi
        req = [need(smp) for smp in samples]
        u = kth(req)
        p_none = peak(0.0, {})
        if not math.isfinite(u):
            chk("design.feasible", False, f"u <= {u_max:g} cannot keep the peak under {cap:g} in enough draws")
            M.runs += Md.runs
            return {"run": True, "feasible": False, **common}
        nominal_peak = peak(u, {})
        chk("design.feasible", nominal_peak <= cap, f"contact reduction u = {u:.3f} from day {t_on:g} ({how}): "
                                                    f"nominal peak prevalence {nominal_peak:.4f} <= {cap:g} (none: "
                                                    f"{p_none:.4f})", u)
        peaks = np.array([peak(u, smp) for smp in fresh])
        pr = float(np.mean(peaks <= cap))
        chk("reverify.probability", pr >= floor, f"{len(fresh)} fresh parameter draws: the peak stays under {cap:g} "
                                                 f"in {pr:.0%} (target {target:.0%}; >= {floor:.0%} at that count)", pr)
        M.runs += Md.runs
        chk("design.horizon", not late, f"every peak inside the {horizon:g}-day design horizon" if not late else
            f"{len(late)} runs still rising at day {horizon:g}: lengthen design['horizon']")
        return {"run": True, "feasible": True, "u": u, "window": [t_on, t_off], "capacity": cap,
                "u_per_draw_p50": float(np.median(req)), "peak_without": p_none, "peak_nominal": nominal_peak,
                "reverify_probability": pr, "reverify_peak_p95": float(np.quantile(peaks, 0.95)), **common,
                "summary": f"u = {u:.3f} keeps the peak under {cap:g} in {pr:.0%} of {len(fresh)} fresh draws"}
    if spec.kind == "pk":
        from . import standard_models as sm
        cmin, cmax, days = float(dz["c_min"]), float(dz["c_max"]), float(dz["days"])
        pk_keys = ("ka", "CL", "V1", "Q", "V2", "F", "ke0", "E0", "Emax", "EC50", "hill")
        horizon = days * 24.0
        dt = float(dz.get("dt_h", 0.05))
        tg = np.arange(0.0, horizon + dt / 2, dt)
        s_one = sm.pk_two_compartment(**{k_: p0[k_] for k_ in pk_keys}, dose=1.0)

        def single(smp):                      # concentration after one unit dose (linear PK: superpose it)
            trj = M.run({**smp, "dose": 1.0}, t=tg, system=s_one)
            return trj["A_c"] / smp.get("V1", p0["V1"])

        def last_interval(C1, tau):
            n = max(2, int(math.ceil(horizon / tau)))
            tl = np.linspace((n - 1) * tau, n * tau, 121)
            C = sum(np.interp(tl - j * tau, tg, C1, left=0.0, right=0.0) for j in range(n))
            return C, n

        def ss(C1, tau):
            C, _ = last_interval(C1, tau)
            return float(C.min()), float(C.max())
        base = [single(smp) for smp in samples]
        rows, best = [], None
        for tau in map(float, dz["intervals"]):
            per = np.array([ss(C1, tau) for C1 in base])           # trough and peak per unit dose
            lo_d, hi_d = cmin / per[:, 0], cmax / per[:, 1]       # draw i is satisfied for lo_d[i] <= dose <= hi_d[i]
            dose, count = math.inf, 0
            for d in np.sort(lo_d):                               # the least dose that satisfies k draws
                c = int(np.sum((lo_d <= d * (1 + 1e-12)) & (d <= hi_d)))
                if c >= k:
                    dose, count = float(d), c
                    break
            n = max(2, int(math.ceil(horizon / tau)))
            row = {"interval_h": tau, "dose_mg": dose, "daily_mg": dose * 24.0 / tau, "draws_satisfied": count,
                   "n_doses": n, "ok": math.isfinite(dose)}
            rows.append(row)
            if row["ok"] and (best is None or row["daily_mg"] < best["daily_mg"]):
                best = row
        if best is None:
            chk("design.feasible", False, f"no interval in {dz['intervals']} h keeps the trough >= {cmin:g} and the "
                                          f"peak <= {cmax:g} mg/L in {k} of {len(samples)} draws")
            return {"run": True, "feasible": False, "candidates": rows, **common}
        row = best
        tau, dose = row["interval_h"], row["dose_mg"]
        chk("design.feasible", True, f"{dose:.4g} mg every {tau:g} h ({row['daily_mg']:.4g} mg/day, the least of "
                                     f"{len(rows)} intervals; trough >= {cmin:g} and peak <= {cmax:g} mg/L in "
                                     f"{row['draws_satisfied']} of {len(samples)} draws, {how})", row["daily_mg"])
        # superposition against an explicit multiple-dose simulation of the chosen regimen (nominal parameters)
        s_reg = sm.pk_two_compartment(**{k_: p0[k_] for k_ in pk_keys}, dose=dose, interval=tau, n_doses=row["n_doses"])
        tl = np.linspace((row["n_doses"] - 1) * tau, row["n_doses"] * tau, 121)
        explicit = M.run({}, t=np.concatenate([[0.0], tl]), system=s_reg, tight=True)["A_c"][1:] / p0["V1"]
        sup = dose * last_interval(single({}), tau)[0]
        e_sup = float(np.max(np.abs(sup - explicit)) / max(float(np.max(explicit)), 1e-300))
        chk("design.superposition", e_sup < 1e-3, f"the steady state by superposing single doses agrees with the "
                                                  f"explicit {row['n_doses']}-dose simulation to {e_sup:.2g} "
                                                  f"(< 1e-3; linear PK)", e_sup)
        fr = np.array([ss(single(smp), tau) for smp in fresh]) * dose
        ok = float(np.mean((fr[:, 0] >= cmin) & (fr[:, 1] <= cmax)))
        chk("reverify.probability", ok >= floor, f"{len(fresh)} fresh draws: trough >= {cmin:g} and peak <= "
                                                 f"{cmax:g} mg/L in {ok:.0%} (target {target:.0%}; >= {floor:.0%} at "
                                                 f"that count)", ok)
        return {"run": True, "feasible": True, "interval_h": tau, "dose_mg": dose, "daily_mg": row["daily_mg"],
                "candidates": rows, "reverify_probability": ok, "superposition_error": e_sup,
                "reverify_trough_p05": float(np.quantile(fr[:, 0], 0.05)),
                "reverify_peak_p95": float(np.quantile(fr[:, 1], 0.95)), **common,
                "summary": f"{dose:.4g} mg every {tau:g} h meets both limits in {ok:.0%} of {len(fresh)} fresh draws"}
    # excitable: stimulus amplitude that fires with the target probability
    amp_max = float(dz["amp_max"])
    thr = [_threshold(M, amp_max, p=smp, iters=14) for smp in samples]
    amp = kth(thr) * 1.001                    # just above the bisection's bracket
    if not math.isfinite(amp):
        chk("design.feasible", False, f"too many draws do not fire at amplitude {amp_max:g}")
        return {"run": True, "feasible": False, **common}
    chk("design.feasible", amp <= amp_max, f"pulse amplitude {amp:.4g} ({how}; nominal threshold "
                                           f"{_threshold(M, amp_max, iters=14):.4g})", amp)
    fires = np.array([M.qoi({**smp, "amp": amp})["fires"] for smp in fresh])
    pr = float(np.mean(fires))
    chk("reverify.probability", pr >= floor, f"{len(fresh)} fresh draws: the cell fires in {pr:.0%} (target "
                                             f"{target:.0%}; >= {floor:.0%} at that count)", pr)
    return {"run": True, "feasible": True, "amp": amp, "thresholds_p05_p50_p95": np.quantile(thr, [0.05, 0.5, 0.95])
            .tolist(), "reverify_probability": pr, **common,
            "summary": f"amplitude {amp:.4g} fires in {pr:.0%} of {len(fresh)} fresh draws"}
