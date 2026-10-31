"""Nonlinear PK/PD inference with uncertainty, and identifiability before the experiment.

    python examples/biology/pkpd_inference.py [--seeds 20]

Model (``pk_one_compartment_mm``): oral dosing, first-order absorption, Michaelis-Menten elimination, effect
compartment and sigmoid Emax. Synthetic subject with known parameters, so the inference can be scored.

1. Design before data: Fisher-information identifiability of (Vmax, Km, V, ka) for a low-dose study (C far below
   Km, elimination nearly linear: only Vmax/Km is expected to be learnable) and for the same study followed by a high
   dose at 24 h (C above Km).
2. Inference on the identifiable design: weighted least squares, then adaptive Metropolis for the posterior
   (flat priors in the bounds); repeated over ``--seeds`` noise draws to measure the coverage of the 95 % credible
   intervals (fraction of draws whose interval contains the true value).
3. PD: E0, Emax, EC50 and ke0 from effect measurements with the PK parameters fixed at their estimates.
4. Minimal PBPK: mass balance and AUC against dose / (CL_ws + CLr) as checks of the model itself.
Writes results/pkpd_inference.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pinneapple_analysis.inverse_problems.estimation import least_squares_fit, metropolis
from pinneapple_analysis.inverse_problems.forward import Observations, Parameter, ParameterSet
from pinneapple_analysis.inverse_problems.identifiability import analyze_identifiability
from pinneapple_analysis.inverse_problems.ode_model import ODEForwardModel
from pinneapple_physics.biology.models import minimal_pbpk, pk_one_compartment_mm

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = {"Vmax": 10.0, "Km": 2.0, "V": 50.0, "ka": 1.0}
PK = list(TRUE)
BOUNDS = {"Vmax": (1.0, 100.0), "Km": (0.1, 20.0), "V": (10.0, 200.0), "ka": (0.1, 10.0)}
T_LOW = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24.0])
T_BOTH = np.concatenate([T_LOW, 24 + T_LOW[:-1], [36.0, 48.0, 60.0, 72.0]])
CV = 0.10            # proportional measurement error (10 %)


def subject(high_dose: bool):
    return pk_one_compartment_mm(dose=10.0, doses=((24.0, 500.0),) if high_dose else (), **TRUE)


def params():
    return [Parameter(k, TRUE[k] * 1.3, *BOUNDS[k], log=True, truth=TRUE[k]) for k in PK]


def design(t, high):
    s = subject(high)
    c = s.simulate(t, max_step=0.25)["C"]
    obs = Observations({"t": t}, CV * c, values=c, observable=np.array(["C"] * len(t), object))
    rep = analyze_identifiability(ODEForwardModel(s, ["C"], PK, max_step=0.25), obs,
                                  [Parameter(k, TRUE[k], *BOUNDS[k], log=True) for k in PK],
                                  profile=False, posterior=False, fit_first=False)
    return {"rank": rep.rank, "condition_number": rep.condition_number, "summary": rep.summary,
            "status": {p["name"]: p["status"] for p in rep.parameters},
            "crb_rel_sd": {p["name"]: p.get("std_rel", p.get("crb_std")) for p in rep.parameters}}


def infer(seed: int, n_mcmc: int):
    s = subject(True)
    c = s.simulate(T_BOTH, max_step=0.25)["C"]
    rng = np.random.default_rng(seed)
    y = c * (1 + CV * rng.standard_normal(c.size))
    obs = Observations({"t": T_BOTH}, CV * np.abs(y), values=y, observable=np.array(["C"] * len(y), object))
    model = ODEForwardModel(s, ["C"], PK, max_step=0.25)
    ps = ParameterSet(params())
    fit = least_squares_fit(model, ps, obs)
    J = fit.jac / obs.sigma[:, None]
    cov = np.linalg.pinv(J.T @ J)
    post = metropolis(model, ps, obs, z0=fit.z, cov0=cov, n_steps=n_mcmc, seed=seed, max_seconds=600)
    summ = post.summary()
    out = {"fit": fit.theta, "chi2_per_dof": fit.chi2 / fit.dof, "posterior": {}}
    for i, k in enumerate(summ["names"]):
        lo, hi = summ["lo"][i], summ["hi"][i]
        out["posterior"][k] = {"median": summ["median"][i], "lo95": lo, "hi95": hi,
                               "covers_truth": bool(lo <= TRUE[k] <= hi)}
    out["acceptance"] = summ["acceptance"]
    return out


def pd_fit(theta_pk: dict):
    s = pk_one_compartment_mm(dose=10.0, doses=((24.0, 500.0),), **theta_pk)
    true_pd = {"E0": 5.0, "Emax": 80.0, "EC50": 3.0, "ke0": 0.3}
    truth = pk_one_compartment_mm(dose=10.0, doses=((24.0, 500.0),), **TRUE, **true_pd, hill=1.0)
    t = T_BOTH
    e = truth.simulate(t, max_step=0.25)["E"]
    rng = np.random.default_rng(123)
    y = e + 2.0 * rng.standard_normal(e.size)
    obs = Observations({"t": t}, 2.0, values=y, observable=np.array(["E"] * len(t), object))
    model = ODEForwardModel(s, ["E"], list(true_pd), max_step=0.25,
                            bounds={"E0": (0.1, 50.0), "Emax": (10.0, 300.0), "EC50": (0.1, 30.0),
                                    "ke0": (0.01, 5.0)})
    ps = ParameterSet([Parameter(k, v * 1.4, *model.bounds[k], log=True) for k, v in true_pd.items()])
    fit = least_squares_fit(model, ps, obs)
    return {"truth": true_pd, "fit": fit.theta, "chi2_per_dof": fit.chi2 / fit.dof,
            "rel_error": {k: abs(fit.theta[k] / true_pd[k] - 1) for k in true_pd}}


def pbpk_checks():
    pb = minimal_pbpk(dose=100.0)
    t = np.linspace(0, 300, 30001)
    r = pb.simulate(t)
    vol = {k: pb.parameters[v] for k, v in pb.meta["volumes"].items()}
    total = sum(vol[k] * r[k] for k in vol) + r["A_elim"]
    auc = float(np.trapezoid(r["C_plasma_art"], t))
    expected = 100.0 / (float(r["CL_ws"][0]) + pb.parameters["CLr"])
    return {"mass_balance_max_rel_error": float(np.max(np.abs(total - 100.0)) / 100.0), "auc": auc,
            "auc_expected_dose_over_cl": expected, "auc_rel_error": abs(auc - expected) / expected}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--mcmc", type=int, default=3000)
    a = ap.parse_args(argv)
    out = {"scale": "full" if a.seeds >= 20 and a.mcmc >= 3000 else
           f"reduced-scale ({a.seeds} noise draws x {a.mcmc} MCMC steps of 20 x 3000)",
           "design_low_dose": design(T_LOW, False), "design_low_then_high": design(T_BOTH, True)}
    print("low dose:", out["design_low_dose"]["summary"])
    print("low + high:", out["design_low_then_high"]["summary"])
    # each finished seed is saved, so a killed run resumes at the next one
    part = os.path.join(HERE, "results", f"pkpd_inference.partial-{a.mcmc}.json")
    runs = json.load(open(part)) if os.path.exists(part) else []
    os.makedirs(os.path.dirname(part), exist_ok=True)
    for s in range(len(runs), a.seeds):
        runs.append(infer(s, a.mcmc))
        with open(part, "w") as f:
            json.dump(runs, f)
    runs = runs[:a.seeds]
    out["inference"] = runs
    out["coverage_95"] = {k: float(np.mean([r["posterior"][k]["covers_truth"] for r in runs])) for k in PK}
    out["median_abs_rel_error"] = {k: float(np.median([abs(r["posterior"][k]["median"] / TRUE[k] - 1)
                                                       for r in runs])) for k in PK}
    print("coverage of 95 % intervals over", a.seeds, "noise draws:", out["coverage_95"])
    print("median |relative error| of posterior medians:", out["median_abs_rel_error"])
    out["pd"] = pd_fit({k: float(np.median([r["posterior"][k]["median"] for r in runs])) for k in PK})
    print("PD:", out["pd"]["rel_error"])
    out["pbpk"] = pbpk_checks()
    print("PBPK:", out["pbpk"])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "pkpd_inference.json"), "w") as f:
        json.dump(out, f, indent=1)
    os.remove(part)
    return out


if __name__ == "__main__":
    main()
