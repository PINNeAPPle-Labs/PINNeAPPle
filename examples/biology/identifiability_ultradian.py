"""Identifiability of the ultradian glucose-insulin model before any fit (SBINN chapter targets T2 and T3).

    python examples/biology/identifiability_ultradian.py

1. Fisher information from the forward sensitivities of the ODE (not finite differences), with the chapter's
   setting: G observed at 1/5 of the minutes of [0, 1800] chosen at random plus both ends, noise 1 % of std(G),
   17 parameters (all but Vp, Vi, Vg). The chapter (arXiv:2202.01723, Fig 17) finds one eigenvalue close to zero
   whose eigenvector is C2 alone.
2. The scaling invariances of Eq (7) tested as null directions of the Fisher matrix (log parameters): when a
   combination leaves G unchanged, the Fisher matrix restricted to those parameters has a zero eigenvalue whose
   eigenvector is that combination.
3. ``analyze_identifiability`` on the identifiable subset: rank, Cramer-Rao bounds, the most informative extra
   measurement.
"""
from __future__ import annotations

import json
import os

import numpy as np

from pinneapple_analysis.inverse_problems.forward import (
    Observations,
    Parameter,
    ParameterSet,
    jacobian,
)
from pinneapple_analysis.inverse_problems.identifiability import analyze_identifiability
from pinneapple_analysis.inverse_problems.ode_model import ODEForwardModel
from pinneapple_physics.biology.models import ULTRADIAN_SEARCH_RANGE, ultradian_glucose_insulin

HERE = os.path.dirname(os.path.abspath(__file__))
FREE = ["E", "tp", "ti", "td", "k", "Rm", "a1", "C1", "C2", "C4", "C5", "Ub", "U0", "Um", "Rg", "alpha", "beta"]


def chapter_observations(system, seed=0):
    t = np.arange(0, 1800.01, 1.0)
    traj = system.simulate(t, max_step=1.0)
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([[0], rng.choice(np.arange(1, len(t) - 1), len(t) // 5, replace=False),
                                  [len(t) - 1]]))
    sigma = 0.01 * float(np.std(traj["G"]))
    return Observations({"t": t[idx]}, sigma, values=traj["G"][idx], observable=np.array(["G"] * len(idx), object))


def fisher(system, obs, names, bounds=None):
    model = ODEForwardModel(system, ["G"], names, max_step=1.0, bounds=bounds)
    ps = ParameterSet(model.default_parameters())
    _, J, how = jacobian(model, ps, ps.z0(), obs)
    Jw = J / obs.sigma[:, None]
    return Jw.T @ Jw, how


def null_direction(F, names):
    w, V = np.linalg.eigh(F)
    v = V[:, 0] * np.sign(V[np.argmax(np.abs(V[:, 0])), 0])
    return float(w[0] / w[-1]), dict(zip(names, np.round(v, 4).tolist(), strict=True))


def main():
    system = ultradian_glucose_insulin()
    obs = chapter_observations(system)
    out = {}

    # 1. the chapter's practical identifiability (Fisher matrix in physical units, as their Julia code)
    model = ODEForwardModel(system, ["G"], FREE, max_step=1.0, bounds=ULTRADIAN_SEARCH_RANGE)
    ps = ParameterSet(model.default_parameters())
    _, J, how = jacobian(model, ps, ps.z0(), obs)
    theta = ps.theta(ps.z0())
    Jp = J / np.array([theta[n] for n in FREE])[None, :] / obs.sigma[:, None]     # dG/dtheta / sigma
    F = Jp.T @ Jp
    w, V = np.linalg.eigh(F)
    v0 = V[:, 0] * np.sign(V[np.argmax(np.abs(V[:, 0])), 0])
    out["T3"] = {"derivatives": how, "eigenvalues": w.tolist(), "smallest_over_largest": float(w[0] / w[-1]),
                 "second_smallest_over_largest": float(w[1] / w[-1]),
                 "null_vector": dict(zip(FREE, np.round(v0, 4).tolist(), strict=True)),
                 "dominant": FREE[int(np.argmax(np.abs(v0)))]}
    print("T3 eigenvalues (smallest 3, largest):", w[:3], w[-1])
    print("T3 null vector:", {k: v for k, v in out["T3"]["null_vector"].items() if abs(v) > 0.01})

    # 2. scaling invariances of Eq (7), with initial values free where the invariance needs them
    checks = {
        "Rm, Ip(0), Ii(0), C4, C5 (Eq 7a extended to the original system)": ["Rm", "Ip_0", "Ii_0", "C4", "C5"],
        "U0, Um, C3 (Eq 7c)": ["U0", "Um", "C3"],
        "alpha, C5 (Eq 7b; the chapter says it breaks in the original system)": ["alpha", "C5"],
    }
    out["T2_invariances"] = {}
    for label, names in checks.items():
        Fs, _ = fisher(system, obs, names)
        ratio, vec = null_direction(Fs, names)
        out["T2_invariances"][label] = {"smallest_over_largest_eigenvalue": ratio, "null_vector": vec}
        print(f"{label}: lambda_min/lambda_max = {ratio:.2e}, direction {vec}")

    # 3. full analysis on the 16 parameters left once C2 is set aside
    sub = [n for n in FREE if n != "C2"]
    m16 = ODEForwardModel(system, ["G"], sub, max_step=1.0, bounds=ULTRADIAN_SEARCH_RANGE)
    params = [Parameter(p.name, p.value, p.lower, p.upper, log=True) for p in m16.default_parameters()]
    rep = analyze_identifiability(m16, obs, params, profile=False, posterior=False, fit_first=False)
    out["analysis_16"] = {"rank": rep.rank, "n": rep.n_parameters, "condition_number": rep.condition_number,
                          "summary": rep.summary,
                          "status": {p["name"]: p["status"] for p in rep.parameters}}
    print(rep.summary)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "identifiability_ultradian.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
