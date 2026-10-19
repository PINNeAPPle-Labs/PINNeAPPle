"""Manufacturing tolerances of the Prius rotor through the FEA: Monte-Carlo propagation (JCGM 101:2008) of the magnet
remanence, magnet thickness and width, and air gap to the mean torque and torque ripple at 250 A.

The tolerances are stated assumptions (normal, one standard deviation): remanence 1 % (a +-2 % batch spread at two
sigma), magnet thickness and width 0.025 mm and 0.05 mm, air gap 0.025 mm (the rotor radius moves, the stator stays).
Each sample is one nonlinear FEA (480 gap nodes, 12 rotor angles in 60 electrical degrees, MTPA angle of the nominal
machine). Reports mean, standard deviation and percentiles of both outputs, the probability of meeting a stated
specification with its Wilson interval, and the standardised regression coefficients (linear sensitivity of each
output to each input, in standard deviations) with the R^2 of that linear fit.

    python examples/electrodynamics/17_prius_tolerances.py           # 200 samples, about 25 min
    python examples/electrodynamics/17_prius_tolerances.py --quick   # 24 samples, coarse FEA
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from pinneapple_analysis.uncertainty.tolerance import propagate
from pinneapple_simulation.electromagnetics.machines import prius_2004
from pinneapple_simulation.electromagnetics.machines.robustness import (
    TOLERANCE_VARIABLES,
    tolerance_model,
)

OUT = Path(__file__).parent / "results"
NOMINAL = np.array([1.0, 0.0065, 0.0189, 0.00075])
SIGMA = np.array([0.01, 0.025e-3, 0.05e-3, 0.025e-3])
UNITS = ("-", "m", "m", "m")


def src(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Standardised regression coefficients of ``y`` on the columns of ``X`` and the R^2 of the linear fit."""
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)
    A = np.column_stack([np.ones(len(ys)), Xs])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    r2 = 1.0 - np.sum((ys - A @ coef) ** 2) / np.sum(ys ** 2)
    return coef[1:], float(r2)


def main(quick: bool) -> dict:
    t0 = time.perf_counter()
    g = prius_2004()
    fn = tolerance_model(g, gamma_deg=50.3, n_gap=288 if quick else 480, n_positions=6 if quick else 12)
    T0, r0 = fn(NOMINAL[None, :])[0]
    print(f"nominal: torque {T0:.2f} N m, ripple {100 * r0:.1f} %", flush=True)
    spec = (np.array([0.98 * T0, -np.inf]), np.array([np.inf, 0.25]))
    n = 24 if quick else 200
    res = propagate(fn, NOMINAL, SIGMA, n=n, seed=0, limits=spec)
    ok = np.all(np.isfinite(res.outputs), axis=1)
    X, Y = res.inputs[ok], res.outputs[ok]
    out = {"variables": list(TOLERANCE_VARIABLES), "units": list(UNITS), "nominal": NOMINAL.tolist(),
           "sigma": SIGMA.tolist(), "distribution": "normal", "samples": int(n), "valid_samples": int(ok.sum()),
           "fea": {"n_gap": 288 if quick else 480, "rotor_positions": 6 if quick else 12, "I_peak_A": 250.0,
                   "gamma_deg": 50.3},
           "nominal_outputs": {"torque_Nm": float(T0), "ripple": float(r0)},
           "specification": {"torque_min_Nm": float(spec[0][0]), "ripple_max": 0.25},
           "outputs": {}}
    for j, name in enumerate(("torque_Nm", "ripple")):
        y = Y[:, j]
        coef, r2 = src(X, y)
        out["outputs"][name] = {"mean": float(y.mean()), "std": float(y.std(ddof=1)),
                                "std_error_of_mean": float(y.std(ddof=1) / np.sqrt(len(y))),
                                "percentiles": {f"{p:g}": float(v[j]) for p, v in res.percentiles.items()},
                                "p_within_spec": float(res.p_within_each[j]),
                                "src": dict(zip(TOLERANCE_VARIABLES, map(float, coef), strict=True)),
                                "src_r2": r2}
        print(name, json.dumps(out["outputs"][name]), flush=True)
    out["p_within_spec_joint"] = res.p_within
    out["p_within_spec_joint_wilson95"] = list(res.p_within_ci)
    out["samples_inputs"] = X.tolist()
    out["samples_outputs"] = Y.tolist()
    out["fea_calls"] = int(fn.calls)
    out["seconds"] = time.perf_counter() - t0
    OUT.mkdir(exist_ok=True)
    (OUT / "17_prius_tolerances.json").write_text(json.dumps(out, indent=1, default=float))
    _figure(out, X, Y)
    return out


def _figure(out, X, Y):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    T0, r0 = out["nominal_outputs"]["torque_Nm"], out["nominal_outputs"]["ripple"]
    ax[0].hist(Y[:, 0], bins=20, color="C0", alpha=0.8)
    ax[0].axvline(T0, color="k", lw=1, label="nominal")
    ax[0].axvline(out["specification"]["torque_min_Nm"], color="r", ls="--", lw=1, label="spec: >= 98 % of nominal")
    ax[0].set_xlabel("mean torque at 250 A (N m)")
    ax[0].set_ylabel("samples")
    ax[0].legend(fontsize=8)
    ax[1].hist(100 * Y[:, 1], bins=20, color="C1", alpha=0.8)
    ax[1].axvline(100 * r0, color="k", lw=1, label="nominal")
    ax[1].axvline(25, color="r", ls="--", lw=1, label="spec: <= 25 %")
    ax[1].set_xlabel("torque ripple (% pk-pk)")
    ax[1].legend(fontsize=8)
    names = out["variables"]
    w = 0.38
    k = np.arange(len(names))
    for j, (name, c) in enumerate((("torque_Nm", "C0"), ("ripple", "C1"))):
        o = out["outputs"][name]
        ax[2].bar(k + (j - 0.5) * w, [o["src"][v] for v in names], w, color=c,
                  label=f"{name.split('_')[0]} (R^2 {o['src_r2']:.3f})")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xticks(k, ["remanence", "magnet thickness", "magnet width", "air gap"])
    ax[2].set_ylabel("standardised regression coefficient")
    ax[2].legend(fontsize=8)
    fig.text(0.01, 0.01, f"{out['valid_samples']} Monte-Carlo samples (JCGM 101), one FEA each; normal inputs, sigma: "
             "remanence 1 %, thickness 0.025 mm, width 0.05 mm, air gap 0.025 mm (stated assumptions, not measured "
             "tolerances). Joint probability of meeting both specifications "
             f"{out['p_within_spec_joint']:.3f} (Wilson 95 %: {out['p_within_spec_joint_wilson95'][0]:.3f}-"
             f"{out['p_within_spec_joint_wilson95'][1]:.3f}).", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "17_prius_tolerances.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main("--quick" in sys.argv)
