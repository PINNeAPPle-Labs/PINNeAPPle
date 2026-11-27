"""Physics audit on pipe friction: models judged by their physics, not only by their past errors.

The Darcy friction factor of a rough pipe (relative roughness 1e-3) is predicted for a stream of cases whose Reynolds
number moves through the laminar, transitional and turbulent regimes and back. The truth is Churchill's (1977)
correlation, which covers every regime and includes roughness. The models:

* ``laminar``: f = 64 / Re (Hagen-Poiseuille; assumes laminar flow, Re <= 2300);
* ``blasius``: f = 0.316 Re^-0.25 (smooth-pipe turbulence, 4000 <= Re <= 1e5); it does not know about roughness;
* ``surrogate``: a polynomial in log Re fitted to Churchill on 4000 <= Re <= 2e5 (a stand-in for a learned model,
  with that training envelope);
* ``solver``: the reference itself with 1 % noise, 100 times more expensive (a high-fidelity run).

Three ensembles see the same 160 cases, one at a time, and learn from each case's truth after predicting it:
history only (``PhysicsEnsemble``), history plus the physics-aware router (compatibility of the case with each
model's assumptions and envelope, sleeping experts, escalation to the solver in the transition range), and the
router alone. Then the audit explains two failures: the laminar law at Re = 6000 (regime change) and Blasius at
Re = 5e5 in a rough pipe (outside its envelope, and its smooth-wall equation misses the roughness).

Writes docs/assets/physics_audit/pipe_regimes.png. Runs in a few seconds.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
from pinneapple_physics.physics_audit import (
    Envelope,
    ModelProfile,
    PhysicsAwareRouter,
    PhysicsValidityMonitor,
    default_graph,
    diagnose,
    hydraulically_smooth,
    improvement_plan,
    laminar,
    residual_check,
    roughness_reynolds,
    turbulent,
)
from pinneapple_tools.visualization.style import use_theme

use_theme("report")   # one theme for every figure of the library (#918)

EPS_D = 1e-3                                                     # relative roughness


def churchill(re, eps_d=EPS_D):
    """Churchill (1977): laminar, transitional and turbulent (rough) friction factor in one formula."""
    a = (2.457 * np.log(1.0 / ((7.0 / re) ** 0.9 + 0.27 * eps_d))) ** 16
    b = (37530.0 / re) ** 16
    return 8.0 * ((8.0 / re) ** 12 + 1.0 / (a + b) ** 1.5) ** (1.0 / 12)


def smooth_law_residual(f, re):
    """Residual of the smooth-pipe Prandtl-Karman law 1/sqrt(f) = 2 log10(Re sqrt(f)) - 0.8, relative."""
    lhs = 1.0 / np.sqrt(f)
    return float(abs(lhs - (2.0 * np.log10(re * np.sqrt(f)) - 0.8)) / lhs)


rng = np.random.default_rng(0)
log_train = np.linspace(np.log10(4000), np.log10(2e5), 40)
coef = np.polyfit(log_train, np.log10(churchill(10 ** log_train)), 3)

MODELS = {
    "laminar": lambda q: np.array([64.0 / q["Re"]]),
    "blasius": lambda q: np.array([0.316 * q["Re"] ** -0.25]),
    "surrogate": lambda q: np.array([10 ** np.polyval(coef, np.log10(q["Re"]))]),
    "solver": lambda q: np.array([churchill(q["Re"]) * (1 + 0.01 * rng.standard_normal())]),
}
PROFILES = [
    ModelProfile("laminar", "closed form", ("Hagen-Poiseuille / laminar correlations",), (laminar(2300), "incompressible"),
                 envelope=Envelope({"Re": (1.0, 2300.0)}), cost=1),
    ModelProfile("blasius", "closed form", ("smooth-pipe friction correlations (Blasius, Prandtl-Karman)",),
                 (turbulent(4000), hydraulically_smooth(), "incompressible"), envelope=Envelope({"Re": (4000.0, 1e5)}),
                 cost=1),
    ModelProfile("surrogate", "learned surrogate", (), (turbulent(4000), "incompressible"),
                 mechanisms=("turbulent transport (eddy viscosity)", "wall roughness"),
                 envelope=Envelope({"Re": (4000.0, 2e5)}), cost=1),
    ModelProfile("solver", "high-fidelity solver", ("RANS with eddy viscosity", "Colebrook-White correlation"),
                 ("incompressible",), cost=100),
]

# 160 cases: laminar -> transition -> turbulent -> high-Re rough -> back to laminar
segments = [(300, 2000, 30), (2300, 4000, 20), (5000, 9e4, 40), (1.5e5, 6e5, 30), (400, 1800, 40)]
re_stream = np.concatenate([np.exp(rng.uniform(np.log(a), np.log(b), n)) for a, b, n in segments])


def features(re):
    return {"Re": float(re), "Ma": 0.05, "eps_plus": roughness_reynolds(re, EPS_D)}


cases = [features(r) for r in re_stream]
truth = [np.array([churchill(q["Re"])]) for q in cases]


COST_WEIGHT = 0.03                                               # prefer the cheap models when they are good enough


def make(router):
    experts = [from_callable(n, f, cost=p.cost) for (n, f), p in zip(MODELS.items(), PROFILES, strict=True)]
    return PhysicsEnsemble(experts, mode="select", router=router, explore_every=1, cost_weight=COST_WEIGHT)


def router(strength=1.0):
    return PhysicsAwareRouter(PROFILES, strength=strength, abstain_below=0.05, escalate_below=0.9, escalate_to="solver")


runs = {"history only": make(None).run(cases, truth), "history + physics router": make(router()).run(cases, truth)}
names = list(MODELS)
err_single = {n: [float(abs(MODELS[n](q)[0] - t[0]) / t[0]) for q, t in zip(cases, truth, strict=True)]
              for n in names if n != "solver"}
r_only = router()
chosen_r = [r_only.route(q).chosen for q in cases]
err_router_only = [float(abs(MODELS[c](q)[0] - t[0]) / t[0]) for c, q, t in zip(chosen_r, cases, truth, strict=True)]

print("mean relative error over the 160 cases (and share of cases sent to the 100x solver)")
for n, e in err_single.items():
    print(f"  {n:<26s} {100 * np.mean(e):6.2f} %")
for k, r in runs.items():
    solver_share = np.mean([a == "solver" for a in r.active])
    print(f"  {k:<26s} {100 * np.nanmean(r.ensemble_errors):6.2f} %   solver on {100 * solver_share:.0f} % of cases")
print(f"  {'physics router only':<26s} {100 * np.mean(err_router_only):6.2f} %   "
      f"solver on {100 * np.mean([c == 'solver' for c in chosen_r]):.0f} % of cases")
bounds = np.cumsum([0] + [s[2] for s in segments])
labels = ["laminar", "transition", "turbulent", "high Re, rough", "laminar again"]
print("\nper regime:      " + "".join(f"{lab:>16s}" for lab in labels))
for k, r in runs.items():
    print(f"  {k:<26s}" + "".join(f"{100 * np.nanmean(r.ensemble_errors[a:b]):14.2f} %" for a, b in zip(bounds[:-1], bounds[1:], strict=True)))
for k, r in runs.items():
    i0 = bounds[-2]                                                # first cases back in the laminar regime
    print(f"  {k}: first 5 cases back in laminar flow: "
          + ", ".join(f"{100 * e:.1f} % ({a})" for e, a in zip(r.ensemble_errors[i0:i0 + 5], r.active[i0:i0 + 5], strict=True)))

# ----------------------------------------------------------------------------------------- diagnosis of two failures
mon = PhysicsValidityMonitor([residual_check(lambda p, q: smooth_law_residual(p[0], q["Re"]), 0.03,
                                             name="smooth_pipe_law")])
graph = default_graph(PROFILES)
diagnoses = []
for name, re in [("laminar", 6000.0), ("blasius", 5e5), ("blasius", 6e4)]:
    case = features(re)
    pred = MODELS[name](case)
    err = float(abs(pred[0] - churchill(re)) / churchill(re))
    mon = PhysicsValidityMonitor([residual_check(lambda p, q: smooth_law_residual(p[0], q["Re"]), 0.03,
                                                 name="smooth_pipe_law")]) if name == "blasius" else None
    d = diagnose(PROFILES[names.index(name)], case, prediction=pred, query=case, monitor=mon, error=err, graph=graph)
    diagnoses.append(d)
    print("\n" + d.report())
print("\nimprovement plan over the three cases:")
for r in improvement_plan([(d.attribution, d.recommendations) for d in diagnoses], top=5):
    print("  " + r.line())

# ----------------------------------------------------------------------------------------- figure
out = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "physics_audit")
os.makedirs(out, exist_ok=True)
colors = {"laminar": "#4FB3BF", "blasius": "#E8735A", "surrogate": "#B59BE0", "solver": "#F2B233"}
fig, ax = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
k = np.arange(len(cases))
ax[0].semilogy(k, re_stream, color="0.3", lw=1)
for a, b, lab in [(2300, 4000, "transition")]:
    ax[0].axhspan(a, b, color="0.85", zorder=0)
    ax[0].text(2, 3000, lab, va="center", fontsize=9, color="0.35")
r = runs["history + physics router"]
for n in names:
    m = np.array([a == n for a in r.active])
    ax[0].scatter(k[m], re_stream[m], s=18, color=colors[n], label=f"chosen: {n}", zorder=3)
ax[0].set_ylabel("Reynolds number")
ax[0].legend(ncol=4, fontsize=8, loc="upper left")
ax[0].set_title("Pipe friction factor: the physics router picks the model whose assumptions fit each case")
for key, ls in [("history only", "-"), ("history + physics router", "-")]:
    ax[1].semilogy(k, 100 * np.maximum(runs[key].ensemble_errors, 1e-4), ls, lw=1.4,
                   label=f"{key}: mean {100 * np.nanmean(runs[key].ensemble_errors):.1f} %")
ax[1].set_ylabel("relative error (%), floor 0.01 %")
ax[1].set_xlabel("case")
ax[1].legend(fontsize=8, loc="upper right")
for x in np.cumsum([s[2] for s in segments])[:-1]:
    for a in ax:
        a.axvline(x - 0.5, color="0.6", lw=0.6, ls=":")
fig.tight_layout()
fig.savefig(os.path.join(out, "pipe_regimes.png"), dpi=130)
print("\nwrote docs/assets/physics_audit/pipe_regimes.png")
