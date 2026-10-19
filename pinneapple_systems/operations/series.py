"""The days 13-31 series end to end on one asset: Campaign Planner -> (runs) -> Surrogate Studio -> Uncertainty ->
Optimizer -> Digital Twin, every hand-over a ``pinneapple_data.contracts`` file.

Each step is the library call the corresponding app makes, so the chain runs whether or not those apps are
deployed; :func:`run_series_flow` reports for every step whether the app exists in this checkout (``apps/<dir>``)
or the step ran as a library stand-in. Numbers come from the run itself: held-out metrics, empirical coverage on a
trajectory the surrogate never saw, and a physics check of the optimum against the closed-form steady state.

Steps
-----
1. **Campaign** (day 13): a ``ParameterSpace`` of operating inputs planned with the Campaign Planner's
   ``pinneapple_data.doe.plan_campaign`` (Latin hypercube, coverage measured), executed as runs of the heated tank
   (:mod:`.plant`), stored as a Parquet table with a ``DatasetSpec``.
2. **Surrogate** (day 14): a controlled operator-inference ROM (:mod:`.rom`) fitted on whole runs, others held out:
   a ``ModelRef`` with held-out metrics and validity ranges.
3. **Uncertainty** (day 15): the Uncertainty Lab's ``pinneapple_analysis.uncertainty.study.uncertainty_study``:
   split-conformal 90 % intervals of the one-step prediction calibrated on two held-out runs, coverage measured on a
   third.
4. **Optimizer** (day 17): Bayesian optimisation (``pinneapple_design.design_optimizer``) of the operating point
   with the surrogate, polished with SLSQP: least heater power that keeps the outlet hot enough *with margin* and
   the level safe; checked against the closed-form optimum of the physical model.
5. **Twin** (day 20): the ROM deployed in :class:`.twin.TwinRuntime` on live data at the optimum; when the inlet
   water leaves the surrogate's training range the twin asks for revalidation and re-scores the model.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pinneapple_analysis.study_models import StudyModel
from pinneapple_analysis.uncertainty.study import uncertainty_study
from pinneapple_data.contracts import (
    Column,
    DatasetRef,
    DatasetSpec,
    FileRef,
    Parameter,
    ParameterSpace,
    Provenance,
    Splits,
    StudyResult,
    Table,
    write_table,
)
from pinneapple_data.doe import plan_campaign

from .contracts import check, load_model, new_id, save_model, twin_study
from .plant import (
    SENSOR_NAMES,
    Scenario,
    SensorNoise,
    TankParams,
    TankSimulator,
    simulate,
    steady_state,
)
from .rom import fit_controlled_opinf, residual_sigma
from .twin import AlertRules, SurrogateModel, TwinRuntime

STEPS = [("campaign", 13, "Simulation Campaign Planner", r"(simulation_)?campaign(_\w+)?"),
         ("surrogate", 14, "Surrogate Studio", r"surrogate(_\w+)?"),
         ("uncertainty", 15, "Engineering Uncertainty Lab", r"uncertainty(_\w+)?"),
         ("optimize", 17, "Optimizer", r"optimi[sz](e|er|ation)(_\w+)?"),
         ("twin", 20, "Digital Twin Runtime", r"(digital_)?twin(_\w+)?")]
LIBRARY = {"campaign": "pinneapple_data.doe.plan_campaign",
           "surrogate": "pinneapple_systems.operations.rom.fit_controlled_opinf",
           "uncertainty": "pinneapple_analysis.uncertainty.study.uncertainty_study",
           "optimize": "pinneapple_design BayesianDesignOptimizer + SciPy SLSQP",
           "twin": "pinneapple_systems.operations.twin.TwinRuntime"}

INPUTS = ["q_in", "Q", "T_in", "T_amb"]
STATES = ["h", "T"]


def operating_space() -> ParameterSpace:
    return ParameterSpace(name="heated-tank operating box", description="inputs a campaign excites the tank with",
                          parameters=[
                              Parameter(name="q_in", low=0.0045, high=0.0085, unit="m3/s", group="operating",
                                        description="pump flow; keeps the level between ~0.6 m and the 2 m overflow"),
                              Parameter(name="Q", low=1.5e5, high=5.5e5, unit="W", group="operating"),
                              Parameter(name="T_in", low=9.0, high=17.0, unit="degC", group="boundary"),
                              Parameter(name="T_amb", low=15.0, high=25.0, unit="degC", group="boundary")],
                          budget=144)


def app_available(pattern: str, apps_dir: Path | None) -> str | None:
    """Name of the ``apps/`` directory of a series app (matched by its whole name), if this checkout has it."""
    if apps_dir is None or not apps_dir.is_dir():
        return None
    for d in sorted(apps_dir.iterdir()):
        if d.is_dir() and re.fullmatch(pattern, d.name.lower()) and (d / "Dockerfile").exists():
            return d.name
    return None


def run_series_flow(workdir: str | Path, *, seed: int = 0, n_runs: int = 12, steps_per_run: int = 12,
                    hold: float = 300.0, dt: float = 20.0, T_min: float = 28.0, q_min: float = 0.006,
                    apps_dir: str | Path | None = None,
                    progress: Callable[[float, str], None] | None = None) -> dict:
    """Run the chain and write every contract under ``workdir``. Returns a JSON-ready summary."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    apps = Path(apps_dir) if apps_dir else None
    prog = progress or (lambda f, m: None)
    p = TankParams()
    out: dict[str, dict] = {}

    # 1. campaign -------------------------------------------------------------------------------------------------
    prog(0.05, "campaign: Latin hypercube over the operating box")
    space = operating_space()
    space.save(wd / "operating.space.json")
    plan = plan_campaign(space, n_runs * steps_per_run, "lhs", seed=seed, name="heated-tank operating campaign")
    cases = plan.cases[space.names].reset_index(drop=True)
    cov = plan.summary()["coverage"]["planned"]
    frames = []
    trajectories = []
    for r in range(n_runs):
        block = cases.iloc[r * steps_per_run:(r + 1) * steps_per_run].reset_index(drop=True)
        rng = np.random.default_rng(seed + 1000 + r)

        def level(col, block=block):
            return lambda t: float(block[col].iloc[min(int(t // hold), len(block) - 1)])

        sc = Scenario(duration=hold * steps_per_run - dt, dt=dt, x0=(rng.uniform(0.8, 1.4), rng.uniform(22, 30)),
                      T_in=level("T_in"), T_amb=level("T_amb"), q_in=level("q_in"), Q=level("Q"))
        tr = simulate(sc, p, noise=SensorNoise({k: 0.0 for k in SENSOR_NAMES}), seed=seed + r)
        trajectories.append(tr)
        import pandas as pd

        frames.append(pd.DataFrame({"run_id": r, "t": tr["t"], **{k: tr[k] for k in STATES + INPUTS}}))
    import pandas as pd

    table = pd.concat(frames, ignore_index=True)
    test_runs = list(range(n_runs - 3, n_runs))
    ds = DatasetSpec(name="heated-tank campaign runs", columns=[
        Column(name="run_id", role="id"), Column(name="t", role="meta", unit="s", description="time")]
        + [Column(name=k, role="output", unit=u) for k, u in (("h", "m"), ("T", "degC"))]
        + [Column(name=k, role="input", unit=u) for k, u in (("q_in", "m3/s"), ("Q", "W"), ("T_in", "degC"),
                                                             ("T_amb", "degC"))],
        n_rows=len(table), parameter_space=space, status="complete",
        splits=Splits(test=test_runs, id_column="run_id", seed=seed),
        provenance=Provenance(app="campaign", source="LHS over the operating box, executed on the lumped tank "
                                                      "model (noise-free)"))
    try:                                    # Parquet when pyarrow is installed, else CSV with its contract sidecar
        import pyarrow  # noqa: F401
        table_name = "campaign_runs.parquet"
    except ImportError:
        table_name = "campaign_runs.csv"
    path = write_table(table, wd / table_name, ds)
    ds.files = [FileRef.of(path, root=wd)]
    ds.save(wd / "campaign_runs.dataset.json")
    camp = StudyResult(id=new_id("campaign"), study_type="campaign", app="campaign", title="Operating campaign",
                       parameter_space=space, dataset=ds, metrics={"n_runs": n_runs, "n_cases": len(cases),
                                "discrepancy": cov.get("discrepancy"), "min_distance": cov.get("min_distance")},
                       checks=[check("cases inside the parameter space", bool(space.feasible(cases).all()),
                                     value=int(space.feasible(cases).sum()), threshold=len(cases),
                                     evidence="operating.space.json")],
                       limits=["noise-free simulated runs; a real campaign adds measurement noise"])
    camp.save(wd / "campaign.study.json")
    out["campaign"] = {"contract": "StudyResult + DatasetSpec", "files": ["operating.space.json",
                       table_name, "campaign_runs.dataset.json", "campaign.study.json"],
                       "summary": f"{n_runs} runs x {steps_per_run} LHS operating points ({len(table)} rows)"}

    # 2. surrogate ------------------------------------------------------------------------------------------------
    prog(0.25, "surrogate: operator inference with inputs, 3 runs held out")
    rom = fit_controlled_opinf(trajectories, STATES, INPUTS, dt, holdout=3)
    ref = save_model(rom, wd / "tank_rom.json", "heated-tank ROM", app="surrogate",
                     dataset=DatasetRef(name=ds.name, version=ds.version, content_hash=ds.content_hash(),
                                        uri="campaign_runs.dataset.json"),
                     trust_decision="REVIEW")
    v = rom.validation
    out["surrogate"] = {"contract": "ModelRef", "files": ["tank_rom.json", "tank_rom.json.modelref.json"],
                        "summary": f"held-out rollout RMSE h {v['rollout_rmse']['h'] * 1000:.2f} mm, "
                                   f"T {v['rollout_rmse']['T']:.3f} K on {v['n_test_trajectories']} unseen runs",
                        "metrics": ref.metrics}

    # 3. uncertainty ----------------------------------------------------------------------------------------------
    prog(0.4, "uncertainty: split-conformal intervals of the ROM, coverage on an unseen run")
    cal, test = trajectories[-3:-1], trajectories[-1:]
    sig = residual_sigma(rom, cal)

    def pairs(trs):
        X = np.concatenate([np.column_stack([tr[n][:-1] for n in STATES + INPUTS]) for tr in trs])
        Y = np.concatenate([np.column_stack([tr[n][1:] for n in STATES]) for tr in trs])
        return X, Y

    smodel = StudyModel(name="heated-tank ROM", kind="rom", input_names=STATES + INPUTS,
                        output_names=[f"{n}_next" for n in STATES],
                        predict_fn=lambda X: rom.step(X[:, :len(STATES)], X[:, len(STATES):]))
    Xc, Yc = pairs(cal)
    Xt, Yt = pairs(test)
    conf = {}
    for j, n in enumerate(STATES):
        r = uncertainty_study(smodel, f"{n}_next", Xc, Yc[:, j], Xt, Yt[:, j], alpha=0.1, seed=seed)
        conf[n] = r["conformal"]
    unc = StudyResult(id=new_id("uncertainty"), study_type="uncertainty", app="uncertainty",
                      title="Split-conformal intervals of the ROM's one-step prediction", models=[ref],
                      metrics={**{f"sigma_{k}": v2 for k, v2 in sig.items()},
                               **{f"conformal_q_{k}": c["q"] for k, c in conf.items()},
                               **{f"coverage_{k}": c["coverage"] for k, c in conf.items()}},
                      checks=[check(f"90 % conformal coverage on {n} (unseen run)", bool(c["meets_target"]),
                                    value=c["coverage"], threshold=c["target_coverage"],
                                    evidence=f"{c['n_test']} one-step predictions on run {n_runs - 1}, "
                                             f"calibrated on {c['n_cal']}")
                              for n, c in conf.items()],
                      limits=["exchangeability between calibration and test runs is assumed",
                              "calibrated on 2 held-out runs only"])
    unc.save(wd / "uncertainty.study.json")
    out["uncertainty"] = {"contract": "StudyResult", "files": ["uncertainty.study.json"],
                          "summary": "90 % split-conformal intervals (pinneapple_analysis.uncertainty.study), "
                                     "coverage on an unseen run: " +
                                     ", ".join(f"{k} {c['coverage']:.1%}" for k, c in conf.items()),
                          "sigma": sig}

    # 4. optimizer ------------------------------------------------------------------------------------------------
    prog(0.55, "optimizer: Bayesian optimisation of the operating point on the surrogate")
    from pinneapple_design.design_optimizer.optimizer import (
        BayesianDesignOptimizer,
        DesignOptimizerConfig,
    )

    T_in_nom, T_amb = 15.0, 20.0
    margin_T = 2 * float(v["rollout_rmse"]["T"]) + 2 * sig["T"]
    bounds = np.array([[q_min, 0.0085], [1.5e5, 5.5e5]])

    def steady(q, Q):
        x = np.array([1.2, 25.0])
        for _ in range(240):
            x = rom.step(x, [q, Q, T_in_nom, T_amb])
        return x

    def objective(z):
        h, T = steady(*z)
        pen = max(0.0, T_min + margin_T - T) * 10.0 + max(0.0, 0.6 - h) * 100 + max(0.0, h - 1.6) * 100
        return z[1] / 1e5 + pen, h, T

    bo = BayesianDesignOptimizer(DesignOptimizerConfig(method="bayesian", n_iters=30, n_initial_random=10),
                                 seed=seed)
    evals = []
    for _ in range(30):
        z = bo.propose(bounds)
        f, h, T = objective(z)
        bo.update(z, f)
        evals.append((float(f), float(z[0]), float(z[1]), float(h), float(T)))
    feas = [e for e in evals if e[4] >= T_min + margin_T and 0.6 <= e[3] <= 1.6]
    best = min(feas or evals)
    # local polish from the best feasible point: SLSQP on the surrogate with the margin as a hard constraint
    sc = np.array([1e-3, 1e5])
    pol = minimize(lambda z: z[1], np.array(best[1:3]) / sc, method="SLSQP", bounds=bounds / sc[:, None],
                   constraints=[{"type": "ineq", "fun": lambda z: steady(*(z * sc))[1] - (T_min + margin_T)},
                                {"type": "ineq", "fun": lambda z: steady(*(z * sc))[0] - 0.6},
                                {"type": "ineq", "fun": lambda z: 1.6 - steady(*(z * sc))[0]}],
                   options={"maxiter": 50, "ftol": 1e-9})
    z_pol = pol.x * sc
    h_pol, T_pol = steady(*z_pol)
    polished = bool(pol.success and T_pol >= T_min + margin_T - 1e-6 and z_pol[1] <= best[2])
    q_opt, Q_opt = (float(z_pol[0]), float(z_pol[1])) if polished else (best[1], best[2])
    phys = steady_state(p, q_opt, Q_opt, T_in_nom, T_amb)
    surr = steady(q_opt, Q_opt)
    # closed-form optimum of the same problem on the physics model: least Q at the lowest flow that meets T_min+margin
    T_req = T_min + margin_T
    Q_star = p.rho * p.cp * q_min * (T_req - T_in_nom) + p.ua * (T_req - T_amb)
    gap = (Q_opt - Q_star) / Q_star
    opt_checks = [check("optimum feasible on the surrogate", bool(feas), value=best[4], threshold=T_min + margin_T,
                        evidence=f"{len(evals)} Bayesian-optimisation evaluations"),
                  check("physics check: closed-form steady state meets the outlet temperature",
                        bool(phys[1] >= T_min), value=float(phys[1]), threshold=T_min,
                        evidence="pinneapple_systems.operations.plant.steady_state"),
                  check("within 5 % of the closed-form physics optimum", bool(gap <= 0.05), value=float(gap),
                        threshold=0.05, evidence=f"Q* = {Q_star / 1e3:.1f} kW from the steady energy balance",
                        warn=True),
                  check("surrogate vs physics at the optimum (T within margin)",
                        bool(abs(surr[1] - phys[1]) <= margin_T), value=float(abs(surr[1] - phys[1])),
                        threshold=margin_T, evidence="ROM rollout to steady state vs closed form")]
    opt = StudyResult(id=new_id("optimize"), study_type="optimization", app="optimize",
                      title="Least heater power for a hot enough outlet", models=[ref],
                      table=Table(columns={"objective": [e[0] for e in evals], "q_in": [e[1] for e in evals],
                                           "Q": [e[2] for e in evals], "h_ss": [e[3] for e in evals],
                                           "T_ss": [e[4] for e in evals]}),
                      metrics={"q_in_opt": q_opt, "Q_opt": Q_opt, "Q_best_bayesian": best[2], "polished": polished,
                               "Q_closed_form_optimum": Q_star, "gap_to_closed_form": gap, "h_ss_surrogate": float(surr[0]),
                               "T_ss_surrogate": float(surr[1]), "h_ss_physics": float(phys[0]),
                               "T_ss_physics": float(phys[1]), "T_margin": margin_T},
                      checks=opt_checks, limits=["steady state at nominal inlet 15 degC; transients not optimised"])
    opt.save(wd / "optimize.study.json")
    out["optimize"] = {"contract": "StudyResult", "files": ["optimize.study.json"],
                       "summary": f"q_in {q_opt * 1000:.2f} L/s, heater {Q_opt / 1000:.0f} kW -> T {surr[1]:.2f} "
                                  f"degC (surrogate) / {phys[1]:.2f} degC (physics), level {phys[0]:.2f} m; "
                                  f"{gap:+.1%} vs the closed-form optimum ({Q_star / 1000:.0f} kW)",
                       "q_in": q_opt, "Q": Q_opt}

    # 5. twin -----------------------------------------------------------------------------------------------------
    prog(0.7, "twin: deploy the ModelRef on live data at the optimum")
    model = SurrogateModel(load_model(ref, wd))
    rules = AlertRules(limits={"h": (0.4, 1.8), "T": (T_min - 1.0, 40.0)}, param_limits={})
    tw = TwinRuntime(model, STATES, {"h": 0.003, "T": 0.05},
                     {"h": max(3 * sig["h"], 2e-4), "T": max(3 * sig["T"], 0.01)}, rules=rules,
                     asset="tank-1", seed=seed)
    sim = TankSimulator(p, phys, seed=seed + 7)
    change_at, n_steps = 2400.0, int(round(5400.0 / dt))
    first_reval = None
    for _k in range(n_steps):
        T_in = T_in_nom if sim.t < change_at else 6.0           # cold inlet water: outside the training range
        u, d = (q_opt, Q_opt), (T_in, T_amb)
        reading = sim.read(u, d)
        reading["t"] = sim.t
        tw.ingest(reading)
        if first_reval is None and "drift" in tw.active:
            first_reval = sim.t
        sim.step(u, d, dt)
    reval = tw.revalidate(n_last=60)
    tstudy = twin_study(tw, app="twin", model_ref=ref, title="ROM twin of tank-1 at the optimised operating point")
    tstudy.save(wd / "twin.study.json")
    out["twin"] = {"contract": "StudyResult", "files": ["twin.study.json"],
                   "summary": (f"{tw.n_updates} updates; inlet water dropped to 6 degC at t={change_at:.0f} s "
                               f"(training range 9-17 degC); revalidation requested at "
                               f"t={first_reval:.0f} s" if first_reval is not None else
                               f"{tw.n_updates} updates; no revalidation requested") +
                              f"; open-loop re-score {'passed' if reval['passed'] else 'failed'}",
                   "revalidation_requested_at": first_reval, "revalidation": reval, "status": tw.status}

    prog(0.95, "done")
    steps = []
    for key, day, title, pattern in STEPS:
        found = app_available(pattern, apps)
        steps.append({"step": key, "day": day, "app": title,
                      "library": LIBRARY[key],
                      "ran_as": f"app on this checkout (apps/{found}); this step calls the same library function, "
                                f"{LIBRARY[key]}" if found else
                                f"library stand-in ({LIBRARY[key]}): the app is not on main yet", **out[key]})
    summary = {"workdir": str(wd), "steps": steps}
    (wd / "series_flow.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
