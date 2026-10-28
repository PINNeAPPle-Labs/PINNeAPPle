"""The Ahmed-body end-to-end study, stage by stage, on top of :mod:`pinneapple_autolab.study` (the generic stages)
and :mod:`.ahmed` (the simulator). Every function reads the campaign records (``runs`` folder) and writes JSON and
figures into ``res`` (the example folder's ``results``); the example scripts and the Flow Studio nodes call these.

    recs = load_records(runs)
    grid_and_validation(recs, res)            # GCI on the mesh family, CFD vs Ahmed et al. (1984)
    ds = study_dataset(recs)                   # DOE + validation runs, slant > 34 deg held out
    boards = benchmark_all(ds, res)            # families, intervals, physics checks, selection, figures
    plan = next_runs(ds, boards, res)          # Sobol, Pareto picks, acquisition batch + random yardstick
    inv = inverse(ds, recs, res)               # slant from drag: synthetic checks, then the wind-tunnel data
    twin(ds, recs, res, out)                   # the digital-twin page
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from . import ahmed as A

__all__ = ["TARGETS", "OUTPUTS", "BOUNDS", "UNITS", "HOLDOUT_DEG", "load_records", "refresh_fields", "grid_and_validation",
           "study_dataset", "deployment_dataset", "template_integral", "benchmark_all", "next_runs",
           "evaluate_more_runs", "fit_selected", "train_slant_mdn", "inverse", "twin", "flow_scene", "visuals",
           "study_costs", "study_report"]

TARGETS = ["CD", "CL", "CD_pressure", "CD_friction"]
OUTPUTS = TARGETS + ["CD_front", "CD_slant", "CD_base", "CD_roof", "CD_side", "CD_underbody"]
BOUNDS = {v.name: (v.lower, v.upper) for v in A.VARIABLES}
UNITS = {v.name: v.unit for v in A.VARIABLES}
HOLDOUT_DEG = 34.0
PROVENANCE = ("OpenFOAM simpleFoam k-omega SST, Ahmed body half model, L1 mesh with 5 prism layers, "
              "800 iterations (outputs averaged over 400-800)")


def _dump(res: str, name: str, obj: Any) -> str:
    os.makedirs(res, exist_ok=True)
    p = os.path.join(res, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return p


def _fig(res: str, name: str) -> str:
    os.makedirs(os.path.join(res, "figures"), exist_ok=True)
    return os.path.join(res, "figures", name)


def load_records(runs: str, ok_only: bool = False):
    from pinneapple_autolab.study.campaign import load_records as _lr
    return _lr(runs, ok_only=ok_only)


def refresh_fields(recs, runs: str) -> int:
    """Re-sample every successful run's Cp on the current template from its saved flow scene and rewrite its field
    file (runs whose scene is missing keep their file). Returns the number of files rewritten."""
    n = 0
    for r in recs:
        fz = os.path.join(runs, "cases", f"{r.id}.flow.npz")
        if r.ok and r.fields_path and os.path.exists(fz):
            np.savez_compressed(r.fields_path, cp=A.cp_from_flow(r.params, fz))
            n += 1
    return n


# ---------------------------------------------------------------------- 1. grid convergence and validation
def grid_and_validation(recs, res: str, campaign_mesh: str = "L1", tol: float = 0.05) -> dict[str, Any]:
    """GCI of the baseline over the mesh family (records ``mesh_L*``) and the validation sweep (records ``val_*``)
    against :data:`.ahmed.AHMED_1984_CD`; writes ``mesh_study.json``, ``validation.json`` and two figures."""
    from pinneapple_autolab.study import figures as FG
    from pinneapple_autolab.study import mesh_study as M
    by = {r.id: r for r in recs if r.ok}
    levels = [{"name": m, "cells": by[f"mesh_{m}"].meta["cells"], "values": by[f"mesh_{m}"].outputs,
               "cost_s": by[f"mesh_{m}"].cost_s} for m in A.mesh_family() if f"mesh_{m}" in by]
    study: dict[str, Any] = {}
    if len(levels) >= 3:
        study = M.grid_convergence(levels, ["CD", "CL", "CD_pressure", "CD_friction"])
        study["choice"] = M.choose_mesh(study, "CD", tol=tol)
        study["campaign_mesh"] = campaign_mesh
        study["trust"] = {m["name"]: A.run_trust_report(by[f"mesh_{m['name']}"].meta).decisao for m in study["meshes"]}
        study["failed_checks"] = {m["name"]: [c.name for c in A.qa_checks(by[f"mesh_{m['name']}"].meta)
                                              if c.status == "contradicts"] for m in study["meshes"]}
        _dump(res, "mesh_study.json", study)
        FG.grid_convergence(_fig(res, "grid_convergence_cd.png"), study, "CD", measured=A.AHMED_1984_CD[25.0],
                            title="Baseline CD against cell size (25 deg, 60 m/s)")
    names = [m["name"] for m in study.get("meshes", [])]
    l1_err, basis = None, ""
    if campaign_mesh in names:
        i = names.index(campaign_mesh)
        if study["CD"].get("extrapolated") is not None:
            l1_err, basis = study["CD"]["error_vs_extrapolated"][i], "Richardson extrapolation"
        else:
            l1_err, basis = study["CD"]["error_vs_finest"][i], f"finest mesh {names[-1]} (oscillatory: no extrapolation)"
    rows = _val_rows(by, "val_")
    fine = _val_rows(by, "valL2_")
    val = {"source": A.AHMED_1984_SOURCE, "rows": rows, "rows_L2": fine, "campaign_mesh": campaign_mesh,
           "grid_uncertainty_fine_mesh": (study.get("CD") or {}).get("gci_fine"),
           "campaign_mesh_error": l1_err, "campaign_mesh_error_basis": basis}
    _dump(res, "validation.json", val)
    if rows:
        series = {f"OpenFOAM k-omega SST, {campaign_mesh} (campaign)": [(r["slant_deg"], r["CD_cfd"]) for r in rows]}
        if fine:
            series["OpenFOAM k-omega SST, L2"] = [(r["slant_deg"], r["CD_cfd"]) for r in fine]
        FG.validation(_fig(res, "validation_cd.png"), A.AHMED_1984_CD, series, source="Ahmed et al. 1984",
                      title="Drag against slant angle: CFD and wind tunnel")
    return {"mesh_study": study, "validation": val}


def _val_rows(by, prefix):
    rows = []
    for k in sorted(by):
        if not k.startswith(prefix):
            continue
        r = by[k]
        s = r.params["slant_deg"]
        meas = A.AHMED_1984_CD[s]
        rows.append({"slant_deg": s, "CD_cfd": r.outputs["CD"], "CD_measured": meas,
                     "rel_error": (r.outputs["CD"] - meas) / meas, "CL_cfd": r.outputs["CL"],
                     "CD_breakdown": {x: r.outputs[f"CD_{x}"] for x in A.REGIONS} | {"friction": r.outputs["CD_friction"]},
                     "iteration_std": r.meta["iteration_std"]["CD"], "trust": A.run_trust_report(r.meta).decisao,
                     "failed_checks": [c.name for c in A.qa_checks(r.meta) if c.status == "contradicts"],
                     "yplus_p05_p95": [r.meta["yplus"]["p05"], r.meta["yplus"]["p95"]], "cells": r.meta.get("cells")})
    return rows


# ---------------------------------------------------------------------- 2. dataset
def study_dataset(recs):
    """DOE + validation runs (later runs never change the split): slant > :data:`HOLDOUT_DEG` -> extrapolation;
    the validation runs at <= 34 deg -> test; the rest 70 / 15 / 15 train / val / test (seed 0)."""
    from pinneapple_autolab.study import dataset as D
    sim = A.AhmedAero()
    use = [r for r in recs if r.ok and r.purpose in ("doe", "baseline")]
    ds = D.build_dataset(use, [v.name for v in A.VARIABLES], OUTPUTS, field_names=["cp"],
                         points=A.field_vertices(A.BASELINE), regions=sim.regions, units=UNITS, bounds=BOUNDS,
                         provenance={"source": PROVENANCE, "study_id": "ahmed_body_e2e",
                                     "holdout_rule": f"slant_deg > {HOLDOUT_DEG:g} -> extrapolation"})
    in_val = [i for i, p, x in zip(ds.ids, ds.purpose, ds.X, strict=False) if p == "baseline" and x[0] <= HOLDOUT_DEG]
    sp = D.region_split(ds, holdout=lambda X: X[:, 0] > HOLDOUT_DEG, val=0.15, test=0.15, seed=0,
                        fixed={"test": in_val})
    return ds.with_split(sp)


def deployment_dataset(recs):
    """Every successful study run (DOE, validation, acquisition, random, Pareto checks) in ``train``: what the
    twin and the inverse problem on the wind-tunnel data use, after the held-out evaluation is done."""
    from pinneapple_autolab.study import dataset as D
    use = [r for r in recs if r.ok and r.purpose in ("doe", "baseline", "acquisition", "verification")]
    ds = D.build_dataset(use, [v.name for v in A.VARIABLES], OUTPUTS, field_names=["cp"],
                         points=A.field_vertices(A.BASELINE), regions=A.AhmedAero().regions, units=UNITS,
                         bounds=BOUNDS, provenance={"source": PROVENANCE, "study_id": "ahmed_body_e2e_deployment"})
    return ds.with_split({"train": list(ds.ids), "val": [], "test": [], "extrapolation": []})


def template_integral(ds) -> dict[str, float]:
    """How closely the pressure drag integrated from the CFD's Cp on the 1751 template points matches the CFD's
    own pressure drag, on every case of ``ds``; its 95th percentile is the tolerance of the field-scalar
    consistency check (:func:`.ahmed.prediction_checks`), with no added margin."""
    P = [dict(zip(ds.inputs, x, strict=False)) for x in ds.X]
    integ = np.array([A.pressure_drag_from_cp(p, cp) for p, cp in zip(P, ds.fields["cp"], strict=False)])
    gap = np.abs(integ - ds.y("CD_pressure"))
    return {"abs_gap_mean": float(gap.mean()), "abs_gap_p50": float(np.median(gap)), "abs_gap_max": float(gap.max()),
            "abs_gap_p95": float(np.quantile(gap, 0.95)), "signed_gap_mean": float((integ - ds.y("CD_pressure")).mean()),
            "tolerance": float(np.quantile(gap, 0.95)), "n_cases": int(len(gap)),
            "how": "pressure_drag_from_cp of the CFD Cp on the template vs the CFD CD_pressure, every case; the "
                   "tolerance is the 95th percentile (the check compares the same quantile of the surrogate)"}


# ---------------------------------------------------------------------- 3. surrogates
def benchmark_all(ds, res: str, budget_s: float = 30.0, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Scalar families on :data:`TARGETS`, then field families on Cp (the field's physics checks include the
    integral against the selected CD_pressure model); ``benchmark.json`` and the parity / coverage figures."""
    from pinneapple_autolab.study import benchmark as B
    from pinneapple_autolab.study import figures as FG
    integ = template_integral(ds)
    _dump(res, "template_integral.json", integ)
    scal = B.benchmark(ds, TARGETS, None, physics=A.prediction_checks(BOUNDS), log=log)
    cdp = scal.predictor("CD_pressure")
    phys = A.prediction_checks(BOUNDS, cd_pressure=cdp, integral_tol=integ["tolerance"])
    fld = B.benchmark(ds, [], "cp", physics=phys, budget_s=budget_s, log=log)
    sel = {**scal.select()["choice"], **fld.select()["choice"]}
    reasons = {**scal.select()["reasons"], **fld.select()["reasons"]}
    _dump(res, "benchmark.json", {"scalar": scal.to_dict(), "field": fld.to_dict(), "selection": sel,
                                  "reasons": reasons, "rule": B.SELECTION_RULE,
                                  "split": {k: len(v) for k, v in ds.split.items()}})
    te, ex = ds.idx("test"), ds.idx("extrapolation")
    for t in ("CD", "CL"):
        rows = {}
        for f in B.SCALAR_FAMILIES:
            m, h = scal.predictor(t, f)(ds.unit_x()[np.r_[te, ex]])
            rows[f] = (ds.y(t)[np.r_[te, ex]], m, h)
        FG.parity(_fig(res, f"parity_{t}.png"), rows, t, title=f"{t}: test and extrapolation cases")
    cov = {f: {"test": scal.rows[f"CD/{f}"]["test"]["coverage90_conformal"],
               "extrapolation": scal.rows[f"CD/{f}"]["extrapolation"]["coverage90_conformal"]}
           for f in B.SCALAR_FAMILIES}
    FG.calibration(_fig(res, "coverage_cd.png"), cov, title="CD: coverage of the cross-conformal 90 % intervals")
    covf = {f: {"test": fld.rows[f"cp/{f}"]["test"]["coverage90_conformal"],
                "extrapolation": fld.rows[f"cp/{f}"]["extrapolation"]["coverage90_conformal"]}
            for f in B.FIELD_FAMILIES}
    FG.calibration(_fig(res, "coverage_cp.png"), covf, title="Cp: pointwise coverage of the 90 % intervals")
    return {"scalar": scal, "field": fld, "selection": sel}


def _to_full(U):
    """Unit cube of the training region (slant 0-34 deg) -> unit cube of :data:`BOUNDS`."""
    U = np.array(U, float)
    U[:, 0] = U[:, 0] * HOLDOUT_DEG / (BOUNDS["slant_deg"][1] - BOUNDS["slant_deg"][0])
    return U


def next_runs(ds, boards: Mapping[str, Any], res: str, k: int = 8) -> dict[str, Any]:
    """Sobol indices of CD, CL and CD_friction over the training region; the CD-CL Pareto front at 40 m/s
    (calibrated CD half width <= 0.02) with three picks; ``k`` designs where the GP of CD is most uncertain and
    ``k`` random ones. Writes ``sobol.json``, ``pareto_plan.json`` and ``next_runs.json``."""
    from pinneapple_autolab.study import extras as E
    from pinneapple_autolab.study import figures as FG
    scal = boards["scalar"]
    pred = {t: scal.predictor(t) for t in TARGETS}

    def predict(U):
        Uf = _to_full(U)
        return {t: pred[t](Uf) for t in TARGETS}
    train_b = {**BOUNDS, "slant_deg": (BOUNDS["slant_deg"][0], HOLDOUT_DEG)}
    S = E.sobol_on_surrogate(predict, train_b, ["CD", "CL", "CD_friction"], n=4096)
    _dump(res, "sobol.json", S)
    FG.sobol(_fig(res, "sobol_cd.png"), S, "CD", title="What drives CD (Sobol indices on the surrogate)")
    par = E.pareto_on_surrogate(predict, train_b, [("CD", "min"), ("CL", "min")], fixed={"speed": 40.0},
                                max_half_width={"CD": 0.02}, budget=3000)
    _dump(res, "pareto_plan.json", par)
    gp, scale = scal.models["CD/gp"], scal.rows["CD/gp"]["scale"]
    done = ds.unit_x()[np.r_[ds.idx("train"), ds.idx("val")]].copy()
    done[:, 0] *= (BOUNDS["slant_deg"][1] - BOUNDS["slant_deg"][0]) / HOLDOUT_DEG
    al = E.acquire(lambda U: gp.predict(_to_full(U))[1] / scale, train_b, done, k, min_dist=0.15, seed=1)
    rnd = E.random_points(train_b, k, seed=7)
    plan = {"acquisition": al, "random": rnd, "pareto": par["picks"],
            "rule": "GP predictive std of CD over its scale on a Sobol pool, picks >= 0.15 apart (unit cube)"}
    _dump(res, "next_runs.json", plan)
    return {"sobol": S, "pareto": par, "plan": plan}


def evaluate_more_runs(ds, recs, res: str) -> dict[str, Any]:
    """CD (GP) and Cp (POD-GP) errors on the untouched test split when the DOE training set is extended by the
    acquired runs or by the random ones; the Pareto picks predicted against simulated. ``more_cfd.json`` + figures."""
    from pinneapple_autolab.study import benchmark as B
    from pinneapple_autolab.study import dataset as D
    from pinneapple_autolab.study import figures as FG
    te = ds.idx("test")

    def errors(d):
        fit = np.r_[d.idx("train"), d.idx("val")]
        U = d.unit_x()
        gp = B.make_scalar("gp").fit(U[fit], d.y("CD")[fit])
        m, _ = gp.predict(ds.unit_x()[te])
        fm = B.make_field("pod_gp").fit(U[fit], d.fields["cp"][fit])
        M, _ = fm.predict(ds.unit_x()[te])
        Y = ds.fields["cp"][te]
        return {"n_fit": int(len(fit)), "CD_rmse": float(np.sqrt(np.mean((m - ds.y("CD")[te]) ** 2))),
                "cp_rel_l2": float(np.mean(np.linalg.norm(M - Y, axis=1) / np.linalg.norm(Y, axis=1)))}
    out = {"doe": errors(ds), "test_cases": int(len(te))}
    for pre, lab in (("al_", "doe + acquired"), ("rand_", "doe + random")):
        rr = [r for r in recs if r.ok and r.id.startswith(pre)]
        if rr:
            ex = D.build_dataset(rr, ds.inputs, ds.outputs, field_names=["cp"], points=ds.points, regions=ds.regions,
                                 units=ds.units, bounds=ds.bounds)
            out[lab] = errors(ds.extend(ex, "train"))
    plan = json.load(open(os.path.join(res, "next_runs.json")))
    by = {r.id: r for r in recs if r.ok}
    ver = []
    for i, p in enumerate(plan["pareto"]):
        r = by.get(f"pareto_{i:02d}")
        ver.append({"params": {v.name: p[v.name] for v in A.VARIABLES}, "predicted": p["predicted"],
                    "cfd": {k: r.outputs[k] for k in ("CD", "CL")} if r else None})
    out["pareto_verification"] = ver
    _dump(res, "more_cfd.json", out)
    n0 = out["doe"]["n_fit"]
    curves = {lab: [(n0, out["doe"]["CD_rmse"]), (out[lab]["n_fit"], out[lab]["CD_rmse"])]
              for lab in ("doe + acquired", "doe + random") if lab in out}
    if curves:
        FG.error_drop(_fig(res, "active_learning_cd.png"), curves, "CD RMSE on the test split (GP)",
                      title="Eight more CFD runs: chosen by uncertainty or at random")
    par = json.load(open(os.path.join(res, "pareto_plan.json")))
    FG.pareto(_fig(res, "pareto.png"), par["front"], ["CD", "CL"], ver,
              title="CD against CL at 40 m/s: surrogate front and CFD checks")
    return out


# ---------------------------------------------------------------------- 4. inverse
def fit_selected(ds, target: str, family: str, fit_idx=None):
    """The ``family`` model of ``target`` fitted on ``fit_idx`` (default: train + val) with its cross-conformal q
    (5-fold CV on the same cases): ``predict(U) -> (mean, 90 % half width)``."""
    from pinneapple_autolab.study import benchmark as B
    idx = np.r_[ds.idx("train"), ds.idx("val")] if fit_idx is None else np.asarray(fit_idx)
    sub = ds.with_split({"train": [ds.ids[i] for i in idx], "val": [], "test": [], "extrapolation": []})
    if target in sub.outputs:
        board = B.benchmark(sub, [target], None, scalar_families=(family,), log=lambda s: None)
    else:
        board = B.benchmark(sub, [], target, field_families=(family,), log=lambda s: None)
    return board.predictor(target, family)


def train_slant_mdn(predict, sigma_d: float, *, n_sims: int = 40000, budget_s: float = 60.0, seed: int = 0):
    """MDN posterior p(slant, clearance, speed | CD, clearance, speed) (unit cube) from the CD surrogate: each
    draw's CD gets the surrogate's 1-sigma (half width / 1.645) and the model discrepancy ``sigma_d``."""
    from pinneapple_analysis.inverse_problems.amortized import fit_mdn

    def simulate(Z):
        m, h = predict(Z)
        return np.c_[m, h, Z[:, 1], Z[:, 2]]

    def noise(Y, rng):
        sd = np.sqrt((Y[:, 1] / 1.645) ** 2 + sigma_d ** 2)
        return np.c_[Y[:, 0] + sd * rng.standard_normal(len(Y)), Y[:, 2], Y[:, 3]]
    return fit_mdn(simulate, 3, noise=noise, n_sims=n_sims, K=6, budget_s=budget_s, seed=seed)


def _obs(cd, c, U):
    u = A.params_to_unit({"slant_deg": 0.0, "clearance": c, "speed": U})[0]
    return np.array([cd, u[1], u[2]])


def _slant(mdn, cd, c, U, n=4000, seed=0):
    from pinneapple_analysis.inverse_problems.amortized import hdi
    lo, hi = BOUNDS["slant_deg"]
    S = mdn.sample(_obs(cd, c, U), n, seed)[:, 0]
    return lo + S * (hi - lo), [[lo + a * (hi - lo), lo + b * (hi - lo)] for a, b in hdi(S, 0.9)]


def _baselines(predict, cd, c, U, sigma):
    from pinneapple_analysis.inverse_problems.estimation import least_squares_fit, metropolis
    from pinneapple_analysis.inverse_problems.forward import (
        CallableModel,
        Observations,
        Parameter,
        ParameterSet,
    )
    lo, hi = BOUNDS["slant_deg"]
    u = A.params_to_unit({"slant_deg": 0.0, "clearance": c, "speed": U})[0]
    fn = lambda th, inp: predict(np.array([[(th["slant_deg"] - lo) / (hi - lo), u[1], u[2]]]))[0]  # noqa: E731
    model = CallableModel(fn, inputs=["case"], outputs=["CD"])
    obs = Observations({"case": np.zeros(1)}, sigma=np.array([sigma]), values=np.array([cd]))
    ls = []
    for s0 in (2.0, 10.0, 20.0, 30.0, 38.0):
        f = least_squares_fit(model, ParameterSet([Parameter("slant_deg", s0, lo, hi, log=False)]), obs)
        ls.append({"start": s0, "slant_deg": float(f.theta["slant_deg"]), "chi2": float(f.chi2)})
    best = min(ls, key=lambda r: r["chi2"])
    ps = ParameterSet([Parameter("slant_deg", best["slant_deg"], lo, hi, log=False)])
    mc = metropolis(model, ps, obs, z0=ps.z0(), n_steps=6000, max_seconds=30, seed=1)
    return ls, mc


def inverse(ds, recs, res: str, family: str | None = None, budget_s: float = 60.0,
            measurements: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Synthetic checks first (calibration on 300 surrogate draws; the slant of every held-out CFD case from its
    CFD drag, with the surrogate fitted on train + val only), then every wind-tunnel data set in ``measurements``
    (default :data:`.ahmed.MEASUREMENTS`) with the deployment surrogate at the set's speed and clearance; MDN
    against least squares and Metropolis. The model discrepancy ``sigma_d`` is the RMS of CFD - measurement over the
    validation sweep (``validation.json``); for the set that sweep belongs to it is leave-one-out (a measurement
    never calibrates its own inversion). A set without drag values is reported as not measured here, with the
    reason. Writes ``inverse.json`` and the posterior figures."""
    from pinneapple_analysis.inverse_problems.amortized import posterior_coverage
    from pinneapple_autolab.study import figures as FG
    t0 = time.time()
    family = family or json.load(open(os.path.join(res, "benchmark.json")))["selection"]["CD"]
    val = json.load(open(os.path.join(res, "validation.json")))["rows"]
    d_all = np.array([r["CD_cfd"] - r["CD_measured"] for r in val])
    sigma_all = float(np.sqrt(np.mean(d_all ** 2)))
    pred_h = fit_selected(ds, "CD", family)
    mdn = train_slant_mdn(pred_h, sigma_all, budget_s=budget_s)
    rng = np.random.default_rng(11)
    Z = rng.random((300, 3))
    m, h = pred_h(Z)
    Yd = np.c_[m + np.sqrt((h / 1.645) ** 2 + sigma_all ** 2) * rng.standard_normal(len(Z)), Z[:, 1], Z[:, 2]]
    cal = posterior_coverage(mdn, Z, Yd)
    held = []
    test_set = set(ds.idx("test").tolist())
    for i in np.r_[ds.idx("test"), ds.idx("extrapolation")]:
        p = dict(zip(ds.inputs, ds.X[i], strict=False))
        deg, iv = _slant(mdn, ds.y("CD")[i], p["clearance"], p["speed"])
        held.append({"id": ds.ids[i], **p, "CD": float(ds.y("CD")[i]), "median": float(np.median(deg)),
                     "hdi90": iv, "inside": any(a <= p["slant_deg"] <= b for a, b in iv),
                     "split": "test" if int(i) in test_set else "extrapolation"})
    dep = deployment_dataset(recs)
    pred_d = fit_selected(dep, "CD", family, np.arange(len(dep)))
    sets_out: dict[str, Any] = {}
    posts: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    grid = np.linspace(0, 1, 401)
    val_slants = {r["slant_deg"] for r in val}
    for name, ms in (measurements or A.MEASUREMENTS).items():
        entry: dict[str, Any] = {"label": ms["label"], "speed": ms["speed"], "clearance": ms["clearance"],
                                 "source": ms["source"], "verification": ms.get("verification", {})}
        if not ms.get("CD"):
            entry["status"] = "not measured here: the set carries no drag values (see verification)"
            entry["rows"] = []
            sets_out[name] = entry
            continue
        c, U = float(ms["clearance"]), float(ms["speed"])
        own_sweep = all(float(s) in val_slants for s in ms["CD"]) and U == 60.0 and c == 0.05
        u_c = A.params_to_unit({"slant_deg": 0, "clearance": c, "speed": U})[0]
        mg, hg = pred_d(np.c_[grid, np.full_like(grid, u_c[1]), np.full_like(grid, u_c[2])])
        rows = []
        for k, (s, cd) in enumerate(sorted((float(a), float(b)) for a, b in ms["CD"].items())):
            if own_sweep:
                j = [r["slant_deg"] for r in val].index(s)
                sig = float(np.sqrt(np.mean(np.delete(d_all, j) ** 2)))
            else:
                sig = sigma_all
            mdn_k = train_slant_mdn(pred_d, sig, seed=k, budget_s=0.75 * budget_s)
            deg, iv = _slant(mdn_k, cd, c, U)
            ls, mc = _baselines(pred_d, cd, c, U, sig)
            rows.append({"slant_deg": s, "CD_measured": cd, "sigma_d": sig,
                         "sigma_d_basis": "leave-one-out over the validation sweep" if own_sweep else
                         "RMS of CFD - measurement over the 60 m/s validation sweep (no CFD run at these conditions "
                         "was validated against this set)",
                         "mdn_median": float(np.median(deg)), "mdn_hdi90": iv,
                         "inside": any(a <= s <= b for a, b in iv),
                         "least_squares": sorted({round(x["slant_deg"], 2) for x in ls}),
                         "mcmc": {"median": float(np.median(mc.samples[:, 0])),
                                  "lo90": float(np.quantile(mc.samples[:, 0], 0.05)),
                                  "hi90": float(np.quantile(mc.samples[:, 0], 0.95)), "acceptance": mc.acceptance},
                         "measurement_within_model_reach": bool(np.min(mg - hg) - 2 * sig <= cd <= np.max(mg + hg) + 2 * sig),
                         "surrogate_CD_range": [float(mg.min()), float(mg.max())]})
            posts[(name, s)] = (deg, mc.samples[:, 0])
        entry.update(status="measured", rows=rows, inside_hdi90=float(np.mean([r["inside"] for r in rows])))
        sets_out[name] = entry
    main = sets_out.get("ahmed_1984", next(iter(sets_out.values()), {"rows": []}))
    out = {"selected_CD_family": family, "sigma_d_all": sigma_all, "calibration_on_draws": cal,
           "heldout_cfd": held, "heldout_inside_hdi90": float(np.mean([r["inside"] for r in held])) if held else None,
           "measurements": main.get("rows", []), "measurement_sets": sets_out, "mdn": mdn.info,
           "seconds": time.time() - t0}
    _dump(res, "inverse.json", out)
    for (name, s), (deg, mcs) in posts.items():
        if s in (25.0, 30.0):
            tag = "" if name == "ahmed_1984" else f"_{name}"
            FG.posterior(_fig(res, f"posterior_slant_{s:g}{tag}.png"),
                         {"MDN (amortised)": deg, "Metropolis": mcs}, truth=s, xlabel="slant angle [deg]",
                         title=f"Slant inferred from the CD measured at {s:g} deg ({sets_out[name]['label']})")
    return out


# ---------------------------------------------------------------------- 5. twin
def twin(recs, res: str, out_dir: str, api: str = "", grid: Sequence[int] = (21, 6, 9)) -> str:
    """The digital-twin page from the deployment models (the selected families refitted on every run)."""
    from pinneapple_autolab.study import twin as T
    sel = json.load(open(os.path.join(res, "benchmark.json")))["selection"]
    dep = deployment_dataset(recs)
    allx = np.arange(len(dep))
    preds = {t: fit_selected(dep, t, sel[t], allx) for t in ("CD", "CL")}
    fpred = fit_selected(dep, "cp", sel["cp"], allx)
    _, F, j = A.template_faces(A.BASELINE)
    Vh = A.field_vertices(A.BASELINE)
    # the page mirrors the half template; give it the full closed body's triangles on [half, mirrored half]
    mirrored = np.where(A.make_body(A.BASELINE, dense=True)[0][:, 1] < -1e-9, j + len(Vh), j)
    pay = T.twin_payload(lambda U: {t: preds[t](U) for t in preds}, fpred, BOUNDS,
                         lambda p: A.field_vertices(p), mirrored[F], dep.unit_x(), grid=grid,
                         shape_inputs=("slant_deg", "clearance"), units=UNITS, max_distance=0.25,
                         title="Ahmed body: surrogate twin",
                         scalar_labels={"CD": "Drag coefficient CD", "CL": "Lift coefficient CL"},
                         meta={"subtitle": f"OpenFOAM k-omega SST campaign ({len(dep)} runs), selected models "
                                           f"{sel['CD']} (CD), {sel['CL']} (CL), {sel['cp']} (Cp)",
                               "baseline": A.BASELINE, "chart_output": "CD",
                               "note": "Bands: cross-conformal 90 % intervals of the selected models."})
    return T.write_twin(pay, out_dir, api=api)


# ---------------------------------------------------------------------- 6. visuals
def flow_scene(runs: str, rid: str, params: Mapping[str, float], every_line: int = 2):
    """The studio scene of a run: the dense body with the CFD's skin Cp and Cf, every ``every_line``-th
    streamline (thin tubes), the mid-plane and wake slices (half models mirrored)."""
    from pinneapple_simulation.numerical_solvers.external_flow import FlowResult
    from pinneapple_tools.visualization.studio.scene import Scene, Surface
    r = FlowResult.load(os.path.join(runs, "cases", f"{rid}.flow.npz"))
    V, F = A.make_body(params, dense=True)
    sc = r.to_scene(Scene([Surface("ahmed", V, F, "paint")], title=r.info.get("title", "Ahmed body")))
    for L in sc.lines:
        L.points, L.values = L.points[::every_line], L.values[::every_line]
        L.radius, L.alpha = 0.0025, 0.9
    return sc


NARRATION = {
    "intro": "The Ahmed body: the simplified car that has tested aerodynamics codes since 1984. This is an "
             "end-to-end study in PINNeAPPle, from geometry to a digital twin, and back to the wind tunnel.",
    "cfd": "Fifty-eight OpenFOAM runs on a moving road: a mesh family for grid convergence, a validation sweep "
           "against the measured drag curve, and a space-filling design of experiments over the slant angle, "
           "the ground clearance and the speed.",
    "slices": "Steady RANS with k-omega SST resolves the attached flow and the wake, but not the drag peak at the "
              "critical slant angle. The report says by how much.",
    "surrogate": "Surrogate families are trained on the same data and budget, with calibrated uncertainty and "
                 "physics checks on their predictions. The selected one drives the digital twin: here the slant "
                 "sweeps from zero to forty degrees through the surrogate alone.",
    "outro": "Every number in the report was measured. The twin shows its uncertainty band, warns outside the "
             "region the CFD covered, and can run a verifying case at the click of a button.",
}


def visuals(recs, runs: str, res: str, *, rid: str = "val_25.0", samples: int = 64, size=(1600, 900),
            video: bool = True, frames_orbit: int = 48, frames_sweep: int = 41, fps: int = 12, narrate: bool = True,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """Blender Cycles stills of one run on a dark product stage (Cp on the body with streamlines from the front and
    from the rear, mid-plane speed, wake vorticity), a short film in English (title card, an orbit of that CFD run,
    the two slices, the slant swept 0-40 deg through the deployment Cp surrogate, an end card with the measured
    numbers; captions on every shot and, with ``narrate``, an eSpeak NG voice-over whose cues set the cut lengths),
    the three.js viewer and an OpenUSD stage with the run's trust decision. Writes ``visuals.json``."""
    import shutil
    import tempfile

    import pinneapple as pp
    from pinneapple_tools.visualization.studio import captions as CP
    by = {r.id: r for r in recs if r.ok}
    p = by[rid].params
    sc = flow_scene(runs, rid, p)
    out_dir = os.path.join(res, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    done: dict[str, Any] = {}
    t0 = time.time()
    kw = dict(samples=samples, size=size, background="dark", floor=True)
    shot = f"OpenFOAM k-omega SST: slant {p['slant_deg']:g} deg, {p['speed']:g} m/s"
    stills = {"cp_streamlines_front.jpg": (dict(field="cp", field_range=(-1.0, 1.0), cmap="rdbu", lines=True, view="iso"),
                                           "surface pressure coefficient Cp, streamlines coloured by speed"),
              "cp_streamlines_rear.jpg": (dict(field="cp", field_range=(-1.0, 1.0), cmap="rdbu", lines=True, view="iso_back"),
                                          "the slant and the base: Cp and the wake streamlines"),
              "mid_plane_speed.jpg": (dict(slice="mid plane: speed", view="side", floor=False), "speed on the symmetry plane"),
              "wake_vorticity.jpg": (dict(slice="wake: vorticity", view="back", floor=False), "vorticity in the wake plane")}
    for name, (o, sub) in stills.items():
        f = pp.viz.render(sc, os.path.join(out_dir, name), **{**kw, **o})
        CP.caption(f, shot, sub=sub, footer="PINNeAPPle, Blender Cycles")
        log(f"rendered {name}")
    done["stills"] = sorted(stills)
    done["stills_s"] = time.time() - t0
    pp.viz.web_viewer(sc, os.path.join(out_dir, "viewer"))
    try:
        from pinneapple_tools.nvidia.omniverse import FieldStage, validate_stage
        st = FieldStage(os.path.join(out_dir, "ahmed_body.usda"), "Ahmed body, OpenFOAM k-omega SST")
        surf = sc.surfaces[0]
        st.add_mesh("ahmed", surf.vertices, surf.faces, fields={"cp": surf.fields["cp"]}, color_by="cp",
                    cmap="coolwarm", vrange=(-1.0, 1.0), units={"cp": "1"})
        st.add_curves("streamlines", sc.lines[0].points, width=0.004, fields={"speed": sc.lines[0].values},
                      color_by="speed", cmap="viridis")
        rep = A.run_trust_report(by[rid].meta)
        st.set_trust("/World/ahmed", rep.decisao, [{"name": c.name, "status": c.status} for c in rep.checks])
        st.save()
        done["usd"] = {"path": "visuals/ahmed_body.usda", "issues": validate_stage(st.path)}
    except Exception as e:  # noqa: BLE001 - USD needs usd-core; report instead of failing the stage
        done["usd"] = {"error": f"{type(e).__name__}: {e}"}
    if video:
        from pinneapple_tools.visualization.studio.blender import _encode
        from pinneapple_tools.visualization.studio.scene import Scene, Surface
        tmp = tempfile.mkdtemp()
        vsize = (1280, 720)
        lo_kw = dict(samples=max(16, samples // 3), size=vsize, background="dark", floor=True)
        segments: list[tuple[str, list[str]]] = []                 # (narration key, frames)
        b = _load(res, "benchmark.json") or {}
        costs = _load(res, "costs.json") or {}
        n_runs = len([r for r in recs if r.ok])
        cpu_h = costs.get("total", {}).get("cpu_h")
        head = CP.title_card(os.path.join(tmp, "intro.png"), vsize, "Ahmed body",
                             ["From CFD to a digital twin,", "and back to the wind tunnel."],
                             kicker="PINNeAPPle end-to-end study",
                             footer=f"OpenFOAM k-omega SST, {n_runs} runs" + (f", {cpu_h:.1f} CPU-h" if cpu_h else ""))
        segments.append(("intro", [head] * (3 * fps)))
        orbit = []
        for k in range(frames_orbit):
            a = 2 * np.pi * k / frames_orbit - 2.3
            f = os.path.join(tmp, f"o{k:03d}.png")
            pp.viz.render(sc, f, field="cp", field_range=(-1.0, 1.0), cmap="rdbu", lines=True,
                          view=(np.cos(a), np.sin(a), 0.55), **lo_kw)
            CP.caption(f, shot, sub="CFD: surface pressure coefficient Cp, streamlines by speed",
                       progress=(k + 1) / frames_orbit, footer="validation run, L1 mesh")
            orbit.append(f)
        segments.append(("cfd", orbit))
        sl = []
        for name, (o, sub) in list(stills.items())[2:]:
            f = os.path.join(tmp, "sl_" + name.replace(".jpg", ".png"))
            pp.viz.render(sc, f, **{**lo_kw, **o})
            CP.caption(f, shot, sub=sub, footer="validation run, L1 mesh")
            sl += [f] * int(2.5 * fps)
        segments.append(("slices", sl))
        sel = b.get("selection", {})
        sweep = []
        if sel.get("cp"):
            dep = deployment_dataset(recs)
            fpred = fit_selected(dep, "cp", sel["cp"], np.arange(len(dep)))
            for k, s in enumerate(np.linspace(0.0, 40.0, frames_sweep)):
                q = {**p, "slant_deg": float(s)}
                cp, hw = fpred(A.params_to_unit(q))
                V, F, jj = A.template_faces(q)
                sc2 = Scene([Surface("ahmed", V, F, "paint", fields={"cp": cp[0][jj]})], title="surrogate")
                f = os.path.join(tmp, f"s{k:03d}.png")
                pp.viz.render(sc2, f, field="cp", field_range=(-1.0, 1.0), cmap="rdbu", view="iso_back", **lo_kw)
                CP.caption(f, f"Surrogate ({sel['cp']}): slant {s:.0f} deg, {p['speed']:g} m/s",
                           sub=f"predicted Cp; mean 90 % half width {float(np.mean(hw)):.3f}",
                           progress=(k + 1) / frames_sweep, footer="no CFD in this shot")
                sweep.append(f)
            sweep += [sweep[-1]] * fps
        segments.append(("surrogate", sweep))
        lines = []
        for t in ("CD", "CL"):
            f = sel.get(t)
            r = b.get("scalar", {}).get("rows", {}).get(f"{t}/{f}") if f else None
            if r and r.get("test"):
                lines.append(f"{t} ({f}): test RMSE {r['test']['rmse']:.4f}, 90 % band covers {r['test']['coverage90_conformal']:.0%}")
        f = sel.get("cp")
        r = b.get("field", {}).get("rows", {}).get(f"cp/{f}") if f else None
        if r and r.get("test"):
            lines.append(f"Cp ({f}): test rel. L2 {r['test']['rel_l2_mean']:.3f}, coverage {r['test']['coverage90_conformal']:.0%}")
        tail = CP.title_card(os.path.join(tmp, "outro.png"), vsize, "Measured, not promised",
                             lines or ["see results/report.html"], kicker="what the study found",
                             footer="results/report.html, apps/ahmed_twin")
        segments.append(("outro", [tail] * (4 * fps)))
        cues, frames, t_seg, t_voice = [], [], 0.0, 0.0
        narrated = False
        if narrate:
            try:
                # each line starts at its shot (or when the previous line has ended) and may run into the next
                # shot; a shot is held only when its line could not start inside it, and the last one until the
                # voice ends
                for k, (key, segf) in enumerate(segments):
                    wav = CP.narrate(NARRATION[key], os.path.join(tmp, f"{key}.wav"))
                    dur = CP.audio_duration(wav)
                    start = max(t_seg + 0.3, t_voice + 0.4)
                    end_needed = start + (dur + 0.6 if k == len(segments) - 1 else 1.0)
                    need = int(np.ceil((end_needed - t_seg) * fps))
                    if segf and len(segf) < need:
                        segf += [segf[-1]] * (need - len(segf))
                    cues.append((start, wav))
                    t_voice = start + dur
                    frames += segf
                    t_seg += len(segf) / fps
                narrated = True
            except RuntimeError as e:
                log(f"no narration: {e}")
                frames = [f for _, segf in segments for f in segf]
        else:
            frames = [f for _, segf in segments for f in segf]
        silent = os.path.join(out_dir, "ahmed_body_silent.mp4" if narrated else "ahmed_body.mp4")
        _encode(frames, silent, fps)
        if narrated:
            CP.mux_audio(silent, cues, os.path.join(out_dir, "ahmed_body.mp4"))
            os.remove(silent)
        _encode(frames[::2], os.path.join(out_dir, "ahmed_body.gif"), fps / 2, gif=True, gif_width=640)
        shutil.rmtree(tmp, ignore_errors=True)
        done["video"] = {"mp4": "visuals/ahmed_body.mp4", "gif": "visuals/ahmed_body.gif", "frames": len(frames),
                         "fps": fps, "seconds": len(frames) / fps, "narrated": narrated, "language": "en",
                         "segments": {k: len(v) for k, v in segments}}
    done["seconds"] = time.time() - t0
    _dump(res, "visuals.json", done)
    return done


# ---------------------------------------------------------------------- 7. costs, report, evidence
def _load(res: str, name: str):
    p = os.path.join(res, name)
    return json.load(open(p)) if os.path.exists(p) else None


def study_costs(recs, res: str) -> dict[str, Any]:
    """CPU-hours of every CFD purpose (:func:`~pinneapple_autolab.study.campaign.cost_table`) plus the seconds the
    other stages recorded in their JSON files (surrogate fits and CV, the inverse stage, the renders), one core
    each; writes ``costs.json``."""
    from pinneapple_autolab.study import campaign as C
    extra: dict[str, float] = {}
    b = _load(res, "benchmark.json")
    if b:
        extra["surrogates (fit + CV, 1 core)"] = float(sum(r.get("fit_s", 0) + r.get("cv_s", 0)
                                                           for part in ("scalar", "field") for r in b[part]["rows"].values()))
    inv = _load(res, "inverse.json")
    if inv and inv.get("seconds"):
        extra["inverse (MDN + baselines, 1 core)"] = float(inv["seconds"])
    vis = _load(res, "visuals.json")
    if vis and vis.get("seconds"):
        extra["visuals (Cycles renders, 4 cores)"] = float(vis["seconds"]) * 4
    t = C.cost_table(recs, extra)
    env = _load(os.path.join(res, "cfd_runs"), "environment.json") or {}
    t["environment"] = {k: env.get(k) for k in ("cpu", "cpu_count", "mem_gb", "platform") if k in env}
    _dump(res, "costs.json", t)
    return t


def _row(d: Mapping[str, Any], keys: Sequence[str], digits: int = 4):
    out = []
    for k in keys:
        v = d.get(k, "") if isinstance(d, Mapping) else ""
        out.append(round(v, digits) if isinstance(v, float) else v)
    return out


def study_report(recs, res: str, out: str | None = None, title: str = "Ahmed body: an end-to-end CFD surrogate study"
                 ) -> dict[str, Any]:
    """The evidence report of the study from the JSON files in ``res`` (nothing is computed here, so a stage that
    was not run reads as "not measured"): one section per stage with its table, figures and the trust decision
    where one exists; ``report.html`` (self-contained), ``evidence.zip`` (every JSON hashed,
    :func:`~pinneapple_autolab.study.report.evidence_package`) and ``summary.json`` (the measured numbers with
    units and how they were measured, for the experiment manifest and the README). Returns the summary."""
    from pinneapple_autolab.study import report as R
    out = out or os.path.join(res, "report.html")
    S: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    missing: list[str] = []

    def metric(name, value, unit, how):
        if value is not None and np.isfinite(float(value)):
            metrics.append({"name": name, "value": float(value), "unit": unit, "how": how})

    ok = [r for r in recs if r.ok]
    by = {r.id: r for r in ok}
    env = _load(os.path.join(res, "cfd_runs"), "environment.json") or {}
    # 1. case
    S.append({"title": "The case", "text": (
        f"Ahmed, Ramm & Faltin (1984) body ({A.LENGTH} x {A.WIDTH} x {A.HEIGHT} m, {A.NOSE_RADIUS} m front radii, "
        f"{A.SLANT_LENGTH} m rear slant), half model with a symmetry plane, moving road, no stilts. Steady RANS "
        f"k-omega SST with wall functions in OpenFOAM simpleFoam; design variables "
        + ", ".join(f"{v.name} {v.lower:g}-{v.upper:g} {v.unit}" for v in A.VARIABLES)
        + f".\n\nRuns: {len(ok)} successful of {len(recs)} recorded on "
        + (f"{env.get('cpu', 'unknown CPU')} ({env.get('cpu_count', '?')} cores; code at git {env.get('git_commit', '')[:8]})"
           if env else "an unrecorded machine") + ". Every number below was measured by the scripts in "
        "examples/end_to_end/ahmed_body; a stage without its JSON file reads as not measured.")})
    # 2. geometry
    cad = _load(res, "cad_import.json")
    if cad:
        S.append({"title": "Geometry and CAD import", "decision": "APPROVED" if not cad.get("issues") else "REVIEW",
                  "text": f"OpenCASCADE B-rep through gmsh, imported back from its STL: {cad['triangles']} triangles, "
                          f"volume {cad['volume_m3']:.5f} m3 against {cad['parametric_volume_m3']:.5f} parametric, "
                          f"symmetry {cad['symmetry_mm']:.2f} mm; issues: {cad.get('issues') or 'none'}. "
                          + (f"CFD on the CAD body at the baseline: CD {by['cad_baseline'].outputs['CD']:.4f} against "
                             f"{by['mesh_L1'].outputs['CD']:.4f} parametric on the same mesh "
                             f"({(by['cad_baseline'].outputs['CD'] / by['mesh_L1'].outputs['CD'] - 1):+.1%})."
                             if "cad_baseline" in by and "mesh_L1" in by else "")})
        if "cad_baseline" in by and "mesh_L1" in by:
            metric("cad_vs_parametric_cd_rel", by["cad_baseline"].outputs["CD"] / by["mesh_L1"].outputs["CD"] - 1, "1",
                   "CD of the imported CAD body over the parametric body, baseline, L1 mesh, OpenFOAM")
    else:
        missing.append("CAD import")
    # 3. mesh
    ms = _load(res, "mesh_study.json")
    if ms:
        cd = ms["CD"]
        rows = [[m["name"], m["cells"], round(m["h_rel"], 3), round(v, 4), round(e, 4) if e is not None else "",
                 ms["trust"].get(m["name"], ""), ", ".join(ms["failed_checks"].get(m["name"], [])) or "none",
                 round(m["cost_s"] / 60, 1)]
                for m, v, e in zip(ms["meshes"], cd["values"], cd["error_vs_extrapolated"] if cd.get("extrapolated")
                                   else cd["error_vs_finest"], strict=False)]
        basis = "Richardson value" if cd.get("extrapolated") else "finest mesh"
        text = (f"Mesh family with a constant refinement ratio sqrt(2) at the baseline (25 deg, 50 mm, 60 m/s, 1000 "
                f"iterations). CD convergence: {cd['status']}; {cd.get('note') or ''}. "
                + (f"Observed order p = {cd['p']:.2f}, Richardson value {cd['extrapolated']:.4f}, GCI on the finest "
                   f"mesh {cd['gci_fine']:.2%}." if cd.get("extrapolated") else "")
                + " Triplets: " + "; ".join(f"{'-'.join(t['meshes'])}: {t.get('status')}"
                                             + (f", p {t['p']:.2f}" if t.get("p") else "") for t in cd["triplets"])
                + f". Campaign mesh: {ms['campaign_mesh']} (chosen for cost; its CD error against the {basis} is in "
                  f"the table).")
        S.append({"title": "Mesh convergence", "decision": "APPROVED" if cd.get("extrapolated") else "REVIEW", "text": text,
                  "table": {"columns": ["mesh", "cells", "h / h_finest", "CD", f"error vs {basis}", "run checks",
                                        "failed checks", "wall min"], "rows": rows},
                  "figures": ["figures/grid_convergence_cd.png"]})
        for m, v in zip(ms["meshes"], cd["values"], strict=False):
            metric(f"cd_baseline_{m['name']}", v, "1", f"OpenFOAM, {m['cells']} cells, window mean of iterations 500-1000")
        if cd.get("gci_fine") is not None:
            metric("gci_cd_finest", cd["gci_fine"], "1", "grid-convergence index of CD on the finest mesh (Celik 2008)")
    else:
        missing.append("mesh convergence")
    # 4. validation
    val = _load(res, "validation.json")
    if val and val["rows"]:
        cols = ["slant deg", "CD CFD", "CD measured", "rel. error", "CFD checks", "failed", "iteration std"]
        rows = [[r["slant_deg"], round(r["CD_cfd"], 4), r["CD_measured"], f"{r['rel_error']:+.1%}", r["trust"],
                 ", ".join(r["failed_checks"]) or "none", round(r["iteration_std"], 4)] for r in val["rows"]]
        rows_l2 = [[r["slant_deg"], round(r["CD_cfd"], 4), r["CD_measured"], f"{r['rel_error']:+.1%}", r["trust"],
                    ", ".join(r["failed_checks"]) or "none", round(r["iteration_std"], 4)] for r in val.get("rows_L2", [])]
        err = np.array([r["rel_error"] for r in val["rows"]])
        text = (f"{val['source']}\n\nCampaign mesh {val['campaign_mesh']}: |relative error| median {np.median(np.abs(err)):.1%}, "
                f"max {np.max(np.abs(err)):.1%} (at {val['rows'][int(np.argmax(np.abs(err)))]['slant_deg']:g} deg). "
                f"Campaign-mesh numerical error ({val.get('campaign_mesh_error_basis', '')}): "
                + (f"{val['campaign_mesh_error']:.1%}." if val.get("campaign_mesh_error") is not None else "not measured.")
                + (" The sweep was repeated on the L2 mesh (second table): the part of the error that the mesh "
                   "explains is the difference between the two." if rows_l2 else " The sweep on L2 was not run."))
        S.append({"title": "Validation against the wind tunnel", "decision": "REVIEW", "text": text,
                  "table": {"columns": cols, "rows": rows}, "figures": ["figures/validation_cd.png"]})
        if rows_l2:
            S.append({"title": "Validation sweep on the L2 mesh", "table": {"columns": cols, "rows": rows_l2}})
        metric("validation_cd_rel_error_median", float(np.median(np.abs(err))), "1",
               "median |CD_cfd / CD_measured - 1| over the six slant angles of Ahmed et al. (1984), L1 mesh")
        metric("validation_cd_rel_error_max", float(np.max(np.abs(err))), "1", "largest of the same")
    else:
        missing.append("validation sweep")
    # 5. DOE and run checks
    doe = _load(res, "doe_plan.json")
    if doe:
        cov, rnd = doe["coverage"], doe["random_reference"]
        dec = {}
        fails: dict[str, int] = {}
        for r in ok:
            d = A.run_trust_report(r.meta).decisao
            dec[d] = dec.get(d, 0) + 1
            for c in A.qa_checks(r.meta):
                if c.status == "contradicts":
                    fails[c.name] = fails.get(c.name, 0) + 1
        S.append({"title": "Design of experiments and run checks", "text": (
            f"{len(doe['cases'])} {doe['method']} Latin-hypercube designs (seed {doe['seed']}): centred L2 discrepancy "
            f"{cov['discrepancy']:.4f} against {rnd['discrepancy']:.4f} for random designs of the same size (median of "
            f"20), minimum distance {cov['min_distance']:.3f} against {rnd['min_distance']:.3f}.\n\nChecks on the "
            f"{len(ok)} successful runs (checkMesh, non-orthogonality, skewness, final residuals <= 1e-3, force "
            f"stationarity <= 1 %, mass balance <= 1e-3, y+ in the wall-function range): decisions "
            + ", ".join(f"{k} {v}" for k, v in sorted(dec.items())) + "; failed checks: "
            + (", ".join(f"{k} in {v}" for k, v in sorted(fails.items())) or "none")
            + ". Limits were not widened; a run that fails a check stays in the dataset with its record.")})
        metric("doe_discrepancy", cov["discrepancy"], "1", "centred L2 discrepancy of the 48-point maximin LHS")
        metric("doe_discrepancy_random", rnd["discrepancy"], "1", "median over 20 random designs of 48 points")
    else:
        missing.append("DOE")
    # 6. dataset
    dsj = _load(os.path.join(res, "dataset"), "ahmed_body.json")
    if dsj:
        x = dsj.get("x_pinneapple_study", {})
        sp = {k: len(dsj["splits"].get(k) or []) for k in ("train", "val", "test")}
        sp["extrapolation"] = len(x.get("extrapolation", []))
        S.append({"title": "Dataset", "text": (
            f"{dsj['n_rows']} cases (DOE + validation sweep): inputs {', '.join(c['name'] for c in dsj['columns'] if c['role'] == 'input')}; "
            f"outputs CD, CL, pressure and friction drag, pressure drag per region, and Cp on 1751 surface points. "
            f"Split {sp} with the held-out rule '{x.get('holdout_rule', '')}' (no model sees the extrapolation split "
            f"before its final evaluation). DatasetSpec with SHA-256 {dsj['files'][0].get('sha256', '')[:16]}... "
            f"and every run id in its provenance.")})
    else:
        missing.append("dataset")
    # 7. surrogates
    b = _load(res, "benchmark.json")
    ti = _load(res, "template_integral.json")
    if b:
        cols = ["target", "family", "CV rel. RMSE / rel. L2", "CV half width", "test RMSE / rel. L2", "test cov. 90 %",
                "extrap. RMSE / rel. L2", "extrap. cov. 90 %", "fit s", "physics"]
        rows = []
        for part, key, ek in (("scalar", "cv_rel_rmse", "rmse"), ("field", "cv_rel_l2", "rel_l2_mean")):
            for k, r in b[part]["rows"].items():
                t, f = k.split("/", 1)
                phys = b[part]["physics"].get(k, [])
                te, ex = r.get("test", {}), r.get("extrapolation", {})
                rows.append([t, f + (" *" if b["selection"].get(t) == f else ""), round(r[key], 4),
                             round(r.get("cv_half_width90", float("nan")), 4), round(te.get(ek, float("nan")), 4),
                             round(te.get("coverage90_conformal", float("nan")), 2), round(ex.get(ek, float("nan")), 4),
                             round(ex.get("coverage90_conformal", float("nan")), 2), round(r.get("fit_s", 0), 1),
                             "pass" if all(c["passed"] for c in phys) else "FAIL: " + ", ".join(c["name"] for c in phys if not c["passed"])])
        sel = b["selection"]
        text = (b["rule"] + "\n\nSelected: " + ", ".join(f"{t}: {f or 'none'}" for t, f in sel.items())
                + ". Reasons: " + "; ".join(f"{t}: {r}" for t, r in b["reasons"].items())
                + ". The test and extrapolation columns were measured once, after the selection (* = selected). "
                  "Budget: every family fits on the same train + val cases with the same 5 folds; the iterative "
                  "family (DeepONet) gets exactly the wall-clock budget, the one-shot families report the time they "
                  "used (fit s).")
        S.append({"title": "Surrogates at equal data and budget, with calibrated uncertainty", "text": text,
                  "table": {"columns": cols, "rows": rows},
                  "figures": ["figures/parity_CD.png", "figures/parity_CL.png", "figures/coverage_cd.png",
                              "figures/coverage_cp.png"]})
        for t, f in sel.items():
            if f:
                part = "field" if t == "cp" else "scalar"
                r = b[part]["rows"][f"{t}/{f}"]
                ek = "rel_l2_mean" if t == "cp" else "rmse"
                for split in ("test", "extrapolation"):
                    if r.get(split):
                        metric(f"{t}_{f}_{split}_{ek}", r[split].get(ek), "1",
                               f"{f} on the {split} split, fitted on train + val, measured once after selection")
                        metric(f"{t}_{f}_{split}_coverage90", r[split].get("coverage90_conformal"), "1",
                               f"share of {split} cases inside the cross-conformal 90 % interval")
        if ti:
            chk = {k: next((c for c in v if c["name"] == "cp_integral_matches_CD_pressure"), None)
                   for k, v in b["field"]["physics"].items()}
            rows = [[k.split("/", 1)[1], round(c["value"], 4), round(c["limit"], 4), "pass" if c["passed"] else "FAIL", c["reason"]]
                    for k, c in chk.items() if c]
            S.append({"title": "The field-scalar consistency check and its tolerance", "text": (
                f"The pressure drag integrated from the CFD's own Cp on the 1751 template points differs from the CFD's "
                f"CD_pressure by {ti['abs_gap_mean']:.4f} on average, {ti['abs_gap_p95']:.4f} at the 95th percentile and "
                f"{ti['abs_gap_max']:.4f} at most over {ti.get('n_cases', '')} cases (signed mean {ti.get('signed_gap_mean', float('nan')):+.4f}: "
                f"the template under-resolves the rear edges). A Cp surrogate is held to that same 95th percentile: "
                f"over 128 Sobol points of the design space, |integrated Cp - predicted CD_pressure| is divided by the "
                f"quadrature sum of that template gap ({ti['tolerance']:.4f}) and the scalar model's calibrated 90 % half "
                f"width at the point (independent bounds), and the 95th percentile of the ratio must be <= 1. Decision "
                f"and why: the earlier formulation compared the maximum over 48 points with p95 + 0.01 = "
                f"{ti['abs_gap_p95'] + 0.01:.4f}; a maximum against a percentile fails even the CFD's own fields (their "
                f"largest gap is {ti['abs_gap_max']:.4f}), and the scalar model's error was counted against the field. "
                f"Under it no Cp family was eligible (POD-GP 0.0554, POD-MLP 0.0563, POD-RBF 0.0992, DeepONet 0.2131 "
                f"against 0.0515). Subtracting the half width instead of adding it in quadrature was tried and rejected: "
                f"it let a field whose raw gap is twice the CFD's pass. The raw p95 and maximum are reported in each "
                f"check's reason."),
                "table": {"columns": ["family", "p95 of |difference| / combined uncertainty", "limit", "result", "detail"], "rows": rows}})
    else:
        missing.append("surrogate benchmark")
    # 8. sensitivity, Pareto, active learning
    so = _load(res, "sobol.json")
    if so:
        rows = [[o, *[round(v, 3) for v in so[o]["S1"]], *[round(v, 3) for v in so[o]["ST"]]] for o in so["outputs"]]
        S.append({"title": "Global sensitivity (Sobol indices on the selected surrogates)", "text": (
            f"Saltelli estimates over the training region ({so['n_evaluations']} surrogate evaluations), bootstrap 95 % "
            f"intervals in sobol.json. These are indices of the surrogate, not of the CFD."),
            "table": {"columns": ["output"] + [f"S1 {k}" for k in so["inputs"]] + [f"ST {k}" for k in so["inputs"]], "rows": rows},
            "figures": ["figures/sobol_cd.png"]})
        for o in so["outputs"]:
            for k, v in zip(so["inputs"], so[o]["ST"], strict=False):
                metric(f"sobol_total_{o}_{k}", v, "1", "total Sobol index on the selected surrogate, Saltelli n=4096")
    more = _load(res, "more_cfd.json")
    par = _load(res, "pareto_plan.json")
    if more:
        ver = more.get("pareto_verification", [])
        rows = [[round(v["params"]["slant_deg"], 1), round(v["params"]["clearance"], 4), round(v["predicted"]["CD"], 4),
                 round(v["cfd"]["CD"], 4) if v.get("cfd") else "not run", round(v["predicted"]["CL"], 4),
                 round(v["cfd"]["CL"], 4) if v.get("cfd") else "not run",
                 f"{abs(v['cfd']['CD'] - v['predicted']['CD']):.4f}" if v.get("cfd") else ""] for v in ver]
        hw = f" CD half width <= {par['picks'][0]['predicted'].get('CD_half_width', float('nan')):.3f} at the picks" if par and par.get("picks") else ""
        S.append({"title": "Pareto front (CD against CL at 40 m/s) and its CFD verification", "text": (
            f"NSGA-II on the surrogate means with the calibrated CD half width constrained to <= 0.02 "
            f"({par['n_evaluations'] if par else '?'} evaluations); three picks (both ends and the knee) run in OpenFOAM.{hw}"),
            "table": {"columns": ["slant deg", "clearance m", "CD predicted", "CD CFD", "CL predicted", "CL CFD", "|dCD|"], "rows": rows},
            "figures": ["figures/pareto.png"]})
        for i, v in enumerate(ver):
            if v.get("cfd"):
                metric(f"pareto_pick{i}_cd_abs_error", abs(v["cfd"]["CD"] - v["predicted"]["CD"]), "1",
                       "|CD CFD - CD predicted| at a Pareto pick, OpenFOAM L1")
        d0 = more["doe"]
        rows = [[lab, more[lab]["n_fit"], round(more[lab]["CD_rmse"], 4), round(more[lab]["cp_rel_l2"], 4)]
                for lab in ("doe", "doe + acquired", "doe + random") if lab in more]
        S.append({"title": "Active learning against random runs", "text": (
            f"The GP of CD and the POD-GP of Cp refitted on the DOE training set alone ({d0['n_fit']} cases), plus the 8 "
            f"runs the GP's uncertainty asked for, plus 8 random runs; errors on the untouched test split "
            f"({more['test_cases']} cases). One batch, one seed: the difference between the two rows is the measured "
            f"effect, no more."),
            "table": {"columns": ["training set", "cases", "CD RMSE (GP)", "Cp rel. L2 (POD-GP)"], "rows": rows},
            "figures": ["figures/active_learning_cd.png"]})
        for lab in ("doe", "doe + acquired", "doe + random"):
            if lab in more:
                metric(f"active_learning_cd_rmse_{lab.replace(' + ', '_plus_')}", more[lab]["CD_rmse"], "1",
                       f"GP of CD fitted on {lab}, RMSE on the test split")
    else:
        missing.append("extra CFD (acquisition, random, Pareto verification)")
    # 9. inverse
    inv = _load(res, "inverse.json")
    if inv:
        cal = inv["calibration_on_draws"]
        held = inv["heldout_cfd"]
        txt = (f"Amortised posterior p(slant | CD, clearance, speed): a mixture density network ({inv['mdn']}) on 40 000 "
               f"draws of the selected CD surrogate ({inv['selected_CD_family']}) with its calibrated noise plus the "
               f"CFD-vs-wind-tunnel discrepancy sigma_d = {inv['sigma_d_all']:.4f} (RMS over the validation sweep).\n\n"
               f"Synthetic checks: coverage of the credible intervals on 300 surrogate draws {cal}; the slant of "
               f"{len(held)} held-out CFD cases recovered from their CFD drag with the truth inside the 90 % HDI in "
               f"{inv['heldout_inside_hdi90']:.0%} of them.")
        S.append({"title": "Inverse problem: the slant angle from a drag measurement", "text": txt,
                  "figures": ["figures/posterior_slant_25.png", "figures/posterior_slant_30.png"]})
        metric("inverse_heldout_inside_hdi90", inv["heldout_inside_hdi90"], "1",
               "share of held-out CFD cases whose slant lies inside the MDN 90 % HDI from their CFD drag")
        for name, ms in inv.get("measurement_sets", {}).items():
            if ms.get("rows"):
                rows = [[r["slant_deg"], r["CD_measured"], round(r["sigma_d"], 4), round(r["mdn_median"], 1),
                         "; ".join(f"[{a:.1f}, {b:.1f}]" for a, b in r["mdn_hdi90"]), "yes" if r["inside"] else "NO",
                         ", ".join(f"{x:g}" for x in r["least_squares"]),
                         f"{r['mcmc']['median']:.1f} [{r['mcmc']['lo90']:.1f}, {r['mcmc']['hi90']:.1f}]",
                         "yes" if r["measurement_within_model_reach"] else "NO"] for r in ms["rows"]]
                S.append({"title": f"Inverse on the measurements of {ms['label']} ({ms['speed']:g} m/s)",
                          "text": (f"{ms['source']}\n\nMeasured slant inside the 90 % HDI in {ms['inside_hdi90']:.0%} of the "
                                   f"points. sigma_d: {ms['rows'][0]['sigma_d_basis']}. A measurement outside the "
                                   f"surrogate's reach at these conditions (last column) says the CFD curve never "
                                   f"takes that value: the posterior then sits where the model is wrong, as it must. "
                                   f"Verification of the reference values: {json.dumps(ms.get('verification', {}))}."),
                          "table": {"columns": ["slant deg", "CD measured", "sigma_d", "MDN median", "MDN 90 % HDI",
                                                "truth inside", "least squares", "Metropolis median [90 %]",
                                                "within model reach"], "rows": rows}})
                metric(f"inverse_{name}_inside_hdi90", ms["inside_hdi90"], "1",
                       f"share of the {ms['label']} measurements whose slant lies inside the MDN 90 % HDI")
            else:
                S.append({"title": f"Inverse on the measurements of {ms['label']} ({ms['speed']:g} m/s)",
                          "decision": "REVIEW", "text": f"{ms.get('status', 'not measured')}.\n\n{ms['source']}\n\n"
                                                          f"Verification: {json.dumps(ms.get('verification', {}))}."})
    else:
        missing.append("inverse")
    # 10. twin and visuals
    tw = _load(os.path.join(res, "twin"), "twin.json")
    if tw:
        ie = tw.get("interpolation_error", {})
        S.append({"title": "Digital twin", "text": (
            f"{tw['meta'].get('subtitle', '')}. Sliders over {', '.join(i['name'] for i in tw['inputs'])}; the Cp field "
            f"on the body (mean or 90 % half width), CD and CL with their calibrated band, a warning beyond "
            f"{tw['max_distance']} (unit cube) from the nearest CFD run, and a button that runs a verifying OpenFOAM "
            f"case through apps/ahmed_twin. The page interpolates a grid of model evaluations; its error against the "
            f"models on random points: " + ", ".join(f"{k} {v:.4g}" for k, v in ie.items()) + ".")})
        for k, v in ie.items():
            metric(f"twin_interpolation_error_{k}", v, "1", "max |page interpolation - model| on 256 random points")
    else:
        missing.append("twin page")
    vis = _load(res, "visuals.json")
    if vis:
        S.append({"title": "Visuals", "text": (
            f"Blender Cycles stills ({', '.join(vis.get('stills', []))}) in {vis.get('stills_s', 0) / 60:.1f} min, "
            + (f"a {vis['video'].get('seconds', 0):.0f} s video ({vis['video']['frames']} frames at {vis['video']['fps']} fps"
               + (", narrated in English" if vis["video"].get("narrated") else "") + ")" if vis.get("video") else "no video")
            + ", the three.js viewer and an OpenUSD stage"
            + (f" ({'valid' if not vis['usd'].get('issues') else vis['usd']['issues']})" if isinstance(vis.get("usd"), dict) and "path" in vis["usd"] else "")
            + f"; {vis.get('seconds', 0) / 60:.1f} min in all."),
            "figures": [f"visuals/{n}" for n in vis.get("stills", [])]})
    else:
        missing.append("visuals")
    # 11. costs
    costs = study_costs(recs, res)
    rows = [[k, v["runs"], v["failed"], round(v["wall_h"], 2), round(v["cpu_h"], 2)] for k, v in costs["stages"].items()]
    rows.append(["total", costs["total"]["runs"], costs["total"]["failed"], round(costs["total"]["wall_h"], 2),
                 round(costs["total"]["cpu_h"], 2)])
    S.append({"title": "Cost accounting", "text": (
        f"Wall time and CPU-hours per stage on {costs['environment']}. CFD rows from the run records (wall x "
        f"processes); the other rows from the seconds each stage recorded, counted on the cores it used."),
        "table": {"columns": ["stage", "runs", "failed", "wall h", "CPU h"], "rows": rows}})
    metric("cfd_cpu_hours", sum(v["cpu_h"] for k, v in costs["stages"].items() if v["runs"]), "h",
           "sum over CFD purposes of wall time x MPI processes")
    # 12. limits
    S.append({"title": "Not measured, and limits", "text": (
        ("Stages without a result file: " + ", ".join(missing) + ".\n\n" if missing else "")
        + "Steady RANS k-omega SST with wall functions cannot reproduce the drag peak at the critical slant angle; "
          "the validation table says by how much. The mesh study is the main numerical uncertainty where its "
          "convergence is oscillatory. The wind-tunnel references were transcribed and only partly re-verified "
          "(per-value notes in the inverse sections). Surrogates are valid in the DOE box only; the twin warns "
          "outside it. Lienhart & Becker drag values were not obtained from this environment.")})
    html = R.write_report(title, S, out, base=res, subtitle="Every number measured by the scripts in "
                          "examples/end_to_end/ahmed_body; evidence.zip holds the hashed sources.")
    files = [os.path.join(res, n) for n in ("mesh_study.json", "validation.json", "doe_plan.json", "template_integral.json",
                                             "benchmark.json", "sobol.json", "more_cfd.json", "inverse.json", "costs.json",
                                             "visuals.json", "cad_import.json") if os.path.exists(os.path.join(res, n))]
    dsp = os.path.join(res, "dataset", "ahmed_body.json")
    if os.path.exists(dsp):
        files.append(dsp)
    pkg = R.evidence_package(title, files, os.path.join(res, "evidence.zip"))
    summary = {"metrics": metrics, "not_measured": missing, "report": os.path.relpath(html, res),
               "evidence_zip": "evidence.zip", "evidence_problems": pkg["problems"], "sections": [s["title"] for s in S]}
    _dump(res, "summary.json", summary)
    return summary
