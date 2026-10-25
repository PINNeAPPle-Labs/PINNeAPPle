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
           "evaluate_more_runs", "fit_selected", "train_slant_mdn", "inverse", "twin"]

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
    own pressure drag; the 95th percentile + 0.01 is the tolerance of the field-scalar consistency check."""
    P = [dict(zip(ds.inputs, x, strict=False)) for x in ds.X]
    integ = np.array([A.pressure_drag_from_cp(p, cp) for p, cp in zip(P, ds.fields["cp"], strict=False)])
    gap = np.abs(integ - ds.y("CD_pressure"))
    return {"abs_gap_mean": float(gap.mean()), "abs_gap_max": float(gap.max()),
            "abs_gap_p95": float(np.quantile(gap, 0.95)), "tolerance": float(np.quantile(gap, 0.95) + 0.01),
            "how": "pressure_drag_from_cp of the CFD Cp on the template vs the CFD CD_pressure, every case"}


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
    phys = A.prediction_checks(BOUNDS, cd_pressure=lambda U: cdp(U)[0], integral_tol=integ["tolerance"])
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


def inverse(ds, recs, res: str, family: str | None = None, budget_s: float = 60.0) -> dict[str, Any]:
    """Synthetic checks first (calibration on 300 surrogate draws; the slant of every held-out CFD case from its
    CFD drag, with the surrogate fitted on train + val only), then the wind-tunnel measurements with the
    deployment surrogate and a leave-one-out discrepancy; MDN against least squares and Metropolis.
    Writes ``inverse.json`` and the posterior figures."""
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
    rows, posts = [], {}
    u_c = A.params_to_unit({"slant_deg": 0, "clearance": 0.05, "speed": 60})[0]
    grid = np.linspace(0, 1, 401)
    mg, hg = pred_d(np.c_[grid, np.full_like(grid, u_c[1]), np.full_like(grid, u_c[2])])
    for k, r in enumerate(val):
        s, cd = r["slant_deg"], r["CD_measured"]
        sig = float(np.sqrt(np.mean(np.delete(d_all, k) ** 2)))
        mdn_k = train_slant_mdn(pred_d, sig, seed=k, budget_s=0.75 * budget_s)
        deg, iv = _slant(mdn_k, cd, 0.05, 60.0)
        ls, mc = _baselines(pred_d, cd, 0.05, 60.0, sig)
        rows.append({"slant_deg": s, "CD_measured": cd, "sigma_d": sig, "mdn_median": float(np.median(deg)),
                     "mdn_hdi90": iv, "inside": any(a <= s <= b for a, b in iv),
                     "least_squares": sorted({round(x["slant_deg"], 2) for x in ls}),
                     "mcmc": {"median": float(np.median(mc.samples[:, 0])),
                              "lo90": float(np.quantile(mc.samples[:, 0], 0.05)),
                              "hi90": float(np.quantile(mc.samples[:, 0], 0.95)), "acceptance": mc.acceptance},
                     "measurement_within_model_reach": bool(np.min(mg - hg) - 2 * sig <= cd <= np.max(mg + hg) + 2 * sig),
                     "surrogate_CD_range_at_60ms": [float(mg.min()), float(mg.max())]})
        posts[s] = (deg, mc.samples[:, 0])
    out = {"selected_CD_family": family, "sigma_d_all": sigma_all, "calibration_on_draws": cal,
           "heldout_cfd": held, "heldout_inside_hdi90": float(np.mean([r["inside"] for r in held])) if held else None,
           "measurements": rows, "mdn": mdn.info, "seconds": time.time() - t0}
    _dump(res, "inverse.json", out)
    for s in (25.0, 30.0):
        if s in posts:
            FG.posterior(_fig(res, f"posterior_slant_{s:g}.png"),
                         {"MDN (amortised)": posts[s][0], "Metropolis": posts[s][1]}, truth=s,
                         xlabel="slant angle [deg]", title=f"Slant inferred from the measured CD at {s:g} deg")
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


def visuals(recs, runs: str, res: str, *, rid: str = "val_25.0", samples: int = 64, size=(1600, 900),
            video: bool = True, frames_orbit: int = 48, frames_sweep: int = 41, fps: int = 12,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """Blender Cycles stills of one run (dark stage: Cp on the body with streamlines from the front and from the
    rear, mid-plane speed, wake vorticity), a video (an orbit of that run, then the slant swept from 0 to 40 deg
    through the deployment Cp surrogate, each frame labelled), the three.js viewer and an OpenUSD stage."""
    import shutil
    import tempfile

    import pinneapple as pp
    by = {r.id: r for r in recs if r.ok}
    p = by[rid].params
    sc = flow_scene(runs, rid, p)
    out_dir = os.path.join(res, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    done: dict[str, Any] = {}
    t0 = time.time()
    kw = dict(samples=samples, size=size, background="dark", floor=True)
    stills = {"cp_streamlines_front.jpg": dict(field="cp", field_range=(-1.0, 1.0), cmap="rdbu", lines=True, view="iso"),
              "cp_streamlines_rear.jpg": dict(field="cp", field_range=(-1.0, 1.0), cmap="rdbu", lines=True,
                                              view="iso_back"),
              "mid_plane_speed.jpg": dict(slice="mid plane: speed", view="side", floor=False),
              "wake_vorticity.jpg": dict(slice="wake: vorticity", view="back", floor=False)}
    for name, o in stills.items():
        pp.viz.render(sc, os.path.join(out_dir, name), **{**kw, **o})
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
        frames = []
        lo_kw = dict(samples=max(16, samples // 3), size=(1280, 720), background="dark", floor=True)
        for k in range(frames_orbit):
            a = 2 * np.pi * k / frames_orbit - 2.3
            f = os.path.join(tmp, f"o{k:03d}.png")
            pp.viz.render(sc, f, field="cp", field_range=(-1.0, 1.0), cmap="rdbu", lines=True,
                          view=(np.cos(a), np.sin(a), 0.55), title=f"CFD: slant {p['slant_deg']:g} deg, "
                          f"{p['speed']:g} m/s, Cp", **lo_kw)
            frames.append(f)
        sel = json.load(open(os.path.join(res, "benchmark.json")))["selection"]
        dep = deployment_dataset(recs)
        fpred = fit_selected(dep, "cp", sel["cp"], np.arange(len(dep)))
        _, Fd, j = A.template_faces(A.BASELINE)
        for k, s in enumerate(np.linspace(0.0, 40.0, frames_sweep)):
            q = {**p, "slant_deg": float(s)}
            cp, hw = fpred(A.params_to_unit(q))
            V, F, jj = A.template_faces(q)
            sc2 = Scene([Surface("ahmed", V, F, "paint", fields={"cp": cp[0][jj]})], title="surrogate")
            f = os.path.join(tmp, f"s{k:03d}.png")
            pp.viz.render(sc2, f, field="cp", field_range=(-1.0, 1.0), cmap="rdbu", view="iso_back",
                          title=f"Surrogate ({sel['cp']}): slant {s:.0f} deg, Cp; 90 % band +-{float(np.mean(hw)):.2f}",
                          **lo_kw)
            frames.append(f)
        _encode(frames, os.path.join(out_dir, "ahmed_body.mp4"), fps)
        _encode(frames[::2], os.path.join(out_dir, "ahmed_body.gif"), fps / 2, gif=True, gif_width=640)
        shutil.rmtree(tmp, ignore_errors=True)
        done["video"] = {"mp4": "visuals/ahmed_body.mp4", "gif": "visuals/ahmed_body.gif", "frames": len(frames),
                         "fps": fps}
    done["seconds"] = time.time() - t0
    _dump(res, "visuals.json", done)
    return done
