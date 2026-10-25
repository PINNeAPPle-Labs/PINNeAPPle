"""The end-to-end study stages (pinneapple_autolab.study), the Ahmed domain and amortised inference.

Fast tests use analytic simulators; the OpenFOAM test runs one tiny Ahmed case and needs PINNEAPPLE_SLOW=1.
"""
import json
import math
import os

import numpy as np
import pytest

from pinneapple_autolab import FunctionSimulator
from pinneapple_autolab.domains import ahmed as A
from pinneapple_autolab.study import benchmark as B
from pinneapple_autolab.study import campaign as C
from pinneapple_autolab.study import dataset as D
from pinneapple_autolab.study import extras as E
from pinneapple_autolab.study import mesh_study as M
from pinneapple_autolab.study import twin as T
from pinneapple_design.design_optimizer.study import Variable
from pinneapple_design.geometry.bodies import ahmed_body

VARS = [Variable("a", 0.0, 1.0), Variable("b", -1.0, 1.0), Variable("c", 2.0, 4.0)]
PTS = np.c_[np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)]


def _fn(p):
    a, b, c = p["a"], p["b"], p["c"]
    y = 1.0 + a ** 2 + 0.5 * b + 0.1 * c
    return {"y": y, "z": math.sin(3 * a) + b * c, "__fields__": {"f": y * np.sin(np.pi * PTS[:, 0]) + 0.2 * b * PTS[:, 0]}}


def _sim(calls):
    def fn(p):
        calls.append(p)
        return _fn(p)
    return FunctionSimulator(fn, ["y", "z"], field_names=["f"], points=PTS)


# ---------------------------------------------------------------------- geometry and domain
def test_ahmed_default_unchanged_and_dense_template_fixed():
    V, F = ahmed_body(25.0)
    assert V.shape == (442, 3) and F.shape == (880, 3)
    counts = {A.field_vertices({"slant_deg": s, "clearance": c, "speed": 40}).shape for s in (0, 20, 40) for c in (0.03, 0.08)}
    assert counts == {(632, 3)}
    with pytest.raises(ValueError):
        ahmed_body(60.0)                                  # the slant would cut through the body


def test_regions_and_pressure_integral():
    P = A.BASELINE
    V = A.field_vertices(P)
    lab = A.region_of(V, P)
    assert set(lab) == set(A.REGIONS)
    assert abs(A.pressure_drag_from_cp(P, np.ones(len(V)))) < 1e-12         # closed surface
    front = (lab == "front").astype(float)
    assert 0.3 < A.pressure_drag_from_cp(P, front) < 1.0                     # push on the nose = drag


def test_ahmed_check_and_qa():
    sim = A.AhmedAero()
    assert sim.check(A.BASELINE) == []
    assert any("outside" in s for s in sim.check({**A.BASELINE, "slant_deg": 50.0}))
    meta = {"mesh_ok": True, "max_non_orthogonality": 50.0, "max_skewness": 2.0,
            "final_residuals": {"p": 1e-4, "Ux": 2e-3}, "convergence": {"CD_mean_drift": 0.002, "window_writes": 10},
            "mass_balance": {"imbalance": 2e-4}, "yplus": {"frac_30_300": 0.9, "mean": 90, "p95": 200}}
    st = {c.name: c.status for c in A.qa_checks(meta)}
    assert st["checkMesh"] == st["mass_balance"] == st["yplus_wall_functions"] == "supports"
    assert st["residuals"] == "contradicts"
    hist = [{"iteration": 50 * i, "CD": 0.3 + (0.01 if i % 2 else -0.01), "CL": 0.2} for i in range(1, 21)]
    fc = A.force_convergence(hist)
    assert fc["CD_rel_change"] < 1e-12 and fc["CD_mean_drift"] < 0.01


# ---------------------------------------------------------------------- campaign, mesh study, dataset
def test_plan_and_resumable_campaign(tmp_path):
    space = C.space_from_variables(VARS)
    plan = C.plan_doe(space, 16, "maximin", seed=1)
    assert len(plan["cases"]) == 16
    assert plan["coverage"]["discrepancy"] < plan["random_reference"]["discrepancy"]
    assert plan["coverage"]["min_distance"] > plan["random_reference"]["min_distance"]
    calls = []
    recs = C.run_campaign(_sim(calls), plan["cases"][:5], str(tmp_path), log=lambda s: None)
    assert len(calls) == 5 and all(r.ok for r in recs)
    recs2 = C.run_campaign(_sim(calls), plan["cases"][:7], str(tmp_path), log=lambda s: None)
    assert len(calls) == 7 and len(recs2) == 7                    # the first five were not run again
    bad = C.run_campaign(_sim(calls), [{"id": "x", "a": 0.5, "b": 0.0, "c": 3.0}], str(tmp_path),
                         log=lambda s: None)
    assert bad[0].ok and os.path.exists(tmp_path / "fields" / "x.npz")
    tab = C.cost_table(C.load_records(str(tmp_path)), extra={"training": 36.0})
    assert tab["stages"]["doe"]["runs"] == 8 and tab["stages"]["training"]["cpu_h"] == pytest.approx(0.01)


def test_grid_convergence_recovers_order():
    N = [1e4, 2.83e4, 8e4, 2.26e5]                                 # h ratio ~ sqrt(2)
    h = (np.array(N) / N[-1]) ** (-1 / 3)
    levels = [{"name": f"L{i}", "cells": n, "values": {"q": 2.0 + 0.3 * hh ** 2}} for i, (n, hh) in enumerate(zip(N, h, strict=False))]
    s = M.grid_convergence(levels, ["q"])
    assert s["q"]["p"] == pytest.approx(2.0, abs=0.05)
    assert s["q"]["extrapolated"] == pytest.approx(2.0, abs=1e-3)
    pick = M.choose_mesh(s, "q", tol=0.35)                         # errors 120, 60, 30, 15 %
    assert pick["mesh"] == "L2" and pick["error"] <= 0.35
    assert M.choose_mesh(s, "q", tol=0.1)["mesh"] == "L3"           # none within 10 %: the finest, with a note


def _dataset(tmp_path, n=40):
    space = C.space_from_variables(VARS)
    plan = C.plan_doe(space, n, "lhs", seed=0)
    recs = C.run_campaign(_sim([]), plan["cases"], str(tmp_path), log=lambda s: None)
    ds = D.build_dataset(recs, ["a", "b", "c"], ["y", "z"], field_names=["f"], points=PTS,
                         regions={"left": PTS[:, 0] < 0.5}, bounds={v.name: (v.lower, v.upper) for v in VARS},
                         provenance={"source": "analytic test"})
    return ds.with_split(D.region_split(ds, holdout=lambda X: X[:, 0] > 0.85, seed=0))


def test_dataset_roundtrip_and_holdout(tmp_path):
    ds = _dataset(tmp_path)
    ex = ds.part("extrapolation")[0]
    assert len(ex) and (ex[:, 0] > 0.85).all()
    assert not set(ds.split["train"]) & set(ds.split["extrapolation"])
    p = ds.save(str(tmp_path / "data" / "toy"))
    spec = json.load(open(p))
    assert spec["kind"] == "DatasetSpec" and spec["files"][0]["sha256"]
    ds2 = D.StudyDataset.load(str(tmp_path / "data" / "toy"))
    assert ds2.ids == ds.ids and np.allclose(ds2.fields["f"], ds.fields["f"]) and ds2.split == ds.split


# ---------------------------------------------------------------------- benchmark and selection
def test_benchmark_select_and_physics_gate(tmp_path):
    from pinneapple_analysis.trust import Check
    ds = _dataset(tmp_path, 40)

    def always_bad(predict):
        return Check("bad", "contradicts")
    board = B.benchmark(ds, ["y"], "f", scalar_families=("gp", "quadratic"), field_families=("pod_gp", "pod_rbf"),
                        k=4, physics={"y": [lambda pr: Check("ok", "supports")], "f": []}, log=lambda s: None)
    sel = board.select()
    assert sel["choice"]["y"] in ("gp", "quadratic") and sel["choice"]["f"] in ("pod_gp", "pod_rbf")
    gp = board.rows["y/gp"]
    assert gp["cv_rel_rmse"] < 0.05 and 0.5 <= gp["test"]["coverage90_conformal"] <= 1.0
    board.physics["y/gp"] = [{"name": "bad", "passed": False}]
    board.physics["y/quadratic"] = [{"name": "bad", "passed": False}]
    assert board.select()["choice"]["y"] is None                      # nothing eligible
    mu, hw = board.predictor("f", "pod_gp")(ds.unit_x()[:3])
    assert mu.shape == (3, len(PTS)) and (hw > 0).all()


def test_conformal_quantile():
    assert B.conformal_q(np.arange(1, 10), 0.1) == 9.0
    assert math.isinf(B.conformal_q(np.arange(1, 5), 0.1))


# ---------------------------------------------------------------------- inverse
def test_mdn_bimodal_posterior_and_coverage():
    from pinneapple_analysis.inverse_problems.amortized import fit_mdn, posterior_coverage
    sim = lambda Z: (Z[:, :1] - 0.5) ** 2                      # noqa: E731 - z and 1 - z give the same y
    noise = lambda Y, rng: Y + 0.005 * rng.standard_normal(Y.shape)  # noqa: E731
    mdn = fit_mdn(sim, 1, noise=noise, n_sims=6000, budget_s=8.0, seed=0)
    S = mdn.sample(np.array([0.09]), 4000)[:, 0]                # z = 0.2 or 0.8
    assert 0.25 < np.mean(S < 0.5) < 0.75
    assert abs(np.median(S[S < 0.5]) - 0.2) < 0.05 and abs(np.median(S[S > 0.5]) - 0.8) < 0.05
    rng = np.random.default_rng(5)
    Z = rng.random((60, 1))
    cov = posterior_coverage(mdn, Z, noise(sim(Z), rng))
    assert 0.75 <= cov["coverage"]["0.9"][0] <= 1.0


# ---------------------------------------------------------------------- extras and twin
def test_sobol_acquire_pareto():
    bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    pred = lambda U: {"y": (U[:, 0] * 4 + 0.1 * U[:, 1], np.full(len(U), 0.01)),  # noqa: E731
                      "z": ((U[:, 0] - 1) ** 2 + U[:, 1], np.full(len(U), 0.01))}
    s = E.sobol_on_surrogate(pred, bounds, ["y"], n=1024)
    assert s["y"]["S1"][0] > 0.95 and s["y"]["ST"][1] < 0.05
    picks = E.acquire(lambda U: U[:, 0], bounds, np.zeros((0, 2)), 3, min_dist=0.2)
    assert len(picks) == 3 and all(p["a"] > 0.6 for p in picks)
    P = np.array([[p["a"], p["b"]] for p in picks])
    assert min(np.linalg.norm(P[i] - P[j]) for i in range(3) for j in range(i)) >= 0.2 - 1e-9
    par = E.pareto_on_surrogate(pred, bounds, [("y", "min"), ("z", "min")], budget=400)
    assert len(par["front"]) >= 3 and 1 <= len(par["picks"]) <= 3


def test_twin_payload_and_page(tmp_path):
    bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    pred = lambda U: {"y": (U[:, 0] + U[:, 1] ** 2, 0.05 + 0 * U[:, 0])}  # noqa: E731
    fpred = lambda U: (np.outer(U[:, 0], PTS[:, 0]) + U[:, 1:2], 0.1 + 0 * np.outer(U[:, 0], PTS[:, 0]))  # noqa: E731
    geo = lambda p: PTS * (1 + p["a"])                                      # noqa: E731
    pay = T.twin_payload(pred, fpred, bounds, geo, np.array([[0, 1, 2]]), np.array([[0.5, 0.5]]), grid=(11, 11),
                         shape_inputs=("a",))
    assert pay["interpolation_error"]["y"] < 0.01 and pay["interpolation_error"]["field_Cp"] < 1e-6
    assert len(pay["geometry"]["vertices"]) == 11 and max(pay["trust"]["distance"]) > 0.6
    T.write_twin(pay, str(tmp_path / "twin"))
    html = open(tmp_path / "twin" / "index.html").read()
    assert "twin.json" in html and os.path.exists(tmp_path / "twin" / "vendor" / "three" / "three.module.min.js")


# ---------------------------------------------------------------------- OpenFOAM (slow)
@pytest.mark.skipif(os.environ.get("PINNEAPPLE_SLOW") != "1", reason="set PINNEAPPLE_SLOW=1 (runs OpenFOAM)")
def test_ahmed_openfoam_run(tmp_path):
    from pinneapple_simulation.numerical_solvers.external_flow import openfoam_available
    if not openfoam_available():
        pytest.skip("OpenFOAM not found")
    sim = A.AhmedAero(mesh="L0", iterations=200, history_every=50, procs=2)
    rec = C.run_campaign(sim, [{"id": "t", **A.BASELINE}], str(tmp_path), log=lambda s: None)[0]
    assert rec.ok, rec.message
    o, m = rec.outputs, rec.meta
    assert 0.15 < o["CD"] < 0.6 and abs(o["CD_pressure"] + o["CD_friction"] - m["CD_final"]) < 1e-6
    assert abs(sum(o[f"CD_{r}"] for r in A.REGIONS) - o["CD_pressure"]) < 1e-9
    assert m["mesh_ok"] and m["mass_balance"]["imbalance"] < 0.01 and m["yplus"]["mean"] > 0
    cp = np.load(rec.fields_path)["cp"]
    assert cp.shape == (632,) and cp.max() <= 1.1


# ---------------------------------------------------------------------- Ahmed study stages on synthetic records
def _fake_records(tmp_path):
    from pinneapple_autolab.ledger import RunRecord
    os.makedirs(tmp_path / "fields", exist_ok=True)
    V0 = A.field_vertices(A.BASELINE)
    meta = {"cells": 0, "iteration_std": {"CD": 1e-3}, "yplus": {"p05": 40, "p95": 200, "frac_30_300": 0.95},
            "mesh_ok": True, "final_residuals": {"p": 1e-4}, "mass_balance": {"imbalance": 1e-4},
            "convergence": {"CD_mean_drift": 1e-3}}
    recs = []

    def rec(rid, purpose, p, cells=2e5):
        s = p["slant_deg"]
        cd = 0.25 + 0.002 * s + 0.02 * np.sin(np.radians(4 * s)) + 0.1 * (p["clearance"] - 0.05) - 2e-4 * p["speed"]
        out = {k: 0.0 for k in OUTS}
        out.update(CD=cd, CL=0.1 + 0.005 * s, CD_pressure=cd - 0.04, CD_friction=0.04 + 1e-3 * (60 - p["speed"]) / 40,
                   CD_slant=0.003 * s, CD_base=0.1)
        cp = 1 - (V0[:, 0] / 1.044) * (1.5 + 0.01 * s)
        fp = str(tmp_path / "fields" / f"{rid}.npz")
        np.savez(fp, cp=cp)
        recs.append(RunRecord(rid, purpose, dict(p), True, out, meta={**meta, "cells": cells}, fields_path=fp))
    from pinneapple_autolab.domains.ahmed_study import OUTPUTS as OUTS
    for m, n in (("L0", 6e4), ("L1", 1.7e5), ("L2", 4.8e5)):
        rec(f"mesh_{m}", "probe", A.BASELINE, cells=n)
        recs[-1].outputs["CD"] = 0.28 + 0.03 * (n / 4.8e5) ** (-2 / 3)
    for s in A.AHMED_1984_CD:
        rec(f"val_{s:04.1f}", "baseline", {"slant_deg": s, "clearance": 0.05, "speed": 60.0})
    rng = np.random.default_rng(0)
    for i in range(30):
        rec(f"doe{i:03d}", "doe", {"slant_deg": 40 * rng.random(), "clearance": 0.03 + 0.05 * rng.random(),
                                    "speed": 20 + 40 * rng.random()})
    return recs


def test_ahmed_study_stages(tmp_path):
    from pinneapple_autolab.domains import ahmed_study as S
    recs = _fake_records(tmp_path)
    gv = S.grid_and_validation(recs, str(tmp_path / "res"))
    assert gv["mesh_study"]["CD"]["p"] == pytest.approx(2.0, abs=0.05)
    assert len(gv["validation"]["rows"]) == 6 and os.path.exists(tmp_path / "res" / "figures" / "validation_cd.png")
    ds = S.study_dataset(recs)
    assert len(ds) == 36 and ds.fields["cp"].shape == (36, 632)
    assert all(ds.X[ds.ids.index(i), 0] > S.HOLDOUT_DEG for i in ds.split["extrapolation"])
    assert {"val_00.0", "val_12.5", "val_25.0", "val_30.0"} <= set(ds.split["test"])
    assert {"val_35.0", "val_40.0"} <= set(ds.split["extrapolation"])
    integ = S.template_integral(ds)
    assert np.isfinite(integ["tolerance"])


def test_report_and_evidence_package(tmp_path):
    from pinneapple_autolab.study import report as R
    ds = _dataset(tmp_path)
    spec = ds.save(str(tmp_path / "d" / "toy"))
    csv = tmp_path / "metrics.csv"
    csv.write_text("method,rmse\ngp,0.01\nquadratic,0.03\n")
    html = R.write_report("Toy", [{"title": "Data", "text": "two\n\nparagraphs", "decision": "REVIEW",
                                   "table": {"columns": ["a", "b"], "rows": [[1, 0.123456]]},
                                   "figures": ["nope.png"]}], str(tmp_path / "r.html"), base=str(tmp_path))
    page = open(html).read()
    assert "missing figure: nope.png" in page and "0.1235" in page and "REVIEW" in page
    pkg = R.evidence_package("Toy", [spec, str(csv)], str(tmp_path / "e.zip"))
    assert pkg["problems"] == [] and os.path.getsize(tmp_path / "e.zip") > 0
