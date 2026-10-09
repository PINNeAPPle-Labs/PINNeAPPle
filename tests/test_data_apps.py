"""Engineering Data Health, Data Standardizer and Simulation Metadata (apps 3-5) and their library modules."""
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from pinneapple_data.physical_units import convert, infer_quantity, parse_unit, split_header, try_parse_unit

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXAMPLES = os.path.join(ROOT, "apps", "simulation_metadata", "examples")


# ── physical units ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("src,dst,x,y", [
    ("°C", "°F", 100.0, 212.0), ("°F", "°C", 32.0, 0.0), ("K", "°C", 273.15, 0.0), ("bar", "psi", 1.0, 14.503773773),
    ("gpm", "m^3/h", 100.0, 22.712470704), ("TR", "kW", 1.0, 3.516852842), ("kWh", "MJ", 1.0, 3.6),
    ("m3/h", "L/s", 3.6, 1.0), ("kg/(m·s)", "cP", 1.0, 1000.0), ("W/(m^2*K)", "W/(m2*K)", 5.0, 5.0),
    ("rpm", "Hz", 60.0, 1.0), ("inH2O", "Pa", 1.0, 249.08891)])
def test_unit_conversions_exact(src, dst, x, y):
    assert convert(x, src, dst) == pytest.approx(y, rel=1e-9, abs=1e-9)
    assert convert(convert(x, src, dst), dst, src) == pytest.approx(x, rel=1e-12, abs=1e-12)


def test_temperature_difference_has_no_offset():
    assert convert(18.0, "°F", "°C", difference=True) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        convert(1.0, "bar", "kW")


@pytest.mark.parametrize("header,name,unit,quantity", [
    ("T_supply [°C]", "T_supply", "°C", "temperature"), ("Pressure (bar)", "Pressure", "bar", "pressure"),
    ("CHW_flow_gpm", "CHW_flow", "gpm", "volumetric_flow_rate"), ("T_out_F", "T_out", "F", "temperature"),
    ("Vibration_mm_s", "Vibration", "mm/s", "velocity"), ("CondP_psig", "CondP", "psig", "pressure"),
    ("Line_A", "Line_A", None, None), ("RH (%)", "RH", "%", "relative_humidity")])
def test_header_units_and_quantities(header, name, unit, quantity):
    n, u = split_header(header)
    assert (n, u) == (name, unit)
    assert infer_quantity(n, try_parse_unit(u)).quantity == quantity
    if unit == "psig":
        assert parse_unit(u).gauge


# ── data health ─────────────────────────────────────────────────────────────

from pinneapple_data.data_health import analyze, example_chiller_plant, load_table  # noqa: E402

FAULT_CHECK = {
    "energy_balance": lambda i: i["category"] == "physics_consistency",
    "stuck": lambda i: "Stuck" in i["title"], "unit_switch": lambda i: "Unit changes" in i["title"],
    "impossible": lambda i: "impossible" in i["title"] and "RH" in (i["column"] or ""),
    "spikes": lambda i: i["title"] == "Spikes", "sentinel": lambda i: "Error code" in i["title"],
    "gap": lambda i: "gap" in i["title"], "duplicates": lambda i: i["title"] == "Duplicate timestamps",
}


def test_health_finds_every_injected_fault():
    df, key = example_chiller_plant()
    r = analyze(df)
    for f in key["faults"]:
        assert any(FAULT_CHECK[f["type"]](i) for i in r["issues"]), f
    assert r["summary"]["sampling"] == "1.0 min"
    assert 60 < r["score"]["overall"] < 85 and r["score"]["level"] == "warn"
    assert r["relations"] and r["relations"][0]["columns"]["flow"] == "CHW_flow [m3/h]"
    json.dumps(r)


def test_health_clean_data_scores_high_without_warnings():
    df, _ = example_chiller_plant(inject=False)
    r = analyze(df)
    assert r["score"]["overall"] > 95
    assert not [i for i in r["issues"] if i["severity"] != "info"]


def test_health_same_result_across_formats():
    pytest.importorskip("openpyxl")
    pytest.importorskip("pyarrow")
    df, _ = example_chiller_plant()
    ref = analyze(df)["score"]["overall"]
    eu = df.copy()
    eu["Timestamp"] = pd.to_datetime(eu["Timestamp"]).dt.strftime("%d/%m/%Y %H:%M")
    variants = {"plant.csv": eu.to_csv(sep=";", decimal=",", index=False).encode("cp1252"),
                "plant.json": df.to_json(orient="records").encode()}
    b = io.BytesIO(); df.to_excel(b, index=False); variants["plant.xlsx"] = b.getvalue()
    b = io.BytesIO(); df.to_parquet(b); variants["plant.parquet"] = b.getvalue()
    for name, data in variants.items():
        x, info = load_table(data, name)
        r = analyze(x, units=info.units_from_file)
        assert r["summary"]["start"] == "2026-03-02 00:00", name
        assert r["score"]["overall"] == pytest.approx(ref, abs=1.0), name


# ── standardizer ─────────────────────────────────────────────────────────────

def _example_run(tz_override=None):
    pytest.importorskip("openpyxl")
    from pinneapple_data.unified import EXAMPLE_TIMEZONES, example_sources, make_recipe, propose_mapping, standardize
    loaded, maps = [], []
    for name, raw in example_sources():
        df, info = load_table(raw, name)
        maps.append(propose_mapping(df, name, tz=(tz_override or EXAMPLE_TIMEZONES).get(name)))
        loaded.append((name, df, raw))
    return standardize(loaded, make_recipe(maps)), maps


def test_standardizer_three_systems_agree():
    res, maps = _example_run()
    scada = {c["source"]: c for c in maps[1]["columns"]}
    assert scada["chws_temp_F"]["quantity"] == "temperature" and scada["chws_temp_F"]["target"] == "chw_supply_temperature"
    assert scada["cooling_load_tons"]["target_unit"] == "kW"
    assert res["resample"] == "5min" and len(res["agreement"]) == 9
    assert all(a["verdict"] == "agree" for a in res["agreement"]), res["agreement"]
    assert res["wide"]["time_utc"].iloc[0] == pd.Timestamp("2026-03-10 00:00")


def test_standardizer_catches_wrong_time_zone():
    res, _ = _example_run(tz_override={"bms_export.csv": "UTC"})
    shifted = [a for a in res["agreement"] if a["verdict"] == "time shift"]
    assert shifted and all(abs(a["shift_hours"]) == 3 for a in shifted)


def test_standardizer_outputs_carry_units():
    pytest.importorskip("pyarrow")
    pytest.importorskip("h5py")
    from pinneapple_data.unified import write
    res, _ = _example_run()
    data, _, _ = write(res, "parquet")
    back, info = load_table(data, "u.parquet")
    assert info.units_from_file["chw_flow@scada_api"] == "m^3/h"
    data, _, _ = write(res, "hdf5")
    back, info = load_table(data, "u.h5")
    assert info.units_from_file["cooling_load@bms_export"] == "kW"
    z = zipfile.ZipFile(io.BytesIO(write(res, "zip")[0]))
    assert {"manifest.json", "recipe.json", "unified_wide.csv"} <= set(z.namelist())


# ── simulation metadata ─────────────────────────────────────────────────────

from pinneapple_data.simulation_metadata import extract, read_upload  # noqa: E402
from pinneapple_data.simulation_metadata.foam_dict import parse  # noqa: E402


def _ex(name):
    with open(os.path.join(EXAMPLES, f"{name}.zip"), "rb") as f:
        return extract(read_upload([(f"{name}.zip", f.read())]))


def test_foam_dict_parser():
    d = parse('''FoamFile { class dictionary; object fvSchemes; }
      divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); } // comment
      solvers { "(U|k)" { solver smoothSolver; tolerance 1e-05; } }
      #include "common" nu [0 2 -1 0 0 0 0] 1e-05;''')
    assert d["divSchemes"]["div(phi,U)"] == "bounded Gauss linearUpwind grad(U)"
    assert d["solvers"]["(U|k)"]["tolerance"] == "1e-05"
    assert d["__directives__"] == ['#include "common"']


@pytest.mark.parametrize("name,status", [
    ("openfoam_pitzDaily_converged", "converged"), ("openfoam_pitzDaily_stopped_early", "still_falling"),
    ("openfoam_pitzDaily_diverged", "diverged"), ("openfoam_cavity_transient", "completed"),
    ("calculix_cantilever_nlgeom", "converged")])
def test_simulation_metadata_verdicts(name, status):
    assert _ex(name)["convergence"]["status"] == status


def test_simulation_metadata_openfoam_record():
    r = _ex("openfoam_pitzDaily_converged")
    assert r["solver"]["application"] == "simpleFoam" and r["solver"]["version"] == "v1912"
    assert r["analysis"]["turbulence"] == {"type": "RAS", "model": "kEpsilon"}
    assert r["mesh"]["cells"] == 12225 and r["mesh"]["quality"]["status"] == "Mesh OK"
    inlet = next(b for b in r["boundary_conditions"] if b["field"] == "U" and b["patch"] == "inlet")
    assert inlet["type"] == "fixedValue" and inlet["value"] == "uniform (10 0 0)"
    assert any(p["name"] == "kinematic viscosity ν" and p["value"] == 1e-5 for p in r["parameters"])
    assert r["convergence"]["fields"]["p"]["met"] is True
    json.dumps(r)


def test_simulation_metadata_calculix_matches_beam_theory():
    r = _ex("calculix_cantilever_nlgeom")
    assert r["mesh"]["nodes"] == 1025 and r["mesh"]["element_types"] == {"C3D8I": 640}
    assert r["loads"][0]["total"] == pytest.approx(-500.0)
    tip = next(x for x in r["results"] if "TIP" in x["name"])["value"]
    e_b = 500 * 200 ** 3 / (3 * 210000 * 10 * 10 ** 3 / 12)          # F L^3 / 3 E I = 7.62 mm
    assert tip == pytest.approx(e_b, rel=0.02)


def test_simulation_metadata_residual_table():
    it = np.arange(1, 301)
    csv = "Inner_Iter,rms[Rho],rms[RhoU],CL\n" + "\n".join(
        f"{i},{-1 - 5 * i / 300:.4f},{-0.5 - 4 * i / 300:.4f},{0.3 + 0.1 * math.exp(-i / 40):.5f}" for i in it)
    r = extract(read_upload([("history.csv", csv.encode())]))
    assert r["detected"]["format"] == "SU2 history"
    assert r["convergence"]["status"] in ("converged", "still_falling")
    assert r["convergence"]["monitors"][0]["name"] == "CL"


# ── web APIs ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def clients():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    pytest.importorskip("openpyxl")
    from fastapi.testclient import TestClient
    out = {}
    for app_dir, mod in (("data_health", "data_health.api"), ("data_standardizer", "data_standardizer.api"),
                         ("simulation_metadata", "simulation_metadata.api")):
        sys.path.insert(0, os.path.join(ROOT, "apps", app_dir))
        out[app_dir] = TestClient(__import__(mod, fromlist=["app"]).app)
    return out


def test_apis(clients):
    h = clients["data_health"]
    r = h.get("/api/example").json()
    assert len(r["answer_key"]["faults"]) == 8 and r["scope"]["validated"]
    csv = h.get("/api/example.csv").content
    assert h.post("/api/analyze", files={"file": ("p.csv", csv)}).json()["score"]["overall"] > 60
    assert h.post("/api/analyze", files={"file": ("x.pdf", b"%PDF")}).status_code == 422

    s = clients["data_standardizer"]
    ins = s.get("/api/example/inspect").json()
    assert len(ins["sources"]) == 3
    prev = s.post("/api/example/standardize", data={"recipe": json.dumps(ins["recipe"]), "format": "preview"}).json()
    assert all(a["verdict"] == "agree" for a in prev["agreement"])
    assert s.post("/api/example/standardize", data={"recipe": json.dumps(ins["recipe"]), "format": "zip"}).content[:2] == b"PK"
    assert s.get("/api/convert", params={"src": "°F", "dst": "°C"}).json()["text"] == "× 0.555556 − 17.7778"

    m = clients["simulation_metadata"]
    assert len(m.get("/api/meta").json()["examples"]) == 5
    assert m.get("/api/example/openfoam_pitzDaily_diverged").json()["convergence"]["status"] == "diverged"
    with open(os.path.join(EXAMPLES, "calculix_cantilever_nlgeom.zip"), "rb") as f:
        z = zipfile.ZipFile(f)
        files = [("files", (n.split("/")[-1], z.read(n))) for n in z.namelist()]
    assert m.post("/api/extract", files=files).json()["solver"]["name"] == "CalculiX"
    assert m.post("/api/extract", files=[("files", ("notes.txt", b"hello"))]).status_code == 422


# ── process optimization (plain ML) ─────────────────────────────────────────

def test_process_optimizer_saving_matches_true_physics():
    pytest.importorskip("sklearn")
    from pinneapple_data import process_optimizer as po
    df, info = po.example_plant_operations()
    rep = analyze(df)
    roles = po.suggest_roles(rep["columns"])
    assert roles["target"] == "Plant_power [kW]" and set(roles["levers"]) == set(info["levers"])
    assert "CHW_return_temp [°C]" not in roles["context"]            # an outcome, not context
    clean, log = po.clean_for_modeling(df, rep)
    assert any("error-code" in x for x in log)
    r = po.fit_and_optimize(clean, info["target"], info["levers"], info["context"], constraints=info["constraints"])
    assert r["model"]["r2"] > 0.95 and r["model"]["mae"] < 0.15 * r["model"]["baseline_mae"]
    cv = r["model"]["cv"]                                             # time-series CV picks the regularisation (#81)
    assert cv["n_folds"] >= 3 and all(f["train_rows"] < r["rows"]["train"] for f in cv["folds"])
    assert [f["train_rows"] for f in cv["folds"]] == sorted(f["train_rows"] for f in cv["folds"])
    assert r["model"]["l2_regularization"] == cv["chosen_l2"] and cv["mean_r2"] > 0.8
    truth = po.true_saving(r)
    assert 4 < r["saving"]["pct"] < 15 and abs(r["saving"]["pct"] - truth["true_saving_pct"]) < 2.5
    assert truth["true_constraint_violations_pct"] < truth["true_constraint_violations_before_pct"]
    assert r["saving"]["range_pct"][0] <= r["saving"]["pct"] <= r["saving"]["range_pct"][1]
