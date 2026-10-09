"""Engineering Data Standardizer web API + single-page UI.

Run:  uvicorn data_standardizer.api:app --port 8083   (from apps/data_standardizer)

Environment (all optional):
  EDS_USER / EDS_PASSWORD   HTTP Basic login on everything except /health
  EDS_MAX_MB                largest upload per request in MB (default 80)
  EDS_MAX_FILES             files per request (default 8)
  EDS_MAX_HEAVY             concurrent runs per worker (default 2)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.data_health import load_table
from pinneapple_data.physical_units import QUANTITIES, UnitError
from pinneapple_data.unified import (EXAMPLE_TIMEZONES, UNIT_CHOICES, UNIT_SYSTEMS, conversion_text, example_sources,
                                     make_recipe, propose_mapping, standardize, write)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_MB = float(os.environ.get("EDS_MAX_MB") or "80")
MAX_FILES = int(os.environ.get("EDS_MAX_FILES") or "8")
FORMATS = ["csv", "parquet", "hdf5", "json", "zip", "manifest", "recipe"]
TIMEZONES = ["UTC", "America/Sao_Paulo", "America/New_York", "America/Chicago", "America/Denver",
             "America/Los_Angeles", "America/Mexico_City", "America/Bogota", "America/Santiago",
             "America/Argentina/Buenos_Aires", "Europe/London", "Europe/Lisbon", "Europe/Madrid", "Europe/Paris",
             "Europe/Berlin", "Europe/Rome", "Europe/Amsterdam", "Europe/Stockholm", "Europe/Warsaw", "Europe/Moscow",
             "Africa/Johannesburg", "Asia/Dubai", "Asia/Kolkata", "Asia/Singapore", "Asia/Shanghai", "Asia/Tokyo",
             "Asia/Seoul", "Australia/Sydney", "Pacific/Auckland"]

app = FastAPI(title="Engineering Data Standardizer", version=VERSION,
              description="Many files, many conventions, one physical schema (PINNeAPPle Unified Physical Data). "
                          "POST files to /api/inspect for a draft mapping, then to /api/standardize with the recipe.")
install(app, prefix="EDS")
_HEAVY = BusyLimiter("EDS_MAX_HEAVY")

VALIDATION = [
    "One chiller recorded by three systems (BMS CSV in local time and °C with decimal commas, SCADA JSON in Unix ms, "
    "°F, gpm and tons, test-rig Excel in K, L/s and W): after standardization every pair agrees within sensor noise "
    "(median differences 0.01–0.04 in the target unit)",
    "A source given the wrong time zone is caught: the sources match best when shifted by exactly the 3 h offset",
    "Unit conversions use exact definitions (NIST SP 811), including temperature offsets; round trips exact to 1e-12",
    "Parquet output carries units in its schema metadata and HDF5 output in dataset attributes, so other tools "
    "(and Engineering Data Health) read them back",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Conversion scope & validation", "noun": "limitation",
            "items_title": "What the standardizer assumes",
            "band": "Every conversion applied is written to the manifest, column by column.",
            "tags": {"conservative": "Safe by default", "optimistic": "Review before use", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Units come from headers or your mapping", "effect": "optimistic",
                 "detail": "A column whose header carries no unit is left out until you give it one; a wrong unit in "
                           "a header is converted faithfully (wrongly).",
                 "today": "Review the mapping table; run Engineering Data Health on the sources to catch unit switches.",
                 "planned": "Unit suggestions from value ranges and from tag dictionaries (BACnet, OPC UA, ISA-5.1)."},
                {"topic": "Local time without an offset", "effect": "check",
                 "detail": "Timestamps without 'Z' or '+hh:mm' are read in the time zone you choose; local times that "
                           "repeat or do not exist at a daylight-saving change are dropped and counted.",
                 "today": "Set each source's time zone; check the cross-source agreement for time-shift warnings.",
                 "planned": "Per-source DST disambiguation using the row order."},
                {"topic": "Temperature differences", "effect": "check",
                 "detail": "A ΔT column must be marked as a difference, or °F → °C applies the −32 offset to it. "
                           "Columns named delta/diff/Δ are marked automatically.",
                 "today": "Tick 'difference' for ΔT columns the name does not reveal.",
                 "planned": "Detection from paired supply/return columns."},
                {"topic": "Resampling", "effect": "check",
                 "detail": "Aligning sources on a common grid averages within each interval (mean), which is right for "
                           "analog signals and wrong for counters and states.",
                 "today": "Choose 'none' to keep original timestamps, or keep counters in a separate run.",
                 "planned": "Per-variable aggregation (mean, last, sum, max)."},
            ]}


def _check_sizes(files: List[Tuple[str, bytes]]) -> None:
    if not files:
        raise HTTPException(422, "No files uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"At most {MAX_FILES} files per request.")
    if sum(len(b) for _, b in files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload larger than {MAX_MB:g} MB in total.")


def _load(files: List[Tuple[str, bytes]]):
    out = []
    for name, raw in files:
        try:
            df, info = load_table(raw, name)
        except Exception as e:
            raise HTTPException(422, f"Could not read '{name}': {e}") from e
        out.append((name, raw, df, info))
    return out


def _preview_rows(df: pd.DataFrame, k: int = 6) -> List[List[str]]:
    return [[("" if pd.isna(v) else str(v))[:40] for v in row] for row in df.head(k).itertuples(index=False)]


def _inspect(files: List[Tuple[str, bytes]], system: str, tz: Dict[str, str]) -> Dict[str, Any]:
    _check_sizes(files)
    if system not in UNIT_SYSTEMS:
        raise HTTPException(422, f"Unknown unit system '{system}'.")
    with _HEAVY:
        out = []
        for name, raw, df, info in _load(files):
            m = propose_mapping(df, name, system=system, units=info.units_from_file, tz=tz.get(name))
            out.append({"mapping": m, "file": {"name": name, "bytes": len(raw), "format": info.format,
                                                "rows": info.rows, "columns": info.columns, "sheet": info.sheet,
                                                "delimiter": info.delimiter, "decimal": info.decimal},
                        "header": list(map(str, df.columns)), "preview": _preview_rows(df)})
    return {"sources": out, "recipe": make_recipe([s["mapping"] for s in out], system=system), "scope": scope()}


def _down(x: pd.Series, y: pd.Series, k: int = 500) -> Dict[str, list]:
    n = len(x)
    if n == 0:
        return {"x": [], "y": []}
    idx = np.linspace(0, n - 1, min(k, n)).astype(int)
    return {"x": [pd.Timestamp(v).strftime("%Y-%m-%d %H:%M") for v in x.iloc[idx]],
            "y": [None if not np.isfinite(v) else float(v) for v in y.iloc[idx].to_numpy(dtype=float)]}


def _charts(res: Dict[str, Any]) -> Dict[str, Any]:
    """Per variable: raw values per source (original units) and standardized values on one time axis."""
    out = {}
    wide, raw = res["wide"], res["raw"]
    grid = wide["time_utc"]
    for v in res["manifest"]["variables"]:
        name = v["name"]
        before, after = [], []
        for s in v["sources"]:
            r = raw[(raw["variable"] == name) & (raw["source"] == s["tag"])].sort_values("time_utc")
            # project raw values on the common grid (nearest earlier sample) so both charts share the axis
            rr = pd.merge_asof(pd.DataFrame({"time_utc": grid}), r[["time_utc", "value"]], on="time_utc",
                               direction="nearest", tolerance=pd.Timedelta("1h"))
            col = f"{name}@{s['tag']}" if f"{name}@{s['tag']}" in wide.columns else name
            mask = wide[col].isna().to_numpy()
            yb = rr["value"].where(~mask)
            before.append({"name": f"{s['tag']} [{s['unit']}]", "y": _down(grid, yb)["y"]})
            after.append({"name": f"{s['tag']} [{v['unit']}]", "y": _down(grid, wide[col])["y"]})
        out[name] = {"x": _down(grid, wide[wide.columns[1]])["x"], "unit": v["unit"], "before": before, "after": after}
    return out


def _standardize(files: List[Tuple[str, bytes]], recipe_text: str, fmt: str, layout: str):
    _check_sizes(files)
    try:
        recipe = json.loads(recipe_text)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"recipe is not valid JSON: {e}") from e
    if fmt not in FORMATS + ["preview"]:
        raise HTTPException(422, f"format must be one of {FORMATS + ['preview']}")
    with _HEAVY:
        loaded = _load(files)
        try:
            res = standardize([(n, df, raw) for n, raw, df, _ in loaded], recipe)
        except (ValueError, UnitError, KeyError) as e:
            raise HTTPException(422, str(e)) from e
        if fmt == "preview":
            head = res["wide"].head(12).copy()
            head["time_utc"] = head["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
            return {"manifest": res["manifest"], "agreement": res["agreement"], "notes": res["notes"],
                    "resample": res["resample"], "rows": int(len(res["wide"])), "columns": list(res["wide"].columns),
                    "head": json.loads(head.to_json(orient="values")), "long_rows": int(len(res["long"])),
                    "charts": _charts(res), "scope": scope()}
        data, media, name = write(res, fmt, layout)
        return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


async def _read(files: List[UploadFile]) -> List[Tuple[str, bytes]]:
    return [(f.filename or f"upload_{i}.csv", await f.read()) for i, f in enumerate(files)]


def _example_files() -> List[Tuple[str, bytes]]:
    return example_sources()


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "unit_systems": {k: v["label"] for k, v in UNIT_SYSTEMS.items()},
            "unit_choices": UNIT_CHOICES, "quantities": sorted(QUANTITIES), "timezones": TIMEZONES,
            "formats": FORMATS, "max_mb": MAX_MB, "max_files": MAX_FILES}


@app.get("/api/convert")
def api_convert(src: str, dst: str, difference: bool = False):
    """Describe a conversion (or why it is impossible)."""
    return {"text": conversion_text(src, dst, difference)}


@app.post("/api/inspect")
async def api_inspect(files: List[UploadFile] = File(...), system: str = Form("metric"),
                      timezones: Optional[str] = Form(None)):
    """Read the files and draft a mapping (the recipe) from their headers."""
    tz = json.loads(timezones) if timezones else {}
    return _inspect(await _read(files), system, tz)


@app.post("/api/standardize")
async def api_standardize(files: List[UploadFile] = File(...), recipe: str = Form(...), format: str = Form("zip"),
                          layout: str = Form("wide")):
    """Apply a recipe. format: preview (JSON summary) | csv | parquet | hdf5 | json | zip | manifest | recipe."""
    return _standardize(await _read(files), recipe, format, layout)


@app.get("/api/example/inspect")
def api_example_inspect(system: str = "metric"):
    return _inspect(_example_files(), system, EXAMPLE_TIMEZONES)


@app.post("/api/example/standardize")
def api_example_standardize(recipe: str = Form(...), format: str = Form("preview"), layout: str = Form("wide")):
    return _standardize(_example_files(), recipe, format, layout)


@app.get("/api/example/{name}")
def api_example_file(name: str):
    for n, b in _example_files():
        if n == name:
            media = {"csv": "text/csv", "json": "application/json"}.get(n.rsplit(".", 1)[-1],
                                                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            return Response(b, media_type=media, headers={"Content-Disposition": f'attachment; filename="{n}"'})
    raise HTTPException(404, "No such example file.")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
