"""HelioTFC: the solar-cycle envelope of the daily sunspot number, decomposed and fitted with an end-point-constrained
extreme learning machine (``pinneapple_systems.time_series.solar_cycle``).

* ``GET /v1/imf_summary``: (E)EMD only; every IMF's measured period and variance share, and the one nearest
  11 years (with a warning when even that one is far off).
* ``POST /v1/decompose_and_forecast`` / ``POST /v1/decompose_and_forecast_upload``: the whole pipeline on the live
  SILSO series, the synthetic demo series, or an uploaded SILSO-format CSV.

Data: ``data_source="live_silso"`` downloads https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv at request time
(SILSO, Royal Observatory of Belgium, Brussels; CC BY-NC 4.0: credit SILSO, non-commercial use only). No SILSO
data is stored in this repository; ``"synthetic"`` is a made-up series of the same shape for trying the API.

Run:  uvicorn helio_tfc.api:app --port 8149   (from apps/helio_tfc)

Environment (all optional; the rest in apps/_shared/README.md):
  HELIO_USER / HELIO_PASSWORD(_HASH)   login on everything except /health
  HELIO_FREE_FORECASTS                 free pipeline runs per caller and month (default 100)
  HELIO_ALLOW_LIVE                     0 disables the live SILSO download (default 1)
"""
from __future__ import annotations

import os
import sys
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from pinneapple_core.jsonable import to_jsonable
from pinneapple_systems.time_series import solar_cycle as sc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
ALLOW_LIVE = (os.environ.get("HELIO_ALLOW_LIVE") or "1").lower() not in ("0", "false", "no")

app = FastAPI(title="HelioTFC", version=VERSION,
              description="Solar-cycle envelope: EMD, Hilbert amplitude, end-point-constrained ELM fit and forecast.")
install(app, prefix="HELIO")
rt = install_runtime(app, prefix="HELIO", name="helio_tfc", units={"forecasts": 100, "summaries": 1000},
                     upload_suffixes={".csv", ".txt"})

Source = Literal["live_silso", "synthetic"]


class ForecastRequest(BaseModel):
    data_source: Source = "synthetic"
    start_date: str | None = None
    end_date: str | None = None
    decomposition_method: Literal["emd", "eemd"] = "emd"
    max_imf: int = Field(6, ge=1, le=12)
    eemd_trials: int = Field(16, ge=2, le=100)
    eemd_noise_width: float = Field(0.05, gt=0, le=1)
    imf_index: int | Literal["auto"] = "auto"
    hidden_units: int = Field(50, ge=2, le=500)
    forecast_horizon_frac: float = Field(1.0, gt=0, le=3)
    seed: int = Field(0, ge=0)


def _data(source: str, start: str | None, end: str | None):
    if source == "live_silso":
        if not ALLOW_LIVE:
            raise HTTPException(403, "the live SILSO download is disabled on this server (HELIO_ALLOW_LIVE=0)")
        try:
            df = sc.fetch_silso_daily()
        except Exception as e:  # noqa: BLE001 -- network or parse failure, reported, never replaced by other data
            raise HTTPException(503, f"could not download the SILSO series: {e}") from None
    else:
        df = sc.synthetic_daily()
    df = df.loc[start:end] if (start or end) else df
    if len(df) < 100:
        raise HTTPException(422, f"only {len(df)} daily values in that range; need at least 100")
    return df


def _credit(source: str) -> str:
    return {"live_silso": sc.SILSO_CREDIT, "upload": "uploaded series (credit its source; SILSO data is CC BY-NC 4.0)"
            }.get(source, "synthetic series, not solar data")


def _run(df, req: ForecastRequest, source: str) -> dict:
    try:
        out = sc.run_pipeline(df, decomposition_method=req.decomposition_method, max_imf=req.max_imf,
                              eemd_trials=req.eemd_trials, eemd_noise_width=req.eemd_noise_width,
                              imf_index=req.imf_index, hidden_units=req.hidden_units,
                              forecast_horizon_frac=req.forecast_horizon_frac, seed=req.seed)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    out["data_credit"] = _credit(source)
    return to_jsonable(out)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/v1/imf_summary")
def imf_summary(request: Request, data_source: Source = "synthetic", start_date: str | None = None,
                end_date: str | None = None, max_imf: int = Query(6, ge=1, le=12),
                method: Literal["emd", "eemd"] = "emd"):
    rt.consume(request, "summaries")
    df = _data(data_source, start_date, end_date)
    _, y, _ = sc.preprocess(df)
    dec = sc.decompose(y, method, max_imf=max_imf)
    dt = float((df.index[-1] - df.index[0]).days) / max(len(df) - 1, 1)
    stats = sc.imf_statistics(dec.imfs, dt_days=dt)
    best, warning = sc.closest_to_schwabe_cycle(stats)
    return to_jsonable({"n_samples": len(df), "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
                        "backend": dec.backend, "n_imfs": int(dec.imfs.shape[0]), "imf_statistics": stats,
                        "closest_to_11yr_cycle": best, "closest_match_warning": warning,
                        "data_credit": _credit(data_source)})


@app.post("/v1/decompose_and_forecast")
def decompose_and_forecast(request: Request, req: ForecastRequest):
    rt.consume(request, "forecasts")
    return _run(_data(req.data_source, req.start_date, req.end_date), req, req.data_source)


@app.post("/v1/decompose_and_forecast_upload")
async def decompose_and_forecast_upload(request: Request, file: UploadFile = File(...),  # noqa: B008
                                        decomposition_method: Literal["emd", "eemd"] = Form("emd"),
                                        max_imf: int = Form(6, ge=1, le=12), hidden_units: int = Form(50, ge=2, le=500),
                                        forecast_horizon_frac: float = Form(1.0, gt=0, le=3), seed: int = Form(0)):
    rt.consume(request, "forecasts")
    try:
        df = sc.parse_silso_csv(await file.read())
    except Exception as e:  # noqa: BLE001 -- not a SILSO-format CSV
        raise HTTPException(422, f"not a SILSO-format daily CSV (';'-separated, 8 columns): {e}") from None
    if len(df) < 100:
        raise HTTPException(422, f"only {len(df)} daily values; need at least 100")
    req = ForecastRequest(decomposition_method=decomposition_method, max_imf=max_imf, hidden_units=hidden_units,
                          forecast_horizon_frac=forecast_horizon_frac, seed=seed)
    return _run(df, req, "upload")
