"""The ENSO study data set: monthly indices and maps built from the cached sources of :mod:`.data`.

:func:`build_dataset` returns an :class:`EnsoDataset` with, from ``start`` to ``end``:

* Nino 3.4 / 3 / 4 / 1+2 SST anomalies [K] from OISST v2.1 (box means of 1 deg monthly means);
* SST anomaly maps on the intermediate model's grid (default 2 deg x 1 deg, 120E-80W, 20S-20N);
* D20 anomalies [m] in 10 deg longitude bins (130E-80W) for 5S-5N and 2S-2N, gaps filled month by month from the
  available bins with the leading EOFs of the base period (:func:`eof_fill`; no month uses another month's data);
  the warm-water-volume proxy ``wwv`` is their zonal mean over 5S-5N, ``h_west`` / ``h_east`` the means west / east
  of 155W (the tilt mode);
* ERA5 zonal wind stress anomalies on the model grid and a Tahiti - Darwin pressure index (1959-2022 only; NaN
  after 2022-12).

All anomalies are taken from calendar-month climatologies of the ``base`` period -- choose it inside the
development period so the test period does not inform them. ``checks`` holds the cross-checks against independent
sources (statsmodels Nino 1+2, NCEI heat content, ERA5 SST).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import data as D
from .intermediate import OceanGrid

__all__ = ["EnsoDataset", "build_dataset", "eof_fill", "to_series", "COLUMNS"]

COLUMNS = ("nino34", "wwv", "h_west", "h_east", "nino4", "nino3", "nino12")


@dataclass
class EnsoDataset:
    months: list[str]
    indices: dict[str, np.ndarray]
    sst_maps: np.ndarray                          # (nt, ny, nx) anomalies on ``grid``
    grid: OceanGrid
    d20_band: np.ndarray                          # (nt, nb) 5S-5N anomalies, gap-filled
    d20_eq: np.ndarray                            # (nt, nb) 2S-2N anomalies, gap-filled
    d20_lon: np.ndarray                           # bin centres
    d20_observed: np.ndarray                      # (nt, nb) bool, bin had casts (5S-5N)
    taux: np.ndarray                              # (nt, ny, nx + 1) ERA5 stress anomalies at u points (NaN after 2022)
    base: tuple[str, str]
    clim: dict[str, np.ndarray] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)

    @property
    def month_of_year(self) -> np.ndarray:
        return np.array([int(m[5:7]) - 1 for m in self.months])

    def index_of(self, month: str) -> int:
        return self.months.index(month)


def eof_fill(X: np.ndarray, base_rows: np.ndarray, n_eofs: int = 4, ridge: float = 0.1) -> np.ndarray:
    """Fill NaNs of each row of X (nt, nb) from its observed entries: least-squares amplitudes of the leading
    ``n_eofs`` EOFs (computed on the complete rows among ``base_rows``), ridge-regularised in units of the EOF
    variances. Observed entries are kept as they are. Rows with fewer observed entries than EOFs keep their NaNs."""
    X = np.asarray(X, float)
    full = base_rows & np.isfinite(X).all(1)
    A = X[full]
    if len(A) < 2 * n_eofs:
        raise ValueError("too few complete rows in the base period for the EOFs")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    E = Vt[:n_eofs].T                                       # (nb, k)
    var = (s[:n_eofs] ** 2) / len(A)
    out = X.copy()
    for t in range(len(X)):
        ok = np.isfinite(X[t])
        if ok.all() or ok.sum() < n_eofs:
            continue
        Eo = E[ok]
        a = np.linalg.solve(Eo.T @ Eo + ridge * np.diag(1.0 / var) * np.mean(var), Eo.T @ X[t, ok])
        out[t, ~ok] = E[~ok] @ a
    return out


def _locf(X: np.ndarray) -> np.ndarray:
    """Rows that are still NaN take the previous row (last observation carried forward; causal); leading NaN rows
    become zero anomalies."""
    X = X.copy()
    for t in range(len(X)):
        bad = ~np.isfinite(X[t])
        if bad.any():
            X[t, bad] = X[t - 1, bad] if t else 0.0
    return X


def _regrid_maps(g_src: D.Gridded, grid: OceanGrid, var: str = "sst") -> np.ndarray:
    """Block means of a 1 deg field onto the model grid (cells of dlon x dlat whole degrees)."""
    ky, kx = int(round(grid.dlat)), int(round(grid.dlon))
    iy0 = int(np.argmin(np.abs(g_src.lat - (grid.lat[0] - grid.dlat / 2 + 0.5))))
    ix0 = int(np.argmin(np.abs(g_src.lon - (grid.lon[0] - grid.dlon / 2 + 0.5))))
    a = g_src.values[var][:, iy0:iy0 + grid.ny * ky, ix0:ix0 + grid.nx * kx]
    a = a.reshape(len(a), grid.ny, ky, grid.nx, kx)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=(2, 4))


def _interp_era5(g: D.Gridded, var: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a global ERA5 field (t, lat, lon) to (lat, lon) points (lon periodic)."""
    from scipy.interpolate import RegularGridInterpolator
    lo = np.concatenate([g.lon - 360, g.lon, g.lon + 360])
    out = np.empty((len(g.time), len(lat), len(lon)))
    P = np.stack(np.meshgrid(lat, lon, indexing="ij"), -1).reshape(-1, 2)
    for t in range(len(g.time)):
        f = np.concatenate([g.values[var][t]] * 3, axis=1)
        out[t] = RegularGridInterpolator((g.lat, lo), f)(P).reshape(len(lat), len(lon))
    return out


def build_dataset(start: str = "1982-01", end: str = "2024-12", base: tuple[str, str] = ("1982-01", "2011-12"),
                  grid: OceanGrid | None = None, *, cache_dir: str | None = None, kinds=("mrb", "xbt", "pfl"),
                  manifests: dict[str, str] | None = None, n_eofs: int = 4, log=print) -> EnsoDataset:
    grid = grid or OceanGrid(dlon=2.0, dlat=1.0, lat_max=20.0)
    man = manifests or {}
    months = D.month_range(start, end)
    tt = np.array(months)
    inb = (tt >= base[0]) & (tt <= base[1])
    # SST
    log("  OISST")
    oi = D.oisst_monthly("1981-09", end, cache_dir=cache_dir, manifest=man.get("oisst"), log=log)
    oi = oi.sel(start, end)
    raw = D.nino_indices(oi)
    indices, clim = {}, {}
    for k, v in raw.items():
        indices[k], clim[k] = D.anomalies(v, months, base)
    maps = _regrid_maps(oi, grid)
    maps_anom, clim["sst_maps"] = D.anomalies(maps, months, base)
    maps_anom = np.where(np.isfinite(maps_anom), maps_anom, 0.0)        # land cells: zero anomaly
    # thermocline
    log("  WOD D20")
    years = range(int(start[:4]) - 1, int(end[:4]) + 1)
    casts = D.wod_d20_casts(years, kinds, cache_dir=cache_dir, manifest=man.get("wod"), log=log)
    edges = tuple(range(130, 281, 10))
    lon_c = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
    out = {}
    for name, la in (("band", (-5.0, 5.0)), ("eq", (-2.0, 2.0))):
        B = D.d20_monthly(casts, months, lat=la, lon_edges=edges)
        A, _ = D.anomalies(B, months, base)
        out[name] = (B, A)
    observed = np.isfinite(out["band"][0])
    d20_band = _locf(eof_fill(out["band"][1], inb, n_eofs))
    d20_eq = _locf(eof_fill(out["eq"][1], inb, n_eofs))
    west = lon_c < 205
    indices["wwv"] = np.nanmean(d20_band, axis=1)
    indices["h_west"] = np.nanmean(d20_band[:, west], axis=1)
    indices["h_east"] = np.nanmean(d20_band[:, ~west], axis=1)
    # atmosphere
    log("  ERA5")
    era = D.era5_monthly(cache_dir=cache_dir, manifest=man.get("era5"), log=log)
    e_months = [m for m in months if m in era.time]
    taux = np.full((len(months), grid.ny, grid.nx + 1), np.nan)
    soi = np.full(len(months), np.nan)
    nino34_era5 = np.full(len(months), np.nan)
    if e_months:
        era_s = era.sel(e_months[0], e_months[-1])
        tx = _interp_era5(era_s, "taux", grid.lat, grid.lon_u)
        ebase = (max(base[0], e_months[0]), min(base[1], e_months[-1]))
        tx_a, _ = D.anomalies(tx, e_months, ebase)
        j0 = months.index(e_months[0])
        taux[j0:j0 + len(e_months)] = tx_a
        soi[j0:j0 + len(e_months)] = D.soi_from_mslp(era_s, base=ebase)
        e34 = D.box_mean(era_s, "sea_surface_temperature", D.NINO_BOXES["nino34"])
        nino34_era5[j0:j0 + len(e_months)] = D.anomalies(e34, e_months, ebase)[0]
    indices["soi_era5"] = soi
    indices["nino34_era5"] = nino34_era5
    # cross-checks
    checks: dict[str, Any] = {}
    ok = np.isfinite(nino34_era5)
    checks["nino34_oisst_vs_era5"] = {"corr": float(np.corrcoef(indices["nino34"][ok], nino34_era5[ok])[0, 1]),
                                      "rmse": float(np.sqrt(np.mean((indices["nino34"][ok] - nino34_era5[ok]) ** 2))),
                                      "months": int(ok.sum())}
    ok = np.isfinite(soi)
    checks["nino34_vs_soi_era5_corr"] = float(np.corrcoef(indices["nino34"][ok], soi[ok])[0, 1])
    try:
        sm = D.statsmodels_nino12()
        common = [m for m in months if m in sm["time"]]
        a = np.array([sm["sst"][sm["time"].index(m)] for m in common])
        b = raw["nino12"][[months.index(m) for m in common]]
        checks["nino12_oisst_vs_statsmodels"] = {"corr": float(np.corrcoef(a, b)[0, 1]),
                                                 "mean_diff_K": float(np.mean(b - a)), "months": len(common),
                                                 "first": common[0], "last": common[-1]}
    except Exception as e:  # noqa: BLE001 -- statsmodels missing: the check is reported as skipped
        checks["nino12_oisst_vs_statsmodels"] = {"skipped": repr(e)}
    try:
        hc = D.ncei_heat_content(cache_dir=cache_dir, manifest=man.get("ncei_ohc"))
        # 3-month mean of the monthly WWV proxy centred on each season's middle month
        pairs = [(hc["hc"][i], np.nanmean(indices["wwv"][months.index(m) - 1: months.index(m) + 2]))
                 for i, m in enumerate(hc["mid_month"]) if m in months and 0 < months.index(m) < len(months) - 1]
        a, b = np.array(pairs).T
        checks["wwv_proxy_vs_ncei_ohc_0_700"] = {"corr": float(np.corrcoef(a, b)[0, 1]), "seasons": len(pairs)}
    except Exception as e:  # noqa: BLE001
        checks["wwv_proxy_vs_ncei_ohc_0_700"] = {"skipped": repr(e)}
    cov = observed.mean(1)
    checks["d20_coverage"] = {"mean_fraction_of_bins": float(cov.mean()),
                              "months_carried_forward": int((observed.sum(1) < n_eofs).sum()),
                              "months_below_half": int((cov < 0.5).sum()),
                              "by_year": {y: float(cov[[m.startswith(str(y)) for m in months]].mean())
                                          for y in range(int(start[:4]), int(end[:4]) + 1)},
                              "casts_by_kind": {k: int((casts["kind"] == k).sum()) for k in kinds}}
    checks["nino34_trend_K_per_decade"] = float(np.polyfit(np.arange(len(months)) / 120.0,
                                                           indices["nino34"], 1)[0])
    sources = {k: D.SOURCES[k] for k in ("oisst", "wod", "era5", "ncei_ohc", "statsmodels_elnino")}
    return EnsoDataset(months, indices, maps_anom.astype(np.float32), grid, d20_band, d20_eq, lon_c, observed,
                       taux.astype(np.float32), base, clim, checks, sources)


def to_series(ds: EnsoDataset, columns=COLUMNS, name: str = "enso"):
    """:class:`~pinneapple_systems.time_series.forecast_lab.SeriesData` with ``columns`` (target = first) and the
    maps in ``meta`` (``month_of_year``, ``sst_maps``, ``d20_band``, ``d20_eq``, ``d20_lon``, ``taux``, ``grid``,
    ``months``)."""
    from ..forecast_lab import SeriesData
    obs = np.column_stack([ds.indices[c] for c in columns])
    if not np.isfinite(obs).all():
        bad = [c for c in columns if not np.isfinite(ds.indices[c]).all()]
        raise ValueError(f"columns with gaps: {bad}")
    meta = {"months": ds.months, "month_of_year": ds.month_of_year, "sst_maps": ds.sst_maps,
            "d20_band": ds.d20_band, "d20_eq": ds.d20_eq, "d20_lon": ds.d20_lon, "d20_observed": ds.d20_observed,
            "taux": ds.taux,
            "grid": ds.grid, "columns": list(columns)}
    return SeriesData(obs, 0, 1.0, name, meta)
