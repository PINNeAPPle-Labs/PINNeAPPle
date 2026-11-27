"""Hurricane Ian (2022) end to end on real data: the official best track, the vortex inside the reanalysis, and the
same storm inside two published global forecasts — with the number each stage earns.

    python examples/weather_forecasting/hurricane_ian.py [--cache DIR] [--out DIR] [--offline] [--coarse]

Writes to ``out`` (default ``examples/weather_forecasting/_out/hurricane_ian``, not committed):

* ``summary.json`` — every number this script measured, with the data behind it (file, SHA-256, licence, date);
* ``figures/ian_globe.png`` — the storm on the Earth-2 globe with the official track and the tracked ones;
* ``figures/ian_globe.gif`` — the reanalysis crossing the globe, 6 hours a frame (the short cut of the video);
* ``figures/ian_track_<model>.png`` — track and intensity error by lead, with the horizons marked;
* ``figures/ian_track_vs_observed.png`` — the official track against the ones the models actually followed;
* ``figures/ian_field_two_grids.png`` — the same instant on the 0.25-degree analysis and on the 5.625-degree archive,
  one cell each, with the official position on both;
* ``figures/ian_confidence_by_lead.png`` — the confidence bar along the lead: green inside both thresholds, amber past
  one of them, red where the grid lost the storm;
* ``figures/ian_stages.mp4`` — the chain as one video, 60 to 90 seconds: the analysis, the forecasts lead by lead and
  the coarse grid, each frame captioned with the instant and the stage, the honest note always on screen.

Data, all public and cited in ``summary.json``:

* **HURDAT2** — NOAA/NHC best track, US Government work, public domain, downloaded from the NHC.
* **ERA5** — Copernicus Climate Change Service reanalysis (Hersbach et al., 2020), read from the ARCO-ERA5 bucket of
  Google Research (``mean_sea_level_pressure``, hourly, 0.25 degrees).
* **IFS HRES and Pangu-Weather** — the 0.25-degree and 5.625-degree forecast archives of WeatherBench2 (Rasp et al.,
  2024), initialised 00/12 UTC. Read over HTTPS and cached outside the repository.

Three stages, three numbers:

1. **Now** (analysis): does the reanalysis hold the storm, and where? Track error in km, central pressure in hPa.
2. **Next days** (forecast): how far the published models drift and how much of the storm's depth they keep, by
   lead, and the lead at which each stops being usable.
3. **The grid** (why the chain hands over, ``--coarse``): the same forecasts on a 5.625-degree grid, where a cell is
   wider than the core, so that "the intensity is wrong" becomes "the field does not contain the storm at all".
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from pinneapple_physics.weather.cyclones import (
    AH_REFERENCE,
    CORE_NOTE,
    HURDAT2_CITATION,
    HURDAT2_URL,
    confidence_by_lead,
    field_vs_grid,
    load_best_track,
    skill_by_lead,
    stages_video,
    storm_horizons,
    track_globe,
    track_skill_figure,
    track_storm,
    track_vs_observed,
    verify_storm,
)

STORM = "AL092022"                                                # Hurricane Ian, Atlantic 2022
BOX = (5.0, 45.0, 275.0, 325.0)                                   # lat_s, lat_n, lon_w, lon_e (lon in [0, 360))
ANALYSIS_TIMES = [f"2022-09-{d:02d}T{h:02d}" for d in range(23, 30) for h in (0, 6, 12, 18)]
INITS = [f"2022-09-{d}T{h}" for d in (24, 25, 26, 27, 28) for h in ("00", "12")]
LEADS = list(range(6, 121, 6))
RADIUS_KM, RADIUS_COARSE_KM, DEEPEST_HPA = 400.0, 600.0, 1008.0

_CACHE_FORMAT = 1                     # bump when the stored arrays change shape or units: a stale npz is a wrong run

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
FORECASTS = {"IFS HRES": "https://storage.googleapis.com/weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
             "Pangu-Weather":
                 "https://storage.googleapis.com/weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"}
COARSE = {"IFS HRES": "https://storage.googleapis.com/weatherbench2/datasets/"
                      "hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
          "Pangu-Weather": "https://storage.googleapis.com/weatherbench2/datasets/"
                           "pangu/2018-2022_0012_64x32_equiangular_conservative.zarr"}
LICENCES = {
    "HURDAT2": HURDAT2_CITATION,
    "ERA5": ("ERA5 reanalysis (Hersbach et al., 2020), Copernicus Climate Change Service (C3S), read from the "
             "ARCO-ERA5 bucket of Google Research; contains modified Copernicus information, neither the European "
             "Commission nor ECMWF is responsible for any use of it"),
    "WeatherBench2": ("IFS HRES and Pangu-Weather forecasts as archived by WeatherBench2 (Rasp et al., 2024), "
                      "https://weatherbench2.readthedocs.io — read, not redistributed"),
}


# ------------------------------------------------------------------------------------------------ data access

def open_zarr(url: str):
    """A public Zarr store over HTTPS by its consolidated metadata: no account, no gcsfs."""
    import xarray as xr

    return xr.open_zarr(url, consolidated=True, chunks=None)


def box_index(lat, lon, box=BOX):
    """The cell indices of ``box`` on a store's own coordinates: what to ask for before reading the field.

    Selecting here rather than after ``.values`` is what keeps the download small: the storm's box is 161 x 201 cells
    of a 721 x 1440 globe, about a thirtieth of the data for the same numbers. The raw fields are never stored --
    :func:`_cached` keeps only this crop, in the cache directory outside the repository.
    """
    iy = np.nonzero((lat >= box[0]) & (lat <= box[1]))[0]
    lo = np.mod(lon, 360.0)
    ix = np.nonzero((lo >= box[2]) & (lo <= box[3]))[0]
    return iy, ix


def _cached(cache: Path, key: str, build, offline: bool):
    """``cache/key.npz`` if it is there, else build it once and keep it: a rerun is offline and free."""
    f = cache / f"{key}.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    if offline:
        raise FileNotFoundError(f"{f} is not cached and --offline was given")
    d = build()
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **{k: np.asarray(v) for k, v in d.items()})
    return d


def fields_of(url: str, init: str | None, leads, cache: Path, offline: bool, tag: str, log=print) -> dict:
    """Sea-level pressure (hPa) on the storm's box: the analysis at ``times`` (``init=None``) or one forecast."""
    def build():
        ds = open_zarr(url)
        lat0 = ds["latitude"].values.astype(float)
        lon0 = ds["longitude"].values.astype(float)
        ascending = lat0[0] < lat0[-1]
        iy, ix = box_index(lat0, lon0)
        lat, clon = lat0[iy][::-1] if ascending else lat0[iy], np.mod(lon0, 360.0)[ix]
        var = "mean_sea_level_pressure" if "mean_sea_level_pressure" in ds else "msl"

        def read(da):
            """Only the box is fetched, in the descending-latitude, ascending-longitude order of ``lat`` and ``clon``.

            The stores do not agree on their dimension order -- the 0.25 degree archives are (lat, lon) and the
            64 x 32 equiangular ones are (lon, lat) -- so the two axes are named rather than assumed, which is what
            :func:`vortex_centre` then checks against.
            """
            sub = da.isel(latitude=iy, longitude=ix)
            if ascending:
                sub = sub.isel(latitude=slice(None, None, -1))
            return sub.transpose(..., "latitude", "longitude").values
        if init is None:
            times = np.array([np.datetime64(t, "h") for t in leads], dtype="datetime64[h]")
            a = np.stack([read(ds[var].sel(time=t)) for t in times])
            log(f"{tag}: {len(times)} analysis times")
        else:
            avail = np.asarray(ds["prediction_timedelta"].values).astype(int)
            pick = [(k, int(np.argmin(np.abs(avail - lead)))) for k, lead in enumerate(leads)
                    if lead in set(avail.tolist())]
            if not pick:
                raise ValueError(f"{tag} {init}: none of the leads {leads} is published")
            times = np.datetime64(init, "h") + np.array([leads[k] for k, _ in pick], "timedelta64[h]")
            a = read(ds[var].sel(time=np.datetime64(init, "h")).isel(prediction_timedelta=[j for _, j in pick]))
            log(f"{tag} {init}: {len(pick)} leads")
        return {"fields": a / 100.0, "times": times, "lat": lat, "lon": clon, "origin": url}
    sha = hashlib.sha256(f"{_CACHE_FORMAT}|{url}|{init}|{leads}".encode()).hexdigest()[:10]
    return _cached(cache, f"{tag.replace(' ', '_').lower()}_{(init or 'analysis').replace(':', '')}_{sha}", build,
                   offline)


def track_and_verify(d: dict, best, name: str, radius_km: float = RADIUS_KM,
                     deepest_hpa: float | None = DEEPEST_HPA, init=None):
    """Follow the vortex from the official position at the first valid time, and score it against the best track.

    ``init`` is the forecast's initialization, which is what a lead time is counted from: the first valid time of a
    6-hourly archive is T+6, so omitting it here would shift every lead by six hours.
    """
    first = np.datetime64(d["times"][0], "h")
    la0, lo0, _, _ = best.interpolate(np.array([first]))
    if not np.isfinite(la0[0]):
        return None, None
    track = track_storm(d["fields"], d["lat"], d["lon"], d["times"], start=(float(la0[0]), float(lo0[0]) % 360.0),
                        radius_km=radius_km, deepest_hpa=deepest_hpa)
    return track, verify_storm(track, best, name=name, init=init)


def summarise(err: dict, label: str) -> dict:
    """One verification as the JSON carries it: every lead, with the not-found ones as -1 and a flag."""
    ok = np.asarray(err.found, bool)
    med = (lambda v: float(np.nanmedian(v)) if ok.any() else None)                       # noqa: E731
    return {"source": label, "init": np.datetime_as_string(err.init, unit="h"),
            "times": [np.datetime_as_string(t, unit="h") for t in err.times],
            "leads_h": np.asarray(err.leads_h, float).tolist(),
            "track_km": np.nan_to_num(err.track_km, nan=-1).tolist(),
            "pressure_hpa": np.nan_to_num(err.pressure_hpa, nan=-1).tolist(),
            "wind_kt": np.nan_to_num(err.wind_kt, nan=-1).tolist(),
            "found": ok.tolist(), "tracked": int(ok.sum()),
            "median_track_km": med(err.track_km), "median_pressure_hpa": med(err.pressure_hpa),
            "median_wind_kt": med(err.wind_kt),
            "max_track_km": float(np.nanmax(err.track_km)) if ok.any() else None}


def _plain(d: dict) -> dict:
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# --------------------------------------------------------------------------------------------------- the study

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "pinneapple" / "hurricane"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "_out" / "hurricane_ian"))
    ap.add_argument("--figures", default=None,
                    help="where the figures and the video go (default: <out>/figures); point it at the example folder "
                         "to publish the media the issue asks for")
    ap.add_argument("--offline", action="store_true", help="use only what is already cached")
    ap.add_argument("--coarse", action="store_true", help="also run the same forecasts on the 5.625 degree grid")
    ap.add_argument("--no-video", action="store_true", help="figures only: skip the 60-to-90-second video")
    ap.add_argument("--fps", type=float, default=12.0, help="frames per second of the video")
    ap.add_argument("--hold", type=int, default=7, help="frames each step of the data is held for")
    args = ap.parse_args(argv)
    cache, out = Path(args.cache), Path(args.out)
    figdir = Path(args.figures) if args.figures else out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    best = load_best_track(STORM, HURDAT2_URL, cache=cache)
    print(f"{best.name} ({best.storm_id}): {len(best)} records from {best.times[0]} to {best.times[-1]}, "
          f"peak {np.nanmax(best.wind_kt):.0f} kt at {np.nanmin(best.pressure_hpa):.0f} hPa", flush=True)

    # 1. Now: what the analysis holds
    an = fields_of(ARCO, None, ANALYSIS_TIMES, cache, args.offline, tag="era5 0.25")
    an_track, an_err = track_and_verify(an, best, "ERA5 0.25 deg (analysis)", deepest_hpa=None)
    print(f"analysis: {an_err.tracked}/{len(an_err)} times held the storm; median track error "
          f"{np.nanmedian(an_err.track_km):.0f} km (max {np.nanmax(an_err.track_km):.0f}), "
          f"central pressure {np.nanmedian(an_err.pressure_hpa):+.0f} hPa, "
          f"equivalent wind {np.nanmedian(an_err.wind_kt):+.0f} kt", flush=True)

    # 2. Next days: the published forecasts
    per_model, skills, horizons, tracks = {}, {}, {}, {}
    for name, url in FORECASTS.items():
        errs, paths = [], []
        for init in INITS:
            d = fields_of(url, init, LEADS, cache, args.offline, tag=name)
            tr, e = track_and_verify(d, best, name, init=np.datetime64(init, "h"))
            if e is not None and e.tracked:
                errs.append(e)
                paths.append(tr)
        if not errs:
            print(f"{name}: nothing to score (no lead held the storm)", flush=True)
            continue
        sk, h = skill_by_lead(errs), None
        h = storm_horizons(sk)
        skills[name], horizons[name] = sk, h
        per_model[name] = [summarise(e, f"{name} from {np.datetime_as_string(e.init, unit='h')}") for e in errs]
        tracks[name] = paths[len(paths) // 2]

        def at(lead, sk=sk):                                                # bound per iteration, not late-bound
            return float(sk["track_km"][int(np.argmin(np.abs(sk["leads_h"] - lead)))])
        print(f"{name}: {len(errs)} forecasts; median track error {at(6):.0f} km at +6 h, {at(24):.0f} km at +24 h, "
              f"{at(48):.0f} km at +48 h, {at(120):.0f} km at +120 h; median intensity error "
              f"{np.nanmedian(sk['wind_kt']):+.0f} kt ({np.nanmedian(sk['pressure_hpa']):+.0f} hPa)", flush=True)
        print(f"   horizons: 500 km at {h['track_hours']:.0f} h, 20 kt at {h['intensity_hours']:.0f} h, "
              f"storm lost by half the forecasts at {h['lost_hours']:.0f} h", flush=True)
        track_skill_figure(figdir / f"ian_track_{name.split()[0].lower()}.png", sk, h,
                           title=f"{name}, Hurricane Ian 2022, {len(errs)} initialisations")

    if skills:
        confidence_by_lead(figdir / "ian_confidence_by_lead.png", skills, horizons,
                           title="Hurricane Ian 2022: the confidence bar along the lead",
                           note="the same record as the two panels of each model, read as the one thing an operator "
                                "wants to know: at which lead it is still inside both thresholds and the storm is "
                                "still there to compare.")

    # 3. The grid: the same forecasts where a cell is wider than the core
    coarse, coarse_samples = {}, {}
    if args.coarse:
        for name, url in COARSE.items():
            errs, samples = [], {}
            for init in INITS:
                d = fields_of(url, init, LEADS, cache, args.offline, tag=f"{name} 64x32")
                tr, e = track_and_verify(d, best, name, radius_km=RADIUS_COARSE_KM, init=np.datetime64(init, "h"))
                if e is not None:
                    errs.append(e)
                    samples[init] = (d, tr)
            if not errs:
                continue
            sk = skill_by_lead(errs)
            coarse[name] = {"forecasts": len(errs), "skill": _plain(sk),
                            "horizons": storm_horizons(sk), "found_fraction_overall": float(np.mean(sk["found_fraction"])),
                            "median_pressure_hpa": float(np.nanmedian(sk["pressure_hpa"]))}
            # one initialisation kept whole -- its fields, its own tracked positions, its valid times -- because the
            # video shows the coarse grid lead by lead and a median curve cannot be drawn on a map
            keep = INITS[len(INITS) // 2] if INITS[len(INITS) // 2] in samples else next(iter(samples))
            coarse_samples[name] = {"fields": samples[keep][0]["fields"], "lat": samples[keep][0]["lat"],
                                    "lon": samples[keep][0]["lon"], "times": samples[keep][0]["times"],
                                    "leads": np.asarray(sk["leads_h"], float), "skill": sk,
                                    "thresholds": coarse[name]["horizons"]["thresholds"],
                                    "horizons": coarse[name]["horizons"], "tracks": {name: samples[keep][1]},
                                    "found_fraction_overall": coarse[name]["found_fraction_overall"],
                                    "median_pressure_hpa": coarse[name]["median_pressure_hpa"], "init": keep}
            print(f"{name} on 64 x 32: the storm is present at {100 * np.mean(sk['found_fraction']):.0f}% of the "
                  f"leads, median central pressure {np.nanmedian(sk['pressure_hpa']):+.0f} hPa", flush=True)

    # the scene: the analysis at the peak hour, with the tracks that were followed
    peak = int(np.nanargmin(best.pressure_hpa))
    j = int(np.argmin(np.abs(an["times"] - best.times[peak])))
    note = ("ERA5 analysis, 0.25 deg, over the storm's box only -- outside it the globe carries no data, which is "
            "not a calm. Tracks: the deepest pressure minimum within 400 km of the previous centre. Intensity is "
            "compared in hPa and, through Atkinson-Holliday, in knots.")
    followed = {"ERA5 0.25 deg (analysis)": an_track} | tracks
    # the display scale is the data's, not the planet's: a fixed 985-1015 washes a basin of 1015-1020 into one tone
    vmin, vmax = (float(np.percentile(an["fields"][np.isfinite(an["fields"])], 0.5)),
                  float(np.percentile(an["fields"][np.isfinite(an["fields"])], 97.0)))
    track_globe(figdir / "ian_globe.png", best, followed, pressure_hpa=an["fields"][j], lat=an["lat"],
                lon=an["lon"], times=an["times"][j:j + 1], centre=(22.0, 282.0), vmin=vmin, vmax=vmax, note=note,
                title=f"Hurricane Ian, {np.datetime_as_string(an['times'][j], unit='h').replace('T', ' ')} UTC")
    track_globe(figdir / "ian_globe.gif", best, followed, pressure_hpa=an["fields"], lat=an["lat"],
                lon=an["lon"], times=an["times"], centre=(24.0, 283.0), size=340, dpi=92, vmin=vmin, vmax=vmax,
                note=note, title=f"Hurricane Ian, {str(best.times[0].astype('datetime64[D]'))} to "
                                 f"{str(best.times[-1].astype('datetime64[D]'))}, 6 hours a frame")

    # the track each model followed, against the one that was flown: the figure a reader checks the mean error against
    captions = {"ERA5 0.25 deg (analysis)": f"median {np.nanmedian(an_err.track_km):.0f} km off the official centre"}
    for name, sk in skills.items():
        captions[name] = (f"median {float(sk['track_km'][int(np.argmin(np.abs(sk['leads_h'] - 48)))]):.0f} km at "
                          f"+48 h")
    track_vs_observed(figdir / "ian_track_vs_observed.png", best, followed, captions=captions, extent=BOX,
                      title="Hurricane Ian 2022: the official track and the ones the models followed",
                      note="one line per model is a real tracked series, not an average: the middle initialisation "
                           "of each, followed by the pressure minimum, against the official HURDAT2 positions.")

    # the cell, next to the storm it cannot hold: the same instant on both grids, with the official position marked
    coarse_name = next(iter(coarse_samples), None)
    if coarse_name is not None:
        sample = coarse_samples[coarse_name]
        want = an["times"][j]
        k = int(np.argmin(np.abs(np.asarray(sample["times"], "datetime64[h]") - want)))
        la_j, lo_j, _, _ = best.interpolate(np.array([sample["times"][k]]))
        both = np.concatenate([np.ravel(an["fields"][j]), np.ravel(sample["fields"][k])])
        field_vs_grid(figdir / "ian_field_two_grids.png",
                      [{"label": "ERA5 reanalysis, 0.25 degree", "field": an["fields"][j], "lat": an["lat"],
                        "lon": an["lon"], "cell_km": 27.8, "position": (la_j[0], lo_j[0]) if np.isfinite(la_j[0])
                        else None},
                       {"label": f"{coarse_name}, 5.625 degree", "field": sample["fields"][k],
                        "lat": sample["lat"], "lon": sample["lon"], "cell_km": 625.0,
                        "position": (la_j[0], lo_j[0]) if np.isfinite(la_j[0]) else None}],
                      best=best, extent=(18.0, 34.0, 276.0, 294.0),
                      vmin=float(np.nanmin(both)) - 1.0, vmax=float(np.nanpercentile(both, 90.0)),
                      title=f"Ian, {np.datetime_as_string(np.datetime64(sample['times'][k], 'h'), unit='h')}"
                            .replace("T", " ") + " UTC: the same storm at two cell sizes",
                      note="the analysis resolves a centre to a fraction of its 28 km cell; the coarse archive is "
                           "asked the same question with a 625 km cell, and answers with a field that has no core in "
                           "it -- which is why the chain hands over to finer models instead of correcting this one. "
                           "The two panels share one colour scale, cut at the 90th percentile of both fields.")

    media = ["ian_globe.png", "ian_globe.gif"] + [f"ian_track_{n.split()[0].lower()}.png" for n in skills] + \
        ["ian_track_vs_observed.png", "ian_confidence_by_lead.png"] + \
        (["ian_field_two_grids.png"] if coarse_name is not None else [])
    if not args.no_video:
        analysis_stage = {"fields": an["fields"], "times": an["times"], "lat": an["lat"], "lon": an["lon"],
                          "track": an_track, "errors": an_err}
        stages_video(figdir / "ian_stages.mp4", best=best, analysis=analysis_stage, tracks=tracks, skills=skills,
                     horizons=horizons, coarse=coarse_samples if args.coarse else None,
                     title="Hurricane Ian 2022, end to end on public data — HURDAT2, ERA5, IFS HRES, Pangu-Weather",
                     note=CORE_NOTE, fps=args.fps, hold=args.hold, n_inits=len(INITS))
        media.append("ian_stages.mp4")


    # every lead time in this file has an origin: the store it was read from, what it carried and the licence it came
    # under, recorded next to the number so the two travel together
    sources = {"ERA5 0.25 deg (analysis)": {"url": ARCO, "variable": "mean_sea_level_pressure", "licence": "ERA5",
                                            "times": [np.datetime_as_string(t, unit="h") for t in an["times"]]}}
    for name, url in FORECASTS.items():
        sources[name] = {"url": url, "variable": "mean_sea_level_pressure", "licence": "WeatherBench2",
                         "inits": INITS, "leads_h": LEADS}
    if args.coarse:
        for name, url in COARSE.items():
            sources[f"{name} (64 x 32)"] = {"url": url, "variable": "mean_sea_level_pressure",
                                            "licence": "WeatherBench2", "inits": INITS,
                                            "leads_h": "the same leads, at the ones this archive publishes"}

    summary = {
        "storm": {"id": best.storm_id, "name": best.name, "basin": best.basin, "records": len(best),
                  "first": str(best.times[0]), "last": str(best.times[-1]),
                  "peak_wind_kt": float(np.nanmax(best.wind_kt)),
                  "peak_pressure_hpa": float(np.nanmin(best.pressure_hpa)),
                  "landfalls": [str(t) for t, r in zip(best.times, best.record, strict=True) if r == "L"]},
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_track_source": best.source,
        "licences": LICENCES,
        "sources": sources,
        "wind_relation": AH_REFERENCE,
        "protocol": {"search_radius_km": RADIUS_KM, "coarse_radius_km": RADIUS_COARSE_KM,
                     "deepest_hpa": DEEPEST_HPA, "leads_h": LEADS, "inits": INITS, "box": BOX,
                     "analysis_times": ANALYSIS_TIMES,
                     "note": "a minimum shallower than deepest_hpa is not counted as the storm; the analysis is "
                             "tracked without that threshold, since a reanalysis always holds a circulation"},
        "analysis": summarise(an_err, "ERA5 0.25 deg (analysis)"),
        "forecasts": per_model,
        "skill": {k: _plain(v) for k, v in skills.items()},
        "horizons": {k: {kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()}
                     for k, v in horizons.items()},
        "coarse_grid": coarse,
        "files": [f"figures/{name}" for name in media],
        "figures_dir": str(figdir),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out / 'summary.json'} + {len(list(figdir.glob('*')))} files in {figdir}", flush=True)
    return summary


if __name__ == "__main__":
    main()
