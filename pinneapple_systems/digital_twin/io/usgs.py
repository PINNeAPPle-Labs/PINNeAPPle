"""Client of the USGS Water Services "Instantaneous Values" API (public, no key): river discharge, gage height and
other parameters of US stream gauges every 5-15 minutes.

    https://waterservices.usgs.gov/nwis/iv/?format=json&sites=<SITE>&parameterCd=<CODE>&period=<ISO 8601 duration>

Parameter codes: ``00060`` discharge (ft^3/s), ``00065`` gage height (ft). USGS data are U.S. Government works in the
public domain (https://www.usgs.gov/information-policy-and-instructions/copyrights-and-credits); they are downloaded
at run time, never stored in this repository. Provisional data can be revised by USGS.

Failures (network, HTTP status, empty series) raise :class:`USGSDataError`; nothing is ever filled in. A small
in-process cache (``cache_ttl_s``, default 90 s) avoids re-querying on rapid reloads.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

__all__ = ["IV_ENDPOINT", "DISCHARGE", "GAGE_HEIGHT", "USGSDataError", "Series", "parse_iv", "fetch_iv",
           "monitoring_url"]

IV_ENDPOINT = "https://waterservices.usgs.gov/nwis/iv/"
DISCHARGE = "00060"
GAGE_HEIGHT = "00065"
_cache: dict[tuple[str, str, str], tuple[float, Series]] = {}


class USGSDataError(RuntimeError):
    """The USGS service could not be reached or returned no usable data."""


@dataclass(frozen=True)
class Series:
    site_id: str
    site_name: str
    parameter_cd: str
    unit: str
    timestamps: list[datetime]     # timezone-aware, ascending
    values: list[float]
    fetched_at: datetime
    from_cache: bool = False


def monitoring_url(site_id: str) -> str:
    """Public USGS page of a gauge, for a reader to check the data."""
    return f"https://waterdata.usgs.gov/monitoring-location/{site_id}/"


def parse_iv(payload: dict[str, Any], site_id: str, parameter_cd: str) -> Series:
    """The first time series of an IV JSON payload, malformed readings skipped, sorted by time."""
    ts = payload.get("value", {}).get("timeSeries", [])
    if not ts:
        raise USGSDataError(f"USGS returned no time series for site {site_id}, parameter {parameter_cd} (gauge "
                            "offline, decommissioned or not reporting this parameter)")
    s = ts[0]
    raw = (s.get("values") or [{}])[0].get("value", [])
    pairs = []
    for point in raw:
        try:
            pairs.append((datetime.fromisoformat(point["dateTime"]), float(point["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not pairs:
        raise USGSDataError(f"USGS returned no usable readings for site {site_id}, parameter {parameter_cd}")
    pairs.sort(key=lambda p: p[0])
    return Series(site_id=site_id, site_name=s.get("sourceInfo", {}).get("siteName", site_id),
                  parameter_cd=parameter_cd, unit=s.get("variable", {}).get("unit", {}).get("unitCode", ""),
                  timestamps=[p[0] for p in pairs], values=[p[1] for p in pairs],
                  fetched_at=datetime.now(timezone.utc))


def fetch_iv(site_id: str, parameter_cd: str = DISCHARGE, period: str = "P5D", timeout_s: float = 10.0,
             cache_ttl_s: float = 90.0) -> Series:
    """Download one parameter of one gauge over the trailing ``period`` (ISO 8601 duration, e.g. ``"P5D"``)."""
    key = (site_id, parameter_cd, period)
    hit = _cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < cache_ttl_s:
        return replace(hit[1], from_cache=True)
    query = urllib.parse.urlencode({"format": "json", "sites": site_id, "parameterCd": parameter_cd,
                                    "period": period})
    req = urllib.request.Request(f"{IV_ENDPOINT}?{query}", headers={"Accept": "application/json",
                                                                    "User-Agent": "pinneapple-usgs/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:   # noqa: S310 (fixed https endpoint)
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise USGSDataError(f"USGS API returned HTTP {exc.code} for site {site_id}, parameter {parameter_cd}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise USGSDataError(f"could not reach the USGS water services API for site {site_id}: {exc}") from exc
    except ValueError as exc:
        raise USGSDataError(f"USGS API returned a non-JSON response for site {site_id}") from exc
    series = parse_iv(payload, site_id, parameter_cd)
    _cache[key] = (time.monotonic(), series)
    return series
