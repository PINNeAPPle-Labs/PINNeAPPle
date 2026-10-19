"""Openly licensed real series, downloaded from a pinned commit and checked against a SHA-256.

* :func:`load_usd_eur_daily` -- daily U.S. dollars per euro, 1999 onwards, from the Federal Reserve H.10 release
  via FRED (series DEXUSEU), as packaged by the `datasets/exchange-rates` data package (licence ODC-PDDL 1.0,
  https://github.com/datasets/exchange-rates). Pinned to commit ``c023dd8b``; the file holds all currencies.
* :func:`load_youtube_trending_us` -- the US file of the Kaggle dataset "Trending YouTube Video Statistics" (user
  ``datasnaek``; licence CC0 1.0 Public Domain as listed on the Kaggle page), taken from a GitHub copy pinned to a
  commit: ``USvideos.csv`` from ``ZhaoyiW/YouTube-Trending-Video-Analysis`` (byte-identical, same SHA-256, to the
  copy in ``waittim/Youtube-trending-videos-analysis``). Aggregated to one row per trending day.
* :func:`load_sunspots_yearly` -- yearly mean sunspot numbers 1700-2008 from the NOAA National Geophysical Data
  Center, in the public domain, as shipped with statsmodels (``statsmodels.datasets.sunspots``); read from the
  installed statsmodels when present, else from statsmodels v0.14.4 on GitHub.

Downloads go to ``cache_dir`` (default ``$PINNEAPPLE_DATA_CACHE`` or ``~/.cache/pinneapple/datasets``) and are
re-used when the checksum matches; a mismatch raises instead of returning different data.
"""
from __future__ import annotations

import csv
import hashlib
import http.client
import io
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["PublicSeries", "PINNED", "fetch_pinned", "load_usd_eur_daily", "load_sunspots_yearly", "log_returns",
           "YouTubeDaily", "youtube_daily_aggregates", "load_youtube_trending_us"]

PINNED: dict[str, dict[str, str]] = {
    "exchange_rates_daily": {
        "url": "https://raw.githubusercontent.com/datasets/exchange-rates/"
               "c023dd8b8f879610641a7391c0394c3bbb42d0f7/data/daily.csv",
        "sha256": "68349ef41ecc0cf35fa00ba0d9d63ec93b45a5182c37d2bb9a368fcd654bfeaa",
        "license": "ODC-PDDL-1.0 (data: Federal Reserve Board H.10 via FRED, public domain)",
        "source": "https://fred.stlouisfed.org/series/DEXUSEU",
    },
    "youtube_trending_us": {
        "url": "https://raw.githubusercontent.com/ZhaoyiW/YouTube-Trending-Video-Analysis/"
               "5dc260b68ed054ba460b53817faa0158e9367c5f/USvideos.csv",
        "sha256": "09b4eb71295752705e472ebefeac9d2afab4177b7a818af795dea62744a48eb2",
        "license": "CC0 1.0 Public Domain (Kaggle dataset 'Trending YouTube Video Statistics', user datasnaek)",
        "source": "https://www.kaggle.com/datasets/datasnaek/youtube-new",
    },
    "sunspots_yearly": {
        "url": "https://raw.githubusercontent.com/statsmodels/statsmodels/v0.14.4/statsmodels/datasets/"
               "sunspots/sunspots.csv",
        "sha256": "f67889b1d9002cd5227f0e0ef54e35b419cdd85a31279adef6f73fb41e5c0a9b",
        "license": "public domain (NOAA NGDC, as distributed with statsmodels)",
        "source": "http://www.ngdc.noaa.gov/stp/solar/solarda3.html",
    },
}


@dataclass
class PublicSeries:
    name: str
    time: list[str]
    values: np.ndarray
    source: dict[str, str] = field(default_factory=dict)


def _cache_dir(cache_dir: str | None) -> Path:
    d = Path(cache_dir or os.environ.get("PINNEAPPLE_DATA_CACHE", Path.home() / ".cache" / "pinneapple" / "datasets"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_pinned(key: str, cache_dir: str | None = None, *, local: str | None = None) -> bytes:
    """Bytes of the pinned file ``key`` (from ``local``, the cache, or the network), checksum verified."""
    spec = PINNED[key]
    if local is not None and Path(local).exists():
        data = Path(local).read_bytes()
        if _sha256(data) == spec["sha256"]:
            return data
    path = _cache_dir(cache_dir) / f"{key}.csv"
    if path.exists():
        data = path.read_bytes()
        if _sha256(data) == spec["sha256"]:
            return data
    data, last = None, None
    for attempt in range(4):                                          # large files: retry a cut-off transfer
        try:
            with urllib.request.urlopen(spec["url"], timeout=300) as r:  # noqa: S310 -- pinned https URL
                data = r.read()
            break
        except (OSError, http.client.IncompleteRead) as e:
            last = e
            time.sleep(2 ** (attempt + 1))
    if data is None:
        raise OSError(f"{key}: download failed: {last!r}")
    got = _sha256(data)
    if got != spec["sha256"]:
        raise ValueError(f"{key}: checksum mismatch (expected {spec['sha256']}, got {got})")
    path.write_bytes(data)
    return data


def load_usd_eur_daily(cache_dir: str | None = None) -> PublicSeries:
    """U.S. dollars per euro on business days (days without a quote are dropped)."""
    data = fetch_pinned("exchange_rates_daily", cache_dir)
    t, v = [], []
    for row in csv.reader(io.StringIO(data.decode("utf-8"))):
        if len(row) == 3 and row[1] == "Euro" and row[2] not in ("", "."):
            t.append(row[0])
            v.append(float(row[2]))
    return PublicSeries("usd_per_eur_daily", t, np.asarray(v), dict(PINNED["exchange_rates_daily"]))


def load_sunspots_yearly(cache_dir: str | None = None) -> PublicSeries:
    local = None
    try:
        import statsmodels.datasets.sunspots as _s
        local = str(Path(_s.__file__).with_name("sunspots.csv"))
    except ImportError:
        pass
    data = fetch_pinned("sunspots_yearly", cache_dir, local=local)
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))[1:]
    return PublicSeries("sunspots_yearly", [str(int(float(r[0]))) for r in rows],
                        np.asarray([float(r[1]) for r in rows]), dict(PINNED["sunspots_yearly"]))


def log_returns(values) -> np.ndarray:
    v = np.asarray(values, float)
    return np.diff(np.log(v))


@dataclass
class YouTubeDaily:
    """One row per trending day: the date, the number of trending entries, their total views and likes, and the
    number of entries per YouTube category id."""
    dates: list[str]
    n_entries: np.ndarray
    total_views: np.ndarray
    total_likes: np.ndarray
    category_counts: dict[int, np.ndarray]
    source: dict[str, str] = field(default_factory=dict)

    def missing_days(self) -> list[str]:
        """Calendar days between the first and last trending day with no data."""
        import datetime as _dt
        d0, d1 = (_dt.date.fromisoformat(d) for d in (self.dates[0], self.dates[-1]))
        have = set(self.dates)
        return [str(d0 + _dt.timedelta(k)) for k in range((d1 - d0).days + 1)
                if str(d0 + _dt.timedelta(k)) not in have]


def youtube_daily_aggregates(csv_text: str) -> YouTubeDaily:
    """Aggregate the rows of a trending-videos CSV (columns ``trending_date`` as yy.dd.mm, ``category_id``,
    ``views``, ``likes``) to one row per trending day, in date order."""
    import datetime as _dt
    agg: dict[str, list] = {}
    cats: dict[int, dict[str, int]] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        yy, dd, mm = row["trending_date"].split(".")
        day = str(_dt.date(2000 + int(yy), int(mm), int(dd)))
        a = agg.setdefault(day, [0, 0, 0])
        a[0] += 1
        a[1] += int(row["views"])
        a[2] += int(row["likes"])
        c = int(row["category_id"])
        cats.setdefault(c, {})
        cats[c][day] = cats[c].get(day, 0) + 1
    days = sorted(agg)
    return YouTubeDaily(days, np.array([agg[d][0] for d in days]), np.array([agg[d][1] for d in days], float),
                        np.array([agg[d][2] for d in days], float),
                        {c: np.array([v.get(d, 0) for d in days]) for c, v in sorted(cats.items())})


def load_youtube_trending_us(cache_dir: str | None = None) -> YouTubeDaily:
    data = fetch_pinned("youtube_trending_us", cache_dir)
    out = youtube_daily_aggregates(data.decode("utf-8"))
    out.source = dict(PINNED["youtube_trending_us"])
    return out
