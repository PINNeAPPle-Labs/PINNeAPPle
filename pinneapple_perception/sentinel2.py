"""Sentinel-2 surface-reflectance images of a point from Google Earth Engine, as arrays for
:mod:`pinneapple_perception.change_detection` (bands B4, B3, B2 and the real NIR band B8).

Credentials come from the environment only, never from code or files in this repository:

* ``EE_PROJECT_ID`` -- the Google Cloud project registered for Earth Engine (https://code.earthengine.google.com/register,
  with the Earth Engine API enabled in that project);
* either ``EE_SERVICE_ACCOUNT_EMAIL`` + ``EE_SERVICE_ACCOUNT_KEY_FILE`` (path to the service account's JSON key,
  kept outside the repository and the image) for a server, or a one-time ``earthengine authenticate`` on the host,
  which stores a refresh token under ``~/.config/earthengine/``.

Without them :func:`ee_initialize` raises :class:`EarthEngineUnavailable` with these steps: no imagery is
fabricated. Needs the optional ``earthengine-api`` and ``geemap`` packages.

Moved here from the PINNeAPPle-apps ``satellite_change_monitoring`` product (OrbitWatch), extracted from a notebook;
images are returned oldest first (the notebook fetched newest first and labelled pairs in both directions).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


class EarthEngineUnavailable(RuntimeError):
    """Raised when ee.Initialize() genuinely fails -- no credential, no
    project, or the Earth Engine API not enabled for the given project.
    Carries the real underlying exception's message plus the concrete
    remediation steps from this module's docstring, so a caller sees
    exactly what a real deployment needs rather than a bare stack trace."""


def ee_initialize(project: str | None = None) -> None:
    """Real ee.Initialize() call. Tries a service account credential if
    EE_SERVICE_ACCOUNT_EMAIL + EE_SERVICE_ACCOUNT_KEY_FILE are set (the
    real, non-interactive path for a server deployment); otherwise falls
    through to ee's default persistent-credential lookup (populated by a
    prior real `earthengine authenticate` run). Never silently succeeds
    with fabricated access -- either a real `ee.Initialize()` call
    genuinely succeeds, or this raises `EarthEngineUnavailable` with a
    concrete remediation message."""
    import ee

    project = project or os.environ.get("EE_PROJECT_ID")
    sa_email = os.environ.get("EE_SERVICE_ACCOUNT_EMAIL")
    sa_key_file = os.environ.get("EE_SERVICE_ACCOUNT_KEY_FILE")

    try:
        if sa_email and sa_key_file:
            credentials = ee.ServiceAccountCredentials(sa_email, sa_key_file)
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as e:  # noqa: BLE001 -- a real, expected credential/config gap
        raise EarthEngineUnavailable(
            "Google Earth Engine is not authenticated/configured in this environment "
            f"(real underlying error: {type(e).__name__}: {e}). A real deployment needs a Google Cloud "
            "project registered for Earth Engine access, plus EITHER a one-time `earthengine authenticate` "
            "run on the host, OR EE_SERVICE_ACCOUNT_EMAIL + EE_SERVICE_ACCOUNT_KEY_FILE env vars pointing at a "
            "real service-account credential, and EE_PROJECT_ID set to that project's id. "
            "See pinneapple_perception.sentinel2's module docstring for the full setup."
        ) from e


@dataclass
class Sentinel2Frame:
    date: str
    rgb: object   # np.ndarray (H, W, 3), bands ordered (red, green, blue) = (B4, B3, B2)
    nir: object   # np.ndarray (H, W), band B8


def fetch_sentinel2_sequence(
    lon: float, lat: float, buffer_m: float = 5000.0,
    start_date: str = "2024-01-01", end_date: str = "2025-07-27",
    max_cloud_pct: float = 10.0, n_images: int = 5, scale: int = 10,
    project: str | None = None,
) -> list:
    """Real fetch of the n_images most recent, least-cloudy Sentinel-2 SR
    images intersecting a buffered point, returned oldest-to-newest
    (chronological order -- the original notebook fetched newest-first
    and then had two of its own cells label "before/after" in opposite
    directions for the same pairs; sorting chronologically here avoids
    reproducing that ambiguity). Same collection/filter/band logic as
    the source notebook, with one real fix: this also fetches the real
    NIR band (B8) alongside B4/B3/B2, instead of that notebook's fake
    `nir = red + 400` placeholder -- see change_detection's module docstring.

    Raises EarthEngineUnavailable if ee.Initialize() fails (see above).
    Raises ValueError if fewer than 2 images are found for the given
    region/date range/cloud filter -- the same real "need >= 2 images"
    check the notebook made explicit in its own cell 6.
    """
    import ee
    import numpy as np

    ee_initialize(project=project)        # credentials first: a missing credential is the common failure
    import geemap

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .select(["B4", "B3", "B2", "B8"])
        .sort("system:time_start", False)  # most recent first, matching the original notebook
    )

    image_list = collection.toList(n_images)
    count = int(image_list.size().getInfo())
    if count < 2:
        raise ValueError(
            f"Only {count} Sentinel-2 image(s) found for this region/date range/cloud filter "
            f"(need at least 2 to compare). Try a wider date range, a higher max_cloud_pct, or a "
            f"different point/buffer."
        )

    frames = []
    for i in range(count):
        image = ee.Image(image_list.get(i)).clip(region)
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        arr = geemap.ee_to_numpy(image, region=region, scale=scale)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        rgb = arr[..., :3]   # B4, B3, B2 -> red, green, blue
        nir = arr[..., 3]    # B8
        frames.append(Sentinel2Frame(date=date, rgb=rgb, nir=nir))

    if len(frames) < 2:
        raise ValueError(f"Only {len(frames)} image(s) converted to real arrays successfully (need at least 2).")

    frames.sort(key=lambda f: f.date)  # oldest -> newest, chronological
    return frames
