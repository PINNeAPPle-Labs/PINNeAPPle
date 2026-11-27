"""The Penmanshiel wind-farm dataset: download, parse, canonicalise, quality-check.

Source
------
Plumley, C. & Takeuchi, R. *Penmanshiel wind farm data*. Zenodo, CC-BY-4.0,
doi:10.5281/zenodo.16807304 (2025 release, SCADA 2016-2024). Data descriptor:
Plumley (2022), *Scientific Data* 9:331.

The farm is 14 Senvion MM82 (2.05 MW, 82 m rotor, 59 m hub) in the Scottish
Borders. There is **no WT03** -- the numbering skips it, and code that assumes
14 turbines named 01..14 will crash on the missing one.

What the export actually contains
---------------------------------
Greenbyte writes one CSV per turbine-year with ~300 signals per 10-minute
sample, including three that change how this POC must be done:

``Potential power default PC (kW)``
    The operator's own expected power at the measured wind speed. It is an
    independent reference curve, not something we fitted, and it is used as such.
``Lost Production to Curtailment (Total) (kWh)`` and ``Turbine Power setpoint``
    Curtailment is *recorded*, so it does not have to be guessed from a power
    deficit heuristic. Curtailed samples are excluded from curve fits and kept in
    the AEP accounting.
``Data Availability`` and ``Lost Production to Downtime``
    Down-time is recorded too, so the operating mask is evidence-based.

``Density adjusted wind speed (m/s)`` is also supplied by the vendor. This
package recomputes its own IEC density correction from temperature (see
:mod:`~pinneapple_systems.wind.power_curve`) so the physics stays inspectable,
and keeps the vendor column as a cross-check rather than as an input.

What this module guarantees
---------------------------
* Canonical column names, so nothing downstream has to know Greenbyte's headers.
* An explicit **operating mask** built from availability, downtime and
  curtailment columns *when they exist*, falling back to physical rules when a
  column is absent -- and saying which path was taken.
* A machine-readable QC report with counts per rule, so a report can state how
  much data each filter removed instead of silently dropping it.

No imputation, no resampling, no smoothing. Missing samples stay missing and are
dropped from fits; every metric is computed on the samples that survived, and
the counts travel with the metrics.
"""
from __future__ import annotations

import csv
import re
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL_COLUMNS",
    "PENMANSHIEL_DOI",
    "ScadaFrame",
    "available_columns",
    "discover_archives",
    "load_farm",
    "load_turbine",
    "operating_mask",
    "qc_report",
    "read_archive",
    "static_layout",
    "summarise_by_turbine",
]

PENMANSHIEL_DOI = "10.5281/zenodo.16807304"
ZENODO_RECORD = 16_807_304

#: Canonical name -> anchored regexes matched against a lower-cased CSV header.
#: Anchors matter: ``^wind speed`` alone would also select the standard-deviation
#: and per-sensor columns and quietly mix three different signals.
_COLUMN_PATTERNS: dict[str, tuple[str, ...]] = {
    "timestamp": (r"^date and time$", r"^time ?stamp$"),
    "power_kw": (r"^power \(kw\)$",),
    "wind_speed_ms": (r"^wind speed \(m/s\)$",),
    "wind_speed_std_ms": (r"^wind speed, standard deviation \(m/s\)$",),
    "wind_speed_min_ms": (r"^wind speed, minimum \(m/s\)$",),
    "wind_speed_max_ms": (r"^wind speed, maximum \(m/s\)$",),
    "wind_speed_1_ms": (r"^wind speed sensor 1 \(m/s\)$",),
    "wind_speed_2_ms": (r"^wind speed sensor 2 \(m/s\)$",),
    "density_adjusted_wind_speed_ms": (r"^density adjusted wind speed \(m/s\)$",),
    "wind_direction_deg": (r"^wind direction \(°\)$",),
    "wind_direction_std_deg": (r"^wind direction, standard deviation \(°\)$",),
    "nacelle_deg": (r"^nacelle position \(°\)$",),
    "yaw_bearing_deg": (r"^yaw bearing angle \(°\)$",),
    "ambient_temperature_c": (r"^nacelle ambient temperature \(°c\)$",),
    "pitch_a_deg": (r"^blade angle \(pitch position\) a \(°\)$",),
    "pitch_b_deg": (r"^blade angle \(pitch position\) b \(°\)$",),
    "pitch_c_deg": (r"^blade angle \(pitch position\) c \(°\)$",),
    "rotor_rpm": (r"^rotor speed \(rpm\)$",),
    "generator_rpm": (r"^generator rpm \(rpm\)$",),
    "grid_frequency_hz": (r"^grid frequency \(hz\)$",),
    "energy_export_kwh": (r"^energy export \(kwh\)$",),
    "potential_power_kw": (r"^potential power default pc \(kw\)$",),
    "power_setpoint_kw": (r"^turbine power setpoint \(kw\)$",),
    "lost_production_curtailment_kwh": (r"^lost production to curtailment \(total\) \(kwh\)$",),
    "lost_production_downtime_kwh": (r"^lost production to downtime \(kwh\)$",),
    "data_availability": (r"^data availability$",),
    "performance_index": (r"^performance index$",),
}

CANONICAL_COLUMNS = tuple(_COLUMN_PATTERNS)

#: Columns the operating mask needs to be evidence-based rather than heuristic.
_CURTAILMENT_COLUMNS = ("lost_production_curtailment_kwh", "power_setpoint_kw")
_DOWNTIME_COLUMNS = ("lost_production_downtime_kwh", "data_availability")


# ---------------------------------------------------------------------------
# Archive discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Archive:
    """One SCADA archive on disk, with the year and turbine group it covers."""

    path: Path
    year: int
    turbines: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.path.name} ({self.year}, {len(self.turbines)} turbines)"


def discover_archives(raw_dir: str | Path) -> list[Archive]:
    """Find ``Penmanshiel_SCADA_<year>_WT...zip`` files and parse their range.

    Anything that does not match the naming pattern is ignored rather than
    guessed at: a mis-parsed archive silently mixing two years would corrupt
    every temporal split downstream. A file that matches the name but is not a
    readable zip (a truncated or still-running download) is also skipped, so a
    half-downloaded year cannot take the whole load down with it.
    """
    raw = Path(raw_dir)
    out: list[Archive] = []
    for path in sorted(raw.glob("*.zip")):
        m = re.search(r"SCADA_(\d{4})_WT_?(\d+)-(\d+)", path.name)
        if not m or not zipfile.is_zipfile(path):
            continue
        year = int(m.group(1))
        lo, hi = int(m.group(2)), int(m.group(3))
        names = tuple(f"T{n:02d}" for n in range(lo, hi + 1) if n != 3)
        out.append(Archive(path=path, year=year, turbines=names))
    return out


def _turbine_from_member(member: str) -> str | None:
    """Turbine id from a Greenbyte member name (``..._Penmanshiel_11_2016-...csv``)."""
    name = Path(member).name
    m = re.search(r"Penmanshiel[ _](?:WT)?[_-]?(\d{1,2})", name)
    if not m:
        return None
    return f"T{int(m.group(1)):02d}"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def available_columns(header: Sequence[str]) -> dict[str, str]:
    """Map canonical name -> source header for the columns a file actually has."""
    lowered = {h: h.strip().lower() for h in header}
    mapping: dict[str, str] = {}
    for canonical, patterns in _COLUMN_PATTERNS.items():
        for pattern in patterns:
            hit = next((h for h, low in lowered.items() if re.search(pattern, low)), None)
            if hit is not None:
                mapping[canonical] = hit
                break
    return mapping


def _split_header(text: str) -> tuple[int, list[str]]:
    """Greenbyte writes comment lines, then a header line that also starts with '#'.

    Returns ``(row_index_of_first_data_line, column_names)``.
    """
    lines = text.splitlines()
    header_at = None
    for i, line in enumerate(lines):
        if line.startswith("#") and ("date and time" in line.lower() or "timestamp" in line.lower()):
            header_at = i
            break
    if header_at is None:
        header_at = 0
    header = lines[header_at].lstrip("#").strip()
    return header_at + 1, next(csv.reader([header]))


def read_archive(path: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read every turbine CSV in one archive into canonical frames keyed by turbine.

    ``Status_*`` members (event logs) are listed in the report and skipped: they
    are a different table with a different schema and mixing them into the SCADA
    frame would silently produce NaN columns.

    Returns ``(frames, report)``. The report records which source columns were
    mapped and which canonical columns are missing for each file -- a silently
    absent ``ambient_temperature_c`` would quietly disable the density
    correction, so it is surfaced instead.
    """
    path = Path(path)
    frames: dict[str, pd.DataFrame] = {}
    report: dict = {"archive": path.name, "files": [], "skipped": [], "unmapped": []}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".csv"):
                continue
            if info.filename.lower().startswith(("status_", "events_")):
                report["skipped"].append(info.filename)
                continue
            turbine = _turbine_from_member(info.filename)
            with zf.open(info) as fh:
                text = fh.read().decode("utf-8-sig", errors="replace")
            first_data_row, header = _split_header(text)
            columns = available_columns(header)
            if turbine is None or "power_kw" not in columns:
                report["unmapped"].append(info.filename)
                continue
            df = _read_csv(text, header, first_data_row, columns)
            report["files"].append({
                "file": info.filename,
                "turbine": turbine,
                "rows": int(len(df)),
                "mapped": sorted(columns),
                "missing": sorted(set(CANONICAL_COLUMNS) - set(columns)),
            })
            frames[turbine] = df if turbine not in frames else pd.concat([frames[turbine], df])
    return frames, report


def _read_csv(text: str, header: Sequence[str], first_data_row: int,
              columns: dict[str, str]) -> pd.DataFrame:
    """Parse one CSV blob keeping only the mapped columns.

    Streamed row by row rather than through ``pandas.read_csv`` on the whole
    text: the export is ~300 columns wide and only ~25 are used, and the wide
    parse dominated the load time of the whole POC.
    """
    header = [h.strip() for h in header]
    keep = {src: canonical for canonical, src in columns.items()}
    idx = {i: keep[name] for i, name in enumerate(header) if name in keep}
    rows = []
    for line in text.splitlines()[first_data_row:]:
        if not line or line.startswith("#"):
            continue
        fields = next(csv.reader([line]))
        if not fields or not fields[0].strip():
            continue
        record = {}
        for i, canon in idx.items():
            if i < len(fields):
                value = fields[i]
                record[canon] = value if value not in ("", "NaN", "nan") else np.nan
        rows.append(record)
    return canonicalise(pd.DataFrame(rows))


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, add derived columns, drop rows without a timestamp.

    Derived columns added here (computed once, used everywhere):

    ``pitch_deg``
        Mean of the three blade angles; NaN only when all three are missing.
    ``yaw_error_deg``
        Signed angle between nacelle and wind, wrapped to [-180, 180].
    ``anemometer_spread``
        Difference between the two nacelle anemometers, a crude proxy for inflow
        variability; NaN when the second sensor is absent.
    ``rho_relative``
        Dry-air density relative to 1.225 kg/m^3, from the nacelle ambient
        temperature and the standard-atmosphere pressure at site elevation. This
        is the package's own IEC correction input -- the vendor's
        ``density_adjusted_wind_speed_ms`` is kept untouched for cross-checking.
    """
    out = df.copy()
    for col in [c for c in out.columns if c != "timestamp"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "timestamp" in out:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", format="mixed")
        out = out.dropna(subset=["timestamp"])
    pitches = [c for c in ("pitch_a_deg", "pitch_b_deg", "pitch_c_deg") if c in out]
    if pitches:
        out["pitch_deg"] = out[pitches].mean(axis=1, skipna=True)
    if {"nacelle_deg", "wind_direction_deg"} <= set(out.columns):
        delta = out["nacelle_deg"] - out["wind_direction_deg"]
        out["yaw_error_deg"] = ((delta + 180.0) % 360.0) - 180.0
    if {"wind_speed_ms", "wind_speed_2_ms"} <= set(out.columns):
        out["anemometer_spread"] = out["wind_speed_ms"] - out["wind_speed_2_ms"]
    return out.sort_values("timestamp").reset_index(drop=True)


def _attach_turbine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, df in frames.items():
        df = df.copy()
        df["turbine"] = name
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["turbine", "timestamp"]).reset_index(drop=True)


def load_turbine(
    raw_dir: str | Path,
    turbine: str = "T01",
    years: Sequence[int] | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load one turbine's SCADA over the requested years (all available by default).

    ``cache_dir`` writes one pickle per (turbine, year) after parsing; a re-run
    then skips the 300-column CSV. The cache is keyed by the archive name, so a
    re-download is picked up rather than shadowed.
    """
    frames: dict[str, pd.DataFrame] = {}
    for archive in discover_archives(raw_dir):
        if turbine not in archive.turbines:
            continue
        if years is not None and archive.year not in years:
            continue
        cached = _cache_path(cache_dir, turbine, archive)
        if cached is not None and cached.exists():
            frames[str(archive.year)] = pd.read_pickle(cached)
            continue
        part, _ = read_archive(archive.path)
        if turbine in part:
            frames[str(archive.year)] = part[turbine]
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                part[turbine].to_pickle(cached)
    if not frames:
        raise FileNotFoundError(f"no archive in {raw_dir} contains {turbine} for years {years}")
    df = pd.concat(frames.values(), ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df.reset_index(drop=True)


def _cache_path(cache_dir, turbine: str, archive: Archive) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{turbine}_{archive.year}_{archive.path.stem}.pkl"


def load_farm(
    raw_dir: str | Path,
    turbines: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load every turbine into one long frame with a ``turbine`` column.

    Delegates to :func:`load_turbine` per machine, so the parse cache is used and
    a turbine whose archive is missing or truncated is skipped with a warning
    rather than aborting the whole farm.
    """
    if turbines is None:
        candidates = sorted({
            name
            for archive in discover_archives(raw_dir)
            if years is None or archive.year in years
            for name in archive.turbines
        })
    else:
        candidates = list(turbines)
    merged: dict[str, pd.DataFrame] = {}
    for name in candidates:
        try:
            merged[name] = load_turbine(raw_dir, name, years, cache_dir=cache_dir)
        except FileNotFoundError:
            continue
    if not merged:
        raise FileNotFoundError(f"no turbine data found in {raw_dir} for years {years}")
    return _attach_turbine(merged)


# ---------------------------------------------------------------------------
# Quality control and the operating mask
# ---------------------------------------------------------------------------

def qc_rules(
    df: pd.DataFrame,
    *,
    rated_power_kw: float,
    cut_in: float = 3.0,
) -> list[dict]:
    """The data-quality rules, each with its own mask and removal count.

    The rules are deliberately conservative and physical: they remove states a
    turbine cannot be in, not states that merely look unusual. Anything subtler
    (icing, sensor drift, sector management) is *recorded in the dataset itself*
    and handled by :func:`operating_mask`, not silently dropped here.
    """
    n = len(df)
    rules: list[dict] = []

    def add(name: str, description: str, bad) -> None:
        mask = np.asarray(bad, dtype=bool)
        if mask.shape != (n,):
            mask = np.zeros(n, dtype=bool)
        rules.append({"name": name, "description": description, "mask": mask,
                      "removed": int(mask.sum())})

    if "timestamp" in df:
        add("missing_timestamp", "timestamp missing or unparseable", df["timestamp"].isna().to_numpy())
    for col in ("power_kw", "wind_speed_ms"):
        if col in df:
            add(f"missing_{col}", f"{col} missing", df[col].isna().to_numpy())
    if "power_kw" in df:
        add("power_negative", "power below -1 % of rated",
            (df["power_kw"].to_numpy() < -0.01 * rated_power_kw))
        add("power_above_rated", "power above 115 % of rated (sensor or scaling fault)",
            (df["power_kw"].to_numpy() > 1.15 * rated_power_kw))
    if "wind_speed_ms" in df:
        add("wind_speed_out_of_range", "nacelle wind speed outside [0, 40] m/s",
            (df["wind_speed_ms"].to_numpy() < 0.0) | (df["wind_speed_ms"].to_numpy() > 40.0))
    if "ambient_temperature_c" in df:
        add("temperature_out_of_range", "nacelle ambient temperature outside [-40, 60] degC",
            (df["ambient_temperature_c"].to_numpy() < -40.0)
            | (df["ambient_temperature_c"].to_numpy() > 60.0))
    _ = cut_in  # the operating window is applied by operating_mask
    return rules


def operating_mask(
    df: pd.DataFrame,
    *,
    rated_power_kw: float,
    cut_in: float = 3.0,
    min_power_fraction: float = 0.005,
    availability_floor: float = 0.95,
) -> np.ndarray:
    """Samples where the machine is genuinely generating, uncurtailed and QC-clean.

    Evidence-based when the export provides the columns (availability, lost
    production to downtime, lost production to curtailment, power setpoint);
    falls back to the physical rules otherwise. The fallback is less precise and
    :func:`mask_provenance` says which path was taken, so a report never claims
    the stronger filter it did not run.

    "Generating" additionally requires positive power and a wind speed above
    cut-in: standstill, idling and down-time would otherwise fill the low-wind
    bins with zeros and drag the measured curve down.
    """
    n = len(df)
    ok = np.ones(n, dtype=bool)
    for rule in qc_rules(df, rated_power_kw=rated_power_kw, cut_in=cut_in):
        ok &= ~rule["mask"]
    if "data_availability" in df:
        ok &= df["data_availability"].to_numpy() >= availability_floor
    if "lost_production_downtime_kwh" in df:
        ok &= ~(df["lost_production_downtime_kwh"].to_numpy() > 1.0)
    if "lost_production_curtailment_kwh" in df:
        ok &= ~(df["lost_production_curtailment_kwh"].to_numpy() > 1.0)
    if "power_setpoint_kw" in df:
        setpoint = df["power_setpoint_kw"].to_numpy()
        ok &= ~np.isfinite(setpoint) | (setpoint >= 0.98 * rated_power_kw)
    if "power_kw" in df:
        ok &= df["power_kw"].to_numpy() > min_power_fraction * rated_power_kw
    if "wind_speed_ms" in df:
        ok &= df["wind_speed_ms"].to_numpy() >= cut_in
    return ok


def mask_provenance(df: pd.DataFrame) -> dict:
    """Which evidence the operating mask actually had available."""
    curtail = [c for c in _CURTAILMENT_COLUMNS if c in df]
    down = [c for c in _DOWNTIME_COLUMNS if c in df]
    return {
        "evidence_based": bool(curtail and down),
        "curtailment_columns": curtail,
        "downtime_columns": down,
        "note": (
            "Curtailment and downtime are taken from the dataset's own recorded lost-production "
            "columns; where they are absent the mask falls back to physical rules only."
        ),
    }


def curtailment_mask(df: pd.DataFrame) -> np.ndarray:
    """Samples the operator curtailed, from the recorded lost-production columns."""
    n = len(df)
    flag = np.zeros(n, dtype=bool)
    if "lost_production_curtailment_kwh" in df:
        flag |= df["lost_production_curtailment_kwh"].to_numpy() > 1.0
    if "power_setpoint_kw" in df:
        flag |= df["power_setpoint_kw"].to_numpy() < 0.0  # explicit zero setpoint
    return flag


def qc_report(df: pd.DataFrame, *, rated_power_kw: float, cut_in: float = 3.0) -> dict:
    """Machine-readable summary of the QC pass, per rule and overall."""
    rules = qc_rules(df, rated_power_kw=rated_power_kw, cut_in=cut_in)
    operating = operating_mask(df, rated_power_kw=rated_power_kw, cut_in=cut_in)
    span = {}
    if "timestamp" in df and len(df):
        start, end = pd.to_datetime(df["timestamp"]).min(), pd.to_datetime(df["timestamp"]).max()
        span = {
            "start": str(start),
            "end": str(end),
            "expected_10min_samples": int((end - start).total_seconds() // 600 + 1),
        }
    curtailed = curtailment_mask(df)
    return {
        "rows": int(len(df)),
        "unique_timestamps": int(df["timestamp"].nunique()) if "timestamp" in df else 0,
        "columns_present": sorted(df.columns),
        "span": span,
        "rules": [
            {"name": r["name"], "description": r["description"], "removed": r["removed"],
             "fraction": r["removed"] / max(len(df), 1)}
            for r in rules
        ],
        "operating_samples": int(operating.sum()),
        "operating_fraction": float(operating.mean()) if len(df) else 0.0,
        "curtailed_samples": int(curtailed.sum()),
        "curtailed_fraction": float(curtailed.mean()) if len(df) else 0.0,
        "mask_provenance": mask_provenance(df),
    }


def add_density_correction(
    df: pd.DataFrame,
    *,
    elevation_m: float = 200.0,
    temperature_col: str = "ambient_temperature_c",
) -> pd.DataFrame:
    """Add ``rho_relative`` and ``u_corrected`` (IEC density-normalised speed).

    Kept as a separate step rather than baked into :func:`canonicalise` because
    it needs the site elevation, which is a property of the farm, not of the CSV.
    """
    from .power_curve import density_corrected_speed, relative_density

    out = df.copy()
    if temperature_col in out:
        out["rho_relative"] = np.asarray(relative_density(out[temperature_col].to_numpy(float), elevation_m))
    else:
        out["rho_relative"] = 1.0
    if "wind_speed_ms" in out:
        out["u_corrected"] = np.asarray(density_corrected_speed(out["wind_speed_ms"].to_numpy(float),
                                                                out["rho_relative"].to_numpy(float)))
    return out


def static_layout(raw_dir: str | Path):
    """The published turbine table as a :class:`~pinneapple_systems.wind.layout.WindFarmLayout`."""
    from .layout import WindFarmLayout

    path = Path(raw_dir) / "Penmanshiel_WT_static.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- download Penmanshiel_WT_static.csv from doi:{PENMANSHIEL_DOI}"
        )
    return WindFarmLayout.from_penmanshiel_csv(path)


@dataclass
class ScadaFrame:
    """A loaded turbine-year with its QC verdict, kept together so they cannot drift."""

    turbine: str
    frame: pd.DataFrame
    qc: dict
    meta: dict = field(default_factory=dict)

    @property
    def operating(self) -> pd.DataFrame:
        mask = operating_mask(self.frame, rated_power_kw=self.meta["rated_power_kw"])
        return self.frame[mask]

    def __len__(self) -> int:
        return len(self.frame)


def summarise_by_turbine(farm: pd.DataFrame, rated_power_kw: float) -> list[dict]:
    """Per-turbine sample counts and mean production, for the exploration section."""
    rows = []
    for name, df in farm.groupby("turbine"):
        op = operating_mask(df, rated_power_kw=rated_power_kw)
        rows.append({
            "turbine": name,
            "samples": int(len(df)),
            "operating": int(op.sum()),
            "operating_fraction": float(op.mean()) if len(df) else 0.0,
            "mean_power_operating_kw": float(df.loc[op, "power_kw"].mean()) if op.any() else None,
            "mean_wind_operating_ms": float(df.loc[op, "wind_speed_ms"].mean()) if op.any() else None,
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        })
    return sorted(rows, key=lambda r: r["turbine"])


def iter_csv_members(path: str | Path) -> Iterable[str]:
    """Names of the CSV members inside an archive (for inspection without extracting)."""
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".csv"):
                yield info.filename
