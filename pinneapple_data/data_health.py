"""Engineering data health: a physics-aware quality report for sensor / test / plant data.

``load_table(bytes, filename)`` reads CSV/TSV (delimiter and decimal comma detected), Excel,
Parquet, JSON and HDF5 (1-D datasets, with ``units`` attributes when present).
``analyze(df)`` returns a JSON-ready report:

* **Units and quantities** of every column, from the header (:mod:`pinneapple_data.physical_units`).
* **Completeness**: missing values and long missing runs.
* **Time axis**: timestamps that go backwards, duplicates, irregular sampling, gaps.
* **Validity**: physically impossible values (below absolute zero, relative humidity above 100 %,
  pressure below vacuum), sentinel / error codes (-999, 9999, 32767...), a unit that does not fit
  the values (a "°C" column that holds kelvin) and a unit that changes mid-recording.
* **Signal quality**: stuck sensors (flat lines), spikes (Hampel filter), clipping at a range limit,
  constant columns, coarse resolution.
* **Physics consistency** (only when the columns allow it): the thermal energy balance
  Q = rho * V * cp * dT between a supply/return temperature pair, the flow and the reported
  heat rate. A flow meter that under-reads produces plausible values one column at a time; only
  the balance shows it.
* **Score**: every issue has a severity and the share of the data it affects. Each category's score
  is 100 minus its issue penalties; the overall score is the weighted mean of the categories that
  apply. The formula is returned with the report (``score.method``).

The checks are deterministic and explainable. Each issue carries its evidence (rows, time spans,
values) and a suggested fix.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pinneapple_core.jsonable import to_jsonable
from pinneapple_core.series import bucket_edges

from .physical_units import QUANTITIES, Unit, convert, infer_quantity, split_header, try_parse_unit

SEVERITY_WEIGHT = {"critical": 30.0, "warning": 10.0, "info": 2.0}
CATEGORY_WEIGHT = {"completeness": 0.20, "time_axis": 0.20, "validity": 0.25, "signal_quality": 0.20,
                   "physics_consistency": 0.15, "metadata": 0.10}
CATEGORY_LABEL = {"completeness": "Completeness", "time_axis": "Time axis", "validity": "Physical validity",
                  "signal_quality": "Signal quality", "physics_consistency": "Physics consistency",
                  "metadata": "Units & metadata"}
SENTINELS = (-9999.0, -999.0, -99.0, 9999.0, 99999.0, -32768.0, 32767.0, 65535.0, -99999.0, 999999.0)


@dataclass
class Issue:
    category: str
    severity: str                      # critical | warning | info
    title: str
    detail: str
    column: Optional[str] = None
    affected_fraction: float = 0.0     # share of the rows (or of the column) affected, 0..1
    count: int = 0
    spans: List[Tuple[str, str]] = field(default_factory=list)
    suggestion: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadInfo:
    format: str
    rows: int
    columns: int
    delimiter: Optional[str] = None
    decimal: Optional[str] = None
    encoding: Optional[str] = None
    sheet: Optional[str] = None
    sheets: List[str] = field(default_factory=list)
    units_from_file: Dict[str, str] = field(default_factory=dict)
    truncated_to: Optional[int] = None
    notes: List[str] = field(default_factory=list)


# ── loading ─────────────────────────────────────────────────────────────────
def _decode(data: bytes) -> Tuple[str, str]:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1"


def _sniff(text: str) -> Tuple[str, str]:
    lines = [l for l in text.splitlines()[:50] if l.strip()]
    sample = "\n".join(lines)
    best, best_score = ",", -1.0
    for d in (",", ";", "\t", "|"):
        counts = [l.count(d) for l in lines]                      # header included: it has no decimal commas
        if not counts or max(counts) == 0:
            continue
        mode = max(set(counts), key=counts.count)
        score = counts.count(mode) / len(counts) * (1 + min(mode, 30) / 30)
        if mode > 0 and score > best_score:
            best, best_score = d, score
    decimal = "."
    if best != ",":
        if len(re.findall(r"\d,\d", sample)) > 2 * max(1, len(re.findall(r"\d\.\d", sample))):
            decimal = ","
    return best, decimal


def load_table(data: bytes, filename: str, sheet: Optional[str] = None,
               max_rows: int = 1_000_000) -> Tuple[pd.DataFrame, LoadInfo]:
    """Read an uploaded table. Supported: .csv .tsv .txt .xlsx .xlsm .xls .parquet .json .jsonl .h5 .hdf5"""
    name = filename.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    bio = io.BytesIO(data)
    info: LoadInfo
    if ext in ("csv", "tsv", "txt", "dat", ""):
        text, enc = _decode(data)
        delim, dec = _sniff(text)
        if ext == "tsv":
            delim = "\t"
        df = pd.read_csv(io.StringIO(text), sep=delim, decimal=dec, low_memory=False,
                         thousands="." if dec == "," and delim != "." else None)
        info = LoadInfo("csv", len(df), df.shape[1], delimiter={"\t": "tab"}.get(delim, delim), decimal=dec, encoding=enc)
    elif ext in ("xlsx", "xlsm", "xls"):
        xl = pd.ExcelFile(bio)
        sh = sheet if sheet in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sh)
        info = LoadInfo("excel", len(df), df.shape[1], sheet=sh, sheets=list(xl.sheet_names))
    elif ext in ("parquet", "pq"):
        df = pd.read_parquet(bio)
        info = LoadInfo("parquet", len(df), df.shape[1])
        try:
            import pyarrow.parquet as pq
            meta = pq.read_schema(io.BytesIO(data)).metadata or {}
            units = json.loads(meta.get(b"pinneapple.units", b"{}")) if meta else {}
            info.units_from_file = {k: v for k, v in units.items() if k in df.columns}
        except Exception:
            pass
    elif ext in ("json", "jsonl", "ndjson"):
        text, enc = _decode(data)
        df = _read_json(text, lines=ext != "json")
        info = LoadInfo("json", len(df), df.shape[1], encoding=enc)
    elif ext in ("h5", "hdf5", "hdf", "he5"):
        df, units, notes = _read_hdf5(data)
        info = LoadInfo("hdf5", len(df), df.shape[1], units_from_file=units, notes=notes)
    else:
        raise ValueError(f"Unsupported file type '.{ext}'. Use CSV, Excel, Parquet, JSON or HDF5.")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed:") or df[c].notna().any()]]
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
        info.truncated_to = max_rows
        info.notes.append(f"Only the first {max_rows:,} rows were analysed.")
    info.rows, info.columns = len(df), df.shape[1]
    return df, info


def _read_json(text: str, lines: bool) -> pd.DataFrame:
    try:
        if lines:
            return pd.read_json(io.StringIO(text), lines=True)
        obj = json.loads(text)
    except ValueError:
        return pd.read_json(io.StringIO(text), lines=True)
    if isinstance(obj, dict):
        for key in ("data", "records", "rows", "values", "items", "results"):
            if isinstance(obj.get(key), list):
                return pd.json_normalize(obj[key])
        if all(isinstance(v, list) for v in obj.values()):
            return pd.DataFrame(obj)
        return pd.json_normalize(obj)
    return pd.json_normalize(obj)


def _read_hdf5(data: bytes) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    import h5py
    cols: Dict[str, np.ndarray] = {}
    units: Dict[str, str] = {}
    notes: List[str] = []
    with h5py.File(io.BytesIO(data), "r") as f:
        items: List[Tuple[str, Any]] = []
        f.visititems(lambda n, o: items.append((n, o)) if isinstance(o, h5py.Dataset) else None)
        for n, ds in items:
            if ds.dtype.names:                               # compound table
                arr = ds[()]
                for fn in ds.dtype.names:
                    cols[f"{n}/{fn}" if len(items) > 1 else fn] = np.asarray(arr[fn])
            elif ds.ndim == 1 or (ds.ndim == 2 and min(ds.shape) == 1):
                cols[n] = np.asarray(ds[()]).reshape(-1)
                u = ds.attrs.get("units", ds.attrs.get("unit"))
                if u is not None:
                    units[n] = u.decode() if isinstance(u, bytes) else str(u)
            else:
                notes.append(f"Skipped '{n}' (shape {ds.shape}): only 1-D signals are analysed.")
    if not cols:
        raise ValueError("No 1-D datasets found in the HDF5 file.")
    lengths = pd.Series({k: len(v) for k, v in cols.items()})
    n = int(lengths.mode().iloc[0])
    keep = {k: (v.astype(str) if v.dtype.kind in "SO" else v) for k, v in cols.items() if len(v) == n}
    for k in set(cols) - set(keep):
        notes.append(f"Skipped '{k}': length {len(cols[k])} differs from the main table ({n}).")
    df = pd.DataFrame({k.split("/")[-1] if sum(1 for j in keep if j.split('/')[-1] == k.split('/')[-1]) == 1 else k: v
                       for k, v in keep.items()})
    units = {k.split("/")[-1] if k.split("/")[-1] in df.columns else k: v for k, v in units.items()}
    return df, units, notes


# ── time axis ───────────────────────────────────────────────────────────────
_TIME_NAME = re.compile(r"^(time|timestamp|date|datetime|date_time|ts|utc|local_time|time_stamp|data|hora|"
                        r"data_hora|datahora|tempo|fecha)(\b|_|\s|\[|\()", re.I)


def _parse_time(s: pd.Series) -> Tuple[Optional[pd.Series], str]:
    """Return a datetime series (naive) or None, and how it was interpreted."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert("UTC").dt.tz_localize(None)
        return out, "datetime column"
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if len(v) and v.between(1e9, 2.2e9).all():
            return pd.to_datetime(s, unit="s", errors="coerce"), "Unix seconds"
        if len(v) and v.between(1e12, 2.2e12).all():
            return pd.to_datetime(s, unit="ms", errors="coerce"), "Unix milliseconds"
        if len(v) and v.between(20000, 80000).all() and (v % 1 != 0).any():
            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce"), "Excel serial date"
        return None, ""
    txt = s.astype(str).str.strip()
    head = txt.dropna().head(5000)
    m = head.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
    options = (False, True)
    if m[0].notna().any():
        first, second = pd.to_numeric(m[0], errors="coerce"), pd.to_numeric(m[1], errors="coerce")
        if (first > 12).any():
            options = (True,)
        elif (second > 12).any():
            options = (False,)
    best, how, best_score = None, "", float("inf")
    for dayfirst in options:
        try:
            p = pd.to_datetime(txt, errors="coerce", dayfirst=dayfirst, format="mixed")
        except (TypeError, ValueError):
            p = pd.to_datetime(txt, errors="coerce", dayfirst=dayfirst)
        bad = float(p.isna().mean())
        d = p.dropna().diff().dt.total_seconds().dropna()
        mono = float((d >= 0).mean()) if len(d) else 0.0
        regular = float((d == d.mode().iloc[0]).mean()) if len(d) else 0.0   # ambiguous d/m: the right order is regular
        score = bad + (1 - mono) + (1 - regular)
        if score < best_score:
            best, best_score = p, score
            how = ("day-first dates (dd/mm)" if dayfirst else "month-first dates (mm/dd)") if m[0].notna().any() \
                else "ISO 8601 timestamps"
    if best is None or best.isna().mean() > 0.2:
        return None, ""
    if getattr(best.dt, "tz", None) is not None:
        best = best.dt.tz_convert("UTC").dt.tz_localize(None)
        how += ", converted to UTC"
    return best, how


def detect_time_column(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series], str]:
    order = sorted(df.columns, key=lambda c: (0 if _TIME_NAME.match(str(c)) else 1, list(df.columns).index(c)))
    for c in order:
        s = df[c]
        if not _TIME_NAME.match(str(c)) and not pd.api.types.is_datetime64_any_dtype(s) and s.dtype != object \
                and not pd.api.types.is_string_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and not _TIME_NAME.match(str(c)):
            continue
        t, how = _parse_time(s)
        if t is not None:
            return c, t, how
    return None, None, ""


def _fmt_t(t) -> str:
    if isinstance(t, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(t).isoformat(sep=" ", timespec="minutes")
    return str(t)


def _fmt_dur(sec: float) -> str:
    if not np.isfinite(sec):
        return "—"
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if sec >= n:
            return f"{sec / n:.1f} {unit}"
    return f"{sec:.3g} s"


# ── helpers ─────────────────────────────────────────────────────────────────
def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if not len(mask):
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _to_numeric(s: pd.Series) -> Tuple[pd.Series, bool]:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float), False
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float), False
    txt = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan, "null": np.nan})
    num = pd.to_numeric(txt, errors="coerce")
    if num.notna().sum() < 0.8 * txt.notna().sum():
        alt = pd.to_numeric(txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
        if alt.notna().sum() > num.notna().sum():
            num = alt
    ok = txt.notna().sum() > 0 and num.notna().sum() >= 0.8 * txt.notna().sum()
    return (num.astype(float), True) if ok else (s, False)


def _span(t: Optional[pd.Series], idx: Sequence[int], a: int, b: int) -> Tuple[str, str]:
    if t is not None:
        return _fmt_t(t.iloc[idx[a]]), _fmt_t(t.iloc[idx[min(b, len(idx)) - 1]])
    return f"row {idx[a] + 2}", f"row {idx[min(b, len(idx)) - 1] + 2}"


# ── analysis ────────────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame, units: Optional[Dict[str, str]] = None, time_column: Optional[str] = None,
            n_bins: int = 160, series_points: int = 600) -> Dict[str, Any]:
    units = dict(units or {})
    issues: List[Issue] = []
    n = len(df)
    if n == 0:
        raise ValueError("The table is empty.")

    # time axis
    if time_column and time_column in df.columns:
        t, how = _parse_time(df[time_column])
        tcol = time_column if t is not None else None
    else:
        tcol, t, how = detect_time_column(df)
    order = np.arange(n)
    summary: Dict[str, Any] = {"rows": n, "columns": int(df.shape[1]), "time_column": tcol, "time_parsed_as": how}
    if t is not None:
        issues += _check_time(t, summary)
        valid = t.notna().to_numpy()
        order = np.argsort(t.fillna(t.max()).to_numpy(), kind="stable")
        order = order[valid[order]]
        t_sorted = t.iloc[order].reset_index(drop=True)
    else:
        t_sorted = None
        issues.append(Issue("time_axis", "warning", "No time column found",
                            "No column could be read as timestamps, so gaps, sampling and DST problems were not "
                            "checked and rows are treated as equally spaced samples.",
                            suggestion="Add a timestamp column (ISO 8601, e.g. 2026-03-01 14:05:00) or pick it in the "
                                       "column list.", affected_fraction=1.0))

    # columns
    columns: List[Dict[str, Any]] = []
    numeric: Dict[str, pd.Series] = {}
    meta_missing_units = []
    for c in df.columns:
        if c == tcol:
            continue
        raw = df[c]
        base, utxt = split_header(c)
        if c in units:
            utxt = units[c]
        unit = try_parse_unit(utxt) if utxt else None
        guess = infer_quantity(base, unit)
        num, was_text = _to_numeric(raw)
        col: Dict[str, Any] = {"name": c, "base_name": base, "unit": utxt, "unit_known": unit is not None,
                               "quantity": guess.quantity, "quantity_confidence": round(guess.confidence, 2),
                               "quantity_reason": guess.reason, "is_difference": guess.is_difference,
                               "dtype": str(raw.dtype), "missing_pct": float(raw.isna().mean() * 100)}
        if not pd.api.types.is_numeric_dtype(num):
            col.update({"kind": "text", "n_unique": int(raw.nunique(dropna=True)),
                        "examples": [str(x) for x in raw.dropna().astype(str).unique()[:5]]})
            columns.append(col)
            continue
        if was_text:
            issues.append(Issue("metadata", "warning", "Numbers stored as text", f"'{c}' holds numbers as text "
                                "(quotes, decimal commas or thousand separators); they were converted for this analysis.",
                                column=c, affected_fraction=1.0,
                                suggestion="Export numbers as numeric values with a dot as decimal separator."))
        v = num.iloc[order].reset_index(drop=True).to_numpy(dtype=float)
        numeric[c] = pd.Series(v)
        finite = v[np.isfinite(v)]
        col.update({"kind": "numeric", "n_unique": int(pd.Series(finite).nunique()),
                    "min": float(finite.min()) if len(finite) else None,
                    "max": float(finite.max()) if len(finite) else None,
                    "mean": float(finite.mean()) if len(finite) else None,
                    "std": float(finite.std()) if len(finite) > 1 else None,
                    "p01": float(np.percentile(finite, 1)) if len(finite) else None,
                    "p99": float(np.percentile(finite, 99)) if len(finite) else None})
        if unit is None:
            meta_missing_units.append(c)
        col_issues = _check_column(c, v, unit, utxt, guess.quantity, guess.is_difference, t_sorted)
        issues += col_issues
        columns.append(col)

    if meta_missing_units:
        k = len(meta_missing_units)
        issues.append(Issue("metadata", "warning" if k > 0.3 * max(1, len(numeric)) else "info",
                            f"{k} column{'s' if k > 1 else ''} without a unit",
                            "No unit was found in the header (e.g. 'T_supply [°C]') or in the file metadata for: "
                            + ", ".join(meta_missing_units[:12]) + ("…" if k > 12 else "")
                            + ". Physical-limit and unit checks were skipped for them.",
                            affected_fraction=k / max(1, len(numeric)), count=k,
                            suggestion="Put the unit in the header, e.g. 'Flow [m3/h]', or provide a units mapping.",
                            evidence={"columns": meta_missing_units}))
    issues += _check_redundancy(numeric)
    relations, rel_issues = _check_energy_balance(columns, numeric, t_sorted)
    issues += rel_issues

    # per-column flags / scores
    by_col: Dict[str, List[Issue]] = {}
    for i in issues:
        if i.column:
            by_col.setdefault(i.column, []).append(i)
    for col in columns:
        ci = by_col.get(col["name"], [])
        col["flags"] = [{"severity": i.severity, "title": i.title} for i in ci]
        pen = sum(SEVERITY_WEIGHT[i.severity] * (0.5 + 0.5 * min(1.0, i.affected_fraction * 10)) for i in ci)
        col["score"] = max(0.0, 100.0 - pen)

    score = _score(issues, applicable_physics=bool(relations))
    report = {
        "summary": summary,
        "score": score,
        "columns": columns,
        "issues": [asdict(i) for i in sorted(issues, key=lambda i: (["critical", "warning", "info"].index(i.severity),
                                                                    -i.affected_fraction))],
        "relations": relations,
        "availability": _availability(numeric, t_sorted, issues, n_bins),
        "series": _series(numeric, t_sorted, issues, series_points),
    }
    return _clean(report)


def _check_time(t: pd.Series, summary: Dict[str, Any]) -> List[Issue]:
    out: List[Issue] = []
    n = len(t)
    nat = int(t.isna().sum())
    if nat:
        out.append(Issue("time_axis", "warning" if nat / n > 0.01 else "info", "Unreadable timestamps",
                         f"{nat} timestamps could not be parsed and their rows were left out of the time checks.",
                         affected_fraction=nat / n, count=nat, suggestion="Use one date format for the whole file."))
    tv = t.dropna()
    if len(tv) < 3:
        return out
    d = tv.diff().dt.total_seconds().to_numpy()[1:]
    back = np.where(d < 0)[0]
    if len(back):
        hours = np.round(-d[back] / 3600, 2)
        dst = int(np.sum(np.isclose(hours, 1.0, atol=0.05)))
        out.append(Issue("time_axis", "critical" if len(back) > 0.001 * n or dst else "warning", "Timestamps go backwards",
                         f"Time jumps backwards {len(back)} time{'s' if len(back) > 1 else ''}"
                         + (f"; {dst} of the jumps are exactly 1 h, the signature of a daylight-saving change logged "
                            "in local time" if dst else "") + ". Rows were sorted by time for the remaining checks.",
                         affected_fraction=len(back) / n, count=len(back),
                         spans=[(_fmt_t(tv.iloc[i]), _fmt_t(tv.iloc[i + 1])) for i in back[:10]],
                         suggestion="Log in UTC (or with a UTC offset) and keep rows in time order."))
    tsorted = tv.sort_values()
    ds = tsorted.diff().dt.total_seconds().to_numpy()[1:]
    dup = int(np.sum(ds == 0))
    if dup:
        dup_times = tsorted[tsorted.duplicated(keep=False)]
        out.append(Issue("time_axis", "warning" if dup / n > 0.001 else "info", "Duplicate timestamps",
                         f"{dup} rows repeat a timestamp that already exists (repeated uploads or a DST fall-back). "
                         "Duplicates bias averages and break resampling.",
                         affected_fraction=dup / n, count=dup,
                         spans=[(_fmt_t(dup_times.iloc[0]), _fmt_t(dup_times.iloc[-1]))],
                         suggestion="Drop exact duplicates; if values differ, keep the last write per timestamp."))
    pos = ds[ds > 0]
    if not len(pos):
        return out
    med = float(np.median(pos))
    summary.update({"start": _fmt_t(tsorted.iloc[0]), "end": _fmt_t(tsorted.iloc[-1]),
                    "duration_s": float((tsorted.iloc[-1] - tsorted.iloc[0]).total_seconds()),
                    "median_dt_s": med, "sampling": _fmt_dur(med)})
    gap_idx = np.where(ds > 3 * med)[0]
    if len(gap_idx):
        gap_total = float(np.sum(ds[gap_idx] - med))
        span = max(1.0, summary["duration_s"])
        big = gap_idx[np.argsort(-ds[gap_idx])][:10]
        frac = gap_total / span
        out.append(Issue("time_axis", "critical" if frac > 0.05 else "warning" if frac > 0.005 else "info",
                         f"{len(gap_idx)} gap{'s' if len(gap_idx) > 1 else ''} in the recording",
                         f"Missing time totals {_fmt_dur(gap_total)} ({frac * 100:.1f} % of the period); the longest is "
                         f"{_fmt_dur(float(ds[big[0]]))}. Sampling is every {_fmt_dur(med)} otherwise.",
                         affected_fraction=frac, count=len(gap_idx),
                         spans=[(_fmt_t(tsorted.iloc[i]), _fmt_t(tsorted.iloc[i + 1])) for i in big],
                         suggestion="Interpolate only short gaps (a few samples); mark long gaps as missing rather "
                                    "than filling them.", evidence={"gap_seconds": [float(ds[i]) for i in big]}))
    regular = pos[pos <= 3 * med]
    irregular = float(np.mean(np.abs(regular - med) > 0.02 * med)) if len(regular) else 0.0
    summary["irregular_pct"] = irregular * 100
    if irregular > 0.1:
        out.append(Issue("time_axis", "warning" if irregular > 0.3 else "info", "Irregular sampling",
                         f"{irregular * 100:.0f} % of the intervals differ from the nominal {_fmt_dur(med)} by more "
                         "than 2 %. Event-based logging or jitter; most models expect a fixed step.",
                         affected_fraction=irregular,
                         suggestion=f"Resample to a fixed {_fmt_dur(med)} grid (mean for analog signals, last for states)."))
    return out


_COMMON_ALT = {
    "temperature": ["°C", "K", "°F"], "pressure": ["bar", "kPa", "Pa", "psi", "MPa", "mbar"],
    "volumetric_flow_rate": ["m^3/h", "L/s", "gpm", "m^3/s", "L/min"], "mass_flow_rate": ["kg/s", "kg/h", "t/h", "lb/h"],
    "power": ["W", "kW", "MW", "BTU/h", "TR"], "length": ["m", "mm", "cm", "in", "ft"], "velocity": ["m/s", "km/h", "ft/s"],
}


def _check_column(c: str, v: np.ndarray, unit: Optional[Unit], utxt: Optional[str], quantity: Optional[str],
                  is_diff: bool, t: Optional[pd.Series]) -> List[Issue]:
    out: List[Issue] = []
    n = len(v)
    idx = np.arange(n)
    isnan = ~np.isfinite(v)
    miss = float(isnan.mean())
    if miss > 0:
        runs = [(a, b) for a, b in _runs(isnan) if b - a >= max(5, int(0.002 * n))]
        sev = "critical" if miss > 0.2 else "warning" if miss > 0.02 else "info"
        out.append(Issue("completeness", sev, "Missing values", f"{miss * 100:.1f} % of '{c}' is empty"
                         + (f", in {len(runs)} run{'s' if len(runs) > 1 else ''} of 5+ samples" if runs else "") + ".",
                         column=c, affected_fraction=miss, count=int(isnan.sum()),
                         spans=[_span(t, idx, a, b) for a, b in sorted(runs, key=lambda r: r[0] - r[1])[:8]],
                         suggestion="Fill only short gaps; keep long ones missing."))
    fin = v[~isnan]
    if len(fin) < 10:
        return out

    # sentinels / error codes
    for s in SENTINELS:
        k = int(np.sum(fin == s))
        if k >= 3 and k < 0.5 * len(fin):
            rr = _runs(v == s)
            out.append(Issue("validity", "critical", f"Error code {s:g} in the data",
                             f"'{c}' contains the value {s:g} {k} times, a typical logger error / no-data code, not "
                             "a measurement. It distorts averages and every model trained on it.",
                             column=c, affected_fraction=k / n, count=k,
                             spans=[_span(t, idx, a, b) for a, b in rr[:8]],
                             suggestion=f"Replace {s:g} with missing (NaN) before any analysis.", evidence={"value": s}))
            v = np.where(v == s, np.nan, v)
    fin = v[np.isfinite(v)]
    if len(fin) < 10:
        return out

    # physically impossible values / unit checks
    if unit is not None and quantity in QUANTITIES:
        q = QUANTITIES[quantity]
        si = unit.to_si(fin, difference=is_diff)
        lo, hi = q.hard
        if quantity == "pressure":
            lo = -101325.0 if (unit.gauge or not unit.offset) else 0.0
        if is_diff:
            lo, hi = None, None
        bad = np.zeros(len(fin), bool)
        if lo is not None:
            bad |= si < lo - 1e-9 * abs(lo or 1)
        if hi is not None:
            bad |= si > hi + 1e-9 * abs(hi or 1)
        if quantity == "relative_humidity" and unit.factor == 1.0 and fin.max() > 1.5:   # RH given as 0-100 without %
            bad = (fin < 0) | (fin > 100)
        if bad.any():
            k = int(bad.sum())
            limit = {"temperature": "absolute zero (−273.15 °C)", "relative_humidity": "0–100 %",
                     "pressure": "full vacuum"}.get(quantity, "the physical limit")
            vv = v.copy()
            mask_full = np.zeros(n, bool)
            mask_full[np.where(np.isfinite(vv))[0][bad]] = True
            out.append(Issue("validity", "critical", "Physically impossible values",
                             f"{k} value{'s' if k > 1 else ''} of '{c}' lie beyond {limit} "
                             f"(range {fin[bad].min():.4g} to {fin[bad].max():.4g} {utxt}). A sensor fault, a wrong unit or "
                             "a scaling error.", column=c, affected_fraction=k / n, count=k,
                             spans=[_span(t, idx, a, b) for a, b in _runs(mask_full)[:8]],
                             suggestion="Check the sensor's range and scaling; treat these values as missing."))
        # unit that does not fit the values (whole column), e.g. kelvin labelled °C
        med = float(np.median(fin))
        if quantity == "temperature" and unit.offset == 273.15 and not is_diff and 240 < med < 400 \
                and np.percentile(fin, 1) > 200:
            out.append(Issue("validity", "critical", "Unit looks wrong: kelvin labelled °C",
                             f"'{c}' is labelled {utxt} but its median is {med:.1f}: as °C that is {med:.0f} °C, as kelvin "
                             f"{med - 273.15:.1f} °C. The values are almost certainly kelvin.",
                             column=c, affected_fraction=1.0, suggestion="Relabel the column as K (or subtract 273.15).",
                             evidence={"median": med}))
        _unit_switch(out, c, v, unit, utxt, quantity, t)

    # flat lines (stuck sensor), clipping, spikes, resolution
    uniq = np.unique(fin)
    if len(uniq) == 1:
        out.append(Issue("signal_quality", "warning", "Constant column",
                         f"'{c}' never changes (always {uniq[0]:g}). A configuration value, a disconnected sensor or "
                         "an unused channel.", column=c, affected_fraction=1.0,
                         suggestion="Drop it from models, or check the sensor."))
        return out
    if len(uniq) <= 6:
        return out                                   # state / binary signal: no analog checks
    dv = np.diff(v)
    same = np.concatenate([[False], np.isclose(dv, 0.0, rtol=0, atol=0)]) & np.isfinite(v)
    rng = float(np.percentile(fin, 99) - np.percentile(fin, 1)) or float(np.ptp(fin)) or 1.0
    noisy = float(np.mean(np.abs(dv[np.isfinite(dv)]) > 0)) > 0.5      # normally changes most samples
    if noisy:
        min_len = max(30, int(0.01 * n))
        stuck = []
        for a, b in _runs(same):
            a0 = a - 1
            if b - a0 >= min_len:
                stuck.append((a0, b))
        if stuck:
            k = sum(b - a for a, b in stuck)
            vals = sorted({round(float(v[a]), 6) for a, b in stuck})
            out.append(Issue("signal_quality", "warning" if k / n > 0.01 else "info", "Stuck sensor (flat line)",
                             f"'{c}' repeats exactly the same value for {len(stuck)} stretch"
                             f"{'es' if len(stuck) > 1 else ''} ({k} samples, longest {max(b - a for a, b in stuck)}) "
                             f"while it normally changes every sample. Frozen value{'s' if len(vals) > 1 else ''}: "
                             + ", ".join(f"{x:g}" for x in vals[:5]) + ". A frozen transmitter or a held last value "
                             "after a communication loss.", column=c, affected_fraction=k / n, count=k,
                             spans=[_span(t, idx, a, b) for a, b in stuck[:8]],
                             suggestion="Treat the stretch as missing; check the transmitter / gateway.",
                             evidence={"values": vals[:5]}))
    for side, val in (("upper", fin.max()), ("lower", fin.min())):
        at = np.isclose(v, val, rtol=0, atol=1e-12 * max(1.0, abs(val)))
        k = int(at.sum())
        runs = [r for r in _runs(at) if r[1] - r[0] >= 3]
        if k >= max(10, 0.01 * len(fin)) and runs and len(uniq) > 20:
            out.append(Issue("signal_quality", "info" if k / n < 0.05 else "warning", f"Clipped at the {side} limit",
                             f"{k} samples of '{c}' sit exactly at {val:g}, the {side} end of the recorded range, in "
                             f"{len(runs)} flat stretches. The true value was beyond the sensor / logger range or an "
                             "actuator was saturated.", column=c, affected_fraction=k / n, count=k,
                             spans=[_span(t, idx, a, b) for a, b in runs[:6]],
                             suggestion="If this is a sensor range limit, the real values are unknown there: flag them "
                                        "as censored rather than using them as measurements."))
    s = pd.Series(v)
    w = 11
    med = s.rolling(w, center=True, min_periods=5).median()
    dev = (s - med).abs()
    mad_global = float(np.nanmedian(dev)) * 1.4826
    mad = (dev.rolling(w * 5, center=True, min_periods=10).median() * 1.4826).clip(lower=0.5 * mad_global)
    spike = (dev > np.maximum(8 * mad, 0.02 * rng)) & (dev > 8 * mad_global) & (dev > 0.15 * rng)
    spike = spike.fillna(False).to_numpy()
    if spike.any():
        k = int(spike.sum())
        runs = _runs(spike)
        if k / n < 0.05:
            out.append(Issue("signal_quality", "warning" if k >= 5 else "info", "Spikes",
                             f"{k} isolated sample{'s' if k > 1 else ''} of '{c}' jump far from their neighbours "
                             f"(more than 8 robust standard deviations and 15 % of the signal range) and come straight "
                             "back. Electrical interference or transmission errors, not process changes.",
                             column=c, affected_fraction=k / n, count=k,
                             spans=[_span(t, idx, a, b) for a, b in runs[:8]],
                             suggestion="Remove with a median (Hampel) filter before computing derivatives or "
                                        "training models.", evidence={"rows": [int(a) for a, _ in runs[:50]]}))
    return out


def _unit_switch(out: List[Issue], c: str, v: np.ndarray, unit: Unit, utxt: Optional[str], quantity: str,
                 t: Optional[pd.Series]) -> None:
    """A stretch whose level equals the rest converted to another unit of the same quantity."""
    alts = [u for u in _COMMON_ALT.get(quantity, []) if try_parse_unit(u) and try_parse_unit(u).dim == unit.dim]
    if not alts:
        return
    s = pd.Series(v)
    n = len(v)
    win = max(15, n // 200)
    s.rolling(win, center=True, min_periods=win // 2).median().to_numpy()
    d = np.abs(np.diff(v))
    fin = d[np.isfinite(d)]
    if not len(fin):
        return
    typical = np.percentile(fin, 99) + 1e-12
    jumps = [i + 1 for i in np.where(d > 20 * typical)[0]]
    if not jumps:
        return
    seg_bounds = sorted(set([0] + jumps + [n]))
    segs = [(a, b) for a, b in zip(seg_bounds[:-1], seg_bounds[1:]) if b - a >= win]
    if len(segs) < 2:
        return
    lens = [b - a for a, b in segs]
    main = int(np.argmax(lens))
    ref = float(np.nanmedian(v[segs[main][0]:segs[main][1]]))
    for j, (a, b) in enumerate(segs):
        if j == main:
            continue
        m = float(np.nanmedian(v[a:b]))
        for alt in alts:
            if alt == utxt:
                continue
            try:
                want = float(convert(ref, utxt or unit.symbol, alt))
            except Exception:
                continue
            if abs(want - m) <= 0.03 * max(abs(want), abs(ref - want) * 0.1, 1e-9):
                out.append(Issue("validity", "critical", "Unit changes mid-recording",
                                 f"'{c}' is labelled {utxt}, but from {_span(t, np.arange(n), a, b)[0]} to "
                                 f"{_span(t, np.arange(n), a, b)[1]} its level ({m:.4g}) equals the rest of the record "
                                 f"({ref:.4g} {utxt}) expressed in {alt}. The logger or a firmware update changed "
                                 "units; averages and models across the switch are wrong.",
                                 column=c, affected_fraction=(b - a) / n, count=b - a,
                                 spans=[_span(t, np.arange(n), a, b)],
                                 suggestion=f"Convert that stretch from {alt} back to {utxt}.",
                                 evidence={"segment_unit": alt, "rows": [int(a), int(b)]}))
                return


def _check_redundancy(numeric: Dict[str, pd.Series]) -> List[Issue]:
    out: List[Issue] = []
    names = [c for c, s in numeric.items() if s.notna().sum() > 10 and s.nunique() > 6]
    seen = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if b in seen:
                continue
            sa, sb = numeric[a], numeric[b]
            both = sa.notna() & sb.notna()
            if both.sum() < 10:
                continue
            if np.allclose(sa[both], sb[both], rtol=0, atol=0):
                seen.add(b)
                out.append(Issue("metadata", "warning", "Duplicate column",
                                 f"'{b}' is identical to '{a}'. A copied tag or a channel mapped twice.",
                                 column=b, affected_fraction=1.0, suggestion=f"Drop '{b}' or fix the tag mapping."))
    return out


# ── physics consistency: thermal energy balance ─────────────────────────────
_SUP = re.compile(r"(supply|sup|inlet|in|ent|entering|ewt|flow_t|chws|hws|cws)$|^(supply|sup|inlet|in|ewt)|_(supply|sup|inlet|in|s)(_|$)", re.I)
_RET = re.compile(r"(return|ret|outlet|out|lv|leaving|lwt|chwr|hwr|cwr)$|^(return|ret|outlet|out|lwt)|_(return|ret|outlet|out|r)(_|$)", re.I)
_WATER = {"rho": 997.0, "cp": 4186.0}


def _prefix(name: str) -> str:
    toks = [t for t in re.split(r"[_\s\-]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()) if t]
    drop = {"supply", "sup", "return", "ret", "inlet", "outlet", "in", "out", "temp", "temperature", "t", "flow",
            "rate", "s", "r", "entering", "leaving", "ewt", "lwt", "load", "power", "heat", "cooling", "q", "duty"}
    return "_".join(t for t in toks if t not in drop)


def _check_energy_balance(columns: List[Dict[str, Any]], numeric: Dict[str, pd.Series],
                          t: Optional[pd.Series]) -> Tuple[List[Dict[str, Any]], List[Issue]]:
    cols = {c["name"]: c for c in columns if c.get("kind") == "numeric" and c.get("unit_known")}
    temps = [c for c in cols.values() if c["quantity"] == "temperature" and not c["is_difference"]]
    flows = [c for c in cols.values() if c["quantity"] in ("volumetric_flow_rate", "mass_flow_rate")]
    powers = [c for c in cols.values() if c["quantity"] == "power"]
    relations: List[Dict[str, Any]] = []
    issues: List[Issue] = []
    for s in temps:
        if not _SUP.search(s["base_name"]):
            continue
        for r in temps:
            if r is s or not _RET.search(r["base_name"]) or _prefix(r["base_name"]) != _prefix(s["base_name"]):
                continue
            pre = _prefix(s["base_name"])
            flow = next((f for f in flows if _prefix(f["base_name"]) == pre), None)
            pw = [p for p in powers if _prefix(p["base_name"]) in (pre, "")
                  and re.search(r"load|heat|cool|duty|q|capacity|thermal", p["base_name"], re.I)]
            if not flow or not pw:
                continue
            p = pw[0]
            uS, uR = try_parse_unit(s["unit"]), try_parse_unit(r["unit"])
            uF, uP = try_parse_unit(flow["unit"]), try_parse_unit(p["unit"])
            Ts, Tr = uS.to_si(numeric[s["name"]].to_numpy()), uR.to_si(numeric[r["name"]].to_numpy())
            F = uF.to_si(numeric[flow["name"]].to_numpy())
            mdot = F * _WATER["rho"] if flow["quantity"] == "volumetric_flow_rate" else F
            q_calc = mdot * _WATER["cp"] * np.abs(Tr - Ts)
            q_rep = np.abs(uP.to_si(numeric[p["name"]].to_numpy()))
            ok = np.isfinite(q_calc) & np.isfinite(q_rep) & (q_rep > 0.1 * np.nanmedian(q_rep))
            if ok.sum() < 20:
                continue
            ratio = np.full(len(q_rep), np.nan)
            ratio[ok] = q_calc[ok] / q_rep[ok]
            rs = pd.Series(ratio).rolling(max(5, len(ratio) // 300), center=True, min_periods=3).median().to_numpy()
            bad = np.isfinite(rs) & (np.abs(rs - 1) > 0.15)
            med = float(np.nanmedian(ratio))
            share = float(bad.sum() / max(1, np.isfinite(rs).sum()))
            rel = {"type": "thermal energy balance", "equation": "Q = rho · V · cp · |T_return − T_supply|",
                   "fluid": "water (rho 997 kg/m³, cp 4186 J/(kg·K))",
                   "columns": {"supply": s["name"], "return": r["name"], "flow": flow["name"], "heat_rate": p["name"]},
                   "median_ratio": med, "violating_pct": share * 100,
                   "ratio_series": _downsample_line(ratio, t, 400)}
            relations.append(rel)
            if share > 0.02 or abs(med - 1) > 0.15:
                runs = [rr for rr in _runs(bad) if rr[1] - rr[0] >= 10]
                issues.append(Issue("physics_consistency", "critical" if share > 0.1 else "warning",
                                    "Energy balance does not close",
                                    f"'{p['name']}' should equal ρ·V·cp·ΔT from '{flow['name']}', '{s['name']}' and "
                                    f"'{r['name']}'. It does on {100 - share * 100:.0f} % of the record, but on "
                                    f"{share * 100:.0f} % the computed heat rate is off by more than 15 % "
                                    f"(typical ratio there {np.nanmedian(rs[bad]) if bad.any() else med:.2f}). "
                                    "Each column looks plausible on its own; together they cannot all be right. "
                                    "Most often a drifting flow meter or a temperature sensor out of its well.",
                                    column=flow["name"], affected_fraction=share, count=int(bad.sum()),
                                    spans=[_span(t, np.arange(len(ratio)), a, b) for a, b in runs[:6]],
                                    suggestion="Check the flow meter first (zero, scaling, fouling), then the two "
                                               "temperature sensors against a reference.",
                                    evidence={"median_ratio": med}))
    return relations, issues


# ── scoring, downsampled views ──────────────────────────────────────────────
def _score(issues: List[Issue], applicable_physics: bool) -> Dict[str, Any]:
    cats = {}
    for cat, w in CATEGORY_WEIGHT.items():
        ci = [i for i in issues if i.category == cat]
        applicable = cat != "physics_consistency" or applicable_physics
        pen = sum(SEVERITY_WEIGHT[i.severity] * (0.5 + 0.5 * min(1.0, i.affected_fraction * 10)) for i in ci)
        cats[cat] = {"label": CATEGORY_LABEL[cat], "score": max(0.0, 100.0 - pen) if applicable else None,
                     "weight": w, "issues": len(ci), "applicable": applicable,
                     "critical": sum(1 for i in ci if i.severity == "critical")}
    app = [c for c in cats.values() if c["applicable"]]
    overall = sum(c["score"] * c["weight"] for c in app) / sum(c["weight"] for c in app)
    crit = sum(1 for i in issues if i.severity == "critical")
    if overall >= 85 and crit == 0:
        verdict, level = "Ready to use: minor issues only.", "ok"
    elif overall >= 60:
        verdict, level = "Usable after cleaning: fix the critical issues before modelling.", "warn"
    else:
        verdict, level = "Not fit for use yet: the issues below would corrupt analysis or training.", "bad"
    return {"overall": overall, "verdict": verdict, "level": level, "categories": cats,
            "method": "Each issue costs critical 30 / warning 10 / info 2 points, scaled from 50 % to 100 % by the "
                      "share of the data it affects (100 % at 10 % or more). A category scores 100 minus its "
                      "penalties (floor 0); the overall score is the weighted mean of the categories that apply "
                      "(physics consistency only when a checkable relation exists)."}


def _downsample_line(v: np.ndarray, t: Optional[pd.Series], k: int) -> Dict[str, Any]:
    n = len(v)
    edges = bucket_edges(n, k)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = v[a:b]
        seg = seg[np.isfinite(seg)]
        xs.append(_fmt_t(t.iloc[a]) if t is not None else int(a))
        ys.append(float(np.median(seg)) if len(seg) else None)
    return {"x": xs, "y": ys}


def _issue_masks(issues: List[Issue], numeric: Dict[str, pd.Series], t: Optional[pd.Series]) -> Dict[str, np.ndarray]:
    """Per column, the rows touched by its issues (from spans)."""
    masks: Dict[str, np.ndarray] = {}
    if t is None:
        tv = None
    else:
        tv = t.to_numpy()
    for i in issues:
        if not i.column or i.column not in numeric or i.severity == "info" and i.category != "signal_quality":
            continue
        m = masks.setdefault(i.column, np.zeros(len(numeric[i.column]), bool))
        for a, b in i.spans:
            if tv is not None:
                try:
                    ta, tb = np.datetime64(pd.Timestamp(a)), np.datetime64(pd.Timestamp(b))
                except (ValueError, TypeError):
                    continue
                m |= (tv >= ta) & (tv <= tb + np.timedelta64(59, "s"))
            else:
                try:
                    ra, rb = int(str(a).split()[-1]) - 2, int(str(b).split()[-1]) - 2
                    m[ra:rb + 1] = True
                except ValueError:
                    continue
        if i.category == "validity" and i.affected_fraction >= 0.999:
            m[:] = True
        rows = i.evidence.get("rows") if isinstance(i.evidence, dict) else None
        if rows and i.title == "Spikes":
            for r in rows:
                m[max(0, r - 1):r + 2] = True
    return masks


def _availability(numeric: Dict[str, pd.Series], t: Optional[pd.Series], issues: List[Issue], n_bins: int) -> Dict[str, Any]:
    if not numeric:
        return {"bins": [], "columns": {}}
    n = len(next(iter(numeric.values())))
    masks = _issue_masks(issues, numeric, t)
    if t is not None and t.notna().sum() > 2:
        tt = t.to_numpy().astype("datetime64[ns]").astype("int64")
        edges = np.linspace(tt.min(), tt.max(), n_bins + 1)
        b = np.clip(np.searchsorted(edges, tt, side="right") - 1, 0, n_bins - 1)
        labels = [_fmt_t(pd.Timestamp(int(e))) for e in edges]
        count = np.bincount(b, minlength=n_bins)
        expected = np.full(n_bins, n / n_bins)
    else:
        b = np.minimum((np.arange(n) * n_bins) // max(1, n), n_bins - 1)
        labels = [f"row {int(i * n / n_bins) + 2}" for i in range(n_bins + 1)]
        count = np.bincount(b, minlength=n_bins)
        expected = count.astype(float)
    out = {}
    for c, s in numeric.items():
        fin = np.isfinite(s.to_numpy())
        present = np.bincount(b, weights=fin.astype(float), minlength=n_bins)
        flagged = np.bincount(b, weights=masks.get(c, np.zeros(n, bool)).astype(float), minlength=n_bins)
        cells = []
        for k in range(n_bins):
            avail = present[k] / max(expected[k], 1e-9) if expected[k] else 0.0
            cells.append([round(float(min(1.0, avail)), 3), round(float(flagged[k] / max(count[k], 1)), 3)])
        out[c] = cells
    return {"bins": labels, "columns": out, "timeline_gaps": bool(t is not None)}


def _series(numeric: Dict[str, pd.Series], t: Optional[pd.Series], issues: List[Issue], k: int) -> Dict[str, Any]:
    masks = _issue_masks(issues, numeric, t)
    out = {}
    for c, s in numeric.items():
        v = s.to_numpy()
        n = len(v)
        edges = bucket_edges(n, k)
        lo, hi, fl, xs = [], [], [], []
        m = masks.get(c, np.zeros(n, bool))
        for a, b in zip(edges[:-1], edges[1:]):
            seg = v[a:b]
            f = seg[np.isfinite(seg)]
            lo.append(float(f.min()) if len(f) else None)
            hi.append(float(f.max()) if len(f) else None)
            fl.append(bool(m[a:b].any()))
            xs.append(_fmt_t(t.iloc[a]) if t is not None else int(a + 2))
        out[c] = {"x": xs, "lo": lo, "hi": hi, "flag": fl}
    return out


def _clean(o: Any) -> Any:
    return to_jsonable(o)


# ── example dataset ─────────────────────────────────────────────────────────
def example_chiller_plant(seed: int = 7, days: int = 7, inject: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """One week of 1-minute data from a chilled-water plant, with eight realistic faults injected.

    Returns the table and the list of injected faults (the answer key used by the tests)."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    t = pd.date_range("2026-03-02 00:00", periods=n, freq="1min")
    h = (t.hour + t.minute / 60).to_numpy()
    day = np.arange(n) / 1440
    oat = 27 + 5 * np.sin((h - 9) / 24 * 2 * np.pi) + 1.5 * np.sin(day * 0.9) + rng.normal(0, 0.15, n)
    rh = np.clip(70 - 2.2 * (oat - 27) + rng.normal(0, 2, n), 25, 99)
    occ = ((h > 7) & (h < 20) & (t.dayofweek < 5)).astype(float)
    load = 380 + 260 * occ + 18 * (oat - 27) + rng.normal(0, 8, n)                      # kW
    load = np.clip(load, 120, None)
    flow_true = 120 + 10 * occ + rng.normal(0, 1.2, n)                                  # m3/h
    chws = 6.7 + rng.normal(0, 0.08, n)
    dT = load * 1e3 / (flow_true / 3600 * 997 * 4186)
    chwr = chws + dT + rng.normal(0, 0.05, n)
    cop = 5.6 - 0.06 * (oat - 27) + rng.normal(0, 0.05, n)
    power = load / cop
    condp = 8.6 + 0.12 * (oat - 27) + 0.002 * (load - 400) + rng.normal(0, 0.05, n)
    pump = np.clip(78 + 12 * occ + rng.normal(0, 1.2, n) + 3 * np.sin(day * 3), 0, 100)
    valve = np.clip(55 + 25 * occ + rng.normal(0, 3, n), 0, 100)
    stage = np.where(load > 560, 2, 1)

    faults: List[Dict[str, Any]] = []
    if not inject:
        df = pd.DataFrame({
            "Timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "OAT [°C]": oat.round(2), "RH_outdoor [%]": rh.round(1),
            "CHW_supply_temp [°C]": chws.round(2), "CHW_return_temp [°C]": chwr.round(2),
            "CHW_flow [m3/h]": flow_true.round(1), "Cooling_load [kW]": load.round(1),
            "Chiller_power [kW]": power.round(1), "Cond_pressure [bar]": condp.round(2), "Pump_speed [%]": pump.round(1),
            "Valve_position [%]": valve.round(1), "Chillers_running": stage})
        return df, {"faults": [], "description": "Chilled-water plant, 1-min data, 7 days, no faults"}
    # 1. flow meter under-reads by 28 % from day 4 noon (only the energy balance shows it)
    flow = flow_true.copy()
    a = int(4.5 * 1440)
    flow[a:] *= 0.72
    faults.append({"type": "energy_balance", "column": "CHW_flow [m3/h]", "rows": [a, n]})
    # 2. CHW supply sensor frozen for 6 h on day 2
    a = 2 * 1440 + 600
    chws[a:a + 360] = round(chws[a], 2)
    faults.append({"type": "stuck", "column": "CHW_supply_temp [°C]", "rows": [a, a + 360]})
    # 3. outdoor temperature logged in kelvin for 10 h on day 5 (firmware update)
    a = 5 * 1440 + 300
    oat[a:a + 600] += 273.15
    faults.append({"type": "unit_switch", "column": "OAT [°C]", "rows": [a, a + 600]})
    # 4. humidity sensor over-range (wet) on day 1 night
    a = 1440 + 120
    rh[a:a + 45] = 100 + rng.uniform(1, 6, 45)
    faults.append({"type": "impossible", "column": "RH_outdoor [%]", "rows": [a, a + 45]})
    # 5. spikes on condenser pressure (electrical interference)
    sp = rng.choice(np.arange(200, n - 200), 14, replace=False)
    condp[sp] += rng.choice([-1, 1], 14) * rng.uniform(9, 16, 14)
    faults.append({"type": "spikes", "column": "Cond_pressure [bar]", "rows": sorted(int(x) for x in sp)})
    # 6. valve position: -999 while the BACnet device was offline
    for a in rng.choice(np.arange(500, n - 500), 5, replace=False):
        valve[a:a + 20] = -999
    faults.append({"type": "sentinel", "column": "Valve_position [%]"})
    df = pd.DataFrame({
        "Timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "OAT [°C]": oat.round(2), "RH_outdoor [%]": rh.round(1),
        "CHW_supply_temp [°C]": chws.round(2), "CHW_return_temp [°C]": chwr.round(2), "CHW_flow [m3/h]": flow.round(1),
        "Cooling_load [kW]": load.round(1), "Chiller_power [kW]": power.round(1), "Cond_pressure [bar]": condp.round(2),
        "Pump_speed [%]": pump.round(1), "Valve_position [%]": valve.round(1), "Chillers_running": stage})
    # 7. logger offline for 3 h on day 3: rows missing
    a = 3 * 1440 + 840
    df = df.drop(index=range(a, a + 180)).reset_index(drop=True)
    faults.append({"type": "gap", "rows": [a, a + 180]})
    # 8. 30 rows re-sent by the gateway (duplicate timestamps)
    a = 6 * 1440
    df = pd.concat([df.iloc[:a], df.iloc[a - 30:a], df.iloc[a:]]).reset_index(drop=True)
    faults.append({"type": "duplicates", "rows": 30})
    return df, {"faults": faults, "description": "Chilled-water plant, 1-min data, 7 days, 8 injected faults"}


__all__ = ["Issue", "LoadInfo", "load_table", "analyze", "detect_time_column", "example_chiller_plant"]
