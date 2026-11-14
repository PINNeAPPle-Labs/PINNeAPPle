"""The levitation-diffusion-chamber data set of Lamb et al. (2026): download, parse, and the 1 Hz mass-ratio series.

The paper fits 290 mass-ratio time series of individual ice crystals grown in a levitation diffusion chamber at
205-240 K and 1.0 < S_i < 1.8. Its data statement points at three archives in the Penn State DataCommons:

===================================  ====================================================  ==========
archive                                DOI                                                   series
===================================  ====================================================  ==========
Harrison et al. (2016), JAS 73           10.26208/dd1w-wa17                                   (not used)
Pokrifka et al. (2018), homogeneous      10.26208/z7bf-nq20                                   18
Pokrifka et al. (2023), low-T, -65..-40 C 10.26208/htw5-q166                                     268
===================================  ====================================================  ==========

Two of them are used here, 18 + 268 = 286 series. The remaining 4 of the paper's 290 are the high-supersaturation
"cold" experiments (T = -63.2, -58.2, -54.0, -60.6 C; S_i = 1.211, 1.397, 1.573, 1.740), which the paper
reprocessed from the raw lower-plate voltage files of its own laboratory directory; those raw files are not in
the archives. The same crystals are present in the low-T archive as processed mass ratios (conditions rows 188,
227, 115 and 72 match the four (T, S_i, r0) triples), so this loader takes all 286 for training and, when the raw
voltage files are supplied with ``cold_dir``, adds the paper's 4 so the count reaches 290.

Downloading needs the network once; afterwards the parsed arrays are cached as ``.npz`` next to the archives. Set
``PINNEAPPLE_ICENODE_DATA`` to move the cache (default ``~/.cache/pinneapple/icenode``).
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ARCHIVES", "ChamberData", "cache_dir", "download", "load", "cold_experiments"]

_BASE = "https://www.datacommons.psu.edu/download/meteorology"

ARCHIVES = {
    "pokrifka2018": {
        "url": f"{_BASE}/harrington/pokrifka-et-al-electrodynamic-levitation-diffusion-homogeneously-nucleated-"
               "ice-crystals-2018/pokrifka-et-al-electrodynamic-levitation-diffusion-homogeneously-nucleated-"
               "ice-crystals-2018.zip",
        "doi": "10.26208/z7bf-nq20",
        "zip": "pokrifka2018.zip",
    },
    "pokrifka2023": {
        "url": f"{_BASE}/pokrifka-et-al-electrodynamic-levitation-diffusion-chamber-measurements-of-the-mass-growth-"
               "of-ice-crystals-grown-from-the-vapor-at-65-to-40c/pokrifka-et-al-electrodynamic-levitation-"
               "diffusion-chamber-measurements-of-the-mass-growth-of-ice-crystals-grown-from-the-vapor-at-65-to-"
               "40c.zip",
        "doi": "10.26208/htw5-q166",
        "zip": "pokrifka2023.zip",
    },
}

RHO_ICE = 910.0          # kg/m3, Table S1 of Lamb et al. (2026)
T_MAX = 1000.0           # s; the paper crops the training window at 500 s and tests extrapolation up to 1000 s
T_TRAIN = 500.0


@dataclass(frozen=True)
class ChamberData:
    """The parsed experiments, all on a 1 Hz grid.

    Attributes
    ----------
    mass_ratio : (n_exp, n_time) ndarray
        ``m/m0`` interpolated to 1 Hz; ``nan`` past the end of each experiment.
    time : (n_exp, n_time) ndarray
        Seconds, ``nan`` past the end of each experiment.
    Si, T, P, r0, m0 : (n_exp,) ndarray
        Ice supersaturation (dimensionless), temperature (K), pressure (Pa), initial spherical-equivalent radius
        (m) and initial mass (kg) of each experiment.
    nucleation : (n_exp,) ndarray of int
        1 homogeneous, 0 heterogeneous (the ``INP`` column of the low-T archive).
    source : (n_exp,) ndarray of str
        Which archive each series came from, for the record.
    """

    mass_ratio: np.ndarray
    time: np.ndarray
    Si: np.ndarray
    T: np.ndarray
    P: np.ndarray
    r0: np.ndarray
    m0: np.ndarray
    nucleation: np.ndarray
    source: np.ndarray

    def __len__(self) -> int:
        return int(self.Si.size)

    def length(self) -> np.ndarray:
        """Number of valid 1 Hz samples of each experiment."""
        return np.sum(np.isfinite(self.mass_ratio), axis=1)

    def up_to(self, t_max: float = T_TRAIN) -> ChamberData:
        """The same experiments restricted to ``t <= t_max`` (``nan`` past it, as the paper masks its loss)."""
        keep = self.time <= t_max
        return ChamberData(np.where(keep, self.mass_ratio, np.nan), np.where(keep, self.time, np.nan),
                           self.Si, self.T, self.P, self.r0, self.m0, self.nucleation, self.source)

    def subset(self, mask) -> ChamberData:
        return ChamberData(self.mass_ratio[mask], self.time[mask], self.Si[mask], self.T[mask], self.P[mask],
                           self.r0[mask], self.m0[mask], self.nucleation[mask], self.source[mask])

    def conditions(self) -> dict:
        """The per-experiment conditions as plain arrays, for a results file."""
        return {"n": len(self), "T_K": self.T.round(3).tolist(), "Si": self.Si.round(4).tolist(),
                "r0_m": self.r0.tolist(), "nucleation": self.nucleation.tolist(),
                "seconds": self.length().tolist(), "source": self.source.tolist()}


def cache_dir() -> Path:
    """Where the archives and the parsed arrays live."""
    return Path(os.environ.get("PINNEAPPLE_ICENODE_DATA", Path.home() / ".cache" / "pinneapple" / "icenode"))


def download(root: str | Path | None = None, *, only: str | None = None) -> Path:
    """Fetch and unzip the archives under :func:`cache_dir`, skipping those already there."""
    root = Path(root) if root is not None else cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in ARCHIVES.items():
        if only is not None and name != only:
            continue
        zip_path, out = root / spec["zip"], root / name
        if out.is_dir() and any(out.rglob("*.csv")):
            continue
        if not zip_path.exists():
            with urllib.request.urlopen(spec["url"], timeout=120) as r:      # noqa: S310 - fixed https URL
                zip_path.write_bytes(r.read())
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    return root


def _interp_1hz(time: np.ndarray, ratio: np.ndarray, t_max: float = T_MAX) -> tuple[np.ndarray, np.ndarray]:
    """The paper's preprocessing: interpolate a series onto ``linspace(0, floor(t_max), floor(t_max) + 1)``.

    The archive holds experiments that ran for hours (the longest, series 106, for 73,429 s); the paper uses at
    most the first 1000 s of each, which is also where its extrapolation test stops, so nothing longer is kept.
    """
    ok = np.isfinite(time) & np.isfinite(ratio)
    time, ratio = time[ok], ratio[ok]
    order = np.argsort(time, kind="stable")
    time, ratio = time[order], ratio[order]
    if time.size < 2:                                    # pragma: no cover - the archives hold no such series
        raise ValueError("series has fewer than two finite samples")
    t_int = np.arange(0.0, min(np.floor(time.max()), t_max) + 1.0, 1.0)
    return t_int, np.interp(t_int, time, ratio)


def _read_pokrifka2018(root: Path):
    """18 homogeneous-nucleation series, ``NN-mass.dat`` (whitespace, no header) + ``conditions.dat``."""
    sub = next(root.rglob("conditions.dat")).parent
    cond = np.loadtxt(sub / "conditions.dat", delimiter=",", ndmin=2)
    ratio, time = [], []
    for i in range(cond.shape[0]):
        dat = np.loadtxt(sub / f"{i + 1:02d}-mass.dat", ndmin=2)
        t, r = _interp_1hz(dat[:, 0], dat[:, 2])
        time.append(t)
        ratio.append(r)
    n = max(len(t) for t in time)
    out_r = np.full((len(time), n), np.nan)
    out_t = np.full((len(time), n), np.nan)
    for i, (t, r) in enumerate(zip(time, ratio, strict=True)):
        out_r[i, :len(r)], out_t[i, :len(t)] = r, t
    return {"mass_ratio": out_r, "time": out_t, "r0": cond[:, 1] * 1e-6, "T": cond[:, 3] + 273.15,
            "Si": cond[:, 4] / 100.0 + 1.0, "nucleation": np.ones(len(time), dtype=int),
            "source": np.array(["pokrifka2018"] * len(time))}


def _read_pokrifka2023(root: Path):
    """268 series, ``N-massratio.csv`` with a one-line header + ``conditions.csv`` with a one-line header."""
    sub = next(root.rglob("lowTdata")).resolve()
    with open(sub / "conditions.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = lambda name: next(k for k in rows[0] if k.strip() == name)                        # noqa: E731
    ratio, time = [], []
    for i in range(len(rows)):
        with open(sub / f"{i + 1}-massratio.csv", newline="") as fh:
            rdr = csv.reader(fh)
            next(rdr)
            rows_i = [(float(a), float(b)) for a, b in ((r[0], r[1]) for r in rdr if len(r) >= 2 and r[0])]
        t, r = _interp_1hz(np.array([x[0] for x in rows_i]), np.array([x[1] for x in rows_i]))
        time.append(t)
        ratio.append(r)
    n = max(len(t) for t in time)
    out_r = np.full((len(time), n), np.nan)
    out_t = np.full((len(time), n), np.nan)
    for i, (t, r) in enumerate(zip(time, ratio, strict=True)):
        out_r[i, :len(r)], out_t[i, :len(t)] = r, t
    inp = np.array([0 if (row[key("INP")] or "").strip() not in ("", "None") else 1 for row in rows], dtype=int)
    return {"mass_ratio": out_r, "time": out_t, "r0": np.array([float(r[key("initial radius (microns)")])
                                                                for r in rows]) * 1e-6,
            "T": np.array([float(r[key("temperature (C)")]) for r in rows]) + 273.15,
            "Si": np.array([float(r[key("supersaturation (%)")]) for r in rows]) / 100.0 + 1.0,
            "nucleation": inp, "source": np.array(["pokrifka2023"] * len(rows))}


#: The four cold experiments of the paper: (file name, T in C, S_i, r0 in microns) -- the paper's own table.
_COLD = (("20210420-130904-crop.dat", -63.17, 1.2111, 12.16),
         ("20210630-145335-crop.dat", -58.19, 1.3971, 13.63),
         ("20200825-150358.dat", -53.99, 1.5732, 9.54),
         ("20200721-150638.dat", -60.58, 1.7404, 12.09))


def cold_experiments(cold_dir: str | Path):
    """The paper's 4 high-supersaturation series from raw chamber files, if they are available locally.

    Each raw file has 13 whitespace-separated columns; column 5 ("Voltage") is proportional to the crystal mass, so
    ``m/m0`` is that column divided by its first non-negative value. Conditions are the paper's, not the archive's.
    """
    cold_dir = Path(cold_dir)
    ratio, time = [], []
    for fname, _Tc, _Si, _r0 in _COLD:
        path = cold_dir / fname
        raw = np.loadtxt(path, ndmin=2)
        volt = raw[:, 5]
        start = np.argmax(volt >= 0)
        volt = volt[start:]
        t, r = _interp_1hz(raw[start:, 0] - raw[start, 0], volt / volt[0])
        time.append(t)
        ratio.append(r)
    n = max(len(t) for t in time)
    out_r = np.full((len(time), n), np.nan)
    out_t = np.full((len(time), n), np.nan)
    for i, (t, r) in enumerate(zip(time, ratio, strict=True)):
        out_r[i, :len(r)], out_t[i, :len(t)] = r, t
    return {"mass_ratio": out_r, "time": out_t, "r0": np.array([c[3] for c in _COLD]) * 1e-6,
            "T": np.array([c[1] for c in _COLD]) + 273.15, "Si": np.array([c[2] for c in _COLD]),
            "nucleation": np.ones(len(_COLD), dtype=int), "source": np.array(["cold"] * len(_COLD))}


def _stack(parts: list[dict]) -> ChamberData:
    n = max(p["mass_ratio"].shape[1] for p in parts)
    pad = lambda a: np.pad(a, ((0, 0), (0, n - a.shape[1])), constant_values=np.nan)          # noqa: E731
    mass = np.vstack([pad(p["mass_ratio"]) for p in parts])
    time = np.vstack([pad(p["time"]) for p in parts])
    cat = lambda k, dtype=float: np.concatenate([np.asarray(p[k], dtype=dtype) for p in parts])  # noqa: E731
    r0 = cat("r0")
    return ChamberData(mass_ratio=mass, time=time, Si=cat("Si"), T=cat("T"),
                       P=np.full(r0.size, 97190.0),      # chamber pressure, estimated by the paper's preprocessing
                       r0=r0, m0=4.0 / 3.0 * np.pi * RHO_ICE * r0**3, nucleation=cat("nucleation", int),
                       source=np.concatenate([np.asarray(p["source"], dtype=str) for p in parts]))


def load(root: str | Path | None = None, *, cold_dir: str | Path | None = None, download_if_missing: bool = True,
         use_cache: bool = True) -> ChamberData:
    """The 286 (or 290 with ``cold_dir``) 1 Hz mass-ratio series, downloading the archives the first time.

    The parsed arrays are cached in ``<root>/levdata.npz``; ``use_cache=False`` re-parses the archives.
    """
    root = Path(root) if root is not None else cache_dir()
    npz = root / ("levdata290.npz" if cold_dir is not None else "levdata.npz")
    if use_cache and npz.exists():
        return _from_npz(npz)
    if download_if_missing:
        download(root)
    parts = [_read_pokrifka2018(root / "pokrifka2018"), _read_pokrifka2023(root / "pokrifka2023")]
    if cold_dir is not None:
        parts.insert(0, cold_experiments(cold_dir))
    data = _stack(parts)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **{k: getattr(data, k) for k in
                                ("mass_ratio", "time", "Si", "T", "P", "r0", "m0", "nucleation", "source")})
    return data


def _from_npz(path: Path) -> ChamberData:
    with np.load(path) as z:
        return ChamberData(**{k: z[k] for k in
                              ("mass_ratio", "time", "Si", "T", "P", "r0", "m0", "nucleation", "source")})


def _parse_fp(text: str) -> tuple[np.ndarray, np.ndarray]:      # pragma: no cover - convenience for notebooks
    """Parse a two-column text block into arrays (kept for interactive use)."""
    rows = [r.split() for r in io.StringIO(text).read().splitlines() if r.strip()]
    return np.array([float(r[0]) for r in rows]), np.array([float(r[1]) for r in rows])
