"""Data model of a wear twin: vessel spec (geometry + zones) and the wear dataset.

A *zone* is a surface of revolution: a profile polyline ``(r, y)`` swept around the vertical axis
and cut into ``n_rows x n_sectors`` cells. That single abstraction covers a cylindrical wall, a cone,
a flat or dished floor (profile along r at fixed y), and an off-axis leg (``center``), which is how
the four reference refractory twins (steel ladle, pig-iron ladle, BOF, RH degasser) are described.

The dataset is the *contract* shared with the web hub of the reference suite
(``to_contract`` / ``from_contract``): measurements are wear depth per cell and per reading ``x``
(heat number, days, ...), never a model output unless ``origin`` says so.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SYNTHETIC = "synthetic"
MEASURED = "measured"


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    label: str
    profile: Tuple[Tuple[float, float], ...]  # (r, y) points from first row to last row, metres
    n_rows: int
    n_sectors: int
    e0: float  # new lining thickness [mm]
    emin: float  # minimum allowed thickness [mm]
    center: Tuple[float, float] = (0.0, 0.0)  # (x, z) of the revolution axis
    theta0: float = 0.0  # angle of sector 0 [rad]
    group: str = ""

    def __post_init__(self):
        if len(self.profile) < 2:
            raise ValueError(f"zone '{self.name}': profile needs at least 2 points")
        if self.n_rows < 1 or self.n_sectors < 3:
            raise ValueError(f"zone '{self.name}': need n_rows >= 1 and n_sectors >= 3")
        if not self.e0 > self.emin >= 0:
            raise ValueError(f"zone '{self.name}': require e0 > emin >= 0 (got {self.e0}, {self.emin})")

    @property
    def usable(self) -> float:
        return self.e0 - self.emin

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_sectors)


@dataclass(frozen=True)
class VesselSpec:
    key: str
    title: str
    zones: Tuple[ZoneSpec, ...]
    x_name: str = "heat"  # what ``x`` counts
    wear_unit: str = "mm"
    notes: str = ""

    def __post_init__(self):
        names = [z.name for z in self.zones]
        if len(set(names)) != len(names):
            raise ValueError("duplicate zone names")

    def zone(self, name: str) -> ZoneSpec:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(name)


@dataclass
class WearDataset:
    """Wear depth [mm] per zone: ``wear[zone]`` has shape (T, rows, sectors); NaN = no reading."""

    spec: VesselSpec
    x: np.ndarray  # (T,) strictly increasing
    wear: Dict[str, np.ndarray]
    origin: str = MEASURED
    labels: Optional[List[str]] = None
    note: str = ""

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim != 1 or len(self.x) < 1:
            raise ValueError("x must be a non-empty 1-D array")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x must be strictly increasing")
        if self.origin not in (SYNTHETIC, MEASURED):
            raise ValueError(f"origin must be '{SYNTHETIC}' or '{MEASURED}'")
        for name, arr in list(self.wear.items()):
            z = self.spec.zone(name)  # KeyError for unknown zone
            a = np.asarray(arr, dtype=float)
            if a.shape != (len(self.x),) + z.shape:
                raise ValueError(f"zone '{name}': expected shape {(len(self.x),) + z.shape}, got {a.shape}")
            if np.any(a[np.isfinite(a)] < 0):
                raise ValueError(f"zone '{name}': negative wear depth")
            self.wear[name] = a
        if not self.wear:
            raise ValueError("dataset has no zones")
        if self.labels is not None and len(self.labels) != len(self.x):
            raise ValueError("labels must have one entry per reading")

    @property
    def zone_names(self) -> List[str]:
        return [z.name for z in self.spec.zones if z.name in self.wear]

    # ---- web-hub contract (digital-solutions/shared/bridge.js) -----------------------------
    def to_contract(self) -> dict:
        zones = {n: {"N": self.spec.zone(n).n_rows, "M": self.spec.zone(n).n_sectors,
                     "e0": self.spec.zone(n).e0, "emin": self.spec.zone(n).emin} for n in self.zone_names}
        snaps = []
        for t, x in enumerate(self.x):
            snaps.append({"x": float(x), "label": (self.labels[t] if self.labels else f"{self.spec.x_name} {x:g}"),
                          "wear": {n: [[None if not np.isfinite(v) else round(float(v), 3) for v in row]
                                       for row in self.wear[n][t]] for n in self.zone_names}})
        return {"origem": "sintético" if self.origin == SYNTHETIC else "carregado", "titulo": self.spec.title,
                "zones": zones, "snaps": snaps}

    @classmethod
    def from_contract(cls, spec: VesselSpec, d: dict) -> "WearDataset":
        if "snaps" not in d or "zones" not in d:
            raise ValueError("contract needs 'zones' and 'snaps'")
        snaps = d["snaps"]
        names = list(d["zones"])
        wear = {}
        for n in names:
            z = spec.zone(n)
            for k in ("N", "M"):
                want = z.n_rows if k == "N" else z.n_sectors
                if d["zones"][n][k] != want:
                    raise ValueError(f"zone '{n}': {k}={d['zones'][n][k]} does not match spec ({want})")
            wear[n] = np.array([[[np.nan if v is None else v for v in row] for row in s["wear"][n]] for s in snaps],
                               dtype=float)
        return cls(spec, [s["x"] for s in snaps], wear,
                   origin=SYNTHETIC if d.get("origem") == "sintético" else MEASURED,
                   labels=[s.get("label", "") for s in snaps])

    # ---- CSV: zona,fiada,setor,valor,corrida (same columns as the reference apps) ----------
    @classmethod
    def from_csv(cls, spec: VesselSpec, text: str) -> "WearDataset":
        rd = csv.reader(io.StringIO(text), delimiter=";" if ";" in text.split("\n", 1)[0] else ",")
        rows = [r for r in rd if r and not r[0].lstrip().startswith("#")]
        head = [h.strip().lower() for h in rows[0]]
        need = ["zona", "fiada", "setor", "valor", "corrida"]
        if any(h not in head for h in need):
            raise ValueError(f"CSV header must contain {need}; got {head}")
        ix = {h: head.index(h) for h in need}
        xs = sorted({float(r[ix["corrida"]]) for r in rows[1:]})
        pos = {x: k for k, x in enumerate(xs)}
        wear: Dict[str, np.ndarray] = {}
        for r in rows[1:]:
            z = spec.zone(r[ix["zona"]].strip())
            i, j = int(r[ix["fiada"]]) - 1, int(r[ix["setor"]]) - 1
            if not (0 <= i < z.n_rows and 0 <= j < z.n_sectors):
                raise ValueError(f"zone '{z.name}': fiada/setor ({i + 1},{j + 1}) outside {z.shape}")
            a = wear.setdefault(z.name, np.full((len(xs),) + z.shape, np.nan))
            v = r[ix["valor"]].strip().replace(",", ".")
            a[pos[float(r[ix["corrida"]])], i, j] = float(v) if v else np.nan
        return cls(spec, xs, wear, origin=MEASURED)
