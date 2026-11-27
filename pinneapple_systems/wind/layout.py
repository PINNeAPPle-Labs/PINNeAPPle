"""Wind-farm layouts: turbines, coordinates and the geometry the wake models need.

A layout is deliberately small and explicit: a turbine is an identity plus the
three numbers the engineering models actually use (position, hub height, rotor
diameter), and everything else about a machine -- rated power, thrust curve --
travels with it as a *named* parameter so a report can always say where a number
came from.

The Penmanshiel scrape (``Penmanshiel_WT_static.csv``) is the reference layout:
14 Senvion MM82, 2.05 MW, 82 m rotor, 59 m hub, no WT03 (the numbering skips it).

Coordinates are metres in a local east/north frame, with the *meteorological*
convention everywhere: a wind direction of 270 deg means the wind blows **from**
the west and travels towards +x (east). Mixing that up silently reverses every
wake in the model, so the conversion lives in exactly one function,
:func:`wind_vector`.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Turbine",
    "WindFarmLayout",
    "power_law_exponent",
    "wind_vector",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def wind_vector(direction_deg: float) -> tuple[float, float]:
    """Unit vector *towards which* the wind travels, for a meteorological direction.

    ``direction_deg`` is where the wind comes **from**, clockwise from north, so
    the flow travels towards ``direction_deg + 180``:

    ==============  ==================  ==========================
    direction_deg   comes from          travels towards (x, y)
    ==============  ==================  ==========================
    0               north               south  (0, -1)
    90              east                west   (-1, 0)
    180             south               north  (0, +1)
    270             west                east   (+1, 0)
    ==============  ==================  ==========================

    Getting this half-turn wrong reverses every wake in the model, and it does
    so *invisibly*: a farm-averaged loss looks the same either way, and only a
    direction-resolved comparison against data catches it.
    """
    rad = math.radians(float(direction_deg))
    return (-math.sin(rad), -math.cos(rad))


def power_law_exponent(
    *,
    z_ref: float = 10.0,
    alpha: float = 0.143,
    height: float = 59.0,
) -> float:
    """Shear factor ``(height / z_ref) ** alpha`` for extrapolating to hub height.

    ``alpha = 0.143`` is the 1/7 power law; it is a *documented default*, not a
    measurement. Pass a measured shear exponent when the site has one.
    """
    return (float(height) / float(z_ref)) ** float(alpha)


# ---------------------------------------------------------------------------
# Turbine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Turbine:
    """One machine: where it stands and what it is.

    ``thrust_peak`` and ``cut_in``/``cut_out``/``rated_speed`` describe a
    reference thrust/power characteristic, **not** measured data. They are
    carried as parameters so a wake result can state which values it used.
    """

    name: str
    x: float
    y: float
    hub_height: float
    rotor_diameter: float
    rated_power_kw: float
    elevation: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    cut_in: float = 3.0
    cut_out: float = 25.0
    rated_speed: float = 13.0
    thrust_peak: float = 0.82
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.rotor_diameter <= 0 or self.hub_height <= 0 or self.rated_power_kw <= 0:
            raise ValueError(f"{self.name}: rotor, hub height and rated power must be positive")
        if not 0 < self.cut_in < self.rated_speed < self.cut_out:
            raise ValueError(f"{self.name}: need 0 < cut-in < rated < cut-out")

    @property
    def rotor_area(self) -> float:
        """Swept area [m^2]."""
        return math.pi * (self.rotor_diameter / 2.0) ** 2

    def thrust_coefficient(self, wind_speed):
        """Thrust coefficient :math:`C_T(U)` of this machine's reference curve.

        Constant at :attr:`thrust_peak` through region 2, then decaying as
        :math:`U^{-2}` because the thrust of a pitch-regulated turbine is close
        to constant once rated power is reached. Below cut-in and above cut-out
        the rotor is not producing thrust into the wake model.

        This is a *model*, not a measured :math:`C_T` table. It is the single
        most influential free parameter of the wake result, which is why it is
        exposed on the turbine rather than buried in the solver.
        """
        u = np.asarray(wind_speed, dtype=float)
        ct = np.where(
            u < self.cut_in,
            0.0,
            np.where(
                u <= self.rated_speed,
                self.thrust_peak,
                self.thrust_peak * (self.rated_speed / np.maximum(u, 1e-9)) ** 2,
            ),
        )
        return np.where(u > self.cut_out, 0.0, np.minimum(ct, 0.999))

    def available_power_fraction(self, wind_speed) -> np.ndarray:
        """Binary operating window: 1 inside [cut-in, cut-out], 0 outside."""
        u = np.asarray(wind_speed, dtype=float)
        return ((u >= self.cut_in) & (u <= self.cut_out)).astype(float)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class WindFarmLayout:
    """An ordered collection of :class:`Turbine` with vector accessors.

    Order is load-bearing: every array returned by the wake models is indexed by
    the same order as :attr:`turbines`, so a per-turbine result can always be
    pasted back onto the map.
    """

    def __init__(self, name: str, turbines: Sequence[Turbine]):
        turbines = list(turbines)
        if not turbines:
            raise ValueError("a layout needs at least one turbine")
        names = [t.name for t in turbines]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate turbine names in layout {name!r}")
        self.name = name
        self.turbines = turbines

    # -- construction -------------------------------------------------------
    @classmethod
    def from_rows(cls, name: str, rows: Iterable[dict]) -> WindFarmLayout:
        """Build from parsed CSV rows with the Penmanshiel column names."""
        turbines = []
        for r in rows:
            lat = _optional_float(r.get("Latitude"))
            lon = _optional_float(r.get("Longitude"))
            turbines.append(
                Turbine(
                    name=str(r["Alternative Title"]).strip(),
                    x=0.0,  # filled by :meth:`with_local_frame`
                    y=0.0,
                    hub_height=float(r["Hub Height (m)"]),
                    rotor_diameter=float(r["Rotor Diameter (m)"]),
                    rated_power_kw=float(r["Rated power (kW)"]),
                    elevation=_optional_float(r.get("Elevation (m)")),
                    latitude=lat,
                    longitude=lon,
                    metadata={
                        "title": r.get("Title"),
                        "model": r.get("Model"),
                        "manufacturer": r.get("Manufacturer"),
                        "identity": r.get("Identity"),
                    },
                )
            )
        layout = cls(name, turbines)
        if all(t.latitude is not None and t.longitude is not None for t in turbines):
            return layout.with_local_frame()
        return layout

    @classmethod
    def from_penmanshiel_csv(cls, path: str | "os.PathLike[str]") -> WindFarmLayout:
        """Read ``Penmanshiel_WT_static.csv`` (Zenodo record 16807304).

        The published file carries trailing empty lines and a UTF-8 BOM; both are
        handled here so the loader can be pointed at the download unchanged.
        """
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.DictReader(fh) if (r.get("Alternative Title") or "").strip()]
        return cls.from_rows("Penmanshiel", rows)

    def with_local_frame(self) -> WindFarmLayout:
        """Project lat/lon onto a local east/north metric frame (equirectangular).

        Over 4 km the equirectangular distortion is far below the rotor diameter,
        so a flat-earth projection is exact enough for wake geometry; it is not
        good enough to be called a survey.
        """
        lat0 = float(np.mean([t.latitude for t in self.turbines]))
        lon0 = float(np.mean([t.longitude for t in self.turbines]))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
        moved = [
            Turbine(
                name=t.name,
                x=(t.longitude - lon0) * m_per_deg_lon,
                y=(t.latitude - lat0) * m_per_deg_lat,
                hub_height=t.hub_height,
                rotor_diameter=t.rotor_diameter,
                rated_power_kw=t.rated_power_kw,
                elevation=t.elevation,
                latitude=t.latitude,
                longitude=t.longitude,
                cut_in=t.cut_in,
                cut_out=t.cut_out,
                rated_speed=t.rated_speed,
                thrust_peak=t.thrust_peak,
                metadata=t.metadata,
            )
            for t in self.turbines
        ]
        return WindFarmLayout(self.name, moved)

    # -- accessors ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.turbines)

    def __iter__(self):
        return iter(self.turbines)

    def __repr__(self) -> str:
        return f"WindFarmLayout({self.name!r}, {len(self)} turbines)"

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.turbines]

    @property
    def positions(self) -> np.ndarray:
        """``(N, 2)`` array of local east/north coordinates [m]."""
        return np.array([[t.x, t.y] for t in self.turbines], dtype=float)

    @property
    def rotor_diameters(self) -> np.ndarray:
        return np.array([t.rotor_diameter for t in self.turbines], dtype=float)

    @property
    def hub_heights(self) -> np.ndarray:
        return np.array([t.hub_height for t in self.turbines], dtype=float)

    @property
    def rated_powers(self) -> np.ndarray:
        """Installed capacity per turbine [kW]."""
        return np.array([t.rated_power_kw for t in self.turbines], dtype=float)

    @property
    def installed_capacity_mw(self) -> float:
        return float(self.rated_powers.sum() / 1000.0)

    @property
    def thrust_peak(self) -> np.ndarray:
        return np.array([t.thrust_peak for t in self.turbines], dtype=float)

    def spacing_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Pairwise ``(distance, bearing_from)`` matrices in metres and degrees.

        ``bearing_from[i, j]`` is the meteorological direction along which wind
        must blow for turbine ``j`` to sit directly downstream of turbine ``i``.
        """
        p = self.positions
        d = p[:, None, :] - p[None, :, :]
        dist = np.hypot(d[..., 0], d[..., 1])
        bearing = (np.degrees(np.arctan2(d[..., 0], d[..., 1]))) % 360.0
        return dist, bearing

    def project(self, direction_deg: float) -> np.ndarray:
        """Upwind coordinate of every turbine for one wind direction.

        Larger value = further upwind. Sorted descending this gives the order in
        which a wake propagates, which is the only ordering the superposition
        needs.
        """
        ux, uy = wind_vector(direction_deg)
        return -(self.positions[:, 0] * ux + self.positions[:, 1] * uy)

    def upwind_order(self, direction_deg: float) -> list[int]:
        """Indices of the turbines from most upwind to most downwind."""
        return list(np.argsort(-self.project(direction_deg)))

    def to_records(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "x_m": round(t.x, 2),
                "y_m": round(t.y, 2),
                "hub_height_m": t.hub_height,
                "rotor_diameter_m": t.rotor_diameter,
                "rated_power_kw": t.rated_power_kw,
                "thrust_peak": t.thrust_peak,
                "elevation_m": t.elevation,
                "latitude": t.latitude,
                "longitude": t.longitude,
            }
            for t in self.turbines
        ]


def _optional_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
