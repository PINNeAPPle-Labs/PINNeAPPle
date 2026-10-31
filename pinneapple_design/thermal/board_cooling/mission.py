"""The cooling mission: a board with an overheating part, the limits, the space, the fan budget, the design space.

    m = default_mission()                       # 160 x 100 mm 4-layer board, 30 W SoC in a 35 x 35 BGA, 35 °C air
    m = parse_mission("SoC de 30 W ... Tj max 95 °C ... dissipador até 70 x 70 x 30 mm ... ventoinha 0.05 W")
    m.space.sample(n, rng)                      # Latin hypercube designs inside the space, geometrically feasible

Every value parsed from the text keeps the words it came from (``m.provenance``); values not in the text are the
defaults of :func:`default_mission` and are listed as such.
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from pinneapple_physics.pcb import Board, Component

# Minimum fin gap of the design space: the forced-convection correlation of the heat sink model is used from
# 0.8 mm (pinneapple_design.thermal.heatsink.surrogate.MODE_RANGES); 1 mm keeps the designs inside it.
MIN_GAP_MM = 1.0


@dataclass
class CoolingDesign:
    """One heat sink on the hot part plus the fan air speed. Lengths in mm, speed in m/s (approach velocity of
    the enclosure fans over the board and through the fins; 0 = natural convection)."""
    base_width_mm: float
    base_depth_mm: float
    base_thickness_mm: float
    n_fins: int
    fin_thickness_mm: float
    height_mm: float                 # base + fins, above the package
    air_velocity_m_s: float
    material: str = "al6063"

    @property
    def fin_height_mm(self) -> float:
        return self.height_mm - self.base_thickness_mm

    @property
    def fin_gap_mm(self) -> float:
        return (self.base_width_mm - self.n_fins * self.fin_thickness_mm) / max(self.n_fins - 1, 1)

    def vector(self) -> np.ndarray:
        return np.array([self.base_width_mm, self.base_depth_mm, self.base_thickness_mm, self.n_fins,
                         self.fin_thickness_mm, self.height_mm, self.air_velocity_m_s], float)

    @classmethod
    def from_vector(cls, x, material: str = "al6063") -> CoolingDesign:
        x = np.asarray(x, float)
        return cls(float(x[0]), float(x[1]), float(x[2]), int(round(x[3])), float(x[4]), float(x[5]), float(x[6]),
                   material)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fin_height_mm": self.fin_height_mm, "fin_gap_mm": self.fin_gap_mm}


VARIABLES = ("base_width_mm", "base_depth_mm", "base_thickness_mm", "n_fins", "fin_thickness_mm", "height_mm",
             "air_velocity_m_s")


@dataclass
class DesignSpace:
    """Box bounds of the 7 design variables (``VARIABLES``) plus the geometric rules (fin gap, fin height)."""
    lo: np.ndarray
    hi: np.ndarray
    min_gap_mm: float = MIN_GAP_MM
    min_fin_height_mm: float = 4.0

    def feasible(self, x) -> bool:
        d = CoolingDesign.from_vector(x)
        return (d.fin_gap_mm >= self.min_gap_mm and d.fin_height_mm >= self.min_fin_height_mm
                and bool(np.all(x >= self.lo - 1e-9)) and bool(np.all(x <= self.hi + 1e-9)))

    def gap_margin(self, x) -> float:
        d = CoolingDesign.from_vector(x)
        return d.fin_gap_mm - self.min_gap_mm

    def sample(self, n: int, rng: np.random.Generator, max_tries: int = 50) -> np.ndarray:
        """``n`` feasible designs, Latin hypercube in the box (infeasible ones redrawn)."""
        from scipy.stats import qmc
        out: list[np.ndarray] = []
        seed = int(rng.integers(2 ** 31))
        for k in range(max_tries):
            u = qmc.LatinHypercube(d=len(self.lo), seed=seed + k).random(max(2 * n, 16))
            X = self.lo + u * (self.hi - self.lo)
            X[:, 3] = np.round(X[:, 3])
            out += [x for x in X if self.feasible(x)]
            if len(out) >= n:
                break
        if len(out) < n:
            raise RuntimeError(f"only {len(out)} feasible designs found in the space")
        return np.array(out[:n])

    def contains(self, x, tol: float = 1e-9) -> bool:
        x = np.asarray(x, float)
        return bool(np.all(x >= self.lo - tol) and np.all(x <= self.hi + tol))


@dataclass
class CoolingMission:
    """Board, parts, the overheating part (``hot``), air temperature, limits and the room for the cure.

    Attributes
    ----------
    t_ambient_c : float [degC]
        Air temperature.
    tj_limit_c : float [degC]
        Junction limit of the hot part.
    max_width_mm : float [mm]
        Heat-sink envelope.
    max_depth_mm : float [mm]
    max_height_mm : float [mm]
    fan_power_budget_w : float [W]
        Air (hydraulic) power through the fins.
    velocity_range_m_s : tuple[float, float] [m/s]
    tim_k_mm2_w : float [K mm2/W]
        Interface material resistance.
    """
    board: Board
    components: list[Component]
    hot: str
    t_ambient_c: float = 35.0
    tj_limit_c: float = 90.0                 # junction limit of the hot part (others: their own tj_max_c)
    max_width_mm: float = 70.0               # heat-sink envelope above the hot part
    max_depth_mm: float = 70.0
    max_height_mm: float = 30.0
    fan_power_budget_w: float = 0.02         # air (hydraulic) power through the fins, dp x flow
    velocity_range_m_s: tuple[float, float] = (0.5, 4.0)
    material: str = "al6063"
    tim_k_mm2_w: float = 10.0                # interface material R'' [K mm^2/W] (grease 5-20)
    text: str = ""
    provenance: dict[str, str] = field(default_factory=dict)

    def hot_component(self) -> Component:
        for c in self.components:
            if c.name == self.hot:
                return c.resolved()
        raise KeyError(self.hot)

    def limits(self) -> dict[str, float]:
        """Junction limit of each part [°C]."""
        return {c.name: (self.tj_limit_c if c.name == self.hot else c.tj_max_c) for c in self.components}

    @property
    def space(self) -> DesignSpace:
        src = self.hot_component()
        lo = np.array([src.w_mm, src.d_mm, 2.0, 6, 0.6, 12.0, self.velocity_range_m_s[0]])
        hi = np.array([self.max_width_mm, self.max_depth_mm, 8.0, 30, 2.5, self.max_height_mm,
                       self.velocity_range_m_s[1]])
        return DesignSpace(lo, hi)

    def with_components(self, comps: list[Component]) -> CoolingMission:
        m = copy.copy(self)
        m.components = comps
        return m

    def to_dict(self) -> dict[str, Any]:
        return {"board": asdict(self.board), "components": [asdict(c) for c in self.components], "hot": self.hot,
                "t_ambient_c": self.t_ambient_c, "tj_limit_c": self.tj_limit_c,
                "envelope_mm": [self.max_width_mm, self.max_depth_mm, self.max_height_mm],
                "fan_power_budget_w": self.fan_power_budget_w, "velocity_range_m_s": list(self.velocity_range_m_s),
                "material": self.material, "tim_k_mm2_w": self.tim_k_mm2_w, "text": self.text,
                "provenance": dict(self.provenance)}


def default_mission(power_w: float = 40.0) -> CoolingMission:
    """A 160 x 100 mm, 4-layer controller board: a ``power_w`` SoC (BGA 35 x 35, junction limit 90 °C), two 2 W
    regulators (QFN-32) and
    a 1.5 W memory (BGA 17 x 17), air at 35 °C inside the enclosure. Package data: the typical values of
    ``pinneapple_physics.closed_form.pcb_thermal.PACKAGES``."""
    comps = [Component("SoC", 60.0, 50.0, power_w, "BGA 35x35", tj_max_c=90.0),
             Component("VRM1", 118.0, 25.0, 2.0, "QFN-32 5x5", tj_max_c=125.0, vias=9),
             Component("VRM2", 118.0, 75.0, 2.0, "QFN-32 5x5", tj_max_c=125.0, vias=9),
             Component("DDR", 18.0, 50.0, 1.5, "BGA 17x17", tj_max_c=95.0)]
    return CoolingMission(Board(160.0, 100.0), comps, "SoC", tj_limit_c=90.0, fan_power_budget_w=0.02)


DEFAULT_TEXT = ("Placa de controle 160 x 100 mm com um SoC de 40 W superaquecendo. Temperatura de junção máxima "
                "90 °C, ar a 35 °C dentro da caixa. Cabe um dissipador de até 70 x 70 x 30 mm sobre o SoC e as "
                "ventoinhas podem gastar no máximo 0.02 W de potência de ar.")

_NUM = r"(\d+(?:[.,]\d+)?)"
_PATTERNS: list[tuple[str, str]] = [            # fan budget first: its watts are removed before the part power
    ("fan_power_budget_w", rf"(?:ventoinha|fan)s?.{{0,60}}?{_NUM}\s*W\b"),
    ("power_w", rf"{_NUM}\s*W\b"),
    ("tj_limit_c", rf"(?:jun[cç][aã]o|junction|tj).{{0,40}}?{_NUM}\s*°?\s*C\b"),
    ("t_ambient_c", rf"(?:\bar\b|\bair\b|ambient[e]?)\s*(?:a|at|de|of|=)?\s*{_NUM}\s*°?\s*C\b"),
    ("envelope", rf"{_NUM}\s*[x×]\s*{_NUM}\s*[x×]\s*{_NUM}\s*mm"),
    ("board", rf"{_NUM}\s*[x×]\s*{_NUM}\s*mm"),
]


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def parse_mission(text: str) -> CoolingMission:
    """Deterministic reading of a cooling request (Portuguese or English): the numbers it states replace the
    defaults; ``provenance`` maps each field to the quoted words or to ``"default"``."""
    m = default_mission()
    prov = {k: "default" for k in ("power_w", "tj_limit_c", "t_ambient_c", "envelope", "board",
                                   "fan_power_budget_w")}
    low = text.replace(" ", " ")
    for key, pat in _PATTERNS:
        hit = re.search(pat, low, flags=re.IGNORECASE)
        if not hit:
            continue
        if key in ("fan_power_budget_w", "envelope"):        # these numbers are not the part power / board size
            low = low[:hit.start()] + " " * (hit.end() - hit.start()) + low[hit.end():]
        g = [_f(v) for v in hit.groups()]
        if key == "power_w":
            hot = next(c for c in m.components if c.name == m.hot)
            hot.power_w = g[0]
        elif key == "envelope":
            m.max_width_mm, m.max_depth_mm, m.max_height_mm = g
        elif key == "board":
            m.board.width_mm, m.board.depth_mm = g
        else:
            setattr(m, key, g[0])
        prov[key] = hit.group(0).strip()
    hot = next(c for c in m.components if c.name == m.hot)
    hot.tj_max_c = m.tj_limit_c
    m.text, m.provenance = text, prov
    return m
