"""Design vector of the 3-D multiphysics motor study, and the machine it builds.

The cross-section of the interior-PM rotor (:class:`~..geometry.VRotor`) is the one of the 2-D studies; this module
adds the two directions those studies do not have and the two load variables that make the problem multiphysics:

* **axial** -- stack length ``L`` and rotor skew ``skew_deg`` (mechanical degrees of twist over the stack);
* **cooling** -- coolant volume flow through the water jacket ``flow_L_min``, which sets the jacket conductance and
  therefore the winding and magnet temperatures;
* **operating point** -- peak phase current ``I_peak``, current angle ``gamma_deg`` and rotor speed ``rpm``.

:data:`VARIABLES` is the single source of the names, units and bounds: the design space of the optimiser, the
columns of the training set and the inputs of the surrogate are all built from it, so a variable that is not in it
cannot leak into one of the three by accident.

:class:`Motor3DDesign` is a frozen dataclass: a design is a value. :meth:`Motor3DDesign.geometry` returns the
cross-section the mesher takes, :meth:`Motor3DDesign.sector_errors` the reasons it cannot be meshed (empty when it
can), so an optimiser can discard a design before it costs a solve.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry import MachineGeometry, geometry_errors, prius_2004

__all__ = ["DesignVariable", "VARIABLES", "NAMES", "BOUNDS", "Motor3DDesign", "to_matrix", "from_matrix"]


@dataclass(frozen=True)
class DesignVariable:
    """One design variable: ``name``, SI ``unit``, the box the study searches, and what it does to the machine."""
    name: str
    unit: str
    low: float
    high: float
    description: str
    group: str = "geometry"          # geometry | axial | operating | cooling

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low must be below high")

    @property
    def doe_group(self) -> str:
        """The group this variable maps to in the library's parameter-space contract (geometry, operating or
        boundary: the DOE report has no separate axial or cooling group)."""
        return {"axial": "geometry", "cooling": "boundary"}.get(self.group, self.group)


VARIABLES: tuple[DesignVariable, ...] = (
    DesignVariable("W4", "m", 0.0140, 0.0220,
                   "magnet width: more magnet flux, more magnet mass", "geometry"),
    DesignVariable("H3", "m", 0.0045, 0.0075,
                   "magnet thickness: flux per pole against demagnetisation margin", "geometry"),
    DesignVariable("H0", "m", 0.0085, 0.0135,
                   "depth of the V apex below the rotor surface: saliency and the iron bridge to the air gap",
                   "geometry"),
    DesignVariable("W0", "m", 0.0360, 0.0450,
                   "chord of the outer pocket ends: how far the magnets reach around the pole", "geometry"),
    DesignVariable("H1", "m", 0.0010, 0.0020,
                   "tangential bridge between the pocket and the air gap: leakage against mechanical retention",
                   "geometry"),
    DesignVariable("L", "m", 0.0600, 0.1100,
                   "stack length: torque and losses grow with it, so does the slot copper and the cooled surface",
                   "axial"),
    DesignVariable("skew_deg", "deg", 0.0, 15.0,
                   "mechanical skew of the rotor over the stack: trades mean torque for torque ripple", "axial"),
    DesignVariable("I_peak", "A", 80.0, 300.0, "peak phase current of the operating point", "operating"),
    DesignVariable("gamma_deg", "deg", 10.0, 75.0,
                   "current angle from the q axis: reluctance torque against magnet torque", "operating"),
    DesignVariable("rpm", "rpm", 500.0, 6000.0, "rotor speed: iron and magnet loss, and the voltage the inverter "
                                                "has to supply", "operating"),
    DesignVariable("flow_L_min", "L/min", 0.5, 12.0,
                   "coolant flow through the water jacket: h grows as flow^0.8, so the temperatures fall with it",
                   "cooling"),
)

NAMES: tuple[str, ...] = tuple(v.name for v in VARIABLES)
BOUNDS: np.ndarray = np.array([[v.low, v.high] for v in VARIABLES], float)


@dataclass(frozen=True)
class Motor3DDesign:
    """One point of the study: the five rotor cross-section variables, the two axial ones, the operating point and
    the cooling. Every value is SI (metres, amperes, revolutions per minute, litres per minute, degrees)."""
    W4: float = 0.0189
    H3: float = 0.0065
    H0: float = 0.01096
    W0: float = 0.0420
    H1: float = 0.0015
    L: float = 0.08382
    skew_deg: float = 0.0
    I_peak: float = 250.0
    gamma_deg: float = 50.0
    rpm: float = 1500.0
    flow_L_min: float = 6.0

    def __post_init__(self) -> None:
        for v in VARIABLES:
            x = getattr(self, v.name)
            if not math.isfinite(x):
                raise ValueError(f"{v.name} must be finite, got {x!r}")

    # ---------------------------------------------------------------------------------------- construction
    @classmethod
    def from_mapping(cls, p: Mapping[str, Any]) -> Motor3DDesign:
        """A design from a mapping that has at least every name of :data:`VARIABLES` (extra keys are ignored)."""
        missing = [n for n in NAMES if n not in p]
        if missing:
            raise KeyError(f"missing design variables: {missing}")
        return cls(**{n: float(p[n]) for n in NAMES})

    @classmethod
    def from_vector(cls, x: Sequence[float]) -> Motor3DDesign:
        """A design from the vector in the order of :data:`NAMES` (the optimiser's ordering)."""
        x = np.asarray(x, float).reshape(-1)
        if x.size != len(NAMES):
            raise ValueError(f"expected {len(NAMES)} values, got {x.size}")
        return cls(**dict(zip(NAMES, x.tolist(), strict=True)))

    @classmethod
    def baseline(cls) -> Motor3DDesign:
        """The 2004 Prius design at its published operating point: the reference every result is compared with."""
        return cls()

    def as_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in NAMES}

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in NAMES], float)

    def clipped(self) -> Motor3DDesign:
        """The design moved onto the box of :data:`VARIABLES` (keeps an optimiser's proposal meshable)."""
        return dataclasses.replace(self, **{v.name: float(np.clip(getattr(self, v.name), v.low, v.high))
                                            for v in VARIABLES})

    # ---------------------------------------------------------------------------------------- the machine
    def geometry(self) -> MachineGeometry:
        """The cross-section: the Prius stator and winding with this design's rotor and stack length."""
        base = prius_2004()
        rotor = dataclasses.replace(base.rotor, W4=self.W4, H3=self.H3, H0=self.H0, W0=self.W0, H1=self.H1)
        return dataclasses.replace(base, rotor=rotor, length=self.L)

    def sector_errors(self) -> list[str]:
        """Reasons the cross-section cannot be meshed (polygons crossing, leaving the pole...); empty when valid."""
        return geometry_errors(self.geometry().rotor.sector(1e-3))

    @property
    def skew_rad(self) -> float:
        return math.radians(self.skew_deg)

    @property
    def omega_mech(self) -> float:
        """Mechanical angular speed (rad/s)."""
        return self.rpm * 2.0 * math.pi / 60.0

    def skew_factor(self, pole_pairs: int = 4) -> float:
        """Fundamental skew factor ``sin(x) / x`` with ``x`` half the electrical skew angle (one half of the stack
        is twisted ``+skew/2`` and the other ``-skew/2``): the fraction of the fundamental flux linkage that the
        twist leaves useful."""
        x = 0.5 * pole_pairs * self.skew_rad
        return 1.0 if abs(x) < 1e-12 else math.sin(x) / x


def to_matrix(designs: Sequence[Motor3DDesign]) -> np.ndarray:
    """``(n, d)`` array of the designs in the order of :data:`NAMES`."""
    return np.array([d.as_vector() for d in designs], float).reshape(len(designs), len(NAMES))


def from_matrix(X: np.ndarray) -> list[Motor3DDesign]:
    """Designs from an ``(n, d)`` array in the order of :data:`NAMES`."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[1] != len(NAMES):
        raise ValueError(f"expected {len(NAMES)} columns, got {X.shape[1]}")
    return [Motor3DDesign.from_vector(row) for row in X]
