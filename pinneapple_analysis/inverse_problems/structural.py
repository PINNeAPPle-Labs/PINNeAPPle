"""Identify the parameters of a structure from displacements measured at a few points (#177, item S28).

The engineering question is the everyday one: a load was applied to a cantilever, a dial gauge or an LVDT
recorded the deflection at a few stations, and the question is what the beam was made of and how hard it
was pushed. This module answers it with the machinery already in
:mod:`pinneapple_analysis.inverse_problems`: a forward model (:class:`~.forward.CallableModel` wrapping
Timoshenko beam theory, or CalculiX itself), :func:`~.recovery.recover` for the fit and the intervals, and
:func:`~.identifiability.analyze_identifiability` for the question that decides whether the intervals mean
anything.

The answer nobody expects: E and P cannot both be identified from displacements
-------------------------------------------------------------------------------
For a cantilever the deflection is ``w(x) = P f(x) / E`` -- linear in the load and linear in the
compliance. Displacements at any number of stations, at **one** load level, therefore see the ratio
``P/E`` and nothing else: doubling E and doubling P leaves every measured number identical. The
identifiability analysis reports it as a dead direction, and :func:`identify_cantilever` refuses to quote
two confident intervals for a pair the data only see as a product. What makes them separable is
information the displacements do not carry: a load cell (``unknown="modulus"``), a material certificate
(``unknown="load"``), or measurements at two different load levels.

What is measured, and what the uncertainty is
----------------------------------------------
The synthetic data come from **CalculiX** (a C3D20R brick cantilever, the case validated in
``tests/test_calculix_bridge.py``), and the forward model used for the inversion is the Timoshenko beam.
Those are not the same model: the 3-D element carries shear, Poisson effects and the local deformation
under the load, which the beam formula only approximates, and the gap measured at the four stations of the
reference case is 3.3 %. Rather than pretend otherwise, the observation
noise handed to the inversion carries the measured model-form gap as a second term
(``sigma = sqrt(sigma_noise^2 + (model_form * |w|)^2)``), which is the ASME V&V 20 practice: a beam model
that claims 0.1 % accuracy against a 3-D solid would give an interval that is confidently wrong, and the
truth would fall outside it.

References
----------
* Timoshenko, S., *Strength of Materials*, 3rd ed., Van Nostrand, 1955 -- the deflection with shear.
* Roache, P.J., *Verification and Validation in Computational Science and Engineering*, Hermosa, 1998;
  ASME V&V 20-2009 -- numerical, input and model-form uncertainty as one budget.
* Raue, A. et al., "Structural and practical identifiability analysis of partially observed dynamical
  models", *Bioinformatics* 25(15), 2009 -- the profile-likelihood and dead-direction reading used by
  ``inverse_problems.identifiability``.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .forward import CallableModel, Observations, Parameter, ParameterSet
from .recovery import RecoveryResult, recover

__all__ = [
    "GEOMETRY",
    "IDENTIFICATION_FIXTURE",
    "Identification",
    "ObservationSet",
    "cantilever_measurements",
    "identify_cantilever",
    "load_measurements",
    "timoshenko_deflection",
    "timoshenko_model",
]

#: the geometry and the material of the reference case (`examples/calculix_cantilever`), SI metres and pascals
GEOMETRY = {"L": 1.0, "b": 0.1, "h": 0.1, "nu": 0.33}

#: where the displacements measured by CalculiX are kept, so the study runs without the solver
IDENTIFICATION_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "data" / "calculix_cantilever_displacements.json"


def timoshenko_deflection(
    x: Sequence[float] | np.ndarray,
    *,
    E: float,
    P: float,
    L: float = GEOMETRY["L"],
    b: float = GEOMETRY["b"],
    h: float = GEOMETRY["h"],
    nu: float = GEOMETRY["nu"],
) -> np.ndarray:
    """Deflection of a cantilever under a tip load, with the shear term:
    ``w(x) = P x^2 (3L - x) / (6 E I) + P x / (kappa G A)``, ``I = b h^3 / 12``, ``kappa = 5/6``.

    The second term is why this model is closer to the 3-D solid than Euler-Bernoulli: at ``L/h = 10``
    shear carries 0.8 % of the tip deflection, and a model that omitted it would put that 0.8 % into the
    recovered modulus.
    """
    xi = np.asarray(x, dtype=np.float64)
    inertia = b * h**3 / 12.0
    shear = E / (2.0 * (1.0 + nu))
    area = b * h
    bending = P * xi**2 * (3.0 * L - xi) / (6.0 * E * inertia)
    transverse = P * xi / ((5.0 / 6.0) * shear * area)
    return bending + transverse


def timoshenko_model(
    *,
    stations: Sequence[float] | np.ndarray,
    L: float = GEOMETRY["L"],
    b: float = GEOMETRY["b"],
    h: float = GEOMETRY["h"],
    nu: float = GEOMETRY["nu"],
    E: float,
    P: float,
) -> CallableModel:
    """The forward model the inversion calls: Timoshenko deflection at ``stations`` from ``(E, P)``.

    ``E`` and ``P`` are the analysis point (the initial guess of an estimation), the same convention the
    rest of :mod:`pinneapple_analysis.inverse_problems` uses.
    """
    x = np.asarray(stations, dtype=np.float64)

    def predict(theta: dict[str, float], inputs: dict[str, np.ndarray]) -> np.ndarray:
        return timoshenko_deflection(x, E=theta["E"], P=theta["P"], L=L, b=b, h=h, nu=nu)

    return CallableModel(predict, inputs=("x",), outputs=("w",), name="timoshenko_cantilever",
                         description="tip-loaded cantilever deflection with shear, at the gauge stations",
                         parameters=[Parameter("E", E, 10e9, 400e9, "Pa", truth=None,
                                               description="Young's modulus"),
                                     Parameter("P", P, 1.0, 5000.0, "N", truth=None,
                                               description="tip load")])


def cantilever_measurements(
    *,
    E: float,
    P: float,
    stations: Sequence[float],
    noise: float,
    model_form: float,
    L: float = GEOMETRY["L"],
    b: float = GEOMETRY["b"],
    h: float = GEOMETRY["h"],
    nu: float = GEOMETRY["nu"],
    seed: int = 0,
) -> ObservationSet:
    """Synthetic displacements at ``stations``, with the noise and the model-form gap declared separately.

    ``noise`` is the gauge resolution in metres (an LVDT is micrometres); ``model_form`` is the measured
    relative gap between the 3-D solver and the beam formula, folded into the sigma the inversion is given
    so that a bias the model cannot represent widens the interval instead of displacing it silently.
    """
    truth = timoshenko_deflection(stations, E=E, P=P, L=L, b=b, h=h, nu=nu)
    generator = np.random.default_rng(seed)
    measured = truth + generator.normal(0.0, noise, size=truth.size)
    sigma = np.sqrt(noise**2 + (model_form * np.abs(truth)) ** 2)
    return ObservationSet(stations=np.asarray(stations, dtype=np.float64), measured=measured, sigma=sigma,
                          truth=truth, noise=noise, model_form=model_form, truth_parameters={"E": E, "P": P})


class ObservationSet:
    """A set of gauge readings: the stations, what they read, and the sigma each reading carries."""

    def __init__(self, *, stations: np.ndarray, measured: np.ndarray, sigma: np.ndarray, truth: np.ndarray,
                 noise: float, model_form: float, truth_parameters: dict[str, float]) -> None:
        self.stations = np.asarray(stations, dtype=np.float64)
        self.measured = np.asarray(measured, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.truth = np.asarray(truth, dtype=np.float64)
        self.noise = float(noise)
        self.model_form = float(model_form)
        self.truth_parameters = dict(truth_parameters)

    @property
    def observations(self) -> Observations:
        """The same readings in the form :func:`~.recovery.recover` takes."""
        return Observations({"x": self.stations}, self.sigma, self.measured,
                            observable=np.array(["w"] * self.stations.size, dtype=object),
                            meta={"source": "cantilever", "noise": self.noise, "model_form": self.model_form})

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form (this is what the CalculiX run is stored as)."""
        return {"stations": [float(v) for v in self.stations],
                "measured": [float(v) for v in self.measured],
                "sigma": [float(v) for v in self.sigma],
                "truth": [float(v) for v in self.truth],
                "noise": self.noise, "model_form": self.model_form,
                "truth_parameters": dict(self.truth_parameters)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObservationSet:
        return cls(stations=np.asarray(payload["stations"], dtype=float),
                   measured=np.asarray(payload["measured"], dtype=float),
                   sigma=np.asarray(payload["sigma"], dtype=float),
                   truth=np.asarray(payload["truth"], dtype=float),
                   noise=float(payload["noise"]), model_form=float(payload["model_form"]),
                   truth_parameters={k: float(v) for k, v in payload["truth_parameters"].items()})


def load_measurements(path: str | Path | None = None) -> ObservationSet:
    """The displacements measured by CalculiX and stored with the tests, so this runs without the solver."""
    target = Path(path) if path is not None else IDENTIFICATION_FIXTURE
    if not target.is_file():
        raise FileNotFoundError(
            f"no measured displacement set at {target}. Generate it with "
            "``python scripts/structural_identification.py --write`` (needs the ccx binary).")
    return ObservationSet.from_dict(json.loads(target.read_text(encoding="utf-8")))


def identify_cantilever(
    measurements: ObservationSet,
    *,
    unknown: str = "modulus",
    guess: dict[str, float] | None = None,
    interval: str = "profile",
) -> Identification:
    """Recover the unknown parameter(s) and return the fit, the intervals and the identifiability analysis.

    ``unknown`` is what the experiment does **not** know:

    ``"modulus"``
        the load is on a load cell and the material is unknown (``E`` free, ``P`` fixed);
    ``"load"``
        the material is certified and the actuator force is not (``P`` free, ``E`` fixed);
    ``"both"``
        neither is known. The fit runs, and the result carries the dead direction that says only ``P/E`` is
        identifiable -- an infinite interval per parameter rather than two confident-looking numbers.

    ``guess`` starts the fit away from the truth on purpose: passing the true values would make every
    ``recover`` look good.
    """
    if unknown not in ("modulus", "load", "both"):
        raise ValueError(f"unknown is 'modulus', 'load' or 'both'; got {unknown!r}")
    truth = measurements.truth_parameters
    start = guess or {}
    E0 = float(start.get("E", 200e9))
    P0 = float(start.get("P", 500.0))
    # a fixed parameter is a *known* one: the load cell reading or the material certificate, not the guess
    E_fixed = unknown == "load"
    P_fixed = unknown == "modulus"
    fitted = ParameterSet([
        Parameter("E", float(truth["E"]) if E_fixed else E0, 1e9, 500e9, "Pa", fixed=E_fixed,
                  truth=truth["E"]),
        Parameter("P", float(truth["P"]) if P_fixed else P0, 1.0, 5000.0, "N", fixed=P_fixed,
                  truth=truth["P"]),
    ])
    model = timoshenko_model(stations=measurements.stations, E=E0, P=P0)
    model._params = fitted.all
    result = recover(model, fitted, measurements.observations, interval=interval)
    return Identification(result=result, parameters=fitted, unknown=unknown)


@dataclass(frozen=True)
class Identification:
    """One identification run: the fit, the parameters it fitted, and the two questions it has to answer."""

    result: RecoveryResult
    parameters: ParameterSet
    unknown: str

    def verdict(self) -> dict[str, Any]:
        """Did it come back, and could it have?

        ``inside`` says, per parameter, whether the truth of the synthetic study falls inside the reported
        interval -- the acceptance criterion of #177. ``dead`` lists the directions the data cannot separate,
        which is why a parameter can sit inside a very wide interval and still not be identified.
        """
        intervals = {i.name: i for i in self.result.intervals}
        rows: dict[str, Any] = {}
        for parameter in self.parameters.free:
            interval = intervals[parameter.name]
            truth = parameter.truth
            rows[parameter.name] = {
                "value": interval.estimate, "low": interval.lower, "high": interval.upper, "truth": truth,
                "inside": None if truth is None else bool(interval.contains(truth)),
                "relative_width": (None if not np.isfinite(interval.width)
                                   else float(interval.width / abs(interval.estimate)
                                              if interval.estimate else math.inf)),
            }
        return {"unknown": self.unknown, "parameters": rows,
                "dead": [d.get("text", "") for d in self.result.dead],
                "condition_number": float(self.result.identifiability.condition_number),
                "interval": self.result.interval_kind, "warnings": list(self.result.warnings)}

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return self.verdict()
