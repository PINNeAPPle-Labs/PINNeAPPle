"""Automatic scaling and conditioning of a problem before it is trained (X22, #206).

Every variable in a simulation arrives in whatever units the engineer works in: a length in metres, a
conductivity in W/(m K), a source in W/m^3, a temperature in kelvin. A network sees only the numbers, so a
residual of ``k T'' + f`` with ``k = 0.15`` and ``f = 3000`` enters the loss as ``9e6`` while the boundary
term enters as ``1e4``: the optimiser spends its whole budget on the term with the biggest units, and the
result is a model that fits the source and ignores the boundary. The remedy is old and unglamorous --
nondimensionalise, normalise the fields, weight the loss terms by what they are worth -- and this module
does it from the numbers of the problem, not from a guess about them.

What it measures, and what it does with the measurement
-------------------------------------------------------
``condition_report``
    Takes the domain (per coordinate), the declared fields (their expected range) and the characteristic
    magnitude of each PDE term, and returns a :class:`ConditionReport`: one :class:`Scale` per variable
    (centre and half-width, so ``apply`` maps the range onto ``[-1, 1]``), one weight per residual term
    (``(target / magnitude)^2``, so the weighted term is O(target)), and the dimensionless groups the
    problem admits.
``dimensionless_groups``
    ``Re``, ``Pe``, ``Ma``, ``Pr``, ``Gr``, ``Ra``, ``Ri``, ``Bi``, ``Kn``, ``Fr`` -- computed from SI
    parameters through :func:`pinneapple_physics.physics_audit.regime.regime_features`, and the ones the
    supplied parameters cannot build are **named as missing**, with the parameters each needs, instead of
    left out or defaulted. "Re = 20,000 from geometry, velocity, density and viscosity" is a report, not a
    formula the user has to remember.
``scaling_benchmark``
    The claim of the issue, measured: the same network, seed, optimiser, budget and loss structure, trained
    on the same 1-D diffusion-reaction problem three ways -- raw physical units, a hand-tuned scaling, and
    the automatic one -- and the error against the analytic solution recorded for each. This is the one
    place in the module that touches torch, and it keeps the comparison honest by changing nothing but the
    scaling between the arms.

What this module does not do
----------------------------
It does not choose a network, a schedule or an adaptive-weight scheme (#1060's balancers are in
``pinneapple_neural.trainer.weight_scheduler``: ``SelfAdaptiveWeights``, ``GradNormBalancer``,
``LossRatioBalancer``, ``NTKWeightBalancer``). Those *re-balance during* training; this module decides
what the numbers should be *before* it starts, which is what makes the two composable. It also does not do
a symbolic nondimensionalisation: it cannot take ``rho u . grad u = -grad p + mu lap u`` apart term by term
and produce ``Re`` -- that would need the equation tree, which lives in
``pinneapple_physics.symbolic_pde``. What it does is the arithmetic part, from magnitudes the caller can
honestly state.

References
----------
* Buckingham, E., "On physically similar systems", *Physical Review* 4(4), 1914 -- the theorem behind
  nondimensionalisation.
* Karniadakis, G.E. et al., "Physics-informed machine learning", *Nature Reviews Physics* 3, 2021 -- why
  the loss balance matters for a PINN.
* Wang, S., Teng, Y. and Perdikaris, P., "Understanding and mitigating gradient flow pathologies in
  physics-informed neural networks", *SIAM J. Sci. Comput.* 43(5), 2021 -- the weighting.
* Fornberg, B. and Martel, J.M., "A stable and hig-accuracy numerical method for the solution of the
  Navier-Stokes equations", *J. Sci. Comput.* 28, 2006 -- why the length scale is the one to fix first.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_physics.physics_audit.regime import regime_features

from .numerics.diagnostics import Diagnostic, explain
from .verification.dimensional_analysis import compute_dimensionless_numbers

__all__ = [
    "BenchmarkResult",
    "ConditionReport",
    "Scale",
    "condition_report",
    "dimensionless_groups",
    "scaling_benchmark",
]


# --------------------------------------------------------------------------- the affine scale
@dataclass(frozen=True)
class Scale:
    """``v -> (v - centre) / scale``: one variable's affine map onto a range of order one."""

    name: str
    centre: float
    scale: float
    low: float
    high: float

    @classmethod
    def from_range(cls, name: str, low: float, high: float) -> Scale:
        """The map that sends ``[low, high]`` to ``[-1, 1]``."""
        if not np.isfinite([low, high]).all():
            raise ValueError(f"{name}: the range must be finite; got [{low}, {high}]")
        if high < low:
            low, high = high, low
        half = 0.5 * (high - low)
        if half == 0.0:
            raise ValueError(f"{name}: the range [{low}, {high}] has zero width, so nothing can be scaled")
        return cls(name=name, centre=0.5 * (low + high), scale=half, low=low, high=high)

    @classmethod
    def from_values(cls, name: str, values: Any) -> Scale:
        """The map built from measured values (the observed range of a field or of a coordinate)."""
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size == 0:
            raise ValueError(f"{name}: no values to measure a scale from")
        return cls.from_range(name, float(np.min(v)), float(np.max(v)))

    def apply(self, values: Any) -> np.ndarray:
        """Map the values, so that the declared range becomes ``[-1, 1]``."""
        return (np.asarray(values, dtype=np.float64) - self.centre) / self.scale

    def invert(self, values: Any) -> np.ndarray:
        """Undo :meth:`apply`: back to physical units."""
        return np.asarray(values, dtype=np.float64) * self.scale + self.centre

    def to_dict(self) -> dict[str, float | str]:
        """JSON-able form."""
        return {"name": self.name, "centre": self.centre, "scale": self.scale,
                "low": self.low, "high": self.high}


# --------------------------------------------------------------------------- the report
#: parameter set each dimensionless group needs, so a missing one can be reported with what it wanted
_GROUP_INPUTS: dict[str, tuple[str, ...]] = {
    "Re": ("velocity", "length", "density", "dynamic_viscosity"),
    "Pe": ("velocity", "length", "thermal_diffusivity"),
    "Ma": ("velocity", "sound_speed"),
    "Pr": ("kinematic_viscosity", "thermal_diffusivity"),
    "Fr": ("velocity", "length", "gravity"),
    "Kn": ("mean_free_path", "length"),
    "Gr": ("gravity", "beta", "dT", "length", "kinematic_viscosity"),
    "Ra": ("gravity", "beta", "dT", "length", "kinematic_viscosity", "thermal_diffusivity"),
    "Ri": ("gravity", "beta", "dT", "length", "velocity"),
    "Bi": ("h", "length", "conductivity"),
}


@dataclass(frozen=True)
class ConditionReport:
    """The scales, the weights and the groups a problem is conditioned with."""

    coordinates: Mapping[str, Scale]
    fields: Mapping[str, Scale]
    residuals: Mapping[str, float]
    groups: Mapping[str, float]
    missing: Mapping[str, tuple[str, ...]]
    target: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def scale(self, name: str) -> Scale:
        """The scale of a coordinate or a field, by name."""
        if name in self.coordinates:
            return self.coordinates[name]
        if name in self.fields:
            return self.fields[name]
        raise KeyError(f"{name!r} is neither a coordinate nor a field of this report; "
                       f"known: {sorted(set(self.coordinates) | set(self.fields))}")

    def weight(self, term: str) -> float:
        """The weight of one residual term."""
        if term not in self.residuals:
            raise KeyError(f"{term!r} is not a residual term of this report; known: {sorted(self.residuals)}")
        return self.residuals[term]

    def apply(self, name: str, values: Any) -> np.ndarray:
        """Scale one variable, by name."""
        return self.scale(name).apply(values)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "coordinates": {k: v.to_dict() for k, v in self.coordinates.items()},
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "residuals": dict(self.residuals),
            "groups": dict(self.groups),
            "missing": {k: list(v) for k, v in self.missing.items()},
            "target": self.target,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def format(self) -> str:
        """A few lines for a CLI: what was scaled, what was weighted, which group is missing."""
        lines = ["coordinates: " + ", ".join(f"{k} -> ({s.centre:g}, {s.scale:g})"
                                             for k, s in self.coordinates.items()),
                 "fields: " + ", ".join(f"{k} -> ({s.centre:g}, {s.scale:g})"
                                        for k, s in self.fields.items()),
                 "residual weights: " + ", ".join(f"{k}={w:.3g}" for k, w in self.residuals.items()),
                 "groups: " + (", ".join(f"{k}={v:.4g}" for k, v in self.groups.items()) or "none")]
        for name, needed in self.missing.items():
            lines.append(f"  ! {name} not computed: needs {', '.join(needed)}")
        return "\n".join(lines)


def dimensionless_groups(parameters: Mapping[str, float]) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """``(computed, missing)`` for every group the SI ``parameters`` admit.

    ``parameters`` is a mapping of SI quantities: ``velocity``, ``length``, ``density``,
    ``dynamic_viscosity``, ``kinematic_viscosity``, ``thermal_diffusivity``, ``sound_speed``,
    ``mean_free_path``, ``gravity``, ``beta``, ``dT``, ``h``, ``conductivity``. ``Reynolds`` is computed
    from ``density`` and ``dynamic_viscosity`` when those are given and from ``kinematic_viscosity``
    otherwise, and the two must agree when both are present -- a report that quietly used one and called it
    the other would be wrong in a way nobody checks.
    """
    p = {k: float(v) for k, v in parameters.items()}
    groups: dict[str, float] = {}
    missing: dict[str, tuple[str, ...]] = {}

    nu = p.get("kinematic_viscosity")
    if "density" in p and "dynamic_viscosity" in p and p["density"] > 0:
        nu_from_si = p["dynamic_viscosity"] / p["density"]
        if nu is not None and not np.isclose(nu, nu_from_si, rtol=1e-6):
            raise ValueError(
                f"kinematic_viscosity={nu:g} disagrees with dynamic_viscosity/density={nu_from_si:g}; "
                "give one of them, not two that contradict each other")
        nu = nu_from_si

    features = regime_features(
        velocity=p.get("velocity"), length=p.get("length"), nu=nu,
        sound_speed=p.get("sound_speed"), diffusivity=p.get("thermal_diffusivity"),
        mean_free_path=p.get("mean_free_path"), g=p.get("gravity", 9.81),
        beta=p.get("beta"), dT=p.get("dT"), h=p.get("h"), k=p.get("conductivity"),
    )
    groups.update({k: float(v) for k, v in features.items() if np.isfinite(v)})

    # the numbers the repository's own entry point computes (it also names the Nusselt correlation it used)
    numbers = compute_dimensionless_numbers(
        velocity=p.get("velocity"), length=p.get("length"), density=p.get("density"),
        dynamic_viscosity=p.get("dynamic_viscosity"), kinematic_viscosity=nu,
        thermal_diffusivity=p.get("thermal_diffusivity"), speed_of_sound=p.get("sound_speed"),
        thermal_conductivity=p.get("conductivity"), gravity=p.get("gravity", 9.81),
    )
    for name, value in (("Re", numbers.reynolds), ("Pr", numbers.prandtl),
                        ("Pe", numbers.peclet_heat), ("Ma", numbers.mach), ("Fr", numbers.froude)):
        if value is not None and name not in groups and np.isfinite(value):
            groups[name] = float(value)

    for name, needed in _GROUP_INPUTS.items():
        if name in groups:
            continue
        absent = _still_missing(name, p, nu)
        if absent:
            missing[name] = absent
    return groups, missing


def _still_missing(name: str, parameters: Mapping[str, float], nu: float | None) -> tuple[str, ...]:
    """The inputs one group still lacks, after the ones that can be derived (``nu``) or defaulted (``g``)."""
    needed = _GROUP_INPUTS[name]
    if name == "Re" and nu is not None:
        needed = ("velocity", "length")
    absent = []
    for key in needed:
        if key == "gravity":                       # has a default, so it is never the reason a group is missing
            continue
        if key == "kinematic_viscosity" and nu is not None:
            continue
        if key not in parameters or not np.isfinite(parameters[key]):
            absent.append(key)
    return tuple(absent)


def condition_report(
    *,
    coordinates: Mapping[str, tuple[float, float]] | None = None,
    fields: Mapping[str, tuple[float, float]] | None = None,
    residual_terms: Mapping[str, float] | None = None,
    parameters: Mapping[str, float] | None = None,
    target: float = 1.0,
) -> ConditionReport:
    """Build the scales, the residual weights and the dimensionless groups of one problem.

    ``coordinates`` and ``fields`` are ``name -> (low, high)`` in physical units: the domain of a coordinate,
    the expected range of a field. ``residual_terms`` is ``name -> characteristic magnitude`` of each term
    of the PDE -- for ``k T'' + f`` that is ``("diffusion", k * dT / L**2)`` and ``("source", f)`` -- and the
    weight returned for it is ``(target / magnitude)**2``, so a loss that sums ``weight * mean(term**2)``
    has every term of order ``target`` before training starts. ``parameters`` is what
    :func:`dimensionless_groups` reads.

    A term whose magnitude is zero or not finite is refused: its weight would be infinite, which is not a
    conditioning decision, it is a bug in the caller.
    """
    if not (coordinates or fields or residual_terms or parameters):
        raise ValueError("nothing to condition: give coordinates, fields, residual_terms or parameters")
    if not np.isfinite(target) or target <= 0:
        raise ValueError(f"target must be positive and finite; got {target}")

    diagnostics: list[Diagnostic] = []
    coord_scales = {name: Scale.from_range(name, low, high) for name, (low, high) in (coordinates or {}).items()}
    field_scales = {name: Scale.from_range(name, low, high) for name, (low, high) in (fields or {}).items()}

    weights: dict[str, float] = {}
    for name, magnitude in (residual_terms or {}).items():
        value = float(magnitude)
        if not np.isfinite(value) or value == 0.0:
            raise ValueError(f"the residual term {name!r} has magnitude {magnitude!r}: a weight of "
                             "1/0 is not a conditioning decision")
        weights[name] = float((target / abs(value)) ** 2)

    groups: dict[str, float] = {}
    missing: dict[str, tuple[str, ...]] = {}
    if parameters:
        groups, missing = dimensionless_groups(parameters)

    spread = [s.scale / max(abs(s.centre), 1e-300) for s in list(coord_scales.values()) + list(field_scales.values())
              if s.centre != 0.0]
    if spread and max(spread) > 1e3:
        diagnostics.append(explain("REGIME.MULTISCALE",
                                   note="a variable spans more than three decades around its centre; the "
                                        "affine scale is the best a single affine map can do",
                                   worst_relative_half_width=max(spread)))
    if weights:
        widest = max(weights.values()) / max(min(weights.values()), 1e-300)
        if widest > 1e6:
            diagnostics.append(explain("LINALG.ILL_CONDITIONED",
                                       note="the residual weights span more than six decades: the loss is "
                                            "still dominated by one term after weighting, so the magnitudes "
                                            "supplied probably describe different terms than intended",
                                       weight_spread=widest))
    return ConditionReport(coordinates=coord_scales, fields=field_scales, residuals=weights, groups=groups,
                           missing=missing, target=float(target), diagnostics=tuple(diagnostics))


# --------------------------------------------------------------------------- the benchmark
@dataclass(frozen=True)
class BenchmarkResult:
    """Three training runs of the same problem that differ only in how the problem was conditioned."""

    errors: Mapping[str, np.ndarray]
    final: Mapping[str, float]
    first_below: Mapping[str, float]
    tolerance: float
    steps: int
    analytic: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "errors": {k: [float(v) for v in e] for k, e in self.errors.items()},
            "final": dict(self.final),
            "first_below": dict(self.first_below),
            "tolerance": self.tolerance,
            "steps": self.steps,
            "analytic": self.analytic,
        }

    def format(self) -> str:
        """The table the issue asks for: when each arm first reached the tolerance, and where it ended."""
        lines = [f"relative L2 error against '{self.analytic}' (budget {self.steps} steps, "
                 f"tolerance {self.tolerance:g})"]
        for arm in self.errors:
            reached = self.first_below.get(arm, float("nan"))
            when = "never" if not np.isfinite(reached) else f"step {int(reached)}"
            lines.append(f"  {arm:12s} first below tolerance: {when:>12s}   final: {self.final[arm]:.3e}")
        return "\n".join(lines)


def _analytic_profile(L: float, k: float, source: float, t_left: float) -> Callable[[np.ndarray], np.ndarray]:
    """``k T'' = -f`` on ``[0, L]`` with ``T(0) = T(L) = t_left``: the exact solution the arms are scored on."""
    def profile(x: np.ndarray) -> np.ndarray:
        return t_left + source * x * (L - x) / (2.0 * k)
    return profile


def scaling_benchmark(
    *,
    steps: int = 1500,
    tolerance: float = 0.05,
    seed: int = 0,
    length: float = 0.1,
    conductivity: float = 0.15,
    source: float = 3000.0,
    ambient: float = 20.0,
    delta_t: float = 100.0,
    width: int = 32,
    device: str = "cpu",
) -> BenchmarkResult:
    """Train the same PINN three ways and report when each reached ``tolerance``.

    The problem is steady conduction in a slab, ``k T'' = -f`` on ``[0, L]`` with both ends held at
    ``ambient``, whose exact solution is ``T = ambient + f x (L - x) / (2 k)`` -- zero at both walls and
    ``f L^2 / 8k`` in the middle. The three arms are:

    ``raw``
        ``x`` in metres and ``T`` in kelvin, loss ``mean(residual**2) + mean(datum**2)``.
    ``hand``
        the textbook rescaling a person does by hand: ``x`` to ``[-1, 1]``, ``T`` divided by the
        temperature rise, and the residual divided by ``f``, all from the problem statement.
    ``auto``
        exactly what :func:`condition_report` returns from the same problem statement -- domain, expected
        range of the field, and the magnitude of each term.

    Everything else is identical: the network, the seed, the optimiser, the number of collocation and data
    points, and the learning rate. A difference between the columns is a difference in the conditioning and
    in nothing else.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    rise = source * length**2 / (8.0 * conductivity)
    profile = _analytic_profile(length, conductivity, source, ambient)

    # the problem statement, as an engineer would state it -- the same for all three arms
    report = condition_report(
        coordinates={"x": (0.0, length)},
        fields={"T": (ambient, ambient + delta_t)},
        residual_terms={"conduction": source},
        parameters={"length": length, "conductivity": conductivity, "dT": delta_t},
    )

    n_col = 64
    n_data = 16
    generator = torch.Generator().manual_seed(seed)
    x_col_phys = torch.rand(n_col, 1, generator=generator) * length
    x_data_phys = torch.rand(n_data, 1, generator=generator) * length
    t_data_phys = torch.as_tensor(profile(x_data_phys.numpy()), dtype=torch.float32)
    x_eval = torch.linspace(0.0, length, 201).reshape(-1, 1)
    t_eval = torch.as_tensor(profile(x_eval.numpy()), dtype=torch.float32)

    def make_net() -> nn.Module:
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                             nn.Linear(width, 1))

    def train(scale_x: float, scale_t: float, weight: float) -> np.ndarray:
        """One arm: scale the inputs, weight the residual, undo the scaling to score."""
        net = make_net()
        optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
        history = []
        for _ in range(int(steps)):
            optimiser.zero_grad()
            x_phys = x_col_phys.clone().requires_grad_(True)
            t = net(x_phys / scale_x) * scale_t        # physical units, as the residual needs them
            d1 = torch.autograd.grad(t, x_phys, torch.ones_like(t), create_graph=True)[0]
            d2 = torch.autograd.grad(d1, x_phys, torch.ones_like(d1), create_graph=True)[0]
            residual = conductivity * d2 + source
            loss = weight * torch.mean(residual**2)
            pred_data = net(x_data_phys / scale_x) * scale_t
            loss = loss + torch.mean((pred_data - t_data_phys) ** 2) / scale_t**2
            # both ends held at ambient, as a datum rather than as a loss weight to be guessed
            x_wall = torch.tensor([[0.0], [length]])
            walls = net(x_wall / scale_x) * scale_t
            loss = loss + torch.mean((walls - ambient) ** 2) / scale_t**2
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                prediction = net(x_eval / scale_x) * scale_t
                error = float(torch.linalg.norm(prediction - t_eval) / torch.linalg.norm(t_eval))
            history.append(error)
        return np.asarray(history)

    # raw: physical units everywhere, no weighting
    raw = train(1.0, 1.0, 1.0)
    # hand: x to [0, 1], T divided by the rise the problem statement implies, residual divided by f
    hand = train(length, rise, (1.0 / source) ** 2)
    # auto: exactly what the report says, from the same statement (the declared field range is the loose
    # one the caller gave, not the exact rise -- the point is that the report does not need it)
    auto = train(report.scale("x").scale, report.scale("T").scale, report.weight("conduction"))

    errors = {"raw": raw, "hand": hand, "auto": auto}
    first_below = {}
    for arm, history in errors.items():
        hits = np.where(history <= tolerance)[0]
        first_below[arm] = float(hits[0]) if hits.size else float("nan")
    return BenchmarkResult(errors=errors, final={k: float(v[-1]) for k, v in errors.items()},
                           first_below=first_below, tolerance=float(tolerance), steps=int(steps),
                           analytic="ambient + f x (L - x) / (2 k)")
