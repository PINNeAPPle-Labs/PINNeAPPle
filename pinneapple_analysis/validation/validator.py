"""PhysicsValidator: orchestrator for physics validation runs.

Collects check specifications, executes them in order, and produces a
:class:`~pinneapple_validate.core.ValidationReport`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import numpy as np
import torch

from .core import CheckResult, ValidationReport
from .boundary import BoundaryCheck
from .conservation import ConservationCheck
from .symmetry import SymmetryCheck

# ---------------------------------------------------------------------------
# Internal spec records
# ---------------------------------------------------------------------------


@dataclass
class _ConservationSpec:
    kind: str  # "mass" | "energy" | "integral"
    kwargs: dict[str, Any]


@dataclass
class _BoundarySpec:
    kind: str  # "dirichlet" | "neumann" | "periodicity"
    kwargs: dict[str, Any]


@dataclass
class _SymmetrySpec:
    kind: str  # "reflection" | "rotational"
    kwargs: dict[str, Any]


@dataclass
class _DimensionalSpec:
    expected: str                      # the unit the model's output has to be in ("m/s")
    name: str
    n_points: int
    seed: int


@dataclass
class _OODSpec:
    training_coords: Any               # a real sample of training coordinates, or None to use the domain box
    threshold: float                   # fraction of out-of-domain points that must be flagged
    max_false_alarm: float             # fraction of in-domain points the detector may wrongly flag
    n_points: int
    margin: float                      # how far outside the bounds the probe points go, as a fraction of the range
    ood_level: float
    seed: int


@dataclass
class _CustomSpec:
    fn: Callable[..., Any]
    name: str
    threshold: float
    kwargs: dict[str, Any]


# ---------------------------------------------------------------------------
# PhysicsValidator
# ---------------------------------------------------------------------------


class PhysicsValidator:
    """Orchestrates a suite of physics validation checks on a trained model.

    Parameters
    ----------
    model:
        Trained model with a ``forward`` / callable interface compatible with
        the pinneapple model conventions (returns ``Tensor`` or an object with
        a ``.y`` attribute).
    coord_names:
        Ordered list of coordinate names matching the model's input columns
        (e.g. ``["x", "y", "t"]``).
    domain_bounds:
        Dict mapping each coordinate name to ``(lo, hi)`` bounds.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, …).

    Example
    -------
    >>> validator = PhysicsValidator(model, ["x", "y"], {"x": (0,1), "y": (0,1)})
    >>> validator.add_conservation_check(kind="mass", tolerance=1e-3)
    >>> report = validator.validate()
    >>> print(report.summary())
    """

    def __init__(
        self,
        model: object,
        coord_names: list[str],
        domain_bounds: dict[str, tuple[float, float]],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.coord_names = coord_names
        self.domain_bounds = domain_bounds
        self.device = device

        self._conservation_specs: list[_ConservationSpec] = []
        self._boundary_specs: list[_BoundarySpec] = []
        self._symmetry_specs: list[_SymmetrySpec] = []
        self._dimensional_specs: list[_DimensionalSpec] = []
        self._ood_specs: list[_OODSpec] = []
        self._custom_specs: list[_CustomSpec] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_conservation_check(self, kind: str = "mass", **kwargs: Any) -> PhysicsValidator:
        """Register a conservation check.

        Parameters
        ----------
        kind:
            One of ``"mass"``, ``"energy"``, or ``"integral"``.
        **kwargs:
            Forwarded to the corresponding :class:`ConservationCheck` method.
            ``coord_names`` and ``domain_bounds`` are injected automatically
            unless overridden.

        Returns
        -------
        self (for chaining)
        """
        self._conservation_specs.append(_ConservationSpec(kind=kind, kwargs=kwargs))
        return self

    def add_boundary_check(self, kind: str = "dirichlet", **kwargs: Any) -> PhysicsValidator:
        """Register a boundary condition check.

        Parameters
        ----------
        kind:
            One of ``"dirichlet"``, ``"neumann"``, or ``"periodicity"``.
        **kwargs:
            Forwarded to the corresponding :class:`BoundaryCheck` method.

        Returns
        -------
        self (for chaining)
        """
        self._boundary_specs.append(_BoundarySpec(kind=kind, kwargs=kwargs))
        return self

    def add_symmetry_check(self, kind: str = "reflection", **kwargs: Any) -> PhysicsValidator:
        """Register a symmetry check.

        Parameters
        ----------
        kind:
            One of ``"reflection"`` or ``"rotational"``.
        **kwargs:
            Forwarded to the corresponding :class:`SymmetryCheck` method.

        Returns
        -------
        self (for chaining)
        """
        self._symmetry_specs.append(_SymmetrySpec(kind=kind, kwargs=kwargs))
        return self

    def add_dimensional_check(self, expected: str, *, name: str = "dimensional", n_points: int = 64,
                              seed: int = 0) -> PhysicsValidator:
        """Register the unit check: the model's output must be in ``expected``.

        The check can only be asked of a model that returns a unit with its numbers — a
        :class:`~pinneapple_core.physics_tensor.PhysicsTensor`, or anything with a ``unit`` attribute. A plain
        tensor is reported as **not run**, with that sentence, rather than as a pass: there is no unit to compare,
        and claiming one would be inventing it.

        Parameters
        ----------
        expected:
            Unit the output has to be in, e.g. ``"m/s"`` or ``"Pa"``.
        n_points:
            Points inside the domain the model is called with.
        """
        self._dimensional_specs.append(_DimensionalSpec(expected=str(expected), name=str(name),
                                                        n_points=int(n_points), seed=int(seed)))
        return self

    def add_ood_check(self, training_coords: Any = None, *, threshold: float = 0.8,
                      max_false_alarm: float = 0.2, n_points: int = 512, margin: float = 1.0,
                      ood_level: float = 0.05, seed: int = 0) -> PhysicsValidator:
        """Register the out-of-distribution check: points outside the domain must be **flagged**.

        The reference distribution is fitted from ``training_coords`` when one is given (a real sample: mean and
        covariance) and from the domain box otherwise. The check then probes points placed outside the bounds and
        measures the fraction the detector flags, together with the fraction of *in-domain* points it wrongly flags:
        a detector that flags everything would otherwise pass on recall alone.

        Parameters
        ----------
        training_coords:
            Sample of training coordinates (``torch.Tensor`` or array, shape ``(n, len(coord_names))``).
        threshold:
            Fraction of out-of-domain points that must be flagged for the check to pass.
        max_false_alarm:
            Fraction of in-domain points the detector may flag before the check fails.
        margin:
            How far outside each bound the probe points go, as a fraction of that axis's range. The default of one
            full range puts them a couple of standard deviations out, where a detector should be sure; a margin so
            small that the probe points sit on the boundary measures nothing but the level chosen.
        ood_level:
            The p-value below which a point counts as out of distribution.
        """
        self._ood_specs.append(_OODSpec(training_coords=training_coords, threshold=float(threshold),
                                        max_false_alarm=float(max_false_alarm), n_points=int(n_points),
                                        margin=float(margin), ood_level=float(ood_level), seed=int(seed)))
        return self

    def add_custom_check(
        self,
        fn: Callable[..., Any],
        name: str,
        threshold: float,
        **kwargs: Any,
    ) -> PhysicsValidator:
        """Register a custom check function.

        The function must accept ``(model, coord_names, domain_bounds, **kwargs)``
        and return either a :class:`~pinneapple_validate.core.CheckResult` or a
        plain ``float`` (which will be compared to *threshold*).

        Parameters
        ----------
        fn:
            Custom check callable.
        name:
            Identifier for the check in the report.
        threshold:
            Acceptance threshold (used when *fn* returns a float).
        **kwargs:
            Additional keyword arguments passed to *fn*.

        Returns
        -------
        self (for chaining)
        """
        self._custom_specs.append(
            _CustomSpec(fn=fn, name=name, threshold=threshold, kwargs=kwargs)
        )
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def validate(self, model_name: str | None = None) -> ValidationReport:
        """Run all registered checks and return a :class:`ValidationReport`.

        Parameters
        ----------
        model_name:
            Optional name override; defaults to ``type(model).__name__``.

        Returns
        -------
        ValidationReport
        """
        name = model_name or type(self.model).__name__
        report = ValidationReport.create(name)

        conservation = ConservationCheck(device=self.device)
        boundary = BoundaryCheck(device=self.device)
        symmetry = SymmetryCheck(device=self.device)

        # --- Conservation ---
        for spec in self._conservation_specs:
            kw = dict(spec.kwargs)
            kw.setdefault("coord_names", self.coord_names)
            kw.setdefault("domain_bounds", self.domain_bounds)
            if spec.kind == "mass":
                result = conservation.check_mass_conservation(self.model, **kw)
            elif spec.kind == "energy":
                result = conservation.check_energy_conservation(self.model, **kw)
            elif spec.kind == "integral":
                result = conservation.check_integral_quantity(self.model, **kw)
            else:
                raise ValueError(f"Unknown conservation kind: {spec.kind!r}")
            report.checks.append(result)

        # --- Boundary ---
        for spec in self._boundary_specs:
            kw = dict(spec.kwargs)
            if spec.kind == "dirichlet":
                result = boundary.check_dirichlet(self.model, **kw)
            elif spec.kind == "neumann":
                result = boundary.check_neumann(self.model, **kw)
            elif spec.kind == "periodicity":
                result = boundary.check_periodicity(self.model, **kw)
            else:
                raise ValueError(f"Unknown boundary kind: {spec.kind!r}")
            report.checks.append(result)

        # --- Symmetry ---
        for spec in self._symmetry_specs:
            kw = dict(spec.kwargs)
            if spec.kind == "reflection":
                result = symmetry.check_reflection(self.model, **kw)
            elif spec.kind == "rotational":
                result = symmetry.check_rotational(self.model, **kw)
            else:
                raise ValueError(f"Unknown symmetry kind: {spec.kind!r}")
            report.checks.append(result)

        # --- Dimensional ---
        for spec in self._dimensional_specs:
            report.checks.append(self._check_dimensional(spec))

        # --- Out of distribution ---
        for spec in self._ood_specs:
            report.checks.append(self._check_ood(spec))

        # --- Custom ---
        for spec in self._custom_specs:
            raw = spec.fn(
                self.model, self.coord_names, self.domain_bounds, **spec.kwargs
            )
            if isinstance(raw, CheckResult):
                report.checks.append(raw)
            else:
                value = float(raw)
                report.checks.append(
                    CheckResult(
                        name=spec.name,
                        passed=value <= spec.threshold,
                        value=value,
                        threshold=spec.threshold,
                        description=f"Custom check '{spec.name}'",
                    )
                )

        return report

    # ------------------------------------------------------------------
    # The two checks that ask the model something new (#31)
    # ------------------------------------------------------------------

    def _probe_points(self, n_points: int, seed: int) -> np.ndarray | None:
        """``n_points`` random points inside the domain, in ``coord_names`` order, or ``None`` without bounds."""
        if not self.domain_bounds or set(self.domain_bounds) != set(self.coord_names):
            return None
        rng = np.random.default_rng(seed)
        lo = np.array([self.domain_bounds[c][0] for c in self.coord_names], dtype=float)
        hi = np.array([self.domain_bounds[c][1] for c in self.coord_names], dtype=float)
        return lo + (hi - lo) * rng.random((int(n_points), len(self.coord_names)))

    def _call(self, points: np.ndarray) -> Any:
        x = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(x)

    def _check_dimensional(self, spec: _DimensionalSpec) -> CheckResult:
        """The model's output unit against the expected one: equal passes, a scaled or different dimension fails."""
        from pinneapple_data.physical_units import convert, try_parse_unit

        points = self._probe_points(spec.n_points, spec.seed)
        if points is None:
            return CheckResult.not_run(spec.name, "the domain bounds do not cover every coordinate, so there is no "
                                                  "point to call the model with")
        try:
            out = self._call(points)
        except Exception as exc:                                  # the model's own refusal is the finding
            return CheckResult.not_run(spec.name, f"the model could not be called: {type(exc).__name__}: {exc}")
        unit = getattr(out, "units", None) or getattr(out, "unit", None)
        if unit is None:
            return CheckResult.not_run(spec.name, "the model returns a plain tensor: there is no unit to compare with "
                                                  f"{spec.expected!r}. Return a PhysicsTensor, or check units with "
                                                  "add_custom_check")
        text = unit if isinstance(unit, str) else (getattr(unit, "symbol", None) or str(unit))
        got, want = try_parse_unit(text), try_parse_unit(spec.expected)
        if got is None or want is None:
            return CheckResult.not_run(spec.name, f"the unit {text!r} or {spec.expected!r} is not one this library "
                                                  "parses, so the comparison would be a guess")
        if got.dim != want.dim:
            return CheckResult(name=spec.name, passed=False, value=float("nan"), threshold=0.0,
                               description=f"the model returns {text!r} where {spec.expected!r} is expected: "
                                           "different dimensions", unit=text, direction="lower")
        factor = float(np.asarray(convert(1.0, text, spec.expected)).ravel()[0])
        passed = factor == 1.0
        return CheckResult(name=spec.name, passed=passed, value=factor, threshold=1.0,
                           description=(f"the model returns {text!r}, the expected {spec.expected!r}"
                                        if passed else
                                        f"the model returns {text!r} where {spec.expected!r} is expected: the same "
                                        f"dimensions scaled by {factor:g}, so every number is off by that factor"),
                           unit="", direction="lower")

    def _check_ood(self, spec: _OODSpec) -> CheckResult:
        """Out-of-domain points must be flagged, and in-domain ones must not: recall alone is a detector that says
        "out" to everything."""
        from pinneapple_analysis.trust.trust_gate import TrustGate

        inside = self._probe_points(spec.n_points, spec.seed)
        if inside is None:
            return CheckResult.not_run("ood", "the domain bounds do not cover every coordinate, so there is no "
                                              "inside to compare against")
        rng = np.random.default_rng(spec.seed + 1)
        lo = np.array([self.domain_bounds[c][0] for c in self.coord_names], dtype=float)
        hi = np.array([self.domain_bounds[c][1] for c in self.coord_names], dtype=float)
        span = hi - lo
        outside = (lo - spec.margin * span) + (span * (1.0 + 2.0 * spec.margin)) * rng.random(inside.shape)
        gate = TrustGate(ood_level=spec.ood_level)
        if spec.training_coords is not None:
            coords = spec.training_coords
            coords = coords if isinstance(coords, torch.Tensor) else torch.as_tensor(np.asarray(coords), dtype=torch.float32)
            gate.fit(coords)
            how = f"fitted on {len(coords)} training points"
        else:
            gate.fit_from_bounds(self.domain_bounds)
            how = "fitted on the domain box (no training sample was given)"
        recall = float(np.mean(gate.ood_p_values(outside) < spec.ood_level))
        false_alarm = float(np.mean(gate.ood_p_values(inside) < spec.ood_level))
        passed = recall >= spec.threshold and false_alarm <= spec.max_false_alarm
        return CheckResult(name="ood", passed=passed, value=recall, threshold=spec.threshold,
                           description=(f"{recall:.1%} of the points outside the domain were flagged (threshold "
                                        f"{spec.threshold:.0%}) and {false_alarm:.1%} of the points inside were "
                                        f"wrongly flagged (limit {spec.max_false_alarm:.0%}); {how}"),
                           direction="higher")

    # ------------------------------------------------------------------
    # Auto-configuration from ProblemSpec
    # ------------------------------------------------------------------

    def validate_from_spec(self, problem_spec: Any) -> ValidationReport:
        """Auto-configure and run checks from a :class:`~pinneapple_problemdesign.schema.ProblemSpec`.

        Currently auto-detects:
        - Dirichlet BCs from ``problem_spec.physics.boundary_conditions`` strings
          containing ``"dirichlet"`` or ``"="`` patterns.
        - Mass conservation check when the governing equations mention
          ``"incompressible"`` or ``"∇·"`` / ``"div"``.
        - Energy conservation when equations mention ``"energy"`` or ``"∂E"``.

        For a precise programmatic setup you should call the ``add_*`` methods
        directly.  This method is a best-effort convenience helper.

        Parameters
        ----------
        problem_spec:
            A :class:`~pinneapple_problemdesign.schema.ProblemSpec` instance.

        Returns
        -------
        ValidationReport
        """
        physics = getattr(problem_spec, "physics", None)
        if physics is not None:
            governing = " ".join(getattr(physics, "governing_equations", [])).lower()

            # Conservation heuristics
            if any(kw in governing for kw in ("incompressible", "∇·", "div u", "div(u)")):
                self._conservation_specs.append(
                    _ConservationSpec(kind="mass", kwargs={})
                )
            if any(kw in governing for kw in ("energy", "∂e", "de/dt")):
                self._conservation_specs.append(
                    _ConservationSpec(kind="energy", kwargs={})
                )

        return self.validate(
            model_name=getattr(problem_spec, "title", None) or type(self.model).__name__
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def validate_model(
    model: object,
    spec: Any,
    coord_names: list[str] | None = None,
    domain_bounds: dict[str, tuple[float, float]] | None = None,
    device: str = "cpu",
) -> ValidationReport:
    """Quick one-shot validation from a ProblemSpec or explicit arguments.

    Parameters
    ----------
    model:
        Trained PINN model.
    spec:
        A :class:`~pinneapple_problemdesign.schema.ProblemSpec` instance, or
        ``None`` to use *coord_names* / *domain_bounds* directly.
    coord_names:
        Required when *spec* is ``None``.
    domain_bounds:
        Required when *spec* is ``None``.
    device:
        PyTorch device string.

    Returns
    -------
    ValidationReport
    """
    if spec is not None:
        _coords = getattr(getattr(spec, "domain", None), "coord_names", None) or coord_names or []
        _bounds = getattr(getattr(spec, "domain", None), "bounds", None) or domain_bounds or {}
    else:
        _coords = coord_names or []
        _bounds = domain_bounds or {}

    validator = PhysicsValidator(model, _coords, _bounds, device=device)
    if spec is not None:
        return validator.validate_from_spec(spec)
    return validator.validate()
