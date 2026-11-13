"""Gates on a *model*, not on its data: accuracy, conservation, boundary conditions and the PDE residual.

The gates in :mod:`pinneapple_data.schema.gates` ask whether a dataset is well-formed. These ask whether a trained
surrogate is worth believing, and they are the half the mapping calls "model-side": a network can reproduce a field
to three digits and still violate the equation it claims to solve, drift the quantity that should be conserved, or
ignore the wall it was given.

Four gates, each with a number and a tolerance:

``accuracy``
    relative L2 against the reference over the held-out region.
``conservation``
    the spatial integral of a conserved field, along the time axis, on the **prediction** -- and the same measure on
    the data. Reporting only the model's drift hides the case where the reference itself leaks; reporting both lets
    a reviewer see which of the two moved.
``boundary``
    for every Dirichlet condition the bundle declares on a structured bundle's edge (``region="x_low"``), the mean
    deviation of the predicted values there from the declared value, relative to the field's own scale.
``residual``
    the governing equation's residual evaluated on the prediction by finite differences, normalised by the size of
    its own terms, so the number is "how much of the equation's own magnitude is left unbalanced" rather than a
    number whose size depends on the units.

Nothing here is imported from the data layer: a bundle is duck-typed (``.manifest``, ``.field_array``, ``.axes``,
``.array``), so this module keeps the property the rest of the package has -- it holds no physics of its own and no
dependency it does not need.

    >>> from pinneapple_veriphysics.surrogate_gates import run_surrogate_gates
    >>> report = run_surrogate_gates(predicted, reference=data, field="density", conserved="density",
    ...                              equation="burgers", parameters={"nu": 0.01})
    >>> print(report.summary())

A gate that cannot be asked (no boundary condition declared, no equation named) is "not run", never "passed" --
the same rule the evidence ledger applies to everything else in this package.
"""
from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as _dc_field
from typing import Any

import numpy as np

__all__ = ["SurrogateGateResult", "SurrogateGateReport", "run_surrogate_gates", "RESIDUALS",
           "advection_residual", "burgers_residual", "diffusion_residual", "relative_l2"]


class NotApplicable(Exception):
    """The model cannot be asked this question (no BCs declared, no equation named)."""


def _cut(array: np.ndarray, region: Any) -> np.ndarray:
    return array if region is None else array[region]


def relative_l2(computed: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(computed, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: prediction {a.shape}, reference {b.shape}")
    denominator = float(np.linalg.norm(b.ravel()))
    if denominator == 0.0:
        raise ValueError("the reference is identically zero: a relative L2 has no meaning")
    return float(np.linalg.norm((a - b).ravel()) / denominator)


# ── the residual operators, by named equation ────────────────────────────────────────────────────────────────
def _grid(bundle: Any) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(bundle.axis_array("t"), dtype=np.float64)
    x = np.asarray(bundle.axis_array("x"), dtype=np.float64)
    return t, x


def _derivatives(field: np.ndarray, t: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """Second-order finite differences on the interior; the edges are excluded from the norm, not wrapped."""
    u = np.asarray(field, dtype=np.float64)
    dt = np.gradient(t) if len(t) > 2 else np.array([t[1] - t[0]] * len(t))
    dx = np.gradient(x)
    u_t = np.gradient(u, axis=0) / dt[:, None]
    u_x = np.gradient(u, axis=1) / dx[None, :]
    u_xx = np.gradient(u_x, axis=1) / dx[None, :]
    u_tt = np.gradient(u_t, axis=0) / dt[:, None]
    return {"u_t": u_t, "u_x": u_x, "u_xx": u_xx, "u_tt": u_tt}


def advection_residual(field: np.ndarray, t: np.ndarray, x: np.ndarray, *, c: float = 1.0) -> np.ndarray:
    """``u_t + c u_x``."""
    d = _derivatives(field, t, x)
    return d["u_t"] + c * d["u_x"]


def burgers_residual(field: np.ndarray, t: np.ndarray, x: np.ndarray, *, nu: float = 0.01) -> np.ndarray:
    """``u_t + u u_x - nu u_xx``."""
    u = np.asarray(field, dtype=np.float64)
    d = _derivatives(u, t, x)
    return d["u_t"] + u * d["u_x"] - nu * d["u_xx"]


def diffusion_residual(field: np.ndarray, t: np.ndarray, x: np.ndarray, *, alpha: float = 0.1) -> np.ndarray:
    """``u_t - alpha u_xx``."""
    d = _derivatives(field, t, x)
    return d["u_t"] - alpha * d["u_xx"]


def wave_residual(field: np.ndarray, t: np.ndarray, x: np.ndarray, *, c: float = 1.0) -> np.ndarray:
    """``u_tt - c^2 u_xx``."""
    d = _derivatives(field, t, x)
    return d["u_tt"] - (c ** 2) * d["u_xx"]


#: The equations a bundle from PDEBench or a solver case usually names. A caller with a different one passes
#: ``residual=`` directly -- this library will not guess an equation from prose.
RESIDUALS: dict[str, Callable[..., np.ndarray]] = {
    "advection": advection_residual,
    "linear_advection": advection_residual,
    "burgers": burgers_residual,
    "diffusion": diffusion_residual,
    "heat": diffusion_residual,
    "wave": wave_residual,
}


def _interior(a: np.ndarray) -> np.ndarray:
    return a[1:-1, 1:-1] if a.ndim == 2 else a


# ── the report ──────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class SurrogateGateResult:
    check_id: str
    passed: bool | None
    value: float | None = None
    tolerance: float | None = None
    detail: str = ""
    tool: str = "pinneapple_veriphysics.surrogate_gates"

    @property
    def status(self) -> str:
        return "not run" if self.passed is None else ("passed" if self.passed else "failed")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status}


@dataclass
class SurrogateGateReport:
    """What was asked of a model and what came back."""

    model: str = ""
    bundle_id: str = ""
    field: str = ""
    results: list[SurrogateGateResult] = _dc_field(default_factory=list)

    def get(self, check_id: str) -> SurrogateGateResult | None:
        return next((r for r in self.results if r.check_id == check_id), None)

    @property
    def ok(self) -> bool:
        ran = [r for r in self.results if r.passed is not None]
        return bool(ran) and all(r.passed for r in ran)

    @property
    def failed(self) -> list[str]:
        return [r.check_id for r in self.results if r.passed is False]

    def summary(self) -> str:
        width = max((len(r.check_id) for r in self.results), default=8)
        lines = [f"surrogate gates for {self.model or 'a model'} on {self.bundle_id} (field {self.field})"]
        for r in self.results:
            value = "" if r.value is None else f"{r.value:.3e}"
            tol = "" if r.tolerance is None else f" tol {r.tolerance:.1e}"
            lines.append(f"  {r.check_id:<{width}}  {r.status:<8} {value:<11}{tol} {r.detail}")
        lines.append(f"  -> {'ok' if self.ok else 'NOT ok'}"
                     + (f"; failed: {', '.join(self.failed)}" if self.failed else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "bundle_id": self.bundle_id, "field": self.field, "ok": self.ok,
                "failed": self.failed, "gates": [r.to_dict() for r in self.results]}

    def record(self, ledger, *, inputs_sha256: str, tool_version: str = "1") -> list[Any]:
        """Seal every gate that ran into an ``EvidenceLedger`` (a gate that did not run gets no entry)."""
        stamp = _now()
        out = []
        for r in self.results:
            if r.passed is None:
                continue
            out.append(ledger.record(
                f"surrogate/{r.check_id}", passed=bool(r.passed), inputs_sha256=inputs_sha256,
                outputs={"value": r.value, "tolerance": r.tolerance, "detail": r.detail, "model": self.model,
                         "field": self.field},
                started_at=stamp, finished_at=stamp, name=r.check_id, category="verification",
                tool=r.tool, tool_version=tool_version))
        return out


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── the gates ─────────────────────────────────────────────────────────────────────────────────────────────
def _accuracy(predicted: Any, reference: Any, field: str, tolerance: float,
              region: Any = None) -> SurrogateGateResult:
    if reference is None:
        raise NotApplicable("no reference was given")
    pred = _cut(np.asarray(predicted.field_array(field), dtype=np.float64), region)
    ref = _cut(np.asarray(reference if not hasattr(reference, "field_array") else reference.field_array(field),
                          dtype=np.float64), region)
    err = relative_l2(pred, ref)
    where = "the compared region" if region is None else f"the compared region ({pred.shape[0]} slices)"
    return SurrogateGateResult("accuracy", err <= tolerance, err, tolerance, f"relative L2 of '{field}' over {where}")


def _conserved_integral(bundle: Any, field: str, axis: str) -> np.ndarray:
    values = np.asarray(bundle.field_array(field), dtype=np.float64)
    coords = np.asarray(bundle.axis_array(axis), dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != coords.shape[0]:
        raise NotApplicable(f"'{field}' is {values.shape} on axis '{axis}' of {coords.shape}: not an integrable "
                            "grid field")
    return np.trapezoid(values, coords, axis=-1)


def _conservation(predicted: Any, reference: Any, field: str, tolerance: float,
                  axis: str, region: Any = None) -> SurrogateGateResult:
    if "t" not in predicted.axes:
        raise NotApplicable("a conservation check needs a time axis")
    integral = _conserved_integral(predicted, field, axis)
    if region is not None:
        integral = integral[region]
    i0 = float(integral[0])
    scale = max(abs(i0), float(np.max(np.abs(integral))))
    if scale <= 1e-12 * max(1.0, float(np.max(np.abs(predicted.field_array(field))))):
        raise NotApplicable("the integral of the prediction is zero to rounding: its drift is not a measurement")
    model_drift = float(np.max(np.abs(integral - i0)) / scale)
    detail = f"relative drift of the integral of '{field}' over {len(integral)} steps"
    if reference is not None:
        try:
            ref = _conserved_integral(reference, field, axis)
            ref = ref if region is None else ref[region]
            ref_scale = max(abs(float(ref[0])), float(np.max(np.abs(ref))))
            detail += f"; the data drifts {float(np.max(np.abs(ref - ref[0])) / ref_scale):.1e} over the same steps"
        except NotApplicable:
            pass
    return SurrogateGateResult("conservation", model_drift <= tolerance, model_drift, tolerance, detail)


def _boundary(predicted: Any, field: str, tolerance: float) -> SurrogateGateResult:
    """Dirichlet conditions on the edges of a structured bundle, checked on the prediction."""
    manifest = predicted.manifest
    declared = [bc for bc in manifest.physics.boundary_conditions if bc.kind == "dirichlet"]
    if not declared:
        raise NotApplicable("the bundle declares no Dirichlet boundary condition to check against")
    values = np.asarray(predicted.field_array(field), dtype=np.float64)
    scale = float(np.max(np.abs(values))) or 1.0
    problems: list[str] = []
    worst = 0.0
    checked = 0
    for bc in declared:
        side = _edge_index(values, bc.region, predicted.axes)
        if side is None:
            problems.append(f"region '{bc.region}' does not name an edge of this bundle, so '{bc.field}' there is "
                            "unverified")
            continue
        target = float(bc.value if bc.value is not None else 0.0)
        deviation = float(np.mean(np.abs(values[side] - target))) / scale
        worst = max(worst, deviation)
        checked += 1
    passed = not problems and checked > 0 and worst <= tolerance
    return SurrogateGateResult("boundary", passed, worst, tolerance,
                               "; ".join(problems) or f"{checked} Dirichlet condition(s) checked on '{field}'")


def _edge_index(values: np.ndarray, region: str, axes: Sequence[str]) -> Any | None:
    """Turn ``"x_low"`` / ``"x_high"`` / ``"t_low"`` / ``"t_high"`` into an index of the field."""
    text = (region or "").lower()
    axis, _, side = text.partition("_")
    if axis not in axes or side not in ("low", "high"):
        return None
    position = list(axes).index(axis)
    index: list[Any] = [slice(None)] * values.ndim
    index[position] = 0 if side == "low" else -1
    return tuple(index)


def _residual(predicted: Any, field: str, equation: str, parameters: Mapping[str, float],
              tolerance: float, residual: Callable[..., np.ndarray] | None,
              region: Any = None) -> SurrogateGateResult:
    if residual is None:
        if not equation:
            raise NotApplicable("no equation was named, and this library will not infer one from the prose in "
                                "physics.governing")
        key = equation.strip().lower().split("(")[0].strip()
        if key not in RESIDUALS:
            raise NotApplicable(f"'{equation}' is not one of the equations this module discretises "
                                f"({', '.join(sorted(RESIDUALS))}); pass residual= with your own operator")
        residual = RESIDUALS[key]
    values = _cut(np.asarray(predicted.field_array(field), dtype=np.float64), region)
    if values.ndim != 2:
        raise NotApplicable(f"'{field}' is {values.shape}: the residual operators work on a (time, space) field")
    t, x = _grid(predicted)
    if region is not None and isinstance(region, slice):
        t = t[region]
    if len(t) != values.shape[0]:
        raise NotApplicable("the region and the time axis disagree: the residual needs the coordinates of the part "
                            "it differentiates")
    r = np.asarray(residual(values, t, x, **dict(parameters)), dtype=np.float64)
    if r.shape != values.shape:
        raise ValueError(f"the residual operator returned {r.shape} for a field of {values.shape}")
    magnitude = float(np.linalg.norm(_interior(r).ravel()))
    scale = float(np.linalg.norm(_interior(np.gradient(values, axis=0)).ravel())) + \
        float(np.linalg.norm(_interior(np.gradient(values, axis=1)).ravel()))
    if scale == 0.0:
        raise NotApplicable("the field has no gradient to compare the residual against (a constant field)")
    value = magnitude / scale
    name = getattr(residual, "__name__", "residual")
    return SurrogateGateResult("residual", value <= tolerance, value, tolerance,
                               f"{name} on the prediction, normalised by the size of its own terms")


def run_surrogate_gates(predicted: Any, *, reference: Any = None, field: str, model: str = "",
                        conserved: str | None = None, conserved_axis: str = "x",
                        equation: str = "", parameters: Mapping[str, float] | None = None,
                        residual: Callable[..., np.ndarray] | None = None,
                        gates: Sequence[str] = ("accuracy", "conservation", "boundary", "residual"),
                        region: Any = None, tolerance: float = 5e-2, conservation_tolerance: float = 1e-2,
                        boundary_tolerance: float = 5e-2, residual_tolerance: float = 1e-1) -> SurrogateGateReport:
    """Run the model-side gates over a prediction (a bundle whose field holds the model's answer).

    ``region`` restricts accuracy, conservation and the residual to a slice of the time axis -- the held-out part,
    which is the only part a model's score means anything on. Boundary conditions are edge conditions and ignore
    it."""
    report = SurrogateGateReport(model=model, bundle_id=getattr(predicted.manifest, "id", ""), field=field)
    for gate in gates:
        try:
            if gate == "accuracy":
                result = _accuracy(predicted, reference, field, tolerance, region)
            elif gate == "conservation":
                result = _conservation(predicted, reference, conserved or field, conservation_tolerance,
                                       conserved_axis, region)
            elif gate == "boundary":
                result = _boundary(predicted, field, boundary_tolerance)
            elif gate == "residual":
                result = _residual(predicted, field, equation, parameters or {}, residual_tolerance, residual, region)
            else:
                raise ValueError(f"unknown gate '{gate}'; known: accuracy, conservation, boundary, residual")
        except NotApplicable as exc:
            result = SurrogateGateResult(gate, None, None, None, str(exc))
        report.results.append(result)
    return report
