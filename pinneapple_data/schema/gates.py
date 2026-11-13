"""Verification gates for a bundle: the checks a physics dataset has to pass before anything is trained on it.

Each gate answers one question about the *data*, not about a model, and produces a numeric value and a tolerance
so the verdict is arithmetic rather than opinion:

=================  ========================================================================================
``integrity``      do the manifest's references resolve, and does the digest still match the bytes?
``units``          does every field and axis declare a unit, and do the spatial axes agree with the geometry?
``location``       does every field say nodal or cell, and does its shape agree with the mesh?
``frame``          are the spatial axes strictly monotonic, uniformly spaced, and coverage-complete?
``provenance``     is there a source, a licence, a citation and a hash of every input?
``conservation``   does the integral of a conserved field stay inside tolerance along the time axis?
``reference``      how far is a field from a reference array (relative L2), and is it inside tolerance?
=================  ========================================================================================

A gate that a bundle cannot satisfy *by its nature* (there is no conserved field to integrate, no reference to
compare with) is reported as "not run", never as "passed": the rule the whole verification layer is built on
(:mod:`pinneapple_veriphysics.ledger`). :meth:`GateReport.record` seals the verdicts into an
:class:`~pinneapple_veriphysics.ledger.EvidenceLedger`, so the words "passed" in a report exist only when the
record of the run does.

    >>> from pinneapple_data.schema import run_gates
    >>> report = run_gates(bundle, conserved="density")
    >>> print(report.summary())                       # one line per gate, with its number
    >>> report.apply(bundle)                          # write CheckResults into bundle.manifest.verification
"""
from __future__ import annotations

import datetime as _dt
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any, Literal

import numpy as np

from .bundle import PhysicsDataBundle
from .models import CheckResult, Verification

__all__ = ["GateResult", "GateReport", "run_gates", "relative_l2", "conservation_drift", "GATES"]

GateName = Literal["integrity", "units", "location", "frame", "provenance", "conservation", "reference"]
GATES: tuple[str, ...] = ("integrity", "units", "location", "frame", "provenance", "conservation", "reference")

DEFAULT_TOLERANCES: dict[str, float] = {
    "integrity": 0.0,          # exact: the digest either matches or it does not
    "units": 0.0,              # exact: every declaration present and dimensionally consistent
    "location": 0.0,           # exact
    # max |dx - mean(dx)| / mean(dx) over an axis. 1e-5 rather than 1e-9 because the archives store coordinates as
    # float32: a 64-point linspace written as float32 already differs from perfectly uniform by ~2e-6 (measured on
    # PDEBench's x-coordinate), and a tolerance below that would report every real dataset as non-uniform.
    "frame_uniformity": 1e-5,
    "provenance": 0.0,
    "conservation": 1e-2,      # relative drift of the integral over the record
    "reference": 1e-2,         # relative L2 error against the reference field
}


class GateNotApplicable(Exception):
    """The bundle cannot be asked this question (no conserved field, no reference, no time axis)."""


@dataclass
class GateResult:
    """One gate outcome, with the number it was decided on."""

    check_id: str
    passed: bool | None            # None == "not run"
    value: float | None = None
    tolerance: float | None = None
    unit: str = "1"
    detail: str = ""
    tool: str = "pinneapple_data.schema.gates"

    @property
    def status(self) -> str:
        return "not run" if self.passed is None else ("passed" if self.passed else "failed")

    def to_check(self) -> CheckResult:
        return CheckResult(check_id=self.check_id, tool=self.tool, passed=self.passed, value=self.value,
                           tolerance=self.tolerance, unit=self.unit, detail=self.detail)


@dataclass
class GateReport:
    """Every gate that ran, in order, with the bundle they ran on identified by its digest."""

    bundle_id: str = ""
    bundle_digest: str = ""
    results: list[GateResult] = _dc_field(default_factory=list)

    def get(self, check_id: str) -> GateResult | None:
        for r in self.results:
            if r.check_id == check_id:
                return r
        return None

    @property
    def ok(self) -> bool:
        """True when at least one gate ran and every gate that ran passed (``not run`` does not fail it)."""
        ran = [r for r in self.results if r.passed is not None]
        return bool(ran) and all(r.passed for r in ran)

    @property
    def failed(self) -> list[str]:
        return [r.check_id for r in self.results if r.passed is False]

    def to_verification(self) -> Verification:
        return Verification(checks=[r.to_check() for r in self.results],
                            required=[r.check_id for r in self.results if r.passed is not None])

    def apply(self, bundle: PhysicsDataBundle) -> PhysicsDataBundle:
        """Write the outcomes into the bundle's manifest (``bundle.manifest.verification``)."""
        bundle.manifest.verification = self.to_verification()
        return bundle

    def record(self, ledger, *, inputs_sha256: str | None = None, tool: str = "pinneapple.schema-gates",
               tool_version: str = "1") -> list[Any]:
        """Seal every gate that ran into an ``EvidenceLedger`` (a "not run" gate gets no entry).

        ``inputs_sha256`` defaults to the bundle digest of the report, which is what the checks actually read."""
        digest = inputs_sha256 or self.bundle_digest
        out = []
        for r in self.results:
            if r.passed is None:
                continue
            started = _utc_now()
            out.append(ledger.record(
                f"bundle/{r.check_id}", passed=bool(r.passed), inputs_sha256=digest,
                outputs={"value": r.value, "tolerance": r.tolerance, "unit": r.unit, "detail": r.detail,
                         "bundle": self.bundle_id},
                started_at=started, finished_at=_utc_now(), name=r.check_id, category="verification",
                tool=tool, tool_version=tool_version))
        return out

    def summary(self) -> str:
        width = max((len(r.check_id) for r in self.results), default=8)
        lines = [f"gates for {self.bundle_id} (digest {self.bundle_digest[:12]}...)"]
        for r in self.results:
            number = "" if r.value is None else f"{r.value:.3e}"
            tol = "" if r.tolerance is None else f" tol {r.tolerance:.1e}"
            lines.append(f"  {r.check_id:<{width}}  {r.status:<8} {number:<11}{tol} {r.detail}")
        lines.append(f"  -> {'ok' if self.ok else 'NOT ok'}" + (f"; failed: {', '.join(self.failed)}" if self.failed else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"bundle_id": self.bundle_id, "bundle_digest": self.bundle_digest,
                "ok": self.ok,
                "gates": [{"check_id": r.check_id, "status": r.status, "value": r.value, "tolerance": r.tolerance,
                           "unit": r.unit, "detail": r.detail} for r in self.results]}


# ── the numbers the gates are decided on ────────────────────────────────────────────────────────────────────
def relative_l2(computed: np.ndarray, reference: np.ndarray) -> float:
    """``||a - b||_2 / ||b||_2`` over the common shape; the error measure every operator benchmark reports."""
    a = np.asarray(computed, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: computed {a.shape}, reference {b.shape}")
    denom = float(np.linalg.norm(b.ravel()))
    if denom == 0.0:
        raise ValueError("reference is identically zero: relative L2 is undefined")
    return float(np.linalg.norm((a - b).ravel()) / denom)


def conservation_drift(bundle: PhysicsDataBundle, field: str, *, axis: str = "x") -> float:
    """Relative drift of the spatial integral of ``field`` from its first time step to its last.

    Trapezoid in the spatial coordinate, then ``max_t |I(t) - I(0)| / |I(0)|``. This is the check that catches a
    solver (or a dataset) that is not conserving the quantity it says it conserves, which no pointwise error
    metric shows."""
    values = np.asarray(bundle.field_array(field), dtype=np.float64)
    if axis not in bundle.axes:
        raise GateNotApplicable(f"no axis '{axis}' to integrate over")
    coords = np.asarray(bundle.axis_array(axis), dtype=np.float64)
    if values.shape[-1] != coords.shape[0]:
        raise GateNotApplicable(f"field '{field}' has trailing size {values.shape[-1]}, axis '{axis}' has "
                                f"{coords.shape[0]}: not a nodal grid field to integrate")
    integral = np.trapezoid(values, coords, axis=-1)
    if integral.ndim > 1:                      # a multi-axis field: integrate the rest out spatially
        integral = integral.reshape(integral.shape[0], -1).sum(axis=1)
    if not integral.size:
        raise GateNotApplicable(f"field '{field}' has no time axis to integrate along")
    i0 = float(integral[0])
    scale = max(abs(i0), float(np.max(np.abs(integral))))
    if scale <= 1e-12 * max(1.0, float(np.max(np.abs(values)))):
        raise GateNotApplicable(
            f"the integral of '{field}' is zero to rounding at every step (a sign-changing field with a zero mean, "
            "such as a travelling wave): the drift of a zero is not a measurement")
    return float(np.max(np.abs(integral - i0)) / scale)


# ── the gates ─────────────────────────────────────────────────────────────────────────────────────────────
def _integrity(bundle: PhysicsDataBundle, tol: Mapping[str, float]) -> tuple[GateResult, dict]:
    report = bundle.verify()
    return GateResult("integrity", report["ok"], 0.0 if report["ok"] else float(len(report["problems"])),
                      tol["integrity"], "1",
                      "; ".join(report["problems"]) or f"digest {report['digest'][:12]}... matches"), report


def _units(bundle: PhysicsDataBundle, tol: Mapping[str, float]) -> tuple[GateResult, dict]:
    problems: list[str] = []
    notes: list[str] = []
    length_units: set[str] = set()
    from .models import UNKNOWN_UNIT

    for ax in bundle.manifest.mesh.axes:
        if not ax.unit:
            problems.append(f"axis '{ax.name}' has no unit")
            continue
        if ax.unit == UNKNOWN_UNIT:          # an imported case whose length unit nobody recorded: a verdict, not a
            problems.append(f"axis '{ax.name}' has an unknown unit")      # crash from the unit parser
            continue
        if ax.name != "t":
            from ..physical_units import parse_unit
            length_units.add(str(parse_unit(ax.unit).dim))
    if len(length_units) > 1:
        problems.append(f"spatial axes do not share a dimension: {sorted(length_units)}")
    if "t" in bundle.axes:
        from ..physical_units import parse_unit
        t_unit = bundle.manifest.mesh.axis("t").unit
        t_dim = parse_unit(t_unit).dim
        time_dim, dimensionless = (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0)
        if t_dim not in (time_dim, dimensionless):
            problems.append(f"time axis is in '{t_unit}', which is neither a time unit nor dimensionless")
        elif t_dim == dimensionless:
            notes.append("the time axis is dimensionless (a non-dimensionalised case)")
    for f in bundle.manifest.fields.variables:
        if not f.unit:
            problems.append(f"field '{f.name}' has no unit")
        elif f.unit == "unknown":
            problems.append(f"field '{f.name}' has an unknown unit: the source did not record one (a solver deck in "
                            "model units), so no conversion or comparison is possible until it is given")
    detail = "; ".join(problems) if problems else (
        "every declaration is present and consistent" + (f" ({'; '.join(notes)})" if notes else ""))
    return GateResult("units", not problems, float(len(problems)), tol["units"], "1", detail), {}


def _location(bundle: PhysicsDataBundle, tol: Mapping[str, float]) -> tuple[GateResult, dict]:
    """Every field says whether it lives on nodes or cells, and its shape agrees with that convention.

    On a structured grid both cell conventions occur in the wild -- a cell field with one value per cell
    (``nx - 1`` on that axis, the finite-volume convention) and one with a value per cell *centre* reusing an
    ``nx``-long axis. Both are accepted; a shape that matches neither is the error this gate exists for. On an
    unstructured mesh the row counts are the contract."""
    problems: list[str] = []
    notes: list[str] = []
    mesh = bundle.manifest.mesh
    for f in bundle.manifest.fields.variables:
        if f.location not in ("nodal", "cell"):
            problems.append(f"field '{f.name}' has location '{f.location}'")
            continue
        body = tuple(f.data.shape[:-1]) if f.is_vector else tuple(f.data.shape)
        if mesh.kind == "unstructured":
            n = mesh.n_nodes if f.location == "nodal" else mesh.n_cells
            if n is not None and (not body or body[-1] != n):
                problems.append(f"field '{f.name}' is {f.location} but its last axis is "
                                f"{body[-1] if body else 'missing'} and the mesh has {n}")
            continue
        if mesh.kind != "grid" or not f.axes:
            continue
        axis_len = [int(mesh.axis(a).data.shape[0]) for a in f.axes if mesh.axis(a).data is not None]
        nodal_shape = tuple(axis_len)
        cell_shapes = {tuple(n - 1 if i == len(axis_len) - 1 else n for i, n in enumerate(axis_len))}
        if f.location == "nodal" and body != nodal_shape:
            problems.append(f"field '{f.name}' is nodal but has {body} instead of {nodal_shape}")
        elif f.location == "cell" and body not in ({nodal_shape} | cell_shapes):
            problems.append(f"field '{f.name}' is cell-centred but has {body}, matching neither the cell count "
                            f"{sorted(cell_shapes)[0]} nor the axis lengths {nodal_shape}")
        elif f.location == "cell" and body == nodal_shape:
            notes.append(f"'{f.name}' reuses an {nodal_shape[-1]}-long axis for cell centres")
    detail = "; ".join(problems) if problems else (
        "every field declares a location consistent with the mesh" + (f" ({'; '.join(notes)})" if notes else ""))
    return GateResult("location", not problems, float(len(problems)), tol["location"], "1", detail), {}


def _float32_noise(coords: np.ndarray, dtype: np.dtype) -> float | None:
    """How non-uniform a *perfectly* uniform axis in this dtype can look.

    A coordinate stored as float32 is rounded to ~eps*|x|, so its neighbour differences differ from the true
    spacing by ~2*eps*|x|; relative to the spacing that is ``2 * eps * max|x| / mean(spacing)``. Comparing a float32
    axis against a tolerance below that reports the storage format as a geometry error."""
    if np.dtype(dtype).itemsize >= 8 or coords.size < 2:
        return None
    spacing = abs(float(np.mean(np.diff(coords.astype(np.float64)))))
    if spacing == 0.0:
        return None
    return 2.0 * float(np.finfo(dtype).eps) * float(np.max(np.abs(coords))) / spacing


def _frame(bundle: PhysicsDataBundle, tol: Mapping[str, float]) -> tuple[GateResult, dict]:
    """Strictly monotonic axes, uniformly spaced to the precision their dtype can hold."""
    if not bundle.axes:
        raise GateNotApplicable("this mesh has no coordinate axes (an unstructured mesh): frame is a property of a "
                                "structured grid, so there is nothing to check rather than something that passed")
    worst = 0.0
    noise_max = 0.0
    problems: list[str] = []
    notes: list[str] = []
    for name in bundle.axes:
        ref = bundle.manifest.mesh.axis(name).data
        if ref is None:
            problems.append(f"axis '{name}' has no coordinates")
            continue
        coords = np.asarray(bundle.array(ref.key))
        if coords.ndim != 1 or coords.size < 2:
            problems.append(f"axis '{name}' has {coords.shape}: expected a 1-D coordinate array of length >= 2")
            continue
        values = coords.astype(np.float64)
        diffs = np.diff(values)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            problems.append(f"axis '{name}' is not strictly monotonic: a flipped or shuffled axis changes every "
                            "spatial comparison downstream")
            continue
        if np.all(diffs < 0):
            notes.append(f"'{name}' decreases")
        mean = abs(float(diffs.mean()))
        worst = max(worst, float(np.max(np.abs(np.abs(diffs) - mean)) / mean))
        noise = _float32_noise(values, coords.dtype)
        if noise:
            noise_max = max(noise_max, noise)
            notes.append(f"'{name}' is {coords.dtype}, which alone accounts for {noise:.1e}")
    effective = max(float(tol["frame_uniformity"]), noise_max)
    passed = not problems and worst <= effective
    detail = "; ".join(problems) if problems else (
        f"axes strictly monotonic and uniformly spaced within {effective:.1e} "
        f"(tolerance {tol['frame_uniformity']:.1e} or the dtype's own rounding, whichever is larger)"
        + (f": {', '.join(notes)}" if notes else ""))
    return GateResult("frame", passed, worst, effective, "1", detail), {}


def _provenance(bundle: PhysicsDataBundle, tol: Mapping[str, float]) -> tuple[GateResult, dict]:
    p = bundle.manifest.provenance
    problems: list[str] = []
    if not p.source:
        problems.append("no source")
    if not p.license or p.license.strip().lower() in ("", "unknown"):
        problems.append(f"licence is '{p.license or 'empty'}': the data cannot be redistributed until it is known")
    if p.kind == "dataset" and not p.citation and not p.url:
        # an external dataset has a place and a paper; a case this machine solved has a path and a solver
        problems.append("no citation and no url: an external dataset has to say where it came from")
    if p.kind in ("dataset", "experiment") and not p.inputs:
        problems.append("no input file hash recorded: a bundle that came from a file has to name the file's hash")
    for ref in p.inputs:
        if not ref.sha256:
            problems.append(f"input '{ref.path}' has no sha256")
    note = (f"source, licence ({p.license}), citation and {len(p.inputs)} input hash(es) recorded" if p.inputs
            else f"source, licence ({p.license}) and citation recorded; no input file for kind '{p.kind}'")
    return GateResult("provenance", not problems, float(len(problems)), tol["provenance"], "1",
                      "; ".join(problems) or note), {}


def _conservation(bundle: PhysicsDataBundle, tol: Mapping[str, float], field_name: str | None,
                  axis: str) -> tuple[GateResult, dict]:
    if field_name is None:
        raise GateNotApplicable("no conserved field was named for this bundle")
    if "t" not in bundle.axes:
        raise GateNotApplicable("a conservation check needs a time axis")
    drift = conservation_drift(bundle, field_name, axis=axis)
    return GateResult("conservation", drift <= tol["conservation"], drift, tol["conservation"], "1",
                      f"relative drift of the integral of '{field_name}' along '{axis}' over {bundle.n_samples()} "
                      "records"), {"field": field_name, "axis": axis}


def _reference(bundle: PhysicsDataBundle, tol: Mapping[str, float], reference: np.ndarray | None,
               field_name: str | None, reference_slice: Any = None) -> tuple[GateResult, dict]:
    if reference is None or field_name is None:
        raise GateNotApplicable("no reference array was given")
    values = bundle.field_array(field_name)
    if reference_slice is not None:
        values = values[reference_slice]
    try:
        err = relative_l2(values, reference)
    except ValueError as exc:
        raise GateNotApplicable(f"'{field_name}' cannot be scored against this reference: {exc}") from None
    return GateResult("reference", err <= tol["reference"], err, tol["reference"], "1",
                      f"relative L2 error of '{field_name}' against the reference"), {"field": field_name}


def run_gates(bundle: PhysicsDataBundle, *, gates: Sequence[str] = GATES, conserved: str | None = None,
              conserved_axis: str = "x", reference: np.ndarray | None = None, reference_field: str | None = None,
              reference_slice: Any = None, tolerances: Mapping[str, float] | None = None) -> GateReport:
    """Run the requested gates over ``bundle`` and return the report.

    ``conserved`` names the field whose spatial integral should not drift (a density, a concentration, a
    mass); ``reference`` is an array to compare a field against (a solver's own answer, an analytical solution),
    and ``reference_slice`` selects the part of the field it covers -- a held-out block of time steps, say, which
    is the only part of a field a model's prediction should be compared with. A gate that cannot be asked is
    "not run", and the report says so rather than claiming a pass.
    """
    tol = dict(DEFAULT_TOLERANCES)
    tol.update(tolerances or {})
    report = GateReport(bundle_id=bundle.manifest.id, bundle_digest=bundle.digest())
    for gate in gates:
        if gate not in GATES:
            raise ValueError(f"unknown gate '{gate}'; known: {GATES}")
        start = time.time()
        try:
            if gate == "integrity":
                result, _ = _integrity(bundle, tol)
            elif gate == "units":
                result, _ = _units(bundle, tol)
            elif gate == "location":
                result, _ = _location(bundle, tol)
            elif gate == "frame":
                result, _ = _frame(bundle, tol)
            elif gate == "provenance":
                result, _ = _provenance(bundle, tol)
            elif gate == "conservation":
                result, _ = _conservation(bundle, tol, conserved, conserved_axis)
            else:
                result, _ = _reference(bundle, tol, reference, reference_field or conserved, reference_slice)
        except GateNotApplicable as exc:
            result = GateResult(gate, None, None, None, "1", str(exc))
        result.detail = f"{result.detail} ({time.time() - start:.2f}s)"
        report.results.append(result)
    return report

def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
