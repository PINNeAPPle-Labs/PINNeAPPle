"""The product surface: one call to commission a twin, one to run it, one to file the evidence.

Everything else in :mod:`pinneapple_systems.digital_twin.product` is a part. This module is
the thing a project uses::

    spec = TwinSpec.from_yaml("exchanger.yaml")
    evidence = commission(spec, model, plant_history)      # preflight + backtest + report
    if evidence.commissioned:                              # gates cleared
        run = run_live(spec, model, streaming_frames, evidence=evidence)
        run.final_mode                                      # NORMAL | WATCH | ALARM | RECALIBRATE

``commission`` is deterministic for a given ``seed``, so a client can re-run it and get the
same number. That is the whole point of shipping the evidence file with the twin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .assimilation import LatentStateTwin
from .commission import backtest, preflight, read_readings, read_timestamp
from .evidence import TwinEvidence

from .policy import DriftMonitor, DriftPolicy, TwinMode
from .spec import TwinSpec

__all__ = ["commission", "run_live", "RunResult", "export_scene", "describe"]


# ---------------------------------------------------------------------------
# Commissioning
# ---------------------------------------------------------------------------

def commission(
    spec: TwinSpec,
    model: Callable[..., Any],
    history: Sequence[Mapping[str, Any]],
    *,
    n_ens: int = 64,
    inflation: float = 1.02,
    seed: int = 0,
    warmup_steps: int = 0,
    calibrate_fraction: float = 0.5,
    notes: str = "",
) -> TwinEvidence:
    """Run both gates and return the evidence file.

    The backtest only runs when the preflight passes: replaying a twin whose sensors are
    mislabelled produces a number that is worse than no number, because it looks like data.

    ``warmup_steps`` frames are assimilated but excluded from the metrics, letting the latent
    state converge. The count is recorded in the evidence so the window is never implicit.
    """
    evidence = TwinEvidence(
        asset_id=spec.asset_id,
        spec_digest=spec.digest(),
        model_id=f"{spec.model.model_id} {spec.model.version}".strip(),
        notes=notes,
    )
    evidence.preflight = preflight(spec, history)

    if not evidence.preflight.passed:
        evidence.unverified = ["tracking: not measured, preflight did not pass"]
        return evidence

    evidence.tracking = backtest(
        spec, model, history, n_ens=n_ens, inflation=inflation, seed=seed,
        warmup_steps=warmup_steps, calibrate_fraction=calibrate_fraction,
    )
    evidence.unverified = _unverified_list(spec, evidence, warmup_steps)
    if evidence.tracking:
        # Hand over the state commissioning converged to: an on-line twin that restarts from a
        # clean prior would alarm on its first frame for no reason but its own cold start.
        evidence.initial_state = dict(evidence.tracking.final_latent)
    return evidence


def _unverified_list(spec: TwinSpec, evidence: TwinEvidence, warmup_steps: int) -> list[str]:
    """The honest footer of a commissioning report."""
    out: list[str] = [
        "the band is a state-uncertainty band scaled once on a held-out window; it is not a "
        "prediction interval for a change of regime",
        f"commissioned on {len(spec.sensors)} channels within the declared envelope only; "
        "no claim is made outside the stated operating range",
    ]
    if warmup_steps:
        out.append(f"the first {warmup_steps} frames were assimilated but excluded from the metrics")
    for m in (evidence.tracking.metrics if evidence.tracking else []):
        if abs(m.band_scale - 1.0) > 1e-9:
            out.append(
                f"{m.field_name}: the band was scaled ×{m.band_scale:.2f} on the first "
                f"{m.calibration_s / 3600:.1f} h of the window and scored on the rest; the raw "
                "ensemble band would have covered "
                f"{100 * (m.coverage_raw or 0):.1f}% of the scored part"
            )
    if spec.model.accuracy_floor is not None:
        out.append(
            f"tracking error includes the model's own error; the spec claims a floor of "
            f"{spec.model.accuracy_floor:g} {spec.fields[0].unit}"
        )
    for f in spec.fields:
        if f.alarm_above is None and f.alarm_below is None:
            out.append(f"no alarm lead measured for {f.name}: no engineering limit was declared")
    if evidence.tracking and not evidence.tracking.alarm_lead_s:
        out.append("no alarm lead was measured on any field")
    if evidence.tracking and evidence.tracking.forecast_error_s:
        out.append(
            "the alarm lead comes from a linear extrapolation of the latent state; its error "
            "against the crossing that actually happened is reported, and it grows with any "
            "change of regime"
        )
    out.append("the model was not re-fitted during commissioning; this is the twin as shipped")
    return out


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """What happened while the twin ran."""

    n_steps: int = 0
    n_observed: int = 0
    n_holdout: int = 0
    final_mode: TwinMode = TwinMode.NORMAL
    coverage: float | None = None
    latent: dict[str, dict[str, float]] = field(default_factory=dict)
    modes_seen: dict[str, int] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "n_observed": self.n_observed,
            "n_holdout": self.n_holdout,
            "final_mode": self.final_mode.value,
            "coverage": self.coverage,
            "latent": self.latent,
            "modes_seen": self.modes_seen,
            "decisions": self.decisions,
        }


def run_live(
    spec: TwinSpec,
    model: Callable[..., Any],
    frames: Sequence[Mapping[str, Any]],
    *,
    n_ens: int = 64,
    inflation: float = 1.02,
    seed: int = 0,
    policy: DriftPolicy | None = None,
    evidence: TwinEvidence | None = None,
    theta0: Sequence[float] | None = None,
) -> RunResult:
    """Assimilate a run of frames, monitor consistency, and record what the twin decided.

    Frames without a held-out reading still move the state; they simply contribute no
    consistency observation. That is the common case on line, where the reference instrument
    is a monthly lab sample.

    When ``evidence`` is supplied its ``decisions`` are updated in place, so the same file
    that commissioned the twin also carries its on-line behaviour. Pass ``theta0`` — normally
    ``evidence.initial_state`` — so the twin starts from the state commissioning converged to
    instead of a cold prior.
    """
    twin = LatentStateTwin(model, spec, n_ens=n_ens, inflation=inflation, seed=seed)
    if theta0 is not None:
        twin.reset(theta0)
    monitor = DriftMonitor(policy or DriftPolicy())
    result = RunResult()

    for frame in frames:
        obs = read_readings(frame, spec)
        if not obs:
            continue
        twin.step(obs, t=read_timestamp(frame))
        result.n_steps += 1
        result.n_observed += len(obs)

        pred, lo, hi = twin.band_at_holdout()
        in_band: list[bool] = []
        for j, s in enumerate(twin.holdout_sensors):
            measured = obs.get(s.sensor_id)
            if measured is None:
                continue
            in_band.append(bool(lo[j] <= measured <= hi[j]))
        result.n_holdout += len(in_band)
        mode = monitor.observe(in_band, t=read_timestamp(frame) if in_band else None)
        result.modes_seen[mode.value] = result.modes_seen.get(mode.value, 0) + 1

    result.final_mode = monitor.mode
    result.coverage = monitor.coverage
    result.latent = twin.latent_dict()
    result.decisions = list(monitor.decisions)

    if evidence is not None:
        evidence.decisions.extend(monitor.decisions)
        evidence.notes = (evidence.notes + "\n" if evidence.notes else "") + (
            f"on-line run: {result.n_steps} steps, final mode {result.final_mode.value}, "
            f"rolling coverage {result.coverage if result.coverage is None else f'{result.coverage:.3f}'}"
        )
    return result


# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------

def describe(
    spec: TwinSpec,
    latent: Mapping[str, Mapping[str, Any]],
    fields: Mapping[str, float] | None = None,
):
    """Describe the twin for the industry standards the platform already speaks.

    Returns a :class:`~pinneapple_systems.digital_twin.standards.TwinDescription`, which
    ``to_aas`` (AAS / IDTA), ``to_dtdl``, ``to_ngsi_ld`` and ``to_ditto_thing`` turn into the
    corresponding document, and which ``OpcUaTwinServer`` serves over OPC UA.

    Estimated parameters go out as ``parameter`` properties with their standard deviation in
    the description; the field estimates go out as ``state``. Sensor identifiers are not
    published directly because the standards require identifier-safe names and plant tags
    contain dashes — the twin's *outputs* are what a platform consumes.
    """
    from ..standards import TwinDescription, TwinModel, TwinProperty

    props: list[Any] = []
    for f in spec.fields:
        if f.kind == "parameter":
            entry = latent.get(f.name)
            if entry is None:
                continue
            props.append(
                TwinProperty(
                    _identifier(f.name), float(entry["value"]), f.unit, "parameter",
                    f"estimated latent parameter (1 sigma {float(entry.get('std', 0.0)):.4g} {f.unit})",
                )
            )
        elif fields and f.name in fields:
            props.append(
                TwinProperty(_identifier(f.name), float(fields[f.name]), f.unit, "state",
                             "twin estimate at the reference point")
            )

    return TwinDescription(
        id=f"urn:pinneapple:twin:{spec.asset_id}",
        name=spec.asset_name,
        asset_type=spec.asset_type or "Asset",
        description=f"PINNeAPPle twin, spec {spec.digest()}",
        properties=props,
        models=[TwinModel(spec.model.model_id, family=spec.asset_type or "", status="UNKNOWN")],
        relationships=[],
    )


def _identifier(name: str) -> str:
    """A standard-safe property name: letters, digits and underscore, starting with a letter."""
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if out[:1].isalpha() else f"p_{out}"


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def export_scene(
    spec: TwinSpec,
    frames: Sequence[Mapping[str, Any]],
    folder: str,
    *,
    mesh: Any = None,
    field_on_mesh: Mapping[str, Sequence[float]] | None = None,
    times: Sequence[float] | None = None,
) -> str:
    """Write a ``pinneapple_twin3d`` scene for this twin: sensors with measured series and envelopes.

    The geometry is the asset's, so it is supplied rather than invented: pass ``mesh`` (a
    trimesh object) and ``field_on_mesh`` (name -> per-vertex values per time step) to see a
    field painted on it. Without a mesh, a wireframe box is drawn around the instrument layout
    so the viewer has something to open — it marks the extent of the measurements, it is not
    the machine, and the scene title says so.

    Sensor coordinates must be 3-D for the viewer; anything else is padded with zeros and the
    scene records that in its title, rather than silently pretending to a position.
    """
    from pinneapple_twin3d import Scene

    padded = 0
    positions: dict[str, tuple[float, float, float]] = {}
    for s in spec.sensors:
        at = list(s.at[:3])
        if len(at) < 3:
            padded += 1
            at = at + [0.0] * (3 - len(at))
        positions[s.sensor_id] = (float(at[0]), float(at[1]), float(at[2]))

    series: dict[str, list[float]] = {s.sensor_id: [] for s in spec.sensors}
    for frame in frames:
        readings = read_readings(frame, spec)
        for s in spec.sensors:
            series[s.sensor_id].append(float(readings.get(s.sensor_id, np.nan)))

    scene_times = list(times) if times is not None else [read_timestamp(f) for f in frames]
    title = spec.asset_name
    if padded:
        title += f" ({padded} sensor(s) without 3-D position, drawn at the origin)"

    sc = Scene(title, times=scene_times, time_unit="s")
    for s in spec.sensors:
        sc.add_sensor(
            s.sensor_id,
            positions[s.sensor_id],
            unit=s.unit,
            label=s.sensor_id,
            series=series[s.sensor_id],
            envelope=(s.lo, s.hi),
        )

    if mesh is None:
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "export_scene needs a mesh, or trimesh to draw the extent box: "
                "pip install trimesh"
            ) from exc
        pts = np.array(list(positions.values()), dtype=float)
        lo_pt, hi_pt = pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5
        extent = np.maximum(hi_pt - lo_pt, 0.5)
        mesh = trimesh.creation.box(extents=extent)
        mesh.apply_translation((lo_pt + hi_pt) / 2.0)
        note = " — instrument extent box, not the asset geometry"
        if note not in sc.title:
            sc.title += note

    sc.add_trimesh("asset", mesh)
    for name, values in (field_on_mesh or {}).items():
        sc.add_field("asset", name, values, unit="")

    return sc.export(folder)
