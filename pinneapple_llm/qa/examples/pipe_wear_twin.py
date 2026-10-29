"""End-to-end reference example: a 3D digital twin of a pipe under wall wear.

What this builds, using only APIs that already exist in this repository:

* a **hollow** pipe mesh (outer wall, inner wall, two annular end faces)
  from :func:`pinneapple_twin3d.primitives.revolve`;
* a wear zone (``pinneapple_twin3d.wear``) whose per-cell wear depth
  evolves over time, with per-cell meshes from
  :func:`pinneapple_twin3d.wear.geometry.cell_mesh`;
* a ``pinneapple_twin3d.Scene`` built by ``scene_from_wear`` and exported
  to real files (``scene.json``, ``geometry.glb``, ``fields.bin``);
* four ultrasonic thickness readings as sensors, one of them reserved as
  holdout, positioned at real cell centres.

Two operating conditions are provided, both synthetic:
``anomaly=False`` (uniform 1.2 mm/year wear, staying under the limit) and
``anomaly=True`` (a localised hotspot that breaches the 2 mm minimum wall).

Honesty note, repeated in the report: this is a **verification** example.
The reference it compares against is the closed-form uniform-wear solution,
so agreement demonstrates that the pipeline, the mesh and the field
plumbing are consistent -- it says nothing about a real pipe. The dataset
is marked ``synthetic`` and the quality level is capped at Q3 by
construction (Q4 needs operational evidence).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..pipeline import QualityPipeline
from ..references import ReferenceData
from ..report import QualityReport
from ..spec import (GeometrySpec, Invariant, Observation, PhysicalSpecification, Provenance,
                    Quantity)
from ..validators.base import SensorReading

# -- design point (explicit, in SI) ----------------------------------------
OUTER_DIAMETER_M = 0.20
WALL_THICKNESS_M = 0.010
LENGTH_M = 2.00
YEARS = 5.0
N_ROWS = 24
N_SECTORS = 48
WEAR_RATE_MM_PER_YEAR = 1.2
MIN_WALL_MM = 2.0
ANOMALY_RATE_MM_PER_YEAR = 2.4


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def hollow_pipe_mesh(*, radius: float = OUTER_DIAMETER_M / 2,
                     thickness: float = WALL_THICKNESS_M,
                     length: float = LENGTH_M,
                     n_theta: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """A genuinely hollow pipe: outer wall, inner wall and two end annuli.

    The profile is a closed polyline revolved around the y axis, which the
    existing primitive turns into a watertight surface with a bore.
    """
    from pinneapple_twin3d.primitives import revolve

    inner = radius - thickness
    if inner <= 0:
        raise ValueError("wall thickness must be smaller than the outer radius")
    profile = [
        (radius, 0.0), (radius, length),
        (inner, length), (inner, 0.0), (radius, 0.0),
    ]
    return revolve(profile, n_theta=n_theta)


def solid_pipe_mesh(*, radius: float = OUTER_DIAMETER_M / 2,
                    length: float = LENGTH_M, n_theta: int = 64
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """A filled cylinder of the same outer dimensions.

    Used by the tests as the deliberate failure the geometry validator must
    catch: it looks like a pipe from outside and is not one.
    """
    from pinneapple_twin3d.primitives import cylinder

    return cylinder((0.0, 0.0, 0.0), (0.0, length, 0.0), radius, n_theta=n_theta, cap=True)


# ---------------------------------------------------------------------------
# Specification (stage A)
# ---------------------------------------------------------------------------
def build_spec(*, anomaly: bool = False) -> PhysicalSpecification:
    e0 = WALL_THICKNESS_M * 1000.0
    limit = e0 - MIN_WALL_MM
    return PhysicalSpecification(
        objective="Track pipe wall wear over five years and keep the residual wall above the "
                  "minimum allowed thickness.",
        phenomenon="erosive/corrosive wall thinning of a straight pipe section",
        geometry=GeometrySpec(
            kind="hollow_pipe",
            coordinate_system="cylindrical",
            dimensions={
                "outer_diameter": Quantity(OUTER_DIAMETER_M, "m", Provenance.USER),
                "outer_radius": Quantity(OUTER_DIAMETER_M / 2, "m", Provenance.USER),
                "wall_thickness": Quantity(WALL_THICKNESS_M, "m", Provenance.USER),
                "length": Quantity(LENGTH_M, "m", Provenance.USER),
            },
            hollow=True,
            wall_thickness=Quantity(WALL_THICKNESS_M, "m", Provenance.USER),
            notes="straight section, y is the pipe axis; wall wear acts on the bore",
        ),
        independent_variables=["t"],
        dependent_variables=["wear_depth", "residual_thickness"],
        parameters={
            "wear_depth": Quantity(0.0, "mm", Provenance.USER,
                                   uncertainty=0.05),
            "residual_thickness": Quantity(e0, "mm", Provenance.USER),
            "wear_rate": Quantity(
                (ANOMALY_RATE_MM_PER_YEAR if anomaly else WEAR_RATE_MM_PER_YEAR),
                "mm/year", Provenance.ESTIMATED,
                source="nominal erosion rate assumed for the example"),
            "initial_wall_thickness": Quantity(e0, "mm", Provenance.USER),
            "minimum_wall": Quantity(MIN_WALL_MM, "mm", Provenance.SOURCE,
                                     source="ASME B31.3 t_min design margin (example value)"),
        },
        materials={"carbon_steel": {"density": Quantity(7850.0, "kg/m^3", Provenance.SOURCE,
                                                        source="standard value")}},
        governing_equations=[
            "dw/dt = rate(t)  (wear depth grows monotonically)",
            "e(t) = e0 - w(t)  (residual thickness)",
        ],
        initial_conditions=["w(t=0) = 0 everywhere on the bore"],
        boundary_conditions=["wear_depth: no wear at the flanges (example assumption)"],
        spatial_validity={
            "y": (Quantity(0.0, "m", Provenance.USER), Quantity(LENGTH_M, "m", Provenance.USER)),
        },
        temporal_validity=(Quantity(0.0, "year", Provenance.USER), Quantity(YEARS, "year", Provenance.USER)),
        invariants=[
            Invariant("wear_nonnegative", "nonnegative", "wear_depth",
                      description="wear depth cannot be negative"),
            Invariant("wear_below_allowance", "max", "wear_depth",
                      params={"limit": limit},
                      description=f"residual wall must stay >= {MIN_WALL_MM} mm"),
            Invariant("wear_monotonic", "monotonic_nondecreasing", "wear_depth",
                      description="thinning is cumulative; no repair is modelled here"),
            Invariant("wear_finite", "finite", "wear_depth"),
            Invariant("wear_units", "unit_compatible", "wear_depth", params={"unit": "mm"}),
            Invariant("residual_above_minimum", "min", "residual_thickness",
                      params={"limit": MIN_WALL_MM},
                      description="residual thickness must never drop below the minimum wall"),
            Invariant("residual_units", "unit_compatible", "residual_thickness",
                      params={"unit": "mm"}),
        ],
        numerical_method={"method": "per-cell linear wear law on a surface of revolution",
                          "provider": "none", "model": "none"},
        validation_metrics={
            "max_wear_mm": {"limit": limit, "severity": "error"},
            "sensor_rmse_mm": {"limit": 0.1, "severity": "error"},
        },
        assumptions=[
            "the wear rate is constant in time and uniform over the bore (nominal case)",
            "the pipe is straight and its wall thickness is initially uniform",
        ],
        missing_data=[],
        required_validations=["units", "numeric", "physics", "geometry", "temporal", "sensors"],
    )


# ---------------------------------------------------------------------------
# Twin construction (stage B, done by code here instead of an LLM)
# ---------------------------------------------------------------------------
def build_twin(*, anomaly: bool = False) -> Dict[str, Any]:
    from pinneapple_twin3d.wear.build import scene_from_wear
    from pinneapple_twin3d.wear.geometry import cell_centers, cell_mesh, cell_to_vertex
    from pinneapple_twin3d.wear.model import SYNTHETIC, VesselSpec, WearDataset, ZoneSpec

    spec = build_spec(anomaly=anomaly)
    radius = OUTER_DIAMETER_M / 2
    e0 = WALL_THICKNESS_M * 1000.0

    zone = ZoneSpec(
        name="wall", label="pipe wall",
        profile=((radius, 0.0), (radius, LENGTH_M)),
        n_rows=N_ROWS, n_sectors=N_SECTORS,
        e0=e0, emin=MIN_WALL_MM, group="wall",
    )
    vessel = VesselSpec("pipe_dn200", "Pipe DN200 - wall wear", (zone,),
                        x_name="year", wear_unit="mm",
                        notes="synthetic uniform wear benchmark")

    times = np.linspace(0.0, YEARS, int(YEARS) + 1)
    rate = ANOMALY_RATE_MM_PER_YEAR if anomaly else WEAR_RATE_MM_PER_YEAR
    wear = rate * times[:, None, None] * np.ones((1, N_ROWS, N_SECTORS))
    if anomaly:
        i0, j0 = N_ROWS // 2, N_SECTORS // 4
        ii, jj = np.meshgrid(np.arange(N_ROWS), np.arange(N_SECTORS), indexing="ij")
        hotspot = np.exp(-(((ii - i0) ** 2) / 6.0 + ((jj - j0) ** 2) / 8.0))
        wear = wear + 3.2 * (times[:, None, None] / YEARS) * hotspot[None, :, :]
    wear = np.clip(wear, 0.0, e0)  # physically: wall wear cannot exceed wall thickness

    ds = WearDataset(vessel, times, {"wall": wear}, origin=SYNTHETIC,
                     note=f"synthetic benchmark ({'anomaly' if anomaly else 'nominal'} condition)")

    scene = scene_from_wear(ds)
    shell_v, shell_f = hollow_pipe_mesh()
    scene.add_part("pipe_body", shell_v, shell_f, group="pipe",
                   color=(0.55, 0.6, 0.68))

    cell_v, cell_f = cell_mesh(zone)
    centers = cell_centers(zone)

    # sensors: four readings evenly spread along the bore, one held out
    sensor_spec = [(2, 0), (N_ROWS // 2, N_SECTORS // 4), (N_ROWS - 3, N_SECTORS // 2),
                   (N_ROWS // 3, (3 * N_SECTORS) // 4)]
    sensors = []
    for k, (i, j) in enumerate(sensor_spec):
        series = e0 - wear[:, i, j]
        sensors.append(SensorReading(
            id=f"UT-{101 + k}", position=tuple(float(c) for c in centers[i, j]),
            unit="mm", quantity="residual_thickness", series=series,
            holdout=(k == len(sensor_spec) - 1),
        ))
    for reading in sensors:
        scene.add_sensor(reading.id, reading.position, unit=reading.unit,
                         quantity=reading.quantity, series=reading.series,
                         envelope=(MIN_WALL_MM, e0))

    fields = {
        "wear_depth": cell_to_vertex(wear),
        "residual_thickness": cell_to_vertex(e0 - wear),
    }
    return {
        "spec": spec,
        "dataset": ds,
        "scene": scene,
        "meshes": {"wall_cells": (cell_v, cell_f), "pipe_body": (shell_v, shell_f)},
        "fields": fields,
        "field_units": {"wear_depth": "mm", "residual_thickness": "mm"},
        "times": times,
        "time_unit": "year",
        "sensors": sensors,
        "payload": {
            "format": "pinneapple-twin3d/1",
            "title": scene.title,
            "times": [float(t) for t in times],
            "parts": [p.name for p in scene.parts],
            "fields": sorted(fields),
            "origin": ds.origin,
        },
        "observations": [
            Observation(
                sensor_id=s.id, quantity=s.quantity, unit=s.unit,
                position=tuple(Quantity(float(c), "m", Provenance.USER) for c in s.position),
                timestamps=tuple(float(t) for t in times), time_unit="year",
                values=tuple(float(v) for v in s.series),
                # these readings come from the synthetic dataset, never from an instrument
                provenance=Provenance.ESTIMATED,
                holdout=s.holdout,
            )
            for s in sensors
        ],
    }


def export_twin(bundle: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """Write the real viewer files and the hub contract; returns ``scene.json``.

    No ``except: pass`` anywhere: if the scene cannot be exported the caller
    sees the exception (and, when run through the pipeline, a failed
    ``export`` step in the report).
    """
    folder = output_dir or os.path.join(os.getcwd(), "pipe_twin")
    os.makedirs(folder, exist_ok=True)
    path = bundle["scene"].export(folder, with_viewer=False, usd=False)
    with open(os.path.join(folder, "wear_dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle["dataset"].to_contract(), fh, separators=(",", ":"))
    return path


# ---------------------------------------------------------------------------
# Analytical reference (stage F)
# ---------------------------------------------------------------------------
def analytical_reference(bundle: Dict[str, Any]) -> ReferenceData:
    """Closed-form uniform-wear residual thickness, per cell and per time.

    This is a *verification* reference for the nominal condition: it derives
    from the same declared law, so it checks plumbing and formulas, not
    physics. It is labelled ``analytical`` and its limitations travel with
    every report that uses it.
    """
    e0 = WALL_THICKNESS_M * 1000.0
    times = np.asarray(bundle["times"], dtype=float)
    residual = e0 - WEAR_RATE_MM_PER_YEAR * times
    cells = N_ROWS * N_SECTORS
    # the artifact's field is per-vertex (cell_to_vertex repeats each cell 4x),
    # so the reference is expanded the same way to keep the comparison 1:1
    values = np.repeat(residual[:, None], cells, axis=1)
    values = np.repeat(values, 4, axis=1).reshape(-1)
    return ReferenceData(
        name="pipe-uniform-wear-residual",
        origin="analytical",
        version="1.0.0",
        values=values,
        unit="mm",
        limitations=[
            "closed-form solution for a constant, uniform wear rate",
            "synthetic input data: this is code verification, not apparatus validation",
        ],
        used_for_calibration=False,
    )


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_example(*, output_dir: Optional[str] = None, anomaly: bool = False,
                write_report: bool = True) -> Tuple[QualityReport, Dict[str, Any]]:
    """Run stages A-G over the synthetic pipe twin and return the report."""
    bundle = build_twin(anomaly=anomaly)
    pipeline = QualityPipeline(
        bundle["spec"],
        reference=analytical_reference(bundle),
        reference_criteria={"rmse": 0.5, "max_abs": 1.0},
        reference_subset="all-synthetic-cells",
        reference_field="residual_thickness",
    )
    report = pipeline.run(
        meshes=bundle["meshes"],
        fields=bundle["fields"],
        field_units=bundle["field_units"],
        times=bundle["times"],
        time_unit=bundle["time_unit"],
        sensors=bundle["sensors"],
        payload=bundle["payload"],
        exporter=(lambda out: export_twin(bundle, out)) if output_dir else None,
        output_dir=output_dir,
        output_manifest=os.path.join(output_dir, "quality_report.json")
        if (output_dir and write_report) else None,
    )
    return report, bundle
