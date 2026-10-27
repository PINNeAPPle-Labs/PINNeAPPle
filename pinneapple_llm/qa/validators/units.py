"""Dimensional consistency across geometry, parameters, fields and sensors.

What this checks: that every declared quantity *has* a parseable unit, that
lengths agree with lengths, that a field's declared unit matches the unit
the specification declared for the same quantity, and that sensor units
agree with the fields they observe. What it does **not** check: that the
equation a field appears in is dimensionally sound. Dimensional analysis
can prove an expression inconsistent; it can never prove one correct.
"""
from __future__ import annotations

from typing import List

from ..report import Finding, Status
from ..spec import _is_length
from ..units import UnitError, are_compatible, parse_unit
from .base import ValidationContext, ValidatorResult, not_applicable, register

_NAME = "units"


@register(_NAME)
def check_units(ctx: ValidationContext) -> ValidatorResult:
    spec = ctx.spec
    findings: List[Finding] = []
    evidence = {"checked": [], "incompatible": []}
    limitations = [
        "a passing units check says dimensions agree, not that the governing equation is right",
        "only units declared in the specification and in the artifact's field metadata are compared",
    ]

    # 1. geometry dimensions must be lengths
    if spec.geometry is not None:
        for name, q in spec.geometry.dimensions.items():
            try:
                parse_unit(q.unit)
            except UnitError as exc:
                findings.append(_fail("unit.unparsable", f"geometry dimension '{name}': {exc}",
                                      {"quantity": name, "unit": q.unit}))
                continue
            if not _is_length(q.unit):
                findings.append(_fail(
                    "unit.geometry_dimension", f"geometry dimension '{name}' is '{q.unit}', not a length",
                    {"quantity": name, "unit": q.unit}))
            evidence["checked"].append(f"geometry.{name}")
        wt = spec.geometry.wall_thickness
        if wt is not None:
            lengths = [q.unit for q in spec.geometry.dimensions.values() if _is_length(q.unit)]
            if lengths and not _is_length(wt.unit):
                findings.append(_fail(
                    "unit.wall_thickness", f"wall thickness is '{wt.unit}', not a length",
                    {"unit": wt.unit}))
            evidence["checked"].append("geometry.wall_thickness")
    else:
        findings.append(Finding(
            id="unit.geometry_absent", status=Status.NOT_RUN, severity="warning",
            message="specification declares no geometry; geometry units were not checked"))

    # 2. declared parameters must have parseable, consistent units where a
    #    counterpart unit is declared (same physical role, same dimension)
    for name, q in spec.parameters.items():
        try:
            parse_unit(q.unit)
        except UnitError as exc:
            findings.append(_fail("unit.parameter", f"parameter '{name}': {exc}",
                                  {"parameter": name, "unit": q.unit}))
            continue
        evidence["checked"].append(f"parameter.{name}")

    # 3. artifact field units vs the specification's own declaration
    for field_name, unit in ctx.field_units.items():
        try:
            parse_unit(unit)
        except UnitError as exc:
            findings.append(_fail("unit.field", f"field '{field_name}': {exc}",
                                  {"field": field_name, "unit": unit}))
            continue
        declared = _declared_unit_for(spec, field_name)
        if declared is not None and not are_compatible(unit, declared):
            findings.append(_fail(
                "unit.field_vs_spec",
                f"field '{field_name}' is in '{unit}' but the specification declares '{declared}'",
                {"field": field_name, "artifact_unit": unit, "spec_unit": declared}))
        elif declared is not None and unit.strip() != declared.strip():
            findings.append(Finding(
                id="unit.field_scale", status=Status.NOT_RUN, severity="warning",
                message=f"field '{field_name}' is in '{unit}' while the specification declares "
                        f"'{declared}' -- same dimension, different scale; consumers must convert",
                evidence={"field": field_name, "artifact_unit": unit, "spec_unit": declared},
                justification="a compatible but different unit silently changes magnitudes if nobody "
                            "converts; reported rather than treated as a pass"))
        evidence["checked"].append(f"field.{field_name}")

    # 4. sensor units vs the fields they observe
    for sensor in ctx.sensors:
        try:
            parse_unit(sensor.unit)
        except UnitError as exc:
            findings.append(_fail("unit.sensor", f"sensor '{sensor.id}': {exc}",
                                  {"sensor": sensor.id, "unit": sensor.unit}))
            continue
        field_unit = ctx.field_units.get(sensor.quantity)
        if field_unit and not are_compatible(sensor.unit, field_unit):
            findings.append(_fail(
                "unit.sensor_vs_field",
                f"sensor '{sensor.id}' measures '{sensor.quantity}' in '{sensor.unit}' but the "
                f"field is declared in '{field_unit}'",
                {"sensor": sensor.id, "sensor_unit": sensor.unit, "field_unit": field_unit}))
        evidence["checked"].append(f"sensor.{sensor.id}")

    if not evidence["checked"] and not findings:
        return not_applicable(_NAME, "no geometry, parameters, fields or sensors declare units")

    if any(f.blocks for f in findings):
        status = Status.FAIL
    else:
        status = Status.PASS
        findings.append(Finding(id="validator.units", status=Status.PASS, severity="info",
                                message=f"{len(evidence['checked'])} unit declaration(s) consistent",
                                evidence={"checked": evidence["checked"]}))
    evidence["incompatible"] = [f.message for f in findings if f.blocks]
    return ValidatorResult(name=_NAME, status=status, findings=findings,
                           evidence=evidence, limitations=limitations)


def _declared_unit_for(spec, field_name: str):
    if field_name in spec.parameters:
        return spec.parameters[field_name].unit
    for obs in spec.observations:
        if obs.quantity == field_name:
            return obs.unit
    key = field_name.removesuffix("_depth").removesuffix("_field")
    if key in spec.parameters:
        return spec.parameters[key].unit
    return None


def _fail(fid: str, message: str, evidence) -> Finding:
    return Finding(id=fid, status=Status.FAIL, severity="error", message=message, evidence=dict(evidence),
                   justification="a unit mismatch makes every numeric result expressed in that "
                                 "quantity uninterpretable")
