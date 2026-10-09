"""Qualitative physics preview of geometries and geometry changes, before the quantitative analysis.

Give it different geometries (or an ``Assembly`` whose parts you change) and a physical goal, and it tells you, from
cheap physical models, which way each geometry or change should push the result, how strongly, through which
mechanism, with what confidence, and where on the surface the physics acts; then ``quantify`` runs the accurate
computation only where it is worth it and checks the qualitative reading.

>>> from pinneapple_design.qualitative import preview, part_sensitivity, Assembly
>>> p = preview({"baseline": body, "rounded rear": body2}, "minimizar arrasto", {"U": 30})
>>> print(p.text()); p.figure("preview.png")
>>> part_sensitivity(Assembly({"base": plate, "fin_1": fin1, "fin_2": fin2}), "maximizar resfriamento").text()
"""
from .descriptors import Assembly, Descriptors, describe, extrude, section_profile
from .models import (
    MODELS,
    Cantilever,
    ConvectiveCooling,
    Evaluation,
    ExternalFlow,
    QualitativeModel,
    ScalingModel,
)
from .preview import (
    Objective,
    PartSensitivity,
    QualitativePreview,
    QuantitativeCheck,
    parse_objective,
    part_sensitivity,
    preview,
)

__all__ = ["Assembly", "Cantilever", "ConvectiveCooling", "Descriptors", "Evaluation", "ExternalFlow", "MODELS",
           "Objective", "PartSensitivity", "QualitativeModel", "QualitativePreview", "QuantitativeCheck",
           "ScalingModel", "describe", "extrude", "parse_objective", "part_sensitivity", "preview", "section_profile"]
