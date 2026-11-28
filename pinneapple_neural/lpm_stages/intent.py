"""Stage 0: the question comes first, and it arrives as a contract rather than as a dictionary.

:mod:`pinneapple_intent` compiles a sentence into a :class:`~pinneapple_intent.contract.StudyContract` — a typed,
versioned description in which **every value carries where it came from** (stated, answered, defaulted with its source,
or missing) and a value nobody gave becomes a question instead of an invention. A design space, in this package, is a
family plus ranges. This stage is the join: the contract supplies the *point* — one geometry, one operating condition,
with its provenance intact — and the family supplies the *space around it*.

Two things follow from taking the contract seriously rather than reading a dict:

* a parameter the contract does not have is a **question**, not a default, so :func:`missing` says exactly which slots
  have to be answered before the model can be asked anything;
* the answer carries the provenance back, so a prediction can say "the chord was stated, the incidence was defaulted
  from the family's reference" instead of presenting both as equally known.

This module does not compile sentences and does not run the intent ladder. It reads the contract that stage produced and
turns it into the one thing the model needs, which is the reason the two capabilities fit: the intent layer knows how to
refuse an under-specified question, and the model layer knows what a well-specified one is worth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Answer", "answer", "design_from_contract", "missing", "spec_from_contract"]


def _needed(family: Any) -> dict[str, str]:
    """Every name the family needs, and which of the two halves of a spec it belongs to."""
    out = {name: "geometry" for name in family.geometry_params}
    out.update({name: "operating" for name in family.conditioning_params})
    return out


def missing(family: Any, contract: Any, mapping: dict[str, str]) -> dict[str, str]:
    """The parameters the contract cannot supply, as ``{family parameter: contract value name}``.

    A slot the contract carries but marks missing is reported here too: the compiler already decided that value is
    unknown, and a default chosen downstream would be this stage overriding that decision.
    """
    out = {}
    for name in _needed(family):
        key = mapping.get(name, "")
        quantity = contract.quantity(key) if key and key in getattr(contract, "values", {}) else None
        if quantity is None or not quantity.known:
            out[name] = key or "(no mapping)"
    return out


def spec_from_contract(contract: Any, family: Any, mapping: dict[str, str]) -> Any:
    """The contract as a :class:`~pinneapple_neural.lpm_pipeline.DesignSpec` for this family.

    The mapping is ``{family parameter: contract value name}`` and has to cover the family: a design whose operating
    point is half known would be predicted at a condition nobody stated, so an uncovered parameter raises with the
    names of all of them at once rather than one at a time.
    """
    from ..lpm_pipeline import DesignSpec

    gaps = missing(family, contract, mapping)
    if gaps:
        detail = ", ".join(f"{name} <- {source}" for name, source in sorted(gaps.items()))
        raise ValueError(f"{family.name}: the contract does not supply every parameter the family needs ({detail}); "
                         f"the contract carries {sorted(getattr(contract, 'values', {}))} and reports its own "
                         f"questions in .questions")
    geometry: dict[str, float] = {}
    operating: dict[str, float] = {}
    for name, half in _needed(family).items():
        quantity = contract.quantity(mapping[name])
        (geometry if half == "geometry" else operating)[name] = float(quantity.value)
    label = f"contract:{getattr(contract, 'id', '')[:12]}"
    # the contract's values are physical, not draws: ``geometry_from_unit`` is the sampler's hook and is not applied
    return DesignSpec(geometry=geometry, operating=operating, label=label)


def design_from_contract(contract: Any, family: Any, mapping: dict[str, str]) -> Any:
    """The contract built into a :class:`~pinneapple_neural.lpm.Design`, through the family's own reference."""
    from ..lpm_pipeline import build

    return build(family, spec_from_contract(contract, family, mapping))


@dataclass
class Answer:
    """What the model says about one contract: the field, the confidence, the verdict, and where each number came from."""

    family: str
    contract_id: str
    prediction: Any
    spread: Any
    confidence: dict[str, Any]
    verdict: str
    quantities: dict[str, dict[str, Any]]
    field_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "contract_id": self.contract_id,
                "field_names": list(self.field_names),
                "confidence": self.confidence, "verdict": self.verdict, "quantities": self.quantities,
                "n_points": int(getattr(self.prediction, "shape", (0,))[0])}


def answer(model: Any, family: Any, contract: Any, mapping: dict[str, str], *, calibration: Any = None) -> Answer:
    """Predict the contract's design and say what the answer is worth.

    The verdict is the same three-way reading the rest of the pipeline uses: inside the trained family or flagged
    outside it (:func:`~pinneapple_neural.lpm_stages.trust.confidence`), and — whatever the flag says — not verified
    until somebody independent reviews the evidence. A field with no verdict is the thing this whole package exists to
    stop shipping.
    """
    from . import trust as trust_stage

    design = design_from_contract(contract, family, mapping)
    # the calibration knots come from a training set and are passed in; fitting them on one design would be a
    # calibration with nothing to calibrate against
    confidence = trust_stage.confidence(model, design, calibration)
    import torch

    with torch.no_grad():
        report = model.predict(design)
    verdict = ("outside the trained family: the answer is an extrapolation"
               if confidence["ood"] else "inside the trained family")
    quantities = {}
    for ours, theirs in mapping.items():
        quantity = contract.quantity(theirs)
        quantities[ours] = {"from": theirs, "value": quantity.value, "unit": quantity.unit,
                            "provenance": quantity.provenance, "source": quantity.source}
    return Answer(family=family.name, contract_id=str(getattr(contract, "id", "")),
                  prediction=report["mean"].detach().cpu().numpy(),
                  spread=report["std"].detach().cpu().numpy(), confidence=confidence, verdict=verdict,
                  quantities=quantities, field_names=tuple(family.field_names))
