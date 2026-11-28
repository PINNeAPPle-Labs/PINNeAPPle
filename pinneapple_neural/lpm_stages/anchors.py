"""Stage 4: a measurement is evidence, and never training data.

A design space drawn here and solved here is scored against its own reference; a held-out split of that is still a
split of our own data. The one thing that can say whether a model is right about the world is a number somebody
*measured* — a bench test, a client's commissioning data, a published coefficient — and the rule this stage exists to
hold is that such a number never enters a fit. An anchor that became a training row would stop being evidence the
moment it was used, which is the whole reason to keep them in a separate object with a separate gate.

The anchor is a **scalar** derived from the field by a named functional (``mean``, ``max``, ``rms``, ...) because that is
what a measurement usually is: a pressure drop, a peak temperature, a mean heat transfer. The error it produces is a
:class:`~pinneapple_veriphysics.validation_case.ReferenceComparison` — the record VeriPhysics already keeps — so an
anchor taken here can be filed in an existing validation case rather than in a table of its own.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

__all__ = ["QUANTITIES", "Anchor", "comparisons", "gate", "measure", "value_of"]

#: The functionals an anchor may name. A measurement is usually one number, and which one has to be stated: the peak
#: and the mean of the same field differ by more than most models' error.
QUANTITIES: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda a: float(np.mean(a)),
    "max": lambda a: float(np.max(a)),
    "min": lambda a: float(np.min(a)),
    "rms": lambda a: float(np.sqrt(np.mean(a ** 2))),
    "abs_max": lambda a: float(np.max(np.abs(a))),
    "l2": lambda a: float(np.linalg.norm(a.ravel())),
}


@dataclass(frozen=True)
class Anchor:
    """One measurement: the design it was taken on, the quantity, the number, and how well it is known.

    ``tolerance_pct`` is the measurement's own tolerance, not the model's — the instrument's error, the repeatability,
    or the acceptance criterion of the test. It is what the prediction is judged against, and stating it is what turns
    "0.8 % off" into a verdict.
    """

    label: str
    design: Any
    quantity: str
    measured: float
    tolerance_pct: float
    source: str = ""
    independent: bool = True
    kind: str = "experimental data"
    field: int = 0
    unit: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.quantity not in QUANTITIES:
            raise ValueError(f"unknown quantity {self.quantity!r}; known: {', '.join(sorted(QUANTITIES))}")
        if self.tolerance_pct <= 0:
            raise ValueError("an anchor without a tolerance is a number nobody can be judged against")
        if not self.source:
            raise ValueError(f"{self.label}: an anchor has to say where the number came from (source=...)")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "design": getattr(self.design, "label", ""), "quantity": self.quantity,
                "measured": float(self.measured), "tolerance_pct": float(self.tolerance_pct), "unit": self.unit,
                "source": self.source, "independent": bool(self.independent), "kind": self.kind,
                "created_at": self.created_at}


def value_of(prediction: Any, quantity: str, field: int = 0) -> float:
    """The named functional of one field of a prediction, ``(Q, f)``."""
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"a prediction is (Q, f), got {values.shape}")
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; known: {', '.join(sorted(QUANTITIES))}")
    return QUANTITIES[quantity](values[:, int(field)])


def measure(anchor: Anchor, model: Any) -> dict[str, Any]:
    """Predict the anchor's design, reduce it the same way, and compare.

    The error is relative to the measured value, in percent, and it is the number the gate asks about. A measured
    value of zero has no relative error, so the comparison falls back to the prediction's magnitude — and says so
    rather than dividing by a floor and calling the result a percentage.
    """
    import torch  # local: this module reads, it does not need torch to import

    with torch.no_grad():
        prediction = model.predict(anchor.design)["mean"]
    predicted = value_of(prediction.detach().cpu().numpy(), anchor.quantity, anchor.field)
    if abs(anchor.measured) > 0:
        error_pct = abs(predicted - anchor.measured) / abs(anchor.measured) * 100.0
        metric = "|predicted - measured| / |measured|, percent"
    else:
        error_pct = abs(predicted) * 100.0
        metric = "the measured value is zero, so the error is the absolute one in percent"
    return {"label": anchor.label, "quantity": anchor.quantity, "predicted": predicted, "measured": float(anchor.measured),
            "error_pct": float(error_pct), "tolerance_pct": float(anchor.tolerance_pct),
            "passed": bool(error_pct <= anchor.tolerance_pct), "metric": metric, "unit": anchor.unit,
            "source": anchor.source, "independent": bool(anchor.independent)}


def gate(anchors: list[Anchor], model: Any, *, tolerance_pct: float | None = None) -> dict[str, Any]:
    """The ``anchor`` gate: what the model says about measurements it never saw, and the verdict on each.

    With no anchor registered the status is ``not applicable`` and the reason says so — an empty anchor list is not a
    pass, and a design space with no measurement is a design space with no external evidence.
    """
    if not anchors:
        return {"check": "anchor", "status": "not applicable", "value": None, "tolerance": tolerance_pct,
                "detail": "no measurement is registered for this design space: the only evidence is the family's own "
                          "reference and a held-out split of it", "rows": []}
    rows = [measure(a, model) for a in anchors]
    worst = max(rows, key=lambda r: r["error_pct"] / max(r["tolerance_pct"], 1e-12))
    failed = [r for r in rows if not r["passed"]]
    return {"check": "anchor", "status": "passed" if not failed else "failed",
            "value": float(np.mean([r["error_pct"] for r in rows])),
            "tolerance": tolerance_pct if tolerance_pct is not None else None,
            "detail": f"{len(rows) - len(failed)}/{len(rows)} measurement(s) within their own tolerance; the worst is "
                      f"{worst['label']} at {worst['error_pct']:.2f} % against {worst['tolerance_pct']:g} %",
            "rows": rows}


def comparisons(anchors: list[Anchor], model: Any) -> list[Any]:
    """The anchors as :class:`~pinneapple_veriphysics.validation_case.ReferenceComparison` records.

    This is the bridge that keeps the evidence in one place: an anchor registered here becomes a comparison a
    validation case can carry, with its source, its independence and its tolerance — instead of a number in a log.
    """
    from pinneapple_veriphysics.validation_case import ReferenceComparison

    out = []
    for anchor, row in zip(anchors, [measure(a, model) for a in anchors], strict=True):
        out.append(ReferenceComparison(
            evidence_id=anchor.label, kind=anchor.kind if anchor.kind != "none" else "experimental data",
            metric=f"{anchor.quantity} of field {anchor.field}" + (f" [{anchor.unit}]" if anchor.unit else ""),
            value_pct=row["error_pct"], tolerance_pct=row["tolerance_pct"], independent=anchor.independent,
            description=row["metric"], source=anchor.source, covers=(anchor.quantity,), created_at=anchor.created_at))
    return out
