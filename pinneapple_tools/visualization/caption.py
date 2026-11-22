"""The caption a figure carries: what the field is, in what unit, at what scale (#67).

A rendered field without its unit is a picture of colours. The rule this module enforces is the one the roadmap
states: **a field is not drawn without a unit** -- and a dimensionless field is what ``"-"`` says, so the absence
of a unit has to be declared, never assumed. The second half of the rule is the scale: a deformed shape is an
exaggeration, and an exaggerated deformation that does not say ``x25`` is a lie of proportion.

    from pinneapple_tools.visualization.caption import figure_caption

    figure_caption("von Mises", "MPa", value_range=(0.0, 512.4), deform=25.0, method="FEM",
                   status="validated").text()
    # 'von Mises [MPa]: 0 to 512.4 MPa, displacement x25, FEM (validated)'

:func:`figure_caption` is what the renderers call; :class:`FigureCaption` is the structured form, so a caller can
put the same facts in a JSON sidecar, an HTML tooltip or a paper's figure list without parsing the sentence back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DIMENSIONLESS = "-"
STATUSES = ("validated", "experimental")


def _number(value: float) -> str:
    """Four significant figures, and no trailing zeros: ``512.4``, ``0.0125``, ``1e+06``."""
    return f"{value:.4g}"


@dataclass(frozen=True)
class FigureCaption:
    """The facts a figure must state about itself. Built by :func:`figure_caption`."""

    field: str
    unit: str
    label: str = ""
    kind: str = "value"
    value_range: tuple[float, float] | None = None
    deform: float | None = None
    length_unit: str = "m"
    time: float | None = None
    time_unit: str = "s"
    method: str = ""
    status: str = ""
    source: str = ""

    @property
    def dimensionless(self) -> bool:
        return self.unit in (DIMENSIONLESS, "")

    @property
    def unit_text(self) -> str:
        """The unit as it goes on the image: ``"-"`` states a dimensionless field, it does not hide one."""
        return "dimensionless" if self.dimensionless else self.unit

    def text(self) -> str:
        """One line: the field and its unit, the range, the exaggeration, the time, the method and its status."""
        parts = [f"{self.label or self.field} [{self.unit_text}]"]
        if self.value_range is not None:
            lo, hi = self.value_range
            parts.append(f"{_number(lo)} to {_number(hi)}" + ("" if self.dimensionless else f" {self.unit}"))
        if self.deform is not None and self.deform != 1.0:
            # The exaggeration and the length unit are two different facts: x25 says how drawn displacements
            # relate to the model, "lengths in mm" says what a coordinate on the box means.
            parts.append(f"deformation x{_number(self.deform)}")
            if self.length_unit:
                parts.append(f"lengths in {self.length_unit}")
        if self.time is not None:
            parts.append(f"t = {_number(self.time)} {self.time_unit}".rstrip())
        if self.method:
            parts.append(f"{self.method} ({self.status})" if self.status else self.method)
        elif self.status:
            parts.append(self.status)
        return ": ".join(parts[:1]) + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "")

    def to_dict(self) -> dict[str, Any]:
        """The same facts as data, for a sidecar file or an application's tooltip."""
        return {"field": self.field, "label": self.label, "unit": self.unit, "kind": self.kind,
                "value_range": list(self.value_range) if self.value_range else None, "deform": self.deform,
                "length_unit": self.length_unit, "time": self.time, "time_unit": self.time_unit,
                "method": self.method, "status": self.status, "source": self.source, "text": self.text()}


def figure_caption(field: str, unit: str, *, label: str = "", kind: str = "value", value_range=None,
                   deform: float | None = None, length_unit: str = "m", time: float | None = None,
                   time_unit: str = "s", method: str = "", status: str = "", source: str = "") -> FigureCaption:
    """The caption for one field, refusing the two things a figure must not leave out.

    ``unit`` has no default on purpose: a caller that does not know the unit of the field is drawing a picture of
    colours, and ``unit="-"`` is how a dimensionless field says so. ``deform`` is the exaggeration factor the image
    used (``25`` for a displacement scaled x25); ``1.0`` means undivided, and is not printed.

    Raises
    ------
    ValueError
        When ``unit`` is ``None`` (use ``"-"`` for dimensionless), when ``status`` is not one of
        :data:`STATUSES`, or when ``value_range`` is not an increasing pair.
    """
    if unit is None:
        raise ValueError(
            "a field is not drawn without a unit: pass its unit (a field of stress is 'MPa'), or '-' for a "
            "dimensionless field. A figure that does not say either is a picture of colours.")
    if status and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES} or empty; got {status!r}")
    rng = None
    if value_range is not None:
        lo, hi = (float(v) for v in value_range)
        if not hi >= lo:
            raise ValueError(f"value_range must be (low, high) with high >= low; got ({lo}, {hi})")
        rng = (lo, hi)
    if deform is not None and float(deform) < 0:
        raise ValueError(f"deform is an exaggeration factor and cannot be negative; got {deform}")
    return FigureCaption(field=str(field), unit=str(unit), label=str(label), kind=str(kind), value_range=rng,
                         deform=None if deform is None else float(deform), length_unit=str(length_unit),
                         time=None if time is None else float(time), time_unit=str(time_unit),
                         method=str(method), status=str(status), source=str(source))


__all__ = ["FigureCaption", "figure_caption", "DIMENSIONLESS", "STATUSES"]
