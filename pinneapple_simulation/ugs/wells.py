"""Wells and schedules for the underground-storage simulator.

A :class:`Well` is a completion in one grid cell, with a Peaceman index taken from the grid (or given), a
minimum and a maximum bottom-hole pressure and a maximum rate. A :class:`Period` says how each well is to be
operated during a stretch of the schedule -- a rate target with a bottom-hole-pressure limit, or a
bottom-hole pressure -- and a :class:`Schedule` is a list of them. A well the period does not name is shut in,
which is what makes a season (injection, shut-in, withdrawal) a few lines.

:func:`operating_point` resolves one well against one cell pressure by the active-set rule every commercial
simulator uses: the target is honoured unless it needs a bottom-hole pressure outside ``[bhp_min, bhp_max]``,
or a rate above ``rate_max``, in which case the binding limit takes over. The derivative it returns is the one
the implicit Jacobian needs, taken on the same branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Well", "Control", "Period", "Schedule", "operating_point"]

_INF = float("inf")


@dataclass(frozen=True)
class Well:
    """One completion in one grid cell.

    ``bhp_min`` and ``bhp_max`` bound the bottom-hole pressure the well may be operated at [Pa]; ``rate_max``
    is the surface rate limit [m3 standard per second], the erosion and sand-control limit of the operating
    issue. ``wi`` overrides the index the grid computes, for a well whose completion is not a plain Peaceman
    block.
    """

    name: str
    cell: int
    r_w: float = 0.1               # wellbore radius [m]
    skin: float = 0.0              # dimensionless skin
    bhp_min: float | None = None   # Pa
    bhp_max: float | None = None   # Pa
    rate_max: float | None = None  # m3 standard / s
    wi: float | None = None        # m3, explicit index; the grid's Peaceman index is used when it is None


@dataclass(frozen=True)
class Control:
    """How one well is operated during one period.

    ``mode`` is ``"rate"`` (``value`` in m3 standard per second, positive for injection) or ``"bhp"``
    (``value`` in Pa). Both are targets: the well's own limits still apply.
    """

    mode: str
    value: float

    def __post_init__(self) -> None:
        if self.mode not in ("rate", "bhp"):
            raise ValueError(f"unknown control mode {self.mode!r}; use 'rate' or 'bhp'")


@dataclass(frozen=True)
class Period:
    """``days`` of a schedule with one control per well; a well left out is shut in."""

    days: float
    controls: dict[str, Control] = field(default_factory=dict)
    label: str = ""


@dataclass(frozen=True)
class Schedule:
    """A list of periods, run in order."""

    periods: tuple[Period, ...]

    @property
    def days(self) -> float:
        return float(sum(p.days for p in self.periods))

    def controls_at(self, day: float) -> dict[str, Control]:
        """The controls in force on ``day`` (0 <= day < ``days``)."""
        t = 0.0
        for period in self.periods:
            if day < t + period.days - 1e-12:
                return period.controls
            t += period.days
        return self.periods[-1].controls if self.periods else {}

    def period_index(self, day: float) -> int:
        t = 0.0
        for i, period in enumerate(self.periods):
            if day < t + period.days - 1e-12:
                return i
            t += period.days
        return max(len(self.periods) - 1, 0)

    def steps(self, dt_days: float) -> list[tuple[float, int]]:
        """``(dt, period index)`` for every time step, split at the period boundaries."""
        out: list[tuple[float, int]] = []
        for i, period in enumerate(self.periods):
            n = max(1, int(round(period.days / dt_days)))
            dt = period.days / n
            out.extend((dt, i) for _ in range(n))
        return out


def operating_point(well: Well, control: Control | None, p_cell: float, mobility: float,
                    dmobility_dp: float, wi: float) -> tuple[float, float, float]:
    """Resolve a well's rate, bottom-hole pressure and ``dq/dp_cell`` at a cell pressure.

    ``mobility`` is ``(1/Bg)/mu`` at the cell and ``wi`` the well index [m3]. Returns ``(q, bhp, dq_dp)`` with
    ``q`` in m3 standard per second, positive for injection. A well with no control is shut in and returns
    zeros.
    """
    if control is None:
        return 0.0, float(p_cell), 0.0
    bhp_lo = -_INF if well.bhp_min is None else float(well.bhp_min)
    bhp_hi = _INF if well.bhp_max is None else float(well.bhp_max)
    q_lo = -_INF if well.rate_max is None else -float(well.rate_max)
    q_hi = _INF if well.rate_max is None else float(well.rate_max)
    if wi <= 0.0 or not np.isfinite(mobility) or mobility <= 0.0:
        raise ValueError(f"well {well.name!r}: index {wi:g} and mobility {mobility:g} must be positive")

    if control.mode == "rate":
        q_target = float(control.value)
        dq_target = 0.0
    else:
        bhp_target = min(max(float(control.value), bhp_lo), bhp_hi)
        q_target = wi * mobility * (bhp_target - p_cell)
        dq_target = wi * (dmobility_dp * (bhp_target - p_cell) - mobility)

    # the rate range the well's own pressure limits allow, intersected with the rate limit
    q_at_lo = wi * mobility * (bhp_lo - p_cell) if np.isfinite(bhp_lo) else -_INF
    q_at_hi = wi * mobility * (bhp_hi - p_cell) if np.isfinite(bhp_hi) else _INF
    lo = max(q_at_lo, q_lo)
    hi = min(q_at_hi, q_hi)
    if lo > hi:
        raise ValueError(f"well {well.name!r}: no feasible operating point at p = {p_cell:g} Pa "
                         f"(rate range [{lo:g}, {hi:g}] m3/s is empty)")

    if q_target >= hi:
        q = hi
        bhp = p_cell + hi / (wi * mobility)
        dq = 0.0 if q_hi == hi else wi * (dmobility_dp * (bhp - p_cell) - mobility)
    elif q_target <= lo:
        q = lo
        bhp = p_cell + lo / (wi * mobility)
        dq = 0.0 if q_lo == lo else wi * (dmobility_dp * (bhp - p_cell) - mobility)
    else:
        q = q_target
        bhp = p_cell + q_target / (wi * mobility)
        dq = dq_target
    return float(q), float(bhp), float(dq)
