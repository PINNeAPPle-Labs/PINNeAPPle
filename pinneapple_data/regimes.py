"""Which physical regimes a dataset covers, and which ones it is missing (#171).

An audit that measures coverage of the *design space* answers "are these samples spread out?". It does not answer
"is there any turbulent case in here?", and that is the question a model answers wrongly: a surrogate trained on
laminar pipe flow and asked about a turbulent one is not interpolating, it is extrapolating with confidence.

A regime is a band of one dimensionless number, so the labels come from a number that already carries the physics —
Reynolds, Mach, Prandtl — and not from a cluster the data happened to form:

    >>> from pinneapple_data.regimes import PRESETS, regime_coverage
    >>> cov = regime_coverage([1200, 1800, 9500], PRESETS["reynolds_pipe"])   # doctest: +SKIP
    >>> cov.counts, cov.missing                                              # doctest: +SKIP
    ({'laminar': 2, 'transitional': 0, 'turbulent': 1}, ['transitional'])

`missing_regions()` turns each empty band back into the interval it covers, which is what a sampler needs to fill it:
the audit says what is missing, and the same object says where to look.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Band", "PRESETS", "RegimeCoverage", "RegimeSpec", "regime_audit", "regime_coverage"]

#: a band thinner than this many samples is reported as thin rather than covered: two points in a regime is a claim
#: about that regime with two points behind it
MIN_COUNT = 2


@dataclass(frozen=True)
class Band:
    """One regime: ``label`` for ``low <= value < high``; ``None`` means unbounded on that side."""

    low: float | None
    high: float | None
    label: str

    def contains(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.low is not None and value < self.low:
            return False
        return not (self.high is not None and value >= self.high)

    def __str__(self) -> str:
        low = "-inf" if self.low is None else f"{self.low:g}"
        high = "inf" if self.high is None else f"{self.high:g}"
        return f"{self.label} [{low}, {high})"


@dataclass(frozen=True)
class RegimeSpec:
    """The bands of one dimensionless number, in order, with no gap between them.

    The bands have to tile the line: a sample that falls in no band is a hole in the spec, not a regime of its own, and
    the constructor refuses it instead of returning "unknown" for a value that is perfectly well defined."""

    name: str
    column: str
    bands: tuple[Band, ...]
    note: str = ""
    reference: str = ""

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError(f"regime {self.name!r} has no bands")
        labels = [b.label for b in self.bands]
        if len(set(labels)) != len(labels):
            raise ValueError(f"regime {self.name!r} repeats a label: {labels}")
        if self.bands[0].low is not None:
            raise ValueError(f"regime {self.name!r} starts at {self.bands[0].low:g}: the first band has to reach -inf")
        if self.bands[-1].high is not None:
            raise ValueError(f"regime {self.name!r} ends at {self.bands[-1].high:g}: the last band has to reach +inf")
        for left, right in zip(self.bands, self.bands[1:], strict=False):
            if left.high != right.low:
                raise ValueError(f"regime {self.name!r} has a gap or an overlap between {left.label!r} and "
                                 f"{right.label!r}: {left.high} != {right.low}")

    @property
    def labels(self) -> list[str]:
        return [b.label for b in self.bands]

    def label(self, value: float) -> str:
        """The band a value falls in; ``"missing"`` for NaN, which is a hole in the data and not a regime."""
        value = float(value)
        if math.isnan(value):
            return "missing"
        for band in self.bands:
            if band.contains(value):
                return band.label
        raise ValueError(f"{value!r} falls in no band of {self.name!r}")     # the constructor rules this out

    def band(self, label: str) -> Band:
        return next(b for b in self.bands if b.label == label)


#: regimes worth naming, with the thresholds the literature uses; a caller can always write its own spec
PRESETS: dict[str, RegimeSpec] = {
    "reynolds_pipe": RegimeSpec(
        "reynolds_pipe", "reynolds", (
            Band(None, 2300.0, "laminar"), Band(2300.0, 4000.0, "transitional"), Band(4000.0, None, "turbulent")),
        note="pipe flow: laminar below 2300, fully turbulent above 4000, the band between is not a regime",
        reference="Schlichting, Boundary-Layer Theory, 8th ed., ch. 16"),
    "reynolds_flat_plate": RegimeSpec(
        "reynolds_flat_plate", "reynolds", (
            Band(None, 5e5, "laminar"), Band(5e5, 3e6, "transitional"), Band(3e6, None, "turbulent")),
        note="flat plate: transition near Re_x = 5e5, fully turbulent past 3e6",
        reference="Incropera et al., Fundamentals of Heat and Mass Transfer, ch. 7"),
    "mach": RegimeSpec(
        "mach", "mach", (
            Band(None, 0.3, "incompressible"), Band(0.3, 0.8, "subsonic"), Band(0.8, 1.2, "transonic"),
            Band(1.2, 5.0, "supersonic"), Band(5.0, None, "hypersonic")),
        note="below 0.3 the density change is under 5 % and the flow is treated as incompressible",
        reference="Anderson, Modern Compressible Flow, 3rd ed., ch. 1"),
    "prandtl": RegimeSpec(
        "prandtl", "prandtl", (
            Band(None, 0.01, "liquid metal"), Band(0.01, 0.1, "low"), Band(0.1, 10.0, "gas or water"),
            Band(10.0, 1e3, "viscous"), Band(1e3, None, "very viscous")),
        note="order-of-magnitude bands, not sharp transitions",
        reference="Incropera et al., table A.4"),
    "rayleigh": RegimeSpec(
        "rayleigh", "rayleigh", (
            Band(None, 1e8, "laminar convection"), Band(1e8, 1e10, "transitional"), Band(1e10, None, "turbulent")),
        note="natural convection: the Rayleigh number plays the role Reynolds plays in forced convection",
        reference="Incropera et al., ch. 9"),
}


@dataclass
class RegimeCoverage:
    """How many samples each regime has, and which ones the dataset does not reach at all."""

    spec: RegimeSpec
    counts: dict[str, int] = field(default_factory=dict)
    n: int = 0
    min_count: int = MIN_COUNT

    @property
    def missing(self) -> list[str]:
        """Regimes with no sample: the dataset cannot say anything about them."""
        return [b.label for b in self.spec.bands if self.counts.get(b.label, 0) == 0]

    @property
    def thin(self) -> list[str]:
        """Regimes with fewer than ``min_count`` samples: present, but not enough to learn from."""
        return [b.label for b in self.spec.bands if 0 < self.counts.get(b.label, 0) < self.min_count]

    @property
    def covered(self) -> list[str]:
        return [b.label for b in self.spec.bands if self.counts.get(b.label, 0) >= self.min_count]

    @property
    def status(self) -> str:
        """``fail`` for a regime the data never reaches, ``warn`` for one it barely reaches, ``pass`` otherwise."""
        if self.missing:
            return "fail"
        return "warn" if self.thin else "pass"

    def share(self, label: str) -> float:
        return self.counts.get(label, 0) / self.n if self.n else 0.0

    def missing_regions(self) -> list[dict[str, Any]]:
        """The empty bands as intervals to sample in: ``{"column", "low", "high", "label"}``, ready for a sampler."""
        return [{"column": self.spec.column, "low": band.low, "high": band.high, "label": band.label}
                for band in (self.spec.band(label) for label in self.missing)]

    def to_dict(self) -> dict[str, Any]:
        return {"regime": self.spec.name, "column": self.spec.column, "n": self.n, "status": self.status,
                "counts": dict(self.counts), "share": {k: self.share(k) for k in self.spec.labels},
                "missing": self.missing, "thin": self.thin, "min_count": self.min_count,
                "sample_here": self.missing_regions()}

    def to_markdown(self) -> str:
        rows = ["| regime | samples | share |", "|---|---|---|"]
        for band in self.spec.bands:
            n = self.counts.get(band.label, 0)
            mark = " **missing**" if band.label in self.missing else (" *thin*" if band.label in self.thin else "")
            rows.append(f"| `{band.label}` ({band.low if band.low is not None else '-inf'} to "
                        f"{band.high if band.high is not None else 'inf'}) | {n}{mark} | {self.share(band.label):.1%} |")
        head = f"**{self.spec.name}** on `{self.spec.column}` — {self.n} samples, {self.status}"
        if self.spec.note:
            head += f"\n\n{self.spec.note}"
        return head + "\n\n" + "\n".join(rows)

    def __str__(self) -> str:
        counts = ", ".join(f"{b.label}={self.counts.get(b.label, 0)}" for b in self.spec.bands)
        tail = f"; missing {self.missing}" if self.missing else ""
        return f"{self.spec.name} over {self.n} samples: {counts}{tail}"


def regime_coverage(values: Sequence[float], spec: RegimeSpec, *, min_count: int = MIN_COUNT) -> RegimeCoverage:
    """Count the samples of ``values`` in each band of ``spec``."""
    counts = {label: 0 for label in spec.labels}
    n = 0
    for value in values:
        n += 1
        counts[spec.label(value)] = counts.get(spec.label(value), 0) + 1
    return RegimeCoverage(spec=spec, counts=counts, n=n, min_count=min_count)


def regime_audit(columns: Mapping[str, Sequence[float]], specs: Sequence[RegimeSpec | str], *,
                 min_count: int = MIN_COUNT) -> list[RegimeCoverage]:
    """Coverage for each spec, over the columns given. A spec whose column is absent is reported, not skipped."""
    out: list[RegimeCoverage] = []
    for spec in specs:
        spec = PRESETS[spec] if isinstance(spec, str) else spec
        values = columns.get(spec.column)
        if values is None:
            out.append(RegimeCoverage(spec=spec, counts={}, n=0, min_count=min_count))
            continue
        out.append(regime_coverage(values, spec, min_count=min_count))
    return out
