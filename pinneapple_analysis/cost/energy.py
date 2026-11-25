"""Joules and emissions per operation, from a method that is named, sourced and printed with the number.

Time and FLOPs answer "how long" and "how much work"; neither answers "what does it cost to run". This module adds
the energy side of the cost report (roadmap item K20, issue #119): one number of joules per operation, produced by
a method that is **printed with the number**, together with the source of its constant and the hardware that
constant describes. An estimate whose method is not in the output is a guess with a decimal point, so
:meth:`EnergyEstimate.report` always writes method, hardware and source under the number.

The three methods (``method="auto"`` tries ``flops``, then ``power_time``; ``rapl`` is asked for by name because it
measures and therefore has to run the operation):

=================  ==========================  ==========================================================================
method             joules =                    when it applies
=================  ==========================  ==========================================================================
``flops``          FLOPs x pJ per FLOP         the operation has a countable FLOP count (``CostRecord.flops``, PyTorch
                                               operators only) and a declared energy per FLOP
``power_time``     power (W) x time (s)        no FLOP count, but the caller declares the draw of the device **and**
                                               where that draw comes from (a meter, a datasheet, a TDP)
``rapl``           measured counter delta      Linux exposes Intel RAPL energy counters (sysfs powercap); the only
                                               method here that measures instead of estimating
=================  ==========================  ==========================================================================

What each number is worth (read before quoting one):

- ``flops`` counts the arithmetic only. One FLOP here is one 32-bit multiplication *or* one addition, the
  convention of ``torch.utils.flop_counter`` (a multiply-add is 2 FLOPs, which is how a matmul is counted). The
  declared table gives the ALU energy at 45 nm; data movement is not in it, and a 32-bit read from DRAM at the
  same node costs 640 pJ, so an operation that streams data costs far more than the arithmetic line says. Treat
  this number as a **floor** -- which is the point of the source below.
- ``power_time`` is only as good as the declared draw, and a draw is a declaration, not a measurement: an idle
  CPU and the same CPU under AVX-512 are an order of magnitude apart. ``power_source`` is required for that
  reason: a watt with no provenance is exactly the invented number this module refuses to print.
- ``rapl`` measures the CPU package (and its subzones, where the kernel exposes them), not the wall: it excludes
  PSU losses and, on a machine with a discrete GPU, the GPU. It is Linux-only and needs the counters readable
  (``/sys/class/powercap/intel-rapl:*/energy_uj`` is often root-only). When it is not available this module
  returns no number and says why, instead of falling back to an estimate silently.

The declared table
------------------
:data:`ENERGY_PER_FLOP` holds the entries this library is willing to stand behind, one today, and a caller can
declare its own with ``energy_per_flop_pj=`` and ``energy_per_flop_source=`` rather than wait for a table entry:

===========  ===========  =============================================================================
key          pJ per FLOP  source
===========  ===========  =============================================================================
``cpu_45nm`` 2.0          Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC
                          2014, 45 nm CMOS table: 3.1 pJ for a 32-bit floating-point multiply and 0.9 pJ
                          for a 32-bit floating-point add, so one multiply-add (2 FLOPs) is 4.0 pJ and
                          one FLOP is 2.0 pJ
===========  ===========  =============================================================================

Some reproductions of that slide quote 3.7 pJ for the multiply; the entry states its own two addends so the
arithmetic can be checked by hand, and a caller who needs the other number declares it explicitly.

Emissions (``co2_g``) are computed only when the caller declares a grid carbon intensity **and** its source:
``joules / 3.6e6 * gCO2_per_kWh``. No default intensity ships here, because a number that depends on where and
when the job ran is the caller's to declare, not the library's to assume.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pinneapple_core.errors import BadValue

__all__ = [
    "DEFAULT_HARDWARE", "ENERGY_PER_FLOP", "EnergyEstimate", "EnergyModel", "RAPLMeter", "RAPL_ROOT",
    "estimate_energy", "fmt_joules", "measure_energy", "rapl_energy_uj", "rapl_packages",
]

#: Where the kernel exposes the RAPL counters (Linux, ``powercap`` class).
RAPL_ROOT = "/sys/class/powercap"

#: The table entry used when the caller does not declare a hardware: named, sourced, and printed as an assumption.
DEFAULT_HARDWARE = "cpu_45nm"

_JOULES_PER_PJ = 1e-12
_JOULES_PER_KWH = 3.6e6


@dataclass(frozen=True)
class EnergyModel:
    """One declared way to turn counted FLOPs into joules, with the source of the number."""
    pj_per_flop: float
    source: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"pj_per_flop": self.pj_per_flop, "source": self.source, "note": self.note}


#: The declared energy-per-FLOP table. Adding a row means adding a source for it, never a plausible number.
ENERGY_PER_FLOP: dict[str, EnergyModel] = {
    "cpu_45nm": EnergyModel(
        pj_per_flop=2.0,
        source=("Horowitz, 'Computing's Energy Problem (and what we can do about it)', ISSCC 2014, 45 nm CMOS: "
                "3.1 pJ per 32-bit floating-point multiply + 0.9 pJ per 32-bit floating-point add = 4.0 pJ per "
                "multiply-add, and a multiply-add is 2 FLOPs, so 2.0 pJ per FLOP"),
        note=("the ALU only: data movement is not counted (a 32-bit DRAM read is 640 pJ at the same node), so an "
              "operation that streams data costs more than this line reports"),
    ),
}

_RAPL_SOURCE = ("Intel RAPL, measured: the energy_uj counters under /sys/class/powercap were read before and after "
                "the call, so the number is a counter delta rather than an estimate")
_RAPL_HARDWARE = "intel-rapl package counters (CPU; excludes PSU losses and any discrete GPU)"

_METHODS = ("auto", "flops", "power_time")


# --------------------------------------------------------------------------- the estimate
@dataclass
class EnergyEstimate:
    """Joules for one operation, with the method, the hardware and the source the number came from.

    ``joules`` is ``None`` whenever no method applies: that is the honest answer for work the library cannot price
    (no FLOP count and no declared draw), and ``reason`` says which inputs would have made it possible.
    """
    joules: float | None = None
    method: str = "unavailable"            # flops | power_time | rapl | unavailable
    source: str = ""
    hardware: str = ""
    hardware_declared: bool = False        # False: the table default was assumed, and ``assumptions`` says so
    measured: bool = False                 # True only for rapl: a counter was read, nothing was estimated
    pj_per_flop: float | None = None
    flops: float | None = None
    time_s: float | None = None
    power_w: float | None = None
    co2_g: float | None = None
    carbon_source: str = ""
    assumptions: list[str] = field(default_factory=list)
    note: str = ""
    reason: str = ""                       # why there is no number, when there is none
    fix: str = ""                          # what would make a number possible

    @property
    def available(self) -> bool:
        return self.joules is not None

    def to_dict(self) -> dict[str, Any]:
        return {"joules": self.joules, "method": self.method, "source": self.source, "hardware": self.hardware,
                "hardware_declared": self.hardware_declared, "measured": self.measured,
                "pj_per_flop": self.pj_per_flop, "flops": self.flops, "time_s": self.time_s,
                "power_w": self.power_w, "co2_g": self.co2_g, "carbon_source": self.carbon_source,
                "assumptions": list(self.assumptions), "note": self.note, "reason": self.reason, "fix": self.fix}

    def derivation(self) -> str:
        """The arithmetic behind ``joules``, written out so a reader can redo it."""
        if self.joules is None:
            return ""
        if self.method == "flops":
            return (f"{_fmt_number(self.flops)} FLOPs x {_fmt_number(self.pj_per_flop)} pJ/FLOP = "
                    f"{fmt_joules(self.joules)}")
        if self.method == "power_time":
            return f"{_fmt_number(self.power_w)} W x {_fmt_number(self.time_s)} s = {fmt_joules(self.joules)}"
        if self.method == "rapl":
            return f"{fmt_joules(self.joules)} read from the RAPL counters (measured, not estimated)"
        return ""

    def report(self) -> str:
        """The number and its provenance, as the lines a cost report prints under the table."""
        if self.joules is None:
            named = f" (method: {self.method})" if self.method not in ("none", "unavailable") else ""
            lines = [f"energy: not reported{named}"]
            if self.reason:
                lines.append(f"  why: {self.reason}")
            if self.fix:
                lines.append(f"  fix: {self.fix}")
            return "\n".join(lines)
        lines = [f"energy: {fmt_joules(self.joules)} (method: {self.method}; "
                 f"{'measured' if self.measured else 'estimated'})",
                 f"  hardware: {self.hardware or 'unspecified'} "
                 f"({'declared' if self.hardware_declared else 'assumed, not declared'})"]
        if self.source:
            lines.append(f"  source: {self.source}")
        derivation = self.derivation()
        if derivation:
            lines.append(f"  derivation: {derivation}")
        if self.co2_g is not None:
            lines.append(f"  emissions: {self.co2_g:.4g} gCO2eq ({self.carbon_source})")
        else:
            lines.append("  emissions: not reported (no grid carbon intensity was declared)")
        if self.note:
            lines.append(f"  worth: {self.note}")
        for assumption in self.assumptions:
            lines.append(f"  assumed: {assumption}")
        return "\n".join(lines)


def _fmt_number(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.4g}" if abs(x) < 1e6 else f"{x:.6g}"


def fmt_joules(joules: float | None) -> str:
    """Joules with a readable unit (pJ ... kJ); ``-`` when there is no number."""
    if joules is None or not math.isfinite(joules):
        return "-"
    for scale, unit in ((1e3, "kJ"), (1.0, "J"), (1e-3, "mJ"), (1e-6, "µJ"), (1e-9, "nJ"), (1e-12, "pJ")):
        if abs(joules) >= scale:
            return f"{joules / scale:.3g} {unit}"
    return f"{joules / 1e-15:.3g} fJ"


# --------------------------------------------------------------------------- the methods
def _model_for(hardware: str | None, pj_per_flop: float | None, source: str,
               assume: list[str]) -> tuple[EnergyModel, str, bool]:
    """``(model, hardware label, declared)``: the table row asked for, or the default one with an assumption."""
    if pj_per_flop is not None:
        if not str(source).strip():
            raise BadValue(
                "energy_per_flop_pj was declared without energy_per_flop_source",
                fix="pass energy_per_flop_source='where this pJ/FLOP comes from' (a measurement, a datasheet, a "
                    "paper) or use hardware= one of " + ", ".join(sorted(ENERGY_PER_FLOP)),
                docs="docs/dev/errors_that_teach.md")
        if not (pj_per_flop > 0 and math.isfinite(pj_per_flop)):
            raise BadValue(f"energy_per_flop_pj must be a positive finite number, got {pj_per_flop}",
                           fix="pass the pJ per FLOP the source states, e.g. energy_per_flop_pj=2.0")
        return EnergyModel(float(pj_per_flop), str(source)), "caller-declared", True
    if hardware is None:
        assume.append(f"no hardware was declared: the table entry {DEFAULT_HARDWARE!r} was used "
                      f"({ENERGY_PER_FLOP[DEFAULT_HARDWARE].pj_per_flop} pJ/FLOP). Declare hardware=... or "
                      f"energy_per_flop_pj= to price this operation on the machine it actually runs on")
        return ENERGY_PER_FLOP[DEFAULT_HARDWARE], DEFAULT_HARDWARE, False
    if hardware not in ENERGY_PER_FLOP:
        raise BadValue(
            f"unknown hardware {hardware!r}: the declared table has {', '.join(sorted(ENERGY_PER_FLOP))}",
            fix="use one of those keys, or declare your own with energy_per_flop_pj= and energy_per_flop_source="
                " (a number the library does not have stays out of the table)",
            docs="docs/dev/errors_that_teach.md")
    return ENERGY_PER_FLOP[hardware], hardware, True


def _with_carbon(estimate: EnergyEstimate, carbon_g_per_kwh: float | None, carbon_source: str) -> EnergyEstimate:
    if carbon_g_per_kwh is None:
        return estimate
    if not str(carbon_source).strip():
        raise BadValue(
            "carbon_g_per_kwh was declared without carbon_source",
            fix="pass carbon_source='where this intensity comes from' (the grid operator's year and region, a "
                "published dataset) -- an intensity with no provenance is an invented emission",
            docs="docs/dev/errors_that_teach.md")
    if not (carbon_g_per_kwh >= 0 and math.isfinite(carbon_g_per_kwh)):
        raise BadValue(f"carbon_g_per_kwh must be a finite number >= 0, got {carbon_g_per_kwh}",
                       fix="pass the intensity in gCO2eq per kWh, e.g. carbon_g_per_kwh=150")
    if estimate.joules is not None:
        estimate.co2_g = estimate.joules / _JOULES_PER_KWH * float(carbon_g_per_kwh)
        estimate.carbon_source = str(carbon_source)
    return estimate


def estimate_energy(*, flops: float | None = None, time_s: float | None = None, method: str = "auto",
                    hardware: str | None = None, energy_per_flop_pj: float | None = None,
                    energy_per_flop_source: str = "", power_w: float | None = None, power_source: str = "",
                    carbon_g_per_kwh: float | None = None, carbon_source: str = "") -> EnergyEstimate:
    """Joules for one operation, by the named method, with the source and hardware it assumed.

    ``method="auto"`` uses ``flops`` when a FLOP count is given, ``power_time`` when only a declared draw is, and
    ``rapl`` never (it measures, so it is asked for by name: run :func:`measure_energy`). Missing inputs do not
    raise: they return an estimate with ``joules=None`` and the ``reason`` and ``fix`` a report can print.

    >>> estimate_energy(flops=1e12, hardware="cpu_45nm").joules
    2.0
    >>> estimate_energy(flops=65536).joules          # the default entry, marked as an assumption
    1.31072e-07
    >>> estimate_energy(time_s=2.0, power_w=65.0, power_source="wall meter on this host").joules
    130.0
    """
    if method not in _METHODS:
        hint = ("measure, not estimate: call measure_energy(fn) and it returns the estimate with method 'rapl'"
                if method == "rapl" else "use one of " + ", ".join(_METHODS))
        raise BadValue(f"unknown energy method {method!r}", fix=hint, docs="docs/dev/errors_that_teach.md")
    if flops is not None and not (flops >= 0 and math.isfinite(flops)):
        raise BadValue(f"flops must be a finite number >= 0, got {flops}",
                       fix="pass the FLOP count a measurement returned (CostRecord.flops), or omit it")
    if power_w is not None:
        if not (power_w >= 0 and math.isfinite(power_w)):
            raise BadValue(f"power_w must be a finite number >= 0, got {power_w}",
                           fix="pass the draw in watts, e.g. power_w=65.0")
        if not str(power_source).strip():
            raise BadValue(
                "power_w was declared without power_source",
                fix="pass power_source='where this draw comes from' (a wall meter, a datasheet TDP, the "
                    "accelerator's board power) -- the library will not price joules on an unsourced watt",
                docs="docs/dev/errors_that_teach.md")

    if method == "auto":
        method = "flops" if flops is not None else ("power_time" if power_w is not None else "none")

    assumptions: list[str] = []
    if method == "none":
        return _with_carbon(EnergyEstimate(
            method="none", reason="no FLOP count and no declared power draw, so no method applies",
            fix="pass flops= (a CostRecord's flops), or time_s with power_w and its power_source, or measure on a "
                "Linux host with measure_energy(fn)"), carbon_g_per_kwh, carbon_source)
    if method == "flops":
        if flops is None:
            return _with_carbon(EnergyEstimate(
                method="flops", reason="the flops method needs a FLOP count and none was given",
                fix="pass flops= (a CostRecord's flops) or use power_time with power_w and its power_source"),
                carbon_g_per_kwh, carbon_source)
        model, label, declared = _model_for(hardware, energy_per_flop_pj, energy_per_flop_source, assumptions)
        joules = float(flops) * model.pj_per_flop * _JOULES_PER_PJ
        return _with_carbon(EnergyEstimate(
            joules=joules, method="flops", source=model.source, hardware=label, hardware_declared=declared,
            pj_per_flop=model.pj_per_flop, flops=float(flops), time_s=time_s, assumptions=assumptions,
            note=model.note), carbon_g_per_kwh, carbon_source)
    if method == "power_time":
        missing = [n for n, v in (("power_w", power_w), ("time_s", time_s)) if v is None]
        if missing:
            return _with_carbon(EnergyEstimate(
                method="power_time", reason=f"the power_time method needs {', '.join(missing)}",
                fix="pass the draw in watts with its power_source and the measured time_s"), carbon_g_per_kwh,
                carbon_source)
        return _with_carbon(EnergyEstimate(
            joules=float(power_w) * float(time_s), method="power_time", source=str(power_source),
            hardware="the device the declared draw describes", hardware_declared=True, power_w=float(power_w),
            time_s=float(time_s), assumptions=assumptions,
            note="the draw is a declaration, not a measurement: an idle device and a loaded one differ by an "
                 "order of magnitude"), carbon_g_per_kwh, carbon_source)
    raise BadValue(f"unknown energy method {method!r}", fix="use one of " + ", ".join(_METHODS),
                   docs="docs/dev/errors_that_teach.md")  # pragma: no cover - _METHODS is checked above


# --------------------------------------------------------------------------- RAPL: the measured method
def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def rapl_packages(root: str | Path = RAPL_ROOT) -> list[Path]:
    """The top-level counter directories (one per CPU package), never the subzones that they already include.

    ``intel-rapl:0`` sums its own subzones (``intel-rapl:0:0`` core, ``:1`` uncore, ...), so adding both would
    count the same joules twice. The ``intel-rapl-mmio`` interface is used only when no ``intel-rapl`` counter is
    readable, because both describe the same packages. AMD's ``amd_energy`` hwmon is not read here.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    for pattern in ("intel-rapl:*", "intel-rapl-mmio:*"):
        found = [d for d in sorted(base.glob(pattern))
                 if d.is_dir() and d.name.count(":") == 1 and (d / "energy_uj").exists()]
        if found:
            return found
    return []


def rapl_energy_uj(root: str | Path = RAPL_ROOT) -> float | None:
    """The sum of the package energy counters in microjoules, or ``None`` when none is readable."""
    values = [v for v in (_read_int(p / "energy_uj") for p in rapl_packages(root)) if v is not None]
    return float(sum(values)) if values else None


class RAPLMeter:
    """Energy a block of code consumed, from the RAPL counters read around it (Linux ``powercap``).

    ``joules`` is ``None`` and ``reason`` says why when the counters are not there or not readable, so a report can
    print the absence instead of an estimate wearing a measurement's clothes.
    """

    def __init__(self, root: str | Path = RAPL_ROOT):
        self.root = root
        self.joules: float | None = None
        self.reason = ""
        self._before: list[tuple[int, int]] = []

    def _snapshot(self) -> list[tuple[int, int]]:
        packages = rapl_packages(self.root)
        if not packages:
            self.reason = (f"no RAPL counter under {self.root} (Linux with the powercap module; AMD's amd_energy "
                           f"hwmon is not read here)")
            return []
        snapshot: list[tuple[int, int]] = []
        for package in packages:
            uj = _read_int(package / "energy_uj")
            if uj is None:
                self.reason = (f"the counter {package / 'energy_uj'} could not be read: it is root-only on many "
                               f"distributions, so read it as root or give this user access")
                return []
            snapshot.append((uj, _read_int(package / "max_energy_range_uj") or 0))
        return snapshot

    def __enter__(self) -> RAPLMeter:
        self._before = self._snapshot()
        return self

    def __exit__(self, *exc) -> None:
        after = self._snapshot()
        if not self._before or len(after) != len(self._before):
            self.joules = None
            return
        total_uj = 0
        for (before, before_max), (now, now_max) in zip(self._before, after, strict=True):
            delta = now - before
            if delta < 0:                    # the counter wrapped: max_energy_range_uj is its modulus
                modulus = before_max or now_max
                delta += modulus if modulus > 0 else 0
            total_uj += delta
        self.joules = total_uj * 1e-6

    @property
    def available(self) -> bool:
        return self.joules is not None


def measure_energy(fn, *args, repeats: int = 3, root: str | Path = RAPL_ROOT, **kwargs) -> EnergyEstimate:
    """Run ``fn(*args, **kwargs)`` ``repeats`` times and return the median RAPL energy of one run, measured.

    This is the only method that measures. It reports the CPU package (and its subzones where the kernel exposes
    them) and nothing else: PSU losses and a discrete GPU are outside the counters, so treat it as a floor of the
    whole-machine draw, not as the wall.
    """
    if repeats < 1:
        raise BadValue(f"repeats must be >= 1, got {repeats}", fix="pass repeats=3 (the default) or any integer >= 1")
    values: list[float] = []
    for _ in range(repeats):
        with RAPLMeter(root=root) as meter:
            fn(*args, **kwargs)
        if meter.joules is None:
            return EnergyEstimate(method="rapl", source=_RAPL_SOURCE, hardware=_RAPL_HARDWARE,
                                  hardware_declared=True, reason=meter.reason,
                                  fix="run on a Linux host with readable /sys/class/powercap counters, or "
                                      "declare power_w with its power_source and use method='power_time'")
        values.append(meter.joules)
    return EnergyEstimate(joules=float(statistics.median(values)), method="rapl", source=_RAPL_SOURCE,
                          hardware=_RAPL_HARDWARE, hardware_declared=True, measured=True, time_s=None,
                          note=(f"median of {len(values)} counter reads; the CPU package only, so PSU losses and "
                                f"any discrete GPU are outside the number"))
