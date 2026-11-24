"""From a measured cost to the machine to ask for (issue #131, roadmap item D8).

The library tags operations by resource kind with a static label ("CPU heavy", "GPU preferred"). A label is a
guess that goes stale; a measurement is not. This module derives, from what :mod:`profiler` actually measured:

* the **kind** of resource an operation wants (:func:`classify`);
* the **number of cores** to ask for, from a measured speedup curve (:func:`recommend_cores`);
* the **memory** to request per run, from the measured peak with a stated headroom (:func:`plan`);
* and a **job profile table** generated from those numbers, recording the machine it was measured on
  (:class:`JobProfileTable`) so a decision is never read as if it were made elsewhere.

The rules
---------
Written out, because a number without its rule is the static label again:

**Kind.** Derived in this order, first match wins. ``flops`` is PyTorch's countable work, ``peak_mem_bytes`` the
measured memory, ``time_s`` the median wall time of one run:

=================================  ==========================================================================
kind                               rule
=================================  ==========================================================================
``gpu_preferred``                  the peak was measured on a CUDA allocator (``mem_source == "cuda"``), or the
                                   measured work rate is at least 10 GFLOP/s (``flops / time_s``), which is where a
                                   CPU stops being the sensible device for the same bytes
``io_bound``                       nothing computes and nothing is held: no countable FLOPs **and** the peak is
                                   under 1 MiB per second of runtime. The time is spent waiting (disk, network,
                                   solver subprocess); more cores do not shorten it, faster storage does
``cpu_heavy``                      at least 0.5 GFLOP/s measured, **or** one run longer than 1 s with a peak
                                   above 64 MiB: it uses the machine rather than passing through it
``cpu_light``                      everything else, including an operation with no PyTorch FLOPs and no
                                   measurable memory (a check, a file edit, an index build)
=================================  ==========================================================================

**Cores.** From measurements of the *same* operation at several worker counts, ``speedup(n) = t(1) / t(n)`` and
``efficiency(n) = speedup(n) / n``. The recommendation is the largest measured ``n`` whose efficiency is still at
least ``efficiency_min`` (default 0.75: three quarters of the added core's capacity is used), reduced to the
smallest ``n`` that meets ``deadline_s`` when one is given. Never more than the largest count actually measured:
this module extrapolates nothing, and says so in the profile's ``notes`` when the deadline cannot be met inside
the measured range.

**Memory.** ``peak * headroom`` (default 1.25) per run, and the table reports the campaign total
``workers * per_run`` separately, because they are requested from different places. A peak that is a lower bound
(``mem_exact=False``, sampled RSS) is requested with the same arithmetic but its profile carries
``memory_is_floor=True``: the request is a floor, not a measurement, and a job that runs out of memory there was
never measured, only sampled.

Usage::

    from pinneapple_analysis.cost import CostLedger, JobProfileTable
    ledger = CostLedger(); ledger.measure("pinn_step", step)
    table = JobProfileTable.from_ledger(ledger, workers=1)
    table.profiles["pinn_step"].kind, table.profiles["pinn_step"].memory_bytes
    table.save("job_profiles.json")                     # with measured_on: the machine the numbers came from
"""
from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .profiler import CostRecord

__all__ = ["RESOURCE_KINDS", "GPU_PREFERRED", "IO_BOUND", "CPU_HEAVY", "CPU_LIGHT", "MachineRecord", "JobProfile",
           "JobProfileTable", "classify", "recommend_cores", "plan"]

GPU_PREFERRED, IO_BOUND, CPU_HEAVY, CPU_LIGHT = "gpu_preferred", "io_bound", "cpu_heavy", "cpu_light"
RESOURCE_KINDS = (CPU_LIGHT, CPU_HEAVY, GPU_PREFERRED, IO_BOUND)

#: Measured work rate from which a GPU is the sensible device (FLOP/s).
GPU_FLOPS_PER_S = 10e9
#: Measured work rate from which one run uses the machine rather than passing through it (FLOP/s).
HEAVY_FLOPS_PER_S = 0.5e9
#: One run longer than this with a peak above ``HEAVY_TIME_S``'s companion is heavy even without FLOP counts.
HEAVY_TIME_S = 1.0
HEAVY_MEMORY_BYTES = 64 * 1024 * 1024
#: Below this many bytes per second held, with no countable FLOPs, the operation is waiting rather than working.
IO_BYTES_PER_S = 1024 * 1024

MIB = 1024 * 1024


@dataclass
class MachineRecord:
    """The machine a number was measured on, so the table it lands in is never read as a universal claim."""
    system: str
    machine: str
    cpu_count: int
    python: str
    total_memory_bytes: int | None = None
    torch: str | None = None
    gpu: str | None = None

    @classmethod
    def here(cls) -> "MachineRecord":
        """This machine, as the profile records it."""
        total = None
        try:                                                 # POSIX: the RAM the scheduler will be asked for
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            total = None
        torch_version, gpu = None, None
        try:
            import torch
            torch_version = torch.__version__
            if torch.cuda.is_available() and torch.cuda.device_count():
                gpu = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return cls(system=platform.system(), machine=platform.machine(), cpu_count=os.cpu_count() or 1,
                   python=platform.python_version(), total_memory_bytes=total, torch=torch_version, gpu=gpu)

    def to_dict(self) -> dict[str, Any]:
        return {"system": self.system, "machine": self.machine, "cpu_count": self.cpu_count, "python": self.python,
                "total_memory_bytes": self.total_memory_bytes, "torch": self.torch, "gpu": self.gpu}


@dataclass
class JobProfile:
    """One operation and the resources its measurement says it needs."""
    name: str
    kind: str
    workers: int
    memory_bytes: int                        # host memory per run
    wall_time_s: float                       # expected wall time at ``workers``
    measured_on: MachineRecord
    device_memory_bytes: int = 0             # GPU memory per run, when the peak came from a CUDA device
    efficiency: float | None = None          # measured speedup per added core, when a curve exists
    memory_is_floor: bool = False            # the peak it came from was a lower bound (sampled RSS)
    basis: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def campaign_memory_bytes(self) -> int:
        """Memory to request for the whole batch of ``workers`` runs at once."""
        return self.memory_bytes * max(1, int(self.workers))

    @property
    def cores(self) -> int:
        return max(1, int(self.workers))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "workers": self.workers,
                "cores": self.cores, "memory_bytes": self.memory_bytes,
                "campaign_memory_bytes": self.campaign_memory_bytes,
                "device_memory_bytes": self.device_memory_bytes, "wall_time_s": self.wall_time_s,
                "efficiency": self.efficiency, "memory_is_floor": self.memory_is_floor,
                "measured_on": self.measured_on.to_dict(), "basis": dict(self.basis), "notes": self.notes}


# --------------------------------------------------------------------------- kind
def classify(record: CostRecord) -> str:
    """Which resource kind a measurement says the operation is (the rules are in the module docstring).

    The measurement decides; nothing is read from a label. ``flops`` is ``None`` for work the FLOP counter cannot
    see, which is exactly why "no FLOPs" is not used as evidence of "light" on its own.
    """
    seconds = max(float(record.time_s), 1e-9)
    peak = int(record.peak_mem_bytes or 0)
    flops = record.flops or 0
    if (record.mem_source or "").startswith("cuda"):
        return GPU_PREFERRED
    if flops:
        rate = flops / seconds
        if rate >= GPU_FLOPS_PER_S:
            return GPU_PREFERRED
        if rate >= HEAVY_FLOPS_PER_S:
            return CPU_HEAVY
    if not flops and peak / seconds < IO_BYTES_PER_S:
        return IO_BOUND
    if record.time_s >= HEAVY_TIME_S and peak >= HEAVY_MEMORY_BYTES:
        return CPU_HEAVY
    return CPU_LIGHT


# --------------------------------------------------------------------------- cores
def recommend_cores(by_workers: Mapping[int, CostRecord], *, deadline_s: float | None = None,
                    efficiency_min: float = 0.75) -> tuple[int, float | None, dict[str, Any]]:
    """``(workers, efficiency, basis)`` from the same operation measured at several worker counts.

    ``by_workers`` maps worker count to the record measured with that many workers (the record's ``size`` is
    ignored). The recommendation is the largest ``n`` whose measured efficiency ``t(1)/t(n)/n`` is at least
    ``efficiency_min``, cut back to the smallest ``n`` that meets ``deadline_s`` when one is given. Nothing is
    extrapolated: the answer is always one of the counts that was measured.
    """
    if not by_workers:
        raise ValueError("recommend_cores needs at least one measurement (worker count -> CostRecord)")
    counts = sorted(int(k) for k in by_workers)
    base = float(by_workers[counts[0]].time_s)
    if base <= 0:
        raise ValueError(f"the {counts[0]}-worker record measured no time")
    times = {c: float(by_workers[c].time_s) for c in counts}
    efficiency = {c: (base / max(times[c], 1e-12)) / c for c in counts}
    basis: dict[str, Any] = {"measured_workers": counts, "times_s": [times[c] for c in counts],
                             "speedup": [round(base / max(times[c], 1e-12), 3) for c in counts],
                             "efficiency": {c: round(efficiency[c], 3) for c in counts},
                             "efficiency_min": efficiency_min, "deadline_s": deadline_s}
    # 1. the throughput choice: the largest count that still uses most of the cores it is given
    usable = [c for c in counts if efficiency[c] >= efficiency_min]
    chosen, notes = (usable[-1] if usable else counts[0]), ""
    # 2. the deadline, when one is given: the fewest cores that is on time, never outside what was measured
    if deadline_s is not None:
        on_time = [c for c in counts if times[c] <= deadline_s]
        within = [c for c in usable if times[c] <= deadline_s]
        if within:
            chosen = within[0]                       # fewer cores is cheaper and still on time
        elif on_time:
            chosen = on_time[0]
            if efficiency[chosen] < efficiency_min:
                notes = (f"the {deadline_s:g} s deadline needs {chosen} workers, where the measured efficiency "
                         f"({efficiency[chosen]:.2f}) is below the {efficiency_min:g} floor")
        else:
            chosen = counts[-1]
            notes = (f"no measured worker count meets the {deadline_s:g} s deadline; asking for the largest "
                     f"measured one ({counts[-1]}) and flagging it rather than extrapolating")
    basis["chosen"] = chosen
    if notes:
        basis["note"] = notes
    return chosen, round(efficiency.get(chosen, 1.0), 3), basis


# --------------------------------------------------------------------------- the profile
def plan(record: CostRecord, *, workers: int = 1, headroom: float = 1.25,
         measured_on: MachineRecord | None = None, efficiency: float | None = None,
         notes: str = "") -> JobProfile:
    """A :class:`JobProfile` from one measurement: kind from :func:`classify`, memory from the peak times headroom."""
    if headroom < 1.0:
        raise ValueError("headroom must be >= 1.0: a request below the measured peak is a guaranteed failure")
    peak = int(record.peak_mem_bytes or 0)
    request = int(math.ceil(peak * headroom))
    device = request if (record.mem_source or "").startswith("cuda") else 0
    basis = {"time_s": float(record.time_s), "peak_mem_bytes": record.peak_mem_bytes,
             "mem_source": record.mem_source, "mem_exact": record.mem_exact, "flops": record.flops,
             "flops_source": record.flops_source, "headroom": headroom}
    extra = notes
    if record.peak_mem_bytes is None:
        extra = (extra + "; " if extra else "") + ("no peak was measurable for this operation: the memory request "
                                                   "is the headroom over an empty value, treat it as unknown")
    elif not record.mem_exact:
        extra = (extra + "; " if extra else "") + (
            f"peak is a lower bound ({record.mem_source}): the memory request is a floor, not a measurement")
    return JobProfile(name=record.name, kind=classify(record), workers=max(1, int(workers)), memory_bytes=request,
                      wall_time_s=float(record.time_s), measured_on=measured_on or MachineRecord.here(),
                      device_memory_bytes=device, efficiency=efficiency,
                      memory_is_floor=not record.mem_exact or record.peak_mem_bytes is None, basis=basis,
                      notes=extra)


class JobProfileTable:
    """Job profiles generated from measured cost, with the machine they were measured on.

    ``from_ledger`` is the intended entry point: a ``CostLedger`` already holds the records of a whole family of
    operations, and the table turns them into the one row per job a scheduler (or a person writing a SLURM header)
    needs. ``to_markdown`` writes the table, ``save`` the same data as JSON, both with ``measured_on``.
    """

    def __init__(self, measured_on: MachineRecord | None = None, *, headroom: float = 1.25):
        self.measured_on = measured_on or MachineRecord.here()
        self.headroom = headroom
        self.profiles: dict[str, JobProfile] = {}
        self.basis: dict[str, Any] = {}

    def add(self, record: CostRecord, *, workers: int = 1, cores: Mapping[int, CostRecord] | None = None,
            deadline_s: float | None = None, efficiency_min: float = 0.75, headroom: float | None = None,
            notes: str = "") -> JobProfile:
        """Add one operation; with a ``cores`` curve the worker count is derived instead of given."""
        efficiency = None
        extra = notes
        if cores:
            derived, efficiency, core_basis = recommend_cores(cores, deadline_s=deadline_s,
                                                              efficiency_min=efficiency_min)
            workers = derived
            self.basis[record.name] = {"cores": core_basis}
            if core_basis.get("note"):
                extra = (extra + "; " if extra else "") + core_basis["note"]
        prof = plan(record, workers=workers, headroom=self.headroom if headroom is None else headroom,
                    measured_on=self.measured_on, efficiency=efficiency, notes=extra)
        if record.name in self.basis:
            prof.basis.update(self.basis[record.name])
        self.profiles[record.name] = prof
        return prof

    def from_records(self, records: Sequence[CostRecord], *, workers: int = 1) -> "JobProfileTable":
        for rec in records:
            self.add(rec, workers=workers)
        return self

    @classmethod
    def from_ledger(cls, ledger, *, workers: int = 1, headroom: float = 1.25,
                    measured_on: MachineRecord | None = None) -> "JobProfileTable":
        """A table from every record of a ``CostLedger`` (the ledger's own environment is kept next to it)."""
        table = cls(measured_on or MachineRecord.here(), headroom=headroom)
        if getattr(ledger, "environment", None):
            table.basis["ledger_environment"] = dict(ledger.environment)
        for name, rec in getattr(ledger, "records", {}).items():
            table.add(rec, workers=workers)
        return table

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        return {"schema": "pinneapple.job_profiles/1", "measured_on": self.measured_on.to_dict(),
                "headroom": self.headroom, "basis": dict(self.basis),
                "profiles": {k: v.to_dict() for k, v in self.profiles.items()}}

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def to_markdown(self) -> str:
        m = self.measured_on
        out = [f"Measured on **{m.system} {m.machine}**, {m.cpu_count} CPUs, "
               f"{'unlimited' if not m.total_memory_bytes else f'{m.total_memory_bytes / (1024 ** 3):.0f} GiB'}, "
               f"Python {m.python}, torch {m.torch or 'not installed'}"
               f"{f', GPU {m.gpu}' if m.gpu else ''}.", "",
               "| job | kind | cores | memory per run | campaign memory | wall time | efficiency | notes |",
               "|---|---|---|---|---|---|---|---|"]
        for name, p in self.profiles.items():
            eff = "-" if p.efficiency is None else f"{p.efficiency:.2f}"
            out.append(f"| `{name}` | {p.kind} | {p.cores} | {p.memory_bytes / MIB:.1f} MiB | "
                       f"{p.campaign_memory_bytes / MIB:.1f} MiB | {p.wall_time_s:.3g} s | {eff} | "
                       f"{p.notes or ''} |")
        return "\n".join(out) + "\n"
