"""The cost report: time, FLOPs and joules per operation, with the method behind the joules printed under them.

Time and FLOPs alone say how long an operation takes and how much work it does; the report adds the energy line and,
under the table, the one thing that makes that line usable -- **how the joules were obtained** (roadmap item K20,
issue #119): the method, the source of its constant and the hardware it describes. A joule with no declared method
is not reported at all: the row says why there is no number and which input would produce one.

Usage::

    from pinneapple_analysis.cost import CostLedger, cost_report
    ledger = CostLedger()
    ledger.measure("matmul_1024", lambda a=torch.rand(1024, 1024): a @ a)
    print(cost_report(ledger, hardware="cpu_45nm"))

The input is a ``CostRecord``, a list of them, a ``CostLedger``, a ``ScalingStudy`` or a ``BudgetRun`` (a training
run: its wall time and counted FLOPs become one row). ``estimate_energy`` does the arithmetic; this module only lays
it out, so a number in the table and the sentence under it can never disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .energy import EnergyEstimate, estimate_energy, fmt_joules

__all__ = ["cost_report"]


@dataclass
class _Row:
    """One operation in the report, whatever kind of measurement it came from."""
    name: str
    time_s: float | None
    flops: float | None
    flops_source: str = ""


def _rows(result: Any) -> tuple[list[_Row], str]:
    """``(rows, title)`` from a record, a ledger, a study, a training run or a list of any of those."""
    if isinstance(result, (list, tuple)):
        rows: list[_Row] = []
        for item in result:
            item_rows, _ = _rows(item)
            rows.extend(item_rows)
        return rows, ""
    records = getattr(result, "records", None)
    if records is not None:                                     # CostLedger (dict) and ScalingStudy (list)
        items = records.values() if isinstance(records, dict) else records
        name = getattr(result, "name", "") or ""
        return [r for r in (_row(item) for item in items) if r is not None], name
    row = _row(result)
    return ([row] if row is not None else []), ""


def _row(item: Any) -> _Row | None:
    if hasattr(item, "time_s") and hasattr(item, "flops"):      # CostRecord
        return _Row(str(getattr(item, "name", "operation")), float(item.time_s), item.flops,
                    str(getattr(item, "flops_source", "") or ""))
    if hasattr(item, "wall_s") and hasattr(item, "flops"):      # equal_budget.BudgetRun: one training run
        return _Row(str(getattr(item, "name", "training")), float(item.wall_s), item.flops, "count_flops")
    return None


def _fmt_flops(flops: float | None) -> str:
    if flops is None:
        return "-"
    for scale, unit in ((1e12, "TFLOP"), (1e9, "GFLOP"), (1e6, "MFLOP"), (1e3, "kFLOP")):
        if flops >= scale:
            return f"{flops / scale:.3g} {unit}"
    return f"{flops:.0f} FLOP"


def _fmt_time(time_s: float | None) -> str:
    if time_s is None:
        return "-"
    if time_s < 1e-3:
        return f"{time_s * 1e6:.3g} µs"
    if time_s < 1.0:
        return f"{time_s * 1e3:.3g} ms"
    return f"{time_s:.3g} s" if time_s < 1e4 else f"{time_s / 3600:.3g} h"


def cost_report(result: Any, *, hardware: str | None = None, method: str = "auto",
                energy_per_flop_pj: float | None = None, energy_per_flop_source: str = "",
                power_w: float | None = None, power_source: str = "", carbon_g_per_kwh: float | None = None,
                carbon_source: str = "", title: str | None = None) -> str:
    """A Markdown table of time, FLOPs and joules per operation, and the declared method under it.

    Every row is priced with the same declared method: ``hardware=`` names the table entry (see
    ``ENERGY_PER_FLOP``), ``energy_per_flop_pj=`` with its ``energy_per_flop_source`` declares a model the library
    does not ship, and ``power_w=`` with ``power_source`` prices the operations that have no FLOP count. Rows that
    no method can price keep their time and FLOPs and say why there is no joule count, instead of showing a guess.
    """
    rows, study_name = _rows(result)
    head = title or study_name or (rows[0].name if len(rows) == 1 else f"{len(rows)} operations")
    lines = [f"## Cost report: {head}", ""]
    if not rows:
        return "\n".join(lines + ["No measurement was passed: nothing to report."])

    priced: list[tuple[_Row, EnergyEstimate]] = []
    for row in rows:
        estimate = estimate_energy(flops=row.flops, time_s=row.time_s, method=method, hardware=hardware,
                                   energy_per_flop_pj=energy_per_flop_pj,
                                   energy_per_flop_source=energy_per_flop_source, power_w=power_w,
                                   power_source=power_source, carbon_g_per_kwh=carbon_g_per_kwh,
                                   carbon_source=carbon_source)
        priced.append((row, estimate))

    lines += ["| operation | time | FLOPs | FLOPs source | energy |", "|---|---|---|---|---|"]
    for row, estimate in priced:
        energy = fmt_joules(estimate.joules) if estimate.available else "not estimable"
        lines.append(f"| `{row.name}` | {_fmt_time(row.time_s)} | {_fmt_flops(row.flops)} | "
                     f"{row.flops_source or '-'} | {energy} |")
    lines.append("")

    # the provenance: one block per distinct method/hardware, so the method is never implied
    groups: dict[tuple, list[str]] = {}
    representatives: dict[tuple, EnergyEstimate] = {}
    unpriced: dict[str, list[str]] = {}
    for row, estimate in priced:
        if not estimate.available:
            unpriced.setdefault(f"{estimate.reason}||{estimate.fix}", []).append(row.name)
            continue
        key = (estimate.method, estimate.hardware, estimate.hardware_declared, estimate.source)
        groups.setdefault(key, []).append(row.name)
        representatives.setdefault(key, estimate)
    for key, names in groups.items():
        lines.append(f"Joules for {len(names)} of {len(priced)} operations ({', '.join(f'`{n}`' for n in names)}):")
        lines.append(_indent(representatives[key].report()))
        lines.append("")
    for reason, names in unpriced.items():
        why, fix = reason.split("||", 1)
        lines.append(f"No joules for {len(names)} of {len(priced)} operations "
                     f"({', '.join(f'`{n}`' for n in names)}): {why}")
        if fix:
            lines.append(f"  fix: {fix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())
