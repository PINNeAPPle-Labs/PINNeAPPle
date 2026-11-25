#!/usr/bin/env python3
"""Measure the cost of every catalog method that has a probe, and gate the table (#52, #53; roadmap D1, D2).

    python scripts/measure_method_costs.py                      # measure everything with a probe, rewrite the table
    python scripts/measure_method_costs.py --only S4 S19        # remeasure a few, leave the rest as they are
    python scripts/measure_method_costs.py --check              # every catalog entry has a row, with its machine,
                                                               # and every unmeasured one its blocker and reason
    python scripts/measure_method_costs.py --gate               # the regression gate against the versioned baseline

``--gate`` is the local stand-in for the CI workflow the repository cannot run (GitHub Actions is off for billing):
it compares a fresh measurement of the probed methods with ``pinneapple_catalog/method_costs_baseline.json`` and
fails when an operation got more expensive -- FLOPs strictly (5 %), time and memory loosely, since they are noisy.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pinneapple_catalog import costs  # noqa: E402

BASELINE = costs.HERE / "method_costs_baseline.json"
DOCS_PATH = ROOT / "docs" / "dev" / "METHOD_COSTS.md"


MACHINE_KEYS = ("platform", "cpu", "cpu_count", "python", "torch")


def _check(path: Path) -> int:
    """The gate: every catalog entry has a row, every measured number travels with its machine, every unmeasured one
    names a blocker from the closed set and a reason.

    The machine is checked at file level because that is where it is recorded -- one file, one run, one machine. A
    table whose header lost the machine has no comparable number in it, whatever the rows say.
    """
    rows = costs.coverage(path)
    doc = costs.load_costs(path)
    machine = doc.get("machine") or {}
    absent = [k for k in MACHINE_KEYS if machine.get(k) in (None, "")]
    if absent:
        print(f"NO MACHINE: the cost table does not record {absent}, so no wall time in it can be compared: "
              f"rerun python scripts/measure_method_costs.py")
    missing = [r for r in rows if r["status"] == "missing"]
    measured = [r for r in rows if r["status"] == "measured"]
    unmeasured = [r for r in rows if r["status"] == "unmeasured"]
    for r in missing:
        print(f"GAP      {r['id']:<6} {r['name']}: {r['problem'] or r['reason']}")
    by_category = collections.Counter(r["category"] for r in rows)
    by_blocker = collections.Counter(r["blocker"] for r in unmeasured)
    print(f"{len(measured)} measured, {len(unmeasured)} unmeasured with a reason, {len(missing)} with no usable row, "
          f"over {len(rows)} catalog entries ({dict(sorted(by_category.items()))})")
    print("unmeasured by blocker: " + ", ".join(f"{k} {v}" for k, v in sorted(by_blocker.items())))
    print(f"machine: {machine.get('cpu') or machine.get('processor')} x{machine.get('cpu_count')}, "
          f"{machine.get('platform')}, commit {doc.get('commit') or '?'}")
    return 1 if missing or absent else 0


def _gate(only, baseline_path: Path, repeats: int) -> int:
    """Measure now, compare with the versioned baseline, fail on a regression: the local stand-in for the CI job
    the repository cannot run (GitHub Actions is off for billing). Two minutes on a laptop, or less with ``--only``.

    The fresh table goes to a file of its own every run, so a leftover from an earlier invocation can never be
    compared as if it had just been measured, and ``--only`` compares exactly the methods it measured.
    """
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}: measure once and copy that table there to start the gate")
        return 1
    baseline = costs.load_costs(baseline_path)
    out = Path(os.environ.get("COST_GATE_OUT") or Path(tempfile.mkdtemp(prefix="cost-gate-")) / "method_costs.json")
    fresh = costs.measure_all(only, repeats=repeats, path=out)
    if only:
        fresh = {**fresh, "methods": {mid: r for mid, r in fresh["methods"].items() if mid in set(only)}}
    rows = costs.compare_with_baseline(fresh, baseline)
    for r in rows:
        limit = ("5 % (FLOPs are deterministic)" if r["quantity"] == "flops"
                 else "50 % (wall time and memory move between runs)")
        print(f"REGRESSION {r['method_id']}: {r['quantity']} {r['baseline']:.4g} -> {r['current']:.4g} "
              f"(x{r['ratio']:.2f}, limit {limit})")
    measured = sum(1 for r in fresh["methods"].values() if r.get("status") == "measured")
    print(f"cost gate: {len(rows)} regression(s) over {measured} measured method(s); fresh table at {out}")
    return 1 if rows else 0


def _ms(value: float) -> str:
    """A time with a unit that keeps it readable: microseconds, milliseconds or seconds."""
    if value < 1e-3:
        return f"{value * 1e6:.1f} µs"
    if value < 1.0:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.2f} s"


def _bytes(value: int | None) -> str:
    if not value:
        return "—"
    for unit, scale in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} B"


def render_doc(doc: dict, rows: list[dict]) -> str:
    """The Markdown one-liner-per-method view of the table: what the project reads, over the JSON it comes from.

    Generated, never hand-written -- ``--docs-check`` compares the file on disk with this function's output, so a
    stale row cannot outlive the measurement that produced it.
    """
    machine = doc.get("machine") or {}
    measured = [r for r in rows if r["status"] == "measured"]
    unmeasured = [r for r in rows if r["status"] != "measured"]
    by_blocker = collections.Counter(r["blocker"] for r in unmeasured)
    records = doc.get("methods", {})
    out = [
        "# Measured cost per catalog method",
        "",
        "> Generated by `python scripts/measure_method_costs.py --docs` from",
        "> `pinneapple_catalog/method_costs.json`. Do not edit by hand: run the script again.",
        "> `--docs-check` fails when this file and the table disagree.",
        "",
        "Every entry of `pinneapple_catalog/methods.py` (`METHODS`) has a row here. A row is one of two things:",
        "",
        "- **measured** -- a probe ran this method on this machine over several problem sizes, and the row carries the",
        "  median time with its interquartile range, the peak memory with its sampling source, the FLOPs with their",
        "  source (`null` when nothing counted them, which is an answer and not a gap), the scaling exponent with a",
        "  bootstrap 95 % confidence interval, the sizes it was fitted over, and the machine it all came from;",
        "- **unmeasured** -- with a blocker from a closed set (`needs_cuda`, `needs_external_solver`,",
        "  `needs_optional_package`, `needs_input`, `not_a_single_operation`, `definition`) and the specific reason.",
        "",
        "## How to reproduce",
        "",
        "```bash",
        "python scripts/measure_method_costs.py                 # measure every method that has a probe, rewrite the table",
        "python scripts/measure_method_costs.py --only S4 S11   # remeasure a few, leave the rest untouched",
        "python scripts/measure_method_costs.py --check         # the gate: a row per catalog entry, the machine recorded",
        "python scripts/measure_method_costs.py --docs          # rewrite this file from the table",
        "python scripts/measure_method_costs.py --gate          # regression gate against method_costs_baseline.json",
        "```",
        "",
        "The instrument is `pinneapple_analysis/cost/profiler.py` (`measure`, `scaling_study` with its bootstrap",
        "interval). It measures wall time, peak memory and PyTorch FLOPs. FLOPs stay `null` where nothing counts them:",
        "the counter knows matmul, conv, attention and FFTs, and NumPy work is not counted at all. Peak memory comes",
        "from `pinneapple_analysis/cost/memory.py`, and `mem_source` says which method produced it (`torch+trace` is",
        "the PyTorch profiler summed with `tracemalloc`, exact for what each one sees; `rss` and `tracemalloc` alone",
        "are lower bounds). The limits of both are established in those two modules and are not re-derived here.",
        "",
        "Wall time is wall clock, so it moves with whatever else the machine is doing. That is why every record",
        "carries the interquartile range beside the median and the exponent carries a bootstrap interval: a number",
        "here is a measurement with its uncertainty, and two runs on two machines are two measurements, not a",
        "regression (`--gate` refuses to compare across machines for exactly that reason).",
        "",
        "## The machine these numbers came from",
        "",
        "A wall time without its machine cannot be compared with anything, so it travels with the file:",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for key in ("cpu", "cpu_count", "platform", "system", "machine", "python", "torch", "cuda", "memory_bytes"):
        out.append(f"| `{key}` | {machine.get(key)} |")
    out += [
        "",
        f"Measured at `{doc.get('measured_at', '?')}` on commit `{doc.get('commit') or '?'}`.",
        "",
        f"## Measured ({len(measured)} of {len(rows)})",
        "",
        "Each `id` is the key of that method's record in `pinneapple_catalog/method_costs.json`, where the per-size",
        "times with their interquartile range live.",
        "",
        "| id | method | what ran (the probe) | `n` means | sizes | exponent | 95 % CI | R² | declared | verdict "
        "| median time | peak memory | FLOPs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in measured:
        rec = records.get(r["id"], {})
        ci = rec.get("exponent_ci") or [float("nan"), float("nan")]
        exponent = rec.get("exponent")
        flops = rec.get("flops")
        flops_cell = (f"{flops:,} ({rec.get('flops_source') or '?'})" if flops
                      else "— (not countable)")
        sizes = ", ".join(str(int(s)) for s in rec.get("sizes", []))
        out.append(
            f"| **{r['id']}** | {_cell(r['name'])} | {_cell(rec.get('probe') or '')} "
            f"| {_cell(rec.get('size_axis') or '')} | {sizes} "
            f"| {exponent:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {rec.get('r2', float('nan')):.3f} "
            f"| {rec.get('declared') or '—'} | {rec.get('verdict') or '—'} "
            f"| {_ms(rec.get('median_time_s') or 0.0)} | {_bytes(rec.get('peak_mem_bytes'))} "
            f"| {flops_cell} |")
    out += [
        "",
        "The median time is the median over the sizes' medians, so it is the order of magnitude of one call, not one",
        "size; the per-size numbers (with the IQR) are in the JSON under `per_size`. The exponent is the log-log slope",
        "of time against `n` over the sizes listed, and `declared` is the complexity the probe says the method has,",
        "with `verdict` saying whether the fit agrees with it inside the interval.",
        "",
        f"## Unmeasured ({len(unmeasured)} of {len(rows)})",
        "",
        "A closed set of blockers, so the gap can be counted and cannot be waved away:",
        "",
        "| blocker | methods | what it means |",
        "|---|---|---|",
    ]
    for blocker, count in sorted(by_blocker.items()):
        out.append(f"| `{blocker}` | {count} | {_cell(costs.BLOCKERS.get(blocker, '?'))} |")
    out += [
        "",
        "| id | method | blocker | why there is no number |",
        "|---|---|---|---|",
    ]
    for r in unmeasured:
        out.append(f"| **{r['id']}** | {_cell(r['name'])} | `{r['blocker']}` | {_cell(r['reason'])} |")
    out.append("")
    return "\n".join(out)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _docs(path: Path, out_path: Path, check: bool) -> int:
    """Write this file from the table, or (``--docs-check``) fail when the file on disk is out of date."""
    doc = costs.load_costs(path)
    text = render_doc(doc, costs.coverage(path))
    if check:
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if current != text:
            print(f"{out_path} is out of date with {path}: run python scripts/measure_method_costs.py --docs")
            return 1
        print(f"{out_path} matches the cost table")
        return 0
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(costs.COSTS_PATH), help="where to write the cost table")
    ap.add_argument("--only", nargs="+", default=None, help="measure only these method ids")
    ap.add_argument("--repeats", type=int, default=None, help="timed runs per size (default: the probe's own)")
    ap.add_argument("--check", action="store_true", help="gate: one row per catalog entry, machine recorded, "
                                                        "blocker and reason on every unmeasured one")
    ap.add_argument("--gate", action="store_true", help="compare against the versioned baseline and fail on a regression")
    ap.add_argument("--baseline", default=str(BASELINE), help="the baseline the gate compares against")
    ap.add_argument("--docs", action="store_true", help="rewrite the Markdown view of the table")
    ap.add_argument("--docs-check", action="store_true", help="fail when the Markdown view is out of date")
    ap.add_argument("--docs-out", default=str(DOCS_PATH), help="the Markdown view the two --docs flags write and check")
    a = ap.parse_args()

    out = Path(a.out)
    if a.check:
        sys.exit(_check(Path(a.out)))
    if a.gate:
        sys.exit(_gate(a.only, Path(a.baseline), a.repeats))
    if a.docs or a.docs_check:
        sys.exit(_docs(Path(a.out), Path(a.docs_out), check=a.docs_check))
    costs.measure_all(a.only, repeats=a.repeats, path=out)
    print(f"wrote {out} ({sum(1 for r in costs.coverage(out) if r['status'] == 'measured')} measured)")
