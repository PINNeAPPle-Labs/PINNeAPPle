#!/usr/bin/env python3
"""Measure the cost of every catalog method that has a probe, and gate the table (#52, #53; roadmap D1, D2).

    python scripts/measure_method_costs.py                      # measure everything with a probe, rewrite the table
    python scripts/measure_method_costs.py --only S4 S19        # remeasure a few, leave the rest as they are
    python scripts/measure_method_costs.py --check              # every method measured, or unmeasured with a reason
    python scripts/measure_method_costs.py --gate               # the regression gate against the versioned baseline

``--gate`` is the local stand-in for the CI workflow the repository cannot run (GitHub Actions is off for billing):
it compares a fresh measurement of the probed methods with ``pinneapple_catalog/method_costs_baseline.json`` and
fails when an operation got more expensive -- FLOPs strictly (5 %), time and memory loosely, since they are noisy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pinneapple_catalog import costs  # noqa: E402

BASELINE = costs.HERE / "method_costs_baseline.json"


def _check(path: Path) -> int:
    rows = costs.coverage(path)
    missing = [r for r in rows if r["status"] == "missing"]
    unmeasured = [r for r in rows if r["status"] == "unmeasured"]
    measured = [r for r in rows if r["status"] == "measured"]
    for r in missing:
        print(f"MISSING  {r['id']:<6} {r['name']}: {r['reason']}")
    for r in unmeasured:
        if not str(r.get("reason") or "").strip():
            print(f"NO REASON {r['id']:<6} {r['name']}: an unmeasured method must say why")
    print(f"{len(measured)} measured, {len(unmeasured)} unmeasured with a reason, {len(missing)} missing")
    return 1 if missing or any(not str(r.get("reason") or "").strip() for r in unmeasured) else 0


def _gate(current_path: Path, baseline_path: Path) -> int:
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}: measure once and copy the table there to start the gate")
        return 1
    current, baseline = costs.load_costs(current_path), costs.load_costs(baseline_path)
    rows = costs.compare_with_baseline(current, baseline)
    for r in rows:
        limit = "5 % (FLOPs are deterministic)" if r["quantity"] == "flops" else "50 % (wall time and memory are noisy)"
        print(f"REGRESSION {r['method_id']}: {r['quantity']} {r['baseline']:.4g} -> {r['current']:.4g} "
              f"(x{r['ratio']:.2f}, limit {limit})")
    print(f"cost gate: {len(rows)} regression(s)")
    return 1 if rows else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(costs.COSTS_PATH), help="where to write the cost table")
    ap.add_argument("--only", nargs="+", default=None, help="measure only these method ids")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per size")
    ap.add_argument("--check", action="store_true", help="gate: every catalog method measured or explained")
    ap.add_argument("--gate", action="store_true", help="compare against the versioned baseline and fail on a regression")
    ap.add_argument("--baseline", default=str(BASELINE), help="the baseline the gate compares against")
    a = ap.parse_args()

    out = Path(a.out)
    if not a.check and not a.gate:
        costs.measure_all(a.only, repeats=a.repeats, path=out)
        print(f"wrote {out} ({sum(1 for r in costs.coverage(out) if r['status'] == 'measured')} measured)")
    if a.check:
        sys.exit(_check(out))
    if a.gate:
        sys.exit(_gate(out, Path(a.baseline)))
