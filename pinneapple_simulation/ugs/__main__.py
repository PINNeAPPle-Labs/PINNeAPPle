"""``python -m pinneapple_simulation.ugs``: run the verification report and print it.

    python -m pinneapple_simulation.ugs            # ~2 s: every case at its published resolution
    python -m pinneapple_simulation.ugs --quick    # ~1 s: the same cases at a size a test can afford
    python -m pinneapple_simulation.ugs --json     # the report as data, for a job that reads it

Exits 1 when a case fails, so a CI step can gate on it.
"""
from __future__ import annotations

import json
import sys
import time

from . import verification


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in args
    as_json = "--json" in args
    start = time.perf_counter()
    report = verification.run(quick=quick)
    elapsed = time.perf_counter() - start
    if as_json:
        print(json.dumps({**report, "seconds": elapsed}, indent=1, default=float))
    else:
        print(verification.summary(report))
        print(f"\n{elapsed:.1f} s ({'quick' if quick else 'published'} resolutions); "
              f"details with --json")
    ok = all(bool(case.get("ok")) for case in report.values() if isinstance(case, dict))
    if not ok:
        failed = [name for name, case in report.items() if isinstance(case, dict) and not case.get("ok")]
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
