"""Run the reference example and print its quality report::

    python -m pinneapple_llm.qa.examples.pipe_wear_twin_main [output_dir] [--anomaly]

Exit code 0 when the run produced a report, 1 when a required step failed.
The anomaly condition (``--anomaly``) is expected to fail its invariants;
that failure is the demonstration, so it exits 0 while printing FAIL steps.
"""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    anomaly = "--anomaly" in argv
    args = [a for a in argv if not a.startswith("--")]
    output_dir = args[0] if args else None

    from .pipe_wear_twin import run_example

    report, bundle = run_example(output_dir=output_dir, anomaly=anomaly)
    print(report.summary())
    print()
    print("artifacts:")
    for artifact in report.artifacts:
        print(f"  {artifact.name}  {artifact.sha256[:12]}  {artifact.size_bytes} bytes")
    print()
    print("reproduce:", report.reproduce)
    if output_dir:
        print("written to:", output_dir)
    return 0 if report.overall_status.value in ("PASS", "FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
