"""04 — The same pipeline from a YAML file, stage by stage, resumed, and from the command line.

What this demonstrates
----------------------
- a pipeline as a file (``configs/burgers_in_viscosity.yaml``), checked in and re-runnable
- ``run_until("data")``: generate the reference campaign now, train later
- ``resume()``: the finished stages are not repeated (the reference solves are the expensive part)
- ``status()`` and ``get_artifacts()`` on a folder from an earlier session
- the CLI: ``python -m pinneapple.pipeline validate|plan|run|report``

Run from the repo root:
    python examples/surrogate_pipeline/04_yaml_and_cli.py

Pass a path to a YAML with ``--config`` to drive your own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pinneapple as pp  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "burgers_in_viscosity.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="pipeline from YAML")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = Path(args.config)

    pipeline = pp.SurrogatePipeline.from_yaml(config)
    print(pipeline.describe())
    print()
    print(f"dry run: {'ready' if pipeline.dry_run()['ok'] else 'not ready'}")
    print(f"estimated cost: {pipeline.estimate_cost()}")

    print("\n=== stage 1: everything up to the dataset ===")
    partial = pipeline.run_until("data")
    for record in partial.stages:
        print(f"  {record.stage:<12} {record.status:<8} {record.seconds:6.2f} s")
    dataset = pipeline.get_artifacts().get("dataset")
    print(f"  dataset: {dataset}")

    print("\n=== stage 2: resume from the dataset and finish ===")
    result = pipeline.resume()
    for record in result.stages:
        print(f"  {record.stage:<12} {record.status:<8} {record.seconds:6.2f} s")
    print(f"\n{result.summary()}")

    print("\n=== the same file from the command line ===")
    cli = [sys.executable, "-m", "pinneapple.pipeline", "validate", str(config)]
    out = subprocess.run(cli, cwd=REPO_ROOT, capture_output=True, text=True)
    print(f"  {' '.join(cli[1:])}  ->  exit {out.returncode}")
    print("  " + (out.stdout or out.stderr).strip().splitlines()[0])

    print("\n=== the report of the last run ===")
    report = Path(result.artifacts["report"])
    print(f"  {report} ({report.stat().st_size} bytes)")
    return 0 if result.status in ("completed", "completed-with-findings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
