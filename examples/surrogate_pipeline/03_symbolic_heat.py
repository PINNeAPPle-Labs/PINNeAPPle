"""03 — A PDE written as text, trained and validated by the PINN validation suite.

What this demonstrates
----------------------
- ``problem.source: symbolic``: the heat equation written as ``d(u, t) - alpha * lap(u)``, parsed by
  ``pinneapple_physics.symbolic_pde`` (the accepted syntax is **not** ``Derivative(u, t)``: the parser knows
  ``d``, ``diff``, ``lap``, ``laplacian`` and a set of elementary functions)
- the ``physics`` gate taking its evidence from ``pinneapple_analysis.validation.pinn.validate_pinn``: residual at
  fresh points, boundary and initial misfit, and the "near-trivial solution" check that catches a network which
  learned to answer zero
- why a model can look trained and still be refused

Run from the repo root (about a minute on a laptop, CPU):
    python examples/surrogate_pipeline/03_symbolic_heat.py

The same run with a shorter budget is the counter-example: ``training.epochs: 300`` and no L-BFGS polish leaves the
boundary and initial conditions traded off against the residual, and the pipeline says so instead of shipping it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pinneapple as pp  # noqa: E402

PRESET = "heat_2d_symbolic"


def main() -> int:
    pipeline = pp.SurrogatePipeline.from_preset(
        PRESET, execution={"out_dir": str(REPO_ROOT / "examples" / "surrogate_pipeline" / "_out")})
    print(pipeline.describe())
    print()

    result = pipeline.run()
    print(result.to_markdown())

    print("\n=== undertrained on purpose: what the same gates say ===")
    short = pp.SurrogatePipeline.from_preset(
        PRESET, name="heat_2d_symbolic_short",
        execution={"out_dir": str(REPO_ROOT / "examples" / "surrogate_pipeline" / "_out")},
        training={"epochs": 300, "lbfgs_steps": 0},
        verification={"mode": "balanced", "fail_policy": "report"})
    short_result = short.run()
    gate = short_result.gate("physics")
    print(f"  physics gate: {gate.status}")
    for line in gate.evidence:
        print(f"    {line}")
    print(f"  run status: {short_result.status}")

    return 0 if result.status in ("completed", "completed-with-findings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
