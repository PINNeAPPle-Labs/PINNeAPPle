"""02 — A physics-only PINN, scored against a solution it never saw, and exported to ONNX.

What this demonstrates
----------------------
- ``data.source: physics_only``: no reference data, the equations and the conditions only
- the ``model`` gate scoring the PINN against the Cole-Hopf closed form (``relative_l2``)
- the ``physics`` gate: the boundary/initial misfit of the trained model on points it did not train on
- the ``export`` gate: the ONNX file is compared against the in-memory model before it is called shipped

Run from the repo root (about a minute on a laptop, CPU):
    python examples/surrogate_pipeline/02_physics_only_pinn.py

The point of the example is the last two gates. A PINN can drive its training loss to 1e-5 and still be wrong; what
licenses it is the error against an independent solution and the conditions it satisfies, both of which the gates
measure after training.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pinneapple as pp  # noqa: E402

PRESET = "burgers_physics_only"


def main() -> int:
    pipeline = pp.SurrogatePipeline.from_preset(
        PRESET, execution={"out_dir": str(REPO_ROOT / "examples" / "surrogate_pipeline" / "_out")})
    print(pipeline.describe())
    print()

    result = pipeline.run()
    print(result.to_markdown())

    print("\n=== the gates that license this model ===")
    for name in ("model", "physics", "export"):
        gate = result.gate(name)
        if gate is not None:
            print(gate.to_markdown())

    print("\n=== a query inside the domain ===")
    answer = pipeline.predict(x=0.25, t=0.5)
    print(f"  u(0.25, 0.5) = {answer['outputs']}")

    print("\n=== training ===")
    training = result.metrics.get("training", {})
    print(f"  epochs {pipeline.config.training.epochs}, {training.get('total_seconds', 0):.1f} s, "
          f"final loss {training.get('final_loss')}")
    return 0 if result.status in ("completed", "completed-with-findings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
