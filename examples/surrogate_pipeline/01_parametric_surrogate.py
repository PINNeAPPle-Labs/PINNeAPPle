"""01 — A parametric surrogate in one call: viscosity in, the Burgers velocity field at t = 1 out.

What this demonstrates
----------------------
- ``pp.SurrogatePipeline.from_preset(...)``: a whole configuration, named
- the seven stages and what each one wrote (``result.stages``)
- the VeriPhysics gates on a supervised run, including the ones that do not apply and why
- ``pipeline.predict_verified(nu=...)``: a prediction with the decision that licenses using it

Run from the repo root:
    python examples/surrogate_pipeline/01_parametric_surrogate.py

It takes about ten seconds on a laptop: 18 reference solves of the preset's own finite-difference solver, then a
Gaussian process and an MLP compared on cases neither trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pinneapple as pp  # noqa: E402

PRESET = "burgers_in_viscosity"


def main() -> int:
    pipeline = pp.SurrogatePipeline.from_preset(
        PRESET, execution={"out_dir": str(REPO_ROOT / "examples" / "surrogate_pipeline" / "_out")})
    print(pipeline.describe())
    print()

    result = pipeline.run()
    print(result.to_markdown())

    print("\n=== the model answers a query ===")
    answer = pipeline.predict_verified(nu=0.03)
    print(f"  nu = 0.03 -> {answer['outputs']}")
    print(f"  decision: {answer['decision']}")
    for reason in answer["reasons"]:
        print(f"    {reason}")
    print(f"  validated for: {answer['validated_domain'].get('bounds')}")

    print("\n=== what the run left on disk ===")
    for name, path in result.artifacts.items():
        print(f"  {name:<16} {path}")
    return 0 if result.status in ("completed", "completed-with-findings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
