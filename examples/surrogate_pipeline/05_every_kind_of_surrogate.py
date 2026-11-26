"""05 — Every kind of surrogate model the pipeline makes, from one import, and each one served.

What this demonstrates
----------------------
==========================  =====================================================================
a parametric surrogate      reference solves -> dataset -> study over GP/MLP -> held-out error
a physics-only PINN         the equations, no data, scored against the Cole-Hopf solution
a PDE written as text       trained and validated by the PINN validation suite, exact solution included
adaptive learning           cases added where the response varies most, until the target is met
``pinneapple check``        the same model, loaded by an independent linter, as gate evidence
deployment                  the run as a model service whose trust function is the run's own gates
==========================  =====================================================================

Run from the repo root::

    python examples/surrogate_pipeline/05_every_kind_of_surrogate.py            # the whole tour (~2 min)
    python examples/surrogate_pipeline/05_every_kind_of_surrogate.py --fast     # smallest budgets (~40 s)
    python examples/surrogate_pipeline/05_every_kind_of_surrogate.py --only serve

Every number printed comes from the run it just did; nothing here is cached.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pinneapple as pp  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out" / "tour"
BOLD, RESET = ("\033[1m", "\033[0m") if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else ("", "")


def rule(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}\n" + "─" * len(title))


def gates(result) -> str:
    """``PASS 7 · WARN 1 · N/A 2`` — the shape of a run in one line."""
    counts: dict[str, int] = {}
    for gate in result.verification:
        counts[gate.status] = counts.get(gate.status, 0) + 1
    return " · ".join(f"{status.replace('_', ' ')} {n}" for status, n in
                      sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------------------------- the four models
def parametric(fast: bool):
    """A surrogate of one family of problems: what changes is viscosity, what we want is the whole field."""
    rule("1. a parametric surrogate — 18 reference solves, then the study")
    pipeline = pp.SurrogatePipeline.from_preset(
        "burgers_in_viscosity", execution={"out_dir": str(OUT), "resume": False},
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12 if fast else 18},
                  "validation_sampling": {"test_fraction": 0.2, "val_fraction": 0.15, "shift": "auto"}},
        training={"budget_seconds": 1.0 if fast else 3.0},
        verification={"fail_policy": "report"})
    result = pipeline.run()
    print(f"  {result.summary()}")
    print(f"  gates: {gates(result)}")
    for row in result.metrics["ranking"][:3]:
        print(f"  {row['method']:<18} held-out nrmse {row['error']:.3g}")
    print(f"  report: {result.artifacts['report']}")
    return pipeline


def physics_only(fast: bool):
    """No data at all: the model is trained on the equation and scored against a closed form."""
    rule("2. a physics-only PINN — the equations, no reference campaign")
    pipeline = pp.SurrogatePipeline.from_preset(
        "burgers_physics_only", execution={"out_dir": str(OUT), "resume": False},
        training={"epochs": 400 if fast else 1200, "n_collocation": 1024 if fast else 2048},
        verification={"fail_policy": "report"}, export={"enabled": False, "formats": []})
    result = pipeline.run()
    print(f"  {result.summary()}")
    print(f"  relative L2 against Cole-Hopf: {result.metrics['relative_l2']:.4f}")
    print(f"  boundary/initial misfit: {result.metrics['physics']['bc_misfit']:.4g}")
    return pipeline


def symbolic(fast: bool):
    """The PDE as text, a declared exact solution, and the validity suite grading the result."""
    rule("3. a PDE written as text — trained and graded by the validation suite")
    pipeline = pp.SurrogatePipeline.from_preset(
        "heat_2d_symbolic", execution={"out_dir": str(OUT), "resume": False},
        training={"epochs": 600 if fast else 2000, "lbfgs_steps": 100 if fast else 400},
        verification={"fail_policy": "report", "check_physics": "none"})
    result = pipeline.run()
    print(f"  {result.summary()}")
    model_gate, physics_gate = result.gate("model"), result.gate("physics")
    print(f"  error against the exact solution: {model_gate.metrics.get('relative_l2')}")
    print(f"  physics gate: {physics_gate.status} — {physics_gate.evidence[0] if physics_gate.evidence else ''}")
    return pipeline


def adaptive(fast: bool):
    """Where the response varies most is where the next simulations go."""
    rule("4. adaptive learning — cases added until the target is met")
    pipeline = pp.SurrogatePipeline.from_preset(
        "burgers_in_viscosity", name="adaptive", execution={"out_dir": str(OUT), "resume": False},
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12},
                  "validation_sampling": {"test_fraction": 0.2, "val_fraction": 0.15, "shift": "auto"}},
        architecture={"family": "gp", "candidates": []}, training={"budget_seconds": 1.0},
        adaptive_learning={"enabled": True, "max_rounds": 2 if fast else 4, "samples_per_round": 3,
                           "target_error": 0.0008},
        verification={"fail_policy": "report"})
    result = pipeline.run()
    adaptive_metrics = result.metrics["adaptive_learning"]
    print(f"  {result.summary()}")
    print(f"  target nrmse {adaptive_metrics['target']:g}, reached: {adaptive_metrics['reached_target']}")
    for row in adaptive_metrics["rounds"]:
        if "skipped" in row:
            print(f"  round {row['round']}: skipped — {row['skipped']}")
        else:
            print(f"  round {row['round']}: +{row['added']} case(s), nrmse "
                  f"{row.get('error_before')} -> {row.get('error_after')}, total {row['total_cases']}")
    return pipeline


# ------------------------------------------------------------------------------------------ the second opinion
def second_opinion(pipeline):
    """``pinneapple check`` reads the same model through a plain array function and applies its own rules."""
    rule("5. pinneapple check — an independent linter over the same artefacts")
    check = pipeline.state.metrics["pinneapple_check"]
    print(f"  {check['n_checks']} outcome(s) from {len(check['by_rule'])} rule(s), {check['n_failed']} not holding")
    for rule_id, row in sorted(check["by_rule"].items()):
        print(f"  {rule_id}  {row['checks']:>2} check(s)  {row['failed']} failed")
    print(f"  files: {Path(check['config']).parent}")
    return check


def served(pipeline):
    """The same model as a service: predict, and the trust verdict of the run it came from."""
    rule("6. deployed — the run answers with its own verification attached")
    deployment = pipeline.deploy()
    inside = pipeline.state.validated_domain["bounds"]
    query = {name: 0.5 * (bounds[0] + bounds[1]) for name, bounds in inside.items()}
    answer = deployment.predict(**query)
    print(f"  predict({query}) -> decision {answer['trust']['decision']}")
    print(f"  {answer['outputs']}")
    outside = {name: bounds[1] * 2.0 for name, bounds in inside.items()}
    try:
        deployment.predict(**outside)
    except ValueError as exc:
        print(f"  predict({outside}) -> refused: {str(exc)[:90]}...")
    return deployment


def main() -> int:
    parser = argparse.ArgumentParser(description="the tour of surrogate models")
    parser.add_argument("--fast", action="store_true", help="smallest budgets")
    parser.add_argument("--only", default="all",
                        choices=["all", "parametric", "physics", "symbolic", "adaptive", "serve"])
    args = parser.parse_args()
    t0 = time.perf_counter()
    if OUT.exists() and "--fast" in sys.argv:         # a tour is meant to run now, not to resume what is on disk
        import shutil
        shutil.rmtree(OUT)

    parametric_pipeline = None
    if args.only in ("all", "parametric"):
        parametric_pipeline = parametric(args.fast)
    if args.only in ("all", "physics"):
        physics_only(args.fast)
    if args.only in ("all", "symbolic"):
        symbolic(args.fast)
    if args.only in ("all", "adaptive"):
        adaptive_pipeline = adaptive(args.fast)
        if args.only == "all" and parametric_pipeline is None:
            parametric_pipeline = adaptive_pipeline
    if args.only in ("all", "serve") and parametric_pipeline is None:
        parametric_pipeline = parametric(args.fast)
    if args.only in ("all", "serve") and parametric_pipeline is not None:
        second_opinion(parametric_pipeline)
        deployment = served(parametric_pipeline)
        if deployment.server is not None:
            deployment.stop()
    print(f"\ndone in {time.perf_counter() - t0:.1f} s; every run is under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
