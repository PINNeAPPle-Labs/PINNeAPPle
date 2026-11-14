"""Trusted execution bridge for industrial/research workflow blueprints.

Register explicit host callbacks. Never execute model-produced Python, shell or
import paths. A verification failure blocks downstream optimization and reports.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pinneapple_core.workflows.adapters import Adapter, StepContext, StepFailed, register_adapter
from pinneapple_core.workflows.dag import Step, Workflow
from pinneapple_core.workflows.engine import WorkflowRunner

STAGES = ("scope", "design", "data", "simulate", "model", "verify", "optimize", "report")
Handler = Callable[[dict[str, Any], StepContext], dict[str, Any]]


class TrustedExperimentAdapter(Adapter):
    kind = "experiment_stage"

    def __init__(self, handlers: Mapping[str, Handler]):
        if set(handlers) != set(STAGES):
            raise ValueError(f"handlers must cover exactly {STAGES}")
        if not all(callable(handler) for handler in handlers.values()):
            raise TypeError("each stage handler must be callable")
        self.handlers = dict(handlers)

    def run(self, ctx: StepContext) -> dict[str, Any]:
        stage = ctx.params.get("stage")
        if stage not in self.handlers:
            raise StepFailed(f"unregistered experiment stage: {stage!r}")
        if stage == "optimize":
            evidence = ctx.upstream.get("verify", {}).get("outputs") or {}
            if evidence.get("approved") is not True:
                raise StepFailed("VeriPhysics verification gate denied optimization")
        if stage == "report":
            evidence = ctx.upstream.get("verify", {}).get("outputs") or {}
            if evidence.get("approved") is not True:
                raise StepFailed("unverified results cannot be reported as approved")
        result = self.handlers[stage](dict(ctx.params), ctx)
        if not isinstance(result, dict):
            raise StepFailed(f"{stage} handler must return a dict")
        if stage == "verify" and result.get("approved") is not True:
            raise StepFailed("verification rejected candidate; no downstream execution")
        return result


def executable_workflow(blueprint: Workflow) -> Workflow:
    """Convert only a known blueprint into trusted adapter steps."""
    if blueprint.metadata.get("execution_mode") != "plan_only":
        raise ValueError("expected a planning blueprint")
    if [s.id for s in blueprint.steps] != list(STAGES):
        raise ValueError("unexpected workflow stages")
    steps = [
        Step(id=s.id, kind="experiment_stage",
             params={**s.params, "stage": s.id},
             needs=list(s.needs) + (["verify"] if s.id == "report" else []), retries=0)
        for s in blueprint.steps
    ]
    return Workflow(
        name=blueprint.name, steps=steps, fail_fast=True, max_parallel=1,
        metadata={**blueprint.metadata, "execution_mode": "trusted_handlers"},
    )


def run_experiment(blueprint: Workflow, handlers: Mapping[str, Handler],
                   root: str, *, run_id: str | None = None) -> dict[str, Any]:
    """Run using existing SQLite runner, logs, retries/resume and manifests.

    This registers a process-local adapter; only use in an isolated worker
    process per tenant. Do not share handler registries across tenants.
    """
    register_adapter(TrustedExperimentAdapter(handlers))
    runner = WorkflowRunner(root, cache=False)
    return runner.run(executable_workflow(blueprint), run_id=run_id)
